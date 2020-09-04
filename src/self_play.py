import multiprocessing
import os
import time
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from functools import partial

from src import paths
from src.config import ConfigGeneral, ConfigMCTS, ConfigSelfPlay
from src.utils import (
    set_gpu_index,
    best_saved_model,
    visualize_mcts_iteration,
)
from src.mcts.mcts import MCTS
from src.visualize_mcts import MctsVisualizer
from src.serving.factory import get_run_id, append_queue

if ConfigGeneral.game == "chess":
    from src.chess.board import Board
    from src.chess.move import Move
    from src.chess.utils import get_all_possible_moves
elif ConfigGeneral.game == "connect_n":
    from src.connect_n.board import Board
    from src.connect_n.move import Move

    get_all_possible_moves = Board.get_all_possible_moves
else:
    raise NotImplementedError


def play_game(
    process_id: int,
    all_possible_moves: List[Move],
    mcts_iterations: int,
    run_id: str,
    plays_inferences: Optional[Dict[str, Tuple[np.ndarray, float]]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MCTS]:
    # we seed each process with an unique value to ensure each MCTS will be different
    np.random.seed(int((process_id + 1) * time.time()) % (2 ** 32 - 1))
    if not ConfigGeneral.http_inference:
        model = best_saved_model(run_id)
    else:
        model = None
    mcts = MCTS(
        board=Board(),
        all_possible_moves=all_possible_moves,
        concurrency=ConfigGeneral.concurrency,
        plays_inferences=plays_inferences,
        model=model,
        use_solver=ConfigMCTS.use_solver,
    )
    states_game, policies_game = [], []
    while not mcts.board.is_game_over():
        mcts.search(mcts_iterations)
        # mcts.board.fullmove_number starts at 0
        greedy = mcts.board.fullmove_number >= ConfigMCTS.index_move_greedy
        parent_state, child_state, policy, last_move = mcts.play(
            greedy, return_details=True
        )
        states_game.append(parent_state)
        policies_game.append(policy)
    states_game, policies_game = np.asarray(states_game), np.asarray(policies_game)
    # we are assuming reward must be either 0 or 1 because last move must have to led to victory or draw
    reward = mcts.board.get_result(keep_same_player=True)
    rewards_game = np.repeat(reward, len(states_game))
    # reverse rewards as odd positions starting from the end
    rewards_game[-2::-2] = -rewards_game[-2::-2]
    rewards_game = (
        rewards_game
        * ConfigSelfPlay.discounting_factor ** np.arange(len(states_game))[::-1]
    )
    if not ConfigGeneral.http_inference:
        # multi process cannot serialize mcts.model
        mcts.model = None
    return states_game, policies_game, rewards_game, mcts


def play(
        run_id: str, plays_inferences: Optional[Dict[str, Tuple[np.ndarray, float]]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[MCTS]]:
    if ConfigGeneral.mono_process:
        states, policies, rewards, mcts_tree = play_game(
            process_id=0,
            all_possible_moves=get_all_possible_moves(),
            mcts_iterations=ConfigSelfPlay.mcts_iterations,
            run_id=run_id,
            plays_inferences=plays_inferences
        )
        mcts_trees = [mcts_tree]
    else:
        processes = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(processes=processes)
        results = pool.map(
            partial(
                play_game,
                all_possible_moves=get_all_possible_moves(),
                mcts_iterations=ConfigSelfPlay.mcts_iterations,
                run_id=run_id,
                plays_inferences=plays_inferences
            ),
            range(processes),
        )
        pool.close()
        pool.join()
        states, policies, rewards, mcts_trees = list(zip(*results))
        states, policies, rewards = (
            np.vstack(states),
            np.vstack(policies),
            np.concatenate(rewards),
        )
        mcts_trees = list(mcts_trees)
    return states, policies, rewards, mcts_trees


if __name__ == "__main__":
    set_gpu_index(ConfigGeneral.self_play_gpu_index)
    if not ConfigGeneral.mono_process:
        # https://bugs.python.org/issue33725
        # https://stackoverflow.com/a/47852388/5490180
        multiprocessing.set_start_method("spawn")
        plays_inferences = multiprocessing.Manager().dict()
    else:
        plays_inferences = {}
    mcts_visualizer = MctsVisualizer(is_updated=False)
    run_id = get_run_id()
    assert run_id is not None, "Could not get the run if from the server"
    print(f"Starting self play with id={run_id}")
    self_play_iteration = 0
    while True:
        starting_time = time.time()
        os.makedirs(
            paths.get_self_play_iteration_path(run_id, self_play_iteration),
            exist_ok=True,
        )
        states, policies, rewards, mcts_trees = play(
            run_id, plays_inferences=plays_inferences
        )
        # we choose a MCST tree randomly to be traced afterwards
        # each tree results from a fixed state of the neural network, so there is no need to keep them all
        mcts_tree = random.choice(mcts_trees)
        print(
            f"Collected {len(states)} samples in {time.time() - starting_time:.2f} seconds"
        )
        if (self_play_iteration + 1) % ConfigSelfPlay.samples_checkpoint_frequency == 0:
            print("Saving current samples...")
            np.savez(
                paths.get_self_play_samples_path(run_id, self_play_iteration),
                states=states,
                policies=policies,
                values=rewards,
            )
            print("Samples saved")
        append_queue(states, policies, rewards)
        # we pick the previously chosen MCTS tree to visualize it and save it under iteration name
        mcts_visualizer.build_mcts_graph(
            mcts_tree.root, mcts_name=f"mcts_iteration_{self_play_iteration}"
        )
        visualize_mcts_iteration(
            mcts_visualizer,
            mcts_tree=mcts_tree,
            mcts_name=f"mcts_iteration_{self_play_iteration}",
            iteration_path=paths.get_self_play_iteration_path(
                run_id, self_play_iteration
            ),
            run_id=run_id,
        )
        # todo: add light version of visualization with name f"mcts_iteration_light_{iteration}"
        mcts_visualizer = MctsVisualizer(is_updated=False)
        self_play_iteration += 1
