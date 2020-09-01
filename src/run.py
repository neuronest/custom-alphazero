import multiprocessing
import os
import time
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from functools import partial
from datetime import datetime

from src.config import ConfigGeneral, ConfigMCTS, ConfigPath
from src.utils import (
    last_saved_model,
    visualize_mcts_iteration,
)
from src.mcts.mcts import MCTS
from src.visualize_mcts import MctsVisualizer

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

if ConfigGeneral.run_with_http:
    from src.serving.factory import train_run_samples_post as train_run_samples
else:
    from src.model.tensorflow.train import train_run_samples_local
    from src.utils import LocalState

    train_run_samples = partial(train_run_samples_local, local_state=LocalState())


def play_game(
    process_id: int,
    all_possible_moves: List[Move],
    mcts_iterations: int,
    run_id: str,
    plays_inferences: Optional[Dict[str, Tuple[np.ndarray, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MCTS]:
    # we seed each process with an unique value to ensure each MCTS will be different
    np.random.seed(int((process_id + 1) * time.time()) % (2 ** 32 - 1))
    if not ConfigGeneral.run_with_http:
        model = last_saved_model(
            os.path.join(ConfigPath.results_path, ConfigGeneral.game, run_id)
        )
    else:
        model = None
    mcts = MCTS(
        board=Board(),
        all_possible_moves=all_possible_moves,
        concurrency=ConfigGeneral.concurrency,
        use_solver=ConfigMCTS.use_solver,
        model=model,
        state_priors_value=plays_inferences,
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
        * ConfigGeneral.discounting_factor ** np.arange(len(states_game))[::-1]
    )
    if not ConfigGeneral.run_with_http:
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
            mcts_iterations=ConfigGeneral.mcts_iterations,
            run_id=run_id,
            plays_inferences=plays_inferences,
        )
        mcts_trees = [mcts_tree]
    else:
        processes = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(processes=processes)
        results = pool.map(
            partial(
                play_game,
                all_possible_moves=get_all_possible_moves(),
                mcts_iterations=ConfigGeneral.mcts_iterations,
                run_id=run_id,
                plays_inferences=plays_inferences,
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


def train_run_queue(
    run_id: str,
    states_queue: np.ndarray,
    policies_queue: np.ndarray,
    rewards_queue: np.ndarray,
    minimum_training_size: int,
) -> Tuple[float, bool, int]:
    training_starting_time = time.time()
    print(
        f"Training on {minimum_training_size} samples taken randomly from the queue..."
    )
    sample_indexes = np.random.choice(
        len(states_queue), minimum_training_size, replace=False
    )
    states_batch, policies_batch, rewards_batch = (
        states_queue[sample_indexes],
        policies_queue[sample_indexes],
        rewards_queue[sample_indexes],
    )
    loss, updated, iteration = train_run_samples(
        run_id=run_id, states=states_batch, labels=[policies_batch, rewards_batch]
    )
    print(f"Training took {time.time() - training_starting_time:.2f} seconds")
    return loss, updated, iteration


if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    mcts_visualizer = MctsVisualizer(is_updated=False)
    states_queue = policies_queue = rewards_queue = None
    latest_experience_amount = 0
    print(f"Starting run with id={run_id}")
    if not ConfigGeneral.mono_process:
        # https://bugs.python.org/issue33725
        # https://stackoverflow.com/a/47852388/5490180
        multiprocessing.set_start_method("spawn")
        plays_inferences = multiprocessing.Manager().dict()
    else:
        plays_inferences = {}
    for _ in range(ConfigGeneral.training_iterations):
        starting_time = time.time()
        states, policies, rewards, mcts_trees = play(
            run_id, plays_inferences=plays_inferences
        )
        # we choose a MCST tree randomly to be traced afterwards
        # each tree results from a fixed state of the neural network, so there is no need to keep them all
        mcts_tree = random.choice(mcts_trees)
        latest_experience_amount += len(states)
        if any(
            sample is None for sample in [states_queue, policies_queue, rewards_queue]
        ):
            states_queue, policies_queue, rewards_queue = (
                states,
                policies,
                rewards,
            )
        else:
            states_queue, policies_queue, rewards_queue = (
                np.vstack([states_queue, states]),
                np.vstack([policies_queue, policies]),
                np.concatenate([rewards_queue, rewards]),
            )
        # we remove oldest samples from the queue
        states_queue, policies_queue, rewards_queue = (
            states_queue[-ConfigGeneral.samples_queue_size :],
            policies_queue[-ConfigGeneral.samples_queue_size :],
            rewards_queue[-ConfigGeneral.samples_queue_size :],
        )
        print(
            f"Collected {len(states)} samples in {time.time() - starting_time:.2f} seconds\n"
            f"Now having {len(states_queue)} samples in the queue and {latest_experience_amount} new experience samples"
        )
        if (
            len(states_queue) >= ConfigGeneral.minimum_training_size
            and latest_experience_amount >= ConfigGeneral.minimum_delta_size
        ):
            loss, updated, iteration = train_run_queue(
                run_id,
                states_queue,
                policies_queue,
                rewards_queue,
                ConfigGeneral.minimum_training_size,
            )
            iteration_path = os.path.join(
                ConfigPath.results_path,
                ConfigGeneral.game,
                run_id,
                f"iteration_{iteration}",
            )
            if updated:
                print(
                    f"Run {run_id}, model has been updated and saved at {iteration_path}"
                )
                plays_inferences = (
                    {}
                    if ConfigGeneral.mono_process
                    else multiprocessing.Manager().dict()
                )
            else:
                print(
                    f"Run {run_id}, model has not been updated, saving mcts trees inferences for reuse"
                )
            print(f"Current loss: {loss:.5f}")
            # we pick the previously chosen MCTS tree to visualize it and save it under iteration name
            visualize_mcts_iteration(
                mcts_visualizer=mcts_visualizer,
                mcts_tree=mcts_tree,
                mcts_name=f"mcts_iteration_{iteration}",
                iteration_path=iteration_path,
                run_id=run_id,
            )
            mcts_visualizer = MctsVisualizer(is_updated=updated)
            states_batch = policies_batch = rewards_batch = None
            latest_experience_amount = 0
