import multiprocessing
import os

from src.utils import last_saved_model, last_iteration_name

import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from functools import partial
from datetime import datetime
import pickle

from src.config import ConfigGeneral, ConfigMCTS, ConfigPath, ConfigServing

from src.mcts.mcts import MCTS
from src.visualize_mcts import MctsVisualizer
from src.model.tensorflow.train import train_and_report
from src.serving import factory
from src.serving.factory import init_model

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

os.environ["CUDA_VISIBLE_DEVICES"] = str(ConfigGeneral.gpu)


def printer(
    board: Board, game_index: int, move_index: int, last_player: str, last_move: Move
):
    print("Game number:\t", game_index + 1)
    print("Moves played:\t", move_index)
    print("Last player:\t", last_player)
    print("Last move:\t", last_move)
    if last_player == "white":
        board.array = board.mirror()
        board.update_array()
    board.display_ascii()
    print()


def last_iteration_inferences(
    run_path: str, not_exist_ok: bool = True
) -> Dict[str, Tuple[np.ndarray, float]]:
    try:
        with open(
            os.path.join(
                run_path,
                last_iteration_name(run_path),
                ConfigPath.saved_inferences_name,
            ),
            "rb",
        ) as f:
            state_priors_value = pickle.load(f)
    except FileNotFoundError as e:
        if not_exist_ok:
            return None
        raise e
    return state_priors_value


def save_mcts_trees_inferences(mcts_trees: List[MCTS], dir_path: str) -> None:
    # dictionary mapping an input state to the inferred priors and value
    state_priors_value = {}
    for mcts_state_priors_value in [mcts.state_priors_value for mcts in mcts_trees]:
        state_priors_value.update(mcts_state_priors_value)

    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, ConfigPath.saved_inferences_name), "wb") as f:
        pickle.dump(state_priors_value, f)


def play_game(
    process_id: int, all_possible_moves: List[Move], mcts_iterations: int, run_id: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MCTS]:
    np.random.seed(int((process_id + 1) * time.time()) % (2 ** 32 - 1))
    model = (
        None
        if ConfigGeneral.run_with_http
        else last_saved_model(
            os.path.join(ConfigPath.results_path, ConfigGeneral.game, run_id)
        )
    )
    state_priors_value = last_iteration_inferences(
        os.path.join(ConfigPath.results_path, ConfigGeneral.game, run_id),
        not_exist_ok=True,
    )
    mcts = MCTS(
        board=Board(),
        all_possible_moves=all_possible_moves,
        concurrency=ConfigGeneral.concurrency,
        model=model,
        state_priors_value=state_priors_value,
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
    if ConfigGeneral.run_with_http:
        train_run_samples = factory.train_run_samples
    loss, updated, iteration = train_run_samples(
        run_id, states_batch, [policies_batch, rewards_batch]
    )
    print(f"Training took {time.time() - training_starting_time:.2f} seconds")
    return loss, updated, iteration


def visualize_mcts_iteration(
    mcts_visualizer: MctsVisualizer, iteration_path: str, run_id: Optional[str] = None,
) -> None:
    mcts_visualizer.save_as_pdf(directory=iteration_path)
    # if mcts generated by an updated model then save it also at runpath level to visualize those trees more easily
    if mcts_visualizer.is_updated:
        assert run_id is not None
        mcts_visualizer.save_as_pdf(
            directory=os.path.join(
                ConfigPath.results_path,
                ConfigGeneral.game,
                run_id,
                ConfigPath.updated_mcts_dir,
            )
        )


def play(run_id):
    if ConfigGeneral.mono_process:
        states, policies, rewards, mcts_tree = play_game(
            process_id=0,
            all_possible_moves=get_all_possible_moves(),
            mcts_iterations=ConfigGeneral.mcts_iterations,
            run_id=run_id,
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
    return states, policies, rewards, mcts_trees


if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    mcts_visualizer = MctsVisualizer(is_updated=False)
    states_queue = policies_queue = rewards_queue = None
    latest_experience_amount = 0
    iteration_mcts_trees = []
    print(f"Starting run with id={run_id}")
    if not ConfigGeneral.mono_process:
        # https://bugs.python.org/issue33725
        # https://stackoverflow.com/a/47852388/5490180
        multiprocessing.set_start_method("spawn")
    for _ in range(ConfigGeneral.training_iterations):
        starting_time = time.time()
        states, policies, rewards, mcts_trees = play(run_id)
        # we choose a MCST tree randomly to be traced afterwards
        # each tree results from a fixed state of the neural network, so there is no need to keep them all
        mcts_tree = mcts_trees[np.random.randint(len(mcts_trees))]
        iteration_mcts_trees += mcts_trees
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
            else:
                print(
                    f"Run {run_id}, model has not been updated, saving mcts trees inferences for reuse"
                )
                save_mcts_trees_inferences(iteration_mcts_trees, iteration_path)

            print(f"Current loss: {loss:.5f}")
            # we pick the previously chosen MCTS tree to visualize it and save it under iteration name
            mcts_visualizer.build_mcts_graph(
                mcts_tree.root, mcts_name=f"mcts_iteration_{iteration}"
            )
            visualize_mcts_iteration(
                mcts_visualizer, iteration, iteration_path, run_id=run_id
            )
            # if model has been updated it will participate in the next self-play phase and in the next mcts tree
            # mcts_name = (
            #    f"mcts_updated_iteration_{iteration + 1}"
            #    if updated
            #    else f"mcts_iteration_{iteration + 1}"
            # )

            mcts_visualizer = MctsVisualizer(is_updated=updated)
            states_batch = policies_batch = rewards_batch = None
            latest_experience_amount = 0
            iteration_mcts_trees = []
