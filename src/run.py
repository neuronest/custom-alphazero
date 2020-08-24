import multiprocessing
import os

from src.utils import last_saved_model, last_iteration_name

import numpy as np
import platform
import time
import argparse
from typing import List, Tuple, Optional, Dict
from functools import partial
from datetime import datetime
import pickle

from src.config import ConfigGeneral, ConfigMCTS, ConfigPath, ConfigServing

from src.mcts.mcts import MCTS
from src.visualize_mcts import MctsVisualizer
from src.model.tensorflow.train import train_and_report_performance
from src.serving import factory

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


def last_iteration_inferences(run_path: str, not_exist_ok: bool = True) -> Dict[str, Tuple[np.ndarray, float]]:
    try:
        with open(
            os.path.join(
                run_path, last_iteration_name(run_path), "state_priors_value.pkl"
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
    with open(os.path.join(dir_path, "state_priors_value.pkl"), "wb") as f:
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
        greedy = mcts.board.fullmove_number > ConfigMCTS.index_move_greedy
        parent_state, child_state, policy, last_move = mcts.play(
            greedy, return_details=True
        )
        states_game.append(parent_state)
        policies_game.append(policy)
    # we are assuming reward must be either 0 or 1 because last move must have to led to victory or draw
    reward = mcts.board.get_result(keep_same_player=True)
    states_game, policies_game = np.asarray(states_game), np.asarray(policies_game)
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


# class intended to reproduce the same global behavior as request.app.state on fastapi server
class App:
    class State:
        number_samples = 0
        iteration = 0


def train_run_samples(
    run_id: str, states: np.ndarray, labels: List[np.ndarray]
) -> Tuple[float, bool, int]:
    policies, values = labels
    App.State.number_samples += len(states)
    run_path = os.path.join(ConfigPath.results_path, ConfigGeneral.game, run_id)
    iteration_path = os.path.join(run_path, f"iteration_{App.State.iteration}")
    tensorboard_path = os.path.join(run_path, ConfigPath.tensorboard_endpath)
    os.makedirs(iteration_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    # load last model that has been used for self-play and now to train
    model = last_saved_model(
        os.path.join(ConfigPath.results_path, ConfigGeneral.game, run_id)
    )
    _, loss, updated = train_and_report_performance(
        model,
        states,
        policies,
        values,
        run_path,
        iteration_path,
        tensorboard_path,
        App.State.iteration,
        App.State.number_samples,
        ConfigServing.training_epochs,
    )
    iteration = App.State.iteration
    App.State.iteration += 1
    return loss, updated, iteration


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
        loss, updated, iteration = factory.train_run_samples(
            run_id, states_batch, [policies_batch, rewards_batch]
        )
    else:
        loss, updated, iteration = train_run_samples(
            run_id, states_batch, [policies_batch, rewards_batch]
        )
    print(f"Training took {time.time() - training_starting_time:.2f} seconds")
    return loss, updated, iteration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mono-process",
        action="store_true",
        help="Disable multiprocessing, used mainly for testing",
    )
    args = parser.parse_args()
    mono_process = args.mono_process
    run_id = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    iteration_mcts_trees = []
    print(f"Starting run with id={run_id}")
    if not mono_process:
        # https://bugs.python.org/issue33725
        # https://stackoverflow.com/a/47852388/5490180
        if platform.system() == "Linux":
            multiprocessing.set_start_method("fork")
        else:
            multiprocessing.set_start_method("spawn")
    all_possible_moves = get_all_possible_moves()
    action_space = len(all_possible_moves)
    input_dim = Board().full_state.shape
    states_queue, policies_queue, rewards_queue = None, None, None
    latest_experience_amount = 0
    for _ in range(ConfigGeneral.iterations):
        starting_time = time.time()
        if mono_process:
            states, policies, rewards, mcts_tree = play_game(
                process_id=0,
                all_possible_moves=all_possible_moves,
                mcts_iterations=ConfigGeneral.mcts_iterations,
            )
        else:
            processes = multiprocessing.cpu_count() - 1
            pool = multiprocessing.Pool(processes=processes)
            results = pool.map(
                partial(
                    play_game,
                    all_possible_moves=all_possible_moves,
                    mcts_iterations=ConfigGeneral.mcts_iterations,
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
            # we choose a MCST tree randomly to be traced afterwards
            # each tree results from a fixed state of the neural network, so there is no need to keep them all
            mcts_tree = mcts_trees[np.random.randint(len(mcts_trees))]
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
            latest_experience_amount = 0
            if updated:
                print("The model has been updated")
            else:
                print(
                    f"Run {run_id}, model has not been updated, saving mcts trees inferences for reuse"
                )
                save_mcts_trees_inferences(iteration_mcts_trees, iteration_path)

            print(f"Current loss: {loss:.5f}")
            # we pick the previously chosen MCTS tree to visualize it and save it under iteration name
            MctsVisualizer(
                mcts_tree.root, mcts_name=f"mcts_iteration_{iteration}",
            ).save_as_pdf(directory=iteration_path)
            states_batch, policies_batch, rewards_batch = (
                None,
                None,
                None,
            )
            iteration_mcts_trees = []
