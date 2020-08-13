import multiprocessing
import numpy as np
import platform
import time
import argparse
from typing import List, Tuple
from functools import partial

from src.config import ConfigGeneral, ConfigMCTS

from src.mcts.mcts import MCTS
from src.serving.factory import train_samples
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


def play_game(
    process_id: int, all_possible_moves: List[Move], mcts_iterations: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[MCTS]]:
    np.random.seed(int(process_id * time.time()) % (2 ** 32 - 1))
    board = Board()
    mcts = MCTS(
        board=board,
        all_possible_moves=all_possible_moves,
        concurrency=ConfigGeneral.concurrency,
    )
    states_straight_game, states_mirror_game, policies_game = [], [], []
    while not board.is_game_over():
        mcts.search(mcts_iterations)
        greedy = board.fullmove_number > ConfigMCTS.index_move_greedy
        state, state_mirror, policy, last_move = mcts.play(greedy, return_details=True)
        states_straight_game.append(state)
        states_mirror_game.append(state_mirror)
        policies_game.append(policy)
    # we are assuming reward must be either 0 or 1 because last move must have to led to victory or draw
    reward = abs(
        board.get_result()
    )  # todo: check this to me there was an error here, reward could be -1
    states_game = (
        states_mirror_game if board.odd_moves_number else states_straight_game
    )  # todo: check this unsure about this
    states_game, policies_game = np.asarray(states_game), np.asarray(policies_game)
    rewards_game = np.repeat(reward, len(states_game))
    # reverse rewards as odd positions as these are views from the opposite player
    rewards_game[1::2] = -rewards_game[1::2]
    rewards_game = rewards_game * ConfigGeneral.discounting_factor ** np.arange(
        len(states_game)
    )
    rewards_game = rewards_game[::-1]
    return states_game, policies_game, rewards_game, [mcts]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mono-process",
        action="store_true",
        help="Disable multiprocessing, used mainly for testing",
    )
    args = parser.parse_args()
    mono_process = args.mono_process
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
    states_batch, policies_batch, rewards_batch = None, None, None
    for iteration in range(ConfigGeneral.iterations):
        starting_time = time.time()
        if mono_process:
            states, policies, rewards, mcts_trees = play_game(
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
                range(1, processes + 1),
            )
            pool.close()
            pool.join()
            states, policies, rewards, mcts_trees = list(zip(*results))
            states, policies, rewards, mcts_trees = (
                np.vstack(states),
                np.vstack(policies),
                np.concatenate(rewards),
                np.concatenate(mcts_trees),
            )
        if any(
            sample is None for sample in [states_batch, policies_batch, rewards_batch]
        ):
            states_batch, policies_batch, rewards_batch, mcts_trees_batch = (
                states,
                policies,
                rewards,
                mcts_trees,
            )
        else:
            states_batch, policies_batch, rewards_batch, mcts_trees_batch = (
                np.vstack([states_batch, states]),
                np.vstack([policies_batch, policies]),
                np.concatenate([rewards_batch, rewards]),
                np.concatenate([mcts_trees_batch, mcts_trees]),
            )
        print(
            "Collected {0} samples in {1:.2f} seconds".format(
                states.shape[0], time.time() - starting_time
            )
        )
        print("Now having {} samples in the stack".format(states_batch.shape[0]))
        if len(states_batch) >= ConfigGeneral.minimum_training_size:
            training_starting_time = time.time()
            print("Training on {} samples...".format(len(states_batch)))
            loss, updated = train_samples(states_batch, [policies_batch, rewards_batch])
            print(
                "Training took {:.2f} seconds".format(
                    time.time() - training_starting_time
                )
            )
            if updated:
                print("The model has been updated")
                # pick one mcts randomly in batch and vizualier and save under iteration name
                MctsVisualizer(
                    mcts_trees_batch[np.random.randint(len(mcts_trees_batch))].root,
                    mcts_name=f"mcts_iteration_{iteration}",
                ).save_as_gv_and_pdf(directory="mcts_visualization")
            else:
                print("The model has not been updated")
            print("Current loss: {0:.5f}".format(loss))

            states_batch, policies_batch, rewards_batch, mcts_trees_batch = (
                None,
                None,
                None,
                None,
            )
