import subprocess
from typing import List, Optional, Tuple

import numpy as np

from custom_alphazero.config import ConfigPath
from custom_alphazero.connect_n.board import Board
from custom_alphazero.connect_n.move import Move

"""
c4solver binary compiled from https://github.com/PascalPons/connect4
7x6.book opening book got from https://github.com/PascalPons/connect4/releases/download/book/7x6.book
"""


def _delete_trailing_eol(string: str) -> str:
    return string[:-1] if string.endswith("\n") else string


def _add_trailing_eol(string: str) -> str:
    return string + "\n" if not string.endswith("\n") else string


def evaluate_boards_with_solution(boards_as_list_moves: str) -> Optional[List[int]]:
    boards_as_list_moves = _add_trailing_eol(boards_as_list_moves)
    number_of_boards = len(boards_as_list_moves.split("\n")) - 1
    try:
        solver_output = subprocess.run(
            [ConfigPath.connect4_solver_bin, "-b", ConfigPath.connect4_opening_book],
            stdout=subprocess.PIPE,
            input=boards_as_list_moves.encode("utf-8"),
        ).stdout.decode()
    except FileNotFoundError:
        print("Solver not found!")
        return None
    solver_output = _delete_trailing_eol(solver_output)
    solver_output = solver_output.split("\n")
    # the output from the solver is normally dependant of the number of boards
    # in addition, we should get 4 responses per board (board, value, number of moves before the end, computed time)
    if len(solver_output) != number_of_boards or not all(
        len(elem.split(" ")) == 4 for elem in solver_output
    ):
        print(f"solver_output: {solver_output}, number_of_boards: {number_of_boards}")
        print("Incorrect input!")
        return None
    # we only keep the second response per board, which is the value
    solver_output = list(map(int, [elem.split(" ")[1] for elem in solver_output]))
    return solver_output


def exact_ranked_moves_and_value(board: Board) -> Tuple[List[int], float]:
    assert not board.is_game_over()
    child_boards = [board.play(move, on_copy=True) for move in board.moves]
    ending_moves = np.array(
        [child_board.is_game_over() for child_board in child_boards]
    )
    all_non_ending_boards = [board] + list(np.array(child_boards)[~ending_moves])
    results = evaluate_boards_with_solution(
        "\n".join(
            [
                all_boards_i.repr_list_played_moves()
                for all_boards_i in all_non_ending_boards
            ]
        )
    )
    assert results is not None
    exact_child_values = np.zeros(len(child_boards))
    exact_child_values[ending_moves] = -np.inf
    exact_child_values[~ending_moves] = results[1:]
    ranked_moves_indexes = list(np.argsort(exact_child_values))
    exact_value = float(
        np.sign(results[0])
    )  # we want a value in {-1, 0, 1} for the board
    return ranked_moves_indexes, exact_value


def exact_policy_and_value(
    board: Board, all_possible_moves: List[Move]
) -> Tuple[np.ndarray, float]:
    ranked_moves_indexes, exact_value = exact_ranked_moves_and_value(board)
    exact_policy = np.zeros(len(all_possible_moves)).astype(float)
    best_move_index = int(ranked_moves_indexes[0])
    best_all_moves_index = all_possible_moves.index(board.moves[best_move_index])
    exact_policy[best_all_moves_index] = 1.0
    return exact_policy, exact_value
