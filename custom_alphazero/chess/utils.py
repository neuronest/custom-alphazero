from itertools import product
from typing import List

import numpy as np

from custom_alphazero.chess.board import Board
from custom_alphazero.chess.move import Move
from custom_alphazero.config import ConfigChess


def get_all_possible_moves() -> List[Move]:
    all_possible_moves = set()
    array = np.zeros((ConfigChess.board_size, ConfigChess.board_size)).astype("int8")
    for i, j, piece in product(
        range(ConfigChess.board_size), range(ConfigChess.board_size), ["Q", "N"]
    ):
        array[i][j] = Board.piece_symbol_to_int(piece)
        all_possible_moves.update(
            set(map(lambda move: Move(uci=move.uci()), Board(array=array).legal_moves))
        )
        array[i][j] = 0
    # underpromotion moves
    array[1, :] = Board.piece_symbol_to_int("P")
    all_possible_moves.update(
        set(map(lambda move: Move(uci=move.uci()), Board(array=array).legal_moves))
    )
    array[0, :] = Board.piece_symbol_to_int("p")
    all_possible_moves.update(
        set(map(lambda move: Move(uci=move.uci()), Board(array=array).legal_moves))
    )
    # no need to add castling moves: they have already be added with queen moves under UCI notation
    return sorted(list(all_possible_moves))
