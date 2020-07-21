import copy
import numpy as np
import chess as python_chess
from collections import deque
from typing import Optional, List

from src.config import ConfigChess
from src.chess.move import Move


class Board(python_chess.Board):
    def __init__(
        self,
        board_fen: Optional[str] = None,
        array: Optional[np.ndarray] = None,
        history_size: int = 8,
    ):
        self.board_size = ConfigChess.board_size
        self.number_unique_pieces = ConfigChess.number_unique_pieces
        if array is not None:
            assert isinstance(array, np.ndarray)
            assert all([dim == self.board_size for dim in array.shape])
            assert np.unique(array).size <= self.number_unique_pieces + 1
            self.array = array.astype("int8")
            board_fen = self.array_to_board_fen(self.array)
            fen = self.get_fen(board_fen)
            python_chess.Board.__init__(self, fen=fen)
        else:
            board_fen = (
                ConfigChess.initial_board_fen if board_fen is None else board_fen
            )
            fen = self.get_fen(board_fen)
            python_chess.Board.__init__(self, fen=fen)
            self.array = self.board_fen_to_array(self.board_fen())
        self.state_history = deque(maxlen=history_size)
        for time_step in range(history_size):
            self.state_history.append(np.zeros(self.state.shape))
        self.state_history.append(self.state)

    @property
    def array_one_hot(self) -> np.ndarray:
        return np.eye(self.number_unique_pieces + 1)[self.array]

    @property
    def moves(self) -> List[Move]:
        return [Move(uci=move.uci()) for move in self.legal_moves]

    @property
    def state(self) -> np.ndarray:
        return np.dstack(
            [
                self.array_one_hot,
                np.full((self.board_size, self.board_size), self.is_repetition()),
            ]
        )

    @property
    def full_state(self) -> np.ndarray:
        return np.dstack(
            [np.dstack(self.state_history)]
            + [
                np.full((self.board_size, self.board_size), feature)
                for feature in [
                    self.has_queenside_castling_rights(self.turn),
                    self.has_kingside_castling_rights(self.turn),
                    self.has_queenside_castling_rights(not self.turn),
                    self.has_kingside_castling_rights(not self.turn),
                    self.fullmove_number,
                    self.halfmove_clock,
                ]
            ]
        )

    @staticmethod
    def get_fen(board_fen: str):
        return " ".join(
            [
                board_fen,
                ConfigChess.initial_turn,
                ConfigChess.initial_castling_rights,
                ConfigChess.initial_ep_quare,
                ConfigChess.initial_halfmove_clock,
                ConfigChess.initial_fullmove_number,
            ]
        )

    @staticmethod
    def from_one_hot(array_oh: np.ndarray) -> np.ndarray:
        array = np.argmax(array_oh, axis=-1)
        array[np.where(array > ConfigChess.number_unique_pieces / 2)] = (
            array[np.where(array > ConfigChess.number_unique_pieces / 2)]
            - ConfigChess.number_unique_pieces
            + 1
        )
        return array

    @staticmethod
    def piece_symbol_to_int(piece_symbol: Optional[str]) -> int:
        if piece_symbol is None:
            return 0
        piece_int = ConfigChess.piece_symbols.index(piece_symbol.lower())
        player = 1 if piece_symbol.isupper() else -1
        return player * piece_int

    @staticmethod
    def int_to_piece_symbol(piece_int: int) -> Optional[str]:
        player, piece_symbol = (
            np.sign(piece_int),
            ConfigChess.piece_symbols[np.abs(piece_int)],
        )
        if piece_symbol is None:
            return piece_symbol
        return piece_symbol if player < 0 else piece_symbol.upper()

    def legal_moves_mask(self, all_possible_moves: List[Move]) -> np.ndarray:
        return np.asarray([move in self.moves for move in all_possible_moves])

    def board_fen_to_array(self, fen: str) -> np.ndarray:
        mat = []
        board_fen = fen.replace("/", "")
        for elem in board_fen:
            if elem.isdigit():
                mat.extend(int(elem) * [0])
            else:
                mat.append(self.piece_symbol_to_int(elem))
        return (
            np.asarray(mat).reshape((self.board_size, self.board_size)).astype("int8")
        )

    def array_to_board_fen(self, array: np.ndarray) -> str:
        fen = ""
        cases_counter, empty_cases_counter = 0, 0
        for piece_int in np.nditer(array):
            piece_symbol = self.int_to_piece_symbol(piece_int)
            if piece_symbol is None:
                empty_cases_counter += 1
            else:
                if empty_cases_counter > 0:
                    fen += str(empty_cases_counter)
                    empty_cases_counter = 0
                fen += piece_symbol
            cases_counter += 1
            if cases_counter % self.board_size == 0:
                if empty_cases_counter:
                    fen += str(empty_cases_counter)
                if cases_counter != self.board_size * self.board_size:
                    fen += "/"
                empty_cases_counter = 0
        return fen

    def update_array(self):
        self.array = self.board_fen_to_array(self.board_fen())
        self.state_history.append(self.state)

    def get_random_move(self) -> Optional[Move]:
        try:
            return np.random.choice(self.moves)
        except ValueError:
            return None

    def play(
        self, move: Move, on_copy: bool = False, keep_same_player: bool = False
    ) -> "Board":
        board = copy.deepcopy(self) if on_copy else self
        board.push_uci(move.uci)
        if keep_same_player:
            board = board.mirror()
            board.turn = True  # virtually, it is always white to play
        board.update_array()
        if not on_copy:
            self.__dict__.update(board.__dict__)
        return board

    def play_random(self) -> "Board":
        return self.play(self.get_random_move())

    def get_result(self):
        if not self.is_game_over():
            return None
        result = self.result()
        # 1-0 or 0-1
        if len(result) == 3:
            if result[0] > result[1]:
                return 1
            else:
                return -1
        # 1/2-1/2
        elif len(result) == 7:
            return 0

    def display_ascii(self):
        for row in self.array:
            print(
                "".join(map(lambda x: self.int_to_piece_symbol(x) if x else ".", row))
            )
