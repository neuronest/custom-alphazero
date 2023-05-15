import hashlib
from copy import deepcopy
from itertools import product
from typing import List, Optional

import numpy as np

from custom_alphazero.config import ConfigConnectN
from custom_alphazero.connect_n.move import Move


class Board:
    def __init__(self, array: Optional[np.ndarray] = None):
        assert (
            2
            <= ConfigConnectN.n
            <= min(ConfigConnectN.board_width, ConfigConnectN.board_height)
        )
        self.board_width = ConfigConnectN.board_width
        self.board_height = ConfigConnectN.board_height
        self.n = ConfigConnectN.n
        self.gravity = ConfigConnectN.gravity
        self.black = ConfigConnectN.black
        self.empty = ConfigConnectN.empty
        self.white = ConfigConnectN.white
        self.pieces = ConfigConnectN.pieces
        self.pieces_to_int = {symbol: value for value, symbol in self.pieces.items()}
        self.played_moves = []
        if array is not None:
            assert isinstance(array, np.ndarray)
            assert (
                array.shape[0] == self.board_height
                and array.shape[1] == self.board_width
            )
            assert np.unique(array).size <= len(self.pieces)
            self.array = array.astype("int8")
        else:
            self.array = np.zeros((self.board_height, self.board_width)).astype("int8")
        self.turn = ConfigConnectN.white
        self.fullmove_number = 0
        self.game_over = False
        self.is_null = None

    def __hash__(self):
        return int(hashlib.md5(repr(self).encode("utf-8")).hexdigest(), 16)

    def __eq__(self, other: "Board"):
        return np.array_equal(self.array, other.array)

    def __repr__(self):
        return "\n".join(
            ["".join(map(lambda x: self.pieces[x], row)) for row in self.array]
        )

    def repr_graphviz(self) -> str:
        def customize_piece(piece):
            if piece == ".":
                # make "." piece wider with spaces on sides to take as much space as other pieces
                piece = " . "
            return piece

        return "\n".join(
            [
                "".join(map(lambda x: customize_piece(self.pieces[x]), row))
                for row in self.array
            ]
        )

    def repr_list_played_moves(self) -> str:
        if not self.gravity:
            raise NotImplementedError
        # the solver needs a string representation with played moves, indexed on 1
        return "".join([str(int(str(move)) + 1) for move in self.played_moves])

    @property
    def turn_mirror(self) -> int:
        return (
            ConfigConnectN.black
            if self.turn == ConfigConnectN.white
            else ConfigConnectN.white
        )

    @property
    def array_one_hot(self) -> np.ndarray:
        return np.eye(len(self.pieces))[self.array]

    @property
    def array_one_hot_mirror(self) -> np.ndarray:
        return np.eye(len(self.pieces))[self.mirror()]

    @property
    def full_state(self) -> np.ndarray:
        return np.dstack(
            [
                self.array_one_hot,
                np.ones((self.board_height, self.board_width)) * self.turn,
            ]
        ).astype("float32")

    @property
    def full_state_mirror(self) -> np.ndarray:
        return np.dstack(
            [
                self.array_one_hot_mirror,
                np.ones((self.board_height, self.board_width)) * self.turn_mirror,
            ]
        ).astype("float32")

    @property
    def odd_moves_number(self) -> bool:
        return bool(self.fullmove_number % 2)

    @property
    def moves(self) -> List[Move]:
        if self.gravity:
            return [
                Move(self.gravity, x)
                for x in next(iter(np.where(self.array[0, :] == ConfigConnectN.empty)))
            ]
        else:
            return [
                Move(self.gravity, x, y)
                for y, x in zip(*np.where(self.array == ConfigConnectN.empty))
            ]

    def last_move(self) -> Optional[Move]:
        if len(self.played_moves):
            return self.played_moves[-1]

    @staticmethod
    def get_all_possible_moves() -> List[Move]:
        if ConfigConnectN.gravity:
            return [
                Move(ConfigConnectN.gravity, x)
                for x in range(ConfigConnectN.board_width)
            ]
        else:
            return [
                Move(ConfigConnectN.gravity, x, y)
                for x, y in list(
                    product(
                        range(ConfigConnectN.board_width),
                        range(ConfigConnectN.board_height),
                    )
                )
            ]

    @staticmethod
    def from_one_hot(array_oh: np.ndarray) -> np.ndarray:
        array = np.argmax(array_oh, axis=-1)
        array[np.where(array > (len(ConfigConnectN.pieces) - 1) / 2)] = -1
        return array

    def legal_moves_mask(self, all_possible_moves: List[Move]) -> np.ndarray:
        return np.asarray([move in self.moves for move in all_possible_moves])

    def update_array(self):
        pass

    def get_random_move(self) -> Optional[Move]:
        try:
            return np.random.choice(self.moves)
        except ValueError:
            return None

    def is_game_over(self) -> bool:
        return self.game_over

    def mirror(self) -> np.ndarray:
        return np.where(
            self.array == ConfigConnectN.black,
            ConfigConnectN.white,
            np.where(
                self.array == ConfigConnectN.white, ConfigConnectN.black, self.array
            ),
        )

    def update_game_over(self, last_move_x: int, last_move_y: int):
        if self.game_over:
            return
        x, y = last_move_x, last_move_y
        for direction in ConfigConnectN.directions:
            number_connexions = 1
            for direction_current in [
                direction,
                tuple(-elem for elem in direction),
            ]:
                while (
                    0 <= x + direction_current[0] < self.board_width
                    and 0 <= y + direction_current[1] < self.board_height
                ):
                    if (
                        self.array[y + direction_current[1], x + direction_current[0]]
                        == self.array[last_move_y, last_move_x]
                    ):
                        number_connexions += 1
                        x, y = x + direction_current[0], y + direction_current[1]
                        if number_connexions >= ConfigConnectN.n:
                            self.game_over = True
                            self.is_null = False
                            return
                    else:
                        break
                x, y = last_move_x, last_move_y

        if not len(self.moves):
            self.game_over = True
            self.is_null = True

    def push(self, move: Move):
        if self.gravity:
            empty_rows_in_col = next(
                iter(np.where(self.array[:, move.x] == ConfigConnectN.empty))
            )
            row_position = -1
            for row_index in range(self.board_height):
                if row_index not in empty_rows_in_col:
                    row_position = row_index - 1
                    break
                row_position = row_index
            assert (
                row_position >= 0
                and self.array[row_position, move.x] == ConfigConnectN.empty
            )
            self.array[row_position, move.x] = self.turn
            self.update_game_over(last_move_x=move.x, last_move_y=row_position)
        else:
            assert self.array[move.y, move.x] == ConfigConnectN.empty
            self.array[move.y, move.x] = self.turn
            self.update_game_over(last_move_x=move.x, last_move_y=move.y)
        self.turn = self.turn_mirror

    def play(
        self,
        move: Optional[Move],
        on_copy: bool = False,
        keep_same_player: bool = False,
    ) -> "Board":
        if move is None or self.game_over:
            return self
        board = deepcopy(self) if on_copy else self
        board.push(move)
        board.fullmove_number += 1
        if keep_same_player:
            board.array = board.mirror()
            board.turn = ConfigConnectN.white  # virtually, it is always white to play
        board.played_moves.append(move)
        if not on_copy:
            self.__dict__.update(board.__dict__)
        return board

    def play_random(
        self, on_copy: bool = False, keep_same_player: bool = False
    ) -> "Board":
        random_move = self.get_random_move()
        return self.play(random_move, on_copy, keep_same_player)

    def get_result(self, keep_same_player: bool = False):
        if self.is_null is None or not self.game_over:
            return None
        elif self.is_null is True:
            return 0
        if keep_same_player:
            return ConfigConnectN.white
        else:
            return (
                ConfigConnectN.white if self.odd_moves_number else ConfigConnectN.black
            )

    def display_ascii(self):
        print(repr(self))
