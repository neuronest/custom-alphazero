from functools import total_ordering
from typing import Tuple, Optional

from src.config import ConfigChess


@total_ordering
class Move:
    def __init__(
        self,
        pos_from: Optional[Tuple[int, int]] = None,
        pos_to: Optional[Tuple[int, int, str]] = None,
        uci: Optional[str] = None,
    ):
        if uci is not None:
            pos_from, pos_to = self.uci_to_coords(uci)
        assert pos_from is not None and pos_to is not None
        self.pos_from = pos_from
        self.pos_to = pos_to

    def __str__(self):
        return "({0}, {1}) -> ({2}, {3}, {4})".format(*self.pos_from, *self.pos_to)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (self.pos_from, self.pos_to) == (other.pos_from, other.pos_to)

    def __lt__(self, other):
        return (self.pos_from, self.pos_to) < (other.pos_from, other.pos_to)

    def __hash__(self):
        return hash((self.pos_from, self.pos_to))

    @property
    def uci(self) -> str:
        return (
            chr(self.pos_from[0] + ord("a"))
            + str(self.pos_from[1] + 1)
            + chr(self.pos_to[0] + ord("a"))
            + str(self.pos_to[1] + 1)
            + self.pos_to[2]
        )

    @staticmethod
    def uci_to_coords(uci: str) -> Tuple[Tuple[int, int], Tuple[int, int, str]]:
        assert 4 <= len(uci) <= 5
        position_from = ord(uci[0]) - ord("a"), int(uci[1]) - 1
        position_to = ord(uci[2]) - ord("a"), int(uci[3]) - 1
        if len(uci) == 5:
            position_to = position_to + (uci[4],)
        else:
            position_to = position_to + ("",)
        return position_from, position_to

    @staticmethod
    def mirror(move: "Move") -> "Move":
        return Move(
            pos_from=(
                ConfigChess.board_size - 1 - move.pos_from[0],
                ConfigChess.board_size - 1 - move.pos_from[1],
            ),
            pos_to=(
                ConfigChess.board_size - 1 - move.pos_to[0],
                ConfigChess.board_size - 1 - move.pos_to[1],
                move.pos_to[2],
            ),
        )
