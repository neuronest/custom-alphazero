from functools import total_ordering
from typing import Optional


@total_ordering
class Move:
    def __init__(
        self,
        gravity: bool,
        x: int,
        y: Optional[int] = None,
    ):
        self.gravity = gravity
        if self.gravity:
            assert y is None
            self.x = x
            self.y = None
        else:
            assert y is not None
            self.x = x
            self.y = y

    def __str__(self):
        if self.gravity:
            return "{}".format(self.x)
        else:
            return "({0}, {1})".format(self.x, self.y)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))
