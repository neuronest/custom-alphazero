import uuid

import numpy as np

from custom_alphazero.config import ConfigGeneral

if ConfigGeneral.game == "chess":
    from custom_alphazero.chess.board import Board
    from custom_alphazero.chess.utils import get_all_possible_moves
elif ConfigGeneral.game == "connect_n":
    from custom_alphazero.connect_n.board import Board

    get_all_possible_moves = Board.get_all_possible_moves
else:
    raise NotImplementedError


class InferenceExample:
    uid = str(uuid.uuid4())
    state = Board().full_state.tolist()
    concurrency = False
    inputs = {"uid": uid, "state": state, "concurrency": concurrency}


class AppendQueueExample:
    all_possible_moves_length = len(get_all_possible_moves())
    state = Board().full_state
    policy = np.eye(all_possible_moves_length, 1).ravel()
    value = np.asarray(1)
    inputs = {
        "states": np.stack([state, state]).tolist(),
        "policies": np.stack([policy, policy]).tolist(),
        "values": np.stack([value, value]).tolist(),
    }
