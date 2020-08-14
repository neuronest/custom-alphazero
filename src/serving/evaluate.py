import os
from typing import Tuple, Optional

from src.config import ConfigGeneral, ConfigMCTS, ConfigServing
from src.mcts.mcts import MCTS
from src.model.tensorflow.model import PolicyValueModel
from src.serving.factory import init_model

if ConfigGeneral.game == "chess":
    from src.chess.board import Board
    from src.chess.utils import get_all_possible_moves
elif ConfigGeneral.game == "connect_n":
    from src.connect_n.board import Board

    get_all_possible_moves = Board.get_all_possible_moves
else:
    raise NotImplementedError


def set_mcts_model(
        mcts: MCTS,
        previous_model: PolicyValueModel,
        current_model: PolicyValueModel,
        game_index: int
) -> int:
    """
    We alternate which model begin first using game_index
    In addition, we also alternate the model at each move played
    For example:
    - game_index 0
        - move 0
            -> previous_model
        - move 1
            -> current_model
        ...
    - game_index 1
        - move 0
            -> current_model
        - move 1
            -> previous_model
        ...
    ...
    Return 1 if it is current_model that is used, -1 if it is previous_model
    """
    if game_index % 2 == 0:
        if mcts.board.odd_moves_number:
            mcts.model = current_model
            return 1
        else:
            mcts.model = previous_model
            return -1
    else:
        if mcts.board.odd_moves_number:
            mcts.model = previous_model
            return -1
        else:
            mcts.model = current_model
            return 1


def evaluate_against_last_model(
    current_model: PolicyValueModel,
    previous_model: Optional[PolicyValueModel] = None,
    path: Optional[str] = None,
) -> Tuple[PolicyValueModel, float]:
    if previous_model is None:
        assert path is not None
        try:
            max_iteration_name = max(
                os.listdir(path), key=lambda x: int(x.split("_")[-1])
            )
            previous_model = init_model(os.path.join(path, max_iteration_name))
        except ValueError:
            previous_model = init_model()
    score_current_model, null_games, index_model = 0, 0, 0
    for game_index in range(ConfigServing.evaluation_games_number):
        mcts = MCTS(
            board=Board(),
            all_possible_moves=get_all_possible_moves(),
            concurrency=False,
        )
        while not mcts.board.is_game_over():
            index_model = set_mcts_model(mcts, previous_model, current_model, game_index)
            mcts.search(ConfigGeneral.mcts_iterations)
            greedy = mcts.board.fullmove_number > ConfigMCTS.index_move_greedy
            _ = mcts.play(greedy)
        result = mcts.board.get_result(keep_same_player=True)
        if result:
            if index_model == 1:
                score_current_model += 1
        else:
            null_games += 1
    try:
        score = score_current_model / (ConfigServing.evaluation_games_number - null_games)
        if score >= ConfigServing.replace_min_score:
            return current_model, score
        else:
            return previous_model, score
    except ZeroDivisionError:
        return previous_model, 0.5
