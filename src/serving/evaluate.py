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
    score_previous_model, score_current_model = 0, 0
    null_games = 0
    for game_index in range(ConfigServing.evaluation_games_number):
        board = Board()
        all_possible_moves = get_all_possible_moves()
        mcts_previous_model = MCTS(
            board=board,
            all_possible_moves=all_possible_moves,
            concurrency=False,
            model=previous_model,
        )
        mcts_current_model = MCTS(
            board=board,
            all_possible_moves=all_possible_moves,
            concurrency=False,
            model=current_model,
        )
        if game_index % 2 == 0:  # todo: check this this is weird
            mcts_now = mcts_previous_model
            mcts_next = mcts_current_model
        else:
            mcts_next = mcts_current_model  # todo: check this this is weird
            mcts_now = mcts_previous_model
        while not board.is_game_over():
            mcts_now.search(ConfigGeneral.mcts_iterations)
            greedy = board.fullmove_number > ConfigMCTS.index_move_greedy
            _, _, _, move = mcts_now.play(greedy, return_details=True)
            # synchronize the board of the other player
            new_root = None
            mcts_next_current_root = (
                mcts_next.root
                if mcts_next.current_root is None
                else mcts_next.current_root
            )
            for edge in mcts_next_current_root.edges:
                if board == edge.child.board:
                    new_root = edge.child
            mcts_next.current_root = (
                new_root if new_root is not None else mcts_next.initialize_root()
            )
            mcts_now, mcts_next = mcts_next, mcts_now
        result = board.get_result()
        if result:
            if game_index % 2 == 0:
                if result == board.white:
                    score_previous_model += 1
                else:
                    score_current_model += 1
            else:
                if result == board.white:
                    score_current_model += 1
                else:
                    score_previous_model += 1
        else:
            null_games += 1
    if null_games == ConfigServing.evaluation_games_number:
        return previous_model, 0.5
    score = score_current_model / (ConfigServing.evaluation_games_number - null_games)
    if score >= ConfigServing.replace_min_score:
        return current_model, score
    else:
        return previous_model, score
