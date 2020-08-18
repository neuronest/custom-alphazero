import os
import numpy as np
from typing import Tuple, Optional

from src.config import ConfigGeneral, ConfigMCTS, ConfigServing
from src.mcts.mcts import MCTS
from src.model.tensorflow.model import PolicyValueModel
from src.serving.factory import init_model
from src.mcts.utils import normalize_probabilities

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
    run_path: Optional[str] = None,
    evaluate_with_mcts: bool = True,
    deterministic: bool = False
) -> Tuple[PolicyValueModel, float]:
    if previous_model is None:
        assert run_path is not None
        try:
            max_iteration_name = max(
                os.listdir(run_path), key=lambda x: int(x.split("_")[-1])
            )
            previous_model = init_model(os.path.join(run_path, max_iteration_name))
        except ValueError:
            previous_model = init_model()
    score_current_model, null_games, index_model = 0, 0, 0
    for game_index in range(ConfigServing.evaluation_games_number):
        model = current_model if game_index % 2 == 0 else previous_model
        if not evaluate_with_mcts:
            board = Board()
            while not board.is_game_over():
                probabilities, _ = model(np.expand_dims(board.full_state, axis=0))
                probabilities = probabilities.numpy().ravel()
                probabilities = probabilities[
                    board.legal_moves_mask(get_all_possible_moves())
                ]
                probabilities = normalize_probabilities(probabilities)
                if deterministic:
                    move = board.moves[int(np.argmax(probabilities))]
                else:
                    move = np.random.choice(board.moves, 1, p=probabilities).item()
                board.play(move, keep_same_player=True)
                if not board.is_game_over():
                    model = previous_model if model is current_model else current_model
        else:
            mcts = MCTS(
                board=Board(),
                all_possible_moves=get_all_possible_moves(),
                concurrency=False,
                model=model
            )
            while not mcts.board.is_game_over():
                mcts.search(ConfigGeneral.mcts_iterations)
                greedy = mcts.board.fullmove_number > ConfigMCTS.index_move_greedy
                _ = mcts.play(greedy, deterministic=deterministic)
                if not mcts.board.is_game_over():
                    model = previous_model if mcts.model is current_model else current_model
                    mcts = MCTS(
                        board=mcts.board,
                        all_possible_moves=get_all_possible_moves(),
                        concurrency=False,
                        model=model
                    )
            board = mcts.board
        result = board.get_result(keep_same_player=True)
        if result:
            if current_model is model:
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
