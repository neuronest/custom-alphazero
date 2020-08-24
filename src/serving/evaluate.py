import os
import numpy as np
from functools import partial
from typing import Tuple, List, Optional

from src.config import ConfigGeneral, ConfigMCTS, ConfigServing, ConfigModel
from src.mcts.mcts import MCTS
from src.model.tensorflow.model import PolicyValueModel
from src.serving.factory import init_model
from src.mcts.utils import normalize_probabilities
from src.exact_solvers.c4_exact_solver import exact_ranked_moves_and_value

if ConfigGeneral.game == "chess":
    from src.chess.board import Board
    from src.chess.move import Move
    from src.chess.utils import get_all_possible_moves
elif ConfigGeneral.game == "connect_n":
    from src.connect_n.board import Board
    from src.connect_n.move import Move

    get_all_possible_moves = Board.get_all_possible_moves
else:
    raise NotImplementedError


def get_last_iteration_name(
    run_path: str, prefix: str = "iteration", sep: str = "_"
) -> str:
    def _is_correct_iteration_directory(
        directory: str, run_path: str, prefix: str
    ) -> bool:
        return directory.startswith(prefix) and os.path.exists(
            os.path.join(run_path, directory, ConfigModel.model_meta)
        )

    return max(
        filter(
            partial(_is_correct_iteration_directory, run_path=run_path, prefix=prefix),
            os.listdir(run_path),
        ),
        key=lambda x: int(x.split(sep)[-1]),
    )


def _single_game_evaluation(
    current_model: PolicyValueModel,
    previous_model: PolicyValueModel,
    game_index: int,
    all_possible_moves: List[Move],
    evaluate_with_mcts: bool,
    evaluate_with_solver: bool,
    deterministic: bool,
) -> Tuple[int, Optional[List[float]]]:
    solver_scores = []
    model = current_model if game_index % 2 == 0 else previous_model
    if not evaluate_with_mcts:
        board = Board()
        while not board.is_game_over():
            probabilities, _ = model(np.expand_dims(board.full_state, axis=0))
            probabilities = probabilities.numpy().ravel()
            legal_probabilities = probabilities[
                board.legal_moves_mask(all_possible_moves)
            ]
            legal_probabilities = normalize_probabilities(legal_probabilities)
            if deterministic:
                move = board.moves[int(np.argmax(legal_probabilities))]
            else:
                move = np.random.choice(board.moves, 1, p=legal_probabilities).item()
            if evaluate_with_solver and model is current_model:
                ranked_moves_indexes, _ = exact_ranked_moves_and_value(board)
                solver_scores.append(
                    1 - ranked_moves_indexes[board.moves.index(move)] / len(board.moves)
                )
            board.play(move, keep_same_player=True)
            if not board.is_game_over():
                model = previous_model if model is current_model else current_model
    else:
        mcts = MCTS(
            board=Board(),
            all_possible_moves=all_possible_moves,
            concurrency=False,
            model=model,
        )
        while not mcts.board.is_game_over():
            mcts.search(ConfigGeneral.mcts_iterations)
            greedy = mcts.board.fullmove_number > ConfigMCTS.index_move_greedy
            _ = mcts.play(greedy, deterministic=deterministic)
            if not mcts.board.is_game_over():
                model = previous_model if mcts.model is current_model else current_model
                mcts = MCTS(
                    board=mcts.board,
                    all_possible_moves=all_possible_moves,
                    concurrency=False,
                    model=model,
                )
        board = mcts.board
    result = board.get_result(keep_same_player=True)
    if result:
        result_current_model = 1 if current_model is model else -1
    else:
        result_current_model = 0
    return result_current_model, solver_scores


def evaluate_against_last_model(
    current_model: PolicyValueModel,
    previous_model: Optional[PolicyValueModel] = None,
    run_path: Optional[str] = None,
    evaluate_with_mcts: bool = False,
    evaluate_with_solver: bool = False,
    deterministic: bool = False,
) -> Tuple[PolicyValueModel, float, Optional[float]]:
    if evaluate_with_solver:
        if evaluate_with_mcts:
            raise NotImplementedError
    if previous_model is None:
        assert run_path is not None
        try:
            max_iteration_name = get_last_iteration_name(run_path)
            previous_model = init_model(os.path.join(run_path, max_iteration_name))
        except ValueError:
            previous_model = init_model()
    all_possible_moves = get_all_possible_moves()
    score_current_model = []
    solver_scores = []  # only used if evaluate_with_mcts is False for now
    for game_index in range(ConfigServing.evaluation_games_number):
        result_current_model_game, solver_scores_game = _single_game_evaluation(
            current_model=current_model,
            previous_model=previous_model,
            game_index=game_index,
            all_possible_moves=all_possible_moves,
            evaluate_with_mcts=evaluate_with_mcts,
            evaluate_with_solver=evaluate_with_solver,
            deterministic=deterministic,
        )
        solver_scores.extend(solver_scores_game)
        score_current_model.append(result_current_model_game)
    if evaluate_with_solver:
        if len(solver_scores):
            solver_score = np.mean(solver_scores)
        else:
            # solver score not implemented for evaluation with MCTS for now
            solver_score = None
    else:
        solver_score = None
    score_current_model = np.array(score_current_model)
    if np.all(score_current_model == 0):
        # there are only draws, we choose to return the previous model with a 50-50 score
        return previous_model, 0.5, solver_score
    score = (score_current_model == 1).sum() / (score_current_model != 0).sum()
    if score >= ConfigServing.replace_min_score:
        return current_model, score, solver_score
    else:
        return previous_model, score, solver_score
