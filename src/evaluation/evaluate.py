import numpy as np
from typing import Tuple, List, Optional

from src.config import ConfigGeneral, ConfigMCTS, ConfigServing, ConfigSelfPlay
from src.mcts.mcts import MCTS
from src.model.tensorflow.model import PolicyValueModel
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
                    1
                    - (ranked_moves_indexes[board.moves.index(move)] + 1)
                    / len(board.moves)
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
            mcts.search(ConfigSelfPlay.mcts_iterations)
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


def evaluate_two_models(
    model: PolicyValueModel,
    other_model: PolicyValueModel,
    evaluate_with_mcts: bool = False,
    evaluate_with_solver: bool = False,
    deterministic: bool = False,
) -> Tuple[float, Optional[float]]:
    if evaluate_with_solver:
        if evaluate_with_mcts:
            raise NotImplementedError
    all_possible_moves = get_all_possible_moves()
    score_current_model = []
    solver_scores = []  # only used if evaluate_with_mcts is False for now
    for game_index in range(ConfigServing.evaluation_games_number):
        result_current_model_game, solver_scores_game = _single_game_evaluation(
            current_model=model,
            previous_model=other_model,
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
        return 0.5, solver_score
    score = (score_current_model == 1).sum() / (score_current_model != 0).sum()
    if score >= ConfigServing.replace_min_score:
        return score, solver_score
    else:
        return score, solver_score
