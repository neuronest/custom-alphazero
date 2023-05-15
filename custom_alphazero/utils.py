import json
import multiprocessing
import os
from functools import partial
from typing import Optional, Union

from custom_alphazero import paths
from custom_alphazero.config import ConfigGeneral, ConfigPath
from custom_alphazero.mcts.mcts import MCTS
from custom_alphazero.model.tensorflow.model import PolicyValueModel
from custom_alphazero.visualize_mcts import MctsVisualizer

if ConfigGeneral.game == "chess":
    from custom_alphazero.chess.board import Board
    from custom_alphazero.chess.utils import get_all_possible_moves
elif ConfigGeneral.game == "connect_n":
    from custom_alphazero.connect_n.board import Board

    get_all_possible_moves = Board.get_all_possible_moves
else:
    raise NotImplementedError


def set_gpu_index(gpu_index: Union[int, str]):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index


def create_all_directories(run_id: str):
    for directory in [
        paths.get_self_play_updated_mcts_path(run_id),
        paths.get_training_path(run_id),
        paths.get_evaluation_path(run_id),
        paths.get_tensorboard_path(run_id),
    ]:
        os.makedirs(directory, exist_ok=True)


def reset_plays_inferences_dict() -> dict:
    return {} if ConfigGeneral.mono_process else multiprocessing.Manager().dict()


def init_model(path: Optional[str] = None) -> PolicyValueModel:
    model = PolicyValueModel(
        input_dim=Board().full_state.shape, action_space=len(get_all_possible_moves())
    )
    if path is not None:
        model.load_with_meta(path)
    return model


def last_saved_model(run_id: str) -> PolicyValueModel:
    training_path = paths.get_training_path(run_id)
    if os.path.exists(os.path.join(training_path, ConfigPath.model_success)):
        model = init_model(training_path)
    else:
        print(
            f"Warning: no model found at {training_path}, initializing last model with random weights\n"
            f"This warning can safely be ignored if the run is just beginning"
        )
        model = init_model()
    return model


def best_saved_model(run_id: str) -> PolicyValueModel:
    # max_iteration_name folder should contain the last evaluation model,
    # which is the best model so far
    evaluation_path = paths.get_evaluation_path(run_id)
    max_iteration_name = last_evaluation_iteration_name(evaluation_path)

    if max_iteration_name is None:
        print(
            f"Warning: no model found at {evaluation_path}, "
            f"initializing best model with random weights\n"
            f"This warning can safely be ignored if the run is just beginning"
        )
        return init_model()

    return init_model(os.path.join(evaluation_path, max_iteration_name))


def best_saved_model_hash(run_id: str) -> Optional[str]:
    # max_iteration_name folder should contain the last evaluation model,
    # which is the best model so far
    evaluation_path = paths.get_evaluation_path(run_id)

    model_hash = None
    if os.path.exists(evaluation_path):
        max_iteration_name = last_evaluation_iteration_name(evaluation_path)

        if os.path.exists(
            os.path.join(evaluation_path, max_iteration_name, ConfigPath.model_meta)
        ):
            with open(
                os.path.join(
                    evaluation_path, max_iteration_name, ConfigPath.model_meta
                ),
                "r",
            ) as fp:
                model_hash = json.loads(fp.read()).get("hash")

    if model_hash is None:
        print(
            f"Warning: no model hash found at {evaluation_path}, returning None instead\n"
            f"This warning can safely be ignored if the run is just beginning"
        )

    return model_hash


def last_evaluation_iteration_name(
    evaluation_path: str, prefix: str = "iteration", sep: str = "_"
) -> Optional[str]:
    def _is_correct_iteration_directory(
        directory: str, evaluation_path: str, prefix: str
    ) -> bool:
        return directory.startswith(prefix) and os.path.exists(
            os.path.join(evaluation_path, directory, ConfigPath.model_success)
        )

    if not os.path.exists(evaluation_path):
        return None

    return max(
        filter(
            partial(
                _is_correct_iteration_directory,
                evaluation_path=evaluation_path,
                prefix=prefix,
            ),
            os.listdir(evaluation_path),
        ),
        key=lambda x: int(x.split(sep)[-1]),
    )


def visualize_mcts_iteration(
    mcts_visualizer: MctsVisualizer,
    mcts_tree: MCTS,
    iteration: int,
    run_id: str,
) -> None:
    mcts_name_full, mcts_name_light = (
        f"mcts_iteration_{iteration}_full",
        f"mcts_iteration_{iteration}_light",
    )
    iteration_path = paths.get_self_play_iteration_path(run_id, iteration)
    updated_mcts_path = paths.get_self_play_updated_mcts_path(run_id)
    # by default save only mcts played edges so that save is fast
    mcts_visualizer.build_mcts_graph(
        mcts_tree.root, mcts_name=mcts_name_light, remove_unplayed_edge=True
    )
    mcts_visualizer.save_as_pdf(directory=iteration_path)
    # if mcts generated by an updated model then save it also at runpath level to visualize those trees more easily
    if mcts_visualizer.is_updated:
        # save light mcts also under updated mcts directory path
        mcts_visualizer.save_as_pdf(directory=updated_mcts_path)
        # save full mcts if it is an updated tree, even if save is slower
        mcts_visualizer.build_mcts_graph(
            mcts_tree.root, mcts_name=mcts_name_full, remove_unplayed_edge=False
        )
        mcts_visualizer.save_as_pdf(directory=updated_mcts_path)
