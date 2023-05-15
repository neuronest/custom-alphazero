import os
from typing import Union

from custom_alphazero.config import ConfigGeneral, ConfigPath


def get_run_path(run_id: str) -> str:
    return os.path.join(ConfigPath.results_dir, ConfigGeneral.game, run_id)


def get_self_play_path(run_id: str) -> str:
    return os.path.join(get_run_path(run_id), ConfigPath.self_play_dir)


def get_training_path(run_id: str) -> str:
    return os.path.join(get_run_path(run_id), ConfigPath.training_dir)


def get_evaluation_path(run_id: str) -> str:
    return os.path.join(get_run_path(run_id), ConfigPath.evaluation_dir)


def get_tensorboard_path(run_id: str) -> str:
    return os.path.join(get_run_path(run_id), ConfigPath.tensorboard_dir)


def get_self_play_iteration_path(
    run_id: str, iteration: Union[str, int], prefix: str = "iteration", sep: str = "_"
) -> str:
    return os.path.join(get_self_play_path(run_id), prefix + sep + str(iteration))


def get_self_play_updated_mcts_path(run_id: str) -> str:
    return os.path.join(get_self_play_path(run_id), ConfigPath.updated_mcts_dir)


def get_self_play_samples_path(run_id: str, iteration: Union[str, int]) -> str:
    return os.path.join(
        get_self_play_iteration_path(run_id, iteration), ConfigPath.samples_file
    )


def get_evaluation_iteration_path(
    run_id: str, iteration: Union[str, int], prefix: str = "iteration", sep: str = "_"
) -> str:
    return os.path.join(get_evaluation_path(run_id), prefix + sep + str(iteration))
