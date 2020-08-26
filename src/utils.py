import os
from functools import partial

from src.config import ConfigModel
from src.serving.factory import init_model
from src.model.tensorflow.model import PolicyValueModel


def last_saved_model(run_path: str) -> PolicyValueModel:
    try:
        # max_iteration_name folder should contain last best saved model
        max_iteration_name = last_iteration_name(run_path)
        last_best_saved_model = init_model(os.path.join(run_path, max_iteration_name))
    except (ValueError, FileNotFoundError):
        print(f"Warning: {run_path} not found, initializing model with random weights")
        last_best_saved_model = init_model()
    return last_best_saved_model


def last_iteration_name(
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