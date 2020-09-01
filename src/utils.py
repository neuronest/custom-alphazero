import os
from typing import Optional
from functools import partial

from src.config import ConfigModel, ConfigPath, ConfigGeneral
from src.mcts.mcts import MCTS
from src.visualize_mcts import MctsVisualizer
from src.model.tensorflow.model import PolicyValueModel

if ConfigGeneral.game == "chess":
    from src.chess.board import Board
    from src.chess.utils import get_all_possible_moves
elif ConfigGeneral.game == "connect_n":
    from src.connect_n.board import Board

    get_all_possible_moves = Board.get_all_possible_moves
else:
    raise NotImplementedError


# class intended to reproduce the same global behavior as request.app.state on fastapi server
class LocalState:
    def __init__(self):
        self.number_samples = 0
        self.iteration = 0
        self.last_model = init_model()
        self.best_model = init_model()


def init_model(path: Optional[str] = None) -> PolicyValueModel:
    model = PolicyValueModel(
        input_dim=Board().full_state.shape, action_space=len(get_all_possible_moves())
    )
    if path is not None:
        model.load_with_meta(path)
    return model


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


def visualize_mcts_iteration(
    mcts_visualizer: MctsVisualizer,
    mcts_tree: MCTS,
    mcts_name: str,
    iteration_path: str,
    run_id: Optional[str] = None,
) -> None:
    # by default save only mcts played edges so that save is fast
    mcts_visualizer.build_mcts_graph(
        mcts_tree.root, mcts_name=f"{mcts_name}_light", remove_unplayed_edge=True
    )
    mcts_visualizer.save_as_pdf(directory=iteration_path)

    # if mcts generated by an updated model then save it also at runpath level to visualize those trees more easily
    if mcts_visualizer.is_updated:
        assert run_id is not None
        # save light mcts also under updated mcts directory path
        updated_mcts_dir_path = os.path.join(
            ConfigPath.results_path,
            ConfigGeneral.game,
            run_id,
            ConfigPath.updated_mcts_dir,
        )
        mcts_visualizer.save_as_pdf(directory=updated_mcts_dir_path)
        # save full mcts if it is an updated tree, even if save is slower
        mcts_visualizer.build_mcts_graph(
            mcts_tree.root, mcts_name=f"{mcts_name}_full", remove_unplayed_edge=False
        )
        mcts_visualizer.save_as_pdf(directory=updated_mcts_dir_path)
