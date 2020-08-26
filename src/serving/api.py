from fastapi import FastAPI

from src.config import ConfigServing, ConfigPath, ConfigGeneral
from src.utils import init_model
from src.serving.factory import InferenceBatch
from src.serving import inference, training

if ConfigGeneral.game == "chess":
    from src.chess.board import Board
elif ConfigGeneral.game == "connect_n":
    from src.connect_n.board import Board

    get_all_possible_moves = Board.get_all_possible_moves
else:
    raise NotImplementedError


def start() -> FastAPI:
    main_app = FastAPI()
    main_app.state.last_model = init_model()
    main_app.state.best_model = init_model()
    main_app.state.inference_batch = InferenceBatch(
        model=main_app.state.best_model, batch_size=ConfigServing.inference_batch_size
    )
    main_app.state.iteration = 0
    main_app.state.number_samples = 0
    main_app.include_router(
        inference.router, tags=["inference"], prefix=ConfigPath.inference_path
    )
    main_app.include_router(
        training.router, tags=["training"], prefix=ConfigPath.training_path
    )
    return main_app


app = start()
