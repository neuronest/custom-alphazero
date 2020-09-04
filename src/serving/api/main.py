from datetime import datetime
from fastapi import FastAPI
from queue import Queue

from src.config import ConfigServing, ConfigPath, ConfigGeneral
from src.utils import set_gpu_index, init_model, create_all_directories
from src.serving.api import run_id, queue, best_model, inference
from src.serving.inference_batch import InferenceBatch

if ConfigGeneral.game == "chess":
    from src.chess.board import Board
elif ConfigGeneral.game == "connect_n":
    from src.connect_n.board import Board

    get_all_possible_moves = Board.get_all_possible_moves
else:
    raise NotImplementedError


def start() -> FastAPI:
    set_gpu_index(ConfigGeneral.serving_gpu_index)
    main_app = FastAPI()
    main_app.state.run_id = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    main_app.state.queue = Queue()
    main_app.state.best_model = init_model()
    main_app.state.inference_batch = InferenceBatch(
        model=main_app.state.best_model, batch_size=ConfigServing.inference_batch_size
    )
    main_app.include_router(
        run_id.router, prefix=ConfigPath.run_id_path, tags=["run-id"]
    )
    main_app.include_router(queue.router, prefix=ConfigPath.queue_path, tags=["queue"])
    main_app.include_router(
        best_model.router, prefix=ConfigPath.best_model_path, tags=["best-model"]
    )
    main_app.include_router(
        inference.router, prefix=ConfigPath.inference_path, tags=["inference"]
    )
    create_all_directories(run_id=main_app.state.run_id)
    return main_app


app = start()
