from fastapi import FastAPI

from src.config import ConfigServing
from src.serving.factory import InferenceBatch, init_model
from src.serving import inference, training


def start() -> FastAPI:
    main_app = FastAPI()
    main_app.state.model = init_model()
    main_app.state.inference_batch = InferenceBatch(
        model=main_app.state.model, batch_size=ConfigServing.inference_batch_size
    )
    main_app.state.iteration = 0
    main_app.state.number_samples = 0
    main_app.include_router(
        inference.router, tags=["inference"], prefix=ConfigServing.inference_path
    )
    main_app.include_router(
        training.router, tags=["training"], prefix=ConfigServing.training_path
    )
    return main_app


app = start()
