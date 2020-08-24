import os
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, Body, Request

from src.config import ConfigGeneral, ConfigServing, ConfigPath
from src.model.tensorflow.train import train
from src.serving.schema import ModelTrainingInputs, ModelTrainingOutputs
from src.serving.example import TrainingExample
from src.serving.evaluate import evaluate_against_last_model
from src.model.tensorflow.train import train_and_report_performance

router = APIRouter()


@router.post("", response_model=ModelTrainingOutputs, name="training")
async def post(
    request: Request,
    inputs: ModelTrainingInputs = Body(..., example=TrainingExample.inputs),
):
    data = inputs.dict()
    run_id = data.pop("run_id")
    states = np.asarray(data.pop("states"))
    policies = np.asarray(data.pop("policies"))
    values = np.asarray(data.pop("values"))
    run_path = os.path.join(ConfigPath.results_path, ConfigGeneral.game, run_id)
    iteration_path = os.path.join(run_path, f"iteration_{request.app.state.iteration}")
    tensorboard_path = os.path.join(run_path, ConfigPath.tensorboard_endpath)
    os.makedirs(iteration_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    request.app.state.number_samples += len(states)

    best_model, loss, updated = train_and_report_performance(
        request.app.state.model,
        states,
        policies,
        values,
        run_path,
        iteration_path,
        tensorboard_path,
        request.app.state.iteration,
        request.app.state.number_samples,
        ConfigServing.training_epochs,
    )
    # instruction to be checked
    request.app.state.model = best_model
    response = {
        "loss": loss,
        "updated": updated,
        "iteration": request.app.state.iteration,
    }
    request.app.state.iteration += 1
    return response
