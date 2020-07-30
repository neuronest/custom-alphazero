import os
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, Body, Request

from src.config import ConfigGeneral, ConfigServing
from src.model.tensorflow.model import train
from src.serving.schema import ModelTrainingInputs, ModelTrainingOutputs
from src.serving.example import TrainingExample
from src.serving.evaluate import evaluate_against_last_model

router = APIRouter()


@router.post("", response_model=ModelTrainingOutputs, name="training")
async def post(
    request: Request,
    inputs: ModelTrainingInputs = Body(..., example=TrainingExample.inputs),
):
    os.makedirs(
        os.path.join(ConfigServing.results_path, ConfigGeneral.game), exist_ok=True
    )
    os.makedirs(ConfigServing.logs_path, exist_ok=True)
    writer = tf.summary.create_file_writer(
        os.path.join(ConfigServing.logs_path, ConfigGeneral.game)
    )
    model = request.app.state.model
    data = inputs.dict()
    states = np.asarray(data.pop("states"))
    policies = np.asarray(data.pop("policies"))
    values = np.asarray(data.pop("values"))
    loss = train(
        model,
        states,
        [policies, values],
        epochs=ConfigServing.training_epochs,
        batch_size=ConfigServing.batch_size,
    )
    request.app.state.iteration += 1
    request.app.state.number_samples += len(states)
    response = {"loss": loss, "updated": False}
    with writer.as_default():
        tf.summary.scalar("loss", loss, step=request.app.state.iteration)
        tf.summary.scalar(
            "number of samples",
            request.app.state.number_samples,
            step=request.app.state.iteration,
        )
        tf.summary.scalar("steps", model.steps, step=request.app.state.iteration)
        tf.summary.scalar(
            "learning rate", model.get_learning_rate(), step=request.app.state.iteration
        )
        writer.flush()
    outputs_path = os.path.join(
        ConfigServing.results_path,
        ConfigGeneral.game,
        "iteration_{}".format(request.app.state.iteration),
    )
    if request.app.state.iteration % ConfigServing.model_checkpoint_frequency == 0:
        best_model, score = evaluate_against_last_model(
            current_model=model,
            path=os.path.join(ConfigServing.results_path, ConfigGeneral.game),
        )
        os.makedirs(outputs_path, exist_ok=True)
        if score >= ConfigServing.replace_min_score:
            print("The current model is better, saving...")
        else:
            print("The previous model was better, saving...")
        best_model.save_with_meta(outputs_path)
        with writer.as_default():
            tf.summary.scalar(
                "last model winning score",
                score,
                step=request.app.state.iteration
                // ConfigServing.model_checkpoint_frequency,
            )
            writer.flush()
        response["updated"] = score >= ConfigServing.replace_min_score
    if request.app.state.iteration % ConfigServing.samples_checkpoint_frequency == 0:
        os.makedirs(outputs_path, exist_ok=True)
        print("Saving current samples...")
        for sample, sample_type in zip(
            [states, policies, values], ["states", "policies", "values"]
        ):
            np.save(
                os.path.join(outputs_path, sample_type), sample,
            )
    return response
