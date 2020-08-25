import os
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, Body, Request

from src.config import ConfigGeneral, ConfigServing, ConfigPath
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
    model = request.app.state.model
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
    updated = False
    request.app.state.number_samples += len(states)
    writer = tf.summary.create_file_writer(tensorboard_path)
    loss = train(
        model,
        states,
        [policies, values],
        epochs=ConfigServing.training_epochs,
        batch_size=ConfigServing.batch_size,
    )
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
    if request.app.state.iteration % ConfigServing.model_checkpoint_frequency == 0:
        best_model, score, solver_score = evaluate_against_last_model(
            current_model=model,
            run_path=run_path,
            evaluate_with_mcts=ConfigServing.evaluate_with_mcts,
            evaluate_with_solver=ConfigServing.evaluate_with_solver,
        )
        if solver_score is not None:
            with writer.as_default():
                tf.summary.scalar(
                    "solver score", solver_score, step=request.app.state.iteration
                )
                writer.flush()
        if score >= ConfigServing.replace_min_score:
            print("The current model is better, saving...")
        else:
            print("The previous model was better, saving...")
        best_model.save_with_meta(iteration_path)
        with writer.as_default():
            tf.summary.scalar(
                "last model winning score",
                score,
                step=request.app.state.iteration
                // ConfigServing.model_checkpoint_frequency,
            )
            writer.flush()
        updated = score >= ConfigServing.replace_min_score
        request.app.state.model = best_model
    if request.app.state.iteration % ConfigServing.samples_checkpoint_frequency == 0:
        print("Saving current samples...")
        np.savez(
            os.path.join(iteration_path, ConfigPath.samples_name),
            states=states,
            policies=policies,
            values=values,
        )
    response = {
        "loss": loss,
        "updated": updated,
        "iteration": request.app.state.iteration,
    }
    request.app.state.iteration += 1
    return response
