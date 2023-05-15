import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from custom_alphazero import paths
from custom_alphazero.config import ConfigModel, ConfigServing
from custom_alphazero.evaluation.evaluate import evaluate_two_models
from custom_alphazero.model.tensorflow.model import PolicyValueModel
from custom_alphazero.utils import best_saved_model


def train(
    model: PolicyValueModel,
    inputs: np.ndarray,
    labels: List[np.ndarray],
    batch_size: int,
    epochs: int,
) -> float:
    policy_labels, value_labels = labels
    # Calling fit instead of apply_gradients seems to be preferred for now in TF2
    # https://github.com/tensorflow/tensorflow/issues/35585
    history = model.fit(
        inputs,
        [policy_labels, value_labels],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        shuffle=False,
    )
    model.steps += epochs * (np.ceil(inputs.shape[0] / batch_size).astype(int))
    loss = history.history["loss"][-1]
    new_learning_rate = {
        ConfigModel.learning_rates[learning_rate]
        for learning_rate in ConfigModel.learning_rates
        if model.steps in learning_rate
    }
    model.update_learning_rate(
        new_learning_rate.pop()
        if len(new_learning_rate)
        else ConfigModel.minimum_learning_rate
    )
    return loss


def train_and_report(
    run_id: str,
    last_model: PolicyValueModel,
    states_batch: np.ndarray,
    policies_batch: np.ndarray,
    values_batch: np.ndarray,
    training_iteration: int,
    evaluation_iteration: int,
) -> Tuple[bool, bool]:
    tensorboard_path = paths.get_tensorboard_path(run_id)
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = tf.summary.create_file_writer(tensorboard_path)
    loss = train(
        last_model,
        states_batch,
        [policies_batch, values_batch],
        epochs=ConfigModel.training_epochs,
        batch_size=ConfigModel.batch_size,
    )
    if (training_iteration + 1) % ConfigServing.model_checkpoint_frequency == 0:
        last_model.save_with_meta(paths.get_training_path(run_id))
    with writer.as_default():
        tf.summary.scalar("loss", loss, step=training_iteration)
        tf.summary.scalar("steps", last_model.steps, step=training_iteration)
        tf.summary.scalar(
            "learning rate", last_model.get_learning_rate(), step=training_iteration
        )
        writer.flush()
    is_evaluated = (
        training_iteration + 1
    ) % ConfigServing.model_evaluation_frequency == 0
    if is_evaluated:
        previous_model = best_saved_model(
            run_id
        )  # best model so far, is going to be challenged in this iteration
        new_evaluation_path = paths.get_evaluation_iteration_path(
            run_id, evaluation_iteration
        )
        os.makedirs(new_evaluation_path, exist_ok=True)
        score, solver_score = evaluate_two_models(
            model=last_model,
            other_model=previous_model,
            evaluate_with_mcts=ConfigServing.evaluate_with_mcts,
            evaluate_with_solver=ConfigServing.evaluate_with_solver,
        )
        is_updated = score >= ConfigServing.replace_min_score
        if is_updated:
            print(
                f"The current model is better, saving best model trained at {new_evaluation_path}..."
            )
            last_model.save_with_meta(new_evaluation_path)
        else:
            print(
                f"The previous model was better, saving best model at {new_evaluation_path}..."
            )
            previous_model.save_with_meta(new_evaluation_path)
        with writer.as_default():
            tf.summary.scalar(
                "last model winning score",
                score,
                step=evaluation_iteration,
            )
            if solver_score is not None:
                tf.summary.scalar(
                    "solver score", solver_score, step=evaluation_iteration
                )
            writer.flush()
    else:
        is_updated = False
    return is_evaluated, is_updated
