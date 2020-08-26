from typing import List, Tuple
import os

import numpy as np
import tensorflow as tf

from src.config import ConfigServing, ConfigPath, ConfigModel
from src.serving.evaluate import evaluate_against_last_model
from src.model.tensorflow.model import PolicyValueModel
from src.utils import last_saved_model


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
    model: PolicyValueModel,
    inputs: np.ndarray,
    policy_labels: np.ndarray,
    value_labels: np.ndarray,
    run_path: str,
    iteration_path: str,
    tensorboard_path: str,
    iteration: int,
    number_samples: int,
) -> Tuple[PolicyValueModel, float, bool]:
    updated = False
    writer = tf.summary.create_file_writer(tensorboard_path)
    loss = train(
        model,
        inputs,
        [policy_labels, value_labels],
        epochs=ConfigServing.training_epochs,
        batch_size=ConfigServing.batch_size,
    )
    with writer.as_default():
        tf.summary.scalar("loss", loss, step=iteration)
        tf.summary.scalar(
            "number of samples", number_samples, step=iteration,
        )
        tf.summary.scalar("steps", model.steps, step=iteration)
        tf.summary.scalar("learning rate", model.get_learning_rate(), step=iteration)
        writer.flush()
    # if not evaluated the model to train next time will be the one that has trained even if it is weaker
    # if evaluated the best model will take part in the next training
    if (iteration + 1) % ConfigServing.model_checkpoint_frequency == 0:
        # model at max iteration name path will be used for evaluation so it has to be the doing the self-play
        best_model, score = evaluate_against_last_model(
            current_model=model,
            run_path=run_path,
            evaluate_with_mcts=ConfigServing.evaluate_with_mcts,
        )
        if score >= ConfigServing.replace_min_score:
            print(
                f"The current model is better, saving best model trained ..."
            )
        else:
            print("The previous model was better, saving best model...")
        best_model.save_with_meta(iteration_path)
        with writer.as_default():
            tf.summary.scalar(
                "last model winning score",
                score,
                step=iteration // ConfigServing.model_checkpoint_frequency,
            )
            writer.flush()
        updated = score >= ConfigServing.replace_min_score
    else:
        # model not evaluated so the last model that has done the self-play phase is loaded and saved at iteration_path
        # to take part in the next self-play phase
        last_saved_model(run_path).save_with_meta(iteration_path)
    if (iteration + 1) % ConfigServing.samples_checkpoint_frequency == 0:
        print("Saving current samples...")
        np.savez(
            os.path.join(iteration_path, ConfigPath.samples_name),
            states=inputs,
            policies=policy_labels,
            values=value_labels,
        )
    return model, loss, updated
