from typing import List, Tuple
import os

import numpy as np
import tensorflow as tf

from src.config import ConfigServing, ConfigPath, ConfigModel
from src.serving.evaluate import evaluate_against_last_model
from src.model.tensorflow.model import PolicyValueModel


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
