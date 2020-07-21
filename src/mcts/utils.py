import numpy as np


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities_sum = probabilities.sum()
    # if there are only zeros, we choose one element randomly to be one
    if probabilities_sum == 0:
        probabilities[np.random.randint(len(probabilities))] = 1.0
        return probabilities
    # keep eventual division per 0 unchanged
    return np.divide(
        probabilities,
        probabilities_sum,
        out=np.zeros_like(probabilities),
        where=probabilities_sum != 0,
    )
