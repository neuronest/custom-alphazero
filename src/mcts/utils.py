import numpy as np


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities_sum = probabilities.sum()
    # if there are only zeros, we return an uniform distribution
    if probabilities_sum == 0:
        return np.array([1 / len(probabilities)] * len(probabilities))
    # keep eventual division per 0 unchanged
    return np.divide(
        probabilities,
        probabilities_sum,
        out=np.zeros_like(probabilities),
        where=probabilities_sum != 0,
    )
