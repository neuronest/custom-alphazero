import time
import numpy as np
from typing import Tuple

from src.config import ConfigGeneral, ConfigServing, ConfigModel
from src.utils import set_gpu_index, last_saved_model
from src.serving.factory import retrieve_queue, get_run_id, update_best_model
from src.model.tensorflow.train import train_and_report


def _append_and_limit_queues(
    local_states_queue: np.ndarray,
    local_policies_queue: np.ndarray,
    local_values_queue: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _limit_queue_size(queue: np.ndarray, max_size: int) -> np.ndarray:
        return queue[-max_size:]

    states, policies, values = retrieve_queue()
    if all(array.size for array in [states, policies, values]):
        local_states_queue = _limit_queue_size(
            np.vstack([local_states_queue, states]),
            max_size=ConfigServing.samples_queue_size,
        )
        local_policies_queue = _limit_queue_size(
            np.vstack([local_policies_queue, policies]),
            max_size=ConfigServing.samples_queue_size,
        )
        local_values_queue = _limit_queue_size(
            np.append(local_values_queue, values),
            max_size=ConfigServing.samples_queue_size,
        )
    return local_states_queue, local_policies_queue, local_values_queue


def training_loop(run_id: str):
    last_model = last_saved_model(run_id)
    local_states_queue, local_policies_queue, local_values_queue = (
        np.empty((0, *last_model.input_dim)),
        np.empty((0, last_model.action_space)),
        np.empty(0),
    )
    training_iteration, evaluation_iteration = 0, 0
    while True:
        (
            local_states_queue,
            local_policies_queue,
            local_values_queue,
        ) = _append_and_limit_queues(
            local_states_queue, local_policies_queue, local_values_queue,
        )
        if len(local_states_queue) >= ConfigServing.minimum_training_size:
            sample_indexes = np.random.choice(
                len(local_states_queue), ConfigModel.batch_size, replace=False,
            )
            states_batch, policies_batch, values_batch = (
                local_states_queue[sample_indexes],
                local_policies_queue[sample_indexes],
                local_values_queue[sample_indexes],
            )
            is_evaluated, is_updated = train_and_report(
                run_id,
                last_model,
                states_batch,
                policies_batch,
                values_batch,
                training_iteration,
                evaluation_iteration,
            )
            if is_updated:
                update_best_model()
            if is_evaluated:
                evaluation_iteration += 1
            training_iteration += 1
        time.sleep(ConfigServing.training_loop_sleep_time)


if __name__ == "__main__":
    set_gpu_index(ConfigGeneral.training_gpu_index)
    run_id = get_run_id()
    assert run_id is not None, "Could not get the run if from the server"
    print(f"Starting train with id={run_id}")
    training_loop(run_id)
