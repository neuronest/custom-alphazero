import time
import numpy as np
from typing import Tuple, Optional, List, Any

from sklearn.model_selection import train_test_split

from src.config import ConfigGeneral, ConfigServing, ConfigModel
from src.utils import set_gpu_index, last_saved_model
from src.serving.factory import retrieve_queue, get_run_id, update_best_model
from src.model.tensorflow.train import train_and_report


def _append_and_limit_queues(
    states_queue: np.ndarray,
    policies_queue: np.ndarray,
    values_queue: np.ndarray,
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    max_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _limit_queue_size(queue: np.ndarray, max_size: int) -> np.ndarray:
        return queue[-max_size:]

    if all(array.size > 0 for array in [states, policies, values]):
        states_queue = _limit_queue_size(
            np.vstack([states_queue, states]), max_size=max_size,
        )
        policies_queue = _limit_queue_size(
            np.vstack([policies_queue, policies]), max_size=max_size,
        )
        values_queue = _limit_queue_size(
            np.append(values_queue, values), max_size=max_size,
        )
    return states_queue, policies_queue, values_queue


def update_training_queues(
    queue: Tuple[np.ndarray, np.ndarray, np.ndarray],
    val_queue: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    val_proportion: float = ConfigServing.val_proportion,
    samples_queue_size: int = ConfigServing.samples_queue_size,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    states, policies, values = retrieve_queue()
    if any(array.size == 0 for array in [states, policies, values]):
        return queue, val_queue
    elif val_queue:
        (
            states_tr,
            states_val,
            policies_tr,
            policies_val,
            values_tr,
            values_val,
        ) = train_test_split(states, policies, values, test_size=val_proportion)
        queue = _append_and_limit_queues(
            *queue, states_tr, policies_tr, values_tr, max_size=samples_queue_size,
        )
        val_queue = _append_and_limit_queues(
            *val_queue,
            states_val,
            policies_val,
            values_val,
            max_size=int(val_proportion * samples_queue_size),
        )
    else:
        queue = _append_and_limit_queues(
            *queue, states, policies, values, max_size=samples_queue_size,
        )
    return queue, val_queue


def data_batch_from_queue(
    queue: Tuple[np.ndarray, np.ndarray, np.ndarray], batch_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    states_queue, policies_queue, values_queue = queue
    sample_indexes = np.random.choice(len(states_queue), batch_size, replace=False)
    return (
        states_queue[sample_indexes],
        policies_queue[sample_indexes],
        values_queue[sample_indexes],
    )


def initialize_training_queues(
    input_dim: Any, action_space: int, with_validation: bool
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    states_queue, policies_queue, values_queue = (
        np.empty((0, *input_dim)),
        np.empty((0, action_space)),
        np.empty(0),
    )
    queue = states_queue, policies_queue, values_queue
    if with_validation:
        val_states_queue, val_policies_queue, val_values_queue = (
            np.empty((0, *input_dim)),
            np.empty((0, action_space)),
            np.empty(0),
        )
        val_queue = val_states_queue, val_policies_queue, val_values_queue
    else:
        val_queue = None
    return queue, val_queue


def queue_as_x_y_repr(
    queue: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    states_queue, policies_queue, values_queue = queue
    return states_queue, [policies_queue, values_queue]


def training_loop(
    run_id: str,
    batch_size: int = ConfigModel.batch_size,
    val_proportion: float = ConfigServing.val_proportion,
    minimum_training_size: int = ConfigServing.minimum_training_size,
    training_loop_sleep_time: float = ConfigServing.training_loop_sleep_time,
) -> None:
    with_validation = val_proportion > 0
    last_model = last_saved_model(run_id)
    local_queue, val_local_queue = initialize_training_queues(
        last_model.input_dim, last_model.action_space, with_validation
    )
    training_iteration, evaluation_iteration = 0, 0
    while True:
        local_queue, val_local_queue = update_training_queues(
            queue=local_queue, val_queue=val_local_queue
        )
        if len(local_queue[0]) <= minimum_training_size:
            time.sleep(training_loop_sleep_time)
            continue
        elif with_validation and len(val_local_queue[0]) >= batch_size:
            validation_data = queue_as_x_y_repr(
                data_batch_from_queue(val_local_queue, batch_size)
            )
        else:
            validation_data = None
        states_batch, policies_batch, values_batch = data_batch_from_queue(
            local_queue, batch_size
        )
        is_evaluated, is_updated = train_and_report(
            run_id,
            last_model,
            states_batch,
            policies_batch,
            values_batch,
            training_iteration,
            evaluation_iteration,
            validation_data=validation_data,
        )
        if is_updated:
            update_best_model()
        if is_evaluated:
            evaluation_iteration += 1
        training_iteration += 1


if __name__ == "__main__":
    set_gpu_index(ConfigGeneral.training_gpu_index)
    run_id = get_run_id()
    assert run_id is not None, "Could not get the run if from the server"
    print(f"Starting train with id={run_id}")
    training_loop(run_id)
