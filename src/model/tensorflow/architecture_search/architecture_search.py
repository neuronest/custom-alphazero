import os

from sklearn.model_selection import train_test_split
import ray
from ray.tune import run, Trainable
from ray.tune.schedulers import PopulationBasedTraining
import tensorflow as tf
import numpy as np

from src.utils import init_model, set_gpu_index, load_samples
from src.config import ConfigArchiSearch
from src.model.tensorflow.base_layers import policy_loss, value_loss
from src.model.tensorflow.architecture_search.render import report_from_ray_analysis
from src.model.tensorflow.architecture_search.utils import (
    save_architecture_search,
    report_filename_with_iterations,
)
from src import paths


def data_to_search_from(run_id, iteration_begin, iteration_end, p_val=0.25):
    for i, iteration in enumerate(range(iteration_begin, iteration_end + 1)):
        states_tmp, policies_tmp, rewards_tmp = load_samples(
            paths.get_self_play_samples_path(run_id, iteration)
        )
        if i == 0:
            states, policies, rewards = states_tmp, policies_tmp, rewards_tmp
        else:
            states = np.concatenate([states, states_tmp], axis=0)
            policies = np.concatenate([policies, policies_tmp], axis=0)
            rewards = np.concatenate([rewards, rewards_tmp], axis=0)
    (
        states_tr,
        states_val,
        policies_tr,
        policies_val,
        rewards_tr,
        rewards_val,
    ) = train_test_split(states, policies, rewards, test_size=p_val, random_state=42)

    x_tr, y_tr = states_tr, [policies_tr, rewards_tr]
    x_val, y_val = states_val, [policies_val, rewards_val]
    return x_tr, y_tr, x_val, y_val


x_tr, y_tr, x_val, y_val = data_to_search_from(
    run_id=ConfigArchiSearch.run_id,
    iteration_begin=ConfigArchiSearch.data_iteration_start,
    iteration_end=ConfigArchiSearch.data_iteration_end,
    p_val=ConfigArchiSearch.validation_percentage / 100,
)


class ArchitectureSearcher(Trainable):
    def setup(self, config):
        # config contains only arguments found in the constructor
        self.model = init_model(**config)

    def step(self):
        _ = self.model.fit(
            x_tr, y_tr, batch_size=16, epochs=1, verbose=1, shuffle=True,
        )
        global_loss, _, _ = self.model.evaluate(x_val, y_val, verbose=0)
        return {"global_loss": global_loss}

    def save_checkpoint(self, checkpoint_dir):
        self.model.save(checkpoint_dir)
        return checkpoint_dir

    def load_checkpoint(self, path):
        self.model = tf.keras.models.load_model(
            path, custom_objects={"policy_loss": policy_loss, "value_loss": value_loss},
        )

    # implement mutations taking effect
    # if not implemented, ray may possibly fall into infinite loop
    def reset_config(self, new_config):
        self.model.optimizer.learning_rate.assign(new_config["maximum_learning_rate"])
        self.model.optimizer.momentum.assign(new_config["momentum"])
        self.config = new_config
        return True


def search_settings() -> dict:
    search_settings = {
        # if resources not available some trials may never end
        "resources_per_trial": ConfigArchiSearch.resources_per_trial,
        "stop": ConfigArchiSearch.search_stop_criterion,
        "config": ConfigArchiSearch.hyperparam_config,
        "num_samples": ConfigArchiSearch.num_samples_from_config,
    }
    return search_settings


def initialize_search(running_mode: str, max_gb_memory=16) -> None:
    def _gb_to_bytes(nb_gb: int) -> int:
        return nb_gb * 1000 * 1024 * 1024

    if running_mode == "debug":
        ray.init(local_mode=True, memory=_gb_to_bytes(max_gb_memory))
    else:
        ray.init(memory=_gb_to_bytes(max_gb_memory))


def handle_search_end(working_dir_before_search: str) -> None:
    # ray has changed the working directory
    if os.getcwd() != working_dir_before_search:
        # reset working directory to the one we want and that has been changed by ray
        os.chdir(working_dir_before_search)


if __name__ == "__main__":
    print(
        f"{os.linesep}Architecture search is starting from run {ConfigArchiSearch.run_id}, iteration "
        f"{ConfigArchiSearch.data_iteration_start} to {ConfigArchiSearch.data_iteration_end}, "
        f"with {len(x_tr) + len(x_val)} samples{os.linesep}"
    )
    set_gpu_index(ConfigArchiSearch.search_gpu_index)
    # check working dir here
    search_working_dir = os.getcwd()
    initialize_search(
        ConfigArchiSearch.running_mode, max_gb_memory=ConfigArchiSearch.max_gb_memory,
    )
    archi_searcher = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="global_loss",
        mode=ConfigArchiSearch.search_mode,
        # perturbation_interval throws an error if set too low
        perturbation_interval=ConfigArchiSearch.perturbation_interval,
        hyperparam_mutations=ConfigArchiSearch.hyperparam_mutations,
    )
    ray_analysis = run(
        ArchitectureSearcher,
        name="alphazero_pbt",
        scheduler=archi_searcher,
        **search_settings(),
        raise_on_failed_trial=False,
    )
    handle_search_end(working_dir_before_search=search_working_dir)
    search_report = report_from_ray_analysis(
        ray_analysis, "global_loss", ConfigArchiSearch.search_mode
    )
    search_report_filename = report_filename_with_iterations(
        ConfigArchiSearch.report_filename,
        ConfigArchiSearch.data_iteration_start,
        ConfigArchiSearch.data_iteration_end,
    )
    save_architecture_search(
        search_report=search_report,
        directory_path=paths.get_architecture_searches_path(ConfigArchiSearch.run_id),
        report_filename=search_report_filename,
    )
    print(
        f"Architecture search done, search report has been saved at: {search_report_filename}"
    )
