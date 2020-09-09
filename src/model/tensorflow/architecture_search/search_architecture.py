import os

from sklearn.model_selection import train_test_split
import ray
from ray.tune import run, Trainable
from ray.tune.schedulers import PopulationBasedTraining
import tensorflow as tf

from src.utils import init_model, save_architecture_search, load_queue
from src.config import ConfigGeneral, ConfigPath, ConfigArchiSearch
from src.model.tensorflow.model import PolicyValueModel
from src.model.tensorflow.base_layers import policy_loss, value_loss

load_model_from_path = PolicyValueModel.load_model_from_path


def data_to_search_from(run_id, iteration, subset_size=int(1e6), p_val=0.25):
    iteration_path = os.path.join(
        ConfigPath.results_path, ConfigGeneral.game, run_id, f"iteration_{iteration}",
    )
    states, policies, rewards = load_queue(iteration_path)
    states, policies, rewards = (
        states[:subset_size],
        policies[:subset_size],
        rewards[:subset_size],
    )

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
    run_id=ConfigArchiSearch.run_id, iteration=ConfigArchiSearch.iteration
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


def report_from_ray_analysis(ray_analysis, searched_metric, search_mode):
    return {
        "best_config": ray_analysis.get_best_config(
            metric=searched_metric, mode=search_mode
        ),
        "trial_dataframes": ray_analysis.trial_dataframes,
        "main_dataframe": ray_analysis.dataframe(),
    }


def initialize_search(running_mode: str) -> None:
    if running_mode == "debug":
        ray.init(local_mode=True)
    else:
        ray.init()


def search_settings() -> dict:
    search_settings = {
        # if resources not available some trials may never end
        # "resources_per_trial": {"cpu": 1, "gpu": 1},
        "resources_per_trial": ConfigArchiSearch.resources_per_trial,
        "stop": ConfigArchiSearch.search_stop_criterion,
        "config": ConfigArchiSearch.hyperparam_config,
        "num_samples": ConfigArchiSearch.num_samples_from_config,
    }
    return search_settings


def handle_search_end(working_dir_before_search: str) -> None:
    # ray has changed the working directory
    if os.getcwd() != working_dir_before_search:
        # reset working directory to the one we want and that has been changed by ray
        os.chdir(working_dir_before_search)


if __name__ == "__main__":
    # check working dir here
    search_working_dir = os.getcwd()
    initialize_search(ConfigArchiSearch.running_mode)
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
    save_architecture_search(
        search_report=search_report,
        directory_path=os.path.join(
            ConfigPath.results_path,
            ConfigGeneral.game,
            ConfigArchiSearch.run_id,
            ConfigArchiSearch.archi_searches_dir,
        ),
        report_filename=f"{ConfigArchiSearch.report_filename}_iteration_{ConfigArchiSearch.iteration}",
    )
