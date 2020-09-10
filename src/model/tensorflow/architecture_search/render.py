import os
from src.model.tensorflow.architecture_search.utils import (
    means_from_discrete,
    means_from_continuous,
)

import pandas as pd

from src import config
from src.config import ConfigArchiSearch


def report_from_ray_analysis(ray_analysis, searched_metric, search_mode):
    return {
        ConfigArchiSearch.report_config: ray_analysis.get_best_config(
            metric=searched_metric, mode=search_mode
        ),
        ConfigArchiSearch.reports_trials: ray_analysis.trial_dataframes,
        ConfigArchiSearch.report_on_trials: ray_analysis.dataframe(),
        ConfigArchiSearch.report_hyperparameters_contribution: mean_metric_per_hyperparam_as_df(
            ray_analysis.dataframe()
        ),
    }


def mean_metric_per_hyperparam_as_df(
    df: pd.DataFrame,
    search_mode: str = ConfigArchiSearch.search_mode,
    hyperparameter_col: str = "hyperparameter",
    value_col: str = "value",
    mean_col: str = "mean",
    distinct_col: str = "hyperparam_nb_distinct",
) -> pd.DataFrame:
    df_mean_losses = pd.DataFrame(
        columns=[hyperparameter_col, value_col, mean_col, distinct_col]
    )
    for hyperparam in [c for c in df if "config/" in c]:
        if (
            hyperparam.replace("config/", "")
            in ConfigArchiSearch.discrete_hyperparameters
        ):
            means_by_values = means_from_discrete(df[hyperparam], df["global_loss"])
        elif (
            hyperparam.replace("config/", "")
            in ConfigArchiSearch.continuous_hyperparameters
        ):
            means_by_values = means_from_continuous(df[hyperparam], df["global_loss"])
        else:
            raise Exception(
                f"hyperparameter {hyperparam} is not recognized as neither discrete nor continuous from the "
                f"config file {config.__file__}"
                f"{os.linesep}Specified discrete hyperparameters are {' '.join(ConfigArchiSearch.discrete_hyperparameters)}"
                f"{os.linesep}Specified continuous hyperparameters are {' '.join(ConfigArchiSearch.continuous_hyperparameters)}"
            )
        for mean, value in means_by_values:
            df_mean_losses.loc[len(df_mean_losses)] = (
                hyperparam,
                value,
                mean,
                # nb of distinct values for that hyperparameter
                len(means_by_values),
            )

    def _sorting_mode(search_mode: str) -> str:
        # if min value was searched, order by ascending
        # to see the best value first
        if search_mode == "min":
            return "ascending"
        elif search_mode == "max":
            return "descending"
        else:
            raise NotImplementedError

    return df_mean_losses.sort_values(
        mean_col, ascending=_sorting_mode(search_mode) == "ascending"
    )
