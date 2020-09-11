from typing import List, Tuple, Optional
import os

import pandas as pd
import pickle

from src.config import ConfigArchiSearch
from src import paths


def report_filename_with_iterations(
    report_filename, iteration_start, iteration_end,
):
    return f"{report_filename}_iteration_{iteration_start}_to_{iteration_end}"


def save_architecture_search(
    search_report: dict,
    directory_path: str,
    report_filename: str = ConfigArchiSearch.report_filename,
):
    os.makedirs(directory_path, exist_ok=True)
    with open(os.path.join(directory_path, f"{report_filename}.pkl"), "wb") as f:
        pickle.dump(search_report, f)


def load_architecture_search(directory_path: str, report_filename: str):
    with open(os.path.join(directory_path, f"{report_filename}.pkl"), "rb") as f:
        analysis = pickle.load(f)
    return analysis


def best_config(
    run_id: str,
    directory_path: Optional[str] = None,
    report_filename: Optional[str] = None,
):
    if directory_path is None:
        directory_path = paths.get_architecture_searches_path(run_id)
    if report_filename is None:
        report_filename = report_filename_with_iterations(
            ConfigArchiSearch.report_filename,
            ConfigArchiSearch.data_iteration_start,
            ConfigArchiSearch.data_iteration_end,
        )
    return load_architecture_search(directory_path, report_filename)[
        ConfigArchiSearch.report_config
    ]


def means_from_discrete(
    values: pd.Series, losses: pd.Series
) -> List[Tuple[int, float]]:
    values = values.astype(str)
    return [(losses[values == value].mean(), value) for value in values.unique()]


def means_from_continuous(
    values: pd.Series, losses: pd.Series, nb_bins: int = 4
) -> List[Tuple[int, float]]:
    return means_from_discrete(pd.cut(values, nb_bins), losses)
