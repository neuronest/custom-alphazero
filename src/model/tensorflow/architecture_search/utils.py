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
    search_report: dict, directory_path: str, report_filename: str = "analysis",
):
    os.makedirs(directory_path, exist_ok=True)
    with open(os.path.join(directory_path, f"{report_filename}.pkl"), "wb") as f:
        pickle.dump(search_report, f)


def load_architecture_search(
    directory_path=".", report_filename: str = ConfigArchiSearch.report_filename,
):
    with open(os.path.join(directory_path, f"{report_filename}.pkl"), "rb") as f:
        analysis = pickle.load(f)
    return analysis


def means_from_discrete(
    values: pd.Series, losses: pd.Series
) -> List[Tuple[int, float]]:
    values = values.astype(str)
    return [(losses[values == value].mean(), value) for value in values.unique()]


def means_from_continuous(
    values: pd.Series, losses: pd.Series, nb_bins: int = 4
) -> List[Tuple[int, float]]:
    return means_from_discrete(pd.cut(values, nb_bins), losses)
