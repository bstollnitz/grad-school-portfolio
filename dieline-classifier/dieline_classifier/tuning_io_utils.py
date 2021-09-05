import csv
import datetime
import pathlib
import pickle
import pprint
import time
from typing import List, Tuple

import numpy as np
import termcolor

def get_last_round_number(prefix: str) -> int:
    """
    Returns the number of the last saved hyperparameter tuning round, or zero
    if none was saved.
    """

    prefix_len = len(prefix)
    p = pathlib.Path(".")
    round_number_list = [path_item.name[prefix_len:] for path_item in p.iterdir() if 
        path_item.is_dir() and path_item.name.startswith(prefix)]
    return 0 if not round_number_list else int(round_number_list[-1])

def create_next_round_dir(prefix: str) -> str:
    """
    Creates the path we will use to save the files for the next round 
    of hyperparameter tuning. Returns the string that represents that path.
    """
    
    round_number = get_last_round_number(prefix) + 1
    save_path_str = f"{prefix}{round_number}"

    save_path = pathlib.Path(save_path_str)
    if not save_path.exists():
        save_path.mkdir()

    return save_path_str

def get_last_round_dir(prefix: str) -> str:
    """
    Returns the string that represents the directory corresponding to the
    last round of hyperparameter tuning.
    """

    round_number = get_last_round_number(prefix)

    if round_number == 0:
        raise Exception("No hyperparameter tuning results available to load.")

    return f"{prefix}{round_number}"

def save_results(param_distributions: dict, scores: List[float],
    params: List[dict], best_index: int, best_history: dict,
    filename: str) -> None:
    """
    Saves results to file.
    """

    with open(filename, "wb") as f:
        pickle.dump(param_distributions, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(best_index, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(best_history, f, pickle.HIGHEST_PROTOCOL)

def load_results(filename: str) -> Tuple[dict, List[float], List[dict], int, dict]:
    """
    Loads the results from file.
    """

    with open(filename, "rb") as f:
        param_distributions = pickle.load(f)
        scores = pickle.load(f)
        params = pickle.load(f)
        best_index = pickle.load(f)
        best_history = pickle.load(f)
    return (param_distributions, scores, params, best_index, best_history)

def save_top(filename: str, scores: List[float], params: List[dict], 
    top_count: int, best_test_accuracy: float) -> None:
    """
    Saves the top parameter values to a CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headings = ["val_accuracy"] + list(params[0].keys()) + ["test_accuracy"]
        writer.writerow(headings)
        top_count = min(top_count, len(scores))
        indices = np.flip(np.argsort(scores))[:top_count]
        for i in range(top_count):
            index = indices[i]
            values = [scores[index]] + list(params[index].values())
            if i == 0:
                values += [best_test_accuracy]
            writer.writerow(values)

def print_configuration(configuration: dict, index: int, count: int,
    start_time: float) -> None:
    """
    Prints the details of a single hyperparameter configuration.
    """

    elapsed_time = round(time.monotonic() - start_time)
    elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
    color = "blue"
    termcolor.cprint(f"\nTrying configuration {index + 1}/{count} " +
        f"({(index / count):.2%}) after {elapsed_time_str}:", color)
    termcolor.cprint(pprint.pformat(configuration), color)
