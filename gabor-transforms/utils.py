import os
import shutil
from typing import List, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

DATA_FOLDER = "data"


def find_or_create_dir(dir_name: str) -> str:
    """
    Creates a directory if it doesn't exist.
    """

    # Get the directory of the current file.
    parent_dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Create a directory if it doesn't exist.
    dir_path = os.path.join(parent_dir_path, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def download_remote_data_file(data_url: str) -> str:
    """
    Gets the data from url if it's not saved locally yet.
    Returns the path to the local file.
    """
    
    # Create a data directory if it doesn't exist.
    data_dir_path = find_or_create_dir(DATA_FOLDER)
    
    # Download the data file if it doesn't exist.
    filename = os.path.basename(urlparse(data_url).path)
    data_file_path = os.path.join(data_dir_path, filename)
    print(f"data_file_path: {data_file_path}")
    if not os.path.exists(data_file_path):
        print("Downloading data file...")
        with urlopen(data_url) as response:
            with open(data_file_path, "wb") as data_file:
                shutil.copyfileobj(response, data_file)
        print("Done downloading data file.")

    return data_file_path


def normalize(my_array: np.ndarray) -> np.ndarray:
    """
    Normalizes an ndarray.
    """

    return np.abs(my_array)/np.max(np.abs(my_array))

