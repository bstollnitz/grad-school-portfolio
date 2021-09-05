import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def download_remote_data_file(local_dir_name: str, 
    remote_dir_url: str, file_name: str) -> Tuple[str, bool]:
    """Downloads data from url if it's not saved locally yet.

    Args:
        local_dir_name (str): The local folder where we'll save the 
        downloaded data.
        remote_dir_url (str): The url prefix of the data file we want to download.
        file_name (str): The name of the data file we want to download.

    Returns:
        The path to the local file, and a bool indicating whether the file
        was downloaded or not.
    """
    # Create a data directory if it doesn't exist.
    local_dir_path = Path('.', local_dir_name)
    local_dir_path.mkdir(exist_ok=True)

    # Download the data file if it doesn't exist.
    remote_file_url = remote_dir_url + file_name
    local_file_path = Path(local_dir_path, file_name)
    downloaded = False
    if not local_file_path.exists():
        print(f'Downloading data file {remote_file_url}...')
        with urlopen(remote_file_url) as response:
            with open(local_file_path, "wb") as data_file:
                shutil.copyfileobj(response, data_file)
        downloaded = True
        print('Done downloading data file.')

    return (local_file_path, downloaded)


def extract_all_zips(dir_name: str) -> bool:
    """Extracts all zip files in a directory into the same directory and
    deleted the file.

    Args:
        dir_name: The name of the local directory where the zips are located.

    Returns:
        True or false, depending on whether any zips were found and extracted.
    """
    # Get the directory of the current file.
    dir_path = Path('.', dir_name)

    # Extract all zip files in directory.
    extracted = False
    if dir_path.exists() and dir_path.is_dir():
        for file_path in dir_path.iterdir():
            if file_path.suffix == '.zip':
                with zipfile.ZipFile(file_path) as zip_obj:
                    zip_obj.extractall(dir_path)
                    extracted = True
    
    return extracted


def save_image(matrix: np.ndarray, folder: str, filename: str) -> None:
    """Saves an image, given a properly shaped array of double values.

    Args:
        matrix (np.ndarray): The matrix containing the information to save
        as an image. 
        Assumes matrix shape is height by width of the image. 
        Assumes matrix values lie between 0 and 1. 
        folder (str): The folder where we want to save the image.
        filename (str): The name of the image file.
    """
    # We apply a colormap.
    cm = plt.get_cmap('gist_heat')
    color_matrix = cm(matrix)
    # We convert the values in the image to go from 0 to 255 and be ints.
    int_matrix = (color_matrix[:, :, :3] * 255).astype('uint8')
    # Save matrix as image.
    image = Image.fromarray(int_matrix)
    save_path = Path(folder, filename)
    image.save(str(save_path))


def print_elapsed_time(start_time: float, end_time: float) -> None:
    """Prints time elapsed between start_time and end_time, formatted as
    HH:MM:SS.

    Args:
        start_time (float): Start time.
        end_time (float): End time.
    """

    elapsed_seconds = int(end_time - start_time)
    (elapsed_minutes, elapsed_seconds) = divmod(elapsed_seconds, 60)
    (elapsed_hours, elapsed_minutes) = divmod(elapsed_minutes, 60)
    print(f'  Elapsed time: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}')
