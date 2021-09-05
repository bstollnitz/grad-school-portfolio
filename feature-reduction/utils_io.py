import os
import shutil
from typing import List, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

import utils_video


DATA_FOLDER = 'data'
PLOTS_FOLDER = 'plots'
VIDEOS_FOLDER = 'videos'


def _find_or_create_dir(dir_name: str) -> str:
    """Creates a directory if it doesn't exist.

    Args:
        dir_name (str): The name of the directory to create if it doesn't
        exist.

    Returns:
        The local path of the directory.
    """
    # Get the directory of the current file.
    parent_dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Create a directory if it doesn't exist.
    dir_path = os.path.join(parent_dir_path, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def create_plots() -> str:
    """Creates a directory to hold plots if it doesn't exist.

    Returns:
        The local path of the directory.
    """
    return _find_or_create_dir(PLOTS_FOLDER)


def download_remote_data_file(data_url: str) -> str:
    """Downloads data from url if it's not saved locally yet.

    Args:
        data_url (str): The url of the data file we want to download.
    
    Returns:
        The path to the local file.
    """
    # Create a data directory if it doesn't exist.
    data_dir_path = _find_or_create_dir(DATA_FOLDER)
    
    # Download the data file if it doesn't exist.
    filename = os.path.basename(urlparse(data_url).path)
    data_file_path = os.path.join(data_dir_path, filename)
    if not os.path.exists(data_file_path):
        print(f'Downloading data file {data_file_path}...')
        with urlopen(data_url) as response:
            with open(data_file_path, "wb") as data_file:
                shutil.copyfileobj(response, data_file)
        print('Done downloading data file.')

    return data_file_path


def save_video(video: np.ndarray, video_name: str) -> str:
    """Creates a folder to hold videos if it doesn't exist yet, and saves
    the video.

    Args: 
        video (np.ndarray): The video to save.
    
    Returns:
        The path to the video.
    """
    # Create a video directory if it doesn't exist.
    video_dir_path = _find_or_create_dir(VIDEOS_FOLDER)
    video_path = os.path.join(video_dir_path, video_name)

    # Save the video to that directory if it's not there already.
    if not os.path.exists(video_path):
        utils_video.save_video(video_path, video)

    return video_path
