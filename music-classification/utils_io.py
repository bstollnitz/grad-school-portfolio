import os
import shutil
from typing import List, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from PIL import Image


def find_or_create_dir(dir_name: str) -> str:
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


def download_remote_data_file(local_data_folder: str, 
    url_prefix: str, filename: str) -> Tuple[str, bool]:
    """Downloads data from url if it's not saved locally yet.

    Args:
        local_data_folder (str): The local folder where we'll save the 
        downloaded data.
        url_prefix (str): The url prefix of the data file we want to download.
        filename (str): The name of the data file we want to download.

    Returns:
        The path to the local file, and a bool indicating whether the file
        was downloaded or not.
    """
    # Create a data directory if it doesn't exist.
    data_dir_path = find_or_create_dir(local_data_folder)
    
    # Download the data file if it doesn't exist.
    data_url = url_prefix + filename
    data_file_path = os.path.join(data_dir_path, filename)
    downloaded = False
    if not os.path.exists(data_file_path):
        print(f'Downloading data file {data_url}...')
        with urlopen(data_url) as response:
            with open(data_file_path, "wb") as data_file:
                shutil.copyfileobj(response, data_file)
        downloaded = True
        print('Done downloading data file.')

    return (data_file_path, downloaded)


def load_wav_file(local_path: str) -> Tuple[int, np.ndarray]:
    """Loads a wav file.

    Args:
        local_path (str): The local path to the wav file.

    Returns:
        The sample rate and an ndarrary containing the information in the 
        wav file.
    """
    # sample_rate is measurements per second.
    (sample_rate, wav_data) = wavfile.read(local_path)

    # 65536 = 2^16. 
    # 32768 = 2^15.
    # Samples in a wav file have 16 bits, with 1 bit for the sign.
    # We scale the values to be between -1 and 1.
    wav_data = wav_data/32768

    # Throw away one of the stereo channels.
    wav_data = wav_data[:,0]

    return (sample_rate, wav_data)


def save_image(matrix: np.ndarray, folder: str, filename: str) -> None:
    """Saves an image, given a properly shaped array of double values.

    Args:
        matrix (np.ndarray): The matrix containing the information to save
        as an image. Must have the shape we want the image to have.
        
        folder (str): The folder where we want to save the image.

        filename (str): The name of the image file.
    """
    # We convert image to Viridis colorscale.
    cm = plt.get_cmap('viridis')
    color_matrix = cm(matrix)
    # We convert the values in the image to go from 0 to 255 and be ints.
    int_matrix = (color_matrix * 255).astype('uint8')
    # Save matrix as image.
    image = Image.fromarray(int_matrix)
    save_path = os.path.join(folder, filename)
    image.save(save_path)
