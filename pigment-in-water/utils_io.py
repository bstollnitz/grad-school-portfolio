import shutil
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlopen

import numpy as np

import cv2


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
            with open(local_file_path, 'wb') as data_file:
                shutil.copyfileobj(response, data_file)
        downloaded = True
        print('Done downloading data file.')

    return (local_file_path, downloaded)
    
    
def show_video(title: str, video: np.ndarray) -> None:
    """Shows the given video.
    
    Args:
        video (np.ndarray): The video, of shape (frames, height, width).
    """
    frame_count = video.shape[0]
    for frame_index in range(frame_count):
        frame = video[frame_index]
        cv2.imshow(f'{title} - press "q" to exit', frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def save_video(file_path: str, video: np.ndarray) -> None:
    """Saves the given video as an AVI file.
    
    Args:
        file_path (str): The path to the file.
        video (np.ndarray): The video to save, of shape (frames, width, height).
    """
    frame_count = video.shape[0]
    size_y = video.shape[1]
    size_x = video.shape[2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        file_path, fourcc, fps=30.0, frameSize=(size_x, size_y)
    )

    for frame_index in range(frame_count):
        frame = video[frame_index]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()
