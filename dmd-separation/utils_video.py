import os
from typing import List, Tuple

import cv2
import numpy as np

DOWNSAMPLE_FRACTION = 0.5

def load_video(folder: str, filename: str) -> Tuple[np.ndarray, float]:
    """Loads a single video.

    Args:
        folder (str): Local folder where video is located.
        filename (str): Name of video file.

    Returns:
        video (np.ndarray): Video with shape (width, height, frames).
        frame_rate (float): Frame rate of video.
    """
    # Initialize the video reader and get the video dimensions.
    print(f'Loading {filename}...')
    file_path = os.path.join(folder, filename)
    capture = cv2.VideoCapture(file_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = capture.get(cv2.CAP_PROP_FPS)

    # Downsample video.
    frame_width = int(frame_width * DOWNSAMPLE_FRACTION)
    frame_height = int(frame_height * DOWNSAMPLE_FRACTION)

    # Read each frame of the video capture. 
    video = np.empty((frame_width, frame_height, frame_count), 
        np.dtype('uint8'))
    frame_index = 0
    success = True
    while frame_index < frame_count and success:
        (success, frame) = capture.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if DOWNSAMPLE_FRACTION != 1:
                frame = cv2.resize(frame, None, fx=DOWNSAMPLE_FRACTION, 
                    fy=DOWNSAMPLE_FRACTION, interpolation=cv2.INTER_CUBIC)
            video[:, :, frame_index] = frame.T
        frame_index += 1
    capture.release()
    
    # Convert to floats from 0 to 1.
    video = video / 255.0

    return (video, frame_rate)


def save_video(folder: str, filename: str, video: np.ndarray) -> None:
    """Saves the given video as an AVI file.
    
    Args:
        folder (str): The folder where we'll save the video file.
        filename (str): The name of the video file.
        video (np.ndarray): The video to save, of shape (width, height, frames).
    """
    print(f'Saving video {filename}...')
    file_path = os.path.join(folder, filename)

    size_x = video.shape[0]
    size_y = video.shape[1]
    frame_count = video.shape[2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        file_path, fourcc, fps=30.0, frameSize=(size_x, size_y)
    )

    for frame_index in range(frame_count):
        frame = video[:, :, frame_index].T

        # Scale each frame to lie between 0 and 1 (temporarily).
        # frame -= np.min(frame)
        # frame /= np.max(frame)

        # Scale to lie between 0 and 255, then clamp and convert to bytes.
        frame = np.clip(frame * 255, 0, 255)
        frame = frame.astype('uint8')
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()
