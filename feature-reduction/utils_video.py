from typing import List, Tuple

import cv2
import numpy as np


def convert_to_grayscale(video: np.ndarray) -> np.ndarray:
    """Converts video to grayscale using y_linear method. 

    Args:
        video (np.ndarray): The video, of shape (height, width, 3, frames).

    Returns:
        video (np.ndarray): The video in grayscale, of shape 
        (height, width, frames).
    """
    grayscale_video = 0.2126*video[:, :, 0, :] + 0.7152*video[:, :, 1, :] + 0.0722*video[:, :, 2, :]
    return grayscale_video


def save_video(file_path: str, video: np.ndarray) -> None:
    """Saves the given video as an AVI file.
    
    Args:
        file_path (str): The path to the file.
        video (np.ndarray): The video to save, of shape (height, width, frames).
    """
    frame_count = video.shape[2]
    size_x = video.shape[1]
    size_y = video.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        file_path, fourcc, fps=30.0, frameSize=(size_x, size_y)
    )

    for frame_index in range(frame_count):
        frame = video[:, :, frame_index]
        frame = frame.astype('uint8')
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()


def track_object(file_path: str, bbox: Tuple[float]) -> np.ndarray:
    """
    Tracks an object in a video defined by the bounding box passed as a 
    parameter. Returns the centers of the bounding box over time.

    Args:
        file_path (str): The path to the file.
        bbox (Tuple[float]): Tuple containing (x, y, width, height).
    """
    # Open video.
    video = cv2.VideoCapture(file_path)

    # Exit if video not opened.
    if not video.isOpened():
        raise Exception(f'Could not open video: {file_path}')
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        raise Exception(f'Cannot read video file: {file_path}')

    # Define tracker.
    tracker = cv2.TrackerCSRT_create()
    ok = tracker.init(frame, bbox)
    if not ok:
        raise Exception(f'Cannot initialize tracker for video file {file_path}.')

    # Get center of first bounding box.
    center = (bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2)
    centers = [center]

    # Read video until the end and gather centers for all bounding boxes.
    while True:
        ok, frame = video.read()
        if not ok:
            break # Reached the end of the video.
        ok, bbox = tracker.update(frame)
        if not ok:
            raise Exception(f'Could not track object in video file {file_path}')
        center = (bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2)
        centers.append(center)
    
    return np.asarray(centers).T
