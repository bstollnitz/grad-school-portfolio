"""Helper functions for videos."""

import cv2
import numpy as np

def show_video(title: str, video: np.ndarray) -> None:
    """Shows the given video.
    
    Args:
        video (np.ndarray): The video, of shape (frames, height, width).
    """
    frame_count = video.shape[0]
    for frame_index in range(frame_count):
        frame = video[frame_index]
        cv2.imshow(f"{title} - press 'q' to exit", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
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

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        file_path, fourcc, fps=30.0, frameSize=(size_x, size_y)
    )

    for frame_index in range(frame_count):
        frame = video[frame_index]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()
