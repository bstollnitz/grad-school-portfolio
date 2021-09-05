import os
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.io import loadmat

import utils_graph
import utils_io
import utils_video


S3_URL = 'https://bea-portfolio.s3-us-west-2.amazonaws.com/feature-reduction/'


def process_data(scenario_index: int) -> List[str]:
    """Creates a data folder if it doesn't exist and loads data for this 
    project if it's not yet present in the data folder. Returns the 
    paths of the local video files.
    """
    # Download mat files from Amazon S3 if they're not yet local.
    index_list = range(1, 4)
    filenames = [f'cam{j}_{scenario_index}.mat' for j in index_list]
    local_paths = []
    for filename in filenames:
        remote_url = S3_URL + filename
        local_path = utils_io.download_remote_data_file(remote_url)
        local_paths.append(local_path)

    # Load each data file into memory.
    print(f'Loading data for scenario {scenario_index}...')
    data_files = [loadmat(local_path, struct_as_record=False) for local_path 
        in local_paths]
    
    # Extract information for each data files into an array.
    # Each arrary is of size (480, 640, 3, frames) = (height, width, 3, frames).
    data_list = [data_files[j-1][f'vidFrames{j}_{scenario_index}'] for j in index_list]

    # Convert to grayscale. Shape now is 
    # (height, width, frames) = (480, 640, frames)
    data_list = [utils_video.convert_to_grayscale(data) for data in data_list]

    # Save grayscale videos to disk as avi files.
    print(f'Saving videos for scenario {scenario_index}...')
    local_paths = []
    for (i, data) in enumerate(data_list):
        local_filename = f'scenario{scenario_index}_cam{i+1}.avi'
        local_path = utils_io.save_video(data, local_filename)
        local_paths.append(local_path)

    return local_paths


def normalize_rows(my_array: np.ndarray) -> np.ndarray:
    """Normalizes the rows of an ndarray to lie between 0 and 1.
    """

    my_array = my_array - np.min(my_array, axis=1, keepdims=True)
    return my_array/np.max(my_array, axis=1, keepdims=True)


def analyze_scenario(dirname: str, scenario_index: int,
    bboxes: List[Tuple[int]], skip_frames: List[int]) -> None:
    """Analyzes the data for a particular scenario.
    """
    local_paths = process_data(scenario_index)

    # Track the center of the bucket in each video.
    print(f'Tracking object for scenario {scenario_index}...')
    centers = []
    index_list = range(3)
    centers = [utils_video.track_object(local_paths[i], bboxes[i]) for i in index_list]

    # Trim beginning of centers so that they all start with bucket at the top.
    centers = list([centers[i][:, skip_frames[i]:] for i in index_list])

    # Trim end of centers to the shortest length.
    frame_count = min([center.shape[1] for center in centers])
    centers = list([center[:, 0:frame_count] for center in centers])

    # Combine centers into a single data matrix.
    data = np.vstack(centers)

    # Graph the vertical component of the centers (just to see if they're
    # aligned).
    print(f'Creating graphs for scenario {scenario_index}...')
    t = np.reshape(np.asarray(range(frame_count)), (1, frame_count))
    row_indices = [1, 3, 4]
    utils_graph.graph_overlapping_lines(
        np.repeat(t, 3, axis=0),
        normalize_rows(data[row_indices, :]),
        ['Camera 1', 'Camera 2', 'Camera 3'],
        'Frame', 'Vertical position', 
        f'Normalized vertical positions in scenario {scenario_index}',
        dirname, f'{scenario_index}_vertical_positions.html')

    # Subtract the mean.
    data = data - np.mean(data, axis=1, keepdims=True)

    # Graph singular values to determine how many modes to keep.
    (u, s, vh) = np.linalg.svd(data / np.sqrt(frame_count-1), full_matrices=False)
    normalized_s = s / np.sum(s)
    utils_graph.graph_2d_markers(
        np.asarray(range(1, len(normalized_s)+1)),
        normalized_s, 'Mode', 'Normalized singular value',
        f'Singular values for scenario {scenario_index}',
        dirname, f'{scenario_index}_singular_values.html')
    
    # We could reduce the number of modes here, but let's look at all of them.
    mode_count = 6
    u_reduced = u[:, 0:mode_count]

    # Project the data onto the reduced U.
    data_reduced = u_reduced.T.dot(data)

    # Plot the two spatial modes of the reduced data over time.
    legend = [f'Mode {i + 1}' for i in range(mode_count)]
    utils_graph.graph_overlapping_lines(
        np.repeat(t, mode_count, axis=0),
        data_reduced,
        legend,
        'Frame', 'Value of mode', 
        f'Modes of scenario {scenario_index}',
        dirname, f'{scenario_index}_dominant_modes.html')


def ideal_scenario(dirname: str) -> None:
    """Analyzes stable video taken from 3 cameras.
    """
    scenario_index = 1
    bboxes = [(313, 210, 70, 100), (260, 260, 70, 100), (310, 255, 84, 63)]
    skip_frames=[10, 19, 10]
    analyze_scenario(dirname, scenario_index, bboxes, skip_frames)


def noisy_scenario(dirname: str) -> None:
    """Analyzes video from 3 cameras with camera shake.
    """
    scenario_index = 2
    bboxes = [(311, 287, 60, 93), (285, 342, 81, 78), (344, 238, 73, 59)]
    skip_frames=[35, 22, 39]
    analyze_scenario(dirname, scenario_index, bboxes, skip_frames)


def horizontal_displacement_scenario(dirname: str) -> None:
    """Analyzes video taken from 3 cameras. In this case, the mass is released
    off-center so as to produce motion in the x-y plane as well as the z 
    direction.
    """
    scenario_index = 3
    bboxes = [(317, 277, 56, 87), (220, 287, 73, 100), (345, 208, 83, 71)]
    skip_frames=[39, 66, 32]
    analyze_scenario(dirname, scenario_index, bboxes, skip_frames)


def horizontal_displacement_rotation_scenario(dirname: str) -> None:
    """Analyzes video taken from 3 cameras. The mass is released off-center
    and rotates.
    """
    scenario_index = 4
    bboxes = [(366, 255, 61, 87), (216, 238, 79, 104), (354, 173, 84, 53)]
    skip_frames=[33, 39, 33]
    analyze_scenario(dirname, scenario_index, bboxes, skip_frames)


def main() -> None:
    """Main program.
    """
    plots_dir_path = utils_io.create_plots()
    ideal_scenario(plots_dir_path)
    noisy_scenario(plots_dir_path)
    horizontal_displacement_scenario(plots_dir_path)
    horizontal_displacement_rotation_scenario(plots_dir_path)


if __name__ == '__main__':
    main()
