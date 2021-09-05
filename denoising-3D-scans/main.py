from typing import List, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.io import loadmat
import os
import shutil

from urllib.request import urlopen

COLOR1 = 'steelblue'


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


def load_data() -> np.ndarray:
    """
    Loads the data for the project.
    """
    
    # Create a data directory if it doesn't exist.
    data_dir_path = find_or_create_dir("data")
    
    # Download the data file if it doesn't exist.
    data_file_path = os.path.join(data_dir_path, "Testdata.mat")
    if not os.path.exists(data_file_path):
        print("Downloading data file...")
        data_url = "https://bea-portfolio.s3-us-west-2.amazonaws.com/denoising-3D-scans/Testdata.mat"
        with urlopen(data_url) as response:
            with open(data_file_path, "wb") as data_file:
                shutil.copyfileobj(response, data_file)
        print("Done downloading data file.")

    # Load data into memory.
    data_file = loadmat(data_file_path, struct_as_record=False)
    data = data_file['Undata']
    # data.shape is 20 x 262144

    return data


def initialize(domain_limit: int, n: int) -> Tuple[List[int], List[int]]:
    """
    Initializes variables typically needed for problem setup.
    n = number of points in spatial or time domain = number of points in 
    frequency domain.
    """

    # Time or spatial domain discretization.
    # Since we have periodic boundary conditions, the first and last points 
    # are the same. So we consider only the first n points in the time domain. 
    t_shifted = np.linspace(-domain_limit, domain_limit, n+1)[0:-1]

    # Frequency domain discretization.
    omega_points = np.linspace(-n/2, n/2, n+1)[0:-1]
    omega_shifted = (2 * np.pi)/(2 * domain_limit) * omega_points
    half_n = n//2
    omega_unshifted = np.concatenate((omega_shifted[half_n:n], 
        omega_shifted[0:half_n]))
    assert np.max(omega_shifted) == np.max(omega_unshifted)
    assert np.min(omega_shifted) == np.min(omega_unshifted)
    assert omega_shifted.size == omega_unshifted.size

    return (t_shifted, omega_shifted)


def normalize(my_array: np.ndarray) -> np.ndarray:
    """
    Takes the absolute value of an ndarray and normalizes it.
    """

    return np.abs(my_array)/np.max(np.abs(my_array))


def plot_spatial_isosurfaces(description: str, domain_values: np.ndarray, 
    u: np.ndarray, filename: str) -> None:
    """
    Plots a few slices of the the data using isosurfaces.
    """

    print(f'Plotting spatial isosurfaces: {description}...')

    (x_grid, y_grid, z_grid) = np.meshgrid(domain_values, domain_values, 
        domain_values, indexing='ij')
    n = len(domain_values)

    num_slices = u.shape[0]
    # We only want to plot the first, middle, and last time slices.
    slices = [0, num_slices//2, num_slices-1]

    titles = [f'{description}: slice {slice}' for slice in slices]

    num_rows = 1
    num_cols = len(slices)
    fig = make_subplots(
        rows=num_rows, 
        cols=num_cols,
        specs=[
            [{'is_3d': True}]*num_cols,
        ]*num_rows,
        subplot_titles=titles,
    )
    for s in range(len(slices)):
        u_slice = np.reshape(u[slices[s],:], (n, n, n))
        fig.add_trace(
            go.Isosurface(
                x=x_grid.flatten(), 
                y=y_grid.flatten(), 
                z=z_grid.flatten(), 
                value=normalize(u_slice).flatten(),
                isomin=0.4,
                isomax=0.4,
                surface_count=1,
                colorscale="Viridis",
            ),
            row=1,
            col=s+1
        )
    pio.write_html(fig, filename)


def get_fft(u: np.ndarray, n: int) -> np.ndarray:
    """
    Gets the fft of the data.
    """

    # We get the fft of each time slice.
    num_slices = u.shape[0]
    ut = np.empty(u.shape, dtype=complex) # shape (20, 262144)
    for s in range(num_slices):
        # We reshape each slice into a 3D cube.
        u_slice = np.reshape(u[s,:], (n, n, n)) # shape (64, 64, 64)
        # We then take the fft of the 3D cube and add it to ut.
        ut_slice = np.fft.fftshift(np.fft.fftn(u_slice)) # shape (64, 64, 64)
        ut[s, :] = ut_slice.flatten()

    return ut

def average_fft(ut: np.ndarray) -> np.ndarray:
    """
    Gets the average fft of the data.
    """

    # We average over each row of ut.
    ut_average = np.average(ut, axis=0) # shape (262144,)

    return ut_average


def plot_fft_isosurface(title: str, omega: np.ndarray, 
    ut: np.ndarray, filename: str) -> None:
    """
    Plots an isosurface 3D graph in frequency domain.
    """

    print(f'Plotting fft isosurface: {title}...')

    (omega_x_grid, omega_y_grid, omega_z_grid) = np.meshgrid(omega, omega, 
        omega, indexing='ij')

    fig = go.Figure()
    fig.add_trace(
        go.Isosurface(
            x=omega_x_grid.flatten(), 
            y=omega_y_grid.flatten(), 
            z=omega_z_grid.flatten(), 
            value=normalize(ut).flatten(),
            opacity=0.5,
            isomin=0.6,
            isomax=0.9,
            surface_count=3,
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        title_text=title,
        scene_xaxis_title_text='omega_x',
        scene_yaxis_title_text='omega_y',
        scene_zaxis_title_text='omega_z',
    )
    pio.write_html(fig, filename)


def get_peak_frequency(ut_average: np.ndarray, 
    omega: np.ndarray) -> Tuple[float, float, float]:
    """
    Gets the peak frequency of the average fft.
    """

    # We get the indices of the peak of the average fft.
    n = len(omega)
    argmax = np.argmax(np.abs(ut_average))
    [index_x, index_y, index_z] = np.unravel_index(argmax, (n, n, n))

    # We then use those indices to get the peak frequency.
    return (omega[index_x], omega[index_y], omega[index_z])


def print_peak_frequency(omega_x: float, omega_y: float, 
    omega_z: float) -> None:
    """
    Prints the peak frequency.
    """

    print(f'The peak frequency is: ({omega_x}, {omega_y}, {omega_z})')


def get_filter(omega_x: float, omega_y: float, omega_z: float,
    omega: np.ndarray) -> np.ndarray:
    """
    Creates the filter used to denoise the data.
    """

    # A 3D Gaussian is the product of three 1D Gaussians.
    variance = 2.5
    c = 1 / (2 * variance) # 0.2
    filter_x = np.exp(-c*np.power(omega-omega_x, 2))
    filter_y = np.exp(-c*np.power(omega-omega_y, 2))
    filter_z = np.exp(-c*np.power(omega-omega_z, 2))
    filter_3d = np.multiply.outer(np.multiply.outer(filter_x, filter_y), 
        filter_z)

    return filter_3d


def denoise_frequency_domain(ut: np.ndarray, 
    filter_3d: np.ndarray) -> np.ndarray:
    """
    Denoise ut by multiplying it by the filter.
    """

    num_rows = ut.shape[0]
    ut_denoised = np.empty(ut.shape, dtype=complex)
    for row in range(num_rows):
        ut_slice_cube = np.reshape(ut[row, :], filter_3d.shape)
        ut_slice_cube_denoised = ut_slice_cube*filter_3d
        ut_denoised[row, :] = ut_slice_cube_denoised.flatten()

    return ut_denoised


def get_denoised_spatial_domain(ut_denoised: np.ndarray, n: int) -> np.ndarray:
    """
    Converts denoised matrix in frequency domain into spatial domain.
    """

    num_rows = ut_denoised.shape[0]
    u_denoised = np.empty(ut_denoised.shape, dtype=complex)
    for row in range(num_rows):
        ut_slice_cube = np.reshape(ut_denoised[row, :], (n, n, n))
        u_denoised_cube = np.fft.ifftn(np.fft.ifftshift(ut_slice_cube))
        u_denoised[row, :] = u_denoised_cube.flatten()

    return u_denoised


def plot_fft_isosurfaces(description: str, omega: np.ndarray, 
    ut: np.ndarray, filename: str) -> None:
    """
    Plots a few slices of the the FFTed data using isosurfaces.
    """

    print(f'Plotting fft isosurfaces: {description}...')

    (omega_x_grid, omega_y_grid, omega_z_grid) = np.meshgrid(omega, omega, 
        omega, indexing='ij')
    n = len(omega)

    num_slices = ut.shape[0]
    # We only want to plot the first, middle, and last time slices.
    slices = [0, num_slices//2, num_slices-1]

    titles = [f'{description}: slice {slice}' for slice in slices]

    num_rows = 1
    num_cols = len(slices)
    fig = make_subplots(
        rows=num_rows, 
        cols=num_cols,
        specs=[
            [{'is_3d': True}]*num_cols,
        ]*num_rows,
        subplot_titles=titles,
    )
    for s in range(len(slices)):
        ut_slice = np.reshape(ut[slices[s],:], (n, n, n))
        fig.add_trace(
            go.Isosurface(
                x=omega_x_grid.flatten(), 
                y=omega_y_grid.flatten(), 
                z=omega_z_grid.flatten(), 
                value=normalize(ut_slice).flatten(),
                opacity=0.5,
                isomin=0.6,
                isomax=0.9,
                surface_count=3,
                colorscale="Viridis",
            ),
            row=1,
            col=s+1
        )
    fig.update_layout(
        scene_xaxis_title_text="omega_x",
        scene_yaxis_title_text="omega_y",
        scene_zaxis_title_text="omega_z",
        scene2_xaxis_title_text="omega_x",
        scene2_yaxis_title_text="omega_y",
        scene2_zaxis_title_text="omega_z",
        scene3_xaxis_title_text="omega_x",
        scene3_yaxis_title_text="omega_y",
        scene3_zaxis_title_text="omega_z",
    )
    pio.write_html(fig, filename)


def get_marble_path(u_denoised: np.ndarray, 
    domain_values: np.ndarray) -> np.ndarray:
    """
    Gets the path of the marble.
    """

    n = len(domain_values)
    num_rows = u_denoised.shape[0]
    marble_path = np.empty((num_rows, 3))
    for row in range(num_rows):
        ut_slice_cube = np.reshape(u_denoised[row, :], (n, n, n))
        argmax = np.argmax(np.abs(ut_slice_cube))
        [index_x, index_y, index_z] = np.unravel_index(argmax, (n, n, n))
        marble_path[row][0] = domain_values[index_x]
        marble_path[row][1] = domain_values[index_y]
        marble_path[row][2] = domain_values[index_z]

    return marble_path


def plot_marble_path(title: str, marble_path: np.ndarray,
    filename: str) -> None:
    """
    Plots the 3D path of the marble.
    """

    print(f'Plotting marble path...')

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=marble_path[:, 0], y=marble_path[:, 1], z=marble_path[:, 2],
            mode="lines+markers",
            line_color=COLOR1,
            line_width=3,
        )
    )
    fig.update_layout(
        title_text=title,
    )
    pio.write_html(fig, filename)


def print_marble_position(marble_path) -> None:
    """
    Prints the 3D coordinates of the marble.
    """

    num_slices = marble_path.shape[0]
    slices = [0, num_slices//2, num_slices-1]
    for slice in slices:
        (x, y, z) = marble_path[slice,:]
        print(f"Position of marble at time {slice}: {x}, {y}, {z}")


def main() -> None:
    """
    Main program.
    """

    n = 64
    (domain_values, omega) = initialize(domain_limit=15, n=n)
    u = load_data()

    plots_dir_path = find_or_create_dir("plots")
    plot_spatial_isosurfaces('Original data in spatial domain', domain_values, u, 
        os.path.join(plots_dir_path, '1_original_spatial.html'))
    ut = get_fft(u, n)
    ut_average = average_fft(ut)
    plot_fft_isosurface('Average fft', omega, ut_average,
        os.path.join(plots_dir_path, '2_average_fft.html'))
    (omega_x, omega_y, omega_z) = get_peak_frequency(ut_average, omega)
    print_peak_frequency(omega_x, omega_y, omega_z)
    filter_3d = get_filter(omega_x, omega_y, omega_z, omega)
    plot_fft_isosurface('Filter', omega, filter_3d, 
        os.path.join(plots_dir_path, '3_filter.html'))
    ut_denoised = denoise_frequency_domain(ut, filter_3d)
    plot_fft_isosurfaces('Denoised fft', omega, ut_denoised,
        os.path.join(plots_dir_path, '4_denoised_fft.html'))
    u_denoised = get_denoised_spatial_domain(ut_denoised, n)
    plot_spatial_isosurfaces('Denoised data in spatial domain', 
        domain_values, u_denoised,
        os.path.join(plots_dir_path, '5_denoised_spatial.html'))
    marble_path = get_marble_path(u_denoised, domain_values)
    plot_marble_path('Marble path', marble_path,
        os.path.join(plots_dir_path, '6_marble_path.html'))
    print_marble_position(marble_path)


if __name__ == '__main__':
    main()
