import os
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.io.wavfile import read

import utils

S3_URL = 'https://bea-portfolio.s3-us-west-2.amazonaws.com/gabor-transforms/'
PLOTS_FOLDER = 'plots'
COLOR1 = '#3F4F8C'


def plot_wav_data(sample_rate: float, data: np.ndarray, title: str, 
    dirname: str, filename: str) -> None:
    """
    Plots the amplitude of the signal in a wav file.
    """

    print(f'Plotting wav file data...')

    path = os.path.join(dirname, filename)

    t = np.arange(0, len(data))/sample_rate

    # Subsample.
    # subsample = 100
    # t_subsampled = t[::subsample]
    # data_subsampled = data[::subsample]
    t_subsampled = t
    data_subsampled = data

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_subsampled,
            y=data_subsampled,
            mode='lines',
            line_color=COLOR1,
            line_width=3,
        )
    )
    fig.update_layout(
        title_text=title,
        xaxis_title_text='Time (sec)',
        yaxis_title_text='Amplitude',
    )
    pio.write_html(fig, path)


def load_wav_file(wav_filename: str, title: str, plots_dir_path: str, 
    plot_filename: str) -> np.ndarray:
    """
    Loads a wav file, saves it locally and plots it.
    """

    # Load wav file from Amazon S3.
    remote_url = S3_URL+wav_filename
    local_path = utils.download_remote_data_file(remote_url)
    # sample_rate is measurements per second.
    (sample_rate, wav_data) = read(local_path)

    # 65536 = 2^16. 
    # Samples in a wav file have 16 bits, so we scale the amplitudes to be 
    # between 0 and 1.
    wav_data = wav_data/65536

    plot_wav_data(sample_rate, wav_data, title, plots_dir_path, plot_filename)

    return (sample_rate, wav_data)


def get_gaussian_filter(b: float, b_list: np.ndarray, 
    sigma: float) -> np.ndarray:
    """
    Returns the values of a Gaussian filter centered at b, with standard 
    deviation sigma.
    """

    a = 1/(2*sigma**2)
    return np.exp(-a*(b_list-b)**2)


def get_box_filter(b: float, b_list: np.ndarray, width: float) -> np.ndarray:
    """
    Returns the values of a box function filter centered on b, with 
    specified width.
    """

    return np.heaviside(width/2-np.abs(b_list-b), 1)


def get_mexican_hat_filter(b: float, b_list: np.ndarray,
    sigma: float) -> np.ndarray:

    return (1-((b_list-b)/sigma)**2) * np.exp(-(b_list-b)**2/(2*sigma**2))


def plot_spectrograms(spectrograms: List[np.ndarray], plot_x: List[np.ndarray], 
    plot_y: List[np.ndarray], plot_titles: List[str], dirname: str, 
    filename: str, frequency_range: List[float]=None) -> None:
    """
    Plots a list of spectrograms.
    """

    print(f'Plotting spectrograms...')

    path = os.path.join(dirname, filename)

    # Subsample.
    subsample = 100

    # Determine range of frequencies.
    if frequency_range == None:
        ymin = np.min(plot_y[0])
        ymax = np.max(plot_y[0])
    else:
        ymin = frequency_range[0]
        ymax = frequency_range[1]

    rows = len(spectrograms)
    fig = make_subplots(rows=rows, cols=1, subplot_titles=plot_titles)
    for row in range(rows):
        # Subsample.
        spectrogram_subsampled = spectrograms[row][::subsample, :]
        plot_y_subsampled = plot_y[row][::subsample]    

        fig.add_trace(
            go.Heatmap(z=spectrogram_subsampled,
                x=plot_x[row],
                y=plot_y_subsampled,
                coloraxis='coloraxis',
            ),
            col=1,
            row=row+1,
        )
    fig.update_yaxes(
        title_text='Frequency (Hz)',
        range=[ymin, ymax]
    )
    fig.update_xaxes(
        title_text='Time (sec)',
    )
    fig.update_layout(
        coloraxis_colorscale='Viridis',
    )
    pio.write_html(fig, path)


def plot_filters(filters: List[np.ndarray], plot_x: List[np.ndarray], 
    plot_titles: List[str], dirname: str, filename: str) -> None:
    """
    Plots a list of filters.
    """

    print(f'Plotting filters...')

    path = os.path.join(dirname, filename)

    rows = len(filters)
    fig = make_subplots(rows=rows, cols=1, subplot_titles=plot_titles)
    for row in range(rows):
        fig.add_trace(
            go.Scatter(
                x=plot_x[row],
                y=filters[row],
            ),
            col=1,
            row=row+1,
        )
    fig.update_yaxes(
        title_text='Filter value',
    )
    fig.update_xaxes(
        title_text='Time (sec)',
    )
    fig.update_traces(
        line_color=COLOR1,
    )
    fig.update_layout(
        showlegend=False,
    )
    pio.write_html(fig, path)


def get_spectrogram_coordinates(sample_rate: float, data: np.ndarray, num_samples: 
    int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets the time and frequency lists used to construct a spectrogram.
    """

    n = len(data)
    max_time = n/sample_rate

    # Get time steps.
    t_list = np.linspace(0, max_time, n)

    # Get frequencies.
    # Angular frequency.
    # omega_list = (2 * np.pi)/max_time * np.linspace(-n/2, n/2, n+1)[0:-1]
    # Frequency in Hz.
    frequency_list = np.linspace(-n/2, n/2, n+1)[0:-1] / max_time

    # Get sampled time steps.
    t_slide = np.linspace(0, max_time, num_samples)

    return (t_list, frequency_list, t_slide)


def try_different_gabor_widths(sample_rate: float, 
    data: np.ndarray, dirname: str, filter_filename: str,
    spectrogram_filename: str) -> None:
    """
    Filters the temporal data using Gaussian Gabor filter with different
    widths, and transforms the result using FFT.
    """

    print('Producing spectrograms for Gaussian Gabor filters with different '+
        'widths...')

    # Number of samples we want to get from the original data.
    num_samples = 200
    # Subsample the data.
    (t_list, frequency_list, t_slide) = get_spectrogram_coordinates(sample_rate, 
        data, num_samples)

    # Gaussian filter standard deviations.
    sigma_list = [0.1, 0.3, 0.7]

    # Lists used to plot the spectrograms and filters.
    spectrograms = []
    spectrograms_x = []
    spectrograms_y = []
    spectrograms_titles = []
    filters = []
    filters_x = []
    filters_titles = []

    # For each Gaussian filter width:
    for sigma in sigma_list:
        spectrogram = np.empty((len(t_list), len(t_slide)))
        # For each time step, slide the Gabor filter so that it's centered at 
        # the desired time, apply it to the function in time domain, and 
        # transform the result using FFT.
        for (j, b) in enumerate(t_slide):
            g = get_gaussian_filter(b, t_list, sigma)
            ug = data * g
            ugt = np.fft.fftshift(np.fft.fft(ug))
            spectrogram[:, j] = utils.normalize(ugt)

        spectrograms.append(spectrogram)
        spectrograms_x.append(t_slide)
        spectrograms_y.append(frequency_list)
        spectrograms_titles.append(f'Spectrogram using Gabor Gaussian filter with standard deviation = {sigma}')

        # Get a Gaussian filter centered in the middle.
        g = get_gaussian_filter(t_list[len(t_list)//2], t_list, sigma)
        filters.append(g)
        filters_x.append(t_list)
        filters_titles.append(f'Gaussian filter with standard deviation = {sigma}')

    plot_spectrograms(spectrograms, spectrograms_x, spectrograms_y, 
        spectrograms_titles, dirname, spectrogram_filename)
    plot_filters(filters, filters_x, filters_titles, dirname, filter_filename)


def try_different_gabor_timesteps(sample_rate: float, 
    data: np.ndarray, dirname: str, spectrogram_filename: str) -> None:
    """
    Filters the temporal data using a Gaussian Gabor filter by sliding it
    with different timesteps, and transforms the result using FFT.
    """

    print('Producing spectrograms for Gaussian Gabor filters evaluated '+
        'at different time steps...')

    # Number of points to subsample from the original data.
    num_samples_list = [50, 400, 1000]

    # Lists used to plot the spectrograms.
    spectrograms = []
    spectrograms_x = []
    spectrograms_y = []
    spectrograms_titles = []

    for num_samples in num_samples_list:
        # Subsample the data.
        (t_list, frequency_list, t_slide) = get_spectrogram_coordinates(sample_rate, 
            data, num_samples)

        spectrogram = np.empty((len(t_list), len(t_slide)))
        # For each time step, slide the Gabor filter so that it's centered at 
        # the desired time, apply it to the function in time domain, and 
        # transform the result using FFT.
        for (j, b) in enumerate(t_slide):
            g = get_gaussian_filter(b, t_list, sigma=0.1)
            ug = data * g
            ugt = np.fft.fftshift(np.fft.fft(ug))
            spectrogram[:, j] = utils.normalize(ugt)

        spectrograms.append(spectrogram)
        spectrograms_x.append(t_slide)
        spectrograms_y.append(frequency_list)
        spectrograms_titles.append(f'Spectrogram using Gabor Gaussian filter and {num_samples} time steps')

    plot_spectrograms(spectrograms, spectrograms_x, spectrograms_y, 
        spectrograms_titles, dirname, spectrogram_filename)


def try_different_gabor_functions(sample_rate: float, 
    data: np.ndarray, dirname: str, filter_filename: str,
    spectrogram_filename: str) -> None:
    """
    Filters the temporal data using different Gabor filters, and transforms 
    the result using FFT.
    """

    print('Producing spectrograms for different Gabor filters...')

    # Number of samples we want to get from the original data.
    num_samples = 200
    # Subsample the data.
    (t_list, frequency_list, t_slide) = get_spectrogram_coordinates(sample_rate, 
        data, num_samples)

    # Filter 1: Gaussian filter.
    gaussian = lambda b: get_gaussian_filter(b, t_list, sigma=0.1)
    # Filter 2: Box function.
    box_function = lambda b: get_box_filter(b, t_list, width=1)
    # Filter 3: Mexican hat function.
    mexican_hat_function = lambda b: get_mexican_hat_filter(b, t_list, sigma=0.1)
    # List of Gabor filters.
    g_list = [
        gaussian,
        box_function,
        mexican_hat_function
    ]
    # List of Gabor filter names (for plotting).
    filter_name_list = ['Gaussian', 'Box function', 'Mexican hat']

    # Lists used to plot the spectrograms and filters.
    spectrograms = []
    spectrograms_x = []
    spectrograms_y = []
    spectrograms_titles = []
    filters = []
    filters_x = []
    filters_titles = []

    for (i, g) in enumerate(g_list):
        spectrogram = np.empty((len(t_list), len(t_slide)))
        # For each time step, slide the Gabor filter so that it's centered at 
        # the desired time, apply it to the function in time domain, and 
        # transform the result using FFT.
        for (j, b) in enumerate(t_slide):
            ug = data * g(b)
            ugt = np.fft.fftshift(np.fft.fft(ug))
            spectrogram[:, j] = utils.normalize(ugt)

        spectrograms.append(spectrogram)
        spectrograms_x.append(t_slide)
        spectrograms_y.append(frequency_list)
        spectrograms_titles.append(f'Spectrogram using {filter_name_list[i]} filter')

        # Get a filter centered in the middle.
        filters.append(g(t_list[len(t_list)//2]))
        filters_x.append(t_list)
        filters_titles.append(f'{filter_name_list[i]} filter')

    plot_spectrograms(spectrograms, spectrograms_x, spectrograms_y, 
        spectrograms_titles, dirname, spectrogram_filename)
    plot_filters(filters, filters_x, filters_titles, dirname, filter_filename)


def part1(plots_dir_path: str) -> None:
    """
    Analyzes music by Handel.
    """

    (handel_sample_rate, handel_data) = load_wav_file(
        'handel.wav', 'Handel music data',
        plots_dir_path, 
        '1_handel_data.html')
    try_different_gabor_widths(handel_sample_rate, handel_data, 
        plots_dir_path, 
        '2_gaussian_filters.html', 
        '3_spectrograms_different_widths.html')
    try_different_gabor_timesteps(handel_sample_rate, handel_data,
        plots_dir_path, 
        '4_spectrograms_different_timesteps.html')
    try_different_gabor_functions(handel_sample_rate, handel_data, 
        plots_dir_path, 
        '5_different_filters.html', 
        '6_spectrograms_gabor_functions.html')


def produce_spectrograms(sample_rates: List, data_list: List, dirname: str, 
    spectrogram_filename: str, filtered_filename: str, 
    zoomed_filtered_filename: str) -> None:
    """
    Produces and plots spectrograms for two versions of 'Mary had a little 
    lamb', on piano and recorder.
    """

    print("Producing spectrograms for 'Mary had a little lamb'...")

    # Number of samples we want to get from the original data.
    num_samples = 200
    # Standard deviation for Gaussian filter.
    sigma = 0.1
    # Lists used to plot the spectrograms.
    spectrograms = []
    spectrograms_x = []
    spectrograms_y = []

    for (i, data) in enumerate(data_list):
        sample_rate = sample_rates[i]
        (t_list, frequency_list, t_slide) = get_spectrogram_coordinates(sample_rate, 
            data, num_samples)

        spectrogram = np.empty((len(t_list), len(t_slide)))
        # For each time step, slide the Gabor filter so that it's centered at 
        # the desired time, apply it to the function in time domain, and 
        # transform the result using FFT.
        for (j, b) in enumerate(t_slide):
            g = get_gaussian_filter(b, t_list, sigma)
            ug = data * g
            ugt = np.fft.fftshift(np.fft.fft(ug))
            spectrogram[:, j] = np.abs(ugt)

        spectrograms.append(np.log(spectrogram))
        spectrograms_x.append(t_slide)
        spectrograms_y.append(frequency_list)

    # Plot spectrograms.
    spectrograms_titles = ['Log of spectrogram for "Mary had a little lamb" on piano', 
        'Log of spectrogram for "Mary had a little lamb" on recorder']
    plot_spectrograms(spectrograms, spectrograms_x, spectrograms_y, 
        spectrograms_titles, dirname, spectrogram_filename)

    # Plot filtered spectrograms.
    filtered_spectrograms = [filter_spectrogram(s) for s in spectrograms]
    filtered_spectrograms_titles = [
        'Filtered spectrogram for "Mary had a little lamb" on piano', 
        'Filtered spectrogram for "Mary had a little lamb" on recorder']
    plot_spectrograms(filtered_spectrograms, spectrograms_x, spectrograms_y, 
        filtered_spectrograms_titles, dirname, filtered_filename)

    # Plot zoomed and filtered spectrograms.
    zoomed_filtered_spectrograms_titles = [
        'Zoomed filtered spectrogram for "Mary had a little lamb" on piano', 
        'Zoomed filtered spectrogram for "Mary had a little lamb" on recorder']
    plot_spectrograms(filtered_spectrograms, spectrograms_x, spectrograms_y, 
        zoomed_filtered_spectrograms_titles, dirname, zoomed_filtered_filename, 
        frequency_range=[150, 1200])


def filter_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """
    Removes small amplitudes and overtones of spectrogram.
    """

    filtered_spectrogram = np.empty(spectrogram.shape)
    rows = spectrogram.shape[0]
    cols = spectrogram.shape[1]
    for col in range(cols):
        spectrum = spectrogram[:, col]
        max_amplitude = np.max(spectrum)
        # Remove small amplitudes.
        spectrum = spectrum * (spectrum > max_amplitude*0.5)
        filtered_spectrogram[:, col] = spectrum
        # Remove overtones. The highest amplitude should happen at the 
        # fundamental frequency. The overtones occur at multiples of the
        # fundamental frequency. So to remove overtones, we can remove
        # frequencies beyond 1.5 times the fundamental frequency.
        top_half_spectrum = spectrum[rows//2:]
        index_fundamental_frequency = np.argmax(top_half_spectrum)
        highest_index_to_keep = rows//2 + int(index_fundamental_frequency*1.5)
        lowest_index_to_keep = rows//2 - int(index_fundamental_frequency*1.5)
        filtered_spectrogram[highest_index_to_keep:, col] = 0
        filtered_spectrogram[:lowest_index_to_keep, col] = 0

    return filtered_spectrogram


def part2(plots_dir_path: str) -> None:
    """
    Analyzes the song 'Mary had a little lamb'.
    """
    
    (mary1_sample_rate, mary1_data) = load_wav_file(
        'music1.wav', 'Mary had a little lamb - piano',
        plots_dir_path, 
        '7_mary1_data.html')
    (mary2_sample_rate, mary2_data) = load_wav_file(
        'music2.wav', 'Mary had a little lamb - recorder',
        plots_dir_path, 
        '8_mary2_data.html')
    produce_spectrograms([mary1_sample_rate, mary2_sample_rate], [mary1_data, 
        mary2_data], plots_dir_path, 
        '9_mary_spectrograms.html',
        '10_filtered_spectrograms.html',
        '11_zoomed_filtered_spectrograms.html',
        )


def main() -> None:
    """
    Main program.
    """
        
    plots_dir_path = utils.find_or_create_dir(PLOTS_FOLDER)

    part1(plots_dir_path)
    part2(plots_dir_path)


if __name__ == '__main__':
    main()
