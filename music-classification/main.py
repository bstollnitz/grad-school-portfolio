import os
import urllib.request
from typing import Dict, List, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import scipy
import sklearn.metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import utils_graph
import utils_io

S3_URL = 'https://bea-portfolio.s3-us-west-2.amazonaws.com/music-classification/'

SONGS_FOLDER = 'songs'
DATA_FOLDER = 'data'
SPECTROGRAMS_FOLDER = 'spectrograms'
PLOTS_FOLDER = 'plots'

# We'll subsample the frequencies so that we end up roughly with 
# NUM_FREQUENCIES frequencies.
NUM_FREQUENCIES = 500
# We'll slide the Gaussian filter NUM_TIMES times.
NUM_TIMES = 500
# Total number of features.
NUM_FEATURES = NUM_FREQUENCIES*NUM_TIMES

GENRES = {'blues': ['aretha_franklin', 'bb_king', 'john_lee_hooker'],
          'classical': ['handel', 'haydn', 'vivaldi'], 
          'electronic': ['aphex_twin', 'daft_punk', 'royksopp'], 
          'grunge': ['nirvana', 'pearl_jam', 'soundgarden']
          }
SONGS = ['song1', 'song2', 'song3']
CLIPS = ['clip1', 'clip2', 'clip3']

CLIP_DATA_FILENAMES = []
for genre in GENRES:
    bands = GENRES[genre]
    for band in bands:
        for song in SONGS:
            for clip in CLIPS:
                CLIP_DATA_FILENAMES.append(f'{genre}-{band}-{song}-{clip}.npz')


FREQUENCY_DATA_FILENAME = 'frequency_list.npz'
TIME_DATA_FILENAME = 'time_list.npz'


def _read_wav_files(folder: str, wav_files: Dict[str, np.ndarray]) -> float:
    """Reads music wav files recursively from disk. Populates a dictionary
    whose keys are the path to each file, and whose values are the 
    corresponding ndarray of audio samples.
    Returns the sample rate, which is assumed to be the same for all files.
    """
    if os.path.exists(folder):
        items = os.listdir(folder)
        for item in items:
            item_path = os.path.join(folder, item)
            if os.path.isdir(item_path):
                sample_rate = _read_wav_files(item_path, wav_files)
            elif os.path.isfile(item_path):
                key = item_path.replace(f'{SONGS_FOLDER}/', 
                    '').replace('.wav', '').replace('/', '-')
                (sample_rate, wav_files[key]) = utils_io.load_wav_file(item_path)

    return sample_rate


def _get_gaussian_filter(b: float, b_list: np.ndarray, 
    sigma: float) -> np.ndarray:
    """
    Returns the values of a Gaussian filter centered at b, with standard 
    deviation sigma.
    """
    a = 1/(2*sigma**2)
    return np.exp(-a*(b_list-b)**2)


def _get_spectrogram_coordinates(sample_rate: float, data: np.ndarray, 
    num_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gets the time and frequency lists used to construct a spectrogram.
    """
    n = len(data)
    max_time = n/sample_rate

    # Get all time steps.
    all_time_list = np.linspace(0, max_time, n)

    # Get sampled time steps. We'll slide the Gabor transform to each of these.
    sampled_time_list = np.linspace(0, max_time, num_samples)

    # Get all frequencies.
    # Angular frequency.
    # omega_list = (2 * np.pi)/max_time * np.linspace(-n/2, n/2, n+1)[0:-1]
    # Frequency in Hz.
    all_frequency_list = np.linspace(-n/2, n/2, n+1)[0:-1] / max_time

    return (all_frequency_list, all_time_list, sampled_time_list)


def _normalize(my_array: np.ndarray) -> np.ndarray:
    """Normalizes an ndarray.
    """
    return np.abs(my_array)/np.max(np.abs(my_array))


def _generate_spectrogram(data: np.ndarray, audio_sample_rate: 
    float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns spectrogram for data passed as a parameter.
    """
    # We'll construct the spectrogram using a Gaussian filter.
    sigma = 0.1
    # The spectrogram is of shape (number of frequencies, sampled number 
    # of times).
    (frequency_list, all_time_list, 
        sampled_time_list) = _get_spectrogram_coordinates(
        audio_sample_rate, data, NUM_TIMES)
    spectrogram = np.empty((len(frequency_list), NUM_TIMES))

    # For each time step, slide the Gabor filter so that it's centered at 
    # the desired time, apply it to the function in time domain, and 
    # transform the result using FFT.
    for (i, b) in enumerate(sampled_time_list):
        g = _get_gaussian_filter(b, all_time_list, sigma)
        ug = data * g
        ugt = np.fft.fftshift(np.fft.fft(ug))
        spectrogram[:, i] = _normalize(ugt)
    
    # We'll reduce the number of frequencies by subsampling every 
    # frequency_subsampling_rate frequencies.
    frequency_subsampling_step = data.shape[0]//NUM_FREQUENCIES
    spectrogram = spectrogram[::frequency_subsampling_step, :]
    sampled_frequency_list = frequency_list[::frequency_subsampling_step]

    return (spectrogram, sampled_frequency_list, sampled_time_list)


def _generate_spectrograms() -> None:
    """Reads music wav files, produces spectrograms and saves them.
    """
    # Read wav files.
    print('Reading wav files...')
    wav_files = {}
    sample_rate = _read_wav_files(SONGS_FOLDER, wav_files)

    # Generate spectrograms.
    print('Generating spectrograms...')
    for wav_file_key in wav_files:
        print(f'Generating spectrogram for {wav_file_key}...')
        (spectrogram, sampled_frequency_list, 
            sampled_time_list) = _generate_spectrogram(
            wav_files[wav_file_key], sample_rate)
        # Save spectrogram npz and image.
        spectrogram_npz_path = os.path.join(DATA_FOLDER, wav_file_key+'.npz')
        np.savez(spectrogram_npz_path, spectrogram)

    # Save frequency and time lists.
    print('Saving frequencies and times...')
    frequencies_path = os.path.join(DATA_FOLDER, FREQUENCY_DATA_FILENAME)
    time_path = os.path.join(DATA_FOLDER, TIME_DATA_FILENAME)
    np.savez(frequencies_path, sampled_frequency_list)
    np.savez(time_path, sampled_time_list)


def _download_spectrograms() -> None:
    """Downloads spectrograms from remote location if they're not yet 
    present locally.
    """
    for item in CLIP_DATA_FILENAMES:
        utils_io.download_remote_data_file(DATA_FOLDER, S3_URL, item)

    utils_io.download_remote_data_file(DATA_FOLDER, S3_URL, FREQUENCY_DATA_FILENAME)
    utils_io.download_remote_data_file(DATA_FOLDER, S3_URL, TIME_DATA_FILENAME)


def _generate_spectrogram_images() -> None:
    """Reads npz files containing spectrogram ndarrays and generates 
    spectrogram images.
    """
    print('Generating and saving spectrogram images...')
    items = os.listdir(DATA_FOLDER)
    for item in items:
        if (item != FREQUENCY_DATA_FILENAME and 
            item != TIME_DATA_FILENAME):
            item_path = os.path.join(DATA_FOLDER, item)
            spectrogram = np.load(item_path)['arr_0']
            image_name = item.replace('.npz', '.png')
            image_path = os.path.join(SPECTROGRAMS_FOLDER, image_name)
            if os.path.exists(image_path):
                break
            print(f'Saving image {image_name}...')
            utils_io.save_image(spectrogram, SPECTROGRAMS_FOLDER, image_name)
    print('Done generating and saving spectrogram images.')


def preprocess_data(method: str) -> None:
    """Generates spectrograms from wav files or downloads them from a remote
    location.
    """
    if method == 'generate':
        _generate_spectrograms()
    elif method == 'download':
        _download_spectrograms()

    _generate_spectrogram_images()


def _split_data_band_classification(genre_band_list: List[str]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gets data and labels for the training and test data set, to be used in 
    band classification.
    Sets aside a song+clip for each band for test, and uses all others
    for training.
    """
    training_data = np.empty((NUM_FEATURES, 0))
    training_labels = np.empty((1, 0))
    test_data = np.empty((NUM_FEATURES, 0))
    test_labels = np.empty((1, 0))
    for genre_band in genre_band_list:
        for song in SONGS:
            for clip in CLIPS:
                filepath = os.path.join(DATA_FOLDER, 
                    f'{genre_band}-{song}-{clip}.npz')
                spectrogram = np.load(filepath)['arr_0'].reshape(-1, 1)
                if song == 'song3' and clip == 'clip3':
                    test_data = np.append(test_data, spectrogram, axis=1)
                    test_labels = np.append(test_labels, genre_band) 
                else:
                    training_data = np.append(training_data, spectrogram, axis=1)
                    training_labels = np.append(training_labels, genre_band)

    return (training_data, training_labels, test_data, test_labels)


def _split_data_genres() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gets data and labels for the training and test data set, to be used in 
    genre classification.
    Sets aside a song+clip for each band for test, and uses all other
    for training.
    """
    training_data = np.empty((NUM_FEATURES, 0))
    training_labels = np.empty((1, 0))
    test_data = np.empty((NUM_FEATURES, 0))
    test_labels = np.empty((1, 0))

    for filename in CLIP_DATA_FILENAMES:
        filepath = os.path.join(DATA_FOLDER, filename)
        spectrogram = np.load(filepath)['arr_0'].reshape(-1, 1)
        (genre, _, song, clip, _) = tuple(filename.replace('.', '-').split('-'))
        if song == 'song3' and clip == 'clip3':
            test_data = np.append(test_data, spectrogram, axis=1)
            test_labels = np.append(test_labels, genre) 
        else:
            training_data = np.append(training_data, spectrogram, axis=1)
            training_labels = np.append(training_labels, genre)

    return (training_data, training_labels, test_data, test_labels)


def _reduce_data(u: np.ndarray, training_data: np.ndarray, test_data: 
    np.ndarray, num_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reduces the training and test data by projecting into U_reduced. 
    U_reduced keeps only the first few columns of matrix U.
    """
    u_reduced = u[:, :num_modes] 
    training_data_reduced = u_reduced.T.dot(training_data)
    test_data_reduced = u_reduced.T.dot(test_data)
    return (training_data_reduced, test_data_reduced)


def _calculate_accuracy(actual: np.ndarray, predictions: np.ndarray) -> None:
    """Calculates the accuracy of predictions.
    """
    accuracy = sklearn.metrics.accuracy_score(actual, predictions)
    print(f'    Predicted test labels: {predictions}')
    print(f'    Actual test labels:    {actual}')
    print(f'    Accuracy: {accuracy * 100:.2f}%')


def _lda_classification_1(training_data: np.ndarray, training_labels:
    np.ndarray, test_data: np.ndarray, test_labels: np.ndarray,
    title: str) -> None:
    """Classifies using builtin LDA from scikit-learn.
    """
    print('- Scikit-learn\'s LDA')
    lda = LinearDiscriminantAnalysis(solver='eigen')
    lda.fit(training_data.T, training_labels)
    predicted_labels = lda.predict(test_data.T)
    _calculate_accuracy(test_labels, predicted_labels)

    # Visualize classes.
    projected_training_data = lda.transform(training_data.T).T
    projected_test_data = lda.transform(test_data.T).T
    projected_centroids = lda.transform(lda.means_).T
    classes = lda.classes_
    utils_graph.graph_classes(projected_training_data, training_labels, 
        projected_test_data, test_labels, 
        projected_centroids, classes, title + ' (sklearn)', PLOTS_FOLDER,
        f'sklearn_reduced_subspace_{title.replace(" ", "_")}.html')


def _lda_classification_2(training_data: np.ndarray, training_labels: 
    np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, 
    title: str) -> None:
    """Classifies using our own n-class LDA.
    """
    print('- My LDA')
    classes = np.unique(training_labels)
    num_classes = len(classes)
    num_features = training_data.shape[0]

    # Calculate the mean spectrogram for each class.
    mu_list = np.empty((num_features, 0))
    for c in classes:
        indices = np.argwhere(training_labels == c).flatten()
        data_in_class = training_data[:, indices]
        mu = np.mean(data_in_class, axis=1, keepdims=True)
        mu_list = np.append(mu_list, mu, axis=1)
    
    # Mean of the class means.
    mu = np.mean(mu_list, axis=1)

    # Calculate sb, the variance of the difference between a class mean and 
    # the average mean.
    # Calculate sw, the variance within each class.
    sb = np.zeros((num_features, num_features))
    sw = np.zeros((num_features, num_features))
    for (i, c) in enumerate(classes):
        diff = mu_list[:, i] - mu
        indices = np.argwhere(training_labels == c).flatten()
        data_in_class = training_data[:, indices]
        num_instances = data_in_class.shape[1]
        sb += num_instances * diff.dot(diff.T)
        sw += np.cov(data_in_class)

    # Calculate the eigenvectors, and sort them by magnitude of eigenvalues.
    (eigenvalues, eigenvectors) = scipy.linalg.eig(sb, b=sw)
    sorted_indices = np.flip(np.argsort(np.abs(eigenvalues)))
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvectors = eigenvectors[:, :num_classes-1]

    # Project training data, test data and centroids into eigenvectors.
    projection_matrix = eigenvectors.T
    projected_training_data = projection_matrix.dot(training_data)
    projected_centroids = projection_matrix.dot(mu_list)
    projected_test_data = projection_matrix.dot(test_data)

    # Classify the test data.
    predicted_labels = np.empty((0, ))
    for i in range(test_data.shape[1]):
        # Find the closest centroid.
        distances = np.linalg.norm(projected_centroids - 
            projected_test_data[:, i].reshape(-1, 1), 
            ord=2, axis=0)
        index_min_distance = np.argmin(distances)
        predicted_labels = np.append(predicted_labels, classes[index_min_distance])

    # Calculate accuracy of prediction.
    _calculate_accuracy(test_labels, predicted_labels)

    # Visualize classes.
    utils_graph.graph_classes(projected_training_data, training_labels, 
        projected_test_data, test_labels, 
        projected_centroids, classes, title, PLOTS_FOLDER,
        f'reduced_subspace_{title.replace(" ", "_")}.html')


def _classify(training_data: np.ndarray, training_labels: np.ndarray, 
    test_data: np.ndarray, test_labels: np.ndarray, num_modes: int, 
    title: str) -> None:
    """Classifies bands and genres using SVD and LDA.
    """
    # Perform SVD.
    (u, s, vh) = np.linalg.svd(training_data, full_matrices=False)

    # Plot the normalized singular values.
    normalized_s = s / np.sum(s)
    utils_graph.graph_2d_markers(
        np.asarray(range(1, len(normalized_s)+1)),
        normalized_s, 'Mode', 'Normalized singular value',
        f'Singular values for classification of {title}',
        PLOTS_FOLDER,
        f'singular_values_{title.replace(" ", "_")}.html')

    # Reduce the training and test data.
    (training_data_reduced, test_data_reduced) = _reduce_data(u, training_data, 
        test_data, num_modes)

    # Classify using LDA.
    _lda_classification_1(training_data_reduced, training_labels, 
        test_data_reduced, test_labels, title)
    _lda_classification_2(training_data_reduced, training_labels, 
        test_data_reduced, test_labels, title)


def classify_bands_different_genres() -> None:
    """Classifies bands from different genres using SVD and LDA.
    """
    print('\n*** Classifying bands of different genres ***')

    # What to classify.
    genre_band_list = ['electronic-royksopp', 'blues-bb_king', 'grunge-soundgarden']

    # Split data into training and test sets, and get labels for both sets.
    (training_data, training_labels, test_data, 
        test_labels) = _split_data_band_classification(genre_band_list)

    # Classify.
    _classify(training_data, training_labels, test_data, 
        test_labels, 21, 'bands of different genres')


def classify_bands_same_genre() -> None:
    """Classifies bands from the same genre using SVD and LDA.
    """
    print('\n*** Classifying bands of the same genre ***')

    for genre in GENRES:
        print(f'\nGenre: {genre}')

        # What to classify.
        genre_band_list = [f'{genre}-{band}' for band in GENRES[genre]]

        # Split data into training and test sets, and get labels for both sets.
        (training_data, training_labels, test_data, 
            test_labels) = _split_data_band_classification(genre_band_list)

        _classify(training_data, training_labels, test_data, 
            test_labels, 20, f'bands of {genre} genre')


def classify_genres() -> None:
    """Classifies music genres using SVD and LDA.
    """
    print('\n*** Classifying genres ***')

    # Split data into training and test sets, and get labels for both sets.
    (training_data, training_labels, test_data, 
        test_labels) = _split_data_genres()

    # Classify.
    # sklearn: 12 modes -> 50% accuracy
    # mine: 9 or 11 modes -> 50% accuracy
    _classify(training_data, training_labels, test_data, test_labels, 11,
        'genres')


def main() -> None:
    """Main program.
    """
    utils_io.find_or_create_dir(DATA_FOLDER)
    utils_io.find_or_create_dir(SPECTROGRAMS_FOLDER)
    utils_io.find_or_create_dir(PLOTS_FOLDER)
    preprocess_data(method='download')
    classify_bands_different_genres()
    classify_bands_same_genre()
    classify_genres()


if __name__ == '__main__':
    main()
