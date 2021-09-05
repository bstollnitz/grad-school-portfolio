import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import h5py
import pywt
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

import utils_graph
import utils_io
import utils_nn
from cnn import CNN
from gram_dataset import GramDataset
from hyperparameters import Hyperparameters
from signal_data import SignalData

PLOTS_FOLDER = 'plots'
USE_CUDA = torch.cuda.is_available()

SPECTROGRAMS_IMAGES_FOLDER = 'spectrograms/images'
SPECTROGRAMS_DATA_FOLDER = 'spectrograms/data'
SPECTROGRAMS_TRAIN_FILE_NAME = 'train_spectrograms.hdf5'
SPECTROGRAMS_TEST_FILE_NAME = 'test_spectrograms.hdf5'

SCALEOGRAMS_IMAGES_FOLDER = 'scaleograms/images'
SCALEOGRAMS_DATA_FOLDER = 'scaleograms/data'
SCALEOGRAMS_TRAIN_FILE_NAME = 'train_scaleograms.hdf5'
SCALEOGRAMS_TEST_FILE_NAME = 'test_scaleograms.hdf5'



def _save_grams(signals: np.ndarray, file_name: str, gram_type: str):
    """Computes and saves spectrograms or scaleograms for all the signals.
    """
    if gram_type == 'spectrograms':
        data_folder = SPECTROGRAMS_DATA_FOLDER
        create_gram_func = _create_spectrogram
    elif gram_type == 'scaleograms':
        data_folder = SCALEOGRAMS_DATA_FOLDER
        create_gram_func = _create_scaleogram
    else:
        raise Exception('gram_type must be "spectrograms" or "scaleograms"')

    gram_path = Path(data_folder, file_name)

    if not gram_path.exists():
        print(f'  Generating and saving {gram_type} to {file_name}.')
        Path(data_folder).mkdir(exist_ok=True, parents=True)
        # 2947 x 9 x 128
        (num_instances, num_components, num_timesteps) = signals.shape
        # 2947 x 9 x 128 x 128
        grams = np.zeros((num_instances, num_components,
            num_timesteps, num_timesteps))

        graph_gaussian_signal = True
        for instance in range(num_instances):
            for component in range(num_components):
                signal = signals[instance, component, :]
                # 128 x 128
                gram = create_gram_func(signal, graph_gaussian_signal)
                grams[instance, component, :, :] = gram
                graph_gaussian_signal = False

        with h5py.File(gram_path, 'w') as group:
            group.create_dataset(name=gram_type, shape=grams.shape, 
                dtype='f', data=grams)


def _save_gram_images(labels: np.ndarray, activity_names: dict,
    gram_type: str) -> None:
    """Saves a few spectrogram or scaleogram images for each component if this 
    hasn't been done already, for debugging purposes.
    Number of images saved: number of activities (6) x number of sets per
    activity (3) x number of components (9).
    """
    if gram_type == 'spectrograms':
        data_path = Path(SPECTROGRAMS_DATA_FOLDER, SPECTROGRAMS_TEST_FILE_NAME)
        images_folder = Path(SPECTROGRAMS_IMAGES_FOLDER)
    elif gram_type == 'scaleograms':
        data_path = Path(SCALEOGRAMS_DATA_FOLDER, SCALEOGRAMS_TEST_FILE_NAME)
        images_folder = Path(SCALEOGRAMS_IMAGES_FOLDER)
    else:
        raise Exception('gram_type must be "spectrograms" or "scaleograms"')
    
    # Create images folder if it doesn't exist.
    images_folder.mkdir(exist_ok=True, parents=True)

    # Open data file.
    with h5py.File(data_path, 'r') as gram_file:
        # If there are no images in the folder:
        images = [item for item in images_folder.iterdir() if item.suffix == '.png']
        if len(images) == 0:
            print('  Saving images.')
            num_sets_per_activity = 3
            # Find all the unique activity numbers in our labels.
            activities = np.unique(labels)
            # For each activity present in the labels:
            for activity in activities:
                instance_indices = np.nonzero(labels == activity)[0][0:num_sets_per_activity]
                # For each instance of that activity:
                for instance_index in instance_indices:
                    # Read the image values from data file.
                    activity_grams = gram_file[gram_type][instance_index, :, :, :]
                    # For each of the 9 components: 
                    num_components = activity_grams.shape[0]
                    for component in range(num_components):
                        gram = activity_grams[component, :, :]
                        activity_name = activity_names[activity]
                        file_name = f'{activity_name}_{instance_index + 1}_{component + 1}.png'
                        # Save the spectrogram or scaleogram.
                        utils_io.save_image(gram, images_folder, file_name)


def _normalize(my_array: np.ndarray) -> np.ndarray:
    """Normalizes an ndarray to values between 0 and 1.
    The max value maps to 1, but the min value may not hit 0.
    """
    return np.abs(my_array)/np.max(np.abs(my_array))


def _train_cnn_network(hyperparameter_dict: dict, full_train_labels: np.ndarray, 
    gram_type: str) -> Tuple[CNN, List, List, List, List]:
    """Trains a CNN using the specified hyperparameters.
    """
    # Ensure reproducibility by giving PyTorch the same seed every time we train.
    torch.manual_seed(1)

    # Choose the data path.
    if gram_type == 'spectrograms':
        full_train_data_path = Path(SPECTROGRAMS_DATA_FOLDER, SPECTROGRAMS_TRAIN_FILE_NAME)
    elif gram_type == 'scaleograms':
        full_train_data_path = Path(SCALEOGRAMS_DATA_FOLDER, SCALEOGRAMS_TRAIN_FILE_NAME)
    else:
        raise Exception('gram_type must be "spectrograms" or "scaleograms"')

    # Print hyperparameters.
    print(f'Hyperparameters: {hyperparameter_dict}')

    # Get hyperparameters.
    learning_rate = hyperparameter_dict['learning_rate']
    batch_size = hyperparameter_dict['batch_size']
    optimizer_str = hyperparameter_dict['optimizer']

    # There are 6 labels, and Pytorch expects them to go from 0 to 5.
    full_train_labels = full_train_labels - 1

    # Get generators.
    full_training_data = GramDataset(full_train_data_path, full_train_labels)
    (training_generator, validation_generator) = utils_nn.get_trainval_generators(
        full_training_data, batch_size, num_workers=0, training_fraction=0.8)

    # Crete CNN.
    cnn = CNN()

    # Parameters should be moved to GPU before constructing the optimizer.
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    cnn = cnn.to(device)

    # Get optimizer.
    optimizer = None
    if optimizer_str == 'adam':
        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    elif optimizer_str == 'sgd':
        optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
    else:
        raise Exception(f'Specified optimizer not valid: {optimizer_str}')

    training_accuracy_list = []
    training_loss_list = []
    validation_accuracy_list = []
    validation_loss_list = []
    max_epochs = 10
    for epoch in range(max_epochs):
        print(f'Epoch {epoch}')

        # Training data.
        (training_accuracy, training_loss) = utils_nn.fit(cnn, 
            training_generator, optimizer, USE_CUDA)
        training_accuracy_list.append(training_accuracy)
        training_loss_list.append(training_loss)

        # Validation data.
        (validation_accuracy, validation_loss) = utils_nn.evaluate(cnn, 
            validation_generator, 'Validation', USE_CUDA)
        validation_accuracy_list.append(validation_accuracy)
        validation_loss_list.append(validation_loss)

    return (cnn, training_accuracy_list, training_loss_list, 
        validation_accuracy_list, validation_loss_list)


def _get_cnn_hyperparameters() -> Hyperparameters:
    """Returns hyperparameters used to tune the network.
    """
    # Spectrograms

    # First pass:
    # hyperparameter_values = Hyperparameters({
    #     'learning_rate': [0.1, 0.01, 0.001],
    #     'batch_size': [32, 64, 128],
    #     'optimizer': ['adam', 'sgd']
    #     })
    # Results: 
    # optimizer: adam, batch size: 64, learning rate: 0.001
    # Adam with learning rate 0.001 seems to work best, regardless of batch size.

    # Second pass:
    # hyperparameter_values = Hyperparameters({
    #     'learning_rate': [0.001],
    #     'batch_size': [8, 16, 32, 64, 256],
    #     'optimizer': ['adam']
    #     })

    # Best: 
    # optimizer: adam, batch size: 64, learning rate: 0.001

    # Scaleograms 

    # First pass:
    # hyperparameter_values = Hyperparameters({
    #     'learning_rate': [0.1, 0.01, 0.001],
    #     'batch_size': [32, 64, 128],
    #     'optimizer': ['adam', 'sgd']
    #     })
    # Results: 
    # optimizer: adam, batch size: 32, learning rate: 0.001
    # Adam with learning rate 0.001 seems to work best, regardless of batch size.

    # Second pass:
    hyperparameter_values = Hyperparameters({
        'learning_rate': [0.001],
        'batch_size': [8, 16, 32, 256],
        'optimizer': ['adam']
        })

    # Best: 
    # optimizer: adam, batch size: 32, learning rate: 0.001

    return hyperparameter_values


def _tune_cnn_hyperparameters(full_train_labels: np.ndarray, 
    gram_type: str) -> None:
    """Classifies spectrograms or scaleograms using a CNN.
    """
    print('  Tuning hyperparameters.')
    start_time = time.time()

    # Hyperparameters to tune.
    hyperparameter_values = _get_cnn_hyperparameters()
    hyperparameter_combinations = hyperparameter_values.sample_combinations()

    # Create Tensorboard writer.
    with SummaryWriter(f'runs/{gram_type}', filename_suffix='') as writer:
        # Hyperparameter loop.
        for hyperparameter_dict in hyperparameter_combinations:
            (_, _, _, validation_accuracy_list, _) = _train_cnn_network(
                hyperparameter_dict, full_train_labels, gram_type)

            writer.add_hparams(hyperparameter_dict,
                {f'hparam/{gram_type}/validation_accuracy': validation_accuracy_list[-1]})

    utils_io.print_elapsed_time(start_time, time.time())


def _test_cnn_network(cnn: CNN, test_labels: np.ndarray, hyperparameter_dict: dict, 
    gram_type: str) -> Tuple[float, float]:
    """Returns accuracy and loss of specified CNN for specified test data and
    specified hyperparameters.
    """
    if gram_type == 'spectrograms':
        test_data_path = Path(SPECTROGRAMS_DATA_FOLDER, SPECTROGRAMS_TEST_FILE_NAME)
    elif gram_type == 'scaleograms':
        test_data_path = Path(SCALEOGRAMS_DATA_FOLDER, SCALEOGRAMS_TEST_FILE_NAME)
    else:
        raise Exception('gram_type must be "spectrograms" or "scaleograms"')

    # There are 6 labels, and Pytorch expects them to go from 0 to 5.
    test_labels = test_labels - 1

    # Get test generator.
    batch_size = hyperparameter_dict['batch_size']
    test_data = GramDataset(test_data_path, test_labels)
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    test_generator = data.DataLoader(test_data, **params)

    (test_avg_accuracy, test_avg_loss) = utils_nn.evaluate(cnn, test_generator, 
        'Test', USE_CUDA)

    return (test_avg_accuracy, test_avg_loss)


def _test_best_cnn_hyperparameters(full_train_labels: np.ndarray, 
    test_labels: np.ndarray, gram_type: str) -> None:
    """Use CNN with best hyperparameters to predict labels for test data.
    Produces accuracy and loss graphs for training and validation data, as 
    well as accuracy and loss values for test data.
    """
    hyperparameter_dict = {}
    if gram_type == 'spectrograms':
        hyperparameter_dict = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'optimizer': 'adam',
            }
    elif gram_type == 'scaleograms':
        hyperparameter_dict = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam',
            }
    else:
        raise Exception('gram_type must be "spectrograms" or "scaleograms"')

    (cnn, training_accuracy_list, 
        training_loss_list, 
        validation_accuracy_list, 
        validation_loss_list) = _train_cnn_network(hyperparameter_dict, 
        full_train_labels, gram_type)

    utils_graph.graph_nn_results(training_accuracy_list, validation_accuracy_list, 
        f'Training and validation accuracy of classification of {gram_type}', 
        'Accuracy', PLOTS_FOLDER, f'{gram_type}_accuracy.html')

    utils_graph.graph_nn_results(training_loss_list, validation_loss_list, 
        f'Training and validation loss of classification of {gram_type}', 
        'Loss', PLOTS_FOLDER, f'{gram_type}_loss.html')

    _test_cnn_network(cnn, test_labels, hyperparameter_dict, gram_type)

    with SummaryWriter(f'runs/{gram_type}', filename_suffix='') as writer:
        num_epochs_train_val = len(training_accuracy_list)
        for i in range(num_epochs_train_val):
            writer.add_scalars(f'{gram_type}/accuracy', {
                'training': training_accuracy_list[i],
                'validation': validation_accuracy_list[i]
                }, i)
            writer.add_scalars(f'{gram_type}/loss', {
                'training': training_loss_list[i],
                'validation': validation_loss_list[i]
                }, i)

    # Spectrograms
    # Test accuracy: 87.49%
    # Test loss: 0.36

    # Scaleograms 
    # Test accuracy: 89.26%
    # Test loss: 0.44


def _get_gaussian_filter(b: float, b_list: np.ndarray, 
    sigma: float) -> np.ndarray:
    """Returns the values of a Gaussian filter centered at b, with standard 
    deviation sigma.
    """
    a = 1/(2*sigma**2)
    return np.exp(-a*(b_list-b)**2)


def _graph_gaussian_signal(signal: np.ndarray, g: np.ndarray) -> None:
    """Saves a graph containing a signal and the Gaussian function used to 
    filter it.
    """
    # Plot Gaussian filter and signal overlayed in same graph.
    time_list = np.arange(len(signal))
    signal = _normalize(signal) 
    x = np.append([time_list], [time_list], axis=0)
    y = np.append([g], [signal], axis=0)
    utils_graph.graph_overlapping_lines(x, y, 
        ['Gaussian filter', 'Signal'],
        'Time', 'Amplitude', 
        'Example of a signal and corresponding Gaussian filter',
        PLOTS_FOLDER, 'sample_gaussian_signal.html')


def _create_spectrogram(signal: np.ndarray, 
    graph_gaussian_signal: bool) -> np.ndarray:
    """Creates spectrogram for signal.
    """
    n = len(signal)
    # Times of the input signal.
    time_list = np.arange(n)
    # Horizontal axis of the output spectrogram (times where we will center the 
    # Gabor filter).
    time_slide = np.arange(n)
    # The vertical axis is the frequencies of the FFT, which is the same size
    # as the input signal.
    spectrogram = np.zeros((n, n), dtype=complex)
    for (i, time) in enumerate(time_slide):
        sigma = 3
        g = _get_gaussian_filter(time, time_list, sigma)
        ug = signal * g
        ugt = np.fft.fftshift(np.fft.fft(ug))
        spectrogram[:, i] = ugt
        if i == n//2 and graph_gaussian_signal == True:
            _graph_gaussian_signal(signal, g)
    # We normalize to get real values between 0 and 1.
    spectrogram = _normalize(spectrogram)
    return spectrogram


def _create_scaleogram(signal: np.ndarray, graph_wavelet_signal: bool) -> np.ndarray:
    """Creates scaleogram for signal.
    """
    # Length of the signal: 128
    n = len(signal)
    time_list = np.arange(n)
    # Scale 1 corresponds to a wavelet of width 17 (lower_bound=-8, upper_bound=8).
    # Scale n corresponds to a wavelet of width n*17.
    scale_list = np.arange(0, n) / 8 + 1
    wavelet = 'mexh'
    scaleogram = pywt.cwt(signal, scale_list, wavelet)[0]
    scaleogram = _normalize(scaleogram)

    if graph_wavelet_signal:
        signal = _normalize(signal)
        x = np.append([time_list], [time_list], axis=0)

        # Graph the narrowest wavelet together with the signal.
        [wav_narrowest, _] = pywt.ContinuousWavelet(wavelet).wavefun(
            length=n*int(scale_list[0])) 
        y = np.append([wav_narrowest], [signal], axis=0)
        utils_graph.graph_overlapping_lines(x, y, 
            ['Mexican hat wavelet', 'Signal'],
            'Time', 'Scale', 
            'Example of a signal and narrowest wavelet',
            PLOTS_FOLDER, 'sample_narrowest_wavelet.html')

        # Graph the widest wavelet together with the signal.
        # wavefun gives us the original wavelet, with a width of 17 (scale=1).
        # We want to stretch that signal by scale_list[n-1].
        # So we oversample the wavelet computation and take the n points in 
        # the middle.
        [wav_widest, _] = pywt.ContinuousWavelet(wavelet).wavefun(
            length=n*int(scale_list[n-1]))
        middle = len(wav_widest) // 2
        lower_bound = middle - n // 2
        upper_bound = lower_bound + n 
        wav_widest = wav_widest[lower_bound:upper_bound]
        y = np.append([wav_widest], [signal], axis=0)
        utils_graph.graph_overlapping_lines(x, y, 
            ['Mexican hat wavelet', 'Signal'],
            'Time', 'Scale', 
            'Example of a signal and widest wavelet',
            PLOTS_FOLDER, 'sample_widest_wavelet.html')

    return scaleogram

def _save_wavelets() -> None:
    """Saves three different kinds of mother wavelets to be used in the 
    theoretical part of the report.
    """
    n = 100
    wavelet_names = ['gaus1', 'mexh', 'morl']
    titles = ['Gaussian wavelet', 'Mexican hat wavelet', 'Morlet wavelet']
    file_names = ['gaussian.html', 'mexican_hat.html', 'morlet.html']
    for i in range(len(wavelet_names)):
        file_name = file_names[i]
        path = Path(PLOTS_FOLDER, file_name)
        if not path.exists():
            wavelet_name = wavelet_names[i]
            wavelet = pywt.ContinuousWavelet(wavelet_name)
            [wavelet_fun, x] = wavelet.wavefun(length=n)
            utils_graph.graph_2d_line(x, wavelet_fun, 
                'Time', 'Amplitude', titles[i], 
                PLOTS_FOLDER, file_name)


def scenario2(data: SignalData) -> None:
    """Creates spectrograms for each of the signals, and uses a CNN to 
    classify them.
    """
    print('Scenario 2: spectrograms + CNN')
    _save_grams(data.train_signals, SPECTROGRAMS_TRAIN_FILE_NAME, 'spectrograms')
    _save_grams(data.test_signals, SPECTROGRAMS_TEST_FILE_NAME, 'spectrograms')
    _save_gram_images(data.test_labels, data.activity_labels, 'spectrograms')

    # _tune_cnn_hyperparameters(data.train_labels, 'spectrograms')
    _test_best_cnn_hyperparameters(data.train_labels, data.test_labels, 'spectrograms')


def scenario3(data: SignalData) -> None:
    """Creates scaleograms for each of the signals, and uses a CNN to 
    classify them.
    """
    print('Scenario 3: scaleograms + CNN')
    _save_grams(data.train_signals, SCALEOGRAMS_TRAIN_FILE_NAME, 'scaleograms')
    _save_grams(data.test_signals, SCALEOGRAMS_TEST_FILE_NAME, 'scaleograms')
    _save_gram_images(data.test_labels, data.activity_labels, 'scaleograms')
    _save_wavelets()
    # _tune_cnn_hyperparameters(data.train_labels, 'scaleograms')
    _test_best_cnn_hyperparameters(data.train_labels, data.test_labels, 'scaleograms')
