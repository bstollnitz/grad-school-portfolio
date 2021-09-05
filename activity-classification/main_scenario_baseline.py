import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

import utils_graph
import utils_io
import utils_nn
from feed_forward import FeedForward
from hyperparameters import Hyperparameters
from signal_data import SignalData
from signal_dataset import SignalDataset

PLOTS_FOLDER = 'plots'
USE_CUDA = torch.cuda.is_available()


def _train_ff_network(hyperparameter_dict: dict, 
    data: SignalData) -> Tuple[FeedForward, List, List, List, List]:
    """Trains a feed-forward network using the specified hyperparameters.
    """
    # Ensure reproducibility by giving PyTorch the same seed every time we train.
    torch.manual_seed(1)

    # Print hyperparameters.
    print(f'Hyperparameters: {hyperparameter_dict}')

    # Get hyperparameters.
    learning_rate = hyperparameter_dict['learning_rate']
    batch_size = hyperparameter_dict['batch_size']
    optimizer_str = hyperparameter_dict['optimizer']

    # There are 6 labels, and Pytorch expects them to go from 0 to 5.
    full_train_labels = data.train_labels - 1

    # Get generators.
    signal_dataset = SignalDataset(data.train_signals, full_train_labels)
    (training_generator, validation_generator) = utils_nn.get_trainval_generators(
        signal_dataset, batch_size, num_workers=0, training_fraction=0.8)

    # Crete feed forward network.
    input_size = data.num_timesteps * data.num_components
    feed_forward = FeedForward(input_size, input_size, data.num_activity_labels)
    print(feed_forward)

    # Parameters should be moved to GPU before constructing the optimizer.
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    feed_forward = feed_forward.to(device)

    # Get optimizer.
    optimizer = None
    if optimizer_str == 'adam':
        optimizer = torch.optim.Adam(feed_forward.parameters(), lr=learning_rate)
    elif optimizer_str == 'sgd':
        optimizer = torch.optim.SGD(feed_forward.parameters(), lr=learning_rate)
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
        (training_accuracy, training_loss) = utils_nn.fit(feed_forward, 
            training_generator, optimizer, USE_CUDA)
        training_accuracy_list.append(training_accuracy)
        training_loss_list.append(training_loss)

        # Validation data.
        (validation_accuracy, validation_loss) = utils_nn.evaluate(feed_forward, 
            validation_generator, 'Validation', USE_CUDA)
        validation_accuracy_list.append(validation_accuracy)
        validation_loss_list.append(validation_loss)

    return (feed_forward, training_accuracy_list, training_loss_list, 
        validation_accuracy_list, validation_loss_list)


def _get_ff_hyperparameters() -> Hyperparameters:
    """Returns hyperparameters used to tune the feed-forward network.
    """
    # First pass:
    hyperparameter_values = Hyperparameters({
        'learning_rate': [0.1, 0.01, 0.001],
        'batch_size': [32, 64, 128],
        'optimizer': ['adam', 'sgd']
        })
    # Best: 
    # optimizer: sgd, batch size: 64, learning rate: 0.1

    # Second pass:
    hyperparameter_values = Hyperparameters({
        'learning_rate': [0.05, 0.1, 0.2],
        'batch_size': [16, 32, 64],
        'optimizer': ['sgd']
        })

    # Best:
    # optimizer: sgd, batch size: 16, learning rate: 0.1

    return hyperparameter_values


def _tune_ff_hyperparameters(data: SignalData) -> None:
    """Classifies temporal signals using a feed-forward network.
    """
    print('  Tuning hyperparameters.')
    start_time = time.time()

    # Hyperparameters to tune.
    hyperparameter_values = _get_ff_hyperparameters()
    hyperparameter_combinations = hyperparameter_values.sample_combinations()

    # Create Tensorboard writer.
    with SummaryWriter(f'runs/signals', filename_suffix='') as writer:
        # Hyperparameter loop.
        for hyperparameter_dict in hyperparameter_combinations:
            (_, _, _, validation_accuracy_list, _) = _train_ff_network(
                hyperparameter_dict, data)

            writer.add_hparams(hyperparameter_dict,
                {'hparam/signals/validation_accuracy': validation_accuracy_list[-1]})

    utils_io.print_elapsed_time(start_time, time.time())


def _test_ff_network(feed_forward: FeedForward, signal_data: SignalData, 
    hyperparameter_dict: dict) -> Tuple[float, float]:
    """Returns accuracy and loss of specified network for specified test data 
    and specified hyperparameters.
    """
    # There are 6 labels, and Pytorch expects them to go from 0 to 5.
    test_labels = signal_data.test_labels - 1

    # Get test generator.
    batch_size = hyperparameter_dict['batch_size']
    test_data = SignalDataset(signal_data.test_signals, test_labels)
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    test_generator = data.DataLoader(test_data, **params)

    (test_avg_accuracy, test_avg_loss) = utils_nn.evaluate(feed_forward, 
        test_generator, 'Test', USE_CUDA)

    return (test_avg_accuracy, test_avg_loss)


def _test_best_ff_hyperparameters(data: SignalDataset) -> None:
    """Use network with best hyperparameters to predict labels for test data.
    Produces accuracy and loss graphs for training and validation data, as 
    well as accuracy and loss values for test data.
    """
    hyperparameter_dict = {
        'learning_rate': 0.1,
        'batch_size': 16,
        'optimizer': 'sgd',
        }

    (feed_forward, training_accuracy_list, 
        training_loss_list, 
        validation_accuracy_list, 
        validation_loss_list) = _train_ff_network(hyperparameter_dict, 
        data)

    utils_graph.graph_nn_results(training_accuracy_list, validation_accuracy_list, 
        f'Training and validation accuracy of classification of temporal signals', 
        'Accuracy', PLOTS_FOLDER, f'signals_accuracy.html')

    utils_graph.graph_nn_results(training_loss_list, validation_loss_list, 
        f'Training and validation loss of classification of temporal signals', 
        'Loss', PLOTS_FOLDER, f'signals_loss.html')

    _test_ff_network(feed_forward, data, hyperparameter_dict)

    with SummaryWriter(f'runs/signals', filename_suffix='') as writer:
        num_epochs_train_val = len(training_accuracy_list)
        for i in range(num_epochs_train_val):
            writer.add_scalars(f'signals/accuracy', {
                'training': training_accuracy_list[i],
                'validation': validation_accuracy_list[i]
                }, i)
            writer.add_scalars(f'signals/loss', {
                'training': training_loss_list[i],
                'validation': validation_loss_list[i]
                }, i)

    # Test accuracy: 87.25%
    # Test loss: 0.45


def scenario1(data: SignalData) -> None:
    """Uses a simple feed forward network to classify the raw signal.
    """
    print('Scenario 1: feed forward network on raw signal')

    # _tune_ff_hyperparameters(data)
    _test_best_ff_hyperparameters(data)