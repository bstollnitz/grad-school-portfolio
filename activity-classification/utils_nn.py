
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


def get_trainval_generators(full_training_data: torch.utils.data.Dataset,
    batch_size: int, num_workers: int, training_fraction: float) -> Tuple[data.DataLoader, 
    data.DataLoader]:
    """Splits the training images and labels into training and validation sets,
    and returns generators for those.
    """
    full_training_len = len(full_training_data)
    training_len = int(full_training_len * training_fraction)
    validation_len = full_training_len - training_len
    (training_data, validation_data) = data.random_split(full_training_data, 
        [training_len, validation_len])

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_workers}
    training_generator = data.DataLoader(training_data, **params)
    validation_generator = data.DataLoader(validation_data, **params)

    return (training_generator, validation_generator)


def _calculate_accuracy(output: torch.Tensor, actual_labels: torch.Tensor) -> float:
    """Calculates accuracy of multiclass prediction.

    Args:
        output (torch.Tensor): Output predictions from neural network.
        actual_labels (torch.Tensor): Actual labels.
    """
    predicted_labels = torch.argmax(output, dim=1)
    num_correct = (predicted_labels == actual_labels).sum()
    num_train = len(actual_labels)
    accuracy = (float(num_correct) / float(num_train)) * 100
    return accuracy


def fit(nn: torch.nn.Module, generator: data.DataLoader,
    optimizer: torch.optim.Optimizer, use_cuda: bool) -> Tuple[float, float]:
    """Trains specified neural network on data returned by specified
    generator.
    Returns the average accuracy and loss across all minibatches.
    Used in training data.

    Args:
        nn (torch.nn.Module): Trained neural network we'll train.
        generator (data.DataLoader): DataLoader used to generate data we'll
        use to get the data to train.
        optimizer (torch.optim.Optimizer): Optimizer we'll use in training.
        use_cuda (bool): Whether to use cuda.
    """
    # Use CUDA.
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    accuracy_list = []
    loss_list = []
    for batch_data, batch_labels in tqdm(generator):
        # Transfer input data and labels to GPU.
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.long().to(device) # Loss function expects long.
        # Zero-out the optimizer's gradients.
        optimizer.zero_grad()
        # Forward pass, backward pass, optimize.
        batch_output = nn(batch_data.float())
        batch_loss = nn.loss_function(batch_output, batch_labels)
        batch_loss.backward()
        optimizer.step()
        # Gather statistics.
        batch_accuracy = _calculate_accuracy(batch_output, batch_labels)
        accuracy_list.append(batch_accuracy)
        loss_list.append(batch_loss.item())

    avg_accuracy = np.average(accuracy_list)
    avg_loss = np.average(loss_list)
    print(f'  Training accuracy: {avg_accuracy:0.2f}' + 
        f'  Training loss: {avg_loss:0.2f}')

    return (avg_accuracy, avg_loss)


def evaluate(nn: torch.nn.Module, generator: data.DataLoader,
    data_type: str, use_cuda: bool) -> Tuple[float, float]:
    """Uses specified neural network to make predictions on data returned by 
    specified generator.
    Returns the average accuracy and loss across all minibatches.
    Used in validation and test sets.

    Args:
        nn (torch.nn.Module): Trained neural network we'll use to make 
        predictions.
        generator (data.DataLoader): DataLoader used to generate data we'll
        use to make predictions on.
        data_type (str): 'Validation' or 'Test'.
        use_cuda (bool): Whether to use cuda.
    """
    # Use CUDA.
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    accuracy_list = []
    loss_list = []
    for batch_data, batch_labels in tqdm(generator):
        # Transfer input data and labels to GPU.
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.long().to(device) # Loss function expects long.
        # Predict.
        batch_output = nn(batch_data.float())
        batch_loss = nn.loss_function(batch_output, batch_labels)
        # Gather statistics.
        batch_accuracy = _calculate_accuracy(batch_output, batch_labels)
        accuracy_list.append(batch_accuracy)
        loss_list.append(batch_loss.item())

    avg_accuracy = np.average(accuracy_list)
    avg_loss = np.average(loss_list)
    print(f'  {data_type} accuracy: {avg_accuracy:0.2f}' + 
        f'  {data_type} loss: {avg_loss:0.2f}')

    return (avg_accuracy, avg_loss)