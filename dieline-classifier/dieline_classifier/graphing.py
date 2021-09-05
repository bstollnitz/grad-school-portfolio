from typing import Dict, List, Tuple

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff

from vector_dataset import VectorDataset

# train_color = "DeepSkyBlue"
# test_color = "Orange"
train_color = "#4b2e83"
test_color = "#F00"

def plot_category_histograms(data: VectorDataset, filename: str) -> None:
    """
    Plots histograms of the category labels in the training and test data.
    """

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Training", "Test"))

    train_hist = np.histogram(data.train_y, bins=range(len(data.categories) + 1))
    test_hist = np.histogram(data.test_y, bins=range(len(data.categories) + 1))
    fig.add_trace(go.Bar(x = data.categories, y = train_hist[0], name="Training",
        marker_color=train_color, showlegend=False), col=1, row=1)
    fig.add_trace(go.Bar(x = data.categories, y = test_hist[0], name="Test",
        marker_color=test_color, showlegend=False), col=1, row=2)

    fig.update_layout(title="Distribution of category labels")
    pio.write_html(fig, filename)

def plot_nn(history: dict, filename: str) -> None:
    """
    Plots accuracy and loss of a neural network over epochs, for the training
    and test sets.
    """

    train_loss = history["loss"]
    train_acc = history["accuracy"]
    val_loss = history["val_loss"]
    val_acc = history["val_accuracy"]
    epochs = list(range(len(train_loss)))

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss", "Accuracy"))

    fig.add_trace(go.Scatter(
        x=epochs, 
        y=train_loss, 
        line=dict(color=train_color),
        name="Training"), 
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=val_loss, 
        line=dict(color=test_color),
        name="Test"), 
        row=1, col=1)
    fig.update_xaxes(
        title_text="Epoch", 
        row=1, 
        col=1)
    fig.update_yaxes(
        title_text="Loss", 
        row=1, 
        col=1)

    fig.add_trace(go.Scatter(
        x=epochs, 
        y=train_acc, 
        line=dict(color=train_color),
        name="Training", 
        showlegend=False), 
        row=2, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=val_acc, 
        line=dict(color=test_color),
        name="Test", 
        showlegend=False), 
        row=2, col=1)
    fig.update_xaxes(
        title_text="Epoch", 
        row=2, 
        col=1)
    fig.update_yaxes(
        title_text="Accuracy", 
        row=2, 
        col=1)

    fig.update_layout(title="Best neural network results")
    pio.write_html(fig, filename)

def get_correlations(param_distributions: dict, scores: dict, 
    params: List[dict]) -> Tuple[List[str], List[float]]:
    """
    Creates a list indicating how much each hyperparameter is correlated to 
    the validation accuracy scores.
    """

    param_count = len(param_distributions)
    score_count = len(scores)
    param_names = list(param_distributions)
    
    # We create matrix m, which contains a row per hyperparameter, and a column
    # per run. Each cell of matrix m contains the value of a particular
    # hyperparameter for a particular run. 
    # The matrix values must be numerical, so for string-valued parameters, we
    # add a row for each possible value, using 1 when the run's parameter value
    # equals that value. For example, the row labeled "optimizer=SGD" contains
    # a 1 for each run that used SGD.
    m = np.empty((0, score_count))
    used_param_names = []
    for i in range(param_count):
        param_name = param_names[i]
        possible_values = param_distributions[param_name]
        if len(possible_values) == 1:
            # There's only one possible value, so we can skip this parameter.
            continue
        experiment_values = [param[param_name] for param in params]
        if isinstance(possible_values[0], str):
            # The parameter values are strings. Add a row for each possible
            # value, converting values to their one-hot encoding.
            for j in range(len(possible_values)):
                row = [possible_values.index(value) == j for value in
                    experiment_values]
                m = np.append(m, [row], axis=0)
                used_param_names.append(f"{param_name}={possible_values[j]}")
        else:
            # The parameter values are numeric. Use them without change.
            m = np.append(m, [experiment_values], axis=0)
            used_param_names.append(param_name)

    # We add an extra row to m containing the scores.
    m = np.append(m, [scores], axis=0)

    # The correlation matrix contains the correlation between the rows of 
    # matrix m. We're only interested in its bottom row (or rightmost column,
    # because this matrix is symmetric) because we want the correlation between
    # scores and everything else.
    correlation = np.corrcoef(m)
    correlation_values = correlation[-1:][0].tolist()

    # We exclude the correlation of scores with itself, which is the last value.
    correlation_values = correlation_values[:-1]

    # We order the information from the largest to smallest absolute value.
    abs_values = [abs(value) for value in correlation_values]
    sorted_indices = np.flip(np.argsort(abs_values))
    used_param_names = [used_param_names[i] for i in sorted_indices]
    correlation_values = [correlation_values[i] for i in sorted_indices]

    return (used_param_names, correlation_values)

def plot_correlations(param_distributions: dict, scores: dict, 
    params: List[dict], filename: str) -> None:
    """
    Plots how much each hyperparameter is correlated to the validation
    accuracy scores.
    """

    (param_names, correlations) = get_correlations(param_distributions, scores, params)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y = param_names, 
        x = correlations,
        marker_color=train_color, 
        orientation="h",
        ))
    fig.update_xaxes(
        zeroline=True,
        zerolinecolor=train_color,
        zerolinewidth=2,
    )
    fig.update_layout(
        title_text="Correlation between accuracy and hyperparameters",
        xaxis_title_text="Correlation",
    )
    pio.write_html(fig, filename)


def plot_accuracy(scores: dict, filename: str) -> None:
    """
    Plots the distribution of validation accuracy of the neural network 
    for all hyperparameter combination experiments.
    """

    bins = 250
    (hist, _) = np.histogram(scores, bins=bins, range=(0, 1))
    x = np.linspace(0, 1, bins)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x,
        y = hist,
        marker_color=train_color,
    ))
    fig.update_layout(
        title_text="Distribution of validation accuracy over experiments",
        xaxis_title_text="Validation accuracy",
        yaxis_title_text="Number of experiments",
    )
    pio.write_html(fig, filename)