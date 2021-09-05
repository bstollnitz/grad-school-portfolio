from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

COLORS = ['#f4792e', '#24624f', '#c7303b', '#457abf', '#298964', '#ffd769']


def graph_nn_results(train_results: List[float], validation_results: List[float], 
    title: str, y_title_test: str, dir_name: str, file_name: str) -> None:
    """
    Plots accuracy or loss of a neural network over epochs, for training and
    validation sets.
    Args:
        train_results (List[float]): List of train results.
        validation_results (List[float]): List of validation results.
        yaxis_title (str): The title of the y axis, typically Accuracy or Loss.
        title (str): The title of the graph.
        dir_name (str): The folder where we want to save the graph.
        file_name (str): The name of the file where we'll save the graph.
    """
    epochs = list(range(len(train_results)))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs, 
        y=train_results, 
        line=dict(color=COLORS[0]),
        name='Training'))
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=validation_results, 
        line=dict(color=COLORS[1]),
        name='Validation'))
    fig.update_xaxes(
        title_text='Epoch')
    fig.update_yaxes(
        title_text=y_title_test)

    fig.update_layout(
        title=title,
        yaxis_rangemode='tozero'
        )

    local_dir_path = Path('.', dir_name)
    local_dir_path.mkdir(exist_ok=True, parents=True)
    path = Path(local_dir_path, file_name)
    pio.write_html(fig, str(path))


def graph_overlapping_lines(x: np.ndarray, y: np.ndarray, legend: List[str],
    xaxis_title: str, yaxis_title: str, title: str, 
    dir_name: str, file_name: str) -> None:
    """Adds as many traces to the graph as rows in x and y. The x value
    of the trace is given by the corresponding row in x, and the y value is
    given by the corresponding row in y.
    Args:
        x (np.ndarray): A vector with the x values.
        y (np.ndarray): A vector with the y values.
        legend (List[str]): A list containing the strings to be used in the 
        legend.
        xaxis_title (str): The title of the x axis.
        yaxis_title (str): The title of the y axis.
        title (str): The title of the graph.
        dir_name (str): The folder where we want to save the graph.
        file_name (str): The name of the file where we'll save the graph.
    """
    fig = go.Figure()
    for i in range(len(x)):
        fig.add_trace(
            go.Scatter(
                x=x[i, :],
                y=y[i, :],
                mode='lines',
                line_color=COLORS[i],
                name=legend[i],
            )
        )
    fig.update_layout(
        xaxis_title_text=xaxis_title,
        yaxis_title_text=yaxis_title,
        title_text=title,
    )

    local_dir_path = Path('.', dir_name)
    local_dir_path.mkdir(exist_ok=True, parents=True)
    path = Path(local_dir_path, file_name)
    pio.write_html(fig, str(path))


def graph_2d_line(x: np.ndarray, y: np.ndarray, 
    xaxis_title: str, yaxis_title: str, title: str, 
    dir_name: str, file_name: str) -> None:
    """Creates a simple 2D plot using lines.
    Args:
        x (np.ndarray): A vector with the x values.
        y (np.ndarray): A vector with the y values.
        xaxis_title (str): The title of the x axis.
        yaxis_title (str): The title of the y axis.
        title (str): The title of the graph.
        dir_name (str): The folder where we want to save the graph.
        file_name (str): The name of the file where we'll save the graph.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            marker_color=COLORS[0],
            marker_size=10,
        )
    )
    fig.update_layout(
        title_text=title,
        xaxis_title_text=xaxis_title,
        yaxis_title_text=yaxis_title,
    )

    local_dir_path = Path('.', dir_name)
    local_dir_path.mkdir(exist_ok=True, parents=True)
    path = Path(local_dir_path, file_name)
    pio.write_html(fig, str(path))