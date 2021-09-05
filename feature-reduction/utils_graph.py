import os
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


COLORS = ['#f4792e', '#24324f', '#ffd769', '#c7303b', '#457abf', '#298964']


def graph_overlapping_lines(x: np.ndarray, y: np.ndarray, legend: List[str],
    xaxis_title: str, yaxis_title: str, title: str, 
    dirname: str, filename: str) -> None:
    """Adds as many traces to the graph as rows in x and y. The x value
    of the trace is given by the corresponding row in x, and the y value is
    given by the corresponding row in y.
    """
    path = os.path.join(dirname, filename)

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
    pio.write_html(fig, path)


def graph_2d_markers(x: np.ndarray, y: np.ndarray, 
    xaxis_title: str, yaxis_title: str, title: str, 
    dirname: str, filename: str) -> None:
    """Creates a simple 2D plot using markers.
    """
    path = os.path.join(dirname, filename)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker_color=COLORS[0],
            marker_size=10,
        )
    )
    fig.update_layout(
        title_text=title,
        xaxis_title_text=xaxis_title,
        yaxis_title_text=yaxis_title,
    )
    pio.write_html(fig, path)
