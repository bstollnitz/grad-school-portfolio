import os
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


COLORS = ['#f4792e', '#24324f', '#ffd769', '#c7303b', '#457abf', '#298964']


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
        yaxis_scaleanchor = 'x',
        yaxis_scaleratio = 1,
    )
    pio.write_html(fig, path)
