import os

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial import Voronoi

COLORS = ['#f4792e', '#24624f', '#c7303b', '#457abf', '#298964', '#ffd769']


def graph_2d_markers(x: np.ndarray, y: np.ndarray, 
    xaxis_title: str, yaxis_title: str, title: str, 
    dirname: str, filename: str) -> None:
    """Creates a simple 2D plot using markers.

    Args:
        x (np.ndarray): A vector with the x values.

        y (np.ndarray): A vector with the y values.

        xaxis_title (str): The title of the x axis.

        yaxis_title (str): The title of the y axis.

        title (str): The title of the graph.

        dirname (str): The folder where we want to save the graph.

        filename (str): The name of the file where we'll save the graph.
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


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Source: https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def add_voronoi_diagram(fig: go.Figure, points: np.ndarray) -> None:
    """Adds the Voronoi diagram of the given points to the figure.

    Args:
        fig (go.Figure): The Figure object where we'll add the trace with the
        Voronoi diagram.

        points (np.ndarray): The center points of each polygon in the Voronoi
        diagram.
    """
    vor = Voronoi(points.T)
    radius = 1000
    (regions, vertices) = voronoi_finite_polygons_2d(vor, radius)
    for region in regions:
        region.append(region[0])
        polygon = vertices[region]
        x = polygon[:, 0]
        y = polygon[:, 1]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
            line_color='black', line_width=0.5, showlegend=False))


def graph_classes(training_data: np.ndarray, training_labels: np.ndarray,
    test_data: np.ndarray, test_labels: np.ndarray,
    centroids: np.ndarray, classes: np.ndarray, title: str,
    dirname: str, filename: str) -> None:
    """Plots decision boundaries between classes that we'll use to classify 
    new points.

    Args:
        training_data (np.ndarray): The training data.

        training_labels (np.ndarray): The training labels.

        test_data (np.ndarray): The test data.

        test_labels (np.ndarray): The test labels.

        centroids (np.ndarray): The means of each class.

        classes (np.ndarray): The names of each class.

        title (str): The title of the graph.

        dirname (str): The local folder where we'll save the graph.

        filename (str): The name of the file where we'll save the graph.
    """
    path = os.path.join(dirname, filename)

    (u, s, vh) = np.linalg.svd(centroids)
    u_reduced = u[:, :2]
    pca_centroids = u_reduced.T.dot(centroids)
    pca_training_data = u_reduced.T.dot(training_data)
    pca_test_data = u_reduced.T.dot(test_data)

    min_data = np.floor(np.amin(pca_training_data, axis=1) - 1)
    max_data = np.ceil(np.amax(pca_training_data, axis=1) + 1)

    fig = go.Figure()
    # Plot training and test data.
    for (i, c) in enumerate(classes):
        indices_of_training_class = np.nonzero(training_labels == c)
        x_training = pca_training_data[0, indices_of_training_class].flatten()
        y_training = pca_training_data[1, indices_of_training_class].flatten()
        indices_of_test_class = np.nonzero(test_labels == c)
        x_test = pca_test_data[0, indices_of_test_class].flatten()
        y_test = pca_test_data[1, indices_of_test_class].flatten()
        fig.add_trace(
            go.Scatter(
                x=x_training,
                y=y_training,
                mode='markers',
                marker_color=COLORS[i],
                marker_size=10,
                name=f'{c} training',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_test,
                y=y_test,
                mode='markers',
                marker_color=COLORS[i],
                marker_symbol='star-open',
                marker_line_width=2,
                marker_size=15,
                name=f'{c} test',
            )
        )
    # Plot centroids.
    fig.add_trace(
        go.Scatter(
            x=pca_centroids[0, :],
            y=pca_centroids[1, :],
            mode='markers',
            marker_line_color=COLORS,
            marker_line_width=4,
            marker_color='white',
            marker_size=20,
            showlegend=False,
        )
    )
    # Plot Voronoi diagram edges.
    add_voronoi_diagram(fig, pca_centroids)
    fig.update_layout(
        title_text=f'Classification of {title}',
        xaxis_title_text='Canonical coordinate 1',
        yaxis_title_text='Canonical coordinate 2',
        yaxis_scaleanchor = 'x',
        yaxis_scaleratio = 1,
        xaxis_range=[min_data[0], max_data[0]],
        yaxis_range=[min_data[1], max_data[1]],
    )
    pio.write_html(fig, path)
