import collections
from typing import List, Tuple
from pathlib import Path

import altair as alt
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
from _plotly_future_ import v4_subplots  # For titles on subplots.
import plotly.io as pio
import plotly.plotly as py
import sklearn.linear_model as sk
import sklearn.preprocessing as pre
from plotly import offline, tools

from cached_dataset import CachedDataset

FITTER = 'Lasso' # Lasso, Ridge, or LinearRegression
ALPHA = 0.0001 # used in Lasso

class ModelExplorer:
    """Uses model discovery to find a PDE that predicts video data."""

    def __init__(
        self,
        train_dataset: CachedDataset,
        test_dataset: CachedDataset,
        output_folder: str,
    ) -> None:
        """Initializes the class.
        
        Args:
            train_dataset (CachedDataset): The training dataset.
            test_dataset (CachedDataset): The test dataset.
            output_folder (str): Output folder.
        """
        self._train = train_dataset
        self._test = test_dataset
        self._output_folder = output_folder

    def explore(self) -> None:
        """Performs model discovery."""
        self._graph_all(self._train)
        self._graph_cross_sections('test', 'u_t', self._test.u_t[0])

        # Build a library, including x, and y derivatives and second derivatives
        print('Building library.')
        (A_train, self._term_names) = self._build_library(self._train)
        (A_test, self._term_names) = self._build_library(self._test)

        print(f'cond(A_train) = {np.linalg.cond(A_train)}')
        print(f'cond(A_test) = {np.linalg.cond(A_test)}')

        # Flatten u_t.
        b = self._train.u_t.reshape(-1, 1)

        # Learn coefficients for a first-order PDE.
        print('Fitting model.')
        x = self._fit(A_train, b)
        print(f'  u_t = {self._get_pde_string(x)}')

        # Graph coefficients for the PDE.
        self._graph_coefficients(x)

        # Compare the actual and predicted derivatives of train and test for
        # various slices through a pixel in the middle of the grid.
        u_t_train_predicted = self._fitter.predict(A_train).reshape(self._train.u_t.shape)
        u_t_test_predicted = self._fitter.predict(A_test).reshape(self._test.u_t.shape)
        self._graph_cross_sections('train prediction', 'u_t', u_t_train_predicted[0])
        self._graph_cross_sections('test prediction', 'u_t', u_t_test_predicted[0])

    def _build_library(self, dataset: CachedDataset) -> np.ndarray:
        """Builds a library of potential dynamical system terms. 
        
        Args:
            u (np.ndarray): The pixel values.
            u_x (np.ndarray): The x derivative of the pixel values.
            u_xx (np.ndarray): The second x derivative of the pixel values.
            u_y (np.ndarray): The y derivative of the pixel values.
            u_yy (np.ndarray): The second y derivative of the pixel values.
        
        Returns:
            np.ndarray: The library of potential terms.
        """
        # Generate x, y, and t variables.
        video_count = dataset.video_count
        frame_count = dataset.frame_count
        height = dataset.height
        width = dataset.width
        x = np.arange(width) - width/2
        y = np.arange(height) - height/2
        t = np.arange(frame_count)
        (y, t, x) = np.meshgrid(y, t, x)
        x = np.tile(x, (video_count, 1, 1, 1))
        y = np.tile(y, (video_count, 1, 1, 1))
        t = np.tile(t, (video_count, 1, 1, 1))

        # Create a library of named terms.
        n = dataset.u.size
        terms = (
            ('1', np.ones((n, 1))),
            ('u', dataset.u),
            ('u_x', dataset.u_x),
            ('u_xx', dataset.u_xx),
            ('u_y', dataset.u_y),
            ('u_yy', dataset.u_yy),
            ('x * u_x', x * dataset.u_x),
            ('y * u_y', y * dataset.u_y),
        )

        # Get the name of each term.
        term_names = list(map(lambda term: term[0], terms))

        # Flatten the values of each term into a column vector.
        term_values = list(map(lambda term: term[1].reshape(n, 1), terms))

        # Build the 2D matrix A by stacking the column vectors side by side.
        A = np.hstack(term_values)
        return (A, term_names)

    def _fit(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Finds coefficients that fit the given library of terms A to the
        vector b.
        
        Args:
            A (np.ndarray): A library of terms.
            b (np.ndarray): A vector.
        
        Returns:
            np.ndarray: A vector of coefficients.
        """
        # Initialize fitter.
        if FITTER is 'Lasso':
            print(f'  Lasso(alpha={ALPHA})')
            self._fitter = sk.Lasso(alpha=ALPHA, fit_intercept=False)
        elif FITTER is 'Ridge':
            print('  Ridge')
            self._fitter = sk.Ridge(fit_intercept=False)
        else:
            print('  LinearRegression')
            self._fitter = sk.LinearRegression(fit_intercept=False)

        # Solve for coefficients x.
        self._fitter.fit(A, b)
        score = self._fitter.score(A, b)
        print(f'  Score: {score}')
        return self._fitter.coef_.flatten()

    def _get_pde_string(self, coefficients: np.ndarray) -> str:
        """Returns the PDE formula.
        
        Args:
            coefficients (np.ndarray): Coefficients of the terms in the PDE.
        
        Returns:
            str: The PDE formula as a string.
        """
        pde = ''
        coefficients = np.round(coefficients, decimals=8)
        has_term = False
        for (i, term_name) in enumerate(self._term_names):
            c = coefficients[i]
            if c != 0:
                if has_term:
                    if c < 0:
                        pde += ' - '
                        c = -c
                    else:
                        pde += ' + '
                pde += str(c) + ' ' + (term_name if term_name != '1' else '')
                has_term = True
        if not has_term:
            pde = '0'
        return pde

    def _graph_coefficients(self, coefficients: np.ndarray) -> None:
        """Graphs the PDE coefficients we learned.
        
        Args:
            coefficients (np.ndarray): Coefficients we want to graph. Size m, 
            where m is the number of terms we consider.
        """
        df = pd.DataFrame({
            'Coefficient': coefficients,
            'Term': self._term_names
        })
        chart = alt.Chart(df).mark_bar().encode(
            x='Term',
            y='Coefficient',
        ).properties(width=300)
        path = Path(self._output_folder, 'md_coefficients.html')
        chart.configure(background='#fff').save(
            str(path),
            scale_factor=2.0
        )

    def _graph_all(self, dataset: CachedDataset) -> None:
        """Creates heatmaps of u and its derivatives for the first video in a
        dataset.
        
        Args:
            dataset (CachedDataset): A dataset.
        """
        video_index = 0
        self._graph_cross_sections(dataset.train_or_test, 'u', dataset.u[video_index])
        self._graph_cross_sections(dataset.train_or_test, 'u_t', dataset.u_t[video_index])
        self._graph_cross_sections(dataset.train_or_test, 'u_x', dataset.u_x[video_index])
        self._graph_cross_sections(dataset.train_or_test, 'u_xx', dataset.u_xx[video_index])
        self._graph_cross_sections(dataset.train_or_test, 'u_y', dataset.u_y[video_index])
        self._graph_cross_sections(dataset.train_or_test, 'u_yy', dataset.u_yy[video_index])

    def _graph_cross_sections(self, train_or_test: str, label: str, u: np.ndarray) -> None:
        """Creates three heatmaps showing cross sections of the given data.
        
        Args:
            train_or_test (str): 'train' or 'test.
            label (str): The name of the independent variable.
            u (np.ndarray): The data.
        """
        # Choose indices for cross sections in t, y, and x.
        i = int(u.shape[0] / 2)
        j = int(u.shape[1] / 3)
        k = int(u.shape[2] / 3)

        # Extract cross sections.
        section1 = u[i, :, :]
        section2 = u[:, j, :]
        section3 = u[:, :, k]

        # Create the three graphs.
        graph1 = self._graph_cross_section(section1, showscale=True)
        graph2 = self._graph_cross_section(section2)
        graph3 = self._graph_cross_section(section3)

        # Make a figure with three subplots.
        rows = 1
        cols = 3
        subplot_titles = (f't = {i}', f'y = {j}', f'x = {k}')
        fig = tools.make_subplots(rows=rows, cols=cols,
            subplot_titles=subplot_titles)
        fig.append_trace(graph1, 1, 1)
        fig.append_trace(graph2, 1, 2)
        fig.append_trace(graph3, 1, 3)

        # Add title and axis labels.
        fixed_label = self._fix_subscript(label)
        title = f'{train_or_test} {fixed_label}(x, y, t)'
        fig.layout.update(title=title, width=2400, height=800)
        self._set_axis_labels(fig, '', 'x', 'y')
        self._set_axis_labels(fig, '2', 'x', 't')
        self._set_axis_labels(fig, '3', 'y', 't')

        # Save the plot in a file.
        train_or_test = train_or_test.replace(' ', '_')
        pio.write_image(fig, f'{self._output_folder}/md_{label}_{train_or_test}.png')

    def _fix_subscript(self, s: str) -> str:
        """Replaces a LaTeX-style subscript with an HTML subscript.
        
        Args:
            s (str): A string.
        
        Returns:
            str: A modified string.
        """
        if '_' in s:
            return s.replace('_', '<sub>') + '</sub>'
        return s

    def _set_axis_labels(self, fig, axis_number: str, xlabel: str, ylabel: str) -> None:
        """Sets the X and Y axis labels of a plotly graph.
        
        Args:
            fig (plotly.Figure): A figure.
            axis_numer (str): The number associated with the axis.
            xlabel (str): The label for the X axis.
            ylabel (str): The label for the Y axis.
        """
        fig.layout[f'xaxis{axis_number}'].update({ 'title': { 'text': xlabel } })
        fig.layout[f'yaxis{axis_number}'].update({ 'title': { 'text': ylabel } })

    def _graph_cross_section(self, u: np.ndarray, showscale: bool = False) -> dict:
        """Creates a heatmap for the given data.
        
        Args:
            u (np.ndarray): The data.
            showscale (bool, optional): Whether to show the color scale.
            Defaults to False.
        
        Returns:
            dict: A dictionary for a plotly heatmap.
        """
        x = np.arange(u.shape[1])
        y = np.arange(u.shape[0])
        return dict(type='heatmap', x=x, y=y, z=u, colorscale='RdBu',
            showscale=showscale)
