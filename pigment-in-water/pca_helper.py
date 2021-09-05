from typing import Tuple
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd


class PCAHelper:
    """Performs PCA on training data and test data."""

    @property
    def U_t(self):
        return self._U_t

    @property
    def normalization(self):
        return self._normalization

    def __init__(self, train_data: np.ndarray, test_data: np.ndarray,
        output_folder: str) -> None:
        """Initializer.
        
        Args:
            train_data (np.ndarray): The training data.
            test_data (np.ndarray): The test data.
            output_folder (str): Output folder.
        """
        self._train_data = train_data
        self._test_data = test_data
        self._output_folder = output_folder

    def get_coefficients(self, retain_threshold: float) -> Tuple[np.ndarray,
        np.ndarray, np.ndarray, np.ndarray]:
        """Projects the video data into its main spatial modes, which can be 
        found in the truncated U matrix.

        Args:
            retain_threshold (float): The fraction of the sum of singular values
            to retain.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The projected train and test data.
        """
        print('Calculating SVD.')
        self._calculate_svd(retain_threshold)

        # Note that we normalize the coefficients. This is to prevent underflow
        # and overflow issues.
        train_coefficients = self._U_t.T.dot(self._train_data)
        self._normalization = np.amax(np.abs(train_coefficients))
        train_coefficients /= self._normalization

        test_coefficients = self._U_t.T.dot(self._test_data) 
        test_coefficients /= self._normalization
        return (train_coefficients, test_coefficients)

    def _calculate_svd(self, retain_threshold: float) -> None:
        """Calculates the SVD and stores the truncated U matrix.

        Args:
            retain_threshold (float): The fraction of the sum of singular values
            to retain.
        """
        # Compute the 'economy-size' SVD of the training data.
        (U, S, _) = np.linalg.svd(self._train_data, full_matrices=False)

        # We want to pick as many modes as needed to retain the specified
        # fraction of the sum of singular values.
        self._plot_singular_values(S)
        retain_sum_threshold = np.sum(S) * retain_threshold
        retain_sum = 0
        retain_count = 0
        while retain_sum < retain_sum_threshold or retain_count < 4:
            retain_sum = retain_sum + S[retain_count]
            retain_count = retain_count + 1

        print(f'Retaining {retain_count} modes.')
        self._U_t = U[:, 0:retain_count]

    def _plot_singular_values(self, S: np.ndarray) -> None:
        """Plots a graph containing the normalized singular values of the
        training data, so that we can get visual intuition for how many modes we
        want to keep.
        """
        normalized_singular_values = S / np.sum(S)
        df = pd.DataFrame({
            'Normalized singular value': normalized_singular_values,
            'Index': range(len(normalized_singular_values))
        })
        chart = alt.Chart(df).mark_point().encode(
            x='Index',
            y=alt.Y('Normalized singular value', scale=alt.Scale(type='log', base=10)),
        )
        path = Path(self._output_folder, 'singular_values.html')
        chart.configure(background='#fff').save(str(path),
            scale_factor=2.0)
