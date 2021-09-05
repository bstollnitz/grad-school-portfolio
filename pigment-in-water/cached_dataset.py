from pathlib import Path

import numpy as np


class CachedDataset:
    """Loads a dataset from a file that was saved by DatasetCacher."""

    @property
    def train_or_test(self) -> str:
        return self._train_or_test

    @property
    def video_count(self) -> int:
        return self._u.shape[0]

    @property
    def frame_count(self) -> int:
        return self._u.shape[1]

    @property
    def height(self) -> int:
        return self._u.shape[2]

    @property
    def width(self) -> int:
        return self._u.shape[3]

    @property
    def u(self) -> int:
        return self._u

    @property
    def u_t(self) -> int:
        return self._u_t

    @property
    def u_x(self) -> int:
        return self._u_x

    @property
    def u_xx(self) -> int:
        return self._u_xx

    @property
    def u_y(self) -> int:
        return self._u_y

    @property
    def u_yy(self) -> int:
        return self._u_yy

    def __init__(self, name: str, train_or_test: str, data_folder: str) -> None:
        """Initializer.
        
        Args:
            name (str): 'ffn' or 'md'.
            train_or_test (str): 'train' or 'test'.
        """
        self._train_or_test = train_or_test

        # Read derivatives from a data file if it exists already.
        file_path = Path(data_folder, 
            f'dataset-{name}-{train_or_test}.npz')
        print(f'  Reading data from {file_path}.')
        with np.load(file_path) as data:
            self._u = data['u']
            if 'u_t' in data:
                self._u_t = data['u_t']
                self._u_x = data['u_x']
                self._u_xx = data['u_xx']
                self._u_y = data['u_y']
                self._u_yy = data['u_yy']
