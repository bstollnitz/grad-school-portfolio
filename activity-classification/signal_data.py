import os
from pathlib import Path
from torch.utils.data import Dataset

import utils_io
import numpy as np

TRAIN_DATA_DIR = 'UCI HAR Dataset/train/Inertial Signals'
TRAIN_LABELS = 'UCI HAR Dataset/train/y_train.txt'
TEST_DATA_DIR = 'UCI HAR Dataset/test/Inertial Signals'
TEST_LABELS = 'UCI HAR Dataset/test/y_test.txt'
ACTIVITY_LABELS = 'UCI HAR Dataset/activity_labels.txt'

class SignalData():

    # 6
    @property
    def num_activity_labels(self):
        return len(self.activity_labels)

    # 7352
    @property
    def num_train(self):
        return len(self.train_labels)

    # 2947
    @property
    def num_test(self):
        return len(self.test_labels)

    # 9
    @property
    def num_components(self):
        return self.train_signals.shape[2]

    # 128
    @property
    def num_timesteps(self):
        return self.train_signals.shape[1]

    def __init__(self, local_dir_name: str, remote_dir_url: str, 
        file_name: str):
        """Class constructor.
        """
        # Download zip file and extract files.
        if not Path(local_dir_name, ACTIVITY_LABELS).exists():
            utils_io.download_remote_data_file(local_dir_name, remote_dir_url, 
                file_name)
            utils_io.extract_all_zips(local_dir_name)

        # Instance variables.
        # 6
        self.activity_labels = self._read_activity_labels(local_dir_name, ACTIVITY_LABELS)
        # 7352
        self.train_labels = self._read_labels(local_dir_name, TRAIN_LABELS)
        # 2947
        self.test_labels = self._read_labels(local_dir_name, TEST_LABELS)
        # 7352 x 9 x 128
        self.train_signals = self._read_signals(local_dir_name, TRAIN_DATA_DIR)
        # 2947 x 9 x 128
        self.test_signals = self._read_signals(local_dir_name, TEST_DATA_DIR)

    def _read_activity_labels(self, local_dir_name: str, relative_path: str) -> dict:
        """Reads activity labels into a dictionary.
        """
        activity_labels_path = Path(local_dir_name, ACTIVITY_LABELS)
        activity_labels = {}
        with activity_labels_path.open() as f:
            activity_labels_list = f.readlines()
        for label in activity_labels_list:
            (key, value) = label.split(' ')
            key = int(key)
            activity_labels[key] = value.rstrip()
        return activity_labels

    def _read_labels(self, local_dir_name: str, relative_path: str) -> np.ndarray:
        """Reads labels into an ndarray.
        """
        labels_file_path = Path(local_dir_name, relative_path)
        with labels_file_path.open() as f:
            labels_list = f.readlines()
        return np.array([int(label.rstrip()) for label in labels_list])

    def _read_signals(self, local_dir_name: str, relative_path: str) -> np.ndarray:
        """Reads signals into an ndarray.
        """
        signals_dir_path = Path(local_dir_name, relative_path)
        signals_list = []
        for file_path in signals_dir_path.iterdir(): # 9
            rows = []
            with file_path.open() as f:
                lines = f.readlines()
            for line in lines: # 2947 or 7352
                row = [float(reading) for reading in line.split()] # 128
                rows.append(row)
            signals_list.append(rows)
        # signals is now 9 x samples x 128
        signals = np.array(signals_list)
        # we want signals to have dims samples x 9 x 128
        signals = np.transpose(signals, (1, 0, 2))
        return signals

