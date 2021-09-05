import numpy as np
from torch.utils import data

class SignalDataset(data.Dataset):

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        """Class constructor.
        """
        # Instance variables.
        self.signals = signals
        self.labels = labels

    def __len__(self):
        """Returns length of data.
        """
        return self.signals.shape[0]

    def __getitem__(self, index):
        """Returns signals and corresponding label at 
        specified index.
        """
        return (self.signals[index, :, :].flatten(), self.labels[index])
