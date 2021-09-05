import numpy as np
from torch.utils import data
from pathlib import Path
import h5py

class GramDataset(data.Dataset):

    def __init__(self, grams_path: Path, labels: np.ndarray):
        """Class constructor.
        """
        # Instance variables.
        f = h5py.File(grams_path, 'r')
        key = list(f.keys())[0]
        self.grams = f[key]
        self.labels = labels

    def __len__(self):
        """Returns length of data.
        """
        return self.grams.shape[0]

    def __getitem__(self, index):
        """Returns spectrogram or scaleogram and corresponding label at 
        specified index.
        """
        return (self.grams[index, :, :, :], self.labels[index])
