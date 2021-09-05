import random

import numpy as np
import torch

from main_scenario_baseline import scenario1
from main_scenario_grams import scenario2, scenario3
from signal_data import SignalData

DATA_FOLDER = 'data'
S3_URL = 'https://bea-portfolio.s3-us-west-2.amazonaws.com/activity-classification/'
S3_FILENAME = 'activity-dataset.zip'


def ensure_reproducibility() -> None:
    """Ensures reproducibility of results.
    """
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    """Main program.
    """
    ensure_reproducibility()
    data = SignalData(DATA_FOLDER, S3_URL, S3_FILENAME)
    # Scenario 1: 2-layer feed forward network on raw signal.
    # scenario1(data)
    # Scenario 2: classify using CNN on spectrograms.
    # scenario2(data)
    # Scenario 3: classify using CNN on scaleograms.
    scenario3(data)


if __name__ == '__main__':
    main()
