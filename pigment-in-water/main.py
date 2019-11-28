import os
import numpy as np
from tensorflow import set_random_seed

from dataset_cacher import DatasetCacher
from cached_dataset import CachedDataset
from ffn_analyzer import FFNAnalyzer
from model_explorer import ModelExplorer


DERIVATIVE_METHOD = "polynomial" # h2, h4, robust, savitzky, polynomial


def _generate_cached_data() -> None:
    """Reads video files and generates preprocessed data files."""
    # Preprocess data for the feed-forward neural network.
    ffn_cacher = DatasetCacher("ffn", False, None)
    ffn_cacher.compute_and_save()

    # Preprocess data for model discovery.
    md_cacher = DatasetCacher("md", True, DERIVATIVE_METHOD)
    md_cacher.compute_and_save()


def main() -> None:
    """Main program."""
    np.random.seed(1)
    set_random_seed(1)

    try:
        os.mkdir("Images")
    except OSError:
        pass

    print("Feed-forward network")
    print("====================")
    ffn_train_dataset = CachedDataset("ffn", "train")
    ffn_test_dataset = CachedDataset("ffn", "test")
    ffn = FFNAnalyzer(ffn_train_dataset, ffn_test_dataset)
    ffn.train_network()
    ffn.predict_train()
    ffn.predict_test()

    print("Model discovery")
    print("===============")
    md_train_dataset = CachedDataset("md", "train")
    md_test_dataset = CachedDataset("md", "test")
    md = ModelExplorer(md_train_dataset, md_test_dataset)
    md.explore()


if __name__ == '__main__':
    main()
