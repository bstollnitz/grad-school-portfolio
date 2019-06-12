import os
import numpy as np
from tensorflow import set_random_seed

from dataset_cacher import DatasetCacher
from cached_dataset import CachedDataset
from ffn_analyzer import FFNAnalyzer
from model_explorer import ModelExplorer

DERIVATIVE_METHOD = "polynomial" # h2, h4, robust, savitzky, polynomial

def main() -> None:
    """Main program."""
    np.random.seed(1)
    set_random_seed(1)

    try:
        os.mkdir("Images")
    except OSError:
        pass

    what = "md"
    if what is "cache_md":
        md_cacher = DatasetCacher("md", True, DERIVATIVE_METHOD)
        md_cacher.compute_and_save()
    elif what is "cache_ffn":
        ffn_cacher = DatasetCacher("ffn", False, None)
        ffn_cacher.compute_and_save()
    elif what is "md":
        train_dataset = CachedDataset("md", "train")
        test_dataset = CachedDataset("md", "test")
        md = ModelExplorer(train_dataset, test_dataset)
        md.explore()
    elif what is "ffn":
        train_dataset = CachedDataset("ffn", "train")
        test_dataset = CachedDataset("ffn", "test")
        ffn = FFNAnalyzer(train_dataset, test_dataset)
        ffn.train_network()
        ffn.predict_train()
        ffn.predict_test()

if __name__ == '__main__':
    main()
