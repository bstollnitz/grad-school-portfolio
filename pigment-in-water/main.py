import numpy as np
import tensorflow as tf

from cached_dataset import CachedDataset
from dataset_cacher import DatasetCacher
from ffn_analyzer import FFNAnalyzer
from model_explorer import ModelExplorer
import utils_io

DERIVATIVE_METHOD = 'polynomial' # h2, h4, robust, savitzky, polynomial
DATA_FOLDER = 'data'
OUTPUT_FOLDER = 'output'
VIDEO_FILENAMES = ['red_diffusion_1.mp4', 'red_diffusion_2.mp4', 
'red_diffusion_3.mp4', 'red_diffusion_4.mp4', 'red_diffusion_5.mp4']
S3_URL = 'https://bea-portfolio.s3-us-west-2.amazonaws.com/pigment-in-water/'


def ensure_reproducibility() -> None:
    """Ensures reproducibility of results.
    """
    np.random.seed(1)
    tf.random.set_seed(1)


def download_videos() -> None:
    """Downloads videos from remote location if they're not yet 
    present locally.
    """
    for item in VIDEO_FILENAMES:
        utils_io.download_remote_data_file(DATA_FOLDER, S3_URL, item)


def generate_cached_data() -> None:
    """Reads video files and generates preprocessed data files."""
    # Preprocess data for the feed-forward neural network.
    ffn_cacher = DatasetCacher('ffn', False, None, DATA_FOLDER)
    ffn_cacher.compute_and_save()

    # Preprocess data for model discovery.
    md_cacher = DatasetCacher('md', True, DERIVATIVE_METHOD, DATA_FOLDER)
    md_cacher.compute_and_save()


def main() -> None:
    """Main program."""
    ensure_reproducibility()
    download_videos()
    generate_cached_data()

    print('Feed-forward network')
    print('====================')
    ffn_train_dataset = CachedDataset('ffn', 'train', DATA_FOLDER)
    ffn_test_dataset = CachedDataset('ffn', 'test', DATA_FOLDER)
    ffn = FFNAnalyzer(ffn_train_dataset, ffn_test_dataset, OUTPUT_FOLDER)
    ffn.train_network()
    ffn.predict_train()
    ffn.predict_test()

    print('Model discovery')
    print('===============')
    md_train_dataset = CachedDataset('md', 'train', DATA_FOLDER)
    md_test_dataset = CachedDataset('md', 'test', DATA_FOLDER)
    md = ModelExplorer(md_train_dataset, md_test_dataset, OUTPUT_FOLDER)
    md.explore()


if __name__ == '__main__':
    main()
