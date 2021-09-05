import datetime
import pprint
import random
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
import termcolor
from sklearn import linear_model, model_selection
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.wrappers import scikit_learn

import graphing
import tuning_io_utils
from image_dataset import ImageDataset
from image_processing import tune_image_hyperparameters
from vector_dataset import VectorDataset
from vector_processing import tune_vector_hyperparameters

# We only need to preprocess data once.
REPROCESS_DATA = True
# We retrain once per hyperparameter configuration round.
# Running the program with this set to false re-generates the graphs for the
# previous round.
RETRAIN = True

# Vector hyperparameter configuration for round 1.
VECTOR_ROUND1_PARAM_DISTRIBUTIONS = {
    "epochs": [60],
    "batch_size": [32, 64, 128, 512, 1024],
    "learning_rate": [0.01, 0.1, 1, 10],
    "decay": [0, 0.1, 0.01, 0.001],
    "activation": ["relu", "tanh"],
    "hidden_layers": [1, 2, 3, 4],
    "units": [32, 64, 128],
    "optimizer": ["adam", "rmsprop", "SGD"],
}

# Vector hyperparameter configuration for round 2.
VECTOR_ROUND2_PARAM_DISTRIBUTIONS = {
    "epochs": [60],
    "batch_size": [16, 32, 64, 128],
    "learning_rate": [1, 3, 5, 7, 10],
    "decay": [0.01, 0.1],
    "activation": ["relu", "tanh"],
    "hidden_layers": [3, 4, 5],
    "units": [128],
    "optimizer": ["adam"],
    "beta_1": [0.9, 0.99],
    "beta_2": [0.999, 0.9999],
    "epsilon": [1e-10, 1e-7, 1e-4],
}

# Image hyperparameter configuration for round 1.
IMAGE_ROUND1_PARAM_DISTRIBUTIONS = {
    "epochs": [10],
    "batch_size": [32, 64],
    "learning_rate": [1, 5, 10],
    "decay": [0, 0.1],
    "optimizer": ["adam"],
    "model": ["simple1", "simple2", "resnet50"],
}

# Image hyperparameter configuration for round 2.
IMAGE_ROUND2_PARAM_DISTRIBUTIONS = {
    "epochs": [20],
    "batch_size": [32],
    "learning_rate": [1, 2, 3, 4, 5],
    "decay": [0.05, 0.1],
    "optimizer": ["adam"],
    "model": ["simple1"],
    "epsilon": [1e-10, 1e-7],
}

# We save this many best hyperparameter combinations.
TOP_COUNT = 20

GRAPH_HISTOGRAM_FILENAME = "graphs/category-histograms.html"

# These are saved inside directories named "round1", "round2", etc.
ROUND_CORRELATION_FILENAME = "nn-correlation.html"
ROUND_ACCURACY_FILENAME = "accuracy.html"
ROUND_BEST_FILENAME = "best-nn-history.html"
ROUND_RESULTS_FILENAME = "results.pickle"
ROUND_TOP_FILENAME = "top-params.csv"

def ensure_reproducibility() -> None:
    """
    Ensure reproducibility of results.

    Also, set the following environment variables in launch.json:
    PYTHONHASHSEED = 0
    CUDA_VISIBLE_DEVICES = ""

    Keep in mind that to ensure reproducibility, we lose the ability to 
    multithread and to run in the GPU. So call this only when absolutely 
    needed.
    """

    # Ensure that Numpy generated random numbers are reproducible.
    np.random.seed(1)
    # Ensure that Python generated random numbers are reproducible.
    random.seed(1)
    # Ensure that TensorFlow generated numbers are reproducible.
    tf.random.set_seed(1)
    # Force TensorFlow to use a single thread because multiple threads are a 
    # potential source of non-reproducible results.
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

def try_logistic_regression(data: VectorDataset) -> None:
    """
    Does logistic regression on the train data, and evaluates the fit on the
    test data.
    """

    classifier = linear_model.LogisticRegression(multi_class="multinomial", 
        solver="lbfgs", max_iter=5000).fit(data.train_x, data.train_y)
    score = classifier.score(data.test_x, data.test_y)
    termcolor.cprint(f"Accuracy using LogisticRegression: {score:.2%}", "yellow")

def print_best(params: List[dict], best_index: int,
    best_test_accuracy: float) -> None:
    """ 
    Prints the best accuracy we achieved and the associated parameters.
    """

    best_params = params[best_index]
    termcolor.cprint(f"\nBest test accuracy using neural network: {best_test_accuracy:.2%}", 
        "yellow")
    color = "blue"
    termcolor.cprint(f"Best parameters:", color)
    termcolor.cprint(pprint.pformat(best_params), color)

def try_neural_networks(data: Union[VectorDataset, ImageDataset],
    param_distributions: dict) -> None:
    """
    Tries neural networks using several hyperparameter combinations.
    """

    if isinstance(data, VectorDataset):
        tune_hyperparameters = tune_vector_hyperparameters
        round_prefix = "vector_round"
    elif isinstance(data, ImageDataset):
        tune_hyperparameters = tune_image_hyperparameters
        round_prefix = "image_round"
    else:
        raise Exception(f"Unknown data type: {type(data)}")

    if RETRAIN:
        # Select the hyperparameter combinations we want to try and train the 
        # network using those. Save the results.
        (scores, params, best_index, best_history) = tune_hyperparameters(
            data, param_distributions)

        path_str = tuning_io_utils.create_next_round_dir(round_prefix)
        tuning_io_utils.save_results(param_distributions, scores, params, 
            best_index, best_history, path_str + "/" + ROUND_RESULTS_FILENAME)

    else:
        # Load previously saved results.
        path_str = tuning_io_utils.get_last_round_dir(round_prefix)
        (param_distributions, scores, params, best_index,
            best_history) = tuning_io_utils.load_results(
            path_str + "/" + ROUND_RESULTS_FILENAME)

    # Save several plots related to this round of hyperparameter tuning.
    best_test_accuracy = np.max(best_history["val_accuracy"])
    print_best(params, best_index, best_test_accuracy)
    tuning_io_utils.save_top(path_str + "/" + ROUND_TOP_FILENAME, scores,
        params, TOP_COUNT, best_test_accuracy)
    graphing.plot_correlations(param_distributions, scores, params, 
        path_str + "/" + ROUND_CORRELATION_FILENAME)
    graphing.plot_accuracy(scores, path_str + "/" + ROUND_ACCURACY_FILENAME)
    graphing.plot_nn(best_history, path_str + "/" + ROUND_BEST_FILENAME)

def main() -> None:
    """
    Main program.
    """

    ensure_reproducibility()

    # --- Vector data ---

    # Load the vector dataset.
    vector_data = VectorDataset(REPROCESS_DATA)

    # Plot histogram that show distribution of labels in train and test sets.
    # If the two distributions are similar, we know that we've done a good job
    # with the splitting.
    graphing.plot_category_histograms(vector_data, GRAPH_HISTOGRAM_FILENAME)

    # Try using logistic regression.
    try_logistic_regression(vector_data)

    # Try using neural networks. This should work better.
    try_neural_networks(vector_data, VECTOR_ROUND1_PARAM_DISTRIBUTIONS)


    # --- Image data ---

    # Load the image dataset.
    image_data = ImageDataset(REPROCESS_DATA)

    # Try using convolutional neural networks.
    try_neural_networks(image_data, IMAGE_ROUND1_PARAM_DISTRIBUTIONS)

if __name__ == '__main__':
    main()
