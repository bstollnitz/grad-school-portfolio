import datetime
import time
from typing import List, Tuple

import termcolor
from sklearn import model_selection
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.wrappers import scikit_learn

import tuning_io_utils
from vector_dataset import VectorDataset

# The total number of models trained is RANDOM_SAMPLES * K_FOLD_VALIDATION.
RANDOM_SAMPLES = 1000
K_FOLD_VALIDATION = 3

def create_optimizer(optimizer: str, learning_rate: float, decay: float, 
    epochs: int, beta_1: float, beta_2: float, epsilon: float) -> optimizers.Optimizer:
    """
    Returns an optimizer according to the specification in the 
    hyperparameter configuration.
    """

    decay = decay / epochs

    if optimizer == "SGD":
        opt_instance = optimizers.SGD(decay=decay)
    elif optimizer == "adam":
        opt_instance = optimizers.Adam(decay=decay, beta_1=beta_1,
            beta_2=beta_2, epsilon=epsilon)
    elif optimizer == "rmsprop":
        opt_instance = optimizers.RMSprop(decay=decay)
    else:
        raise Exception(f"Unknown optimizer type: {optimizer}")

    opt_instance.learning_rate = learning_rate * opt_instance.learning_rate

    return opt_instance

def create_and_compile_model(epochs: int, learning_rate: float, decay: float,
    activation: str, hidden_layers: int, units: int, optimizer: str,
    beta_1: float, beta_2: float, epsilon: float,
    input_dim: int, output_dim: int, index_storage: dict) -> models.Sequential:
    """
    Returns a compiled model according to the specification in the
    hyperparameter configuration.
    """

    index = index_storage["index"]
    max_index = index_storage["max_index"]
    start_time = index_storage["start_time"]
    configuration = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "decay": decay,
        "activation": activation,
        "hidden_layers": hidden_layers,
        "units": units,
        "optimizer": optimizer,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "epsilon": epsilon,
    }
    tuning_io_utils.print_configuration(configuration, index,
        max_index, start_time)

    index_storage["index"] = index + 1

    opt_instance = create_optimizer(optimizer, learning_rate, decay, epochs,
        beta_1, beta_2, epsilon)

    model = models.Sequential()
    
    for i in range(hidden_layers):
        input_dim = input_dim if i == 0 else units
        dense = layers.Dense(units, input_dim=input_dim, activation=activation)
        model.add(dense)

    model.add(layers.Dense(output_dim, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=opt_instance, 
        metrics=["accuracy"])

    return model

def get_print_accuracy(classifier, x, y):
    """
    Returns and prints the accuracy of the classifier passed as a parameter.
    """

    accuracy = classifier.score(x, y, verbose=0)
    termcolor.cprint(f"accuracy: {accuracy:.2%}", "yellow")
    return accuracy

def tune_vector_hyperparameters(data: VectorDataset,
    param_distributions: dict) -> Tuple[List[float], List[dict], int, dict]:
    """
    Performs randomized hyperparameter search using k-fold cross-validation 
    for the current hyperparameter specification.
    Evaluates the best model using the test set.
    """

    # Get dimensions.
    input_dim = data.train_x.shape[1]
    output_dim = len(data.categories)
    n_iter = RANDOM_SAMPLES
    k_fold_cv = K_FOLD_VALIDATION
    # The total number of models created is the number of random parameter
    # samples (n_iter) times the number of k-fold cross validations.
    max_index = n_iter * k_fold_cv
    start_time = time.monotonic()
    index_storage = {
        "index": 0,
        "max_index": max_index,
        "start_time": start_time
    }

    build_fn = (lambda epochs, learning_rate, decay, activation, hidden_layers,
        units, optimizer, beta_1=0.9, beta_2=0.999, epsilon=1e-7: create_and_compile_model(epochs,
        learning_rate, decay, activation, hidden_layers, units, optimizer,
        beta_1, beta_2, epsilon, input_dim, output_dim, index_storage))

    classifier = scikit_learn.KerasClassifier(build_fn=build_fn)

    random_search = model_selection.RandomizedSearchCV(estimator=classifier, 
        param_distributions=param_distributions, n_iter=n_iter,
        scoring=get_print_accuracy, n_jobs=1, cv=k_fold_cv, refit=False)

    random_search.fit(data.train_x, data.train_y_one_hot, verbose=0)

    # Get the index and values of the best hyperparameters.
    best_index = random_search.best_index_
    results = random_search.cv_results_
    best_params = results["params"][best_index]

    # Fit the model to all of the training data using the best hyperparameters,
    # evaluating against the test data as well.
    index_storage = { "index": 0, "max_index": 1, "start_time": start_time }
    classifier.set_params(**best_params)
    history = classifier.fit(data.train_x, data.train_y_one_hot, verbose=0,
        validation_data=(data.test_x, data.test_y_one_hot))
    best_history = history.history

    return (results["mean_test_score"], results["params"], best_index,
        best_history)
