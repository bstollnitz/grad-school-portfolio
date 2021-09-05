import time
from typing import List, Tuple

import numpy as np
import tensorflow.keras.backend as K
import termcolor
from sklearn import model_selection
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing import image

import tuning_io_utils
from hyperparameters import Hyperparameters
from image_dataset import ImageDataset

IMAGE_SIZE = 64
COLOR_MODE = "rgb" # could be "grayscale"
COLOR_CHANNELS = 3 # change to 1 for grayscale
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, COLOR_CHANNELS)
RANDOM_SAMPLES = 1000
VALIDATION_SPLIT = 0.1

def create_optimizer(configuration: dict) -> optimizers.Optimizer:
    """
    Returns an optimizer according to the specification in the 
    hyperparameter configuration.
    """

    optimizer = configuration["optimizer"]
    epochs = configuration["epochs"]
    decay = configuration["decay"] / epochs
    learning_rate = configuration["learning_rate"]

    if optimizer == "SGD":
        opt_instance = optimizers.SGD(decay=decay)
    elif optimizer == "adam":
        opt_instance = optimizers.Adam(decay=decay)
        if "beta_1" in configuration:
            opt_instance.beta_1 = configuration["beta_1"]
        if "beta_2" in configuration:
            opt_instance.beta_2 = configuration["beta_2"]
        if "epsilon" in configuration:
            opt_instance.epsilon = configuration["epsilon"]
    elif optimizer == "rmsprop":
        opt_instance = optimizers.RMSprop(decay=decay)
    else:
        raise Exception(f"Unknown optimizer type: {optimizer}")

    opt_instance.learning_rate = learning_rate * opt_instance.learning_rate

    return opt_instance

def create_model_simple1(output_size: int) -> models.Sequential:
    """
    Creates a simple CNN that is known to perform well on MNIST.
    Adapted from sample code at https://keras.io/examples/mnist_cnn/.
    """

    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu",
        input_shape=INPUT_SHAPE))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_size, activation="softmax"))
    return model

def create_model_simple2(output_size: int) -> models.Sequential:
    """
    Creates a slightly more complex CNN.
    """

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu",
        input_shape=INPUT_SHAPE))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_size, activation="softmax"))
    return model

def create_model_resnet50(output_size: int) -> models.Sequential:
    """
    Loads a ResNet50V2 network with weights pretrained on ImageNet, freezes
    those weights, and adds an output layer of the size we need. (ResNet V2
    adds batch normalization before convolution layers.)
    """

    # Create the base model from ResNet50V2 without its last layer.
    base_model = ResNet50V2(weights="imagenet", include_top=False)

    # Average and use fully-connected layers to reach our desired output size.
    average = layers.GlobalAveragePooling2D()(base_model.output)
    dense1 = layers.Dense(512, activation="relu")(average)
    dense2 = layers.Dense(64, activation="relu")(dense1)
    new_output = layers.Dense(output_size, activation='softmax')(dense2)

    # Make a new model that goes from the base model's input to the new output.
    model = models.Model(inputs=base_model.input, outputs=new_output)

    # Freeze all the weights in the base model.
    for layer in base_model.layers:
        layer.trainable = False

    return model

def print_parameter_summary(model) -> None:
    """
    Prints the number of trainable and non-trainable parameters in the model.
    """

    # Note: use model.summary() for a detailed summary of layers.
    trainable_count = np.sum([K.count_params(w) for w in
        model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in
        model.non_trainable_weights])
    total_count = trainable_count + non_trainable_count

    print(f"Total params: {total_count:,}")
    print(f"Trainable params: {trainable_count:,}")
    print(f"Non-trainable params: {non_trainable_count:,}")

def create_and_compile_model(configuration: dict,
    output_size: int) -> models.Sequential:
    """
    Returns a compiled model according to the specification in the
    hyperparameter configuration.
    """

    model_creation_functions = {
        "simple1": create_model_simple1,
        "simple2": create_model_simple2,
        "resnet50": create_model_resnet50,
    }
    model_creation_function = model_creation_functions[configuration["model"]]
    model = model_creation_function(output_size)
    print_parameter_summary(model)

    opt_instance = create_optimizer(configuration)
    model.compile(loss="categorical_crossentropy", optimizer=opt_instance,
        metrics=["accuracy"])
    return model

def evaluate_hyperparameters(data: ImageDataset, configuration: dict) -> dict:
    """
    Fits a single model to the training data, returning the final loss and
    accuracy of training and validation.
    """

    batch_size = configuration["batch_size"]

    train_data, validation_data = model_selection.train_test_split(data.train,
        test_size=VALIDATION_SPLIT)

    category_list = data.categories.tolist()
    data_generator = image.ImageDataGenerator(rescale=1./255)
    train_generator = data_generator.flow_from_dataframe(
        dataframe=train_data,
        directory=f"./data/labeled-aliased-{IMAGE_SIZE}-png",
        x_col="Filename",
        y_col="Category",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode=COLOR_MODE,
        classes=category_list,
        class_mode="categorical",
        batch_size=batch_size
    )
    validation_generator = data_generator.flow_from_dataframe(
        dataframe=validation_data,
        directory=f"./data/labeled-aliased-{IMAGE_SIZE}-png",
        x_col="Filename",
        y_col="Category",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode=COLOR_MODE,
        classes=category_list,
        class_mode="categorical",
        batch_size=batch_size
    )

    output_size = len(data.categories)
    model = create_and_compile_model(configuration, output_size)
    epochs = configuration["epochs"]
    train_steps = train_generator.n // train_generator.batch_size
    validation_steps = validation_generator.n // validation_generator.batch_size
    history = model.fit_generator(generator=train_generator,
        steps_per_epoch=train_steps, validation_data=validation_generator,
        validation_steps=validation_steps, epochs=epochs)

    result = {
        "train_loss": history.history["loss"][-1],
        "train_accuracy": history.history["accuracy"][-1],
        "val_loss": history.history["val_loss"][-1],
        "val_accuracy": history.history["val_accuracy"][-1]
    }
    termcolor.cprint(f"Validation accuracy: {result['val_accuracy']:.2%}", "yellow")

    return result

def train_and_evaluate(data: ImageDataset, configuration: dict) -> dict:
    """
    Fits a single model using all the training data, then evaluates it using the
    test data. Returns the entire history of loss and accuracy on training and
    test data.
    """

    batch_size = configuration["batch_size"]

    category_list = data.categories.tolist()
    data_generator = image.ImageDataGenerator(rescale=1./255)
    train_generator = data_generator.flow_from_dataframe(
        dataframe=data.train,
        directory=f"./data/labeled-aliased-{IMAGE_SIZE}-png",
        x_col="Filename",
        y_col="Category",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode=COLOR_MODE,
        classes=category_list,
        class_mode="categorical",
        batch_size=batch_size
    )
    test_generator = data_generator.flow_from_dataframe(
        dataframe=data.test,
        directory=f"./data/labeled-aliased-{IMAGE_SIZE}-png",
        x_col="Filename",
        y_col="Category",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode=COLOR_MODE,
        classes=category_list,
        class_mode="categorical",
        batch_size=batch_size
    )

    output_size = len(data.categories)
    model = create_and_compile_model(configuration, output_size)
    epochs = configuration["epochs"]
    train_steps = train_generator.n // train_generator.batch_size
    test_steps = test_generator.n // test_generator.batch_size
    history = model.fit_generator(generator=train_generator,
        steps_per_epoch=train_steps, validation_data=test_generator,
        validation_steps=test_steps, epochs=epochs)
    return history.history

def tune_image_hyperparameters(data: ImageDataset,
    param_distributions: dict) -> Tuple[List[float], List[dict], int, dict]:
    """
    Performs randomized hyperparameter search for the current hyperparameter
    specification. Evaluates the best model using the test set.
    """

    hyperparameters = Hyperparameters(param_distributions)
    print(f"Number of combinations: {len(hyperparameters.combinations)}")
    configurations = hyperparameters.sample_combinations(RANDOM_SAMPLES)
    configuration_count = len(configurations)
    print(f"Sampled combinations: {configuration_count}")

    results = []
    start_time = time.monotonic()
    for (index, configuration) in enumerate(configurations):
        tuning_io_utils.print_configuration(configuration, index,
            configuration_count, start_time)

        result = evaluate_hyperparameters(data, configuration)
        results.append(result)

    # Figure out the index of the configuration that produced the best score.
    scores = [result["val_accuracy"] for result in results]
    best_index = np.argmax(scores)

    # Retrain the best configuration using all the training data and measure
    # accuracy on the test data.
    best_history = train_and_evaluate(data, configurations[best_index])

    return (scores, configurations, best_index, best_history)
