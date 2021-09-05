import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import utils_graph
import utils_io

S3_URL = 'https://bea-portfolio.s3-us-west-2.amazonaws.com/eigenfaces/'
FILE_ZIP = 'yalefaces_cropped.zip'
FILE_TAR = 'yalefaces_uncropped.tar'
FOLDER_CROPPED = 'CroppedYale'
FOLDER_UNCROPPED = 'yalefaces'
DATA_FOLDER = 'data'
PLOTS_FOLDER = 'plots'
DATA_FILE = 'data.npz'
UNCROPPED_WIDTH = 320
UNCROPPED_HEIGHT = 243
UNCROPPED_SIZE = UNCROPPED_WIDTH * UNCROPPED_HEIGHT # 77760
# Number of uncropped images = 165
CROPPED_WIDTH = 168
CROPPED_HEIGHT = 192
CROPPED_SIZE = CROPPED_WIDTH * CROPPED_HEIGHT # 32256
# Number of cropped images = 2414

def preprocess_data() -> None:
    """Downloads and preprocesses data.
    """
    print('\nDownloading and preprocessing data.')

    (_, downloaded1) = utils_io.download_remote_data_file(DATA_FOLDER, S3_URL+FILE_ZIP)
    if downloaded1:
        utils_io.unpack_zip_file(DATA_FOLDER, FILE_ZIP)
        # Convert .pgm images to .png so we can easily visualize them.
        path_cropped = os.path.join(DATA_FOLDER, FOLDER_CROPPED)
        utils_io.convert_images(path_cropped, '.png')

    (_, downloaded2) = utils_io.download_remote_data_file(DATA_FOLDER, S3_URL+FILE_TAR)
    if downloaded2:
        utils_io.unpack_tar_file(DATA_FOLDER, FILE_TAR)
        # Add a .gif extension to uncropped images if it hasn't been added
        # already.
        path_uncropped = os.path.join(DATA_FOLDER, FOLDER_UNCROPPED)
        utils_io.append_to_all_files(path_uncropped, '.gif')

    print('Done downloading and preprocessing data.')


def _recursive_read_cropped(folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function that reads the cropped data and merges it into a single 
    matrix, recursively.
    """
    xc = np.empty((CROPPED_SIZE, 0))
    xc_labels = np.empty((0,))
    items = os.listdir(folder)
    for item in items:
        item = os.path.join(folder, item)
        if os.path.isdir(item):
            (new_xc, new_label) = _recursive_read_cropped(item)
            xc = np.append(xc, new_xc, 1)
            xc_labels = np.append(xc_labels, new_label)
        elif os.path.isfile(item) and item.endswith('.pgm'):
            try:
                image = Image.open(item)
                pixels = image.getdata()
                array = np.asarray(pixels)
                array = np.reshape(array, (array.shape[0], 1))
                xc = np.append(xc, array, 1)
                parent_dir = os.path.dirname(item)
                label = parent_dir[-3:]
                xc_labels = np.append(xc_labels, label)
            except IOError:
                print(f'Cannot open file {item}.')
    return (xc, xc_labels)


def _read_cropped() -> Tuple[np.ndarray, np.ndarray]:
    """Reads the cropped data and labels.
    """
    print('\nReading cropped images.')
    path_cropped = os.path.join(DATA_FOLDER, FOLDER_CROPPED)
    result = _recursive_read_cropped(path_cropped)
    print('Done reading cropped images.')
    return result


def _read_uncropped() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads the uncropped data and labels.
    """
    print('\nReading uncropped images.')
    xu = np.empty((UNCROPPED_SIZE, 0))
    # This will contain the subject number of the image.
    xu_labels_1 = np.empty((0,))
    # This will contain a tag such as 'happy', or 'glasses'.
    xu_labels_2 = np.empty((0,))
    path_uncropped = os.path.join(DATA_FOLDER, FOLDER_UNCROPPED)
    items = os.listdir(path_uncropped)
    for item in items:
        item = os.path.join(path_uncropped, item)
        if os.path.isfile(item):
            try:
                image = Image.open(item)
                image = image.getdata()
                image = np.asarray(image)
                image = np.reshape(image, (image.shape[0], 1))
                xu = np.append(xu, image, 1)
                item_sections = item.split('.')
                xu_labels_1 = np.append(xu_labels_1, item_sections[0][-2:])
                xu_labels_2 = np.append(xu_labels_2, item_sections[1])
            except IOError:
                print(f'Cannot open file {item}.')
    print('Done reading uncropped images.')
    return (xu, xu_labels_1, xu_labels_2)


def read_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
    np.ndarray]:
    """Reads the cropped and uncropped data and labels.
    
    Returns: 
        A tuple with the following ndarrays:
        xc: The cropped images. Each image is in a different column.
        xc_labels: Labels that identify each of the 40 individuals photographed,
        from 'B01' to 'B39'.
        xu: The uncropped images. Each image is in a different column.
        xu_labels_1: Labels that identify each of the 15 individuals 
        photographed, from '01' to '15'.
        xu_labels_2: Labels that identify different aspects of the photo, such
        as 'happy', 'glasses', 'sleepy', etc.
    """
    data_file = os.path.join(DATA_FOLDER, DATA_FILE)
    data = ()
    if os.path.exists(data_file):
        data_dict = np.load(data_file) 
        data = (data_dict[key] for key in data_dict)
    else:
        (xc, xc_labels) = _read_cropped()
        (xu, xu_labels_1, xu_labels_2) = _read_uncropped()
        data = (xc, xc_labels, xu, xu_labels_1, xu_labels_2)
        np.savez(data_file, *data)
    return data


def _save_image(array: np.ndarray, filename: str, image_type: str) -> Image:
    """Saves an image, given a flat array of double values.
    """
    # We reshape the array according to the image type.
    if image_type == 'cropped':
        image_shape = (CROPPED_HEIGHT, CROPPED_WIDTH)
    elif image_type == 'uncropped':
        image_shape = (UNCROPPED_HEIGHT, UNCROPPED_WIDTH)
    matrix = np.reshape(array, image_shape)

    # We convert the values in the image to go from 0 to 255 and be ints.
    matrix -= matrix.min()
    matrix *= 255/matrix.max()
    matrix = matrix.astype('uint8')
    image = Image.fromarray(matrix, mode='L')
    image.save(os.path.join(PLOTS_FOLDER, filename))


def svd_analysis(x: np.ndarray, image_type: str) -> None:
    """Decomposes matrix x using SVD and plots the singular values and modes.
    """
    print(f'\nAnalyzing SVD of {image_type} images.')
    # SVD.
    (u, s, vh) = np.linalg.svd(x, full_matrices=False)

    # Plot the normalized singular values. (Just the first 100 singular values.)
    normalized_s = s / np.sum(s)
    normalized_s = normalized_s[:100]
    utils_graph.graph_2d_markers(
        np.asarray(range(1, len(normalized_s)+1)),
        normalized_s, 'Mode', 'Normalized singular value',
        f'Singular values for {image_type} images', 
        PLOTS_FOLDER, 
        f'singular_values_{image_type}.html')

    # Save the first few spatial modes.
    mode_count = 6
    for mode_number in range(mode_count):
        spatial_mode = u[:, mode_number]
        filename = f'spatial_mode_{image_type}_{mode_number}.png'
        _save_image(spatial_mode, filename, image_type)

    # Plot the coefficients for the first few spatial modes for all images.
    # (Just the first 100.)
    x_projected = u.T.dot(x)
    mode_count = 2
    num_images = vh.shape[1]
    images = np.reshape(np.asarray(range(num_images)), (1, num_images))
    legend = [f'Mode {i + 1}' for i in range(mode_count)]
    utils_graph.graph_overlapping_lines(
        np.repeat(images, mode_count, axis=0)[:, :100],
        x_projected[:, :100],
        legend,
        'Images', 'Coefficient of mode', 
        f'Coefficients of spatial modes for all {image_type} images', 
        PLOTS_FOLDER, 
        f'coef_{image_type}.html')

    print(f'Done analyzing SVD of {image_type} images.')


def image_reduction_analysis(x: np.ndarray, image_type: str) -> None:
    """Reduces images to just a few spatial modes. 
    """
    print(f'\nReducing {image_type} images.')
    (u, s, vh) =  np.linalg.svd(x, full_matrices=False)
    print(f'  Shape of U: {u.shape}')
    print(f'  Shape of Sigma: {s.shape}')
    print(f'  Shape of V*: {vh.shape}')

    # We'll pick a photo to analyze and save the original.
    image_index = 0
    image = x[:, image_index]
    filename = f'reduced_{image_type}_original.png'
    _save_image(image, filename, image_type)

    # We'll analyze the recreated image when we retain just a few modes.
    print(f'  Error between {image_type} image with index {image_index} and ' +
        'its reconstruction: ')
    num_modes_list = [10, 20, 30, 40, 50]
    for num_modes in num_modes_list:
        u_reduced = u[:, :num_modes]
        x_reduced = u_reduced.T.dot(x)
        image_reduced = x_reduced[:, image_index]
        image_reconstructed = u_reduced.dot(image_reduced)
        filename = f'reduced_{image_type}_{num_modes}_modes.png'
        _save_image(image_reconstructed, filename, image_type)
        # Calculate the mean squared error between the original image and
        # the image reconstructed with just a few modes.
        mse = mean_squared_error(image, image_reconstructed)
        print(f'    {num_modes} modes: {mse}')
    
    print(f'Done reducing {image_type} images')


def classify_images(x: np.ndarray, labels: np.ndarray, image_type: str) -> None:
    """Classifies images using SVM.
    """
    print(f'\nClassifying {image_type} images.')

    # Split into train and test sets.
    (x_train, x_test, y_train, y_test) = train_test_split(x.T, labels, 
        test_size=0.2, random_state=42)

    # PCA.
    num_components = 50
    pca = PCA(n_components = num_components, svd_solver = 'randomized', 
        whiten = True)
    pca.fit(x_train)
    x_train_reduced = pca.transform(x_train)
    x_test_reduced = pca.transform(x_test)

    # Train an SVM classifier.
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(x_train_reduced, y_train)

    # Make predictions on the test set.
    y_test_predictions = clf.predict(x_test_reduced)

    # print(f'  Actual labels: {y_test[:5]}')
    # print(f'  Predicted labels: {y_test_predictions[:5]}')

    # Get accuracy of prediction.
    accuracy = accuracy_score(y_test, y_test_predictions)
    print(f'  Accuracy of prediction: {accuracy}.')

    print(f'Done classifying {image_type} images.')


def main() -> None:
    """Main program.
    """
    utils_io.find_or_create_dir(PLOTS_FOLDER)
    preprocess_data()
    (xc, xc_labels, xu, xu_labels_1, xu_labels_2) = read_data()
    svd_analysis(xc, image_type='cropped')
    svd_analysis(xu, image_type='uncropped')
    image_reduction_analysis(xc, image_type='cropped')
    image_reduction_analysis(xu, image_type='uncropped')
    classify_images(xc, xc_labels, image_type='cropped')
    classify_images(xu, xu_labels_1, image_type='uncropped')


if __name__ == '__main__':
    main()
