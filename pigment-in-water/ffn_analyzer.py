import altair as alt
import numpy as np
import pandas as pd
from keras import callbacks, layers, models
from pathlib import Path

import cv2
from cached_dataset import CachedDataset
from pca_helper import PCAHelper
from utils_io import save_video, show_video

RETAIN_THRESHOLD = 0.9

class FFNAnalyzer:
    """Analyzes data using a feed forward neural network."""

    def __init__(
        self,
        train_dataset: CachedDataset,
        test_dataset: CachedDataset,
        output_folder: str,
    ) -> None:
        """Initializes the class.
        
        Args:
            train_dataset (CachedDataset): The training dataset.
            test_dataset (CachedDataset): The test dataset.
            output_folder (str): Output folder.
        """
        self._train = train_dataset
        self._test = test_dataset
        self._output_folder = output_folder

        # Create the output directory if it doesn't exist.
        output_dir_path = Path('.', output_folder)
        output_dir_path.mkdir(exist_ok=True)

        # Flatten the data.
        flat_train_video = self._train.u.reshape(
            self._train.video_count * self._train.frame_count, -1).T
        flat_test_video = self._test.u.reshape(
            self._test.video_count * self._test.frame_count, -1).T

        # Get the train and test coefficients from the dataset, as well
        # as information to convert coefficients into a video.
        self._pca_helper = PCAHelper(flat_train_video, flat_test_video, output_folder)
        (self._train_coefficients, 
        self._test_coefficients) = self._pca_helper.get_coefficients(RETAIN_THRESHOLD)

        self._plot_modes(self._pca_helper._U_t)

    def _plot_modes(self, U: np.ndarray) -> None:
        """Plot the first four modes.
        
        Args:
            U (np.ndarray): The reduced U matrix obtained from SVD.
        """
        chart = None
        for i in range(4):
            mode = U[:, i]

            (x, y) = np.meshgrid(range(self._train.width), range(self._train.height))
            x = x.ravel()
            y = y.ravel()
            df = pd.DataFrame({
                'x': x,
                'y': y,
                'mode': mode,
            })
            new_chart = alt.Chart(df).mark_rect().encode(
                x='x:O',
                y='y:O',
                color='mode',
            ).properties(
                title=f'Mode {i + 1}'
            )
            if chart:
                chart |= new_chart
            else:
                chart = new_chart

        chart = chart.configure_axis(labels=False, ticks=False)
        path = Path(self._output_folder, 'modes.html')
        chart.configure(background='#fff').save(str(path))

    def train_network(self) -> None:
        """Trains a neural network to predict the future of the time series
        data.
        """
        # Parameters used to tweak the network.
        epochs = 10
        validation_split = 0.1

        # The training coefficients are given to us in a two-dimensional
        # array of shape (coefficient count, video count x frame count). We want
        # to slice off the first and last frame of each video, so we reshape
        # to (coefficient count, video count, frame count).
        coefficient_count = self._train_coefficients.shape[0]
        reshaped_coefficients = self._train_coefficients.reshape(
            coefficient_count, self._train.video_count, self._train.frame_count
        )

        # Slice off the last frame of each video for the input, and the first
        # frame of each video for the output.
        train_input = reshaped_coefficients[:, :, :-1]
        train_output = reshaped_coefficients[:, :, 1:]

        # Reshape so that we have (video count x frame count) training examples
        # in columns of size (coefficient count).
        train_input = train_input.reshape(coefficient_count, -1)
        train_output = train_output.reshape(coefficient_count, -1)

        # Note that the input shape should be the shape of one frame of
        # coefficients, not the shape of a sequence of frames for all time.
        self._ffn = models.Sequential()
        self._ffn.add(layers.Dense(units=1500, activation='relu',
            input_shape=(train_input.shape[0],)))
        self._ffn.add(layers.Dense(units=train_output.shape[0]))

        self._ffn.compile(loss='mean_squared_error', optimizer='Adam')

        print(self._ffn.summary())
        
        # Notice the transpose - fit expects each training example to be in 
        # its own row.
        history = self._ffn.fit(train_input.T, train_output.T, epochs=epochs,
            validation_split=validation_split, verbose=1)

        self._plot_metric(history, 'loss')

    def _plot_metric(self, history: callbacks.History, metric_name: str) -> None:
        """Plots accuracy or loss over time.
        
        Args:
            trained_model (callbacks.History): The trained neural network.
            metric_name (str): 'acc' or 'loss'.
        """
        df1 = pd.DataFrame({
            'epoch': np.array(history.epoch) + 1,
            metric_name: history.history[metric_name],
            'data': 'training'
        })
        df2 = pd.DataFrame({
            'epoch':  np.array(history.epoch) + 1,
            metric_name: history.history[f'val_{metric_name}'],
            'data': 'validation'
        })
        df = df1.append(df2)

        chart = alt.Chart(df).mark_line().encode(
            x='epoch',
            y=alt.Y(metric_name, scale=alt.Scale(zero=False)),
            color='data'
        )

        path = Path(self._output_folder, f'ffn_{metric_name}.html')
        chart.configure(background='#fff').save(
            str(path),
            scale_factor=2.0
        )

    def predict_train(self) -> None:
        """Predicts a sequence of coefficients for the train set."""
        for video_index in range(self._train.video_count):
            self._predict('train', video_index)

    def predict_test(self) -> None:
        """Predicts a sequence of coefficients for the test set."""
        for video_index in range(self._test.video_count):
            self._predict('test', video_index)

    def _predict(self, test_or_train: str, video_index: int) -> None:
        """Predicts a sequence of coefficients for one video in the train or
        test set."""
        # Decide on whether we predict train or test coefficients.
        if test_or_train == 'train':
            dataset = self._train
            actual_coefficients = self._train_coefficients
        elif test_or_train == 'test':
            dataset = self._test
            actual_coefficients = self._test_coefficients
        else:
            raise Exception('Pass "train" or "test" to _predict.')

        # Extract the coefficients corresponding to video_index.
        start_index = video_index * dataset.frame_count
        end_index = start_index + dataset.frame_count
        actual_coefficients = actual_coefficients[:, start_index:end_index]

        # Predict a whole sequence of values. Use the previous prediction
        # as input for the next prediction.
        size_prediction = actual_coefficients.shape[0]
        num_predictions = actual_coefficients.shape[1]
        predicted_coefficients = np.ndarray(shape=(size_prediction, num_predictions))
        input_coefficient = actual_coefficients[:, 0:1]
        predicted_coefficients[:, 0:1] = input_coefficient
        for i in range(1, num_predictions):
            output_coefficient = self._ffn.predict(input_coefficient.T).T
            predicted_coefficients[:, i] = output_coefficient.flatten()
            input_coefficient = output_coefficient

        self._plot_actual_predicted_coefficients(actual_coefficients, 
            predicted_coefficients, test_or_train, video_index)

        self._show_actual_predicted_videos(predicted_coefficients, 
            test_or_train, video_index)

    def _plot_actual_predicted_coefficients(self,
            actual_coefficients: np.ndarray,
            predicted_coefficients: np.ndarray,
            test_or_train: str,
            video_index: int) -> None:
        """Plots actual and predicted coefficients.
        
        Args:
            actual_coefficients (np.ndarray): The actual coefficients.
            predicted_coefficients (np.ndarray): The coefficients we predicted
            earlier.
            test_or_train (str): 'train' if we're plotting the coefficients
            for the training set, and 'test' for the test set.
        """
        num_predictions = predicted_coefficients.shape[1]

        chart = None
        for idx in np.arange(4):
            df1 = pd.DataFrame({
                'time': np.arange(num_predictions),
                'coefficients': actual_coefficients[idx, :],
                'legend': 'actual'
            })
            df2 = pd.DataFrame({
                'time': np.arange(num_predictions),
                'coefficients': predicted_coefficients[idx, :],
                'legend': 'predicted'
            })
            df = df1.append(df2)
            new_chart = alt.Chart(df).mark_line().encode(
                x='time',
                y='coefficients',
                color='legend'
            ).properties(
                title=f'Coefficient {idx + 1} - {test_or_train} video {video_index + 1}'
            ).interactive()
            if chart is None:
                chart = new_chart
            else:
                chart = alt.hconcat(chart, new_chart).resolve_scale(y='shared')

        path = Path(self._output_folder, f'ffn_coefficients_{test_or_train}_{video_index + 1}.html')
        chart.configure(background='#fff').save(
            str(path),
            scale_factor=2.0
        )

    def _show_actual_predicted_videos(self, predicted_coefficients: np.ndarray, 
        test_or_train: np.ndarray, video_index: int) -> None:
        """ Get the predicted video from coefficients. Display actual and 
        predicted videos side by side.
        
        Args:
            predicted_coefficients (np.ndarray): The coefficients we predicted
            earlier.
            test_or_train (np.ndarray): 'test' or 'train' string.
        """
        # Decide on whether to show the train or test video.
        if test_or_train == 'train':
            dataset = self._train
        elif test_or_train == 'test':
            dataset = self._test
        else:
            raise Exception('Pass "train" or "test" to _show_actual_predicted_videos.')
        actual_video = dataset.u[video_index]

        # Undo the coefficient normalization we added earlier.
        predicted_coefficients *= self._pca_helper.normalization
        # Multiplying by U_t on the left converts coefficients to video.
        predicted_video = self._pca_helper.U_t.dot(predicted_coefficients).T
        # Reshape the video from 2D to 3D ndarray.
        frame_count = predicted_video.shape[0]
        shape = (frame_count, dataset.height, dataset.width)
        predicted_video = predicted_video.reshape(shape)

        both_videos = np.concatenate((actual_video, predicted_video), axis=2)

        # Clip to prevent clamped pixels.
        both_videos = np.clip(both_videos * 255, 0, 255)
        # Convert from float values to uint8.
        both_videos = both_videos.astype('uint8')

        show_video(f'{test_or_train} {video_index + 1}', both_videos)
        file_path = f'{self._output_folder}/ffn_video_{test_or_train}_{video_index + 1}.avi'
        save_video(file_path, both_videos)
