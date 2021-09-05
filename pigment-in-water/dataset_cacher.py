from typing import List, Tuple
from pathlib import Path

import numpy as np

import cv2

# How many frames to read for each frame that is stored in the dataset:
SPEEDUP_FACTOR = 50

# How many points to use when constructing a local polynomial:
POLYNOMIAL_SAMPLES = 9

class DatasetCacher:
    """Processes videos and saves data to be used in CachedDataset."""

    def __init__(
        self,
        name: str,
        downsample_video: bool,
        derivative_method: str,
        data_folder: str
    ) -> None:
        """Processes the video files and saves u and its derivatives.
        
        Args:
            name (str): A name for the dataset.
            downsample_video (bool): Whether to downsample the videos.
            derivative_method (str): 'polynomial' (preferred), 'h2', 'h4',
                'robust', 'savitzky', or None.
            data_folder (str): The data folder.
        """
        self._name = name
        self._downsample_video = downsample_video
        self._derivative_method = derivative_method
        self._data_folder = data_folder

    def compute_and_save(self) -> None:
        """Loads videos, computes derivatives, and saves everything."""
        file_path = Path(self._data_folder, 
            f'dataset-{self._name}-train.npz')
        if file_path.exists():
            print(f'Using cached data for {self._name}.')
            return

        # Load videos.
        print('Loading videos.')
        u_train = self._load_videos([1, 2, 3, 4])
        u_test = self._load_videos([5])

        # Process and save caches for the training and test data.
        self._cache_dataset('train', u_train)
        self._cache_dataset('test', u_test)

    def _load_videos(self, indices: List[int]) -> np.ndarray:
        """Loads a sequence of videos.
        
        Args:
            indices (List[int]): The indices of the videos to load.
        
        Returns:
            np.ndarray: An array of shape (video count, frame count, y, x).
        """
        videos = []
        for index in indices:
            videos.append(self._load_video(index))

        # Stack all videos in a four-dimensional array, where the dimensions
        # are: (video count, frame count, y, x).
        return np.stack(videos, axis=0)

    def _load_video(self, index: int) -> np.ndarray:
        """Loads a single video.
        
        Args:
            index (int): The index of the video to load.
        
        Returns:
            np.ndarray: An array of shape (frame count, y, x).
        """
        # Initialize the video reader and get the video dimensions.
        file_path = Path(self._data_folder, f'red_diffusion_{index}.mp4')
        print(f'  Loading {file_path}.')
        capture = cv2.VideoCapture(str(file_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Construct a 3D ndarray with space for the video frames, after
        # accounting for a speedup factor.
        speedup_frame_count = int((frame_count + SPEEDUP_FACTOR) / SPEEDUP_FACTOR)
        if self._downsample_video:
            frame_width = int(frame_width / 2)
            frame_height = int(frame_height / 2)
        video = np.empty((speedup_frame_count, frame_height, frame_width), 
            np.dtype('uint8'))

        # Read each frame of the video capture. If the frame index is a multiple
        # of our speedup factor, convert the frame to grayscale and store it in
        # the 3D ndarray constructed earlier.
        frame_index = 0
        speedup_frame_index = 0
        success = True
        while frame_index < frame_count and success:
            (success, frame) = capture.read()
            if success and frame_index % SPEEDUP_FACTOR == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                if self._downsample_video:
                    frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_CUBIC)
                video[speedup_frame_index] = frame
                speedup_frame_index += 1
            frame_index += 1
        capture.release()
        
        # Invert brightness of frames and convert to floats.
        max = np.amax(video)
        video = (max - video) / 255.0

        return video

    def _cache_dataset(self, train_or_test: str, u: np.ndarray):
        """Saves u and, optionally, its derivatives in a file.
        
        Args:
            train_or_test (str): 'train' or 'test'.
            u (np.ndarray): The pixel values.
        """

        if self._derivative_method:
            print(f'Calculating derivatives for {train_or_test}.')
            h = 1
            k = 1
            if self._derivative_method is 'polynomial':
                (u_t, u_x, u_xx, u_y, u_yy) = self._compute_polynomial_derivatives(u)
            elif self._derivative_method is 'h2':
                (u_t, u_x, u_xx, u_y, u_yy) = self._compute_h2_derivatives(u, h, k)
            elif self._derivative_method is 'h4':
                (u_t, u_x, u_xx, u_y, u_yy) = self._compute_h4_derivatives(u, h, k)
            elif self._derivative_method is 'robust':
                (u_t, u_x, u_xx, u_y, u_yy) = self._compute_robust_derivatives(u, h, k)
            elif self._derivative_method is 'savitzky':
                (u_t, u_x, u_xx, u_y, u_yy) = self._compute_savitzky_derivatives(u, h, k)

            # Trim u so that its dimensions match the derivatives.
            if self._derivative_method is 'h2':
                u = u[:, 1:-1, 1:-1, 1:-1]
            elif self._derivative_method is 'polynomial':
                margin = int((POLYNOMIAL_SAMPLES - 1) / 2)
                u = u[:, margin:-margin, margin:-margin, margin:-margin]
            else:
                u = u[:, 2:-2, 2:-2, 2:-2]

        file_path = Path(self._data_folder, 
            f'dataset-{self._name}-{train_or_test}.npz')
        print(f'  Saving data to {file_path}.')
        if self._derivative_method:
            np.savez(file_path, u=u, u_t=u_t, u_x=u_x, u_xx=u_xx, u_y=u_y, u_yy=u_yy)
        else:
            np.savez(file_path, u=u)

    def _compute_h2_derivatives(self, u: np.ndarray, h: float,
        k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray]:
        """Computes derivatives of u using second-order finite differences.
        
        Args:
            u (np.ndarray): The data.
            h (float): The spatial step size.
            k (float): The time step size.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray]:
            (u_t, u_x, u_xx, u_y, u_yy)
        """
        # See Table 6.3 'Central-difference formulas of order O(h^2)' in
        # http://mathfaculty.fullerton.edu/mathews/n2003/differentiation/NumericalDiffProof.pdf.
        u_t = (u[:, 2:, 1:-1, 1:-1] - u[:, 0:-2, 1:-1, 1:-1]) / (2 * k)
        u_x = (u[:, 1:-1, 1:-1, 2:] - u[:, 1:-1, 1:-1, 0:-2]) / (2 * h)
        u_xx = (u[:, 1:-1, 1:-1, 0:-2] - 2 * u[:, 1:-1, 1:-1, 1:-1] + u[:, 1:-1, 1:-1, 2:]) / (h**2)
        u_y = (u[:, 1:-1, 2:, 1:-1] - u[:, 1:-1, 0:-2, 1:-1]) / (2 * h)
        u_yy = (u[:, 1:-1, 0:-2, 1:-1] - 2 * u[:, 1:-1, 1:-1, 1:-1] + u[:, 1:-1, 2:, 1:-1]) / (h**2)
        return (u_t, u_x, u_xx, u_y, u_yy)

    def _compute_h4_derivatives(self, u: np.ndarray, h: float,
        k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray]:
        """Computes derivatives of u using fourth-order finite differences.
        
        Args:
            u (np.ndarray): The data.
            h (float): The spatial step size.
            k (float): The time step size.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray]:
            (u_t, u_x, u_xx, u_y, u_yy)
        """
        # See Table 6.4 'Central-difference formulas of order O(h^4)' in
        # http://mathfaculty.fullerton.edu/mathews/n2003/differentiation/NumericalDiffProof.pdf.
        u_t = (+ 1 * u[:, 0:-4, 2:-2, 2:-2]
            - 8 * u[:, 1:-3, 2:-2, 2:-2]
            + 8 * u[:, 3:-1, 2:-2, 2:-2]
            - 1 * u[:, 4:  , 2:-2, 2:-2]) / (12 * k)
        u_x = (+ 1 * u[:, 2:-2, 2:-2, 0:-4]
            - 8 * u[:, 2:-2, 2:-2, 1:-3]
            + 8 * u[:, 2:-2, 2:-2, 3:-1]
            - 1 * u[:, 2:-2, 2:-2, 4:  ]) / (12 * h)
        u_xx = (-  1 * u[:, 2:-2, 2:-2, 0:-4]
                + 16 * u[:, 2:-2, 2:-2, 1:-3]
                - 30 * u[:, 2:-2, 2:-2, 2:-2]
                + 16 * u[:, 2:-2, 2:-2, 3:-1]
                -  1 * u[:, 2:-2, 2:-2, 4:  ]) / (12 * h**2)
        u_y = (+ 1 * u[:, 2:-2, 0:-4, 2:-2]
            - 8 * u[:, 2:-2, 1:-3, 2:-2]
            + 8 * u[:, 2:-2, 3:-1, 2:-2]
            - 1 * u[:, 2:-2, 4:  , 2:-2]) / (12 * h)
        u_yy = (-  1 * u[:, 2:-2, 0:-4, 2:-2]
                + 16 * u[:, 2:-2, 1:-3, 2:-2]
                - 30 * u[:, 2:-2, 2:-2, 2:-2]
                + 16 * u[:, 2:-2, 3:-1, 2:-2]
                -  1 * u[:, 2:-2, 4:  , 2:-2]) / (12 * h**2)
        return (u_t, u_x, u_xx, u_y, u_yy)

    def _compute_robust_derivatives(self, u: np.ndarray, h: float,
        k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray]:
        """Computes derivatives of u using robust finite differences.
        
        Args:
            u (np.ndarray): The data.
            h (float): The spatial step size.
            k (float): The time step size.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray]:
            (u_t, u_x, u_xx, u_y, u_yy)
        """
        # See table 'Smooth noise-robust differentiators (n=2, exact on 1, x, x^2)'
        # and table 'Second-order smooth differentiators (exact on 1, x, x^2, x^3)'
        # in http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/.
        u_t = (- 1 * u[:, 0:-4, 2:-2, 2:-2]
            - 2 * u[:, 1:-3, 2:-2, 2:-2]
            + 2 * u[:, 3:-1, 2:-2, 2:-2]
            + 1 * u[:, 4:  , 2:-2, 2:-2]) / (8 * k)
        u_x = (- 1 * u[:, 2:-2, 2:-2, 0:-4]
            - 2 * u[:, 2:-2, 2:-2, 1:-3]
            + 2 * u[:, 2:-2, 2:-2, 3:-1]
            + 1 * u[:, 2:-2, 2:-2, 4:  ]) / (8 * h)
        u_xx = (+ 1 * u[:, 2:-2, 2:-2, 0:-4]
                - 2 * u[:, 2:-2, 2:-2, 2:-2]
                + 1 * u[:, 2:-2, 2:-2, 4:  ]) / (4 * h**2)
        u_y = (- 1 * u[:, 2:-2, 0:-4, 2:-2]
            - 2 * u[:, 2:-2, 1:-3, 2:-2]
            + 2 * u[:, 2:-2, 3:-1, 2:-2]
            + 1 * u[:, 2:-2, 4:  , 2:-2]) / (8 * h)
        u_yy = (+ 1 * u[:, 2:-2, 0:-4, 2:-2]
                - 2 * u[:, 2:-2, 2:-2, 2:-2]
                + 1 * u[:, 2:-2, 4:  , 2:-2]) / (4 * h**2)
        return (u_t, u_x, u_xx, u_y, u_yy)

    def _compute_savitzky_derivatives(self, u: np.ndarray, h: float,
        k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray]:
        """Computes derivatives of u using Savitzky's smooth finite differences.
        
        Args:
            u (np.ndarray): The data.
            h (float): The spatial step size.
            k (float): The time step size.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray]:
            (u_t, u_x, u_xx, u_y, u_yy)
        """
        # See table 'Coefficients for 1st derivative' 
        # and table 'Coefficients for 2nd derivative'
        # both with window size 5
        # in https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
        u_t = (- 2 * u[:, 0:-4, 2:-2, 2:-2]
            - 1 * u[:, 1:-3, 2:-2, 2:-2]
            + 1 * u[:, 3:-1, 2:-2, 2:-2]
            + 2 * u[:, 4:  , 2:-2, 2:-2]) / (10 * k)
        u_x = (- 2 * u[:, 2:-2, 2:-2, 0:-4]
            - 1 * u[:, 2:-2, 2:-2, 1:-3]
            + 1 * u[:, 2:-2, 2:-2, 3:-1]
            + 2 * u[:, 2:-2, 2:-2, 4:  ]) / (10 * h)
        u_xx = (  2 * u[:, 2:-2, 2:-2, 0:-4]
                - 1 * u[:, 2:-2, 2:-2, 1:-3]
                - 2 * u[:, 2:-2, 2:-2, 2:-2]
                - 1 * u[:, 2:-2, 2:-2, 3:-1]
                + 2 * u[:, 2:-2, 2:-2, 4:  ]) / (7 * h**2)
        u_y = (- 2 * u[:, 2:-2, 0:-4, 2:-2]
            - 1 * u[:, 2:-2, 1:-3, 2:-2]
            + 1 * u[:, 2:-2, 3:-1, 2:-2]
            + 2 * u[:, 2:-2, 4:  , 2:-2]) / (10 * h)
        u_yy = (  2 * u[:, 2:-2, 0:-4, 2:-2]
                - 1 * u[:, 2:-2, 2:-2, 2:-2]
                - 2 * u[:, 2:-2, 2:-2, 2:-2]
                - 1 * u[:, 2:-2, 2:-2, 2:-2]
                + 2 * u[:, 2:-2, 4:  , 2:-2]) / (7 * h**2)
        return (u_t, u_x, u_xx, u_y, u_yy)

    def _compute_polynomial_derivatives(self, u: np.ndarray) -> Tuple[np.ndarray, 
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes the first and second derivatives of u in space and time
        using polynomial fitting.
        
        Args:
            u (np.ndarray): The pixel values.
        
        Returns:
            Tuple[np.ndarray, np.ndarray,np.ndarray, np.ndarray, np.ndarray]: The
            derivatives, (u_t, u_x, u_xx, u_y, u_yy).
        """

        # Polynomial approximation is described briefly under 'Numerical
        # evaluation of derivatives' in
        # https://advances.sciencemag.org/content/3/4/e1602614. They cite
        # http://emis.ams.org/journals/EJDE/conf-proc/21/k3/knowles.pdf.

        # Calculate sizes.
        n = POLYNOMIAL_SAMPLES
        margin = int((n - 1) / 2)
        (video_count, frame_count, y_count, x_count) = u.shape
        result_shape = (video_count, frame_count - 2 * margin,
            y_count - 2 * margin, x_count - 2 * margin)

        # Initialize results to empty arrays.
        u_t = np.empty(result_shape)
        u_x = np.empty(result_shape)
        u_xx = np.empty(result_shape)
        u_y = np.empty(result_shape)
        u_yy = np.empty(result_shape)

        # Construct vectors of x, y, and t values. Also zeros and ones.
        x = np.arange(-margin, margin + 1)
        y = x
        t = x
        (y, t, x) = np.meshgrid(y, t, x)
        x = x.flatten()
        y = y.flatten()
        t = t.flatten()
        zero = np.zeros(x.shape)
        one = np.ones(x.shape)

        # Create columns of A using all quadratic (or lower degree) combinations
        # of one, x, y, and t. Also construct derivatives of A.
        A = np.stack((one, x, y, t, x * y, x * t, y * t, x**2, y**2, t**2), axis=1)
        A_t  = np.stack((zero, zero, zero, one, zero, x, y, zero, zero, 2 * t), axis=1)
        A_x  = np.stack((zero, one, zero, zero, y, t, zero, 2 * x, zero, zero), axis=1)
        A_xx = np.stack((zero, zero, zero, zero, zero, zero, zero, 2 * one, zero, zero), axis=1)
        A_y  = np.stack((zero, zero, one, zero, x, zero, t, zero, 2 * y, zero), axis=1)
        A_yy = np.stack((zero, zero, zero, zero, zero, zero, zero, zero, 2 * one, zero), axis=1)

        # We're only interested in the derivatives evaluated at x = 0, y = 0,
        # and t = 0, so we can reduce the derivative matrices to just their
        # center row.
        midpoint = int((n**3) / 2)
        A_t  = A_t[midpoint, :]
        A_x  = A_x[midpoint, :]
        A_xx = A_xx[midpoint, :]
        A_y  = A_y[midpoint, :]
        A_yy = A_yy[midpoint, :]

        # Process each pixel in each frame of each video.
        for v in range(video_count):
            for t in range(margin, frame_count - margin):
                print(f'  v={v}, t={t}')
                for y in range(margin, y_count - margin):
                    for x in range(margin, x_count - margin):
                        u_block = u[v,
                            t - margin: t + margin + 1,
                            y - margin:y + margin + 1,
                            x - margin:x + margin + 1]
                        u_block = u_block.flatten()
                        coefs = np.linalg.lstsq(A, u_block, rcond=None)[0]
                        u_t[v, t - margin, y - margin, x - margin] = A_t.dot(coefs)
                        u_x[v, t - margin, y - margin, x - margin] = A_x.dot(coefs)
                        u_xx[v, t - margin, y - margin, x - margin] = A_xx.dot(coefs)
                        u_y[v, t - margin, y - margin, x - margin] = A_y.dot(coefs)
                        u_yy[v, t - margin, y - margin, x - margin] = A_yy.dot(coefs)

        return (u_t, u_x, u_xx, u_y, u_yy)
