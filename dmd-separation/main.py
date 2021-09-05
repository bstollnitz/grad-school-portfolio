from typing import List, Tuple

import numpy as np

import utils_io
import utils_video
import utils_graph


S3_URL = 'https://bea-portfolio.s3-us-west-2.amazonaws.com/dmd-separation/'
DATA_FOLDER = 'data'
PLOTS_FOLDER = 'plots'
VIDEO_FILENAMES = ['elephant.mp4', 'lion.mp4', 'monkey-giraffe.mp4']


def download_videos() -> None:
    """Downloads videos from remote location if they're not yet 
    present locally.
    """
    for item in VIDEO_FILENAMES:
        utils_io.download_remote_data_file(DATA_FOLDER, S3_URL, item)


def _plot_complex(eigenvalues: np.ndarray, variable_name: str, 
    filename: str) -> None:
    """Plots complex mu.
    """
    print('Plotting eigenvalues...')
    real = np.real(eigenvalues)
    imaginary = np.imag(eigenvalues)

    utils_graph.graph_2d_markers(real, imaginary, f'Real({variable_name})', 
        f'Imaginary({variable_name})', 
        f'Complex values of {variable_name}', PLOTS_FOLDER, 
        f'{filename}_{variable_name}.html')


def _perform_dmd(video_matrix: np.ndarray, frame_rate: 
    float, filename: str) -> Tuple[np.ndarray, np.ndarray, 
    np.ndarray]:
    """Performs DMD computations. Returns the DMD eigenvalues, eigenvectors
    and initial coefficients.
    """
    print('Performing DMD computations...')
    # Perform DMD computations.
    slices = video_matrix.shape[1]
    video_matrix_1 = video_matrix[:, :-1]
    video_matrix_2 = video_matrix[:, 1:]
    (u, sigma, vh) = np.linalg.svd(video_matrix_1, full_matrices=False)
    uh = u.conj().T
    v = vh.conj().T
    s = uh.dot(video_matrix_2).dot(v).dot(np.diag(1/sigma))
    (eig_values, eig_vectors) = np.linalg.eig(s)
    mu = eig_values
    # mu = mu / np.maximum(np.abs(mu), 1)
    
    # omega contains the DMD eigenvalues.
    # phi contains the DMD eigenvectors (modes).
    # b contains the initial coefficients.
    dt = 1
    omega = np.log(mu) / dt
    psi = u.dot(eig_vectors)
    b = np.linalg.pinv(psi).dot(video_matrix[:, 0])

    # Plot the complex mu and omega.
    _plot_complex(mu, 'μ', filename)
    _plot_complex(omega, 'ω', filename)

    return (omega, psi, b)


def _reconstruct_video(omega: np.ndarray, psi: np.ndarray, 
    b: np.ndarray) -> np.ndarray:
    """Reconstruct a video using the DMD eigenvalues (omega), 
    eigenvectors (psi) and intial coefficients (b).
    """
    print('Reconstructing video...')
    pixel_count = psi.shape[0]
    frame_count = psi.shape[1]
    reconstructed_video = np.zeros((pixel_count, frame_count), dtype=complex)
    for frame in range(frame_count):
        reconstructed_video[:, frame] = psi.dot(b * np.exp(omega * frame))

    return reconstructed_video


def _save_video(video: np.ndarray, width: int, height: int, 
    filename: str) -> None:
    """Saves a video.
    """
    video_shape = (width, height, video.shape[1])
    reshaped_video = np.real(np.reshape(video, video_shape))

    utils_video.save_video(PLOTS_FOLDER, f'{filename}.avi', 
        reshaped_video)


def _get_background(omega: np.ndarray, psi: np.ndarray, 
    b:np.ndarray) -> np.ndarray:
    """Uses DMD to get background.
    """
    index_background = [np.argmin(np.abs(omega))]
    print(f'Using mode {index_background} for background.')
    omega_background = omega[index_background]
    psi_background = psi[:, index_background]
    b_background = b[index_background]
    # This background reconstruction is perfect. It's just 1 frame.
    background = _reconstruct_video(omega_background,
        psi_background, b_background)

    print(f'Range of imaginary part of background: {np.min(background.imag)}' +
        f' to {np.max(background.imag)}')
    
    return background


def _split_video_1(original_video: np.ndarray, omega: np.ndarray,
    psi: np.ndarray, b: np.ndarray,
    width: int, height: int, filename: str) -> None:
    """Splits foreground and background from video.
    """
    print('Splitting foreground and background of video (1)...')

    # This is the perfect one frame background.
    background = _get_background(omega, psi, b)

    # We subtract the background from the original video, and we get a video
    # with black background, positive pixels where the foreground is brighter
    # than the background, and negative pixels where the foreground is darker.
    foreground = original_video - np.abs(background)

    # We can't display negative pixels, so we clamp the foreground to zero.
    # We do this by subtracting the negative residual (or adding the positive 
    # residual) to the negative pixels in the foreground.
    # The resulting foreground is black on every pixel that was darker than the
    # background, and it's the actual foreground on every pixel that was 
    # lighter than the background.
    residual = np.minimum(foreground, 0)
    foreground = foreground - residual

    # We add the (negative) residuals back to the background. This darkens
    # every background pixel where the original image was darker than the 
    # background, essentially transfering the darker parts of the foreground 
    # onto the background.
    background = residual + np.abs(background)

    # The resulting foreground video has black pixels except where the original
    # foreground is lighter than the original background.
    _save_video(foreground, width, height, f'{filename}_1_foreground')

    # The result background video has the actual background plus dark pixels
    # where the original foreground is darker than the original background.
    _save_video(background, width, height, f'{filename}_1_background')


def _split_video_2(original_video: np.ndarray, omega: np.ndarray,
    psi: np.ndarray, b: np.ndarray, 
    width: int, height: int, filename: str) -> None:
    """Splits foreground and background from video.
    """
    print('Splitting foreground and background of video (2)...')

    # This is the perfect one frame background.
    background = _get_background(omega, psi, b)

    # foreground is black where the original video matches the background, 
    # and it takes the values of the original video elsewhere.
    foreground = np.where(np.abs(background - original_video) < 0.1, 0, 
        original_video)

    _save_video(foreground, width, height, f'{filename}_2_foreground')
    _save_video(background, width, height, f'{filename}_2_background')


def _process_video(filename: str) -> None:
    """Processes a single video.
    """
    # Load video. 
    (video_matrix, frame_rate) = utils_video.load_video(DATA_FOLDER, filename)
    filename = filename.split('.')[0]

    # Original video shape: (width x height x frames) = (640 x 360 x 360).
    (width, height, frame_count) = video_matrix.shape

    # Reshaped video shape: (width*height x frames) = (230400, 360).
    video_matrix = np.reshape(video_matrix, (-1, frame_count))

    # Save original video.
    # _save_video(video_matrix, width, height, f'original_{filename}')

    # Get DMD eigenvalues, eigenvectors and initial coefficients.
    (omega, psi, b) = _perform_dmd(video_matrix, frame_rate, filename)

    # Reconstruct original video using DMD modes.
    # This reconstruction is not accurate at all. Many pixels are clamped
    # to black or white.
    reconstructed_video = _reconstruct_video(omega, psi, b)
    _save_video(reconstructed_video, width, height, 
        f'{filename}_reconstructed')

    # Split foreground and background in 2 ways.
    _split_video_1(video_matrix, omega, psi, b, width, height, filename)
    _split_video_2(video_matrix, omega, psi, b, width, height, filename)


def process_videos() -> None:
    """Processes all videos.
    """
    for item in VIDEO_FILENAMES:
        _process_video(item)


def main() -> None:
    """Main program.
    """
    utils_io.find_or_create_dir(DATA_FOLDER)
    utils_io.find_or_create_dir(PLOTS_FOLDER)
    download_videos()
    process_videos()


if __name__ == '__main__':
    main()
