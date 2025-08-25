import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter


def denoise_spectra(cube, median_window=3, sg_window=11, sg_polyorder=2):
    """
    Apply median and Savitzky–Golay filters along the spectral axis of a hyperspectral cube.

    Parameters
    ----------
    cube : np.ndarray
        Input datacube of shape (H, W, B).
    median_window : int
        Size of the median filter window (must be odd).
    sg_window : int
        Window length of the Savitzky–Golay filter (must be odd and > sg_polyorder).
    sg_polyorder : int
        Polynomial order for the Savitzky–Golay filter.

    Returns
    -------
    denoised_cube : np.ndarray
        Output denoised datacube of same shape.
    """
    H, W, B = cube.shape

    # Apply median filter along the spectral axis (axis=2)
    cube_median = median_filter(cube, size=(1, 1, median_window))

    # Apply Savitzky-Golay filter along the spectral axis
    cube_sg = savgol_filter(
        cube_median, window_length=sg_window, polyorder=sg_polyorder, axis=2
    )

    return cube_sg
