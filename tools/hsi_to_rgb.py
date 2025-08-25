import numpy as np
from scipy.interpolate import CubicSpline
import os

import base64
from io import BytesIO
from PIL import Image

"""
Converts the hyperspectral image to an RGB image as a float32 numpy array. 
REQUIRES:
    cie_cmf.txt
        File that contains the eye response functions to one wavelength
        every 5 nm between 380 nm and 780 nm
INPUTS:
    datacube: np.array, shape: H*W*WL
        WL is the number of wavelengths in the 3rd dimension
    wl_cal: np.array, shape: WL
        wavelength calibration. It contains the wavelengths corresponding
        to each of the WL
RETURNS
    data_RGB_float32: numpy array, shape H*W*3
    The hyperspectral image converted to RGB.

"""


def hsi_to_rgb(datacube, wl_cal):  # enter the datacube and calibration
    base_path = os.path.dirname(__file__)
    cmf_path = os.path.join(base_path, "cie-cmf.txt")

    # Load eye response data
    cie_cmf = np.loadtxt(cmf_path)  # [wl R G B]

    # # load eye response data
    # cie_cmf = np.loadtxt(r"tools/cie-cmf.txt")  # [wl R G B]

    # convert data to functions
    responseR = CubicSpline(cie_cmf[:, 0], cie_cmf[:, 1])
    responseG = CubicSpline(cie_cmf[:, 0], cie_cmf[:, 2])
    responseB = CubicSpline(cie_cmf[:, 0], cie_cmf[:, 3])

    # finds the values of the color response functions at the corresponding wavelenghts
    wl_idx = (wl_cal > 380) & (wl_cal < 780)
    wl_cal = np.squeeze(wl_cal[wl_idx])  # only select the visible wavelengths
    delta_wl = (wl_cal[-1] - wl_cal[0]) / wl_cal.shape[0]

    proj_R = responseR(wl_cal)
    proj_G = responseG(wl_cal)
    proj_B = responseB(wl_cal)
    if len(datacube.shape) == 3:
        data_RGB = datacube.reshape(-1, datacube.shape[2])  # reshape to WxH, WL
        data_RGB = data_RGB[:, wl_idx]
    elif len(datacube.shape) == 2:
        data_RGB = datacube[:, wl_idx]
    elif len(datacube.shape) == 1:
        data_RGB = datacube[np.squeeze(wl_idx)]
    data_RGB = delta_wl * (
        data_RGB @ np.array([proj_R, proj_G, proj_B]).T
    )  # project each spectrum onto the responses to get color values

    # reshape from matrix to cube
    if len(datacube.shape) == 3:
        data_RGB = data_RGB.reshape(datacube.shape[0], datacube.shape[1], 3)
    elif len(datacube.shape) == 2:
        data_RGB = data_RGB.reshape(datacube.shape[0], 3)

    # normalization to 0-1 interval of the whole stack
    # data_RGB = (data_RGB - data_RGB.min()) / (data_RGB.max() - data_RGB.min())
    data_RGB = data_RGB / data_RGB.max()

    # clipping from 0 to 1
    data_RGB = np.clip(data_RGB, 0, 1)

    data_RGB_float32 = (data_RGB).astype(np.float32)
    return data_RGB_float32


def numpy_to_b64_img(arr):
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + encoded


def pil_image_to_base64(pil_img):
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + encoded
