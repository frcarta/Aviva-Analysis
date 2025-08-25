import numpy as np


def normalize_np(arr):
    """
    Normalizes the values of an array such that the max = 1 and min = 0

    Input:
    -------------
    arr: np array

    Output:
    -----------------
    normalized array with same type as input
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)
