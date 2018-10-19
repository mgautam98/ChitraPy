import numpy as np

def rgb2gray(img):
    """
    Converts a given numpy array colored image to gray image

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: Gray scale numpy image matrix
    """
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])
