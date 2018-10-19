import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(img):
    """
    Converts a given numpy array colored image to gray image

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: Gray scale numpy image matrix
    """
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def display_gray(img):
    """
    Displays only single channel of a numpy image matrix

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: Gray scale numpy image matrix
    """
    plt.imshow(img, cmap = plt.get_cmap('gray'))

    return

def InvertGrayImg(img):
    """
    Inverts a single channel image

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: Gray scale numpy image matrix
    """

    for ix in range(img.shape[0]):
        for iy in range(img.shape[1]):
            pixel = img[ix, iy]
            img[ix, iy] = 255 - pixel
    return img
