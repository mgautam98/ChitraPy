import numpy as np
from numba import jit
import matplotlib.pyplot as plt


@jit(nopython=True)
def rgb2gray(img):
    """
    Converts a given numpy array colored image to gray image

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: Gray scale numpy image matrix
    """
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])



@jit(nopython=True)
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



@jit(nopython=True)
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



@jit(nopython=True)
def histogram(img):

    """
    Plots intensity plots for a color image.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    """

    valr = []
    valg = []
    valb = []
    bins = [i for i in range(256)]
    for ix in range(img.shape[0]):
        for iy in range(img.shape[1]):
            valr.append(img[ix, iy, 0])
            valg.append(img[ix, iy, 1])
            valb.append(img[ix, iy, 2])

    f, axarr = plt.subplots(3, sharex=True)
    f.suptitle('Intensity Plots')
    axarr[0].hist(valr, bins=bins, color = 'red')
    axarr[1].hist(valg, bins=bins, color = 'green')
    axarr[2].hist(valb, bins=bins, color = 'blue')
