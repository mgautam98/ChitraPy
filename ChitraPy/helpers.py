import numpy as np
from numba import jit
import matplotlib.pyplot as plt


@jit
def rgb2gray(img):
    """
    Converts a given numpy array colored image to gray image

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: Gray scale numpy image matrix
    """
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])



@jit
def display(img):
    """
    Displays only single channel/ three channel of a numpy image matrix

    Parameters:
    arg1 (np.array): numpy image matrix

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



@jit(nopython=True)
def find_closest_palette_color(oldpixel):

    """
    To find the closest palette color.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Closet palette color.
    """

    return round(oldpixel/255)*255



@jit
def get_pmf_cdf(img, display=False):

    """
    Calculates PMF and CDF for an image.

    Parameters:
    arg1 (np.array): Numpy image matrix.
    bool : display, default it is False.

    Returns:
    Dictionary : PMF and CDF.
    """

    pmf = {}

    for i in range(256):
        pmf[i] = 0

    for ix in range(img.shape[0]):
        for iy in range(img.shape[1]):
            pmf[img[ix, iy]] += 1

    total_pix = img.shape[0]*img.shape[1]

    for i in range(256):
        pmf[i] /=total_pix

    cdf = pmf.copy()
    for i in range(1, 256):
        cdf[i] += cdf[i-1]

    #for displaying
    if(display):
        f, axarr = plt.subplots(2, sharex=True)
        f.suptitle('PMF and CDF')
        axarr[0].plot(pmf.keys(), pmf.values())
        axarr[1].plot(cdf.keys(), cdf.values())

    return pmf, cdf


def save(img, name = "image"):

    """
    Saves an image in .jpg format in current working directory.

    Parameters:
    arg1 (np.array): Numpy image matrix.
    arg2 (String) : Name of image. Default it is image.

    """

    import os
    i = 1
    if os.path.exists(name + ".jpg") :
        while(os.path.exists(name + "(" + str(i) + ")" + ".jpg")):
            i +=1

        name+= "(" + str(i) + ")"
    plt.imsave(name + ".jpg", img, cmap = plt.get_cmap('gray'))
