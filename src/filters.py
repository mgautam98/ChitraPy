import numpy as np
import matplotlib.pyplot as plt
from utils.utils import rgb2gray


def left_sobel(img):

    """
    Applies left sobel filter to a colored numpy image

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: filterd left sobel image
    """

    kernel = np.array([[1., 0., -1.],
                   [2., 0., -2.],
                   [1., 0., -1.]])

    img = rgb2gray(img)

    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1))

    for ix in range(new_image.shape[0]):
        for iy in range(new_image.shape[1]):
            im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1]]
            h_prod = im_patch * kernel

            new_image[ix, iy] = max(0, h_prod.sum())
    return new_image
