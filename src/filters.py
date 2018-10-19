import numpy as np
import matplotlib.pyplot as plt
import utils.utils as utils


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

    img = utils.rgb2gray(img)

    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1))

    for ix in range(new_image.shape[0]):
        for iy in range(new_image.shape[1]):
            im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1]]
            h_prod = im_patch * kernel

            new_image[ix, iy] = max(0, h_prod.sum())
    new_image = new_image.astype(np.uint8)
    new_image = utils.InvertGrayImg(new_image)
    return new_image


def right_sobel(img):

    """
    Applies right sobel filter to a colored numpy image

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: filterd right sobel image
    """

    kernel = np.array([[-1., 0., 1.],
                   [-2., 0., 2.],
                   [-1., 0., 1.]])

    img = utils.rgb2gray(img)

    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1))

    for ix in range(new_image.shape[0]):
        for iy in range(new_image.shape[1]):
            im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1]]
            h_prod = im_patch * kernel

            new_image[ix, iy] = max(0, h_prod.sum())
    new_image = new_image.astype(np.uint8)
    new_image = utils.InvertGrayImg(new_image)
    return new_image


def top_sobel(img):

    """
    Applies top sobel filter to a colored numpy image

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: filterd top sobel image
    """

    kernel = np.array([[1., 2., 1.],
                   [0., 0., 0.],
                   [-1., -2., -1.]])

    img = utils.rgb2gray(img)

    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1))

    for ix in range(new_image.shape[0]):
        for iy in range(new_image.shape[1]):
            im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1]]
            h_prod = im_patch * kernel

            new_image[ix, iy] = max(0, h_prod.sum())
    new_image = new_image.astype(np.uint8)
    new_image = utils.InvertGrayImg(new_image)
    return new_image


def gray_scale(img):
    """
    Applies gray scale filter to a colored numpy image

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: filterd gray scale image
    """
    return utils.rgb2gray(img)


def outline(img):

    """
    Applies Outline filter to a colored numpy image

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: filtered outlined image
    """

    kernel = np.array([[-1., -1., -1.],
                   [-1., 8., -1.],
                   [-1., -1., -1.]])

    img = utils.rgb2gray(img)

    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1))

    for ix in range(new_image.shape[0]):
        for iy in range(new_image.shape[1]):
            im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1]]
            h_prod = im_patch * kernel

            new_image[ix, iy] = max(0, h_prod.sum())
    new_image = new_image.astype(np.uint8)
    new_image = utils.InvertGrayImg(new_image)
    return new_image


def rotate(img):
    """
    Rotates the image by 90 degrees

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: filterd gray scale image
    """
    new_image=np.zeros_like(img)
    new_image=new_image.reshape(img.shape[1],img.shape[0],3)
    imgr=img[:,:,0]
    imgg=img[:,:,1]
    imgb=img[:,:,2]
    imgr=np.transpose(imgr)
    imgg=np.transpose(imgg)
    imgb=np.transpose(imgb)
    new_image[:,:,0]=imgr
    new_image[:,:,1]=imgg
    new_image[:,:,2]=imgb
    return new_image



def invert(img):

    """
    Rotates the image by 180 degrees

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: filterd gray scale image
    """

    new_img = np.zeros_like(img)

    for ix in range(new_img.shape[0]):
        for iy in range(new_img.shape[1]):
            new_img[ix, iy] = img[img.shape[0]-ix-1, img.shape[1]-iy-1]
    return new_img
