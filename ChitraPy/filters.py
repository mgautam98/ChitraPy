import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import ChitraPy.helpers


@jit
def left_sobel(img):

    """
    Applies left sobel filter to a colored numpy image.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Filterd left sobel image.
    """

    kernel = np.array([[1., 0., -1.],
                   [2., 0., -2.],
                   [1., 0., -1.]])

    img = helpers.rgb2gray(img)

    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1))

    for ix in range(new_image.shape[0]):
        for iy in range(new_image.shape[1]):
            im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1]]
            h_prod = im_patch * kernel

            new_image[ix, iy] = max(0, h_prod.sum())
    new_image = new_image.astype(np.uint8)
    new_image = helpers.InvertGrayImg(new_image)
    return new_image


@jit
def right_sobel(img):

    """
    Applies right sobel filter to a colored numpy image.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Filterd right sobel image.

    """

    kernel = np.array([[-1., 0., 1.],
                   [-2., 0., 2.],
                   [-1., 0., 1.]])

    img = helpers.rgb2gray(img)

    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1))

    for ix in range(new_image.shape[0]):
        for iy in range(new_image.shape[1]):
            im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1]]
            h_prod = im_patch * kernel

            new_image[ix, iy] = max(0, h_prod.sum())
    new_image = new_image.astype(np.uint8)
    new_image = helpers.InvertGrayImg(new_image)
    return new_image


@jit
def top_sobel(img):

    """
    Applies top sobel filter to a colored numpy image.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Filterd top sobel image.
    """

    kernel = np.array([[1., 2., 1.],
                   [0., 0., 0.],
                   [-1., -2., -1.]])

    img = helpers.rgb2gray(img)

    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1))

    for ix in range(new_image.shape[0]):
        for iy in range(new_image.shape[1]):
            im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1]]
            h_prod = im_patch * kernel

            new_image[ix, iy] = max(0, h_prod.sum())
    new_image = new_image.astype(np.uint8)
    new_image = helpers.InvertGrayImg(new_image)
    return new_image


@jit
def gray_scale(img):
    """
    Applies gray scale filter to a colored numpy image.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Filterd gray scale image.
    """
    return np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)


@jit
def outline(img):

    """
    Applies Outline filter to a colored numpy image.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Filtered outlined image.
    """

    kernel = np.array([[-1., -1., -1.],
                   [-1., 8., -1.],
                   [-1., -1., -1.]])

    img = helpers.rgb2gray(img)

    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1))

    for ix in range(new_image.shape[0]):
        for iy in range(new_image.shape[1]):
            im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1]]
            h_prod = im_patch * kernel

            new_image[ix, iy] = max(0, h_prod.sum())
    new_image = new_image.astype(np.uint8)
    new_image = helpers.InvertGrayImg(new_image)
    return new_image


@jit(nopython=True)
def rotate(img):
    """
    Rotates the image by 90 degrees.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Rotated image.
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


@jit(nopython=True)
def invert(img):

    """
    Rotates the image by 180 degrees.

    Parameters:
    arg1 (np.array): numpy image matrix.

    Returns:
    np.array: Inverted image.
    """

    new_img = np.zeros_like(img)

    for ix in range(new_img.shape[0]):
        for iy in range(new_img.shape[1]):
            new_img[ix, iy] = img[img.shape[0]-ix-1, img.shape[1]-iy-1]
    return new_img



@jit(nopython=True)
def negative(img):

    """
    Negates the given image.

    Parameters:
    arg1 (np.array): numpy image matrix.

    Returns:
    np.array: Negative of the given image.
    """

    new_img = np.zeros_like(img)

    for ix in range(new_img.shape[0]):
        for iy in range(new_img.shape[1]):
            for iz in range(new_img.shape[2]):
                new_img[ix, iy, iz] = 255 - img[ix, iy, iz]

    return new_img.astype(np.uint8)



@jit(nopython=True)
def emboss(img):

    """
    Applies Emboss filter to a colored numpy image.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Filtered Embossed image.
    """

    kernel = np.array([[-2., -1., 0.],
                   [-1., 1., 1.],
                   [0., 1., 2.]])

    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1, 3))

    for iz in range(new_image.shape[2]):
        for ix in range(new_image.shape[0]):
            for iy in range(new_image.shape[1]):
                im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1], iz]
                h_prod = im_patch * kernel

                if h_prod.sum() < 0:
                    new_image[ix, iy, iz] = max(0, h_prod.sum())
                else:
                    new_image[ix, iy, iz] = min(255, h_prod.sum())

    return new_image.astype(np.uint8)



@jit(nopython=True)
def sharpen(img):

    """
    Applies Sharpen filter to a colored numpy image.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Filtered Sharpen image.
    """

    kernel = np.array([[0., -1., 0.],
                   [-1., 5. , -1.],
                   [0., -1., 0.]])

    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1, 3))

    for iz in range(new_image.shape[2]):
        for ix in range(new_image.shape[0]):
            for iy in range(new_image.shape[1]):
                im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1], iz]
                h_prod = im_patch * kernel

                if h_prod.sum() < 0:
                    new_image[ix, iy, iz] = max(0, h_prod.sum())
                else:
                    new_image[ix, iy, iz] = min(255, h_prod.sum())

    return new_image.astype(np.uint8)



@jit(nopython=True)
def sepia(img):

    """
    Applies sepia filter to the given image.

    Parameters:
    arg1 (np.array): numpy image matrix.

    Returns:
    np.array: Image with sepia filter applied.
    """

    new_img = np.zeros_like(img)

    for ix in range(new_img.shape[0]):
        for iy in range(new_img.shape[1]):
            new_img[ix, iy, 0] = min(255, 0.393*img[ix,iy,0] + 0.769*img[ix,iy,1] + 0.189*img[ix,iy,2])
            new_img[ix, iy, 1] = min(255, 0.349*img[ix,iy,0] + 0.686*img[ix,iy,1] + 0.168*img[ix,iy,2])
            new_img[ix, iy, 2] = min(255, 0.272*img[ix,iy,0] + 0.534*img[ix,iy,1] + 0.131*img[ix,iy,2])

    return new_img.astype(np.uint8)



@jit(nopython=True)
def identity(img):

    """
    Applies identity filter to a colored numpy image.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Identity image.
    """

    kernel = np.array([[0., 0., 0.],
                   [0., 1. , 0.],
                   [0., 0., 0.]])

    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1, 3))

    for iz in range(new_image.shape[2]):
        for ix in range(new_image.shape[0]):
            for iy in range(new_image.shape[1]):
                im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1], iz]
                h_prod = im_patch * kernel
                new_image[ix, iy, iz] = max(0, h_prod.sum())

    return new_image.astype(np.uint8)



@jit(nopython=True)
def crop(img, top_left , bottom_right):

    """
    Crops the image to given dimension.

    Parameters:
    arg1 (np.array): Numpy image matrix.
    arg2 (np.array): Top left corner co-ordinates.
    arg3 (np.array): Top right corner co-ordinates.

    Returns:
    np.array: Cropped image.
    """

    if(top_left[0]<0 or top_left[1]<0 or bottom_right[0]<0 or bottom_right[1]<0):
        print("\nCan not Crop!")
        return img

    if(bottom_right[0] > img.shape[0] or bottom_right[1] > img.shape[1] or top_left[0] > bottom_right[0] or top_left[1] > bottom_right[1]):
        print("\nCan not Crop!")
        return img

    new_img = np.zeros((bottom_right[0] - top_left[0], bottom_right[1] - top_left[1], 3))
    for ix in range(new_img.shape[0]):
        for iy in range(new_img.shape[1]):
            new_img[ix, iy, :] = img[ix + top_left[0], iy + top_left[1], :]

    return new_img.astype(np.uint8)



@jit(nopython=True)
def quick_blur(img):
    """
    Blurs the  image

    Parameters:
    arg1 (np.array): numpy image matrix

    Returns:
    np.array: Gray scale numpy image matrix
    """
    kernel=np.array([[0.0625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.125,0.0625]])
    new_image = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1, 3))

    for iz in range(new_image.shape[2]):
        for ix in range(new_image.shape[0]):
            for iy in range(new_image.shape[1]):
                    im_patch = img[ix:ix+kernel.shape[0], iy:iy+kernel.shape[1],iz]
                    h_prod = im_patch * kernel
                    new_image[ix, iy,iz] = max(0, h_prod.sum())
    new_image = new_image.astype(np.uint8)
    return new_image



@jit
def monochrome(img):

    """
    Applies monochrome filter to a colored numpy image.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Monochrome applied image.
    """
    img = gray_scale(img)
    img = img.astype(np.uint8)
    new_img = np.zeros_like(img)

    for ix in range(new_img.shape[0]):
        for iy in range(new_img.shape[1]):
            if(img[ix, iy]>128):
                new_img[ix, iy] = 255
            else:
                new_img[ix, iy] = 0

    return new_img.astype(np.uint8)



@jit(nopython=True)
def sliding_contrast(img, percentage = 0):

    """
    Changes contrast of the image.

    Parameters:
    arg1 (np.array): Numpy image matrix.
    arg2 (int): Percentage of contrast to change -100 to 100.

    Returns:
    np.array: Monochrome applied image.
    """

    new_img = np.zeros_like(img)

    for iz in range(new_img.shape[2]):
        for ix in range(new_img.shape[0]):
            for iy in range(new_img.shape[1]):
                if percentage>0:
                    new_img[ix, iy, iz] = min(255, img[ix, iy, iz] + percentage*2.55)
                else:
                    new_img[ix, iy, iz] = max(0, img[ix, iy, iz] + percentage*2.55)

    return new_img.astype(np.uint8)



@jit
def dither(img):

    """
    Applies dither filter to a colored numpy image.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Dithered image.
    """
    img = helpers.rgb2gray(img)

    pixel = np.copy(img)
    w, h = img.shape

    for iy in range(h-1):
        for ix in range(1,w-1):
            oldpixel = pixel[ix][iy]
            newpixel = helpers.find_closest_palette_color(oldpixel)
            pixel[ix][iy] = newpixel
            err = oldpixel - newpixel
            if ix+1 < w:
                pixel[ix+1][iy] += err*(7./16)
            if ix-1 > 0 & iy+1 < h:
                pixel[ix-1][iy+1] += err*(3./16)
            if iy+1 < h:
                pixel[ix][iy+1] += err*(5./16)
            if ix+1 < w:
                pixel[ix+1][iy+1] += err*(1./16)

    return pixel



def contrast_enhancement(img):

    """
    Enhances contrast of an image using histogram equalization.

    Parameters:
    arg1 (np.array): Numpy image matrix.

    Returns:
    np.array: Contrast enhanced gray image.
    """

    try:
        if img.shape[2]:
            img = gray_scale(img)
    except:
        pass

    pmf, cdf = helpers.get_pmf_cdf(img)

    new_dic = cdf.copy()

    for i in range(256):
        new_dic[i]*=255
        new_dic[i] = np.floor(new_dic[i])

    for ix in range(img.shape[0]):
        for iy in range(img.shape[1]):
            img[ix, iy] = new_dic[img[ix, iy]]

    return img
