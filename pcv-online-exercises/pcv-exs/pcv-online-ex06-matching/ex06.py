#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jan 04 19:19:15 2022

@author: Nacriema

Refs:

This script contains all functions that I use for doing task in Jupyter note
"""
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from numpy import unravel_index


def standard_deviation(image_patch):
    """
    Compute the standard deviation of intensity values given by an image patch

    Parameters
    ----------
    image_patch: np ndarray

    Returns
    -------
    SD: scalar
    """
    patch_mean = np.mean(image_patch)
    rs = np.mean((image_patch - patch_mean)**2)
    return np.sqrt(rs)


def intensity_covariance(original_patch, query_patch):
    """
    Compute the Covariance between intensity values of original patch of image at overlapped when stride the query patch
    on.

    Parameters
    ----------
    original_patch: np ndarray
    query_patch: np ndarray

    Returns
    -------
    a scalar
    """
    assert original_patch.shape == query_patch.shape

    original_mean = np.mean(original_patch)
    query_mean = np.mean(query_patch)
    rs = np.mean(np.multiply((original_patch - original_mean), (query_patch - query_mean)))
    return rs


def normalized_cross_correlation(source_patch, query_patch, epsilon=1e-9):
    """
    Parameters
    ----------
    epsilon
    source_patch
    query_patch

    Returns
    -------
    """
    assert source_patch.shape == query_patch.shape
    numerator = intensity_covariance(source_patch, query_patch)
    denominator = standard_deviation(query_patch) * standard_deviation(source_patch)
    if numerator == 0:
        return 0
    return (numerator + epsilon) / (denominator + epsilon)


def imageCorrelation(image, patch, epsilon=1e-7):
    """
    Computes the correlation between an image and a patch. Notice, I just compute the correlation values only for the
    pixels where the neighborhood is well-defined.

    Based on the normalized cross correlation in the Lecture Note.

    Parameters
    ----------
    epsilon
    image: a full original image
    patch: a query patch image

    Returns
    -------
    all positions denote by [u, v] where the location of the query patch in the original image
    """
    rs = np.zeros_like(image)
    im_r, im_c = image.shape
    patch_r, patch_c = patch.shape
    patch_sd = standard_deviation(patch)

    for i in range(0, im_r - patch_r + 1):
        for j in range(0, im_c - patch_c + 1):
            numerator = intensity_covariance(image[i: i + patch_r, j: j + patch_c], patch)
            denominator = patch_sd * standard_deviation(image[i: i + patch_r, j: j + patch_c])
            if numerator == 0:
                rs[i, j] = 0
            else:
                rs[i, j] = (numerator + epsilon) / (denominator + epsilon)   # Use epsilon to avoid numerical

    # Return just for the region where the pixel is well-defined
    return rs


def imageCorrelation_2(image, patch):
    """
    This is the update version of the first one, I rotate the array before doing the correlation

    Computes the correlation between an image and a patch. Notice, I just compute the correlation values only for the
    pixels where the neighborhood is well-defined.

    Based on the normalized cross correlation in the Lecture Note.

    Parameters
    ----------
    epsilon
    image: a full original image
    patch: a query patch image

    Returns
    -------
    all positions denote by [u, v] where the location of the query patch in the original image
    """
    rs = np.zeros_like(image)
    im_r, im_c = image.shape
    patch_r, patch_c = patch.shape

    for i in range(0, im_r - patch_r + 1):
        for j in range(0, im_c - patch_c + 1):
            arr = [normalized_cross_correlation(image[i: i + patch_r, j: j + patch_c], patch),
                   normalized_cross_correlation(image[i: i + patch_r, j: j + patch_c], np.rot90(patch))]
            rs[i, j] = max(arr)
    # Return just for the region where the pixel is well-defined
    return rs


if __name__ == '__main__':
    img = skimage.io.imread('./images/text.png', as_gray=True)
    letter_a = img[13:22, 66:75]
    rs = imageCorrelation_2(img, letter_a)

    # Apply threshold
    # See: https://stackoverflow.com/questions/17568612/how-to-make-numpy-argmax-return-all-occurrences-of-the-maximum

    rs = rs > 0.83
    print(np.argwhere(rs == np.amax(rs)))
    print(unravel_index(rs.argmax(), rs.shape))
    print(normalized_cross_correlation(letter_a, letter_a))
    print(normalized_cross_correlation(letter_a, np.zeros_like(letter_a)))
    plt.imshow(rs, cmap='jet')
    plt.show()


