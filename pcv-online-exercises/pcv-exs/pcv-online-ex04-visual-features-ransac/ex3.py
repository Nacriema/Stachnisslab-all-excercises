#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
from scipy import ndimage, misc, signal

from skimage.color import rgb2gray


def compute_corners(I, type='harris', T=0.2):
    """
    Compute corner keypoints using Harris and Shi-Tomasi criteria
    See Cyrill's Lecture: https://www.youtube.com/watch?v=nGya59Je4Bs&list=PLgnQpQtFTOGTPQhKBOGgjTgX-mzpsOGOX&index=3
    See The Best Explanation for SSD:  https://www.youtube.com/watch?v=_qgKQGsuKeQ
    See The Best Implementation, but in this post, there are some mistake: https://viblo.asia/p/ung-dung-thuat-toan-harris-corner-detector-trong-bai-toan-noi-anh-phan-i-ByEZkyME5Q0
    Parameters
    ----------
    I : float [MxN] 
        grayscale image

    type :  string
            corner type ('harris' or 'Shi-Tomasi')

    T:  float
        threshold for corner detection
    
    Returns
    -------
    corners : numpy array [num_corners x 2] 
              Coordinates of the detected corners.    
    """
    # TODO: Implement the feature corner detection
    # 1. Compute horizontal and vertical derivatives of images, use scharr or sobel kernel
    scharr = 1 / 32. * np.array([[3 + 3j, 10 + 0j, 3 - 3j],
                                 [0 + 10j, 0 + 0j, 0 - 10j],
                                 [-3 + 3j, -10 + 0j, -3 - 3j]])

    grad = signal.convolve2d(ascent, scharr, boundary='symm', mode='same')
    grad_x = grad.real
    grad_y = grad.imag

    # 2. Compute three images corresponding to three terms in matrix M, use element-wise multiply
    grad_x_square = np.multiply(grad_x, grad_x)
    grad_y_square = np.multiply(grad_y, grad_y)
    grad_x_mul_grad_y = np.multiply(grad_x, grad_y)

    # 3. Convolve these three images with a larger Gaussian (window), say I choose window size = 5
    # See: https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
    grad_x_square = ndimage.gaussian_filter(grad_x_square, sigma=0.8, truncate=3)
    grad_y_square = ndimage.gaussian_filter(grad_y_square, sigma=0.8, truncate=3)
    grad_x_mul_grad_y = ndimage.gaussian_filter(grad_x_mul_grad_y, sigma=0.8, truncate=3)

    # 4. Compute scalar cornerness value use one of the R measures. I choose R = det(M) - k * (trace(M))^2,
    # k in range [0.04, 0.06]

    # M = |I_x I_x    I_x I_y|
    #     |I_x I_y    I_y I_y|

    det_M = np.multiply(grad_x_square, grad_y_square) - np.multiply(grad_x_mul_grad_y, grad_x_mul_grad_y)
    trace_M = grad_x_square + grad_y_square
    R = det_M - 0.05 * np.multiply(trace_M, trace_M)

    # 5. Apply threshold on R to get the corner, where
    # R approx 0: Flat region
    # R < 0: Edge
    # R >> 0: Corner
    R[R.max() / 500. >= R] = 0

    # 6. Find local maxima above some threshold as detected interest point
    # See a smart way to do it: https://stackoverflow.com/questions/38179797/numpy-max-pooling-convolution
    R_ = R * (R == ndimage.maximum_filter(R, size=5))

    # 7. Return the coordinate of corner in R_
    # See https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
    corners = np.argwhere(R_ != 0)

    # I return more result than the original function (for visualization purpose)
    return corners, R_


def compute_descriptors(I, corners):
    """
    Computes a 128 bit descriptor as described in the lecture.
    Parameters
    ----------
    I : float [MxN]
        grayscale image as a 2D numpy array
    corners : numpy array [num_corners x 2] 
              Coordinates of the detected corners. 
    
    Returns
    -------
    D : numpy array [num_corners x 128]
        128 bit descriptors  corresponding to each corner keypoint

    """

    # TODO: Implement descriptor computation for each corner feature keypoint

    return D


def compute_matches(D1, D2):
    """
    Computes matches for two images using the descriptors.
    Uses the Lowe's criterea to determine the best match.

    Parameters
    ----------
    D1 : numpy array [num_corners x 128]
         descriptors for image 1 corners
    D2 : numpy array [num_corners x 128]
         descriptors for image 2 corners
 
    Returns
    ----------
    M : numpy array [num_matches x 2]
        [cornerIdx1, cornerIdx2] each row contains indices of corresponding keypoints 
    """

    return M


def plot_matches(I1, I2, C1, C2, M):
    """ 
    Plots the matches between the two images
    """
    # Create a new image with containing both images
    W = I1.shape[1] + I2.shape[1]
    H = np.max([I1.shape[0], I2.shape[0]])
    I_new = np.zeros((H, W), dtype=np.uint8)
    I_new[0:I1.shape[0], 0:I1.shape[1]] = I1
    I_new[0:I2.shape[0], I1.shape[1]:I1.shape[1] + I2.shape[1]] = I2

    # plot matches
    plt.imshow(I_new, cmap='gray')
    for i in range(M.shape[0]):
        p1 = C1[M[i, 0], :]
        p2 = C2[M[i, 1], :] + np.array([I1.shape[1], 0])
        plt.plot(p1[0], p1[1], 'rx')
        plt.plot(p2[0], p2[1], 'rx')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-y')


def compute_homography_ransac(C1, C2, M):
    """
    Implements a RANSAC scheme to estimate the homography and the set of inliers.

    Parameters
    ----------
    C1 : numpy array [num_corners x 2]
         corner keypoints for image 1 
    C2 : numpy array [num_corners x 2]
         corner keypoints for image 2
    M  : numpy array [num_matches x 2]
        [cornerIdx1, cornerIdx2] each row contains indices of corresponding keypoints 
 
    Returns
    ----------
    H_final : numpy array [3 x 3]
            Homography matrix which maps in point image 1 to image 2 
    M_final : numpy array [num_inlier_matches x 2]
            [cornerIdx1, cornerIdx2] each row contains indices of inlier matches
    """

    max_iter = 100
    min_inlier_ratio = 0.6
    inlier_thres = 5

    H_final = None
    M_final = None

    # TODO: Complete RANSAC algorithm to estimate the best estimaton

    return H_final, M_final


# Calculate the geometric distance between estimated points and original points, namely residuals.
def compute_residual(P1, P2, H):
    """
    Compute the residual given the Homography H
    
    Parameters
    ----------
    P1: [num_points x 2]
        Points (x,y) from Image 1. The keypoint in the ith row of P1
        corresponds to the keypoint ith row of P2
    P2: [num_points x 2]
        Points (x,y) from Image 2
    H: [numpy array 3x3]
        Homography which maps P1 to P2
    
    Returns:
    ----------
    residual : float
                residual computed for the corresponding points P1 and P2 
                under the transformation given by H
    """

    # TODO: Compute residual given Homography H

    return residual


def calculate_homography_four_matches(P1, P2):
    """
    Estimate the homography given four correspondening keypoints in the two images.

    Parameters
    ----------
    P1: [num_points x 2]
        Points (x,y) from Image 1. The keypoint in the ith row of P1
        corresponds to the keypoint ith row of P2
    P2: [num_points x 2]
        Points (x,y) from Image 2

    Returns:
    ----------
    H: [numpy array 3x3]
        Homography which maps P1 to P2 based on the four corresponding points
    """

    if P1.shape[0] or P2.shape[0] != 4:
        print('Four corresponding points needed to compute Homography')
        return None

    # loop through correspondences and create assemble matrix
    # A * h = 0, where A(2n,9), h(9,1)

    A = []
    for i in range(P1.shape[0]):
        p1 = np.array([P1[i, 0], P1[i, 1], 1])
        p2 = np.array([P2[i, 0], P2[i, 1], 1])

        a2 = [
            0, 0, 0, -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2],
                     p2[1] * p1[0], p2[1] * p1[1], p2[1] * p1[2]
        ]
        a1 = [
            -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2], 0, 0, 0,
            p2[0] * p1[0], p2[0] * p1[1], p2[0] * p1[2]
        ]
        A.append(a1)
        A.append(a2)

    A = np.array(A)

    # svd composition
    # the singular value is sorted in descending order
    u, s, v = np.linalg.svd(A)

    # we take the “right singular vector” (a column from V )
    # which corresponds to the smallest singular value
    H = np.reshape(v[8], (3, 3))

    # normalize and now we have H
    H = (1 / H[2, 2]) * H

    return H


if __name__ == '__main__':
    fig = plt.figure()
    plt.gray()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # ascent = misc.ascent()
    ascent = imageio.imread('checkerboard.png')
    ascent = rgb2gray(ascent)
    corners, img = compute_corners(I=ascent)
    print(corners.shape)
    ax1.imshow(ascent)
    ax2.imshow(img, cmap='jet')
    plt.show()
