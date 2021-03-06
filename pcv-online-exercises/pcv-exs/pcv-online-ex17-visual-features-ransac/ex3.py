#!/usr/bin/env python3
from builtins import print

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import skimage.transform
import imutils
from scipy import ndimage, misc, signal
from numpy.linalg import norm, inv
import random

from skimage.color import rgb2gray


# DONE
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
            corner type ('harris' or 'shi-tomasi')

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

    grad = signal.convolve2d(I, scharr, boundary='symm', mode='same')
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
    # k in range [0.04, 0.06], I choose k=0.05

    # M = |I_x I_x    I_x I_y|
    #     |I_x I_y    I_y I_y|

    det_M = np.multiply(grad_x_square, grad_y_square) - np.multiply(grad_x_mul_grad_y, grad_x_mul_grad_y)
    trace_M = grad_x_square + grad_y_square

    if type == 'harris':
        R = det_M - 0.05 * (trace_M ** 2)

        # 5. Apply threshold on R to get the corner, where
        # R approx 0: Flat region
        # R < 0: Edge
        # R >> 0: Corner
        R[R.max() * T >= R] = 0

        # 6. Find local maxima above some threshold as detected interest point
        # See a smart way to do it: https://stackoverflow.com/questions/38179797/numpy-max-pooling-convolution
        R_ = R * (R == ndimage.maximum_filter(R, size=5))

        # 7. Return the coordinate of corners in R_
        # See https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
        corners = np.argwhere(R_ != 0)

    elif type == 'shi-tomasi':
        # 8. Find Lambda min, for simplify I reuse the variable R to store array of lambda min
        R = trace_M / 2.0 - 1 / 2. * np.sqrt(trace_M ** 2 - 4 * det_M)
        # 9. Apply thresh hold
        R[R.max() * T >= R] = 0
        # 10. Apply non-maxima suppression
        R_ = R * (R == ndimage.maximum_filter(R, size=5))
        # 11. Return the coordinate of corners in R_
        corners = np.argwhere(R_ != 0)

    else:
        raise NotImplementedError(f'Invalid mode: {type}')

    # I return more result than the original function (for visualization purpose), return the grad for later use
    return corners, R_


# Helper functions for compute_descriptors function
def compute_feature_4_4(I):
    """
    Computes 8 bit descriptor for the small (4 x 4 patch).
    Parameters
    ----------
    I : float [4x4]
        gradient orientation as a 2D numpy array

    Returns
    -------
    D : numpy array [8 x 1]
        8 bit descriptors  corresponding to (4 x 4) patch
    """
    bin_edges = np.array([-180., -135., -90, -45, 0, 45, 90, 135, 180])
    I_digitize = np.digitize(I, bin_edges, True)  # This return the orient which each element in quad belong

    # Create an empty array contain the frequent of each orient happen in quad_digitize
    # See https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
    values, counts = np.unique(I_digitize.flatten(), return_counts=True)
    I_ori_count = np.zeros(8)
    for i in range(len(values)):
        I_ori_count[values[i] - 1] += counts[i]

    return I_ori_count / I_ori_count.sum()


def compute_feature_16_16(I):
    """
        Computes 128 bit descriptor for the small (16 x 16 patch).
        Parameters
        ----------
        I : float [16x16]
            gray scale image as a 2D numpy array

        Returns
        -------
        D : numpy array [128 x 1]
            128 bit descriptors  corresponding to (16 x 16) patch
        """
    scharr = 1 / 32. * np.array([[3 + 3j, 10 + 0j, 3 - 3j],
                                 [0 + 10j, 0 + 0j, 0 - 10j],
                                 [-3 + 3j, -10 + 0j, -3 - 3j]])

    # Compute gradient for each pixel, x y is the row and col of image array
    grad = signal.convolve2d(I, scharr, boundary='symm', mode='same')

    # Compute the gradient direction for each patch (16x16)
    # See: https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html
    patch_angle = np.arctan2(grad.real, grad.imag) * 180 / np.pi  # Orientation of gradient in degree

    # Separate to 16 x 16 patch into 16 small patch (each is 4 x 4), for each small patch compute the
    # histogram about the orient of gradient
    feature_128 = []
    for x in range(0, 15, 4):
        for y in range(0, 15, 4):
            four_by_four = patch_angle[x:x + 4, y:y + 4]
            feature_128.append(compute_feature_4_4(four_by_four))
    feature_128 = np.stack(feature_128, axis=0).flatten()
    return feature_128


# TODO: NOT SURE IF I WILL USE IT
def pad_with(vector, pad_width, iasis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


# DONE
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
    # Add small patch around the image to handle the numerical cases
    D = []
    # TODO: Padding will lead to the corner index wrong !!!
    # I = np.pad(I, 10, pad_with, padder=100/255.)
    for cor_point in corners:
        corner_patch = I[cor_point[0] - 7: cor_point[0] + 9, cor_point[1] - 7: cor_point[1] + 9]
        D.append(compute_feature_16_16(corner_patch))
    D = np.stack(D, axis=0)
    assert D.shape == (len(corners), 128)
    return D, I  # For plot purpose


def lowe_criteria(q, P, T=2.):
    """
    Compute the closed point with the given q from the point set P.

    Lowe's Ratio Test has 3 steps:
    * Find closet to descriptors to q, called p_1 and p_2 based on the Euclidean distances (p_1 is best match and p_2 is
    second-best match)
    * Test if distance to the best match is smaller than a Threshold T (hyper-param here):
            d(q, p_1) < T ?
    * Accept match only if the best match p_1 is substantially better than the second:
            d(q, p_1) / d(q, p_2) < 1 / 2

    If p_1 passed all criteria then accept that p_1 closed to the pivot q

    Parameters
    ----------
    q :  numpy array [128 x 1]
         128 bit feature of the pivot corner q from image 1
    P :  numpy array [num_corners x 128]
         descriptors for image 2 corners set
    T :  scalar
         threshold to accept the closest point to pass step 2 of lowe

    Returns
    -------
    idx:  scalar
          index of the best match corner from P with the given q, if not pass the Lowe, return None

    """
    distances = norm((P - q), axis=1)

    # TODO: If distances is all nan, then return None
    if np.isnan(distances).all():
        return None

    # Get indices for top-2 smallest
    idx_distance_sorted = np.argsort(distances)  # Return the index for the item in distance from small to big
    p_1_idx = idx_distance_sorted[0]  # Smallest distance
    p_2_idx = idx_distance_sorted[1]

    # Test if distance to best match is smaller than threshold
    if distances[p_1_idx] >= T:
        return None
    elif distances[p_1_idx] / distances[p_2_idx] < 0.5:
        # Accept the p_1 only the best match p_1 is substantially better than p_2
        return p_1_idx
    else:
        return None


# DONE
def compute_matches(D1, D2):
    """
    Computes matches for two images using the descriptors.
    Uses the Lowe's criteria to determine the best match.

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
    M = []
    for i in range(len(D1)):
        corr_idx = lowe_criteria(D1[i], D2)  # Corresponding point to current index
        if corr_idx is not None:
            M.append(np.array([i, corr_idx]))
    M = np.stack(M, axis=0)
    return M


# DONE
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
        p2 = C2[M[i, 1], :] + np.array([0, I1.shape[1]])
        plt.plot(p1[1], p1[0], 'rx', markersize=2)
        plt.plot(p2[1], p2[0], 'rx', markersize=2)
        plt.plot([p1[1], p2[1]], [p1[0], p2[0]], '-y', color='#5bff4d', linewidth=1)


# TODO: Implement RANSAC here
def compute_homography_ransac(C1, C2, M, inlier_thres=0.9):
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

    inlier_thres  : Inlier condition for stop the procedure and return result
    Returns
    ----------
    H_final : numpy array [3 x 3]
            Homography matrix which maps in point image 1 to image 2 
    M_final : numpy array [num_inlier_matches x 2]
            [cornerIdx1, cornerIdx2] each row contains indices of inlier matches
    """
    best_inlier_percent = -1.

    out_lier_ratio = 0.3
    sample_num = 4
    p = 0.9  # Our expected success ratio
    max_iter = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - out_lier_ratio)**sample_num)))  # Number of trial needed

    H_final = None
    M_final = None

    # Base on C1, C2, M, create P1 and P2 (i th row of P1 correspond to i th row of P2), P1 and P2 then used for
    # compute_residual function
    P1 = C1[M[:, 0], :]
    P2 = C2[M[:, 1], :]

    # TODO: Complete RANSAC algorithm to estimate the best estimation
    # TODO: Delete this line later

    for i in range(max_iter):
        # Uniform random choose 4 points, random without replacement
        # See: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        idx = np.random.choice(len(P1), sample_num, replace=False)
        H = calculate_homography_four_matches(P1[idx], P2[idx])

        # Base on the estimated H, compute the errors when mapping from P1 to P2 (this is the Euclidean of H @ P1 - P2)
        residual = compute_residual(P1, P2, H)
        # Set the threshold T to filter the distance, say if the distance is greater than 2 pixel, then not use
        T = 2.
        inlier_num = sum(residual < T)
        inliner_percent = inlier_num / len(P1)

        if inliner_percent > inlier_thres:
            # Stop and return the current H and current M as the final
            # Get position where is False, remove them from Original M
            indexes_false = np.where(np.invert(residual < T))[0]
            # See: https://numpy.org/doc/stable/reference/generated/numpy.delete.html
            M_ = np.delete(M, indexes_false, axis=0)
            return H, M_

        elif inliner_percent > best_inlier_percent:
            best_inlier_percent = inliner_percent
            indexes_false = np.where(residual < T)[0]
            M_ = np.delete(M, indexes_false, axis=0)
            H_final = H
            M_final = M_

    # If the inlier threshold not reach, then return the best version
    return H_final, M_final


# TODO: Calculate the geometric distance between estimated points and original points, namely residuals.
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
    residual : maybe the array of float [num_points x 1]
                residual computed for the corresponding points P1 and P2 
                under the transformation given by H
    """
    # Change point in P1, P2 to Homogeneous

    # TODO: Compute residual given Homography H
    # Basically get H @ P1 then compare to P2
    residual = None
    # Convert to homogeneous coordinate
    P1 = np.c_[P1, np.ones(len(P1))]
    P2 = np.c_[P2, np.ones(len(P2))]

    P1_ = (H @ P1.T).T
    # Normal to homogeneous
    P1_ = np.divide(P1_, P1_[:, -1].reshape(-1, 1))
    residual = norm((P2 - P1_), axis=1)
    return residual


# DONE
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

    if P1.shape[0] != 4  or P2.shape[0] != 4:
        print('Four corresponding points needed to compute Homography')
        return None

    # loop through correspondences and create assemble matrix
    # A * h = 0, where A(2n,9), h(9,1)

    A = []
    for i in range(P1.shape[0]):
        # ORIGINAL
        # p1 = np.array([P1[i, 0], P1[i, 1], 1])
        # p2 = np.array([P2[i, 0], P2[i, 1], 1])

        # Change x, y axis, due to the ambiguties of the wrapProjective function
        # See: https://github.com/OlehOnyshchak/ImageTransformations for more information
        # Or see PerspectiveTransformation.inpynb

        p1 = np.array([P1[i, 1], P1[i, 0], 1])
        p2 = np.array([P2[i, 1], P2[i, 0], 1])

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

    # SVD composition
    # The singular value is sorted in descending order
    u, s, v = np.linalg.svd(A)

    # Ee take the ???right singular vector??? (a column from V )
    # Which corresponds to the smallest singular value
    H = np.reshape(v[8], (3, 3))

    # Normalize and now we have H
    H = (1 / H[2, 2]) * H

    return H


if __name__ == '__main__':
    fig = plt.figure()
    plt.gray()

    I1 = imageio.imread('1.JPG')
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)

    I2 = imageio.imread('2.JPG')
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)

    T_harris = 0.05  # TODO:  Choose a suitable threshold
    T_shi_tomasi = 0.05  # TODO:  Choose a suitable threshold

    C1_harris, C1_harris_key = compute_corners(I1_gray, 'harris', T_harris)
    C2_harris, C2_harris_key = compute_corners(I2_gray, 'harris', T_harris)

    C1_shi_tomasi, C1_shi_tomasi_key = compute_corners(I1_gray, 'shi-tomasi', T_shi_tomasi)
    C2_shi_tomasi, C2_shi_tomasi_key = compute_corners(I2_gray, 'shi-tomasi', T_shi_tomasi)

    # Compute the descriptor for the two images
    D1_harris, _ = compute_descriptors(I1_gray, C1_harris)
    D2_harris, _ = compute_descriptors(I2_gray, C2_harris)

    D1_shi_tomasi, _ = compute_descriptors(I1_gray, C1_shi_tomasi)
    D2_shi_tomasi, _ = compute_descriptors(I2_gray, C2_shi_tomasi)
    # TODO: D has rows equals to corners, some corners give the nan result in D, Notice for that

    # TEST plot_matches function
    M_harris = compute_matches(D1_harris, D2_harris)
    M_shi_tomasi = compute_matches(D1_shi_tomasi, D2_shi_tomasi)

    # plot_matches(I1_gray, I2_gray, C1_shi_tomasi, C2_shi_tomasi, M_shi_tomasi)
    H, M = compute_homography_ransac(C1_shi_tomasi, C2_shi_tomasi, M_shi_tomasi, inlier_thres=0.999)
    # H, M = compute_homography_ransac(C2_harris, C1_harris, M_harris, inlier_thres=0.999)

    print('TEST VALUE OF H')
    print(H)

    print('TEST VALUE OF M')
    print(M)
    # Transform image
    h1, w1 = I1.shape[:2]
    h2, w2 = I2.shape[:2]
    # result = warpPerspective(I1, H, (h1 + h2, w1))
    # Need to transform from 2 to 1 then concat

    # Warp I2 to I1
    result = cv2.warpPerspective(I2, inv(H), (w1 + w2, h1))
    result[0:h1, 0:w1] = I1

    # OR Warp I1 to I2
    # result = cv2.warpPerspective(I1, H, (w1 + w2, h1))
    # result[0:w2, 0:h2] = I2

    plt.imshow(result)
    plt.show()
