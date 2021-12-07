# Compute camera extrinsics using P3P (spatial resection)
from _multibytecodec import __create_codec

import numpy as np
import numpy.testing
from numpy.linalg import norm, inv, svd, matrix_rank, det
from itertools import combinations
from math import sqrt


def _normal_point_to_hc(x):
    return np.array([x[0], x[1], 1.])


def _get_triplet(v, alpha, beta, a, b):
    """
    Compute the triplet s1, s2, s3 given the v value computed in the first step
    :return:
    """
    s1 = sqrt(b**2 / (1 + v**2 - 2 * v * np.cos(beta)))
    s3 = v * s1
    coeff = [1.0, -2*s3*np.cos(alpha), -(a**2 - s3**2)]
    s2_roots = np.roots(coeff)

    # TODO: Due to the result is > 0 then choose all, need to better config later
    # print(f's2 values: {s2_roots}')
    triplets = np.array([
        [s1, s2_roots[0], s3],
        [s1, s2_roots[1], s3]
    ])
    return triplets


def _create_coeff(a, b, c, alpha, beta, gamma):
    a2_add_c2_sur_b2 = (a**2 + c**2) / (b**2)
    a2_sub_c2_sur_b2 = (a**2 - c**2) / (b**2)
    b2_sub_c2_sur_b2 = (b**2 - c**2) / (b**2)
    b2_sub_a2_sur_b2 = (b**2 - a**2) / (b**2)

    A4 = (a2_sub_c2_sur_b2 - 1)**2 - (4 * (c**2 / b**2)) * np.cos(alpha)**2
    A3 = 4 * (a2_sub_c2_sur_b2 * (1 - a2_sub_c2_sur_b2) * np.cos(beta) - (1 - a2_add_c2_sur_b2) * np.cos(alpha) * np.cos(gamma) + (2 * (c**2/b**2)) * np.cos(alpha)**2 * np.cos(beta))
    A2 = 2 * (a2_sub_c2_sur_b2 ** 2 - 1 + 2 * (a2_sub_c2_sur_b2**2) * np.cos(beta)**2 + 2 * b2_sub_c2_sur_b2 * np.cos(alpha)**2 - (4 * a2_add_c2_sur_b2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)) + 2 * b2_sub_a2_sur_b2 * np.cos(gamma)**2)
    A1 = 4 * (- a2_sub_c2_sur_b2 * (1 + a2_sub_c2_sur_b2) * np.cos(beta) + (2 * (a**2 / b**2)) * np.cos(gamma)**2 * np.cos(beta) - (1 - a2_add_c2_sur_b2) * np.cos(alpha) * np.cos(gamma))
    A0 = (1 + a2_sub_c2_sur_b2) ** 2 - (4 * (a**2 / b**2)) * np.cos(gamma)**2

    return A4, A3, A2, A1, A0


# Add Umeyama method to solve Absolute Orientation problem
def umeyama_absolute_orientation_solver(source_point_set, target_point_set):
    """
    This function will return the Rigid Transformation params to map from source_point_set to target_point_set by the given function:
    See the original paper: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    See Cyrill's lecture: https://www.youtube.com/watch?v=K5WG40gMlz8&list=PLgnQpQtFTOGTPQhKBOGgjTgX-mzpsOGOX&index=26

    X_target = Lambda * R @ X_source + T (More info, see the code in create_sample_data.py)
    Notice: The point set here have the same cardinality and has form of (x_n, y_n) pair x_n in source and y_n in target
    :param source_point_set: np array shape (n_points, 3)
    :param target_point_set: np array shape (n_points, 3)
    :return: The params include:  scale, rotation matrix and translation vector
    """
    assert source_point_set.shape == target_point_set.shape, \
        "Source point set and target point set must have the same shape !!!"

    N, m = source_point_set.shape

    # 1. Compute the mean point for each set - origin for each point set
    # TODO: Here we not consider the weight for each point, in later version, we need take it into account
    x_0 = np.mean(source_point_set, axis=0)
    y_0 = np.mean(target_point_set, axis=0)

    # 2. Compute scale factor, lambda
    nominator = np.trace((target_point_set - y_0).T @ (target_point_set - y_0))
    denominator = np.trace((source_point_set - x_0).T @ (source_point_set - x_0))
    lbd = sqrt(nominator / denominator)

    # 3. Compute the Rotation Matrix
    a_n = source_point_set - x_0  # Shape (n_points, 3)
    b_n = target_point_set - y_0  # Shape (n_points, 3)

    cov_matrix = b_n.T.dot(a_n) / N  # Shape (3, 3)

    u, d, v_t = svd(cov_matrix, full_matrices=True)
    cov_rank = matrix_rank(cov_matrix)
    s = np.eye(m)

    if cov_rank >= m - 1 and det(cov_matrix) < 0:
        s[m - 1, m - 1] = -1
    elif cov_rank < m - 1:
        raise ValueError("Collinearity detected in covariance matrix:\n{}".format(cov_matrix))

    R = u.dot(s).dot(v_t)

    # 4. Compute the Translation Matrix
    T = y_0 - lbd * R @ x_0

    return lbd, R, T


def spatial_resection(x, X, K):
    """ Implements the P3P algorithm to estimate the extrinsic given
    4 3D-2D point correspondences """
    c = K[0, 0]  # c is the focal length
    # 1. Compute angles and distances to the 3D points
    # Compute the unit vector from camera projection center to 3D points: k x_i s = -sign(c) * Norm(K^-1 @ x_i)
    unit_vecs = []
    for point in x:
        point_hc = _normal_point_to_hc(point)
        vec = inv(K) @ point_hc
        print(f'Vec = {vec}')
        unit_vec = vec / norm(vec)
        unit_vecs.append(-np.sign(c) * unit_vec)
    unit_vecs = np.stack(unit_vecs, axis=0)  # Shape (4, 3) w.r.t to the input point
    # print(f'Unit vectors: {unit_vecs}')
    # points_in_cam_coord = np.stack(points_in_cam_coord, axis=0)
    # print(f'Points in camera coordinate system: {points_in_cam_coord}')

    # Notice that we just use 3 points to perform
    pair_list = list(combinations(range(3), 2))   # (0, 1), (0, 2), (1, 2)
    angles_list = []   # angles w.r.t pair_list
    distance_list = []  # distance w.r.t pair_list
    for pair in pair_list:
        angles_list.append(np.arccos(unit_vecs[pair[0]] @ unit_vecs[pair[1]]))
        distance_list.append(norm(X[pair[0]] - X[pair[1]]))
    angles_list = np.array(angles_list)
    distance_list = np.array(distance_list)

    print(f'Unit vectors: \n {unit_vecs}')
    print(f'Angle list: \n {angles_list}')
    print(f'Distance list: \n {distance_list}')
    # Build Quadratic equation: A_4 * v^4 + A_3 * v^3 + A_2 * v^2 + A_1 * v_1 + A_0 = 0
    c, b, a = distance_list
    print(f'a = {a}, b = {b}, c = {c}')
    gamma, beta, alpha = angles_list
    print(f'Alpha = {alpha}, Beta = {beta}, Gamma = {gamma}')
    A4, A3, A2, A1, A0 = _create_coeff(a, b, c, alpha, beta, gamma)

    coeff = np.array([A4, A3, A2, A1, A0])
    roots = np.roots(coeff)

    # print(roots)
    # Get only the root with im part is zero
    result_set = [np.real(_) for _ in roots if np.imag(_) == 0]
    # print(result_set)

    # 2. Identify the correct solution using the 4th point
    triplets = []
    for v in result_set:
        triplets.append(_get_triplet(v, alpha, beta, a, b))
    triplets = np.vstack(triplets)
    print(f'All triplets: \n{triplets}')

    # 3. Compute the coordinate transformation to obtain 𝑅 and 𝑋0
    # TODO: This is a problem and the link to it https://www.youtube.com/watch?v=roEjHhvLWmA \
    #  and the updated version: https://www.youtube.com/watch?v=K5WG40gMlz8
    # TODO: Suppose that I choose the triplet from step 2, this means I need a solution for step 2 !!!
    # Ok, I should make the code for "Absolute Orientation" problem. And then I'll go back to this

    # Uh mm, it seems and 1 and 2 was correct
    s_ = triplets[1][:, np.newaxis]
    X_i = unit_vecs[:3, ] * s_

    print(f'X_i = \n {X_i}')
    # Use Umeyama to compute the params for Absolute Transformation
    lbd, R, X0 = umeyama_absolute_orientation_solver(source_point_set=X_i, target_point_set=X[:3, ])

    return lbd, R, X0


if __name__ == '__main__':
    K = [[1280, 0, 272],
         [0, 1280, 480],
         [0, 0, 1]]

    K = np.array(K, dtype=np.float)

    X = np.array([[0, 0, 0],
         [7, 0, 0],
         [7, 0, 7],
         [7, 7, 7]])

    x = np.array([[489.69006223, 307.0092151],
                  [453.03577044, 565.74539245],
                  [222.32934563, 634.74170641],
                  [61.33794639, 482.37484642]])

    lbd, R, X0 = spatial_resection(x, X, K)
    print(f'Lambda value: \n{lbd}')
    print(f'Rotation matrix: \n{R}')
    print(f'Translation matrix: \n{X0}')

    # TEST
    X0_ = -X0[:, np.newaxis]
    I_X0 = np.concatenate((np.eye(3), X0_), axis=1)
    P = lbd * K @ R.T @ I_X0

    P_1 = np.array([7., 7., 7., 1.])
    x_hat_1 = P @ P_1
    x_hat_1 = x_hat_1 / x_hat_1[-1]

    print("\n Reprojected image coordinates 1: \n", x_hat_1)

