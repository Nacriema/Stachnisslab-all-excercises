# Compute camera extrinsics using P3P (spatial resection)
import numpy as np
import numpy.testing
from numpy.linalg import norm, inv
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
    print(f's2 values: {s2_roots}')
    triplets = np.array([
        [s1, s2_roots[0], s3],
        [s1, s2_roots[1], s3]
    ])
    return triplets


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
        unit_vec = vec / norm(vec)
        unit_vecs.append(-np.sign(c) * unit_vec)
    unit_vecs = np.stack(unit_vecs, axis=0)  # Shape (4, 3) w.r.t to the input point
    print(f'Unit vectors: {unit_vecs}')
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
    # Build Quadratic equation: A_4 * v^4 + A_3 * v^3 + A_2 * v^2 + A_1 * v_1 + A_0 = 0
    a, b, c = distance_list
    alpha, beta, gamma = angles_list

    A4 = ((a**2 - c**2) / b**2 - 1)**2 - (4*(c**2) / b**2 * np.cos(alpha)**2)
    A3 = 4 * ((a**2 - c**2)/b**2 * (1 - (a**2-c**2)/b**2) * np.cos(beta) - (1 - ((a**2 + c**2) / b**2))*np.cos(alpha)*np.cos(gamma) + 2 * c**2 / b**2 * np.cos(alpha)**2 * np.cos(beta))
    A2 = 2 * ( ((a**2 - c**2)/b**2)**2 - 1 + 2*((a**2 - c**2)/b**2)**2 * np.cos(beta)**2 + 2 * ((b**2 - c**2)/b**2) * np.cos(alpha)**2 - 4 * ((a**2 + c**2)/b**2) * np.cos(alpha) * np.cos(beta) * np.cos(gamma) + 2 * ((b**2 - a**2)/b**2) * np.cos(gamma)**2)
    A1 = 4 * ( -((a**2 - c**2)/b**2) * (1 + (a**2 - c**2) / b**2)*np.cos(beta) + 2*(a**2/b**2)*np.cos(gamma)**2 * np.cos(beta) - (1 - ((a**2 + c**2)/b**2)) * np.cos(alpha) * np.cos(gamma))
    A0 = (1 + (a**2 - c**2)/b**2)**2 - 4 * a**2 / b**2 * np.cos(gamma)**2

    coeff = np.array([A4, A3, A2, A1, A0])
    roots = np.roots(coeff)
    print(roots)
    # Get only the root with im part is zero
    result_set = [np.real(_) for _ in roots if np.imag(_) == 0]
    print(result_set)

    # 2. Identify the correct solution using the 4th point
    test_point = x[-1]
    triplets = []
    for v in result_set:
        triplets.append(_get_triplet(v, alpha, beta, a, b))
    triplets = np.vstack(triplets)
    print(f'All triplets: {triplets}')

    # 3. Compute the coordinate transformation to obtain 𝑅 and 𝑋0
    # TODO: This is a problem and the link to it https://www.youtube.com/watch?v=roEjHhvLWmA \
    #  and the updated version: https://www.youtube.com/watch?v=K5WG40gMlz8
    # TODO: Suppose that I choose the triplet from step 2, this means I need a solution for step 2 !!!
    # Ok, I should make the code for "Absolute Orientation" problem. And then I'll go back to this
    s1, s2, s3 = triplets[0]
    R = np.eye(3, dtype=np.float)
    X0 = np.zeros((3, 1), dtype=np.float)
    return R, X0


if __name__ == '__main__':
    K = [[-1280, 0, 272],
         [0, -1280, 480],
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

    spatial_resection(x, X, K)
