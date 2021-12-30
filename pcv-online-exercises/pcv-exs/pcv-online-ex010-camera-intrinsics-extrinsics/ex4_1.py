# Implementation of the DLT method to compute the camera projection matrix 

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, svd, qr


def decompose_P(P):
    # Decompose Camera projection matrix P into intrinsic (c, s, m , x_0, y_0)
    # and extrinsic  parameters (R, X0).
    # Input:  Camera projection matrix (P:= 3x4)
    # Output: Intrinsics (K:= 3x3)
    #         Origin of camera in world frame (X0:= 3x1)
    #         Orientation of the camera in the world frame (R:=3x3)
    # Notes:
    # Camera projection equation
    # X (3D point in world) -> x (2d position in image)
    # x = KR[I|-X0]X

    # We have P = KR[I|-X0] = [KR| -KRX0] = [H|h] (H = KR (a 3x3 matrix), h=-KRX0 (a 3x1 matrix))
    # Then X0 = -H^{-1}h
    # See: https://stackoverflow.com/questions/32160576/how-to-select-a-portion-of-a-numpy-array-efficiently
    H = P[:, 0:3]
    print(f'H = {H}')
    h = P[:, 3]
    print(f'h = {h}')
    X0 = -inv(H) @ h

    # Do the QR decomposition on H^-1 = R^T * K^-1  to Q = R^T and R = K^-1 is the triangle matrix
    q, r = qr(inv(H))
    R = np.transpose(q)
    K = inv(r)
    temp = K[2, 2]
    K = K / temp
    R = R * temp
    return K, R, X0
    
    
def dlt(x, X):
    """
    Given set of point X in 3D coordinate and their corresponding 2D point in image coordinate. Use DLT method to
    estimate the Projection matrix
    :param x: [N_points, 2] array
    :param X: [N_points, 3] array
    :return: Projection matrix P shape (3x4)
    """
    # See https://stackoverflow.com/questions/58083743/what-is-the-fastest-way-to-stack-numpy-arrays-in-a-loop
    # For better np array stack with for loop
    M = []
    assert x.shape[0] == X.shape[0]
    # 1. Build linear system Mp = 0, exactly we will build the M matrix shape (2*N_points, 12)
    for i in range(x.shape[0]):
        M.append([-X[i][0], -X[i][1], -X[i][2], -1, 0., 0, 0, 0, x[i][0] * X[i][0], x[i][0] * X[i][1], x[i][0] * X[i][2], x[i][0]])
        M.append([0., 0, 0, 0, -X[i][0], -X[i][1], -X[i][2], -1, x[i][1] * X[i][0], x[i][1] * X[i][1], x[i][1] * X[i][2], x[i][1]])
    M = np.stack(M, axis=0)

    # 2. Perform SVD M = USV'
    # See docs: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    u, s, vh = svd(M, full_matrices=False)
    # 3. Extract solution
    # Choose the last item of vh, which has the smallest eigen value
    P = vh[-1].reshape(3, 4)
    return P


if __name__ == '__main__':
    # TEST
    # Define 3D control points (Minimum of 6 Points)
    X = [[0, 0, 0],
         [7, 0, 0],
         [7, 0, 7],
         [0, 0, 7],
         [7, 9, 7],
         [0, 9, 7],
         [0, 9, 0]]
    X = np.array(X)

    # Load predefined 2D point
    x = np.load('corres_points.npy')

    P = dlt(x, X)
    print("\n The estimated projection matrix: \n", P)

    K, R, X0 = decompose_P(P)

    print('Intrinsic matrix: \n', K)
    print('Extrinsic matrix: ', "\n R: \n", R, "\n X0: \n", X0)

    X0_ = -X0[:, np.newaxis]
    I_X0 = np.concatenate((np.eye(3), X0_), axis=1)
    print(I_X0)

    print('COMPARE P')
    print(P)
    print(K @ R @ I_X0)

    I = plt.imread('./data/checkerboard_cube/cube_origin.png')
    plt.imshow(I)
    plt.show()