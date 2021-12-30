# Compute camera intrinsics (K) using Zhang's method using a known checkerboard pattern.
import numpy as np
import matplotlib.pyplot as plt


def compute_homography(x, X):
    """
    Computes homography H (3x3) given 2D-3D correspondences
    The idea is the same of the dlt function in ex4_1.py
    """
    # Build linear system
    M = []
    assert x.shape[0] == X.shape[0]
    for i in range(x.shape[0]):
        M.append([-X[i, 0], -X[i, 1], -1, 0., 0, 0, x[i, 0] * X[i, 0], x[i, 0] * X[i, 1], x[i, 0]])
        M.append([0., 0, 0, -X[i, 0], -X[i, 1], -1, x[i, 1] * X[i, 0], x[i, 1] * X[i, 1], x[i, 1]])
    M = np.stack(M, axis=0)
    print(f'Shape of M: {M.shape}')
    # Perform SVD
    u, s, vh = np.linalg.svd(M, full_matrices=False)
    # Extract solution
    H = vh[-1].reshape(3, 3)
    return H


def get_V(H, i, j):
    """
    v_ij = [h1i * h1j,
           h1i * h2j + h2i * h1j,
           h3i * h1j + h1i * h3j,
           h2i * h2j,
           h3i * h2j + h2i * h3j,
           h3i * h3j]
    """
    return np.array([[H[0, i-1] * H[0, j-1],
                     H[0, i-1] * H[1, j-1] + H[1, i-1] * H[0, j-1],
                     H[1, i - 1] * H[1, j - 1],
                     H[2, i-1] * H[0, j-1] + H[0, i-1] * H[2, j-1],
                     H[2, i-1] * H[1, j-1] + H[1, i-1] * H[2, j-1],
                     H[2, i-1] * H[2, j-1]]])


def compute_V(H):
    """ Computes the constraint matrix (2x6) given the homography for each image
    |v_12       |
    |v_11 - v_22|

    Where v_ij = [h1i * h1j,
                  h1i * h2j + h2i * h1j,
                  h3i * h1j + h1i * h3j,
                  h2i * h2j,
                  h3i * h2j + h2i * h3j,
                  h3i * h3j]

    And H = | h11  h12  h13 |
            | h21  h22  h23 |
            | h31  h32  h33 |
    """
    V = np.concatenate([get_V(H, 1, 2), get_V(H, 1, 1) - get_V(H, 2, 2)])
    assert V.shape == (2, 6)
    return V


def compute_K(V):
    """ Extracts K from the constraint matrix V """
    # Solve for Vb = 0
    u, s, vh = np.linalg.svd(V, full_matrices=False)
    b = vh[-1]  # b = [b11, b12, b13, b22, b23, b33]
    # B is a symmetric matrix
    B = np.array([
        [b[0], b[1], b[2]],
        [b[1], b[3], b[4]],
        [b[2], b[4], b[5]]
    ])
    # Decompose B to obtain K, Chol(B) = A.A^T => A = K^{-T}
    A = np.linalg.cholesky(B)
    K = np.linalg.inv(A.T)
    # Rotate to have a negative camera constant c
    print('TEST VALUE OF K:')
    print(K)
    return K


if __name__ == '__main__':
    # TEST
    # checkerboard board params
    board = [5, 7]  # (Leaving out the outer columns and rows.)
    square = 0.027  # (27 mm)

    # Set 3D points (You may choose other points if you prefer)
    X = [[0, 0, 0],
         [0, board[1] * square, 0],
         [board[0] * square, board[1] * square, 0],
         [board[0] * square, 0, 0]]

    # 0, 0, 0; 0, 7, 0; 5, 7, 0; 5, 0, 0
    X = np.array(X)
    print('3D points: X = ', X)

    x0 = np.load('checkerboard0_points.npy')
    x1 = np.load('checkerboard1_points.npy')
    x2 = np.load('checkerboard2_points.npy')
    x3 = np.load('checkerboard3_points.npy')

    H0 = compute_homography(x0, X)
    H1 = compute_homography(x1, X)
    H2 = compute_homography(x2, X)
    H3 = compute_homography(x3, X)

    print('X0 = ', x0)
    print('X1 = ', x1)
    print('X2 = ', x2)
    print('X3 = ', x3)

    print('H0 = ', H0)
    print('H1 = ', H1)
    print('H2 = ', H2)
    print('H3 = ', H3)

    # Compute constraint matrix V
    V0 = compute_V(H0)
    V1 = compute_V(H1)
    V2 = compute_V(H2)
    V3 = compute_V(H3)

    # stack V from each image
    V = np.concatenate([V0, V1, V2, V3])

    print('Constraint matrix V = ', V)

    K = compute_K(V)
    print('Camera Intrinsics (K) := ', K)


