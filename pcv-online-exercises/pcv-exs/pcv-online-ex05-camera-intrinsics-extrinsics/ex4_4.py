# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, svd, matrix_rank
import cv2 as cv


def E_from_point_pairs(xs, xss, K):
    """Compute Essential Matrix from points pairs."
    See Cyrill's lecture: https://www.youtube.com/watch?v=gYYumFSDsvA&list=PLgnQpQtFTOGTPQhKBOGgjTgX-mzpsOGOX&index=19
    See Chapter 13.3.2.3 of the book Photogrammetric and Computer vision
    See Lecture CS231A Epipolar Geometry: https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
    Input:
        xs  - [N x 3] points in first image
        xss - [N x 3] points in second image
        K   - [3 x 3] calibration matrix
    Output:
        E  - [3 x 3] essential matrix of camera pair
    """
    # 1. Compute direction of ray in camera coordinate system: k_x = inv(K).x", k_x has shape (N_point, 4) (3x3) x (
    # 3, 1)
    k_xs = []
    k_xss = []
    for i in range(len(xs)):
        k_xs.append(inv(K) @ xs[i])
        k_xss.append(inv(K) @ xss[i])
    k_xs = np.stack(k_xs, axis=0)  # Shape (7, 3)
    k_xss = np.stack(k_xss, axis=0)  # Shape (7, 3)
    print(np.shape(k_xs))
    print(np.shape(k_xss))
    A = []
    for i in range(len(k_xs)):
        A.append(np.kron(k_xss[i], k_xs[i]))
    A = np.stack(A, axis=0)
    print(np.shape(A))
    print(A)
    # 2. Compute SVD of A",
    u, d, v_t = svd(A, full_matrices=False)
    print(f'D = {d}')
    # print(u @ np.diag(d) @ v_t)
    # 3. Compute essential Matrix, there are 7 Essential Matrix, but just one of them is valid. So we need to choose one

    # Choose the least value, E_hat will have rank 3
    E_hat = v_t[-1].reshape(3, 3)
    print(f'Rank of E_hat: {matrix_rank(E_hat)}')

    # Use SVD on E_hat
    u, d, v_t = svd(E_hat, full_matrices=False)
    E = u @ np.diag([1.0, 1.0, 0]) @ v_t  # By doing this, Essential Matrix will have rank 2

    print(u.T)
    print('===============================================')
    print(v_t)

    print(f'ESTIMATED E: {E}')
    print('=====================================')
    print(f'Rank of E: {matrix_rank(E)}')

    # Apply conditions to E
    # Final essential matrix
    return E


def vector_to_sym_matrix(b):
    """
    Convert a vector (3,) to symmetric matrix (3, 3)
    :param b:
    :return:
    """
    return np.array([
        [0., -b[2], b[1]],
        [b[2], 0, -b[0]],
        [-b[1], b[0], 0]
    ])


# TODO: Compute relative orientation when given Essential Matrix
def relative_orientation_from_E(E, Z, W):
    """Compute the rotation and direction given E.
    See Hartley & Zisserman book, Part IV, Chapter 9.6.1 and 9.6.2

    Suppose that
    SVD(E) = U @ diag(1, 1, 0) @ V^T.

    There are 2 possible factorization E = S @ R as follows:
    S = U @ Z @ U^T , R = U @ W @ V^T or R = U @ W^T @ V^T

    With Z = |0   1  0|  is the skew-symmetric - this is the base element of S
             |-1  0  0|
             |0   0  0|

         W = |0   -1   0|  is the Orthogonal matrix  - this is the base element of R
             |1    0   0|
             |0    0   1|

    Then when estimate the Relative Orientation, we use the general form. Camera 1 is the root of Coordinate System,
    we just estimate the Extrinsic P' of Camera 2 w.r.t Camera 1.  Extrinsic of Camera 1 is P = [I_3 | 0], there are 4
    possible choices for the Camera 2 Extrinsic P':

    P' = [U @ W @ V^T | +u3] or [U @ W @ V^T | -u3] or [U @ W^T @ V^T | +u3]  or [U @ W^T @ V^T | -u3]

    With t = u3 = U @ (0, 0, 1)T - is the last column of U

    Then we need to check which P' is our exact solution

    Input:
        E - [3 x 3] essential matrix of camera pair
        Z - [3 x 3] Solution by Hartley & Zisserman
        W - [3 x 3] Solution by Hartley & Zisserman
    Output:
        R  - [3 x 3] rotation matrix
        Sb - [3 x 3] skew sym matrix which represents direction of 2nd cam
    """
    # This array is used to check the sign of u3, and we use it to compute S
    Z_ = np.array([[0, 1., 0],
                   [-1, 0, 0],
                   [0, 0, 0]])

    # Compute svd of Essential Matrix
    u, d, v_t = svd(E, full_matrices=False)

    # Compute rotation matrix: R = U @ W @ V^T
    R = u @ W @ v_t

    # Compute skew-sym matrix
    if np.array_equal(Z, Z_):
        t = u @ np.array([0, 0, 1.])  # Since b is (3, ) then it will do the same
    else:
        t = - u @ np.array([0, 0, 1.])
    Sb = vector_to_sym_matrix(t)

    print(f'R = {R}')
    print(f't = {t}')
    print(f'Sb = {Sb}')
    print('===========================================================')
    return R, Sb


def compute_intersection(X1, r, X2, s):
    """ Compute the intersection of rays of two cameras
    Given 2 rays from 2 camera 1 and 2 (from position and direction vector for each camera)
    X1: (0, 0, 0)
    X2: derived from Sb Shape (3, )
    r: direction vector - derived from image point of 1st cam
    s: direction vector - derived from image point of 2nd cam
    These 2 rays come from the same point in 3D, so we can find the 3D point that the intersection of 2 ray

    This function is used by the triangulate_points function
    Input:
        X1 - [3 x 1] position of 1st camera
        r  - [3 x 1] direction vector of 1st cam
        X2 - [3 x 1] position of 2nd camera
        s  - [3 x 1] direction vector of 2nd cam
    Output:
        X - [3 x 1] point of intersection

    # TODO: This comment must be include in the README
    See Hartley & Zisserman book, Chapter 12.3, they said:
    * Due to the fact that the typical observation points consists of noisy point correspondence x <-> x', which does
    not in general satisfy the epipolar constraint (coplanar constraint). In reality, the corresponding image point
    should be point x_ <-> x'_ lying close to the measured point x <-> x' and satisfy the coplanar constraint (epipolar
    constrain) exactly.

    Then our job is to find the correct point x_ and x'_ near the point x and x' that minimized the sum distance
    function and satisfy the coplanar constrain exactly. If we have them, then we can compute the exact point 3D X_hat (
    in Cam 1 coordinate system - This is so called the photogrammetric model (Cyrill's Lesson)).

    Harley & Zisserman suggest 2 way to handle that: Sampson approximation (first-order geometric correction), this
    method will give the approximation of 2d point for each image, but new points still not satisfy the epipolar
    relation exactly.

    We then must move to the Optimal Solution - a costly one (costly to compute one)

    For simply, Cyrill want us to approximate the 3D point - the intersection point of 2 3D line (may be it's not
    perfectly intersect, but we can approximate this point): I will choose the line that perpendicular to these 2 lines
    (Then that line contain the sortest distance between 2 lines), then we compute the mean point and choose this point
    as our X (X represented w.r.t Camera 1 coordinate system)
    """

    # 1. Normalize the Direction vector to Unit vector
    r = r / np.linalg.norm(r)   # r - direction vector of Cam 1 (in Cam 1 c.s)
    s = s / np.linalg.norm(s)   # s - direction vector of Cam 2 (in Cam 1 c.s)

    # 2. Derived from geometrical constraints
    # Find the point of line 1 (has direction r and pass through X1) and line 2 (has direction s and pass through X2),
    # and the distance of these 2 point is smallest.
    # See here: https://math.stackexchange.com/questions/1033419/line-perpendicular-to-two-other-lines-data-sufficiency
    # And here: https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    A = np.array([[np.dot(r, r), -np.dot(r, s)],
                  [np.dot(r, s), -np.dot(s, s)]])
    b = np.array([np.dot(r, (X2 - X1)), np.dot(s, (X2 - X1))])
    x = np.linalg.solve(A, b)

    # 3. Compute each point for each light ray
    # X0 - point in r where the lines should intersect
    # X1 - point in s where the lines should intersect
    F = X1 + x[0] * r
    H = X2 + x[1] * s

    # 4. We need a mean point if two lines do not intersect in 3D
    if np.allclose(F, H):
        X = F
    else:
        X = (F + H) / 2.0

    return X


def triangulate_points(xs, xss, K, R21, X2):
    """Triangulate a batch of rays

    Input:
        xs  - [N x 3] points in first image
        xss - [N x 3] points in second image
        K   - [3 x 3] calibration matrix
        # TODO: Need inspect the meaning of Rotation Matrix here !!!
        R21 - [3 x 3] rotation of 2nd cam (w.r.t. 1st cam)
        X2  - [3 x 1] projection center of 2nd cam (w.r.t 1st cam)
      
    Output:
        X - [N x 3] 3D points in global frame (w.r.t 1st cam)
    """
    # 1. Compute direction in corresponding camera frame (w.r.t each coordinate system):
    # k_x = inv(K).x", k_x has shape (3, ) due to (3x3) x (3, ), stack them together to (N_point, 3)
    k_xs = []
    k_xss = []
    for i in range(len(xs)):
        k_xs.append(inv(K) @ xs[i])
        k_xss.append(inv(K) @ xss[i])
    k_xs = np.stack(k_xs, axis=0)  # Shape (N_point, 3)
    k_xss = np.stack(k_xss, axis=0)  # Shape (N_point, 3)

    # 2. Rotate rays of 2nd cam so that they are represented in the coord system of 1st cam
    k_xss_rotated = []
    for i in k_xss:
        k_xss_rotated.append(R21 @ i)
    k_xss_rotated = np.stack(k_xss_rotated, axis=0)

    # 3. Intersect point, use the compute_intersection function
    assert len(k_xs) == len(k_xss_rotated), "Number of pair must be equivalent !!!!"
    X = []
    for i in range(len(k_xs)):
        X.append(compute_intersection(X1=np.array([0., 0., 0]), r=k_xs[i], X2=X2, s=k_xss_rotated[i]))
    X = np.stack(X, axis=0)
    return X


def point_in_front_of_cam(X):
    """Check whether points lie in front of the camera, mean that point is in front of both Cam 1 and Cam 2 w.r.t
    Cam 1's coordinate system.

    Inputs:
        X - [N x 3] Points in camera coordinate system
    """
    check = X[:, 2] >= 0
    return np.all(check)


def gen_homogeneous_pts(pts):
    homo_pts = np.ones((pts.shape[0], pts.shape[1] + 1))
    homo_pts[:, :pts.shape[1]] = pts
    return homo_pts


if __name__ == '__main__':
    K = np.array(
        [[-1.23543537e+03, 1.08226965e+01, 2.82869653e+02],
         [-0.00000000e+00, -1.23745635e+03, 4.69737272e+02],
         [+0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    pts1 = np.load('point_pair_cube_0.npy')
    pts2 = np.load('point_pair_cube_1.npy')

    print("\n Points in the first image:\n ", pts1)
    print("\n Points in the second image:\n ", pts2)

    E_ = E_from_point_pairs(pts1, pts2, K)

    # Use OpenCV estimate Essential matrix
    pts_2d_1 = np.array([[492.09903121, 307.98018972],
                         [451.78454589, 565.25990514],
                         [221.62548423, 634.89401616],
                         [236.28529707, 346.82869375],
                         [59.63455229, 483.16495321],
                         [58.90156165, 241.27804127],
                         [280.99772625, 221.48729393]])

    pts_2d_2 = np.array([[530.21454461, 397.40504807],
                         [463.51239617, 676.67448276],
                         [226.75641872, 758.76943469],
                         [277.33277304, 428.92364569],
                         [11.2571699, 565.99289578],
                         [33.24688917, 291.85439559],
                         [267.80389469, 286.72346109]])

    # F, mask = cv.findFundamentalMat(pts_2d_1, pts_2d_2, cv.FM_7POINT, 0.1, 0.99)
    # print(f'Fundamental matrix: {F}')
    # # F is Rank 2
    # print(f'Rank of F: {matrix_rank(F)}')
    # E = cv.findEssentialMat(pts_2d_1, pts_2d_2, K)
    # print(f'Estimated E: {E}')

    # relative_orientation_from_E(E_, 1, 2)

    Z = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 0]])
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    W_CHECK = [W, W.T]
    Z_CHECK = [Z, Z.T]

    num_solution = 0
    R_final = None
    b_final = None
    for w in W_CHECK:
        for z in Z_CHECK:
            R, Sb = relative_orientation_from_E(E_, z, w)
            b = np.array([-Sb[1, 2], Sb[0, 2], -Sb[0, 1]])
            # Triangulate points in coord system of 1st cam
            X0 = triangulate_points(pts1, pts2, K, R, b)
            print(f'X0 = {X0}')
            # Triangulated points in coord system of 2nd cam
            X1 = np.dot(R.T, (X0 - b).T).T
            print(f'X1 = {X1}')
            if point_in_front_of_cam(X0) and point_in_front_of_cam(X1):
                num_solution += 1
                print('Final Rotation:\n', R)
                R_final = R
                print('Final Baseline:\n', b)
                b_final = b

    print(f'NUMBER OF SOLUTIONS : {num_solution}')
    print(f'FINAL ROTATION:\n{R_final}')
    print(f'FINAL BASELINE:\n{b_final}')
