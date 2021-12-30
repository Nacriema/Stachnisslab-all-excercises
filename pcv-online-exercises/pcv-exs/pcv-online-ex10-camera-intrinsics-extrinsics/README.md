# MUST KNOW 

## Direct Linear Transform (DLT)
In terms of projection geometric, this is used for **Affine Camera**, where the process do not have any **non-linear errors**.

When we use this term for **Camera Calibration**, this is a method to estimate the projection matrix P. In homogeneous 
coordinate system, this matrix has shape **(3 x 4)**. It contains 11 degrees of freedom (5 for Intrinsic and 6 for Extrinsic).

We need at least ceil(11/2) = 6 points to estimate P. 

The idea for the **DLT** method: 

We have: 

![](https://latex.codecogs.com/svg.image?x_i%20=%20P%20X_i%20=%20%5Cbegin%7Bbmatrix%7Dp_%7B11%7D%20&%20p_%7B12%7D%20&p_%7B13%7D%20%20&%20p_%7B14%7D%20%5C%5Cp_%7B21%7D%20&%20p_%7B22%7D%20&%20p_%7B23%7D%20&%20p_%7B24%7D%20%5C%5Cp_%7B31%7D%20&%20p_%7B32%7D%20&%20p_%7B33%7D%20&%20p_%7B34%7D%20%5C%5C%5Cend%7Bbmatrix%7D%20X_%7Bi%7D)
**(1)**

Let's call: 

![](https://latex.codecogs.com/svg.image?A%20=%20%5Cbegin%7Bbmatrix%7Dp_%7B11%7D%20&%20p_%7B12%7D%20&%20p_%7B13%7D%20&%20p_%7B14%7D%20%5C%5C%5Cend%7Bbmatrix%7D%5E%7BT%7D,%20B%20=%20%5Cbegin%7Bbmatrix%7Dp_%7B21%7D%20&%20p_%7B22%7D%20&%20p_%7B23%7D%20&%20p_%7B24%7D%20%5C%5C%5Cend%7Bbmatrix%7D%5E%7BT%7D%20,%20C%20=%20%5Cbegin%7Bbmatrix%7Dp_%7B31%7D%20&%20p_%7B32%7D%20&%20p_%7B33%7D%20&%20p_%7B34%7D%20%5C%5C%5Cend%7Bbmatrix%7D%5E%7BT%7D)

Then **(1)** becomes:

![](https://latex.codecogs.com/svg.image?x_%7Bi%7D%20=%20%5Cbegin%7Bbmatrix%7DA%5E%7BT%7D%20%5C%5C%20B%5E%7BT%7D%5C%5CC%5E%7BT%7D%5Cend%7Bbmatrix%7DX_%7Bi%7D)

In Euclidian Coordinate, then the 2D coordinates of x_i:

![](https://latex.codecogs.com/svg.image?x_%7Bi%7D%20=%20%5Cfrac%7BA%5E%7BT%7DX_%7Bi%7D%7D%7BC%5E%7BT%7DX_%7Bi%7D%7D,%20y_i%20=%20%5Cfrac%7BB%5ETX_i%7D%7BC%5ETX_i%7D)

Rewrite this to the linear equaltion: 

![](https://latex.codecogs.com/svg.image?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20-X_i%5ETA&plus;x_iX_i%5ETC%20=0%20%5C%5C-X_i%5ETB&plus;y_iX_i%5ETC=0%5Cend%7Bmatrix%7D%5Cright.)
 **(2)**

With X_i, x_i, y_i are known params, we need to calculate A, B, C. 

Notice x_i, y_i are scalars, X_i has shape (4, 1).

Let's call: 

![](https://latex.codecogs.com/svg.image?p%20=%20%5Cbegin%7Bbmatrix%7DA%20%5C%5CB%20%5C%5CC%5Cend%7Bbmatrix%7D%20=%20vec(P%5ET))

Then **(2)** becomes: 

![](https://latex.codecogs.com/svg.image?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7Da_%7Bx_i%7D%5ETp%20=%200%20%5C%5Ca_%7By_i%7D%5ETp=0%5Cend%7Bmatrix%7D%5Cright.)

with: 

![](https://latex.codecogs.com/svg.image?a_%7Bx_i%7D%5ET%20=%20(-X_i%5ET,%200%5ET,%20x_iX_i%5ET)=(-X_i;%20-Y_i,%20-Z_i,%20-1,%200,%200,%200,%200,%20x_iX_i,%20x_iY_i,%20x_iZ_i,%20x_i))

And: 

![](https://latex.codecogs.com/svg.image?a_%7By_i%7D%5ET%20=%20(0%5ET,%20-X_i%5ET,%20y_iX_i%5ET)=(0,%200,%200,%200,%20-X_i;%20-Y_i,%20-Z_i,%20-1,%20y_iX_i,%20y_iY_i,%20y_iZ_i,%20y_i))

For each point in 2D image, we have a pair of condition like this, then when we stack together, we have: 

![](https://latex.codecogs.com/svg.image?%5Cbegin%7Bbmatrix%7Da_%7Bx_i%7D%5ET%20%5C%5Ca_%7By_i%7D%5ET%20%5C%5C...%5Cend%7Bbmatrix%7Dp%20=%20Mp%20=%200)

Matrix M has shape (2I, 12) with I: number of 2D points. p has shape of (12, 1).

This is form of Ax = 0, x is the null space of A (**kernel** of A), see wiki: https://en.wikipedia.org/wiki/Kernel_(linear_algebra)

We can solve this by **SVD** or use **Least Square Approximation** methods. 

With SVD, we decompose M into U @ S @ V^T. Then choose p = v_{12}, which minimize the M.p (M.p = a.p with a is the smallest).

Then from p, we reshape it to (3, 4), this is the actual result.

### Notice:
* M has rank 11 if the number of 2D points >=6, otherwise we couldn't estimate these params. 
* We have no solution if all selected 2D point lie on the same plane or they approximate lie on the same plane.

From P, to get the K, R and X_0:

P = [KR | -KRX_0] = [H | h]

So:

![](https://latex.codecogs.com/svg.image?X_0%20=%20-H%5E%7B-1%7Dh)

H = KR -> Use **QR decomposition** on H^{-1}

![](https://latex.codecogs.com/svg.image?H%5E%7B-1%7D%20=%20(KR)%5E%7B-1%7D%20=%20R%5E%7B-1%7DK%5E%7B-1%7D%20=%20R%5ETK%5E%7B-1%7D%20=%20QR)

### Notice: 

K in form of H.C Coordinate -> Uniform it by multiply with K[3, 3]. Then simultaneously multiply K[3, 3] with the R to have the final result of K and R.

