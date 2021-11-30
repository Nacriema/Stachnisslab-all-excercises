# MUST KNOW 

## Direct Linear Transform (DLT)
In terms of projection geometric, this is used for **Affine Camera**, where the process do not have any **non-linear errors**.

When we use this term for **Camera Calibration**, this is a method to estimate the projection matrix P. In homogeneous 
coordinate system, this matrix has shape **(3 x 4)**. It contains 11 degrees of freedom (5 for Intrinsic and 6 for Extrinsic).

We need at least ceil(11/2) = 6 points to estimate P. 

The idea for the **DLT** method: 

We have: 

![](https://latex.codecogs.com/svg.image?x_i%20=%20P%20X_i%20=%20%5Cbegin%7Bbmatrix%7Dp_%7B11%7D%20&%20p_%7B12%7D%20&p_%7B13%7D%20%20&%20p_%7B14%7D%20%5C%5Cp_%7B21%7D%20&%20p_%7B22%7D%20&%20p_%7B23%7D%20&%20p_%7B24%7D%20%5C%5Cp_%7B31%7D%20&%20p_%7B32%7D%20&%20p_%7B33%7D%20&%20p_%7B34%7D%20%5C%5C%5Cend%7Bbmatrix%7D%20X_%7Bi%7D)




