# MUST KNOW 

## Direct Linear Transform (DLT)
In terms of projection geometric, this is used for **Affine Camera**, where the process do not have any **non-linear errors**.

When we use this term for **Camera Calibration**, this is a method to estimate the projection matrix P. In homogeneous 
coordinate system, this matrix has shape **(3 x 4)**. It contains 11 degrees of freedom (5 for Intrinsic and 6 for Extrinsic).

We need at least ceil(11/2) = 6 points to estimate P. 

The idea for the **DLT** method: 

We have: 

![](https://latex.codecogs.com/svg.image?x_i&space;=&space;P&space;X_i&space;=&space;\begin{bmatrix}p_{11}&space;&&space;p_{12}&space;&p_{13}&space;&space;&&space;p_{14}&space;\\p_{21}&space;&&space;p_{22}&space;&&space;p_{23}&space;&&space;p_{24}&space;\\p_{31}&space;&&space;p_{32}&space;&&space;p_{33}&space;&&space;p_{34}&space;\\\end{bmatrix}&space;X_{i})





