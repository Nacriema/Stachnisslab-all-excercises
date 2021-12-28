#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Dec 24 00:07:51 2021

@author: Nacriema

Refs:

This script is used for experiment.
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio
import timeit

gray_im = imageio.imread('images/shapes.png')
gray_im[gray_im > 100] = 1


# Write function connected_component
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


# Brush_fire a.k.a floodfill
def connected_components_1(binary_img, neighborhood):
    """
    .----> j
    |
    |
    v i
    Parameters
    ----------
    binary_img (np.ndarray): 2D binary images (0, 1)
    neighborhood (str): N4 or N8

    Returns
    -------
    k: array label for given binary image

    """
    K = 0
    N4_idx = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    N8_idx = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]

    # To avoid the numerical, then we extend the binary_img by padding 0 around it
    binary_img = np.pad(binary_img, 1, pad_with, padder=0)
    k = np.zeros_like(binary_img, dtype=np.uint8)

    nr, nc = binary_img.shape
    if neighborhood == "N4":
        N = N4_idx
    elif neighborhood == "N8":
        N = N8_idx
    else:
        raise NotImplementedError

    # 1. Find (i, j) | b(i, j) = 1, k(i + 1, j + 1) = 0
    for i in range(1, nr - 1):
        for j in range(1, nc - 1):
            if binary_img[i, j] == 1 and k[i, j] == 0:
                S = [(i, j)]
                K += 1
                k[i, j] = K
                while len(S) != 0:
                    curr_idx = S.pop()    # Index in binary image
                    for idx in N:
                        if binary_img[curr_idx[0] + idx[0], curr_idx[1] + idx[1]] == 1 and k[curr_idx[0] + idx[0], curr_idx[1] + idx[1]] == 0:
                            k[curr_idx[0] + idx[0], curr_idx[1] + idx[1]] = K
                            S.append((curr_idx[0] + idx[0], curr_idx[1] + idx[1]))
    return k[1:-1, 1:-1]


def reduce_equivalence_graph(E, A):
    replace = False
    rm_item = None
    new_item = None
    for item in E:
        if set(item).intersection(set(A)):
            replace = True
            rm_item = item
            new_item = tuple(set(item).union(set(A)))
    if replace:
        # TODO: WARNING ABOUT THE ORDER OF THESE 2 FUNCTION !!!
        E.remove(rm_item)
        E.add(new_item)
    else:
        E.add(tuple(A))


# Upgrade function
def connected_components(binary_img, neighborhood):
    """

    Parameters
    ----------
    binary_img (np.ndarray): 2D binary images (0, 1)
    neighborhood (str): N4 or N8

    Returns
    -------

    """
    K = 0
    N4_idx = [(0, -1), (-1, 0)]
    N8_idx = [(0, -1), (-1, -1), (-1, 0), (-1, 1)]
    binary_img = np.pad(binary_img, 1, pad_with, padder=0)
    k = np.zeros_like(binary_img, dtype=np.uint8)
    nr, nc = binary_img.shape
    E = set()  # Equivalent table
    if neighborhood == "N4":
        N = N4_idx
    elif neighborhood == "N8":
        N = N8_idx
    else:
        raise NotImplementedError

    for i in range(1, nr - 1):
        for j in range(1, nc - 1):
            if binary_img[i, j] == 1:
                # Search all value
                A = set()
                for idx in N:
                    # A contains labels value of neighborhood (i, j)
                    if k[i + idx[0], j + idx[1]] != 0:
                        A.add(k[i + idx[0], j + idx[1]])
                A = sorted(A)   # Sort
                if len(A) == 0:
                    # All neighbor not labelled, then assign a new label
                    K += 1
                    k[i, j] = K
                elif len(A) >= 1:
                    # Contains the labeled neighbor
                    # Copy the label, ex I copy the MIN label
                    k[i, j] = A[0]
                    # List is not hashable, so add tuple to set instead
                    # Before append, I should merge the new A with the previous one in E which has the item
                    reduce_equivalence_graph(E, A)
                    # E.add(tuple(A))
    return k[1:-1, 1:-1], E


# rs, E = connected_components(gray_im, "N4")
# print(E)
# plt.imshow(rs, cmap='gray')
# plt.show()


# Distance Transformation
def distance_transformation(binary_img, neighborhood="N4"):
    """

    Parameters
    ----------
    binary_img (np.ndarray): 2D binary images (0, 1)
    neighborhood (str): N4 or N8

    Returns
    -------
    """

    # Iterate 2 pass:
    # 1. Top - down, Left - right
    # 2. Down - top, Right - left
    # Just different with the neighbor, fist for 1st pass, later for 2nd pass

    # Init step
    d = np.pad(binary_img, 1, pad_with, padder=0)  # Use this to avoid numerical indexing
    nr, nc = d.shape

    # d array currently like the binary image mask
    # Perform 1st pass
    for r in range(1, nr - 1):
        for c in range(1, nc - 1):
            if d[r, c] == 1:
                # Compare with the other value
                # Check type
                if neighborhood == "N4":
                    d[r, c] = min(d[r, c-1] + 1, d[r-1, c] + 1)
                elif neighborhood == "N8":
                    d[r, c] = min(d[r, c-1]+1, d[r-1, c]+1, d[r-1, c-1]+1, d[r-1, c+1]+1)
                else:
                    raise NotImplementedError
    # Perform 2nd pass
    for r in range(nr - 1, 0, -1):
        for c in range(nc - 1, 0, -1):
            if d[r, c] != 0:
                if neighborhood == "N4":
                    d[r, c] = min(d[r, c], d[r, c + 1] + 1, d[r + 1, c] + 1)
                elif neighborhood == "N8":
                    d[r, c] = min(d[r, c], d[r, c+1]+1, d[r+1, c]+1, d[r+1, c+1]+1, d[r+1, c-1]+1)
                else:
                    raise NotImplementedError

    return d[1:-1, 1:-1]


N4_dis = distance_transformation(gray_im, neighborhood="N4")
N8_dis = distance_transformation(gray_im, neighborhood="N8")
EDT_Approx = 1./2 * (N4_dis + N8_dis)
plt.imshow(EDT_Approx, cmap='gray')
plt.show()

