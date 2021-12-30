#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Dec 07 21:02:07 2021

@author: Nacriema

Refs:

This script I use to debug the result of my implementation in ex4_3.py
Plot 3D points use matplotlib
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import norm

# Index_0
# X_i = np.array([[-5.02074091, 3.98980965, -29.52155139],
#                 [-4.78001023, -2.26399375, -33.79670811],
#                 [1.17998148, -3.67606085, -30.4078196]])

# Index_1
X_i = np.array([[-5.02074091, 3.98980965, -29.52155139],
                [-3.63769166, -1.72294845, -25.72002936],
                [1.17998148, -3.67606085, -30.4078196]])


def multiDimenDist(point1, point2):
    # find the difference between the two points, its really the same as below
    deltaVals = [point2[dimension] - point1[dimension] for dimension in range(len(point1))]
    runningSquared = 0
    # because the pythagarom theorm works for any dimension we can just use that
    for coOrd in deltaVals:
        runningSquared += coOrd ** 2
    return runningSquared ** (1 / 2)


def get_distance(point_a, point_b):
    return norm(point_a - point_b)


def get_angle(vect_a, vect_b):
    a_unit = vect_a / norm(vect_a)
    b_unit = vect_b / norm(vect_b)
    return np.arccos(a_unit @ b_unit)


print(get_distance(X_i[0], X_i[1]))
print(get_angle(X_i[0], X_i[1]))

print(get_distance(X_i[0], X_i[2]))
print(get_angle(X_i[0], X_i[2]))

print(get_distance(X_i[1], X_i[2]))
print(get_angle(X_i[1], X_i[2]))

distance_list = np.array([get_distance(X_i[0], X_i[1]), get_distance(X_i[0], X_i[2]), get_distance(X_i[2], X_i[1])])
angle_list = np.array([get_angle(X_i[0], X_i[1]), get_angle(X_i[0], X_i[2]), get_angle(X_i[1], X_i[2])])
print(f'Angle list: \n{angle_list}')
print(f'Distance list: \n{distance_list}')


soa = np.array([[0, 0, 0, -5.02074091, 3.98980965, -29.52155139],
                [0, 0, 0, -3.63769166, -1.72294845, -25.72002936],
                [0, 0, 0, 1.17998148, -3.67606085, -30.4078196],
                [-5.02074091, 3.98980965, -29.52155139, 1.38304925, -5.7127581,  3.80152203],
                [-3.63769166, -1.72294845, -25.72002936, 4.81767314, -1.9531124, -4.68779024]])

X, Y, Z, U, V, W = zip(*soa)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W, color=['red', 'green', 'blue', 'white', 'yellow'], arrow_length_ratio=0.05)
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-40, 0])
plt.show()
