#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jan 12 20:13:02 2022

@author: Nacriema

Refs:

"""
import numpy as np
from numpy.linalg import inv
from math import sqrt


def kalman_filter(prev_mean, prev_cov, control_vec, observe, process_noise_cov, sensor_noise_cov, A_t, B_t, C_t):
    # Prediction step
    pred_mean = A_t @ prev_mean + B_t * control_vec
    pred_cov = A_t @ prev_cov @ A_t.T + process_noise_cov

    # Compute Kalman Gain
    K_t = pred_cov @ C_t.T @ inv(C_t @ pred_cov @ C_t.T + sensor_noise_cov)

    # Update step
    update_mean = pred_mean + K_t @ (observe - C_t @ pred_mean)
    update_cov = pred_cov - K_t @ C_t
    return pred_mean, update_mean, update_cov


if __name__ == '__main__':
    A = np.array([[1., 1], [0, 1]])
    B = np.array([[1 / 2], [0]])

    g = -9.82
    a = 1

    C_t = np.array([[1., 0]])
    z = np.array([96.4, 95.9, 94.4, 87.7, 85.3])

    pro_noise_cov = np.array([[0.1, 0], [0, 0.1]])
    sens_noise_cov = 0.5

    init_state = np.array([[95.5], [0]])
    init_cov = np.array([[0.25, sqrt(0.25 * 0.05)],
                         [sqrt(0.25 * 0.05), 0.05]])
    ctrl_vec = g + a
