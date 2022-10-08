#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot_belief(belief):
    plt.figure()

    ax = plt.subplot(2, 1, 1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")

    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


def motion_model(action, belief):
    # TODO: add code here
    p_ff = p_bb = 0.7  # correct direction
    p_fb = p_bf = 0.1  # opposite direction
    p_s = 0.2  # stay

    belief_out = np.zeros_like(belief)

    # Update the belief based on the belief in the past and the probability
    if action == 'F':    # Anchor is F
        for i in range(len(belief)):
            if 0 < i < len(belief) - 2:
                belief_out[i] = belief[i - 1] * p_ff + belief[i + 1] * p_fb + belief[i] * p_s
            elif i == 0:
                belief_out[i] = belief[i + 1] * p_fb + belief[i] * p_s
            else:
                belief_out[i] = belief[i - 1] * p_ff + belief[i] * p_s
    else:
        for i in range(len(belief)):
            if 0 < i < len(belief) - 2:
                belief_out[i] = belief[i - 1] * p_bf + belief[i + 1] * p_bb + belief[i] * p_s
            elif i == 0:
                belief_out[i] = belief[i + 1] * p_bb + belief[i] * p_s
            else:
                belief_out[i] = belief[i - 1] * p_bf + belief[i] * p_s
    return belief_out


def sensor_model(observation, belief, world):
    # TODO: add code here
    """ Adjust the belief based on the observation model
    Observation - the signal that robot see at specific time
    """

    belief_out = np.zeros_like(belief)
    for i in range(len(belief)):
        if observation == 0:    # See Black
            if world[i] == 0:   # Black
                belief_out[i] = 0.9 * belief[i]
            else:  # White
                belief_out[i] = 0.3 * belief[i]
        else:  # See White
            if world[i] == 0:  # Black
                belief_out[i] = 0.1 * belief[i]
            else:
                belief_out[i] = 0.7 * belief[i]

    return belief_out / sum(belief_out)


def recursive_bayes_filter(actions, observations, belief, world):
    # TODO: add code here
    # Initial position observation/sensor model
    belief_correct = sensor_model(observations[0], belief, world)

    for i, action in enumerate(actions):
        predict_belief = motion_model(action, belief_correct)
        belief_correct = sensor_model(observations[i + 1], predict_belief, world)
    return belief_correct


# if __name__ == '__main__':
#     belief = np.zeros(15)
#
#     # initial known position
#     x_start = 7
#     belief[x_start] = 1.0
#
#     actions = ['F', 'F', 'F', 'F', 'B', 'B', 'F', 'F', 'B']
#
#     for action in actions:
#         belief = motion_model(action, belief)
#         print(np.argmax(belief))

