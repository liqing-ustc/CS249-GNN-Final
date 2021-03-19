"""
Credit to jaesik817 for the physics engine.
Original implementation:
https://github.com/jaesik817/Interaction-networks_tensorflow/blob/master/physics_engine.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import torch
import random
import math
from math import cos, pi, radians, sin, ceil
import argparse
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.autograd import Variable
import matplotlib.animation as manimation
from sklearn.metrics import pairwise_distances

VAL_RATIO, TEST_RATIO = 0.25, 0.25

# number of features, [mass, x, y, vx, vy]
num_features = 5

# G
G = 10 ** 5

def init(args, train, n_body, orbit, ts):
    """
    Initialization on just the first time step; fill all other time steps with 0's
    :param n_body: number of objects
    :param num_features: number of features
    :param orbit: whether simulate planet orbit
    :return: a numpy vector of shape (ts, n_body, num_features)
    """
    data = np.zeros((ts, n_body, num_features), dtype=float)
    if orbit:
        if train:
            data[0][0][0] = args.tr_cm
        else:
            data[0][0][0] = args.tt_cm
        data[0][0][1:5] = 0.0
        for i in range(1, n_body):
            if train:
                data[0][i][0] = np.random.rand() * 8.98 + 0.02
            else:
                data[0][i][0] = np.random.rand() * 17.96 + 0.04
            distance = np.random.rand() * 90.0 + 10.0
            theta = np.random.rand() * 360
            theta_rad = pi / 2 - radians(theta)
            data[0][i][1] = distance * cos(theta_rad)
            data[0][i][2] = distance * sin(theta_rad)
            data[0][i][3] = -1 * data[0][i][2] / norm(data[0][i][1:3]) * (
                    G * data[0][0][0] / norm(data[0][i][1:3]) ** 2) * distance / 1000
            data[0][i][4] = data[0][i][1] / norm(data[0][i][1:3]) * (
                    G * data[0][0][0] / norm(data[0][i][1:3]) ** 2) * distance / 1000
    # data[0][i][3]=np.random.rand()*10.0-5.0
    # data[0][i][4]=np.random.rand()*10.0-5.0
    else:
        for i in range(n_body):
            data[0][i][0] = np.random.rand() * 8.98 + 0.02  # mass
            distance = np.random.rand() * 90.0 + 10.0
            theta = np.random.rand() * 360
            theta_rad = pi / 2 - radians(theta)
            data[0][i][1] = distance * cos(theta_rad)  # x pos
            data[0][i][2] = distance * sin(theta_rad)  # y pos
            data[0][i][3] = np.random.rand() * 6.0 - 3.0  # x vel
            data[0][i][4] = np.random.rand() * 6.0 - 3.0  # y vel
    return data


def norm(x):
    return np.sqrt(np.sum(x ** 2))


def get_f(receiver, sender):
    """
    Return gravitational force between two bodies (in vector form).
    F = G*m1*m2 / r**2
    """
    diff = sender[1:3] - receiver[1:3]  # difference in (x, y)
    distance = norm(diff)
    if distance < 1:
        distance = 1
    return G * receiver[0] * sender[0] / (distance ** 3) * diff


def calc(cur_state, n_body, dt):
    """
    Given current states of n objects, calculate their next states.
    :return: a numpy vector of shape (n_body, num_features)
    """
    next_state = np.zeros((n_body, num_features), dtype=float)
    f_mat = np.zeros((n_body, n_body, 2), dtype=float)
    f_sum = np.zeros((n_body, 2), dtype=float)
    acc = np.zeros((n_body, 2), dtype=float)
    for i in range(n_body):
        for j in range(i + 1, n_body):
            if j != i:
                # i is receiver, j is sender
                f = get_f(cur_state[i][:3], cur_state[j][:3])
                f_mat[i, j] += f
                f_mat[j, i] -= f
        f_sum[i] = np.sum(f_mat[i], axis=0)
        acc[i] = f_sum[i] / cur_state[i][0]  # F = ma
        next_state[i][0] = cur_state[i][0]
        next_state[i][3:5] = cur_state[i][3:5] + acc[i] * dt
        next_state[i][1:3] = cur_state[i][1:3] + next_state[i][3:5] * dt
    return next_state

def gen(args, train):
    """
    Return time-series data for object motions.
    :return: a numpy vector of shape (ts, n_body, num_features)
    """
    # initialize the first time step
    data = init(args, train, args.n_body, args.orbit, args.ts)

    # calculate data for remaining time steps
    for i in range(1, args.ts):
        data[i] = calc(data[i - 1], args.n_body, args.dt)
    return data


def make_video(xy, filename):
    os.system("rm -rf pics/*")
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=45, metadata=metadata)
    fig = plt.figure()
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    fig_num = len(xy)
    color = ['ro', 'bo', 'go', 'ko', 'yo', 'mo', 'co']
    with writer.saving(fig, filename, fig_num):
        for i in range(fig_num):
            for j in range(len(xy[0])):
                plt.plot(xy[i, j, 1], xy[i, j, 0], color[j % len(color)])
            writer.grab_frame()

def format_data_in(data):
    dataset = []
    for idx in range(len(data) - 1):
        obj = data[idx, :, :]
        target = data[idx + 1, :, 3:]
        position = data[idx + 1, :, :3]
        #obj = Variable(torch.FloatTensor(obj))
        #target = Variable(torch.FloatTensor(target))
        #position = Variable(torch.FloatTensor(position))
        dataset.append((obj, target, position))
    return dataset

def sample_data(dataset, ratio, flag=False):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    data_num = math.ceil(ratio*len(dataset))
    if flag:
        data_idx = indices[data_num:]
    else:
        data_idx = indices[:data_num]
    data = [dataset[i] for i in data_idx]
    return data

def get_min_dist(objs):
    pair_dist = pairwise_distances(objs[:, 1:3], metric='l2')
    min_dist = min([pair_dist[i][j] for i in range(len(pair_dist)) for j in range(len(pair_dist[i])) if i != j])
    return min_dist

def generate_videos(args, train):
    videos = []
    for i in range(args.scenes):
        data = gen(args, train)
        dataset = format_data_in(data)
        #dataset = sample_data(dataset, args.sample_ratio, args.reverse)
        videos.extend(dataset)
    return videos

def filter_videos(args, videos, train):
    filtered_videos = []
    for v in videos:
        objs = v[0]
        min_dist = get_min_dist(objs)
        # if min_dist > args.tr_dist:
        #     filtered_videos.append(v)
        if train:
            if min_dist > args.tr_dist:
                filtered_videos.append(v)
        else:
            if min_dist < args.tt_dist and min_dist > 1:
                filtered_videos.append(v)
    return filtered_videos

def generate_data(args, n_samples, train=True):
    filtered_videos = []
    while(len(filtered_videos) < n_samples):
        args.train = train
        videos = generate_videos(args, train)
        filtered_videos.extend(filter_videos(args, videos, train))
    filtered_videos = filtered_videos[:n_samples]
    return filtered_videos

def main():
    parser = argparse.ArgumentParser(description='n-body simulation')

    #Model specifications
    parser.add_argument('--n_body', type=int, default=3, help='number of objects')
    parser.add_argument('--ts', type=int, default=500, help='number of time steps per video')
    parser.add_argument('--dt', type=float, default=0.001, help='time elapse unit')
    parser.add_argument('--sample_ratio', type=float, default=0.8, help='time elapse unit')
    parser.add_argument('--scenes', type=int, default=100, help='number of videos per dataset')
    parser.add_argument('--seed', type=int, default=0, help='random seed for generating the data')
    parser.add_argument('--orbit', type=bool, default=True, help='initial condition for the objects')
    parser.add_argument('--reverse', type=bool, default=False, help='initial condition for the objects')
    parser.add_argument('--output', type=str, default='./data', help='output directory')
    parser.add_argument('--n_samples', type=int, default=10000, help='number of samples for training data')
    parser.add_argument('--tr_dist', type=float, default=30, help='the min dist of each training data must be > tr_dist')
    parser.add_argument('--tt_dist', type=float, default=20, help='the min dist of each test data must be < tt_dist')
    parser.add_argument('--tr_cm', type=int, default=100, help='mass of center object in training data')
    parser.add_argument('--tt_cm', type=int, default=200, help='mass of center object in test data')
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_data = generate_data(args, args.n_samples, train=True)
    val_data = generate_data(args, int(args.n_samples*VAL_RATIO), train=True)
    test_data = generate_data(args, int(args.n_samples*TEST_RATIO), train=False)

    output = "%s/%s.pickle" %(args.output, 'extra_mass')
    
    with open(output, 'wb') as f:
        pickle.dump((train_data, val_data, test_data), f)
    
    print("data saved to %s" %output)
        
if __name__ == '__main__':
    main()
