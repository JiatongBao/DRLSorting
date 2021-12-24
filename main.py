#!/usr/bin/env python
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import os
import random
import threading
import multiprocessing
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
import torch
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
import utils_torch
from datetime import datetime

from matplotlib import pyplot as plt

def main(args):
    # To be released

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=4,                                 help='number of objects to add to simulation')
    parser.add_argument('--fixed_color', dest='fixed_color', action='store_true', default=False,                          help='using fixed colors for objects?')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='192.168.8.113',                     help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,                           help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='192.168.8.113',                     help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,                           help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')
    parser.add_argument('--show_heightmap', dest='show_heightmap', action='store_true', default=False, help='show the background heightmap for collecting a new one and debugging')

    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='reinforcement',                               help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=False,                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,              help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,          help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)
    parser.add_argument('--random_trunk_weights_max', dest='random_trunk_weights_max', type=int, action='store', default=0, help='Max Number of times to randomly initialize the model trunk before starting backpropagaion. 0 disables this feature entirely, we have also tried 10 but more experiments are needed.')
    parser.add_argument('--random_trunk_weights_reset_iters', dest='random_trunk_weights_reset_iters', type=int, action='store', default=0, help='Max number of times a randomly initialized model should be run without success before trying a new model. 0 disables this feature entirely, we have also tried 10 but more experiements are needed.')
    parser.add_argument('--random_trunk_weights_min_success', dest='random_trunk_weights_min_success', type=int, action='store', default=4, help='The minimum number of successes we must have reached before we keep an initial set of random trunk weights.')
    parser.add_argument('--random_actions', dest='random_actions', action='store_true', default=False,                    help='By default we select both the action type randomly, like push or place, enabling random_actions will ensure the action x, y, theta is also selected randomly from the allowed regions.')
    parser.add_argument('--use_commonsense', dest='use_commonsense', action='store_true', default=False, help='use common sense to randomly explore areas with heights for push and grasp?')
    parser.add_argument('--max_iter', dest='max_iter', action='store', type=int, default=-1, help='max iter for training. -1 (default) trains indefinitely.')

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
