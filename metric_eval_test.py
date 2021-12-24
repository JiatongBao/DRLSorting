# Data processing is based on episode.

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import os


ACTION_TO_ID = {'push': 0, 'grasp': 1, 'place': 2}


def read_txt(path):
    file = open(path, "r")
    list_original = file.readlines()
    list_process = []
    for fields in list_original:
        fields = fields.strip()
        fields = fields.strip("\n")
        fields = int(float(fields))
        list_process.append(fields)

    return list_process


def calcul_action_efficiency(clearance_log, progress_log, executed_actions_log):
    actions_efficiency = 0
    start = 0
    push_count = executed_actions_log[:, 0] == ACTION_TO_ID['push']
    for i in clearance_log:
        ideal_actions_num = progress_log[i-1] * 2
        push_count_episode = push_count[start:i].sum()
        efficiency = ideal_actions_num / (i - start - push_count_episode)
        actions_efficiency += efficiency
        start = i
    average_efficiency = actions_efficiency / len(clearance_log)

    return average_efficiency


def calcul_grasp_place_suceess_rate(clearance_log, executed_actions_log, grasp_success_log, place_success_log):
    grasp_success_rate = 0
    place_success_rate = 0
    grasp_count = executed_actions_log[:, 0] == ACTION_TO_ID['grasp']
    place_count = executed_actions_log[:, 0] == ACTION_TO_ID['place']
    start = 0
    count_0_number = 0
    for i in clearance_log:
        trial_grasp_count = grasp_count[start:i].sum()
        trial_place_count = place_count[start:i].sum()
        trial_grasp_success = grasp_success_log[start:i].count(1)
        trial_place_success = place_success_log[start:i].count(1)
        trial_grasp_success_rate = trial_grasp_success / trial_grasp_count
        if trial_place_count != 0:
            trial_place_success_rate = trial_place_success / trial_place_count
        else:
            trial_place_success_rate = 0
            count_0_number += 1
        grasp_success_rate += trial_grasp_success_rate
        place_success_rate += trial_place_success_rate
        start = i

    average_grasp_success_rate = grasp_success_rate / len(clearance_log)
    average_place_success_rate = place_success_rate / (len(clearance_log) - count_0_number)

    return average_grasp_success_rate, average_place_success_rate


def calcul_sort_success_rate(clearance_log, progress_log, num_obj=4):
    sort_success_rate = 0
    for i in clearance_log:
        sort_success_num = progress_log[i-1]
        success_sort_episode = sort_success_num / num_obj
        sort_success_rate += success_sort_episode

    average_sort_success_rate = sort_success_rate / len(clearance_log)

    return average_sort_success_rate


def main(args):
    log_dir = args.log_dir
    object_num = args.object_num

    clearance_log = read_txt(os.path.join(log_dir, 'clearance.log.txt'))
    progress_log = read_txt(os.path.join(log_dir, 'progress.log.txt'))
    grasp_success_log = read_txt(os.path.join(log_dir, 'grasp-success.log.txt'))
    place_success_log = read_txt(os.path.join(log_dir, 'place-success.log.txt'))
    executed_actions_log = np.loadtxt(os.path.join(log_dir, 'executed-action.log.txt'))

    actions_efficiency = calcul_action_efficiency(clearance_log, progress_log, executed_actions_log)
    grasp_success_rate, place_success_rate = calcul_grasp_place_suceess_rate(clearance_log, executed_actions_log, grasp_success_log, place_success_log)
    sort_success_rate = calcul_sort_success_rate(clearance_log, progress_log, num_obj=object_num)

    print('actions_efficiency:', actions_efficiency)
    print('grasp_success_rate:', grasp_success_rate)
    print('place_success_rate:', place_success_rate)
    print('sort_success_rate:', sort_success_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing after training or testing.')

    parser.add_argument('--log_dir', dest='log_dir', action='store', default=None,
                           help='directory containing logger should be to process.')

    parser.add_argument('--object_num', dest='object_num', type=int, action='store', default=4,
                        help='number of objects to add to simulation')

    args = parser.parse_args()
    main(args)