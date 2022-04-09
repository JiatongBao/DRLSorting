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
    # --------------- Setup options ---------------
    is_sim = args.is_sim  # Run in simulation?
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None  # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = args.num_obj
    disturb_num = args.disturb_num
    fixed_color = args.fixed_color if is_sim else None
    tcp_host_ip = args.tcp_host_ip if not is_sim else None  # IP and port to robot arm as TCP client (UR5)
    tcp_port = args.tcp_port if not is_sim else None
    rtc_host_ip = args.rtc_host_ip if not is_sim else None  # IP and port to robot arm as real-time client (UR5)
    rtc_port = args.rtc_port if not is_sim else None
    if is_sim:
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.5]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    else:
        workspace_limits = np.asarray([[-0.10418, 0.17582], [-0.48669, -0.20669], [0.006, 0.5]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = args.heightmap_resolution  # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu
    show_heightmap = args.show_heightmap

    # ------------- Algorithm options -------------
    method = args.method  # 'reinforcement' (reinforcement learning ie Q-learning)
    push_rewards = args.push_rewards if method == 'reinforcement' else None  # Use immediate rewards (from change detection) for pushing?
    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay  # Use prioritized experience replay?
    heuristic_bootstrap = args.heuristic_bootstrap  # Use handcrafted grasping algorithm when grasping fails too many times in a row?
    explore_rate_decay = args.explore_rate_decay
    grasp_only = args.grasp_only
    random_trunk_weights_max = args.random_trunk_weights_max
    random_trunk_weights_reset_iters = args.random_trunk_weights_reset_iters
    random_trunk_weights_min_success = args.random_trunk_weights_min_success
    random_actions = args.random_actions
    max_iter = args.max_iter
    use_commonsense = args.use_commonsense

    # -------------- Testing options --------------
    is_testing = args.is_testing
    max_test_trials = args.max_test_trials  # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot  # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file) if load_snapshot else None
    continue_logging = args.continue_logging  # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations  # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True

    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, fixed_color, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                  is_testing, test_preset_cases, test_preset_file)

    # Initialize trainer
    trainer = Trainer(method, push_rewards, future_reward_discount,
                      is_testing, load_snapshot, snapshot_file, force_cpu)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale)  # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution)  # Save heightmap parameters

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = [2, 2] if not is_testing else [0, 0]
    explore_prob = 0.5 if not is_testing else 0.0

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action': False,
                          'primitive_action': None,
                          'best_pix_ind': None,
                          'push_success': False,
                          'grasp_success': False,
                          'place_success': False,
                          'grasp_correct': False,
                          'grasped_object_ind': None,
                          'grasped_object_color_ind': None,
                          'placed_object_ind': None,
                          'placed_object_color_ind': None,
                          'progress': None,
                          'progress_increase': True,
                          'new_episode': True}

    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:

                # Determine whether grasping or pushing or placing should be executed based on network predictions
                if push_predictions is not None:
                    best_push_conf = np.ma.max(push_predictions)
                    best_grasp_conf = np.ma.max(grasp_predictions)

                # Exploitation (do best action) vs exploration (do random action)
                if is_testing:
                    explore_actions = False
                else:
                    explore_actions = np.random.uniform() < explore_prob

                # If we just did a successful grasp, we always need to place
                if nonlocal_variables['primitive_action'] == 'grasp' and nonlocal_variables['grasp_success'] and nonlocal_variables['grasp_correct']:
                    nonlocal_variables['primitive_action'] = 'place'
                else:
                    nonlocal_variables['primitive_action'] = 'grasp'

                # determine if the network indicates we should do a push or a grasp,
                # otherwise if we are exploring and not placing choose between push and grasp randomly
                if not grasp_only and not nonlocal_variables['primitive_action'] == 'place':
                    if is_testing and method == 'reactive':
                        if best_push_conf > 2 * best_grasp_conf:
                            nonlocal_variables['primitive_action'] = 'push'
                    else:
                        if best_push_conf > best_grasp_conf:
                            nonlocal_variables['primitive_action'] = 'push'

                    if explore_actions:
                        push_frequency_one_in_n = 5
                        nonlocal_variables['primitive_action'] = 'push' if np.random.randint(0, push_frequency_one_in_n) == 0 else 'grasp'

                trainer.is_exploit_log.append([0 if explore_actions else 1])
                logger.write_to_log('is-exploit', trainer.is_exploit_log)

                if random_actions and explore_actions and not is_testing:
                    # explore a random action from the masked predictions
                    nonlocal_variables['best_pix_ind'], each_action_max_coordinate, predicted_value = utils_torch.action_space_explore_random(nonlocal_variables['primitive_action'], push_predictions, grasp_predictions, place_predictions)
                else:
                    # Get pixel location and rotation with highest affordance prediction from the neural network algorithms (rotation, y, x)
                    nonlocal_variables['best_pix_ind'], each_action_max_coordinate, predicted_value = utils_torch.action_space_argmax(nonlocal_variables['primitive_action'], push_predictions, grasp_predictions, place_predictions)

                # If heuristic bootstrapping is enabled: if change has not been detected more than 2 times, execute heuristic algorithm to detect grasps/pushes
                # NOTE: typically not necessary and can reduce final performance.
                if heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'push' and no_change_count[0] >= 2:
                    print('Change not detected for more than two pushes. Running heuristic pushing.')
                    nonlocal_variables['best_pix_ind'] = trainer.push_heuristic(valid_depth_heightmap)
                    no_change_count[0] = 0
                    predicted_value = push_predictions[nonlocal_variables['best_pix_ind']]
                    use_heuristic = True
                elif heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'grasp' and no_change_count[1] >= 2:
                    print('Change not detected for more than two grasps. Running heuristic grasping.')
                    nonlocal_variables['best_pix_ind'] = trainer.grasp_heuristic(valid_depth_heightmap)
                    no_change_count[1] = 0
                    predicted_value = grasp_predictions[nonlocal_variables['best_pix_ind']]
                    use_heuristic = True
                else:
                    use_heuristic = False

                trainer.use_heuristic_log.append([1 if use_heuristic else 0])
                logger.write_to_log('use-heuristic', trainer.use_heuristic_log)

                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)

                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0] * (360.0 / trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]

                primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0], best_pix_y * heightmap_resolution + workspace_limits[1][0], valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]

                # If pushing, adjust start position, and make sure z value is safe and not too low
                if nonlocal_variables['primitive_action'] == 'push':
                    finger_width = 0.02
                    safe_kernel_width = int(np.round((finger_width / 2) / heightmap_resolution))
                    local_region = valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1,valid_depth_heightmap.shape[0]),max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1])]
                    if local_region.size == 0:
                        safe_z_position = workspace_limits[2][0]
                    else:
                        safe_z_position = np.max(local_region) + workspace_limits[2][0]
                    primitive_position[2] = safe_z_position

                # Save executed primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    trainer.executed_action_log.append([0, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]])  # 0 - push
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    trainer.executed_action_log.append([1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]])  # 1 - grasp
                elif nonlocal_variables['primitive_action'] == 'place':
                    trainer.executed_action_log.append([2, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]])  # 2 - place
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if not is_testing and save_visualizations:
                    if trainer.iteration % 50 == 0:
                        if push_predictions is not None:
                            push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap,nonlocal_variables['best_pix_ind'])
                            logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
                            cv2.imwrite('visualization.push.png', push_pred_vis)
                            grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap,nonlocal_variables['best_pix_ind'])
                            logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                            cv2.imwrite('visualization.grasp.png', grasp_pred_vis)
                        if place_predictions is not None:
                            place_pred_vis = trainer.get_place_prediction_vis(place_predictions, color_heightmap,nonlocal_variables['best_pix_ind'])
                            logger.save_visualizations(trainer.iteration, place_pred_vis, 'place')
                            cv2.imwrite('visualization.place.png', place_pred_vis)

                # Initialize variables that influence reward
                nonlocal_variables['push_success'] = False
                nonlocal_variables['grasp_success'] = False
                nonlocal_variables['place_success'] = False
                nonlocal_variables['action_attribute'] = True
                nonlocal_variables['grasp_correct'] = False
                nonlocal_variables['grasped_object_ind'] = -1
                nonlocal_variables['grasped_object_color_ind'] = -1
                nonlocal_variables['placed_object_ind'] = -1
                nonlocal_variables['placed_object_color_ind'] = -1
                nonlocal_variables['progress'] = 0
                nonlocal_variables['progress_increase'] = True
                change_detected = False

                # Execute primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    nonlocal_variables['push_success'], nonlocal_variables['action_attribute'] = robot.push(primitive_position, best_rotation_angle, nonlocal_variables['best_pix_ind'][0], workspace_limits)
                    print('Push successful: %r' % (nonlocal_variables['push_success']))
                    print('Action successful: %r' % (nonlocal_variables['action_attribute']))

                elif nonlocal_variables['primitive_action'] == 'grasp':
                    nonlocal_variables['grasp_success'], nonlocal_variables['grasp_correct'], nonlocal_variables['grasped_object_ind'], nonlocal_variables['grasped_object_color_ind'] = robot.grasp(primitive_position, best_rotation_angle, nonlocal_variables['best_pix_ind'][0], workspace_limits, color_heightmap, valid_depth_heightmap, best_pix_x, best_pix_y)
                    print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))
                    print('Grasp correct: %r' % (nonlocal_variables['grasp_correct']))
                    print('Grasp Object Color: %r' % (nonlocal_variables['grasped_object_ind']))

                elif nonlocal_variables['primitive_action'] == 'place':
                    nonlocal_variables['place_success'], nonlocal_variables['placed_object_ind'], nonlocal_variables['placed_object_color_ind'], prior_place_correct = robot.place(primitive_position, best_rotation_angle, workspace_limits)
                    print('Place successful: %r' % (nonlocal_variables['place_success']))

                if nonlocal_variables['grasp_success']:
                    if not nonlocal_variables['grasp_correct']:
                        if is_sim:
                            print('Because the correctly placed object has been grasped, reset the scene!')
                            robot.restart_sim()
                            robot.add_objects()
                            trainer.clearance_log.append([trainer.iteration])
                            logger.write_to_log('clearance', trainer.clearance_log)
                            nonlocal_variables['new_episode'] = True

                nonlocal_variables['progress'] = robot.check_progress()
                print('Progress: %f' % (nonlocal_variables['progress']))

                trainer.progress_log.append([nonlocal_variables['progress']])
                logger.write_to_log('progress', trainer.progress_log)

                trainer.grasp_success_log.append([int(nonlocal_variables['grasp_success'])])
                logger.write_to_log('grasp-success', trainer.grasp_success_log)

                trainer.grasp_correct_log.append([int(nonlocal_variables['grasp_correct'])])
                logger.write_to_log('grasp-correct', trainer.grasp_correct_log)

                trainer.place_success_log.append([int(nonlocal_variables['place_success'])])
                logger.write_to_log('place-success', trainer.place_success_log)

                trainer.grasped_object_ind_log.append([nonlocal_variables['grasped_object_ind']])
                logger.write_to_log('grasped_object_ind', trainer.grasped_object_ind_log)

                trainer.grasped_object_color_ind_log.append([nonlocal_variables['grasped_object_color_ind']])
                logger.write_to_log('grasped_object_color_ind', trainer.grasped_object_color_ind_log)

                #print('Notify the main thread the action is finished.')
                nonlocal_variables['executing_action'] = False

            time.sleep(0.01)

    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------

    # variables for store information about previous place action, cause its label is calculated at a delayed step.
    recap_idx_place_label_inserted = -1
    recap_place_input = []
    recap_place_best_pix_ind = None
    recap_place_success = None
    recap_progress = None
    recap_progress_increase = True
    recap_place_grasped_color_ind = None

    # other variables
    prev_grasped_object_color_ind = -1
    counter_continuous_grasped = [0 for _ in range(100)]

    # Start main training/testing loop
    while max_iter < 0 or trainer.iteration <= max_iter:

        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim: robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img, color_heightmap, depth_heightmap, valid_depth_heightmap = get_images(robot, workspace_limits, heightmap_resolution)

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')

        # TODO
        # Get the previous scene image for place
        if 'prev_color_img' in locals() and prev_primitive_action == 'grasp' and prev_grasp_success and prev_grasp_correct:
            prev_scene_color_heightmap = prev_color_heightmap.copy()
            prev_scene_depth_heightmap = prev_valid_depth_heightmap.copy()

            before_grasp_color_heightmap = prev_color_heightmap.copy()
            before_grasp_depth_heightmap = prev_valid_depth_heightmap.copy()
            after_grasp_color_heightmap = color_heightmap.copy()
            after_grasp_depth_heightmap = valid_depth_heightmap.copy()
        else:
            prev_scene_color_heightmap = None
            prev_scene_depth_heightmap = None

        if 'prev_color_img' in locals() and prev_primitive_action == 'place':
            recap_idx_place_label_inserted = trainer.iteration - 1
            recap_place_best_pix_ind = prev_best_pix_ind
            recap_place_success = prev_place_success
            recap_progress = prev_progress
            recap_progress_increase = prev_progress_increase
            recap_place_grasped_color_ind = prev_placed_object_color_ind
            recap_place_input = [after_grasp_color_heightmap.copy(), after_grasp_depth_heightmap.copy(),
                                 before_grasp_color_heightmap.copy(), before_grasp_depth_heightmap.copy()]

        # TODO: call the function to update the status for each object whether it's correctly placed or not
        if is_sim:
            robot.check_place_correct_num()
        else:
            robot.check_real_robot_progress()

        # TODO: check if one episode is over when (1) all the objects are correctly placed or (2) some objects are correctly placed while the others are out of workspace
        if 'prev_color_img' in locals() and prev_grasp_correct:
            task_episode_complete = False
        else:
            task_episode_complete = robot.check_task_episode_complete(disturb_num)

        # TODO: if one interrupting object is continuously grasped for many times then the episode should be terminated.
        if is_sim and is_testing and 'prev_color_img' in locals() and prev_grasp_correct:
            for _i in range(len(counter_continuous_grasped)):
                if _i != prev_grasped_object_ind:
                    counter_continuous_grasped[_i] = 0
            if robot.is_interrupt_object(prev_grasped_object_ind):
                counter_continuous_grasped[prev_grasped_object_ind] += 1
                if counter_continuous_grasped[prev_grasped_object_ind] > 5:
                    task_episode_complete = True
                    counter_continuous_grasped[prev_grasped_object_ind] = 0
            else:
                counter_continuous_grasped[prev_grasped_object_ind] += 1
                if counter_continuous_grasped[prev_grasped_object_ind] > 20:
                    task_episode_complete = True
                    counter_continuous_grasped[prev_grasped_object_ind] = 0

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        # print('stuff count:', stuff_count)
        # print('stuff count sum:', np.sum(stuff_count))
        empty_threshold = 300
        if is_sim and is_testing:
            empty_threshold = 10
        if np.sum(stuff_count) < empty_threshold or task_episode_complete or (is_sim and no_change_count[0] + no_change_count[1] > 10):  # original 10
            no_change_count = [0, 0]
            if is_sim:
                #print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                print('One episode is over! Repositioning objects.')
                robot.restart_sim()
                robot.add_objects()
                if is_testing:  # If at end of test run, re-load original weights (before test run)
                    trainer.model.load_state_dict(torch.load(snapshot_file))
            else:
                print('Not enough stuff on the table (value: %d)! Flipping over bin of objects...' % (
                    np.sum(stuff_count)))
                robot.restart_real()
                robot.generate_object_index()
                time.sleep(10)
                if is_testing:  # If at end of test run, re-load original weights (before test run)
                    trainer.model.load_state_dict(torch.load(snapshot_file))

            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True  # Exit after training thread (backprop and saving labels)

            nonlocal_variables['new_episode'] = True
            continue

        # check for possible bugs in the code
        if len(trainer.reward_value_log) < trainer.iteration - 1:
            # check for progress counting inconsistencies
            print('WARNING POSSIBLE CRITICAL DETECTED: log data index and trainer.iteration out of sync!!! Experience Replay may break!'
                   'Check code for errors in indexs, continue statements etc.')
            print('reward_value_log length:', len(trainer.reward_value_log))

        if len(trainer.reward_value_log) != len(trainer.label_value_log):
            print('WARNING POSSIBLE CRITICAL DETECTED: reward_value_log data index and label_value_log data index out of sync!!! Experience Replay may break!'
                  'Check code for errors in indexs, continue statements etc.')

        if not exit_called:
            # TODO: Run forward pass with network to get affordances
            push_predictions, grasp_predictions, place_predictions, state_feat, output_prob = trainer.forward(
                color_heightmap, valid_depth_heightmap, prev_scene_color_heightmap, prev_scene_depth_heightmap,
                is_volatile=True, use_prior=use_commonsense, object_manipulated=prev_grasped_object_color_ind)

            # TODO: Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():
            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.01] = 0
            depth_diff[depth_diff > 0] = 1
            change_threshold = 300
            change_value = np.sum(depth_diff)
            change_detected = change_value > change_threshold or prev_grasp_success
            print('Change detected: %r (value: %d)' % (change_detected, change_value))

            trainer.change_detected_log.append([int(change_detected)])
            logger.write_to_log('change-detected', trainer.change_detected_log)

            if change_detected:
                if prev_primitive_action == 'push':
                    no_change_count[0] = 0
                elif prev_primitive_action == 'grasp' or prev_primitive_action == 'place':
                    no_change_count[1] = 0
            else:
                if prev_primitive_action == 'push':
                    no_change_count[0] += 1
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] += 1

            reward_multiplier = prev_progress

            # TODO: when a new episode is started, the label for the last action in the previous episode should be the immediate reward
            if nonlocal_variables['new_episode']:
                nonlocal_variables['new_episode'] = False
                if prev_primitive_action == 'grasp' or prev_primitive_action == 'push':
                    print('get label value for previous action %s and then backpropagate' % prev_primitive_action)
                    label_value, reward_value = trainer.get_label_value(prev_primitive_action, prev_push_success,
                                                                             prev_grasp_success, prev_grasp_correct,
                                                                             prev_place_success, prev_action_attribute,
                                                                             reward_multiplier, prev_progress_increase, change_detected,
                                                                             None, None,
                                                                             None, None)
                    # Backpropagate
                    trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, None, None,
                                     prev_primitive_action, prev_best_pix_ind, label_value, use_prior=use_commonsense)

                    trainer.label_value_log.append([label_value])
                    trainer.reward_value_log.append([reward_value])
                    logger.write_to_log('label-value', trainer.label_value_log)
                    logger.write_to_log('reward-value', trainer.reward_value_log)
                if recap_idx_place_label_inserted > 0:
                    print('get label value for previous action place and then backpropagate')
                    label_value, reward_value = trainer.get_label_value('place', None,
                                                                        None, None,
                                                                        recap_place_success, None,
                                                                        recap_progress, recap_progress_increase, None,
                                                                        None, None,
                                                                        None, None)
                    # Backpropagate
                    trainer.backprop(recap_place_input[0], recap_place_input[1], recap_place_input[2],
                                     recap_place_input[3],
                                     'place', recap_place_best_pix_ind, label_value, use_prior=use_commonsense,
                                     object_manipulated=recap_place_grasped_color_ind)

                    if prev_primitive_action == 'place':
                        trainer.label_value_log.append([label_value])
                        trainer.reward_value_log.append([reward_value])
                    else:
                        trainer.label_value_log[recap_idx_place_label_inserted] = [label_value]
                        trainer.reward_value_log[recap_idx_place_label_inserted] = [reward_value]

                    logger.write_to_log('label-value', trainer.label_value_log)
                    logger.write_to_log('reward-value', trainer.reward_value_log)
                    # it notice that the information of the previous place action is completed
                    recap_idx_place_label_inserted = -1
            else:
                # Compute training labels ,returns are:
                # label_value == expected_reward (with future rewards)
                # prev_reward_value == current_reward (without future rewards)
                if prev_primitive_action == 'grasp' or prev_primitive_action == 'push':
                    print('get label value for action %s and then backpropagate' % prev_primitive_action)
                    label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_push_success, prev_grasp_success, prev_grasp_correct, prev_place_success, prev_action_attribute, reward_multiplier, prev_progress_increase, change_detected, color_heightmap, valid_depth_heightmap, None, None)

                    # Backpropagate
                    trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, None, None, prev_primitive_action, prev_best_pix_ind, label_value, use_prior=use_commonsense)

                    trainer.label_value_log.append([label_value])
                    trainer.reward_value_log.append([prev_reward_value])

                    logger.write_to_log('label-value', trainer.label_value_log)
                    logger.write_to_log('reward-value', trainer.reward_value_log)

                # the values should be modified later
                if prev_primitive_action == 'place':
                    trainer.label_value_log.append([0])
                    trainer.reward_value_log.append([0])
                    logger.write_to_log('label-value', trainer.label_value_log)
                    logger.write_to_log('reward-value', trainer.reward_value_log)

                # TODO: get label for previous place action and then backpropagate
                if recap_idx_place_label_inserted > 0 and prev_primitive_action == 'grasp' and prev_grasp_success and prev_grasp_correct:
                    print('get label value for previous place action and then backpropagate')
                    label_value, reward_value = trainer.get_label_value('place', None,
                                                                        None, None,
                                                                        recap_place_success, None,
                                                                        recap_progress, recap_progress_increase, None,
                                                                        after_grasp_color_heightmap,
                                                                        after_grasp_depth_heightmap,
                                                                        before_grasp_color_heightmap,
                                                                        before_grasp_depth_heightmap)
                    # Backpropagate
                    trainer.backprop(recap_place_input[0], recap_place_input[1], recap_place_input[2], recap_place_input[3],
                                     'place', recap_place_best_pix_ind, label_value, use_prior=use_commonsense,
                                     object_manipulated=recap_place_grasped_color_ind)

                    trainer.label_value_log[recap_idx_place_label_inserted] = [label_value]
                    trainer.reward_value_log[recap_idx_place_label_inserted] = [reward_value]

                    logger.write_to_log('label-value', trainer.label_value_log)
                    logger.write_to_log('reward-value', trainer.reward_value_log)

                    # it notice that the information of the previous place action is completed
                    recap_idx_place_label_inserted = -1

            # Adjust exploration probability
            if not is_testing:
                explore_prob = max(0.5 * np.power(0.9996, trainer.iteration), 0.01) if explore_rate_decay else 0.5

            # TODO: Do sampling for experience replay
            if experience_replay and not is_testing:
                sample_primitive_action = prev_primitive_action
                log_len = trainer.iteration
                if sample_primitive_action == 'push':
                    sample_primitive_action_id = 0
                    if method == 'reactive':
                        sample_reward_value = 0 if prev_reward_value == 1 else 1  # random.randint(1, 2) # 2
                    elif method == 'reinforcement':
                        sample_reward_value = not change_detected
                    log_to_compare = np.asarray(trainer.change_detected_log)

                elif sample_primitive_action == 'grasp':
                    sample_primitive_action_id = 1
                    if method == 'reactive':
                        sample_reward_value = 0 if prev_reward_value == 1 else 1
                    elif method == 'reinforcement':
                        sample_reward_value = not prev_grasp_success
                    log_to_compare = np.asarray(trainer.grasp_success_log)

                elif sample_primitive_action == 'place':
                    sample_primitive_action_id = 2
                    log_len = trainer.iteration - 1
                    if method == 'reactive':
                        sample_reward_value = 0 if prev_reward_value == 1 else 1
                    elif method == 'reinforcement':
                        sample_reward_value = not prev_place_success
                    log_to_compare = np.asarray(trainer.place_success_log)


                # Get samples of the same primitive but with different results
                sample_ind = np.argwhere(np.logical_and(log_to_compare[0:log_len, 0] == sample_reward_value,
                                                        np.asarray(trainer.executed_action_log)[0:log_len, 0] == sample_primitive_action_id))
                if sample_ind.size > 0:
                    # Find sample with highest surprise value
                    if method == 'reactive':
                        sample_surprise_values = np.abs(
                            np.asarray(trainer.predicted_value_log)[sample_ind[:, 0]] - (1 - sample_reward_value))
                    elif method == 'reinforcement':
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:, 0]] -
                                                        np.asarray(trainer.label_value_log)[sample_ind[:, 0]])

                    sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
                    sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
                    pow_law_exp = 2
                    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1) * (sample_ind.size - 1)))
                    sample_iteration = sorted_sample_ind[rand_sample_ind]
                    print('Experience replay: iteration %d , action: %s, surprise value: %f' % (sample_iteration, str(sample_primitive_action), sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

                    # Load sample RGB-D heightmap
                    sample_color_heightmap, sample_depth_heightmap = get_history_image(logger, sample_iteration)

                    # Load sample RGB-D heightmap
                    if sample_iteration > 0 and sample_primitive_action == 'place':
                        sample_prev_scene_color_heightmap, sample_prev_scene_depth_heightmap = get_history_image(logger, sample_iteration - 1)
                    else:
                        sample_prev_scene_color_heightmap = None
                        sample_prev_scene_depth_heightmap = None

                    # Compute forward pass with sample
                    with torch.no_grad():
                        sample_push_predictions, sample_grasp_predictions, sample_place_predictions, sample_state_feat, sample_output_prob = trainer.forward(
                            sample_color_heightmap, sample_depth_heightmap, sample_prev_scene_color_heightmap,
                            sample_prev_scene_depth_heightmap, is_volatile=True, use_prior=False)

                    # Get labels for sample and backpropagate
                    sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration, 1:4]).astype(int)
                    sample_label_value = trainer.label_value_log[sample_iteration]
                    if sample_primitive_action == 'place':
                        sample_manipulated_object = trainer.grasped_object_color_ind_log[sample_iteration - 1][0]
                    else:
                        sample_manipulated_object = trainer.grasped_object_color_ind_log[sample_iteration][0]

                    trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_prev_scene_color_heightmap,
                                     sample_prev_scene_depth_heightmap, sample_primitive_action, sample_best_pix_ind,
                                     sample_label_value, use_prior=use_commonsense, object_manipulated=sample_manipulated_object)

                    # Recompute prediction value and label for replay buffer
                    if sample_primitive_action == 'push':
                        trainer.predicted_value_log[sample_iteration] = [np.ma.max(sample_push_predictions)]

                    elif sample_primitive_action == 'grasp':
                        trainer.predicted_value_log[sample_iteration] = [np.ma.max(sample_grasp_predictions)]

                    elif sample_primitive_action == 'place':
                        trainer.predicted_value_log[sample_iteration] = [np.ma.max(sample_place_predictions)]

                else:
                    print('Not enough prior training samples for', sample_primitive_action, '. Skipping experience replay.')

            # Save model snapshot
            if not is_testing:
                logger.save_backup_model(trainer.model, method)
                if trainer.iteration % 50 == 0:
                    logger.save_model(trainer.iteration, trainer.model, method)
                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        if exit_called:
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_push_success = nonlocal_variables['push_success']
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_place_success = nonlocal_variables['place_success']
        prev_action_attribute = nonlocal_variables['action_attribute']
        prev_grasp_correct = nonlocal_variables['grasp_correct']
        prev_primitive_action = nonlocal_variables['primitive_action']
        prev_progress = nonlocal_variables['progress'] / num_obj
        prev_progress_increase = nonlocal_variables['progress_increase']
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']
        prev_grasped_object_ind = nonlocal_variables['grasped_object_ind']
        prev_grasped_object_color_ind = nonlocal_variables['grasped_object_color_ind']
        prev_placed_object_ind = nonlocal_variables['placed_object_ind']
        prev_placed_object_color_ind = nonlocal_variables['placed_object_color_ind']

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))


def get_images(robot, workspace_limits, heightmap_resolution):
    color_img, depth_img = robot.get_camera_data()
    depth_img = depth_img * robot.cam_depth_scale  # Apply depth scale from calibration

    # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
    color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

    return color_img, depth_img, color_heightmap, depth_heightmap, valid_depth_heightmap

def get_history_image(logger, iteration):
    color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (iteration)))
    color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_BGR2RGB)
    depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (iteration)), -1)
    depth_heightmap = depth_heightmap.astype(np.float32) / 100000

    return color_heightmap, depth_heightmap

def display_heightmap(logger, iteration, title_name):
    color_image_data = plt.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (iteration)))
    plt.subplot(1,2,1)
    plt.imshow(color_image_data)
    depth_image_data = plt.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (iteration)))
    plt.subplot(1,2,2)
    plt.imshow(depth_image_data)
    plt.title(title_name)
    plt.show()

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
