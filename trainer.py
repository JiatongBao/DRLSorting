import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import CrossEntropyLoss2d
import utils
import utils_torch
from models import reinforcement_net
from scipy import ndimage
from models import init_trunk_weights
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, method, push_rewards, future_reward_discount,
                 is_testing, load_snapshot, snapshot_file, force_cpu):

        self.method = method
        self.is_testing = is_testing

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Fully convolutional Q network for deep reinforcement learning
        if self.method == 'reinforcement':
            self.model = reinforcement_net(self.use_cuda)
            self.push_rewards = push_rewards
            self.future_reward_discount = future_reward_discount

            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduce=False) # Huber loss
            if self.use_cuda:
                self.criterion = self.criterion.cuda()

        # Load pre-trained model
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot_file), False)
            print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []
        self.progress_log = []
        self.change_detected_log = []
        self.grasp_success_log = []
        self.grasp_correct_log = []
        self.place_success_log = []
        self.grasped_object_ind_log = []
        self.grasped_object_color_ind_log = []

        self.ACTION_TO_ID = {'push': 0, 'grasp': 1, 'place': 2}
        self.ID_TO_ACTION = {0: 'push', 1: 'grasp', 2: 'place'}

    def log_list_float_to_int(self, lst):
        res = []
        for i in range(len(lst)):
            res.append([int(lst[i][0])])
        return res

    # Pre-load execution info and RL variables
    def preload(self, transitions_directory):
        #TODO: continue from a new clear scene
        self.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
        self.clearance_log = self.clearance_log[0:self.clearance_log.size-1]
        self.clearance_log.shape = (self.clearance_log.shape[0], 1)
        self.clearance_log = self.clearance_log.tolist()
        self.iteration = int(self.clearance_log[-1][0])
        print("continue from %d" % self.iteration)

        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
        #self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration, :]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration, 1)
        self.label_value_log = self.label_value_log.tolist()

        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration, 1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]

        self.reward_value_log.shape = (self.iteration, 1)
        self.reward_value_log = self.reward_value_log.tolist()

        self.use_heuristic_log = np.loadtxt(os.path.join(transitions_directory, 'use-heuristic.log.txt'), delimiter=' ')
        self.use_heuristic_log = self.use_heuristic_log[0:self.iteration]
        self.use_heuristic_log.shape = (self.iteration, 1)
        self.use_heuristic_log = self.use_heuristic_log.tolist()
        self.is_exploit_log = np.loadtxt(os.path.join(transitions_directory, 'is-exploit.log.txt'), delimiter=' ')
        self.is_exploit_log = self.is_exploit_log[0:self.iteration]
        self.is_exploit_log.shape = (self.iteration, 1)
        self.is_exploit_log = self.is_exploit_log.tolist()


        self.progress_log = np.loadtxt(os.path.join(transitions_directory, 'progress.log.txt'), delimiter=' ')
        self.progress_log = self.progress_log[0:self.iteration]
        self.progress_log.shape = (self.iteration, 1)
        self.progress_log = self.progress_log.tolist()

        self.change_detected_log = np.loadtxt(os.path.join(transitions_directory, 'change-detected.log.txt'), delimiter=' ')
        self.change_detected_log = self.change_detected_log[0:self.iteration]
        self.change_detected_log.shape = (self.iteration, 1)
        self.change_detected_log = self.change_detected_log.tolist()

        self.grasp_success_log = np.loadtxt(os.path.join(transitions_directory, 'grasp-success.log.txt'), delimiter=' ')
        self.grasp_success_log = self.grasp_success_log[0:self.iteration]
        self.grasp_success_log.shape = (self.iteration, 1)
        self.grasp_success_log = self.grasp_success_log.tolist()

        self.grasp_correct_log = np.loadtxt(os.path.join(transitions_directory, 'grasp-correct.log.txt'), delimiter=' ')
        self.grasp_correct_log = self.grasp_correct_log[0:self.iteration]
        self.grasp_correct_log.shape = (self.iteration, 1)
        self.grasp_correct_log = self.grasp_correct_log.tolist()

        self.place_success_log = np.loadtxt(os.path.join(transitions_directory, 'place-success.log.txt'), delimiter=' ')
        self.place_success_log = self.place_success_log[0:self.iteration]
        self.place_success_log.shape = (self.iteration, 1)
        self.place_success_log = self.place_success_log.tolist()

        self.grasped_object_ind_log = np.loadtxt(os.path.join(transitions_directory, 'grasped_object_ind.log.txt'), delimiter=' ')
        self.grasped_object_ind_log = self.grasped_object_ind_log[0:self.iteration]
        self.grasped_object_ind_log.shape = (self.iteration, 1)
        self.grasped_object_ind_log = self.grasped_object_ind_log.tolist()
        self.grasped_object_ind_log = self.log_list_float_to_int(self.grasped_object_ind_log)

        self.grasped_object_color_ind_log = np.loadtxt(os.path.join(transitions_directory, 'grasped_object_color_ind.log.txt'), delimiter=' ')
        self.grasped_object_color_ind_log = self.grasped_object_color_ind_log[0:self.iteration]
        self.grasped_object_color_ind_log.shape = (self.iteration, 1)
        self.grasped_object_color_ind_log = self.grasped_object_color_ind_log.tolist()
        self.grasped_object_color_ind_log = self.log_list_float_to_int(self.grasped_object_color_ind_log)

        #utils.object_failure_ind = np.load(os.path.join(transitions_directory, 'prior-place-wrong-positions.npy'), allow_pickle=True)

    # Compute forward pass through model to compute affordances/Q
    def forward(self, color_heightmap, depth_heightmap, prev_scene_color_heightmap, prev_scene_depth_heightmap, is_volatile=False, specific_rotation=-1, use_prior=False, object_manipulated=-1):

        padding_width, color_heightmap_2x, depth_heightmap_2x, input_color_data, input_depth_data = self.heightmap_process(color_heightmap, depth_heightmap)

        is_place = False
        if prev_scene_color_heightmap is not None and prev_scene_depth_heightmap is not None:
            prev_scene_padding_width, prev_scene_color_heightmap_2x, prev_scene_depth_heightmap_2x, prev_scene_input_color_data, prev_scene_input_depth_data = self.heightmap_process(prev_scene_color_heightmap, prev_scene_depth_heightmap)
            is_place = True
        else:
            prev_scene_input_color_data, prev_scene_input_depth_data = None, None

        # Pass input data through model
        output_prob, state_feat = self.model.forward(input_color_data, input_depth_data, prev_scene_input_color_data, prev_scene_input_depth_data, is_volatile, specific_rotation)

        if self.method == 'reinforcement':

            # Return Q values (and remove extra padding)
            if is_place:
                rotate_idx = 0   # no rotation for place action
                place_predictions = output_prob[rotate_idx][2].cpu().data.numpy()[:, 0, int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2), int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)]
            else:
                for rotate_idx in range(len(output_prob)):
                    if rotate_idx == 0:
                        push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:, 0,int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2),int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)]
                        grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:, 0,int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2),int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)]
                    else:
                        push_predictions = np.concatenate((push_predictions,output_prob[rotate_idx][0].cpu().data.numpy()[:, 0,int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2),int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)]), axis=0)
                        grasp_predictions = np.concatenate((grasp_predictions,output_prob[rotate_idx][1].cpu().data.numpy()[:, 0,int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2),int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)]), axis=0)

        # Mask pixels we know cannot lead to progress
        if is_place:
            push_predictions, grasp_predictions, place_predictions = utils.common_sense_action_space_mask(depth_heightmap, None, None, place_predictions, False, color_heightmap, use_prior, object_manipulated)
        else:
            push_predictions, grasp_predictions, place_predictions = utils.common_sense_action_space_mask(depth_heightmap, push_predictions, grasp_predictions, None, False, color_heightmap, use_prior, object_manipulated)

        return push_predictions, grasp_predictions, place_predictions, state_feat, output_prob

    def get_label_value(self, primitive_action, push_success, grasp_success, grasp_correct, place_success, action_attribute, reward_multiplier, progress_increase, change_detected, curr_color_heightmap, curr_depth_heightmap, prev_color_heightmap, prev_depth_heightmap):

        '''
        reward_multiplier: default: 0.1
                           one success: 1/4
                           two success: 2/4
                           three success: 3/4
                           four success: 4/4
        '''

        if self.method == 'reinforcement':

            # Compute current reward
            current_reward = 0
            if primitive_action == 'push':
                if action_attribute:
                    if change_detected:
                        current_reward = 0.1
            elif primitive_action == 'grasp':
                if grasp_success and grasp_correct:
                    current_reward = 1.0
                elif grasp_success and not grasp_correct:
                    current_reward = -1.0
            elif primitive_action == 'place':
                if place_success:
                    current_reward = 1.5

            if curr_color_heightmap is None and prev_color_heightmap is None:
                print('Current reward: %f' % (current_reward))
                print('Expected reward: %f' % (current_reward))
                return current_reward, current_reward

            # Compute future reward
            if not change_detected and not grasp_success and not place_success:
                future_reward = 0
            else:
                is_place = False
                if prev_color_heightmap is not None and prev_depth_heightmap is not None:
                    is_place = True

                if is_place:
                    next_push_predictions, next_grasp_predictions, next_place_predictions, next_state_feat, output_prob = self.forward(curr_color_heightmap, curr_depth_heightmap, prev_color_heightmap, prev_depth_heightmap, is_volatile=True)
                    future_reward = np.max(next_place_predictions)
                else:
                    next_push_predictions, next_grasp_predictions, next_place_predictions, next_state_feat, output_prob = self.forward(curr_color_heightmap, curr_depth_heightmap, prev_color_heightmap, prev_depth_heightmap, is_volatile=True)
                    future_reward = max(np.max(next_push_predictions), np.max(next_grasp_predictions))

            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            if primitive_action == 'push' and not self.push_rewards:
                expected_reward = self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (0.0, self.future_reward_discount, future_reward, expected_reward))
            else:
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
            return expected_reward, current_reward

    # Compute labels and backpropagate
    def backprop(self, color_heightmap, depth_heightmap, prev_scene_color_heightmap, prev_scene_depth_heightmap, primitive_action, best_pix_ind, label_value, use_prior=False, object_manipulated=-1):

        if self.method == 'reinforcement':

            action_id = self.ACTION_TO_ID[primitive_action]

            is_place = False
            if prev_scene_color_heightmap is not None and prev_scene_depth_heightmap is not None:
                is_place = True

            # Compute labels
            label = np.zeros((1, 320, 320))
            action_area = np.zeros((224, 224))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            # blur_kernel = np.ones((5,5),np.float32)/25
            # action_area = cv2.filter2D(action_area, -1, blur_kernel)
            tmp_label = np.zeros((224, 224))
            tmp_label[action_area > 0] = label_value
            # these are the label values, mostly consisting of zeros, except for where the robot really went which is at best_pix_ind.
            label[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros((224, 224))
            tmp_label_weights[action_area > 0] = 1

            # If the current argmax is masked, the geometry indicates the action would not contact anything.
            # Therefore, we know the action would fail so train the argmax value with 0 reward.
            # This new common sense reward will have the same weight as the actual historically executed action.

            # Do forward pass with specified rotation (to save gradients)
            push_predictions, grasp_predictions, place_predictions, state_feat, output_prob = self.forward(color_heightmap, depth_heightmap, prev_scene_color_heightmap, prev_scene_depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0], use_prior=use_prior, object_manipulated=object_manipulated)

            new_best_pix_ind, each_action_max_coordinate, predicted_value = utils_torch.action_space_argmax(primitive_action, push_predictions, grasp_predictions, place_predictions)
            predictions = {0: push_predictions, 1: grasp_predictions, 2: place_predictions}
            if predictions[action_id].mask[each_action_max_coordinate[primitive_action]]: # mask = 1 means it's invalid.
                # The tmp_label value will already be 0, so just set the weight.
                # for backpropagating error at pixel selected at previous step
                tmp_label_weights[each_action_max_coordinate[primitive_action]] = 1

            label_weights[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label_weights

            # Compute loss and backward pass
            self.optimizer.zero_grad()
            loss_value = 0
            if self.use_cuda:
                loss = self.criterion(output_prob[0][action_id].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss = self.criterion(output_prob[0][action_id].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(), requires_grad=False)

            loss = loss.sum()

            loss.backward()
            loss_value = loss.cpu().data.numpy()

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()

    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):
        # TODO(ahundt) once the reward function is back in the 0 to 1 range, make the scale factor 1 again
        canvas = None
        num_rotations = predictions.shape[0]
        # predictions are a masked arrray, so masked regions have the fill value 0
        predictions = predictions.filled(0.0)
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                # prediction_vis[prediction_vis < 0] = 0 # assume probability
                # prediction_vis[prediction_vis > 1] = 1 # assume probability
                # Reduce the dynamic range so the visualization looks better
                prediction_vis = prediction_vis/np.max(prediction_vis)
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (221,211,238), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas


    def get_place_prediction_vis(self, predictions, color_heightmap, best_pix_ind):
        # TODO(ahundt) once the reward function is back in the 0 to 1 range, make the scale factor 1 again
        canvas = None
        num_rotations = predictions.shape[0]
        # predictions are a masked arrray, so masked regions have the fill value 0
        predictions = predictions.filled(0.0)
        for canvas_row in range(1):
            tmp_row_canvas = None
            for canvas_col in range(1):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                # prediction_vis[prediction_vis < 0] = 0 # assume probability
                # prediction_vis[prediction_vis > 1] = 1 # assume probability
                # Reduce the dynamic range so the visualization looks better
                prediction_vis = prediction_vis/np.max(prediction_vis)
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (221,211,238), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas


    def randomize_trunk_weights(self, backprop_enabled=None, random_trunk_weights_max=6, random_trunk_weights_reset_iters=10, min_success=2):
        """ Automatically re-initialize the trunk weights until we get something useful.
        """
        if self.is_testing or self.iteration > random_trunk_weights_max * random_trunk_weights_reset_iters:
            # enable backprop
            backprop_enabled = {'push': True, 'grasp': True, 'place': True}
            return backprop_enabled
        if backprop_enabled is None:
            backprop_enabled = {'push': False, 'grasp': False, 'place': False}
        if self.iteration < 2:
            return backprop_enabled
        # models_ready_for_backprop = 0
        # executed_action_log includes the action, push grasp or place, and the best pixel index
        max_iteration = np.min([len(self.executed_action_log), len(self.change_detected_log)])
        min_iteration = max(max_iteration - random_trunk_weights_reset_iters, 1)
        actions = np.asarray(self.executed_action_log)[min_iteration:max_iteration, 0]
        successful_push_actions = np.argwhere(np.logical_and(np.asarray(self.change_detected_log)[min_iteration:max_iteration, 0] == 1, actions == self.ACTION_TO_ID['push']))

        time_to_reset = self.iteration > 1 and self.iteration % random_trunk_weights_reset_iters == 0
        # we need to return if we should backprop
        if (len(successful_push_actions) >= min_success):
            backprop_enabled['push'] = True
        elif not backprop_enabled['grasp'] and time_to_reset:
            init_trunk_weights(self.model, 'push-')

        if (np.sum(np.asarray(self.grasp_success_log)[min_iteration:max_iteration, 0]) >= min_success):
            backprop_enabled['grasp'] = True
        elif not backprop_enabled['grasp'] and time_to_reset:
            init_trunk_weights(self.model, 'grasp-')

        if np.sum(np.asarray(self.place_success_log)[min_iteration:max_iteration, 0]) >= min_success:
            backprop_enabled['place'] = True
        elif not backprop_enabled['place'] and time_to_reset:
            init_trunk_weights(self.model, 'place-')

        return backprop_enabled


    def heightmap_process(self, color_heightmap, depth_heightmap):
        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2, 2, 1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2, 2], order=0)
        assert (color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - color_heightmap_2x.shape[0]) / 2)
        color_heightmap_2x_r = np.pad(color_heightmap_2x[:, :, 0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g = np.pad(color_heightmap_2x[:, :, 1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b = np.pad(color_heightmap_2x[:, :, 2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
        for c in range(3):
            input_depth_image[:, :, c] = (input_depth_image[:, :, c] - image_mean[c]) / image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)

        return padding_width, color_heightmap_2x, depth_heightmap_2x, input_color_data, input_depth_data


    def push_heuristic(self, depth_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) - rotated_heightmap > 0.02] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_push_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_push_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                push_predictions = tmp_push_predictions
            else:
                push_predictions = np.concatenate((push_predictions, tmp_push_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
        return best_pix_ind


    def grasp_heuristic(self, depth_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[np.logical_and(rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) > 0.02, rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,25], order=0) > 0.02)] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_grasp_predictions = ndimage.rotate(valid_areas, -romax_train_actionstate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_grasp_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                grasp_predictions = tmp_grasp_predictions
            else:
                grasp_predictions = np.concatenate((grasp_predictions, tmp_grasp_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        return best_pix_ind


    def place_heuristic(self, depth_heightmap):

        num_rotations = 0

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[np.logical_and(rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) > 0.02, rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,25], order=0) > 0.02)] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_place_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_place_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                place_predictions = tmp_place_predictions
            else:
                place_predictions = np.concatenate((place_predictions, tmp_place_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(place_predictions), place_predictions.shape)
        return best_pix_ind