import struct
import math
import numpy as np
import warnings
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import ndimage


def action_space_argmax(primitive_action, push_predictions, grasp_predictions, place_predictions=None):
    # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
    if push_predictions is not None:
        each_action_max_coordinate = {
            'push': np.unravel_index(np.ma.argmax(push_predictions), push_predictions.shape), # push, index 0
            'grasp': np.unravel_index(np.ma.argmax(grasp_predictions), grasp_predictions.shape)
        }
        each_action_predicted_value = {
            'push': push_predictions[each_action_max_coordinate['push']], # push, index 0
            'grasp': grasp_predictions[each_action_max_coordinate['grasp']]
        }
    if place_predictions is not None:
        each_action_max_coordinate = {
            'place': np.unravel_index(np.ma.argmax(place_predictions), place_predictions.shape)
        }
        each_action_predicted_value = {
            'place': place_predictions[each_action_max_coordinate['place']]
        }

    # we will actually execute the best pixel index of the selected action
    best_pixel_index = each_action_max_coordinate[primitive_action]
    predicted_value = each_action_predicted_value[primitive_action]
    return best_pixel_index, each_action_max_coordinate, predicted_value


def random_unmasked_index_in_mask_array(maskarray):
    """ Return an index in a masked array which is selected with a uniform random distribution from the valid aka unmasked entries where the masked value is 0.
    """
    # TODO(ahundt) currently a whole new float mask is created to define the probabilities. There may be a much more efficient way to handle this.
    if np.ma.is_masked(maskarray):
        # Randomly select from only regions which are valid exploration regions
        p = (np.array(1-maskarray.mask, dtype=np.float)/np.float(maskarray.count())).ravel()     #ravel()方法将数组维度拉成一维数组
    else:
        # Uniform random across all locations
        p = None

    return np.unravel_index(np.random.choice(maskarray.size, p=p), maskarray.shape)


def action_space_explore_random(primitive_action, push_predictions, grasp_predictions, place_predictions=None):
    """ Return an index in a masked prediction arrays which is selected with a uniform random distribution from the valid aka unmasked entries where the masked value is 0. (rotation, y, x)
    """
    if push_predictions is not None:
        each_action_rand_coordinate = {
            'push': random_unmasked_index_in_mask_array(push_predictions), # push, index 0
            'grasp': random_unmasked_index_in_mask_array(grasp_predictions),
        }
        each_action_predicted_value = {
            'push': push_predictions[each_action_rand_coordinate['push']], # push, index 0
            'grasp': grasp_predictions[each_action_rand_coordinate['grasp']],
        }
    if place_predictions is not None:
        each_action_rand_coordinate = {
            'place': random_unmasked_index_in_mask_array(place_predictions)
        }
        each_action_predicted_value = {
            'place': place_predictions[each_action_rand_coordinate['place']]
        }
    # we will actually execute the best pixel index of the selected action
    best_pixel_index = each_action_rand_coordinate[primitive_action]
    predicted_value = each_action_predicted_value[primitive_action]
    return best_pixel_index, each_action_rand_coordinate, predicted_value


# def place_space_explore_random(place_predictions):
#
#     # orignal place greedy method
#     # len = place_predictions.size
#     # step = np.random.randint(15000, 20000)
#     # max_ind = np.ma.argmax(place_predictions)
#     #
#     # if max_ind > int(len/2):
#     #     explore_ind = max_ind - step
#     # else:
#     #     explore_ind = max_ind + step
#
#     # Now place greedy method
#     len = place_predictions.size
#     random_num = np.random.uniform()
#     random_index = random_num * len
#     explore_ind = int(random_index)
#
#     max_ind = np.ma.argmax(place_predictions)
#
#     # similar to e_greedy exploration
#     explore_prob = 0.2
#     random_number = np.random.uniform()
#     explore_sapce = random_number < explore_prob
#     # print('place_explore_random_number:', random_number)
#
#     if explore_sapce:
#         # print('Place_Strategy: explore (exploration probability: %f)' % (explore_prob))
#         return explore_ind
#     else:
#         # print('Place_Strategy: exploit (exploration probability: %f)' % (explore_prob))
#         return max_ind


