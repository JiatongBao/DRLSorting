import socket
import select
import struct
import time
import os
import numpy as np
import utils
from simulation import vrep
import math

import urx
import math3d as m3d

import robotiq_gripper
from image_process import Color_Detection

class Robot(object):
    def __init__(self, is_sim, obj_mesh_dir, num_obj, fixed_color, workspace_limits,
                 tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                 is_testing, test_preset_cases, test_preset_file):

        self.is_sim = is_sim
        self.fixed_color = fixed_color
        self.workspace_limits = workspace_limits
        self.sim_home_position = [-0.275, 0.0, 0.45]

        # If in simulation...
        if self.is_sim:

            # Define colors for object meshes (Tableau palette)
            self.color_space = np.asarray([[255.0, 0.0, 0.0],  # red
                                           [0.0, 0.0, 255.0],  # blue
                                           [0.0, 255.0, 0.0],  # green
                                           [255.0, 255.0, 0.0]]) / 255.0  # yellow

            self.color_names = ['red', 'blue', 'green', 'yellow']

            # four boxes
            self.box_num = 4
            self.box_regions = np.asarray(
                [[-0.724, -0.574, 0.074, 0.224],
                 [-0.426, -0.276, 0.074, 0.224],
                 [-0.724, -0.574, -0.224, -0.074],
                 [-0.426, -0.276, -0.224, -0.074]])
            self.box_mesh_colors = self.color_space[np.asarray(range(self.box_num)), :]

            # Read files in object mesh directory
            self.obj_mesh_dir = obj_mesh_dir
            self.num_obj = num_obj
            self.mesh_list = os.listdir(self.obj_mesh_dir)

            # Randomly choose objects to add to scene
            self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)

            # Make sure to have the server side running in V-REP:
            # in a child script of a V-REP scene, add following command
            # to be executed just once, at simulation start:
            #
            # simExtRemoteApiStart(19999)
            #
            # then start simulation, and run this program.
            #
            # IMPORTANT: for each successful call to simxStart, there
            # should be a corresponding call to simxFinish at the end!

            # MODIFY remoteApiConnections.txt

            # Connect to simulator
            vrep.simxFinish(-1)  # Just in case, close all opened connections
            self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP on port 19997
            if self.sim_client == -1:
                print('Failed to connect to simulation (V-REP remote API server). Exiting.')
                exit()
            else:
                print('Connected to simulation.')
                self.restart_sim()

            self.is_testing = is_testing
            self.test_preset_cases = test_preset_cases
            self.test_preset_file = test_preset_file

            # Setup virtual camera in simulation
            self.setup_sim_camera()

            if self.is_testing and not self.test_preset_cases:
                self.test_box_mesh_colors = self.box_mesh_colors

            # If testing, read object meshes and poses from test case file
            if self.is_testing and self.test_preset_cases:
                file = open(self.test_preset_file, 'r')
                file_content = file.readlines()
                self.test_obj_mesh_files = []
                self.test_obj_mesh_colors = []
                self.test_obj_positions = []
                self.test_obj_orientations = []
                self.test_box_mesh_colors = []

                box_num = int(file_content[0])
                self.box_num = box_num
                for box_idx in range(box_num):
                    file_content_curr_object = file_content[box_idx+1].split()
                    self.test_box_mesh_colors.append([float(file_content_curr_object[0]), float(file_content_curr_object[1]),
                         float(file_content_curr_object[2])])

                obj_num = int(file_content[box_num+1])
                self.num_obj = obj_num
                for object_idx in range(obj_num):
                    file_content_curr_object = file_content[object_idx+box_num+2].split()
                    self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir, file_content_curr_object[0]))
                    self.test_obj_mesh_colors.append(
                        [float(file_content_curr_object[1]), float(file_content_curr_object[2]),
                         float(file_content_curr_object[3])])
                    self.test_obj_positions.append(
                        [float(file_content_curr_object[4]), float(file_content_curr_object[5]),
                         float(file_content_curr_object[6])])
                    self.test_obj_orientations.append(
                        [float(file_content_curr_object[7]), float(file_content_curr_object[8]),
                         float(file_content_curr_object[9])])
                file.close()

            # Add objects and set color for plane to simulation environment
            self.set_plane()
            self.add_objects()

            self.obj_status = [0] * self.num_obj
            self.obj_num_should_be_placed = 0
            self.calc_object_num_be_placed()

        # If in real-settings
        else:
            self.num_obj = num_obj
            self.box_num = 4

            # Set the object pose in the workspace
            # the colors of the objects are as follow: Red, Blue, Green, Yellow
            self.object_pose = [[0.02563, -0.37108, 0.26983], [-0.01982, -0.33507, 0.26983], [0.08570, -0.33043, 0.26983], [0.03044, -0.25581, 0.26983]]

            self.real_color_space = np.asarray([[255.0, 0.0, 0.0],  # red
                                                [0.0, 0.0, 255.0],  # blue
                                                [0.0, 255.0, 0.0],  # green
                                                [255.0, 255.0, 0.0]]) / 255.0  # yellow

            self.real_box_regions = np.asarray(
                [[-0.10418, -0.01418, -0.29669, -0.20669],  # red
                 [0.08582, 0.17582, -0.29669, -0.20669],  # blue
                 [0.08582, 0.17582, -0.48669, -0.39669],  # green
                 [-0.10418, -0.01418, -0.48669, -0.39669]])  # yellow

            # Connect to robot client
            self.tcp_host_ip = tcp_host_ip
            self.tcp_port = tcp_port

            # Connect as real-time client to parse state data
            self.rtc_host_ip = rtc_host_ip
            self.rtc_port = rtc_port

            # control real robot using urx
            self.rob_ur3 = urx.Robot(self.tcp_host_ip)
            # control Robotiq 2F-85 gripper
            self.robotiq_gripper = robotiq_gripper.RobotiqGripper()
            self.robotiq_gripper.connect(self.tcp_host_ip, 63352)
            self.robotiq_gripper.activate(auto_calibrate=False)

            # Default home joint configuration
            self.deg2rad = np.pi / 180
            self.home_joint_config = (0.84 * self.deg2rad, -74.51 * self.deg2rad, 57.40 * self.deg2rad, -75.32 * self.deg2rad, -89.74 * self.deg2rad, 16.45 * self.deg2rad)

            self.rob_ur3.set_tcp((0, 0, 0, 0, 0, 0))
            self.rob_ur3.set_payload(1.0, (0, 0, 0.073))

            # Default joint speed configuration
            self.joint_acc = 0.3
            self.joint_vel = 0.8

            # Move robot to home pose
            self.go_home()

            self.open_gripper()

            # Fetch RGB-D data from RealSense camera
            from real.camera import Camera
            self.camera = Camera()
            self.cam_intrinsics = self.camera.intrinsics

            calibration = False
            # Load camera pose (from running calibrate.py), intrinsics and depth scale
            if os.path.exists('real/camera_pose.txt'):
                self.cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
                self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')
            else:
                print('Hand-eye calibration!!!')
                calibration = True

            self.color_detection = Color_Detection()

            if not calibration:
                self.generate_object_index()


    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale


    def add_objects(self):

        # Randomly choose objects to add to scene
        self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
        if self.fixed_color:
            self.obj_color_ind = np.asarray(range(self.num_obj)) % 4
        else:
            self.obj_color_ind = np.random.randint(0, np.size(self.color_space, 0), size=self.num_obj)

        self.obj_mesh_color = self.color_space[self.obj_color_ind, :]

        if self.is_testing and not self.test_preset_cases:
            self.test_obj_mesh_colors = self.obj_mesh_color

        self.object_handles = []
        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        for object_idx in range(self.num_obj):
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            else:
                curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            curr_shape_name = 'shape_%02d' % object_idx

            object_position = None
            object_orientation = None
            object_color = None
            if self.is_testing and self.test_preset_cases:
                object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1], self.test_obj_positions[object_idx][2]]
                object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
                object_color = [self.test_obj_mesh_colors[object_idx][0], self.test_obj_mesh_colors[object_idx][1], self.test_obj_mesh_colors[object_idx][2]]
            else:
                drop_x_max = -0.388
                drop_x_min = -0.612
                drop_y_max = 0.112
                drop_y_min = -0.112
                drop_x = (drop_x_max - drop_x_min - 0.2) * np.random.random_sample() + drop_x_min + 0.1
                drop_y = (drop_y_max - drop_y_min - 0.2) * np.random.random_sample() + drop_y_min + 0.1
                object_position = [drop_x, drop_y, 0.15]
                object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample()]
                object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]

            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()

            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)

            if not (self.is_testing and self.test_preset_cases):
                time.sleep(2)

        self.calc_object_num_be_placed()

    # TODO: Set planes color, plane represents box in vrep
    def set_plane(self):
        for plane_idx in range(self.box_num):
            if self.is_testing and self.test_preset_cases:
                plane_color = [self.test_box_mesh_colors[plane_idx][0], self.test_box_mesh_colors[plane_idx][1], self.test_box_mesh_colors[plane_idx][2]]
            else:
                plane_color = [self.box_mesh_colors[plane_idx][0], self.box_mesh_colors[plane_idx][1], self.box_mesh_colors[plane_idx][2]]

            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer', vrep.sim_scripttype_childscript, 'setPlaneColor', [0, 0, 255, 0], plane_color, ['plane_%02d' % plane_idx], bytearray(), vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new boxes to simulation. Please restart.')
                exit()

    def restart_sim(self):

        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)


    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and gripper_position[0] < self.workspace_limits[0][1] + 0.1 and gripper_position[1] > self.workspace_limits[1][0] - 0.1 and gripper_position[1] < self.workspace_limits[1][1] + 0.1 and gripper_position[2] > self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()


    def get_task_score(self):

        key_positions = np.asarray([[-0.625, 0.125, 0.0], # red
                                    [-0.625, -0.125, 0.0], # blue
                                    [-0.375, 0.125, 0.0], # green
                                    [-0.375, -0.125, 0.0]]) #yellow

        obj_positions = np.asarray(self.get_obj_positions())
        obj_positions.shape = (1, obj_positions.shape[0], obj_positions.shape[1])
        obj_positions = np.tile(obj_positions, (key_positions.shape[0], 1, 1))

        key_positions.shape = (key_positions.shape[0], 1, key_positions.shape[1])
        key_positions = np.tile(key_positions, (1 ,obj_positions.shape[1] ,1))

        key_dist = np.sqrt(np.sum(np.power(obj_positions - key_positions, 2), axis=2))
        key_nn_idx = np.argmin(key_dist, axis=0)

        return np.sum(key_nn_idx == np.asarray(range(self.num_obj)) % 4)


    def check_goal_reached(self):

        goal_reached = self.get_task_score() == self.num_obj
        return goal_reached

    def get_color_idx(self, obj_idx):
        return self.obj_color_ind[obj_idx]

    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations

    def reposition_objects(self, workspace_limits):
        # Move gripper out of the way
        self.move_to([-0.30, 0.0, 0.45], None)

        for object_handle in self.object_handles:

            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
            time.sleep(2)


    def get_camera_data(self):

        if self.is_sim:

            # Get color image from simulation
            sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(np.float)/255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = np.fliplr(color_img)
            color_img = color_img.astype(np.uint8)

            # Get depth image from simulation
            sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.fliplr(depth_img)
            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear

        else:
            # Get color and depth image from ROS service
            color_img, depth_img = self.camera.get_data()

        return color_img, depth_img

    def close_gripper(self, asynch=False):

        if self.is_sim:
            gripper_motor_velocity = -0.5
            gripper_motor_force = 100
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            gripper_fully_closed = False
            while gripper_joint_position > -0.045: # Block until gripper is fully closed
                sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
                # print(gripper_joint_position)
                if new_gripper_joint_position >= gripper_joint_position:
                    return gripper_fully_closed
                gripper_joint_position = new_gripper_joint_position
            gripper_fully_closed = True

        else:
            self.robotiq_gripper.move_and_wait_for_pos(255, 255, 255)
            gripper_status = self.robotiq_gripper.get_current_position()
            if 160 < gripper_status < 210:
                gripper_fully_closed = False
            else:
                gripper_fully_closed = True

        return gripper_fully_closed

    def open_gripper(self, asynch=False):

        if self.is_sim:
            gripper_motor_velocity = 0.5
            gripper_motor_force = 20
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            while gripper_joint_position < 0.03: # Block until gripper is fully open
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)

        else:
            self.robotiq_gripper.move_and_wait_for_pos(110, 255, 255)  # 110

    def move_to(self, tool_position, tool_orientation):

        if self.is_sim:

            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)

            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_magnitude/0.02))

            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)

        else:

            target_pose = m3d.Transform()
            target_pose.pos = tool_position
            target_pose.orient = m3d.Orientation(tool_orientation)
            self.rob_ur3.set_pose(target_pose, acc=self.joint_acc, vel=self.joint_vel)

    def move_joints(self, joint_configuration):
        self.rob_ur3.movej(joint_configuration, self.joint_acc, self.joint_vel)

    def go_home(self):
        if self.is_sim:
            self.move_to(self.sim_home_position, None)
        else:
            self.move_joints(self.home_joint_config)

    def get_highest_object_list_index_and_handle(self):
        """
        Of the objects in self.object_handles, get the one with the highest z position and its handle.
        # Returns
           grasped_object_ind, grasped_object_handle，grasped_object_color
        """
        object_positions = np.asarray(self.get_obj_positions())
        object_positions = object_positions[:,2]
        grasped_object_ind = np.argmax(object_positions)
        grasped_object_handle = self.object_handles[grasped_object_ind]
        return grasped_object_ind, grasped_object_handle

    # Primitives ----------------------------------------------------------

    def grasp(self, position, heightmap_rotation_angle, tool_orientation, workspace_limits, color_image=None, depth_image=None, pixel_x=None, pixel_y=None):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

            # Avoid collision with floor
            position = np.asarray(position).copy()
            position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)

            # Move gripper to location above grasp target
            grasp_location_margin = 0.15
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05*move_direction/move_magnitude
            if move_step[0] > 0:
                num_move_steps = int(np.floor(move_direction[0]/move_step[0]))
            else:
                num_move_steps = 1

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

            # Ensure gripper is open
            self.open_gripper()

            # Approach grasp target
            self.move_to(position, None)

            # Close gripper to grasp target
            gripper_full_closed = self.close_gripper()

            # Move gripper to location above grasp target
            self.move_to(location_above_grasp_target, None)

            # move to the simulator home position
            self.go_home()

            # Check if grasp is successful
            gripper_full_closed = self.close_gripper()
            grasp_success = not gripper_full_closed

            if grasp_success:
                grasped_object_ind, grasped_object_handle = self.get_highest_object_list_index_and_handle()
                grasped_object_color_ind = self.get_color_idx(grasped_object_ind)
                grasp_correct = self.classify_correct(grasped_object_ind)
                if grasp_correct == 1:
                    grasp_correct = True
                elif grasp_correct == 0:
                    grasp_correct = False
            else:
                grasp_correct = False
                grasped_object_ind = -1
                grasped_object_color_ind = -1

        else:
            grasp_signal = True
            # first check whether the object waiting to be grasped is a successfully placed object
            grasping_object_color = self.get_grasp_object_color(color_image, depth_image, pixel_x, pixel_y)
            if grasping_object_color == 0 or grasping_object_color == 1 or grasping_object_color == 2 or grasping_object_color == 3:
                # if self.place_error_object_list[grasping_object_color] == 1:
                if self.check_grasp_pos_in_box(grasping_object_color, position):
                    grasp_signal = False
                    print('Will grasp the successfully placed object, pass!!')

            if pixel_x > 152 and pixel_y < 40:
                grasp_signal = False
                print('Protect Robot!!! PASS')

            if grasp_signal:
                tool_rotation_angle = self.real_robot_tcp_posture(tool_orientation)
                # correction z
                position[2] = position[2] + 0.24
                if position[2] < 0.27:
                    position[2] = 0.27

                position[2] = max(position[2], workspace_limits[2][0])
                position[2] = min(position[2], workspace_limits[2][1])

                self.open_gripper()

                # move gripper to location above grasp target
                grasp_location_margin = 0.03
                location_above_position = [position[0], position[1], position[2] + grasp_location_margin]

                tool_orientation = tool_rotation_angle

                self.move_to(location_above_position, tool_orientation)

                # Attempt grasp
                position = np.asarray(position).copy()
                position[2] = max(position[2], workspace_limits[2][0])

                self.move_to(position, tool_orientation)

                self.close_gripper()

                # move gripper to location above grasp target
                self.move_to(location_above_position, tool_orientation)

                self.go_home()

                # Check if grasp is successful
                gripper_full_closed = self.close_gripper()
                grasp_success = not gripper_full_closed

                if grasp_success:
                    self.real_grasped_object_color = self.get_grasp_object_color(color_image, depth_image, pixel_x, pixel_y)
                    if self.check_grasp_pos_in_box(grasping_object_color, position):
                        grasp_correct = False
                    else:
                        grasp_correct = True
                    grasped_object_ind, grasped_object_color_ind = self.real_grasped_object_color, self.real_grasped_object_color
                else:
                    grasp_correct = False
                    grasped_object_ind, grasped_object_color_ind = -1, -1

            else:
                grasp_success = True
                grasp_correct = False
                grasped_object_ind, grasped_object_color_ind = -1, -1

        return grasp_success, grasp_correct, grasped_object_ind, grasped_object_color_ind


    def push(self, position, heightmap_rotation_angle, tool_orientation, workspace_limits):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

            # Adjust pushing point to be on tip of finger
            position[2] = position[2] + 0.026

            #Check push position is in reliable region
            action_success = self.check_push(position)
            if action_success == 0:
                action_attribute = False
            elif action_success == 1:
                action_attribute = True

            if action_attribute:

                # Compute pushing direction
                push_orientation = [1.0,0.0]
                push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle)])

                # Move gripper to location above pushing point
                pushing_point_margin = 0.1
                location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

                # Compute gripper position and linear movement increments
                tool_position = location_above_pushing_point
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
                move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
                move_magnitude = np.linalg.norm(move_direction)
                move_step = 0.05*move_direction/move_magnitude
                num_move_steps = int(np.floor(move_direction[0]/move_step[0]))

                # Compute gripper orientation and rotation increments
                sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
                rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
                num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

                # Simultaneously move and rotate gripper
                for step_iter in range(max(num_move_steps, num_rotation_steps)):
                    vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                    vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

                # Ensure gripper is closed
                self.close_gripper()

                # Approach pushing point
                self.move_to(position, None)

                # Compute target location (push to the right)
                push_length = 0.1    #original 0.1
                target_x = min(max(position[0] + push_direction[0]*push_length, workspace_limits[0][0]), workspace_limits[0][1])
                target_y = min(max(position[1] + push_direction[1]*push_length, workspace_limits[1][0]), workspace_limits[1][1])
                push_length = np.sqrt(np.power(target_x-position[0],2)+np.power(target_y-position[1],2))

                # Move in pushing direction towards target location
                self.move_to([target_x, target_y, position[2]], None)

                # Move gripper to location above grasp target
                self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

                push_success = True

            else:
                push_success = False

        else:

            pass

            push_success = True
            action_attribute = True

        return push_success, action_attribute

    def place(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: place at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:

            #Ensure gripper is closed
            gripper_fully_closed = self.close_gripper()
            if gripper_fully_closed:
                print('Warning: gripper is fully closed')
                return False, -1, -1, None

            #If the object has been grasped it should be the highest object and held by the gripper
            grasped_object_ind, grasped_object_handle = self.get_highest_object_list_index_and_handle()
            grasped_object_color_ind = self.get_color_idx(grasped_object_ind)

            sim_ret, grasped_object_position = vrep.simxGetObjectPosition(self.sim_client, grasped_object_handle, -1, vrep.simx_opmode_blocking)
            grasped_object_position = np.array(grasped_object_position)

            # Avoid collision with floor
            position[2] = max(position[2] + 0.04 + 0.02, workspace_limits[2][0] + 0.02)

            # Move gripper to location above place target
            place_location_margin = 0.1
            location_above_place_target = (position[0], position[1], position[2] + place_location_margin)

            self.move_to(location_above_place_target, heightmap_rotation_angle)

            # Approach place target
            self.move_to(position, None)

            # Ensure gripper is open
            self.open_gripper()

            # Move gripper to location above place target
            self.move_to(location_above_place_target, None)

            #move to the simulator home position
            self.go_home()

            sim_ret, placed_object_position = vrep.simxGetObjectPosition(self.sim_client, grasped_object_handle, -1, vrep.simx_opmode_blocking)
            placed_object_position = np.array(placed_object_position)

            has_moved = np.linalg.norm(placed_object_position - grasped_object_position, axis=0) > 0.03   #0.03 is the distance to cide whether move

            place_correct = self.check_place_correct(grasped_object_ind, placed_object_position)
            prior_place_correct = self.check_place_correct(grasped_object_ind, position)

            if has_moved and place_correct:
                place_success = True
            else:
                place_success = False

            return place_success, grasped_object_ind, grasped_object_color_ind, prior_place_correct
        else:
            tool_rotation_angle = [2.3425, 2.1167, -0.0301]  # fix orientation
            # correction z
            position[2] = position[2] + 0.24
            if position[2] < 0.28:
                position[2] = 0.28
            if position[2] > 0.3:
                position[2] = 0.3

            # move gripper to location above place target
            place_location_margin = 0.02
            position[2] = max(position[2], workspace_limits[2][0])
            position[2] = min(position[2], workspace_limits[2][1])

            location_above_position = [position[0], position[1], position[2] + place_location_margin]

            tool_orientation = tool_rotation_angle

            self.move_to(location_above_position, tool_orientation)

            # Attempt place
            position = np.asarray(position).copy()
            position[2] = max(position[2], workspace_limits[2][0])

            self.move_to(position, tool_orientation)

            self.open_gripper()

            # move gripper to location above grasp target
            self.move_to(location_above_position, tool_orientation)

            self.go_home()

            place_success = self.check_place_correct(self.real_grasped_object_color, position)

            if place_success:

                self.real_place_correct_count += 1

            return place_success, -1, -1, -1

    def real_robot_tcp_posture(self, index):
        tool_posture = [[2.3425, 2.1167, -0.0301], [1.2106, 2.8753, 0.0060], [0.0118, -3.1532, 0.0214], [2.8730, 1.2700, -0.0254], [2.2853, 2.1770, -0.0615]]
        tool_posture_index = int((index % 8) / 2)
        tool_orientation = tool_posture[tool_posture_index]

        return tool_orientation

    def check_place_correct(self, grasped_object_ind, placed_object_position):
        if self.is_sim:
            if self.is_testing:
                grasped_object_color = self.test_obj_mesh_colors[grasped_object_ind]
            else:
                grasped_object_color = self.obj_mesh_color[grasped_object_ind]

            is_in_box = False
            placed_box_idx = -1
            for box_idx in range(self.box_num):
                if self.box_regions[box_idx][0] < placed_object_position[0] < self.box_regions[box_idx][1] and self.box_regions[box_idx][2] < placed_object_position[1] < self.box_regions[box_idx][3]:
                    is_in_box = True
                    placed_box_idx = box_idx
                    break
            if not is_in_box:
                return False

            if self.is_testing:
                placed_box_color = self.test_box_mesh_colors[placed_box_idx]
            else:
                placed_box_color = self.box_mesh_colors[placed_box_idx]

            if self.compare_color(grasped_object_color, placed_box_color):
                return True
            return False
        else:
            # wait to correct!!!!!!
            is_in_box = False
            if grasped_object_ind > 3:
                is_in_box = False
            else:
                if self.real_box_regions[grasped_object_ind][0] <= placed_object_position[0] <= self.real_box_regions[grasped_object_ind][1] and self.real_box_regions[grasped_object_ind][2] <= placed_object_position[1] <= self.real_box_regions[grasped_object_ind][3]:
                    is_in_box = True

            return is_in_box

    def check_grasp_pos_in_box(self, grasped_object_ind, grasp_object_position):
        in_box = False
        if grasped_object_ind < 0:
            in_box = False
        else:
            if self.real_box_regions[grasped_object_ind][0] <= grasp_object_position[0] <= self.real_box_regions[grasped_object_ind][1] and self.real_box_regions[grasped_object_ind][2] <= grasp_object_position[1] <= self.real_box_regions[grasped_object_ind][3]:
                in_box = True

        return in_box

    def classify_correct(self, grasped_object_ind):
        grasp_correct = 1
        if self.obj_status[grasped_object_ind]:
            grasp_correct = 0
        return grasp_correct

    def check_push(self, position):
        action_success = 1
        if self.is_sim:
            for box_idx in range(self.box_num):
                if self.box_regions[box_idx][0] < position[0] < self.box_regions[box_idx][1] and self.box_regions[box_idx][2] < position[1] < self.box_regions[box_idx][3]:
                    action_success = 0
                    break
        else:
            for box_idx in range(self.box_num):
                if self.real_box_regions[box_idx][0] < position[0] < self.real_box_regions[box_idx][1] and self.real_box_regions[box_idx][2] < position[1] < self.real_box_regions[box_idx][3]:
                    action_success = 0
                    break
        return action_success

    def check_progress(self):
        if self.is_sim:
            success_number = self.check_place_correct_num()
            if success_number > 0:
                progress = success_number
            else:
                progress = 0.1
        else:
            success_number = self.check_real_robot_progress()
            if success_number > 0:
                progress = success_number
            else:
                progress = 0.1

        return progress

    def check_place_correct_num(self):
        obj_positions = np.asarray(self.get_obj_positions())
        correct_count = 0
        for obj_idx in range(self.num_obj):
            self.obj_status[obj_idx] = 0
            if self.check_place_correct(obj_idx, obj_positions[obj_idx]):
                self.obj_status[obj_idx] = 1
                correct_count += 1
        return correct_count

    def compare_color(self, c1, c2):
        if c1[0] == c2[0] and c1[1] == c2[1] and c1[2] == c2[2]:
            return True
        return False

    def is_interrupt_object(self, obj_idx):
        res = True
        if self.is_testing:
            obj_color = self.test_obj_mesh_colors[obj_idx]
        else:
            obj_color = self.obj_mesh_color[obj_idx]
        for box_idx in range(self.box_num):
            if self.is_testing:
                box_color = self.test_box_mesh_colors[box_idx]
            else:
                box_color = self.box_mesh_colors[box_idx]
            if self.compare_color(obj_color, box_color):
                res = False
                break
        return res

    def calc_object_num_be_placed(self):
        cnt = 0
        for obj_idx in range(self.num_obj):
            if self.is_testing:
                obj_color = self.test_obj_mesh_colors[obj_idx]
            else:
                obj_color = self.obj_mesh_color[obj_idx]
            for box_idx in range(self.box_num):
                if self.is_testing:
                    box_color = self.test_box_mesh_colors[box_idx]
                else:
                    box_color = self.box_mesh_colors[box_idx]
                if self.compare_color(obj_color, box_color):
                    cnt += 1

        self.obj_num_should_be_placed = cnt

    # TODO: need to considering the number of interference objects
    def check_task_episode_complete(self, interference_num=0):
        if self.is_sim:
            obj_positions = np.asarray(self.get_obj_positions())
            count = 0
            out_count = 0
            interrupt_count = 0
            for obj_idx in range(self.num_obj):
                if self.check_place_correct(obj_idx, obj_positions[obj_idx]):
                    count += 1

                # if self.is_testing:
                #     if self.is_interrupt_object(obj_idx):
                #         interrupt_count += 1
                if not self.is_interrupt_object(obj_idx):
                    if obj_positions[obj_idx][0] < self.workspace_limits[0][0] or obj_positions[obj_idx][0] > self.workspace_limits[0][1] or obj_positions[obj_idx][1]<self.workspace_limits[1][0] or obj_positions[obj_idx][1] > self.workspace_limits[1][1]:
                        out_count += 1

            print('success count: %d / %d' % (count, self.obj_num_should_be_placed))
            return count == self.obj_num_should_be_placed or count + out_count == self.obj_num_should_be_placed
        else:
            # need to considering the number of interference objects
            need_to_place = self.num_obj - interference_num
            return need_to_place == self.check_real_robot_progress()

    def get_grasp_object_color(self, color_image, depth_image, pixel_x, pixel_y):
        object_color = self.color_detection.getPositionColor(color_image, depth_image, pixel_x, pixel_y)
        return object_color

    def restart_real(self):
        self.open_gripper()

    def generate_object_index(self):

        self.real_place_correct_count = 0

    def check_real_robot_progress(self):
        return self.real_place_correct_count


