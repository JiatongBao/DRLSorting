import cv2
import numpy as np


class Color_Detection(object):
    def __init__(self):
        self.ColorStandard = np.array([[255, 0, 0],      # red
                                       [0, 0, 255],      # blue
                                       [0, 255, 0],      # green
                                       [0, 255, 255]])   # yellow

        # Euclidean Distance, Red, Blue, Green, Yellow
        self.ColorDistance_threshold = [150, 160, 200, 200]

        # HSV space threshold
        # H_min, H_max, S_min, S_max
        self.hsv_threshold = np.array([[120, 130, 160, 180],    # red
                                       [15, 25, 85, 105],    # blue
                                       [30, 40, 200, 220],      # green
                                       [80, 90, 95, 115]])     # yellow

    # according to the pixel point to get the color
    def getPositionColor(self, color_image, depth_image, x, y, m=12):

        hsv_color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        # hsv_depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2HSV)
        hsv_depth_image = depth_image

        if x < m or x > 223-m or y < m or y > 223-m:
            m = min(abs(x-0), abs(x-223), abs(y-0), abs(y-223))
            obj_color = self.get_obj_color(hsv_color_image, hsv_depth_image, x, y, m)

        else:
            obj_color = self.get_obj_color(hsv_color_image, hsv_depth_image, x, y, m)

        return obj_color

    def get_obj_color(self, hsv_color_image, hsv_depth_image, x, y, m):

        # store the color of the sample point
        color = np.zeros((2 * m + 1, 2 * m + 1, 3), dtype=int)

        each_point_H = []
        each_point_S = []

        vaild_depth_pixel = []
        depth_threshold = 0.02      # the height of the blocks

        red_count = 0
        blue_count = 0
        green_count = 0
        yellow_count = 0

        for row in range(-m, m + 1):
            for col in range(-m, m + 1):
                if hsv_depth_image[y+col][x+row] > depth_threshold:
                    vaild_depth_pixel.append([y+col, x+row])

        if vaild_depth_pixel is not None:
            for index in range(len(vaild_depth_pixel)):
                each_point_H.append(hsv_color_image[vaild_depth_pixel[index][0], vaild_depth_pixel[index][1]][0])
                each_point_S.append(hsv_color_image[vaild_depth_pixel[index][0], vaild_depth_pixel[index][1]][1])

            for i in range(len(each_point_H)):
                if self.hsv_threshold[0][0] <= each_point_H[i] <= self.hsv_threshold[0][1]:
                    red_count += 1
                if self.hsv_threshold[1][0] <= each_point_H[i] <= self.hsv_threshold[1][1]:
                    blue_count += 1
                if self.hsv_threshold[2][0] <= each_point_H[i] <= self.hsv_threshold[2][1]:
                    green_count += 1
                if self.hsv_threshold[3][0] <= each_point_H[i] <= self.hsv_threshold[3][1]:
                    yellow_count += 1

            count = [red_count, blue_count, green_count, yellow_count]
            print('count:', count)

            if count.count(0) == 4:
                object_color = -1
            else:
                max_index = count.index(max(count))
                object_color = max_index

        else:
            # -1 represent cannot recognize the color of the grasped object
            object_color = -1

        return object_color