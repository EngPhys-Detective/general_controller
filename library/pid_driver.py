#!/usr/bin/env python3

from typing import Any
import rospy
import cv2
import time
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from matplotlib import pyplot as plt

from image_processor import ImageProcessor
from constants import ImageConstants, ClueConstants, CNNConstants, ClueLocations

class PavedDriver():

    kp_x = 0.025/2.5
    kp_y = 0.005/2.5

    stop_twist = Twist()
    stop_twist.linear.x = 0
    stop_twist.angular.z = 0

    def __init__(self) -> None:
        self.velocity_pub = rospy.Publisher('R1/cmd_vel', Twist, queue_size=1)
        self.twist = Twist()

    def get_error(self, road_mid_point):
        x_error = road_mid_point[0] - 320
        y_error = 280 - road_mid_point[1]
        return [x_error, y_error]
    
    def move(self, error):
        self.twist.angular.z = self.kp_x * error[0]
        self.twist.linear.x = 0.25 + self.kp_y * error[1] - 0.001 * abs(error[0])
        self.velocity_pub.publish(self.twist)

    def drive(self, camera_image):
        error = self.get_error(self.find_road_paved(camera_image))
        self.move(error)

    def stop(self):
        self.velocity_pub.publish(self.stop_twist)
    
    def forward_step(self):
        self.twist.linear.x = 0.15
        self.twist.angular.z = 0
        self.velocity_pub.publish(self.twist)
        rospy.sleep(0.25)
        self.stop()

    def find_road_paved(self, image, show=True):
        lower_bound = ImageConstants.PAVED_ROAD_LOWER_BOUND
        upper_bound = ImageConstants.PAVED_ROAD_UPPER_BOUND
        min_area = 200 * 2.5

        img = cv2.resize(image, (640, 360))

        # convert to grayscale
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # select the colour white
        # threshold the image
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        # find the contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # select the countours that have contour area greater than min_area
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        # sort the contours by area
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2]

        # find the center of the two biggest contours
        if len(contours) == 0:
            mid_point = [600, 180]
        elif len(contours) < 2:
            M = cv2.moments(contours[0])
            mid_point = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
        else:
            center = []
            for c in contours[0:2]:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center.append([cX, cY])

            # find the middle point of the two centers based on the weighted average of the area of the contours
            mid_point_x = (center[0][0] * cv2.contourArea(contours[0]) + center[1][0] * cv2.contourArea(contours[1])) / (cv2.contourArea(contours[0]) + cv2.contourArea(contours[1]))
            mid_point_y = (center[0][1] * cv2.contourArea(contours[0]) + center[1][1] * cv2.contourArea(contours[1])) / (cv2.contourArea(contours[0]) + cv2.contourArea(contours[1]))
            mid_point = [mid_point_x, mid_point_y]

            # plort the center of the two biggest contours and the middle point with cv2
            if show:
                cv2.circle(img, (int(center[0][0]), int(center[0][1])), 5, (0, 0, 255), -1)
                cv2.circle(img, (int(center[1][0]), int(center[1][1])), 5, (0, 0, 255), -1)
    
        if show:
            cv2.circle(img, (int(mid_point[0]), int(mid_point[1])), 10, (255, 0, 0), -1)
            cv2.imshow("Image window", img)
            cv2.waitKey(1)

        return mid_point

class DirtDriver():

    kp_x = 0.0095
    kp_y = 0.00225

    stop_twist = Twist()
    stop_twist.linear.x = 0
    stop_twist.angular.z = 0

    def __init__(self) -> None:
        self.velocity_pub = rospy.Publisher('R1/cmd_vel', Twist, queue_size=1)
        self.twist = Twist()

    def get_error(self, road_mid_point):
        x_error = road_mid_point[0] - 320
        y_error = 300 - road_mid_point[1]
        return [x_error, y_error]
    
    def move(self, error):
        self.twist.angular.z = self.kp_x * error[0]
        self.twist.linear.x = 0.15 
        self.velocity_pub.publish(self.twist)

    def stop(self):
        self.velocity_pub.publish(self.stop_twist)
    
    def forward_step(self):
        self.twist.linear.x = 0.15
        self.twist.angular.z = 0
        self.velocity_pub.publish(self.twist)
        rospy.sleep(0.25)
        self.stop()

    def drive(self, camera_image):
        error = self.get_error(self.find_road_dirt(camera_image))
        self.move(error)

    def find_road_dirt(self, image, show=True):
        img = cv2.resize(image, (640, 360))
        neglecting_rect = np.array([[320-30, 340-20], [320-30, 340+20], [320+30, 340+20], [320+30, 340-20]])

        # convert to grayscale
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # select the colour white
        # threshold the image
        mask = cv2.inRange(hsv, ImageConstants.DIRT_ROAD_LOWER_BOUND, ImageConstants.DIRT_ROAD_UPPER_BOUND)
        # find the contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # select the countours that have contour area greater than 500
        contours = [c for c in contours if cv2.contourArea(c) > 50]
        # remove contours with long axis shorter than 20 pixels
        # find the eccentricity of the contours
        # contours = [c for c in contours if (cv2.minAreaRect(c)[1][0]/cv2.minAreaRect(c)[1][1]) <= 0.8]
        # find the center of the minimum area rectangle
        
        contours = [c for c in contours
                     if (abs(cv2.minAreaRect(c)[0][0]-320) >= 30 and abs(cv2.minAreaRect(c)[0][1]-340) >= 30)]
        # sort the contours by area
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2]

        # find the center of the two biggest contours
        if len(contours) == 0:
            mid_point = [320, 180]
        if len(contours) < 2:
            M = cv2.moments(contours[0])
            mid_point = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
        else:
            center = []
            for c in contours[0:2]:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center.append([cX, cY])

            # find the middle point of the two centers based on the weighted average of the area of the contours
            mid_point_x = (center[0][0] * cv2.contourArea(contours[0]) + center[1][0] * cv2.contourArea(contours[1])) / (cv2.contourArea(contours[0]) + cv2.contourArea(contours[1]))
            mid_point_y = (center[0][1] * cv2.contourArea(contours[0]) + center[1][1] * cv2.contourArea(contours[1])) / (cv2.contourArea(contours[0]) + cv2.contourArea(contours[1]))
            mid_point = [mid_point_x, mid_point_y]

            # plort the center of the two biggest contours and the middle point with cv2
            if show:
                cv2.circle(img, (int(center[0][0]), int(center[0][1])), 5, (0, 0, 255), -1)
                cv2.circle(img, (int(center[1][0]), int(center[1][1])), 5, (0, 0, 255), -1)
        if show:
            cv2.circle(img, (int(mid_point[0]), int(mid_point[1])), 10, (255, 0, 0), -1)

            # draw the contours on the image
            # cv2.drawRect(img, neglecting_rect, (0, 255, 0), 3)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            cv2.imshow("Image window", img)
            cv2.waitKey(1)

        return mid_point