#!/usr/bin/env python3

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

    kp_x = 0.025
    kp_y = 0.005

    stop_twist = Twist()
    stop_twist.linear.x = 0
    stop_twist.angular.z = 0

    def __init__(self) -> None:
        self.velocity_pub = rospy.Publisher('R1/cmd_vel', Twist, queue_size=1)
        self.twist = Twist()

    def get_error(self, road_mid_point):
        x_error = road_mid_point[0] - 128
        y_error = 120 - road_mid_point[1]
        return [x_error, y_error]
    
    def move(self, error):
        self.twist.angular.z = self.kp_x * error[0]
        self.twist.linear.x = 0.25 + self.kp_y * error[1]
        self.velocity_pub.publish(self.twist)

    def drive(self, camera_image):
        error = self.get_error(ImageProcessor.find_road_paved(self, camera_image))
        self.move(error)

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

    def drive(self, camera_image):
        error = self.get_error(ImageProcessor.find_road_dirt(self, camera_image))
        self.move(error)