#!/usr/bin/env python3

import rospy
import cv2
import time
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from matplotlib import pyplot as plt
from constants import ImageConstants

class Driver():

    initial_position = [5.5, 2.5]

    kp_x = 0.0095
    kp_y = 0.0025
     
    stop_msg = Twist()
    stop_msg.linear.x = 0
    stop_msg.angular.z = 0

    def __init__(self, track) -> None:
        self.bridge = CvBridge()
        self.track = track
        
        rospy.init_node('keyboard_controller')
        # self.score_pub = rospy.Publisher('score_tracker', String , queue_size=1)
        self.velocity_pub = rospy.Publisher('R1/cmd_vel', Twist, queue_size=1)
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.twist = Twist()

        # self.reset_position()
        self.prev_center = 0

    def find_road(self, image, track, show=True):
        if track == str(1):
            lower_bound = ImageConstants.PAVED_ROAD_LOWER_BOUND
            upper_bound = ImageConstants.PAVED_ROAD_UPPER_BOUND
            # print("paved")
        else:
            lower_bound = ImageConstants.DIRT_ROAD_LOWER_BOUND
            upper_bound = ImageConstants.DIRT_ROAD_UPPER_BOUND
        
        min_area = 500
        neglecting_rect = np.array([[320-30, 340-20], [320+30, 340+20]])


        img = cv2.resize(image, (640, 360))

        # convert to grayscale
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # select the colour white
        # threshold the image
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        # find the contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # select the countours that have contour area greater than min_area
        contours = [c for c in contours if cv2.contourArea(c) > 50]

        # remove contours with long axis shorter than 20 pixels
        # find the eccentricity of the contours
        # contours = [c for c in contours if (cv2.minAreaRect(c)[1][0]/cv2.minAreaRect(c)[1][1]) <= 0.8]
        # find the center of the minimum area rectangle
        
        contours = [c for c in contours if (abs(cv2.minAreaRect(c)[0][0]-320) >= 100 or cv2.minAreaRect(c)[0][1] <= 280)]


        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:7]

        big_c = []
        small_c = []
        for c in contours:
            area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area
            if (solidity < 0.5):
                small_c.append(c)
            else:
                big_c.append(c)

            center = cv2.minAreaRect(c)[0]
            # draw the ellipse
            if show:
                # cv2.ellipse(img, ellipsis, (233,0,0), 2)
                cv2.circle(img, (int(center[0]), int(center[1])), 7, (111, 112, 111), -1)

        # find the center of the two biggest contours
        if len(contours) == 0:
            mid_point = [500, 180]
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
            # cv2.circle(img, (int(mid_point[0]), int(mid_point[1])), 1, (255, 0, 0), -1)
            cv2.rectangle(img, (320-100, 280), (320+100,360), (0, 0, 255), 3)
            cv2.drawContours(img, big_c, -1, (0, 255, 0), 3)
            cv2.drawContours(img, small_c, -1, (0, 0, 255), 3)
            cv2.circle(img, (int(mid_point[0]), int(mid_point[1])), 10, (255, 0, 0), -1)
            cv2.imshow("Image window", img)
            cv2.waitKey(1)

        # return mid_point
    
    def callback(self, data):
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        mid_point = self.find_road(img, self.track)

        # error = self.get_error(mid_point)
        # self.move(error)

    def get_error(self, road_mid_point):
        x_error = road_mid_point[0] - 320
        y_error = 300 - road_mid_point[1]
        return [x_error, y_error]
    
    def move(self, error):
        self.twist.angular.z = self.kp_x * error[0]
        self.twist.linear.x = 0.5 - self.kp_y * abs(320 - error[0])
        self.velocity_pub.publish(self.twist)

def set_params():
    track = input("Enter track path (paved=1/dirt=2): ")
    
    return track

if __name__ == '__main__': 
    track = set_params()
    driver = Driver(track)

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass