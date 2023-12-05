#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

from image_processor import ImageProcessor
from constants import ImageConstants

class PavedDriver():

    kp_x = 0.0145 # WORKING DO NOT CHANGE HERE
    kp_yx = 0.000675 # WORKING DO NOT CHANGE HERE
    kp_yy = 0.00195 # WORKING DO NOT CHANGE HERE

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
        if error[0] >= 320:
            self.sharp_turn_left()
        else:
            # WORKING DO NOT CHANGE HERE
            self.twist.angular.z = self.kp_x * error[0]
            # WORKING DO NOT CHANGE HERE
            self.twist.linear.x = 0.295 - self.kp_yy * abs(error[1]) - self.kp_yx * abs(error[0]) 
            self.velocity_pub.publish(self.twist)

    def drive(self, camera_image): 
        error = self.get_error(self.find_road_paved(camera_image))
        self.move(error)

    def stop(self):
        self.velocity_pub.publish(self.stop_twist)
    
    def slow_down(self):
        self.twist.linear.x = 0.04
        self.twist.angular.z = 0
        self.velocity_pub.publish(self.twist)
    
    def speed_up(self): 
        self.twist.linear.x = 0.9
        self.twist.angular.z = 0
        self.velocity_pub.publish(self.twist)
        rospy.sleep(0.75)
    
    def turn_right_slightly(self):
        self.twist.linear.x = 0.125
        self.twist.angular.z = -0.4
        self.velocity_pub.publish(self.twist)
        rospy.sleep(0.4)
        
    def speed_up_before_pink(self): # WORKING DO NOT CHANGE
        self.twist.linear.x = 0.3
        self.twist.angular.z = 0
        self.velocity_pub.publish(self.twist)
        rospy.sleep(0.5)
    
    def sharp_turn_left(self):
        self.twist.linear.x = 0.15
        self.twist.angular.z = 0.775
        self.velocity_pub.publish(self.twist)
        rospy.sleep(0.4)

    def find_road_paved(self, image, show=True):
        lower_bound = ImageConstants.PAVED_ROAD_LOWER_BOUND
        upper_bound = ImageConstants.PAVED_ROAD_UPPER_BOUND
        min_area = 500

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
        line_contours = []
        if len(contours) == 0:
            mid_point = [640, 280]
        elif len(contours) < 2:
            M = cv2.moments(contours[0])
            mid_point = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
            line_contours.append(contours[0])
        else:
            center = []
            for c in contours[0:2]:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center.append([cX, cY])
                line_contours.append(c)

            # find the middle point of the two centers based on the weighted average of the area of the contours
            mid_point_x = (center[0][0] * cv2.contourArea(contours[0]) + center[1][0] * cv2.contourArea(contours[1])) / (cv2.contourArea(contours[0]) + cv2.contourArea(contours[1]))
            mid_point_y = (center[0][1] * cv2.contourArea(contours[0]) + center[1][1] * cv2.contourArea(contours[1])) / (cv2.contourArea(contours[0]) + cv2.contourArea(contours[1]))
            mid_point = [mid_point_x, mid_point_y]

            # plort the center of the two biggest contours and the middle point with cv2
            if show:
                cv2.drawContours(img, line_contours, -1, (0, 255, 0), 3)
                cv2.circle(img, (int(center[0][0]), int(center[0][1])), 5, (0, 0, 255), -1)
                cv2.circle(img, (int(center[1][0]), int(center[1][1])), 5, (0, 0, 255), -1)
    
        if show:
            cv2.circle(img, (int(mid_point[0]), int(mid_point[1])), 10, (255, 0, 0), -1)
            cv2.imshow("Image window", img)
            cv2.waitKey(1)

        return mid_point

class DirtDriver():

    kp_x = 0.0135 # WORKING DO NOT CHANGE HERE
    kp_yx = 0.001 # WORKING DO NOT CHANGE HERE
    kp_yy = 0.00125 # WORKING DO NOT CHANGE HERE

    stop_twist = Twist()
    stop_twist.linear.x = 0
    stop_twist.angular.z = 0

    def __init__(self) -> None:
        self.velocity_pub = rospy.Publisher('R1/cmd_vel', Twist, queue_size=1)
        self.teleporter = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.twist = Twist()

    def get_error(self, road_mid_point):
        x_error = road_mid_point[0] - 320
        y_error = 280 - road_mid_point[1]
        return [x_error, y_error]
    
    def move(self, error):
        if abs(error[0]) < 15:
            self.twist.angular.z = (self.kp_x-0.002) * error[0]
            self.twist.linear.x = 0.5
        else:
            # WORKING DO NOT CHANGE HERE
            self.twist.angular.z = self.kp_x * error[0]
            # WORKING DO NOT CHANGE HERE
            self.twist.linear.x = 0.25 - self.kp_yx * abs(error[0]) 
            self.velocity_pub.publish(self.twist)

    def stop(self):
        self.velocity_pub.publish(self.stop_twist)
    
    def slow_down(self):
        self.twist.linear.x = 0.035
        self.twist.angular.z = 0
        self.velocity_pub.publish(self.twist)
    
    def speed_up(self): # WORKING DO NOT CHANGE
        self.twist.linear.x = 0.3
        self.twist.angular.z = 0
        self.velocity_pub.publish(self.twist)
        rospy.sleep(0.5)

    def drive(self, camera_image):
        error = self.get_error(self.find_road_dirt(camera_image))
        self.move(error)

    def find_road_dirt(self, image, show=True):
        img = cv2.resize(image, (640, 360))
        min_area = 500
        # convert to grayscale
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # select the colour white
        # threshold the image
        mask = cv2.inRange(hsv, ImageConstants.DIRT_ROAD_LOWER_BOUND, ImageConstants.DIRT_ROAD_UPPER_BOUND)
        # find the contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # remove contours with long axis shorter than 20 pixels
        # find the eccentricity of the contours
        # contours = [c for c in contours if (cv2.minAreaRect(c)[1][0]/cv2.minAreaRect(c)[1][1]) <= 0.8]
        # find the center of the minimum area rectangle
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        contours = [c for c in contours if (abs(cv2.minAreaRect(c)[0][0]-320) >= 110 or cv2.minAreaRect(c)[0][1] <= 280)]

        # sort the contours by area
        contours = sorted(contours, key = lambda c: max(cv2.minAreaRect(c)[1][0], cv2.minAreaRect(c)[1][1]), reverse = True)[:2]

        # find the center of the two biggest contours
        if len(contours) == 0:
            mid_point = [320, 180]
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

            # draw the contours on the image
            cv2.rectangle(img, (320-110, 280), (320+110,360), (0, 0, 255), 3)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            cv2.imshow("Image window", img)
            cv2.waitKey(1)

        return mid_point
    
    def teleport_to_mountain(self):
            model_state = ModelState()
            model_state.model_name = "R1"
            model_state.pose.position.x = -4.05
            model_state.pose.position.y = -2.27
            model_state.pose.position.z = 0.0525
            model_state.pose.orientation.x = 0.0
            model_state.pose.orientation.y = 0.0
            model_state.pose.orientation.z = 0.0
            model_state.pose.orientation.w = 0.0
            self.teleporter(model_state)
            rospy.sleep(0.25)
    
class MountainDriver():

    kp_x = 0.0075
    kp_y = 0.000225
     
    stop_msg = Twist()
    stop_msg.linear.x = 0
    stop_msg.angular.z = 0
    
    def __init__(self) -> None:
        self.velocity_pub = rospy.Publisher('R1/cmd_vel', Twist, queue_size=1)
        self.teleporter = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.twist = Twist()
    
    def stop(self):
        self.velocity_pub.publish(self.stop_msg)

    def slow_down(self):
        self.twist.linear.x = 0.035
        self.twist.angular.z = 0
        self.velocity_pub.publish(self.twist)
    
    def speed_up(self): # WORKING DO NOT CHANGE
        self.twist.linear.x = 0.95
        self.twist.angular.z = 0
        self.velocity_pub.publish(self.twist)
        rospy.sleep(3)

    def teleport(self):
        model_state = ModelState()
        model_state.model_name = "R1"
        model_state.pose.position.x = -4.05
        model_state.pose.position.y = -2.27
        model_state.pose.position.z = 0.0525
        model_state.pose.orientation.x = 0.0
        model_state.pose.orientation.y = 0.0
        model_state.pose.orientation.z = 0.0
        model_state.pose.orientation.w = 0.0
        self.teleporter(model_state)
        rospy.sleep(0.25)

    def get_error(self, road_mid_point):
        x_error = road_mid_point[0] - 320
        y_error = 280 - road_mid_point[1]
        return [x_error, y_error]
    
    def move(self, error):
        self.twist.angular.z = self.kp_x * error[0]
        self.twist.linear.x = 0.15 - self.kp_y * abs(error[0])
        self.velocity_pub.publish(self.twist)
    
    def drive(self, camera_image): 
        mid_point = self.find_road_mountain(camera_image)
        error = self.get_error(mid_point)
        self.move(error)

    def find_road_mountain(self, image, show=True):
        min_area = 300
        
        img = cv2.resize(image, (640, 360))


        # Tools.trackpad(img)

        # convert to grayscale
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # select the colour white
        # threshold the image
        mask = cv2.inRange(img, ImageConstants.MOUNTAIN_LOWER_BOUND, ImageConstants.MOUNTAIN_UPPER_BOUND)
        kernel = np.ones((3,3),np.uint8)
        mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # find the contours
        contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # select the countours that have contour area greater than min_area
        contours = [c for c in contours if cv2.contourArea(c) > min_area]

        # contours = [c for c in contours if (max(cv2.minAreaRect(c)[1][0], cv2.minAreaRect(c)[1][1])) > 225]

        # remove contours with long axis shorter than 20 pixels
        # find the eccentricity of the contours
        # contours = [c for c in contours if (cv2.minAreaRect(c)[1][0]/cv2.minAreaRect(c)[1][1]) <= 0.8]
        # find the center of the minimum area rectangle
        
        # contours = [c for c in contours if (abs(cv2.minAreaRect(c)[0][0]-320) >= 80 or cv2.minAreaRect(c)[0][1] <= 280)]
        
        # sort the contours based on the greatest of the height or width of the minarea rectangle
        contours = sorted(contours, key = lambda c: max(cv2.minAreaRect(c)[1][0], cv2.minAreaRect(c)[1][1]), reverse = True)[:2]
        # contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
       
        # find the center of the two biggest contours
        # line_c = []
        if len(contours) == 0:
            mid_point = [640, 180]
        elif len(contours) < 2:
            M = cv2.moments(contours[0])
            mid_point = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
            # line_c.append(contours[0])
        else:
            center = []
            for c in contours[0:2]:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center.append([cX, cY])
                # line_c.append(c)

            # find the middle point of the two centers based on the weighted average of the area of the contours
            mid_point_x = (center[0][0] * cv2.contourArea(contours[0]) + center[1][0] * cv2.contourArea(contours[1])) / (cv2.contourArea(contours[0]) + cv2.contourArea(contours[1]))
            mid_point_y = (center[0][1] * cv2.contourArea(contours[0]) + center[1][1] * cv2.contourArea(contours[1])) / (cv2.contourArea(contours[0]) + cv2.contourArea(contours[1]))
            # mid_point_x = (center[0][0] + center[1][0]) / 2
            # mid_point_y = (center[0][1] + center[1][1]) / 2
            mid_point = [mid_point_x, mid_point_y]

            # plort the center of the two biggest contours and the middle point with cv2
            if show:
                cv2.circle(img, (int(center[0][0]), int(center[0][1])), 10, (0, 0, 255), -1)
                cv2.circle(img, (int(center[1][0]), int(center[1][1])), 10, (0, 0, 255), -1)
    
        if show:
            # cv2.circle(img, (int(mid_point[0]), int(mid_point[1])), 1, (255, 0, 0), -1)
            cv2.rectangle(img, (320-100, 280), (320+100,360), (0, 0, 255), 3)
            # cv2.drawContours(img, big_c, -1, (0, 255, 0), 3)
            # cv2.drawContours(img, small_c, -1, (0, 0, 255), 3)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            # cv2.drawContours(img, line_c, -1, (0, 255, 0), 3)
            cv2.circle(img, (int(mid_point[0]), int(mid_point[1])), 10, (255, 0, 0), -1)
            cv2.imshow("Image window", img)
            cv2.waitKey(1)

            # Apply the mask to extract the color
            result = cv2.bitwise_and(img, img, mask=mask1)            
            # show image 
            cv2.imshow('Color Extraction', result)

        return mid_point
"""    
    kp_x = 0.0075
    kp_y = 0.000225
    
    stop_twist = Twist()
    stop_twist.linear.x = 0
    stop_twist.angular.z = 0

    def __init__(self) -> None:
        self.velocity_pub = rospy.Publisher('R1/cmd_vel', Twist, queue_size=1)
        self.teleporter = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.twist = Twist()

    def get_error(self, road_mid_point):
        x_error = road_mid_point[0] - 320
        y_error = 280 - road_mid_point[1]
        return [x_error, y_error]
    
    def move(self, error):
        # # WORKING DO NOT CHANGE HERE
        self.twist.angular.z = self.kp_x * error[0]
        # WORKING DO NOT CHANGE HERE
        self.twist.linear.x = 0.125 - self.kp_y * abs(error[0]) 
        self.velocity_pub.publish(self.twist)

    def stop(self):
        self.velocity_pub.publish(self.stop_twist)
    
    def slow_down(self):
        self.twist.linear.x = 0.035
        self.twist.angular.z = 0
        self.velocity_pub.publish(self.twist)
    
    def speed_up(self): # WORKING DO NOT CHANGE
        self.twist.linear.x = 0.95
        self.twist.angular.z = 0
        self.velocity_pub.publish(self.twist)
        rospy.sleep(3)

    def teleport(self):
        model_state = ModelState()
        model_state.model_name = "R1"
        model_state.pose.position.x = -4.05
        model_state.pose.position.y = -2.27
        model_state.pose.position.z = 0.0525
        model_state.pose.orientation.x = 0.0
        model_state.pose.orientation.y = 0.0
        model_state.pose.orientation.z = 0.0
        model_state.pose.orientation.w = 0.0
        self.teleporter(model_state)
        rospy.sleep(0.25)

    def drive(self, camera_image):
        error = self.get_error(self.find_road_mountain(camera_image))
        self.move(error)

    def find_road_mountain(self, image, show=True):
        img = cv2.resize(image, (640, 360))
        min_area = 300
        
        mask = cv2.inRange(img, ImageConstants.MOUNTAIN_LOWER_BOUND, ImageConstants.MOUNTAIN_UPPER_BOUND)
        kernel = np.ones((3,3),np.uint8)
        mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # find the contours
        contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # select the countours that have contour area greater than min_area
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        contours = [c for c in contours if (abs(cv2.minAreaRect(c)[0][0]-320) >= 80 or cv2.minAreaRect(c)[0][1] <= 280)]
        
        # sort the contours based on the greatest of the height or width of the minarea rectangle
        contours = sorted(contours, key = lambda c: max(cv2.minAreaRect(c)[1][0], cv2.minAreaRect(c)[1][1]), reverse = True)[:2]

        # find the center of the two biggest contours
        if len(contours) == 0:
            mid_point = [640, 180]
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

            # draw the contours on the image
            cv2.rectangle(img, (320-100, 225), (320+100,360), (0, 0, 255), 3)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            cv2.imshow("Image window", img)
            cv2.waitKey(1)

        return mid_point
        """