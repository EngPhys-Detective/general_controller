#!/usr/bin/env python3


from __future__ import print_function
 
from image_processor import ImageProcessor
from constants import ImageConstants, ClueConstants, CNNConstants, ClueLocations
from clue_finder import ClueFinder
from clue_guesser import ClueGuesser
from pid_driver import *
from score_keeper import ScoreKeeper
from tools import Tools
from flags import Flags
from constants import *

import sys
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import cv2


class StateMachine:
        
    def __init__(self, initial_state):
        self.bridge = CvBridge()

        rospy.init_node('master')
        
        self.loop_count = 0
        
        self.current_state = initial_state
        
        self.paved_driver = PavedDriver()
        self.dirt_driver = DirtDriver()
        self.mountain_driver = MountainDriver()
        self.peak_driver = TopDriver()
        self.image_processor = ImageProcessor()
        self.clue_guesser = ClueGuesser()
        self.clue_finder = ClueFinder()

        self.driver = self.paved_driver   
        
        self.banner_image = None
        
        self.last_red_detection_time = 0
        self.last_pink_detection_time = 0
        self.time_of_pedestrian_detection = 0
        self.passed_tunnel = False
        self.saw_last_clue = 0
        
        self.score_keeper = ScoreKeeper()
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.poll)
        
    
    def poll(self,data):
        try:
            camera_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            
            
        self.banner_image = ClueFinder.get_banner_image(camera_image)
        
                
        if (self.get_current_state() == States.NORMAL_DRV):
            self.normal_driving(camera_image)
            
        elif (self.get_current_state() == States.RED_LINE_DRV):
            self.red_line_driving(camera_image)
        
        elif (self.get_current_state() == States.PEDESTRIAN_SAFETY):
            self.wait_for_pedestrian(camera_image)
             
        elif (self.get_current_state() == States.DIRT_DRV):
            self.dirt_driving(camera_image)
            
        elif (self.get_current_state() == States.MOUNTAIN_DRV):
            self.mountain_driving(camera_image)

        elif (self.get_current_state() == States.TRUCK_SAFETY):
            self.wait_for_truck(camera_image)

        elif (self.get_current_state() == States.PEAK_DRV):
            self.peak_driving(camera_image)

        elif (self.get_current_state() == States.FINISH_COURSE):
            self.finish_course()           
            
    def change_state(self, new_state):
        if new_state == self.current_state:
            return False
        
        self.current_state = new_state
        return True
    
    def clue_check(self):
        if (self.banner_image is not None):
            self.driver.slow_down()
            clue_value, clue_topic= self.clue_guesser.guess_clue_values(self.banner_image)
            if clue_value is not None:
                self.score_keeper.publish_clue(clue_value, clue_topic)
        
    def get_current_state(self):
        return self.current_state
    
    def normal_driving(self, camera_image):
        self.driver.drive(camera_image)
        
        if (self.score_keeper.publish_count != 4):
            self.clue_check() 
        
        if (ImageProcessor.detect_red_line(camera_image) and not Flags.pedestrian_crossed):
            self.last_red_detection_time = rospy.get_time()
            self.change_state(States.PEDESTRIAN_SAFETY)
            
        if (ImageProcessor.detect_pink_line(camera_image)):
            # self.driver.turn_right_slightly()            
            self.last_pink_detection_time = rospy.get_time()
            self.driver.speed_up_before_pink()
            self.change_state(States.DIRT_DRV)
            
        if (ImageProcessor.detect_truck(camera_image)):
            self.change_state(States.TRUCK_SAFETY)
        
    def wait_for_truck(self, camera_image):
        self.driver.stop()
        rospy.sleep(0.5)
        
        if (not ImageProcessor.detect_truck(camera_image)):
            self.change_state(States.NORMAL_DRV)
            
    def wait_for_pedestrian(self, camera_image):
        print("entered wait for pedestrian")
        self.driver.stop()
        elapsed_time = rospy.get_time() - self.last_red_detection_time
        print("elapsed_time", elapsed_time)
        if (elapsed_time > 1.5):
            print("status: ", ImageProcessor.detect_pedestrian(camera_image))
            print("passed_pedestrian: ", Flags.pedestrian_crossed)
            if (Flags.pedestrian_crossed):
                print("elapsed time: ", elapsed_time)
                print("entered inner if")
                self.time_of_pedestrian_detection = rospy.get_time()
                rospy.sleep(0.3)
                self.change_state(States.RED_LINE_DRV)
                        
    def red_line_driving(self, camera_image):
        print("red line detected and pedestrian safe")
        self.driver.speed_up()
        
        elapsed_time = rospy.get_time() - self.time_of_pedestrian_detection
        print(elapsed_time)
        if (elapsed_time > 0.5):
            self.change_state(States.NORMAL_DRV)
    
    def dirt_driving(self, camera_image):
        self.driver = self.dirt_driver
        self.driver.drive(camera_image)

        if (self.score_keeper.publish_count != 6):
            self.clue_check()
        
        elapsed_time = rospy.get_time() - self.last_pink_detection_time
        
        if (ImageProcessor.detect_pink_line(camera_image) and elapsed_time > 15):
            self.driver.stop()
            
            self.last_pink_detection_time = rospy.get_time()
            self.driver.teleport_to_mountain()
            self.change_state(States.MOUNTAIN_DRV)
        
    def mountain_driving(self, camera_image):
        self.driver = self.mountain_driver
        self.clue_check()
        
        if (self.score_keeper.publish_count == 7 and not self.passed_tunnel):
            self.driver.speed_up()
            self.passed_tunnel = True
        elif (np.sum(ImageProcessor.blue_filter(camera_image)==255) > ImageConstants.TOP_CLUE_THRESHOLD):
            self.saw_last_clue += 1
            if (self.saw_last_clue > 25):
                self.driver = self.paved_driver
                self.change_state(States.PEAK_DRV)

        self.driver.drive(camera_image)

    def peak_driving(self, camera_image):
        self.driver = self.peak_driver
        self.driver.drive(camera_image)
        self.clue_check()

        if (self.score_keeper.publish_count == 8):
            self.change_state(States.FINISH_COURSE)

    def finish_course(self):
        print("----END OF COURSE----")
        self.driver.stop()
        self.score_keeper.end()
        sys.exit(0)
        


if __name__ == '__main__': 
    state_machine = StateMachine(States.NORMAL_DRV)


    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass