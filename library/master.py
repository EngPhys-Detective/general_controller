#!/usr/bin/env python3

# change current directory to the parent directory of this file 
import os
# os.chdir('/home/fizzer/enph_ws/src/competition_machine/')

# from library.image_processor import ImageProcessor
# from library.constants import ImageConstants, ClueConstants, CNNConstants, ClueLocations
# from library.clue_finder import ClueFinder
# from library.clue_guesser import ClueGuesser
# from library.pid_driver import PavedDriver, DirtDriver
# from library.score_keeper import ScoreKeeper

from image_processor import ImageProcessor
from constants import ImageConstants, ClueConstants, CNNConstants, ClueLocations
from clue_finder import ClueFinder
from clue_guesser import ClueGuesser
from pid_driver import PavedDriver, DirtDriver, MountainDriver, TopDriver
from score_keeper import ScoreKeeper


import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import cv2
import sys
 

class Master:
    counter = 0
    def __init__(self):
        self.bridge = CvBridge()

        rospy.init_node('master')
        self.paved_driver = PavedDriver()
        self.dirt_driver = DirtDriver()
        self.mountain_driver = MountainDriver()
        self.top_driver = TopDriver()
        self.image_processor = ImageProcessor()
        self.clue_guesser = ClueGuesser()
        self.clue_finder = ClueFinder()

        self.driver = self.paved_driver
        self.onDirt = False
        self.pink_count = 0
        self.red_count = 0
        self.clue_7 = 0
        self.passed_truck = False
        self.clue_8 = 0

        self.score_keeper = ScoreKeeper()

        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.poll)

    def poll(self, data):
        try:
            camera_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        if (not self.passed_truck and ImageProcessor.detect_truck(camera_image)):
            print("----TRUCK DETECTED----")
            self.driver.stop()
            rospy.sleep(2)
            
        # cv2.imshow("trcuk filter", ImageProcessor.colour_mask(camera_image, ImageConstants.TRUCK_LOWER_BOUND, ImageConstants.TRUCK_UPPER_BOUND))
        # cv2.waitKey(1)
        elif (ImageProcessor.detect_pink_line(camera_image)):
            print("----RED LINE DETECTED----")
            self.passed_truck = True
            self.onDirt = True
            self.driver = self.dirt_driver
            if self.pink_count == 0:
                self.driver.speed_up()
                self.pink_count += 1
            elif self.score_keeper.publish_count > 4:
                self.driver = self.mountain_driver
                self.driver.stop()
                self.driver.teleport()
                self.clue_7 = 1
                self.driver.slow_down()
            
            self.pink_count += 1
        
        elif (ImageProcessor.detect_red_line(camera_image)):
            print("----RED LINE DETECTED----")
            if self.red_count == 0:
                self.driver.speed_up()
            self.red_count += 1

        elif self.score_keeper.publish_count == 7 and self.clue_7 == 1:
            self.clue_7 += 1
            self.driver.speed_up()
            print(self.clue_7)

        elif np.sum(ImageProcessor.blue_filter(camera_image)==255) > ImageConstants.TOP_CLUE_THRESHOLD and self.clue_7 > 1:
            self.clue_8 += 1
            if self.clue_8 > 15:
                self.driver = self.top_driver

        elif self.score_keeper.publish_count == 8:
            print("----END OF COURSE----")
            self.driver.stop()
            self.score_keeper.end()
            sys.exit(0)
        
        
        self.seeClue = self.clue_check(camera_image)
        if (self.clue_7 != 1):
            self.driver.drive(camera_image)

    def clue_check(self, camera_image):
        banner_image = ClueFinder.get_banner_image(camera_image)
        if banner_image is not None:
            cv2.imshow("Clue", banner_image)
            cv2.waitKey(1)
            self.driver.slow_down()
            clue_value, clue_topic = self.clue_guesser.guess_clue_values(banner_image)
            if clue_value is not None:
                self.score_keeper.publish_clue(clue_value, clue_topic)
            return True
        else:
            return False

if __name__ == '__main__': 
    master = Master()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass