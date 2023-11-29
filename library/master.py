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
from pid_driver import PavedDriver, DirtDriver
from score_keeper import ScoreKeeper


import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
 

class Master:

    def __init__(self):
        self.bridge = CvBridge()

        rospy.init_node('master')
        self.paved_driver = PavedDriver()
        self.dirt_driver = DirtDriver()
        self.image_processor = ImageProcessor()
        self.clue_guesser = ClueGuesser()

        self.driver = self.paved_driver

        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.poll)

        self.onDirt = False
        self.pink_count = 0
        
        self.score_keeper = ScoreKeeper()

    def poll(self, data):
        try:
            camera_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        
        if (ImageProcessor.detect_pink_line(self, camera_image)):
            self.onDirt = True
            # if (self.pink_count > 0):
            #     self.dirt_driver.teleport()
            self.pink_count += 1
            print("pink line detected")

        if (self.onDirt):
            self.driver = self.dirt_driver

        self.driver.drive(camera_image)
        
        clue_finder = ClueFinder(camera_image)
        
        banner_image = clue_finder.get_banner_image()
        if (banner_image is not None): 
            self.driver.stop()
            clue = self.clue_guesser.guess_image(banner_image)
            print(clue)
            self.score_keeper.publish_clue(clue)
            
            # cv2.imwrite("/home/fizzer/enph353_ws/src/my_controller/media/testing_clue_banners/test_img" + str(self.counter) + ".png", banner_image)
            # self.counter += 1


if __name__ == '__main__': 
    master = Master()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass