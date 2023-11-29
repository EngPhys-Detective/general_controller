#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import time

class ScoreKeeper:
    
    def __init__(self) -> None:
        self.start_msg = "TeamName,password,0,NA"
        self.end_msg = "TeamName,password,-1,NA"

        self.score_publisher = rospy.Publisher('score_tracker', String , queue_size=1)
        rospy.sleep(2)  # wait for publisher to initialize
        self.start()
        print("ScoreKeeper initialized")
        self.publish_count = 0

    def start(self):
        try:
            print(self.start_msg)
            self.score_publisher.publish(self.start_msg)
        except Exception as e:
            print(e)
            return False

    def end(self):
        try:
            self.score_publisher.publish(self.end_msg)
            time.sleep(0.5)
            return True
        except Exception as e:
            print(e)
            return False
        
    def publish_clue(self, clue_msg):
        try:
            self.score_publisher.publish("TeamName,password,1," + clue_msg)
            self.publish_count += 1
            return True
        except Exception as e:
            print(e)
            return False