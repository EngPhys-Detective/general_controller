#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import time

class ScoreKeeper:

    START_MSG = "Sharper,password,0,NA"
    END_MSG = "Sharper,password,-1,NA"
    
    def __init__(self) -> None:
        self.score_publisher = rospy.Publisher('score_tracker', String , queue_size=1)
        self.start()
        self.publish_count = 0

    def start(self):
        try:
            self.score_publisher.publish(self.START_MSG)
            time.sleep(0.5)
        except Exception as e:
            print(e)
            return False

    def end(self):
        try:
            self.score_publisher.publish(self.END_MSG)
            time.sleep(0.5)
            return True
        except Exception as e:
            print(e)
            return False
        
    def publish_clue(self, clue_msg):
        try:
            self.score_publisher.publish(clue_msg)
            self.publish_count += 1
            return True
        except Exception as e:
            print(e)
            return False