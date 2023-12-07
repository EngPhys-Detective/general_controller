#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import time

from constants import ClueConstants

class ScoreKeeper:
    
    def __init__(self) -> None:
        self.start_msg = "Sharpener,Invincible,0,NA"
        self.end_msg = "Sharpener,Invincible,-1,NA"
        
        try:
            self.score_publisher = rospy.Publisher('score_tracker', String , queue_size=1)
            rospy.sleep(2)  # wait for publisher to initialize
            print("-----ScoreKeeper SUCCESS-----")
        except Exception as e:
            print(e)
            print("----ScoreKeeper FAILED----")

        self.publish_count = 0
        self.start()

    def start(self):
        try:
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
        
    def publish_clue(self, value, topic):
        try:
            if topic in ClueConstants.CLUE_TOPICS:
                topic_msg = str(ClueConstants.CLUE_TOPICS.index(topic)+1)
                if value is not None:
                    self.score_publisher.publish("Sharpener,Invincible," + topic_msg + "," + value)
                    self.publish_count = int(topic_msg)
                    return True
                return False
            else:
                print("Invalid topic")
                return False
        except Exception as e:
            print(e)
            return False