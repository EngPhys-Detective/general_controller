#!/usr/bin/env python3


from __future__ import print_function
 
import sys
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import random
import string
import math
import os
import tensorflow as tf
from tensorflow import keras as ks
from matplotlib import pyplot as plt
from constants import *

from random import randint
from PIL import Image, ImageFont, ImageDraw


class ClueGuesser:
    """
    Class for guessing clues in an image.
    """

    def __init__(self):
        """
        Initialize the ClueGuesser object.

        Parameters:
        - camera_feed_image: The input camera feed image.
        """
        
        
        try:
            self.conv_model = ks.models.load_model(CNNConstants.CNN_MODEL_PATH)
            print("Model Loaded Successfully")
        except Exception as e:
            print("Error loading model")
            print(str(e))
            
        
    def crop_clue_value(self, clue_img):
        """!
        Crop a clue image into individual character of the clue value.

        Args:
            clue_img (numpy.ndarray): A NumPy array representing the clue image.

        Returns:
            list of numpy.ndarray: A list of NumPy arrays, each representing a cropped character region.
        """

        NUM_CROPS = 12 # This is the maximum number of characters for the clue value

        clue_img_pil = Image.fromarray(clue_img)

        width, height = clue_img_pil.size


        top = height - 150 # 400 - 150 = 250
        bottom = height - 50 # 400 - 50 = 350
        cropped_image_width = ClueConstants.CLUE_VALUE_CROP_WIDTH

        clue_img_pil_cropped = []

        for i in range(NUM_CROPS):
            clue_img_pil_cropped.append(clue_img_pil.crop((30+i*cropped_image_width, top, 75+i*cropped_image_width, bottom)))

        clue_img_cv2_cropped = []
        for i in range(NUM_CROPS):
            clue_img_cv2_cropped.append(np.array(clue_img_pil_cropped[i])) # here we are converting to cv2 image

        return clue_img_cv2_cropped
    
    def get_symbol_from_one_hot_encoder(self,vector):
        assert len(vector) == CNNConstants.CHARACTERS_COUNT

        max_value_index = np.argmax(vector)

        return CNNConstants.CHARACTERS[max_value_index]


    def guess_image(self,input_banner_img):
        
        cropped_input_clue_values = self.crop_clue_value(input_banner_img)
        clue_value_guess = ""

        for cicv in cropped_input_clue_values:
            img_aug = np.expand_dims(cicv, axis=0)
            y_predict = self.conv_model.predict(img_aug)[0]
            clue_value_guess += self.get_symbol_from_one_hot_encoder(y_predict)

        return clue_value_guess
    
def main(args):
    
    clue_guesser = ClueGuesser()
    
    
    for test_img in sorted(os.listdir("/home/fizzer/enph353_ws/src/my_controller/media/testing_clue_banners")):
        nice_image = cv2.imread("/home/fizzer/enph353_ws/src/my_controller/media/testing_clue_banners/" + test_img)
        cv2.imshow("nice_image", nice_image)
        cv2.waitKey(1)
        print(clue_guesser.guess_image(nice_image))
        

if __name__ == '__main__':
    main(sys.argv)