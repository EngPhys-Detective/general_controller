#!/usr/bin/env python3


from __future__ import print_function
 
import sys
import cv2
import numpy as np
from image_processor import ImageProcessor
from constants import *



class Tools:
    """Class that contains useful functions for the controller."""

    def trackpad(self, img):
        cv2.namedWindow('Color Extraction')
        
        # Create trackbars for HSV values
        cv2.createTrackbar('Hue Lower', 'Color Extraction', 0, 180, nothing)
        cv2.createTrackbar('Saturation Lower', 'Color Extraction', 0, 255, nothing)
        cv2.createTrackbar('Value Lower', 'Color Extraction', 0, 255, nothing)
        cv2.createTrackbar('Hue Upper', 'Color Extraction', 0, 180, nothing)
        cv2.createTrackbar('Saturation Upper', 'Color Extraction', 0, 255, nothing)
        cv2.createTrackbar('Value Upper', 'Color Extraction', 0, 255, nothing)
        
        while(True): 
        
            # for button pressing and changing 
            k = cv2.waitKey(1) & 0xFF
            if k == 27: 
                break
            
            hue_lower = cv2.getTrackbarPos('Hue Lower', 'Color Extraction')
            sat_lower = cv2.getTrackbarPos('Saturation Lower', 'Color Extraction')
            val_lower = cv2.getTrackbarPos('Value Lower', 'Color Extraction')
            hue_upper = cv2.getTrackbarPos('Hue Upper', 'Color Extraction')
            sat_upper = cv2.getTrackbarPos('Saturation Upper', 'Color Extraction')
            val_upper = cv2.getTrackbarPos('Value Upper', 'Color Extraction')
            
            # Create HSV lower and upper bounds
            hsv_lower = np.array([hue_lower, sat_lower, val_lower], dtype=np.uint8)
            hsv_upper = np.array([hue_upper, sat_upper, val_upper], dtype=np.uint8)

            # denoised_img = cv2.fastNlMeansDenoising(img, None, 10, 10, 7)
            # cv2.imshow("denoised", denoised_img)
            # cv2.waitKey(1)
            
            # Convert the image to HSV
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Create a binary mask for the specified color range
            color_mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
            
            # Apply the mask to extract the color
            result = cv2.bitwise_and(img, img, mask=color_mask)
            
            # show image 
            cv2.imshow('Color Extraction', np.vstack([cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR), result]))
            # cv2.imshow('Color Extraction', result)
            cv2.waitKey(1)
            
            
def nothing(x):
    pass

def main(args):
    
    
    print("Hello World!")
    
    
    
    
if __name__ == '__main__':
    main(sys.argv)
