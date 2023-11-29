#!/usr/bin/env python3


from __future__ import print_function
 
import cv2
import numpy as np

from constants import *


class ImageProcessor:
          
    def find_road_paved(self, image, show=True):
        lower_bound = ImageConstants.PAVED_ROAD_LOWER_BOUND
        upper_bound = ImageConstants.PAVED_ROAD_UPPER_BOUND
        min_area = 200 * 2.5

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
            cv2.circle(img, (int(mid_point[0]), int(mid_point[1])), 10, (255, 0, 0), -1)
            cv2.imshow("Image window", img)
            cv2.waitKey(1)

        return mid_point

    def find_road_dirt(self, image, show=True):
        img = cv2.resize(image, (640, 360))
        neglecting_rect = np.array([[320-30, 340-20], [320-30, 340+20], [320+30, 340+20], [320+30, 340-20]])

        # convert to grayscale
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # select the colour white
        # threshold the image
        mask = cv2.inRange(hsv, ImageConstants.DIRT_ROAD_LOWER_BOUND, ImageConstants.DIRT_ROAD_UPPER_BOUND)
        # find the contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # select the countours that have contour area greater than 500
        contours = [c for c in contours if cv2.contourArea(c) > 50]
        # remove contours with long axis shorter than 20 pixels
        # find the eccentricity of the contours
        # contours = [c for c in contours if (cv2.minAreaRect(c)[1][0]/cv2.minAreaRect(c)[1][1]) <= 0.8]
        # find the center of the minimum area rectangle
        
        contours = [c for c in contours
                     if (abs(cv2.minAreaRect(c)[0][0]-320) >= 30 and abs(cv2.minAreaRect(c)[0][1]-340) >= 30)]
        # sort the contours by area
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2]

        # find the center of the two biggest contours
        if len(contours) == 0:
            mid_point = [320, 180]
        if len(contours) < 2:
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
            cv2.drawRect(img, neglecting_rect, (0, 255, 0), 3)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            cv2.imshow("Image window", img)
            cv2.waitKey(1)

        return mid_point

    def is_blurry(self, image, fm_threshold):
        """!
        Determines if an image is blurry based on the variance of the Laplacian filter.

        Parameters:
        image (numpy.ndarray): The input image.
        fm_threshold (float): The threshold value for the variance of the Laplacian filter.

        Returns:
        bool: True if the image is blurry, False otherwise.
        """
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()

        if fm < fm_threshold:
            return True
        else:
            return False
    
    
    def colour_mask(self, image, lower_bound, upper_bound):
        """!
        Applies a color filter to the input image.

        Parameters:
        image (numpy.ndarray): The input image.
        lower_bound (numpy.ndarray): The lower bound of the color filter.
        upper_bound (numpy.ndarray): The upper bound of the color filter.

        Returns:
        numpy.ndarray: The binary mask representing the color regions in the image.
        """
        
        
        HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
        color_change_mask = cv2.inRange(HSV_image, lower_bound, upper_bound)
        return color_change_mask
    
        
    def blue_filter(self, image):
        
        """
        Applies a blue color filter to the input image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The binary mask representing the blue color regions in the image.
        """
                    
        lower_bound = np.array([ImageConstants.BLUE_HUE_LOWER_BOUND, ImageConstants.BLUE_SAT_LOWER_BOUND, ImageConstants.BLUE_VAL_LOWER_BOUND])
        upper_bound = np.array([ImageConstants.BLUE_HUE_UPPER_BOUND, ImageConstants.BLUE_SAT_UPPER_BOUND, ImageConstants.BLUE_VAL_UPPER_BOUND])
        
        return ImageProcessor.colour_mask(self, image, lower_bound, upper_bound) 
    
    def red_filter(self, image):
        """
        Applies a red color filter to the input image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The binary mask representing the red color regions in the image.
        """
                    
        lower_bound = np.array([ImageConstants.RED_HUE_LOWER_BOUND, ImageConstants.RED_SAT_LOWER_BOUND, ImageConstants.RED_VAL_LOWER_BOUND])
        upper_bound = np.array([ImageConstants.RED_HUE_UPPER_BOUND, ImageConstants.RED_SAT_UPPER_BOUND, ImageConstants.RED_VAL_UPPER_BOUND])
        
        return ImageProcessor.colour_mask(self, image, lower_bound, upper_bound)
    
    def pink_filter(self, image):
        """
        Applies a pink color filter to the input image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The binary mask representing the pink color regions in the image.
        """
                    
        lower_bound = np.array([ImageConstants.PINK_HUE_LOWER_BOUND, ImageConstants.PINK_SAT_LOWER_BOUND, ImageConstants.PINK_VAL_LOWER_BOUND])
        upper_bound = np.array([ImageConstants.PINK_HUE_UPPER_BOUND, ImageConstants.PINK_SAT_UPPER_BOUND, ImageConstants.PINK_VAL_UPPER_BOUND])
        
        return ImageProcessor.colour_mask(self, image, lower_bound, upper_bound)
    
    def do_perspective_transform(self, image, original_coordinates, final_width, final_height):
        """
        Applies perspective transformation to an image based on the given original coordinates and final width and height.

        Parameters:
            image (numpy.ndarray): The input image.
            original_coordinates (numpy.ndarray): The original coordinates of the image corners.
            final_width (int): The desired width of the transformed image.
            final_height (int): The desired height of the transformed image.

        Returns:
            numpy.ndarray: The transformed image.

        """
        ref_h = final_height
        ref_w = final_width
        
        transform_to = np.float32([[0, 0], [0, ref_h], [ref_w, ref_h], [ref_w, 0]]).reshape(-1, 1, 2) # 4 corners of the reference image
        transform_from = original_coordinates
        
        transformation_matrix = cv2.getPerspectiveTransform(transform_from, transform_to) # matrix transform from the slanted clue banner to the well-shaped 2D image
        imgout = cv2.warpPerspective(image, transformation_matrix, (ref_w, ref_h))

        return imgout
    
    def detect_horizontal_line(self, image):
        """
        Detects the horizontal line in the input image.

        Parameters:
            image (numpy.ndarray): The input image must be binary (thresholded).

        Returns:
            numpy.ndarray: The image with the detected horizontal line.
        """
        is_horizontal_line = False
                
        image_shape = image.shape
        w, h = image_shape[1], image_shape[0]
        
        if (image[h-1][int(w/2)] == 255):
            is_horizontal_line = True
                
        return is_horizontal_line
    
    def detect_red_line(self, image):
        """
        Detects the red line in the input image.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The image with the detected red line.
        """
        
        red_mask = ImageProcessor.red_filter(self, image)
        
        return ImageProcessor.detect_horizontal_line(self, red_mask)
    
    def detect_pink_line(self, image):
        """
        Detects the pink line in the input image.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The image with the detected pink line.
        """
        
        pink_mask = ImageProcessor.pink_filter(self, image)
        
        return ImageProcessor.detect_horizontal_line(self, pink_mask)
        
        
        
