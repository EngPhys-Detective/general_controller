#!/usr/bin/env python3


from __future__ import print_function
 

import cv2
import numpy as np
from constants import *
from image_processor import ImageProcessor

class ClueFinder:
    """
    Class for finding clues in an image.
    """


    def get_bounding_points(approx_polygon):
        """
        Get the bounding points of an approximate polygon.

        Parameters:
        - approx_polygon: The approximate polygon.

        Returns:
        - The bounding points as a numpy array.
        """
        points = []
        for i in range(4):
            points.append((approx_polygon[i][0][0], approx_polygon[i][0][1]))

        # Calculate centroid
        centroid_x = sum(x for x, y in points) / len(points)
        centroid_y = sum(y for x, y in points) / len(points)

        # Classify each point relative to the centroid
        classified_points = []
        for x, y in points:
            if x < centroid_x and y < centroid_y:
                classified_points.append(('bottom', 'left', (x, y)))
            elif x < centroid_x and y > centroid_y:
                classified_points.append(('top', 'left', (x, y)))
            elif x > centroid_x and y < centroid_y:
                classified_points.append(('bottom', 'right', (x, y)))
            elif x > centroid_x and y > centroid_y:
                classified_points.append(('top', 'right', (x, y)))

        # Sort points based on their classifications
        sorted_points = sorted(classified_points, key=lambda x: (x[0], x[1]))

        top_left = sorted_points[0][2]
        top_right = sorted_points[1][2]
        bottom_left = sorted_points[2][2]
        bottom_right = sorted_points[3][2]

        polygon_out = [[top_left[0], top_left[1]]], [[bottom_left[0], bottom_left[1]]], [[bottom_right[0], bottom_right[1]]], [[top_right[0], top_right[1]]]
        return np.float32(polygon_out).reshape(-1, 1, 2)

    def remove_blue_boundaries(image):
        """
        Remove blue boundaries from an image.

        Parameters:
        - image: The input image.

        Returns:
        - The image with blue boundaries removed.
        """
        color_change_mask = ImageProcessor.blue_filter(image)

        edged = cv2.Canny(color_change_mask, 30, 200)

        approx_polygon = ClueFinder.find_approximate_polygon(edged)

        if approx_polygon is None:
            return None

        approx_polygon = ClueFinder.get_bounding_points(approx_polygon)

        imgout = ImageProcessor.do_perspective_transform(image, approx_polygon, ClueConstants.CLUE_BANNER_WIDTH, ClueConstants.CLUE_BANNER_HEIGHT)
        return imgout

    def find_banner(image):
        """
        Find a banner in an image.

        Parameters:
        - image: The input image.

        Returns:
        - The cropped banner image and the number of white pixels in the image.
        """
        color_change_mask = ImageProcessor.blue_filter(image)
        num_white_pixels = cv2.countNonZero(color_change_mask)

        approx_polygon = ClueFinder.find_approximate_polygon(color_change_mask)

        if approx_polygon is None:
            return None, -1

        approx_polygon = ClueFinder.get_bounding_points(approx_polygon)

        imgout = ImageProcessor.do_perspective_transform(image, approx_polygon, ClueConstants.CLUE_BANNER_WIDTH, ClueConstants.CLUE_BANNER_HEIGHT)

        return imgout, num_white_pixels

    def find_approximate_polygon(image):
        """
        Find an approximate polygon in an image.

        Parameters:
        - image: The input image should be color_mask image.

        Returns:
        - The approximate polygon.
        """
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            # print("No contours found")
            return None

        largest_contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx_polygon) != 4:
            # print("Approx polygon is not a quadrilateral")
            return None

        return approx_polygon

    def get_banner_image(camera_feed_image):
        """
        Get the banner image.

        Returns:
        - The banner image if found and not blurry, None otherwise.
        """
        cropped_banner, white_pix = ClueFinder.find_banner(camera_feed_image)

        if cropped_banner is not None:
            if ImageProcessor.is_blurry(cropped_banner, ClueConstants.CLUE_BLURRINESS_THRESHOLD):
                # print("blurry")
                pass
            else:
                # print("not blurry")
                if white_pix < ClueConstants.CLUE_MAX_WHITE_PIXELS:
                    ready_for_CNN = ClueFinder.remove_blue_boundaries(cropped_banner)
                    if ready_for_CNN is not None:
                        return ready_for_CNN

        else:
            # print("No banner found")
            return None
    

    
    