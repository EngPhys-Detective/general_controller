#!/usr/bin/env python3

import numpy as np


class ImageConstants:
    """A class that defines constants related to image processing."""

    PAVED_ROAD_LOWER_BOUND = np.array([0, 0, 220], np.uint8)
    """numpy.ndarray: The lower bound for the paved road color filter."""
    PAVED_ROAD_UPPER_BOUND = np.array([5, 10, 255], np.uint8)
    """numpy.ndarray: The upper bound for the paved road color filter."""

    DIRT_ROAD_LOWER_BOUND = np.array([0, 10, 180], np.uint8)
    """numpy.ndarray: The lower bound for the dirt road color filter."""
    DIRT_ROAD_UPPER_BOUND = np.array([45, 100, 255], np.uint8)

    TRUCK_LOWER_BOUND = np.array([0, 0, 0], np.uint8)
    """numpy.ndarray: The lower bound for the truck color filter."""
    
    TRUCK_UPPER_BOUND = np.array([180, 255, 5], np.uint8)
    """numpy.ndarray: The upper bound for the truck color filter."""
    
    TRUCK_THRESHOLD = 80
    """int: The threshold for determining the number of white pixels in binary mask of the truck image."""

    MOUNTAIN_LOWER_BOUND = np.array([120, 160, 170])
    MOUNTAIN_UPPER_BOUND = np.array([180, 230, 240])

    BLUE_HUE_LOWER_BOUND = 120
    """int: The lower bound for blue hue value in HSV color space."""
    
    BLUE_HUE_UPPER_BOUND = 150
    """int: The upper bound for blue hue value in HSV color space."""
    
    BLUE_SAT_LOWER_BOUND = 50
    """int: The lower bound for blue saturation value in HSV color space."""
    
    BLUE_SAT_UPPER_BOUND = 255
    """int: The upper bound for blue saturation value in HSV color space."""
    
    BLUE_VAL_LOWER_BOUND = 50
    """int: The lower bound for blue value value in HSV color space."""
    
    BLUE_VAL_UPPER_BOUND = 255
    """int: The upper bound for blue value value in HSV color space."""
    
    RED_HUE_LOWER_BOUND = 0
    """int: The lower bound for red hue value in HSV color space."""
    
    RED_HUE_UPPER_BOUND = 0
    """int: The upper bound for red hue value in HSV color space."""
    
    RED_SAT_LOWER_BOUND = 36
    """int: The lower bound for red saturation value in HSV color space."""
    
    RED_SAT_UPPER_BOUND = 255
    """int: The upper bound for red saturation value in HSV color space."""
    
    RED_VAL_LOWER_BOUND = 78
    """int: The lower bound for red value value in HSV color space."""
    
    RED_VAL_UPPER_BOUND = 255
    """int: The upper bound for red value value in HSV color space."""
    
    PINK_HUE_LOWER_BOUND = 75
    """int: The lower bound for pink hue value in HSV color space."""
    
    PINK_HUE_UPPER_BOUND = 180
    """int: The upper bound for pink hue value in HSV color space."""
    
    PINK_SAT_LOWER_BOUND = 111
    """int: The lower bound for pink saturation value in HSV color space."""
    
    PINK_SAT_UPPER_BOUND = 255
    """int: The upper bound for pink saturation value in HSV color space."""
    
    PINK_VAL_LOWER_BOUND = 239
    """int: The lower bound for pink value value in HSV color space."""
    
    PINK_VAL_UPPER_BOUND = 255
    """int: The upper bound for pink value value in HSV color space."""
    
    PEDESTRIAN_HUE_LOWER_BOUND = 0
    """int: The lower bound for pedestrian hue value in HSV color space."""
    
    PEDESTRIAN_HUE_UPPER_BOUND = 0
    """int: The upper bound for pedestrian hue value in HSV color space."""
    
    PEDESTRIAN_SAT_LOWER_BOUND = 0
    """int: The lower bound for pedestrian saturation value in HSV color space."""
    
    PEDESTRIAN_SAT_UPPER_BOUND = 125
    """int: The upper bound for pedestrian saturation value in HSV color space."""
    
    PEDESTRIAN_VAL_LOWER_BOUND = 0
    """int: The lower bound for pedestrian value value in HSV color space."""
    
    PEDESTRIAN_VAL_UPPER_BOUND = 255
    """int: The upper bound for pedestrian value value in HSV color space."""

    TOP_CLUE_THRESHOLD = 1900    
        
class ClueConstants:
    """A class that defines constants related to clues."""
    
    CLUE_BANNER_WIDTH = 600
    """int: The width of the clue banner image."""
    
    CLUE_BANNER_HEIGHT = 400
    """int: The height of the clue banner image."""
    
    CLUE_BLURRINESS_THRESHOLD = 13
    """int: The threshold for determining the blurriness of a clue banner image."""

    CONFIDENCE_THRESHOLD = 0.9
    """float: The minimum threshold for confidence"""
    
    CLUE_MAX_WHITE_PIXELS = 45000
    """int: The maximum number of white pixels allowed in a blue-filtered mask of clue banner image."""

    NUM_CROPS = 12
    
    CLUE_VALUE_CROP_TOP = 250
    
    CLUE_VALUE_CROP_BOTTOM = 350
    
    CLUE_VALUE_CROP_WIDTH = 45
    
    CLUE_VALUE_CROP_HEIGHT = 100

    CLUE_TOPICS = ["SIZE", "VICTIM", "CRIME", "TIME", "PLACE", "MOTIVE" ,"WEAPON", "BANDIT"]
    """list str: The list of clue topics."""

    
class CNNConstants:
    
    
    CHARACTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z','0','1','2','3','4', '5', '6', '7', '8', '9', " "]
    
    CHARACTERS_COUNT = len(CHARACTERS)
        
class States:
    NORMAL_DRV = 0
    DIRT_DRV = 1
    RED_LINE_DRV = 2
    MOUNTAIN_DRV = 3
    PEAK_DRV = 4
    TRUCK_SAFETY = 5
    PEDESTRIAN_SAFETY = 6
    FINISH_COURSE = 7
    DESERT_DRV = 8
    TUNNEL_DRV = 9
       

class ClueLocations:

    FIRST = [5.5,2,0,0,0,0]
    SECOND = [5.5,-1,0,0,0,0]
    THIRD = [4.5,-1.5,0,0,-1.5,0]
    FORTH = [0.5,-1,0,0,-3.25,0]
    FIFTH = [0.5,2,0,0,0,0]
    SIXTH = [-3,1.5,0,0,-1.5,0]
    SEVENTH = [-4.25,-2,0,01.5,0,0]