import cv2 as cv
import numpy as np
from GlobalVariables import *

def is_contour_circle(contour, epsilon_factor=0.02, circularity_threshold=0.4,min_area = 300)-> bool:
    """ Checks if the detected contour is circle or not """
    global radius , max_area
    epsilon = epsilon_factor * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    
    if len(approx) >= 6:  
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        return circularity > circularity_threshold and area > min_area and area < max_area
    return False

def is_hand_inside(hand_on_screen,current_frame,old_frame,threshold = 60,sensitivity=6):
    diff = cv.absdiff(old_frame, current_frame)
    t,diff = cv.threshold(diff,threshold,255,cv.THRESH_BINARY)
    different_pixels_number = sum(sum(diff))
    result  = diff.size/different_pixels_number
    
    if result <= sensitivity and hand_on_screen == False:
        return True
    elif result >= sensitivity and hand_on_screen == True:
        return False
    else:
        return hand_on_screen

def get_square_position(square_index):
        global square_x_scaler,square_y_scaler,square_x_bias,square_y_bias
        x_position = square_index[0] * square_x_scaler + square_x_bias
        y_position = square_index[1] * square_y_scaler + square_y_bias
        return ( x_position , y_position )

def get_contour_radius(contour)->int:
    """ Return the radius of circular contour """

    area =  cv.contourArea(contour)
    perimeter = cv.arcLength(contour,True)
    radius = int((2*area) / perimeter)
    return radius  

def visualize_hand_tracking(old_frame,current_frame):
    diff = cv.absdiff(old_frame, current_frame)
    t,diff = cv.threshold(diff,60,255,cv.THRESH_BINARY)
    cv.imshow("Frame Diff", diff)