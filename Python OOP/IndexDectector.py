import cv2 as cv
import numpy as np
from GlobalFunctions import *

class IndexDetector:
    """
        This class will hold the last 3 index record (including the current one) to return more optimized index result
    """
    white_lastn_square_indexes = list()
    white_result_index = list()
    black_lastn_square_indexes = list()
    black_result_index = list()
    n = 4
    index = 0
    def set_index(self,contours,bird_eye_frame_gray):
        """
            Convert incoming contours into indexes
            Set the indexes into white_indexes according to white_index
        """
        contour_centers = self.get_contours_centers(contours)
        contour_centers_combined = self.combine_close_contours(contour_centers)
        white_checker_index = list()
        black_checker_index = list()
        for cent in contour_centers_combined:
            checker_index = self.get_square_index_from_center(cent)
            is_white = self.is_checker_white(bird_eye_frame_gray,cent)
            if is_white : white_checker_index.append(checker_index)
            else : black_checker_index.append(checker_index)
        if len(self.white_lastn_square_indexes)!= self.n:
            self.white_lastn_square_indexes.append(white_checker_index) 
            self.black_lastn_square_indexes.append(black_checker_index)
        else:
            self.white_lastn_square_indexes[self.index] = white_checker_index
            self.black_lastn_square_indexes[self.index] = black_checker_index
            result = (self.index + 1) % self.n
            self.index = result
    def get_square_index_from_center(self,center):
        """
            Return the sqaure index of incoming center
            Result Example:
                (2,7) = x axis third square and y axis eighth square
                (3,4) = 4 unit right 5 unit below
        """
        global square_x_scaler,square_y_scaler
        return (center[0]//square_x_scaler,center[1]//square_x_scaler)
    
    def is_checker_white(self,frame_gray,point)->bool:
        """ Classify checkers as black(0) or white(1) """
        return frame_gray[point[1],point[0]]>180 and True or False
    
    def combine_close_contours(self,centers,threshold_distance=25)-> list:
        """
            This function is used with circular contours
            If two circular contour is too close two each other , they represent the same checkers.
            So they will be counted as one 
        """
        length = len(centers)
        distance = 0
        result = []
        centers = np.array(centers)
        for i in range(0,length):
            add_flag = True
            distance = 0
            for j in range(i+1,length):
                distance = np.sqrt((centers[i,0] - centers[j,0])**2 + (centers[i,1] - centers[j,1])**2)
                if distance < threshold_distance: # if two contour is close
                    add_flag = False
                    break
            if add_flag:
                result.append([centers[i,0],centers[i,1]])
        
        return np.array(result,dtype=int)    
    
    def get_contours_centers(self,contours)-> list:
        """ Calculate the centroid of the contour """
        contour_centers = []
        for contour in contours:
            M = cv.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                contour_centers.append([cX,cY])
        return np.array(contour_centers,dtype=int)  
              
    def combine_indexes(self):
        combined_white_index = set()
        for x in self.white_lastn_square_indexes:
            combined_white_index = combined_white_index.union(set(x))
        combined_black_index = set()
        for x in self.black_lastn_square_indexes:
            combined_black_index = combined_black_index.union(set(x))
        return list(combined_white_index),list(combined_black_index)
    
    def calculate_indexes(self):
        self.white_result_index,self.black_result_index = self.combine_indexes() 
    
    def visualize_all_checkers(self,radius,bird_eye_frame):
        centers = [get_square_position(x) for x in self.white_result_index]
        for cent in centers:
            cv.putText(bird_eye_frame,"White",(cent[0]-10,cent[1]-20),1,1,(0,255,0),2)
            cv.circle(bird_eye_frame, (cent[0], cent[1]), radius, (0, 0, 255), 3)  
        centers = [get_square_position(x) for x in self.black_result_index]
        for cent in centers:
            cv.putText(bird_eye_frame,"Black",(cent[0]-10,cent[1]-20),1,1,(0,255,0),2)
            cv.circle(bird_eye_frame, (cent[0], cent[1]), radius, (0, 0, 255), 3)              

