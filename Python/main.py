import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class Movement(Enum):
    White = 1 # White move without black defeated
    WhiteWithBlack = 2 # White move with black defeated
    Black = 3 # Black move without white defeated
    BlackWithWhite = 4 # Black move with white defeated
    Wrong = 5 # Wrong movement

class Player(Enum):
    White = 1 # White move without black defeated
    Black = 2 # Black move without white defeated

class Checkers():
    square_index_with_white_checkers_old = list()
    square_index_with_white_checkers_new = list()

    square_index_with_black_checkers_old = list()
    square_index_with_black_checkers_new = list()

    movement = None
    movement_old_index = None
    movement_new_index = None
    defeated_indexes = None

    def get_indexes_changes(self,old_indexes,new_indexes):
        """
            Return the indexes that old_indexes have but new_indexes doesn't
        """
        result = list()
        for old in old_indexes:
            flag = False
            for new in new_indexes:
                if old == new:
                    flag = True
                    break
            if not flag:
                result.append(old)
        return result

    def what_happaned(self,nnw,now,nnb,nob):
        """
            Decide what kind of movement has happened
            nnw = number of new white checkers
            now = number of old white checkers
            nnb = number of new black checkers
            nob = number of old black checkers
        """
        if nnw == 1 and nnb == 0:
            if nob > 0:
                return Movement.WhiteWithBlack
            return Movement.White
        
        elif nnb == 1 and nnw == 0:
            if now > 0:
                return Movement.BlackWithWhite
            return Movement.Black
        
        return Movement.Wrong

    def find_different_white_index(self): 

        # Eskide olup da yenide olmayan 
        old_sqaure_white_indexes = self.get_indexes_changes(self.square_index_with_white_checkers_old,self.square_index_with_white_checkers_new)
        old_sqaure_black_indexes = self.get_indexes_changes(self.square_index_with_black_checkers_old,self.square_index_with_black_checkers_new)
        

        # Yenide olup da eskide olmayan 
        new_sqaure_white_indexes = self.get_indexes_changes(self.square_index_with_white_checkers_new,self.square_index_with_white_checkers_old)
        new_sqaure_black_indexes = self.get_indexes_changes(self.square_index_with_black_checkers_new,self.square_index_with_black_checkers_old)

        number_new_white = len(new_sqaure_white_indexes) 
        number_old_white = len(old_sqaure_white_indexes) 
        number_new_black = len(new_sqaure_black_indexes) 
        number_old_black = len(old_sqaure_black_indexes)   

        self.movement = self.what_happaned(number_new_white,number_old_white,number_new_black,number_old_black)      
        
        print("Movement:",self.movement)

        match self.movement:
            case Movement.White:
                self.movement_old_index = self.get_closest_index(old_sqaure_white_indexes,new_sqaure_white_indexes[0])
                self.movement_new_index = new_sqaure_white_indexes[0]
                self.defeated_indexes = None
                print("White New: ",new_sqaure_white_indexes)
                print("White Old: ",old_sqaure_white_indexes)
            case Movement.WhiteWithBlack:
                self.movement_old_index = self.get_closest_index(old_sqaure_white_indexes,new_sqaure_white_indexes[0])
                self.movement_new_index = new_sqaure_white_indexes[0]
                self.defeated_indexes = old_sqaure_black_indexes
                print("White New: ",new_sqaure_white_indexes)
                print("White Old: ",old_sqaure_white_indexes)
                print("Defeated Blacks: ",old_sqaure_black_indexes)
            case Movement.Black:
                self.movement_old_index = self.get_closest_index(old_sqaure_black_indexes,new_sqaure_black_indexes[0])
                self.movement_new_index = new_sqaure_black_indexes[0]
                self.defeated_indexes = None                
                print("Black New: ",new_sqaure_black_indexes)
                print("Black Old: ",old_sqaure_black_indexes)
            case Movement.BlackWithWhite:
                self.movement_old_index = self.get_closest_index(old_sqaure_black_indexes,new_sqaure_black_indexes[0])
                self.movement_new_index = new_sqaure_black_indexes[0]
                self.defeated_indexes = old_sqaure_white_indexes                
                print("Black New: ",new_sqaure_black_indexes)
                print("Black Old: ",old_sqaure_black_indexes)
                print("Defeated Whites: ",old_sqaure_white_indexes)
            case _:
                self.movement_old_index = None
                self.movement_new_index = None
                self.defeated_indexes = None

        print()
        

    def set_square_white_index(self,new_index:list):
        self.square_index_with_white_checkers_old = self.square_index_with_white_checkers_new
        self.square_index_with_white_checkers_new = new_index

    def set_square_black_index(self,new_index:list):
        self.square_index_with_black_checkers_old = self.square_index_with_black_checkers_new
        self.square_index_with_black_checkers_new = new_index
        
    def get_closest_index(self,indexes,point):
        if len(indexes) == 0:
            return 
        
        result = indexes[0]
        distance_min = abs(indexes[0][0] - point[0]) + abs(indexes[0][1]-point[1])
        for index in indexes:
            distance = abs(index[0] - point[0]) + abs(index[1]-point[1])
            if distance < distance_min:
                result = index
                distance_min = distance
        return result
    
    def visualize_changes(self):
        global bird_eye_frame
        if self.movement_new_index == None or self.movement_old_index == None:
            return
        
        mover_old_position = get_square_position(self.movement_old_index)
        mover_new_position = get_square_position(self.movement_new_index)

        cv.putText(bird_eye_frame,f"Old {self.movement_old_index}",(mover_old_position[0]-10,mover_old_position[1]-20),1,1,(255,0,0),2)
        cv.circle(bird_eye_frame, (mover_old_position[0], mover_old_position[1]), radius, (255, 0, 0), 3)

        cv.putText(bird_eye_frame,f"New {self.movement_new_index}",(mover_new_position[0]-10,mover_new_position[1]-20),1,1,(0,255,0),2)
        cv.circle(bird_eye_frame, (mover_new_position[0], mover_new_position[1]), radius, (0, 255, 0), 3)

        cv.arrowedLine(bird_eye_frame,mover_old_position,mover_new_position,(255,255,0),2)

        if self.defeated_indexes != None: 
            defeated_positions = [get_square_position(x) for x in self.defeated_indexes]
            for cent,position in zip(self.defeated_indexes,defeated_positions):
                cv.putText(bird_eye_frame,f"Defeated {cent}",(position[0]-10,position[1]-20),1,1,(0,0,255),2)
                cv.circle(bird_eye_frame, (position[0], position[1]), radius, (0, 0, 255), 3)     







def is_contour_circle(contour, epsilon_factor=0.02, circularity_threshold=0.4,min_area = 300)-> bool:
    """ Checks if the detected contour is circle or not """
    global radius,max_area
    epsilon = epsilon_factor * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    
    if len(approx) >= 6:  
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        return circularity > circularity_threshold and area > min_area and area < max_area
    return False

def is_hand_inside(current_frame,threshold = 60,sensitivity=6):
    global hand_on_screen,old_frame
    diff = cv.absdiff(old_frame, current_frame)
    t,diff = cv.threshold(diff,threshold,255,cv.THRESH_BINARY)
    different_pixels_number = sum(sum(diff))
    result  = diff.size/different_pixels_number
    
    if result <= sensitivity and hand_on_screen == False:
        hand_on_screen = True
        # print("Hand in")
    elif result >= sensitivity and hand_on_screen == True:
        hand_on_screen = False
        # print("Hand out")
    # print("Result: ",result)

def set_old_frame_for_hand_detection(current_frame):
    global old_frame
    old_frame = current_frame

def get_square_position(square_index):
        global square_x_scaler,square_y_scaler,square_x_bias,square_y_bias
        x_position = square_index[0] * square_x_scaler + square_x_bias
        y_position = square_index[1] * square_y_scaler + square_y_bias
        return ( x_position , y_position )
    


def get_contours_centers(contours)-> list:
    """ Calculate the centroid of the contour """
    contour_centers = []
    for contour in contours:
        M = cv.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            contour_centers.append([cX,cY])

    return np.array(contour_centers,dtype=int)

def get_contour_radius(contour)->int:
    """ Return the radius of circular contour """
    area =  cv.contourArea(contour)
    perimeter = cv.arcLength(contour,True)
    radius = int((2*area) / perimeter)
    return radius   

def combine_close_contours(centers,threshold_distance=25)-> list:
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

def is_checker():
    pass

def is_checker_white(frame_gray,point)->bool:
    """ Classify checkers as black(0) or white(1) """
    return frame_gray[point[1],point[0]]>180 and True or False

def get_square_index_from_contour(contour,square_width,square_height):
    """
        Return the sqaure index of incoming contour
        Result Example:
            (2,7) = x axis third square and y axis eighth square
            (3,4) = 4 unit right 5 unit below
    """
    contour_center = get_contours_centers([contour])[0]
    return (contour_center[0]//square_width,contour_center[1]//square_height)

def get_square_index_from_center(center,x_scaler,y_scaler):
    """
        Return the sqaure index of incoming center
        Result Example:
            (2,7) = x axis third square and y axis eighth square
            (3,4) = 4 unit right 5 unit below
    """
    return (center[0]//x_scaler,center[1]//y_scaler)




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

    def set_index(self,contours):
        """
            Convert incoming contours into indexes
            Set the indexes into white_indexes according to white_index
        """
        contour_centers = get_contours_centers(contours)
        contour_centers_combined = combine_close_contours(contour_centers)

        white_checker_index = list()
        black_checker_index = list()

        for cent in contour_centers_combined:
            checker_index = get_square_index_from_center(cent,square_x_scaler,square_y_scaler)
            is_white = is_checker_white(bird_eye_frame_gray,cent)

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

    def combine_indexes(self):

        combined_white_index = set()
        for x in self.white_lastn_square_indexes:
            combined_white_index = combined_white_index.union(set(x))

        combined_black_index = set()
        for x in self.black_lastn_square_indexes:
            combined_black_index = combined_black_index.union(set(x))

        return list(combined_white_index),list(combined_black_index)

    def get_indexes(self):
        self.white_result_index,self.black_result_index = self.combine_indexes()
        return self.white_result_index,self.black_result_index 
    
    def visualize(self,radius):
        centers = [get_square_position(x) for x in self.white_result_index]
        for cent in centers:
            cv.putText(bird_eye_frame,"White",(cent[0]-10,cent[1]-20),1,1,(0,255,0),2)
            cv.circle(bird_eye_frame, (cent[0], cent[1]), radius, (0, 0, 255), 3)  

        centers = [get_square_position(x) for x in self.black_result_index]
        for cent in centers:
            cv.putText(bird_eye_frame,"Black",(cent[0]-10,cent[1]-20),1,1,(0,255,0),2)
            cv.circle(bird_eye_frame, (cent[0], cent[1]), radius, (0, 0, 255), 3)              
        

# Set available camera                          
video_capture = cv.VideoCapture("source/vid1.avi")
# Read the first frame
success, frame = video_capture.read()

# Set the width and height for bird'eye view
width,heigth = 500,500
# Parameters for checkers positioning and locationing
square_x_scaler = width//8
square_x_bias = square_x_scaler//2
square_y_scaler = heigth//8
square_y_bias = square_y_scaler//2

# Checker Manager
manager = Checkers()
index_detector = IndexDetector()

# Parameters for preprocessing 
b_k = 7  # Gaussian Blur Kernel Size
threshold1 = 60  # Threshold value 1 for Canny Edge Detecter
threshold2 = 120 # Threshold value 2 for Canny Edge Detecter
kernel_morp = np.ones((3,3), np.uint8) # Morphological operation kernel

# Detect Chessboard Coordinates
# For now use static point
points = np.array([[113, 14], [51, 248], [437, 246], [357, 17]],dtype=np.float32)  
new_points = np.array([[0, 0], [0, heigth], [width, heigth], [width, 0]],dtype=np.float32)

# Calculate the transformation matrix
transformation_matrix = cv.getPerspectiveTransform(points,new_points)

frame_count = 1
radius = (width//20)
max_area =  (np.pi * radius ** 2) + (np.pi * radius ** 2)/2
old_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
hand_on_screen = False
hand_on_screen_previous = False

while success:
    success, frame = video_capture.read()
    frame_original = frame.copy()
    full_view_frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    # Using transformation matrix convert frame into "Bird'Eye View"
    bird_eye_frame = cv.warpPerspective(frame,transformation_matrix,(width,heigth))
    
    # Check if hand is in
    hand_on_screen_previous = hand_on_screen
    is_hand_inside(full_view_frame_gray)

    if hand_on_screen == True:
        frame_count = 0
    elif frame_count == 0 and hand_on_screen == False and hand_on_screen_previous == True:
        frame_count += 1

    # Hand went out
    if frame_count != 0 and hand_on_screen == False: 

        # Make frame gray scale
        bird_eye_frame_gray = cv.cvtColor(bird_eye_frame,cv.COLOR_BGR2GRAY)
        # Apply gamma correction to reduce the shaddow effect
        bird_eye_frame_gray_gamma_corrected = np.array(np.power(bird_eye_frame_gray/np.max(bird_eye_frame_gray),0.7) * 255).astype(np.uint8)
        # Blur the frame for making Canny algorithm more precise
        bird_eye_frame_blur = cv.GaussianBlur(bird_eye_frame_gray_gamma_corrected,(b_k,b_k),0)
        # Detect Edges 
        edges = cv.Canny(bird_eye_frame_blur,threshold1,threshold2)
        # Apply Morphological operations to fix broken edges
        dilated_edges = cv.dilate(edges, kernel_morp, iterations=2)
        eroded_edges = cv.erode(dilated_edges, kernel_morp, iterations=1)
        # Find Circlular Contours
        contours, _ = cv.findContours(dilated_edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        circle_contours = [contour for contour in contours if is_contour_circle(contour)]
    
        index_detector.set_index(circle_contours)
        radius = get_contour_radius(circle_contours[0])

        frame_count+= 1


    # Display changes
    if frame_count == index_detector.n + 1 and hand_on_screen == False:
        frame_count = 0
        index_detector.get_indexes()
        manager.set_square_white_index(index_detector.white_result_index)
        manager.set_square_black_index(index_detector.black_result_index)
        manager.find_different_white_index()
        set_old_frame_for_hand_detection(full_view_frame_gray)

    # For visualization
    diff = cv.absdiff(old_frame, full_view_frame_gray)
    t,diff = cv.threshold(diff,60,255,cv.THRESH_BINARY)
    # Show all detected checkers
    # index_detector.visualize(radius)
    # Show changes
    manager.visualize_changes()

    # Show the result
    cv.imshow("Frame ",frame)
    cv.imshow("Frame Diff", diff)
    cv.imshow("Chessboard Bird Eye", bird_eye_frame)
    # cv.imshow("dilated_edges", dilated_edges)

    # old_frame = full_view_frame_gray
    key = cv.waitKey(100)
    if key == 27:
        break

    