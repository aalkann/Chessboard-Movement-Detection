import cv2 as cv
import numpy as np
from GlobalVariables import heigth,width
from PreporcessingVariables import *
from GameMovementRecorder import GameMovementRecorder
from GlobalFunctions import is_hand_inside,is_contour_circle,get_contour_radius,visualize_hand_tracking
from Checkers import Checkers
from IndexDectector import IndexDetector

def main():
    # Set available camera                          
    video_capture = cv.VideoCapture("source/vid1.avi")
    # Read the first frame
    success, frame = video_capture.read()

    # Parameter to track hand
    current_full_frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    old_full_frame_gray = current_full_frame_gray
    hand_on_screen = False
    hand_on_screen_previous = False
    frame_count_while_hand_out = 1

    # Initiate Tracker Objects
    manager = Checkers()
    index_detector = IndexDetector()
    recorder = GameMovementRecorder()

    # Detect Chessboard Coordinates
    points = np.array([[113, 14], [51, 248], [437, 246], [357, 17]],dtype=np.float32)  
    new_points = np.array([[0, 0], [0, heigth], [width, heigth], [width, 0]],dtype=np.float32)
    
    # Calculate the transformation matrix
    transformation_matrix = cv.getPerspectiveTransform(points,new_points)
    
    while success:
        # Read next frame
        success, frame = video_capture.read()
        
        # Update current frame
        current_full_frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        # Using transformation matrix convert frame into "Bird'Eye View"
        bird_eye_frame = cv.warpPerspective(frame,transformation_matrix,(width,heigth))
        
        # Check if hand is in
        hand_on_screen_previous = hand_on_screen
        hand_on_screen = is_hand_inside(hand_on_screen,current_full_frame_gray,old_full_frame_gray)

        # According to the hand status update frame_count_while_hand_out
        if hand_on_screen == True:
            frame_count_while_hand_out = 0
        elif frame_count_while_hand_out == 0 and hand_on_screen == False and hand_on_screen_previous == True:
            frame_count_while_hand_out += 1

        # If hand is currently out , calculate the positions of checkers and save for later prediction
        if frame_count_while_hand_out != 0 and hand_on_screen == False: 
            # Make frame gray scale
            bird_eye_frame_gray = cv.cvtColor(bird_eye_frame,cv.COLOR_BGR2GRAY)
            # Apply gamma correction to reduce the shaddow effect
            bird_eye_frame_gray_gamma_corrected = np.array(np.power(bird_eye_frame_gray/np.max(bird_eye_frame_gray),0.7) * 255).astype(np.uint8)
            # Blur the frame for making Canny algorithm more precise
            bird_eye_frame_blur = cv.GaussianBlur(bird_eye_frame_gray_gamma_corrected,(guassian_blur_kernel_size,guassian_blur_kernel_size),0)
            # Detect Edges 
            edges = cv.Canny(bird_eye_frame_blur,threshold1_canny,threshold2_canny)
            # Apply Morphological operations to fix broken edges
            dilated_edges = cv.dilate(edges, kernel_morp, iterations=2)
            # Find Circlular Contours
            contours, _ = cv.findContours(dilated_edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            circle_contours = [contour for contour in contours if is_contour_circle(contour)]
            # Calculate and save the index(positions of checkers)
            index_detector.set_index(circle_contours,bird_eye_frame_gray)
            # Calculate radius for better visualization
            radius = get_contour_radius(circle_contours[0])
            # Increase frame_count_while_hand_out by one 
            frame_count_while_hand_out+= 1

        # After index_detector.n frame waiting while hand is out , calculate the precise result(move) and save it
        if frame_count_while_hand_out == index_detector.n + 1 and hand_on_screen == False:
            # Reset frame_count_while_hand_out
            frame_count_while_hand_out = 0
            # Calculate indexes
            index_detector.calculate_indexes()
            # Set the current index positions into the manager 
            manager.set_square_white_index(index_detector.white_result_index)
            manager.set_square_black_index(index_detector.black_result_index)
            # Manager will compare previous and current positions and return a movement record
            # Record = (player , old position , new position , defeated checker positions)
            record = manager.find_different_index()

            # If calculation is successful , save the record
            if record != None:
                recorder.record(record[0],record[1],record[2],record[3])
            
            # After saving the record update old full frame which is used for hand tracking
            old_full_frame_gray = current_full_frame_gray

        # For hand detection visualization
        visualize_hand_tracking(old_full_frame_gray,current_full_frame_gray)

        # Apply changes on the bird eye frame
        manager.apply_changes(bird_eye_frame,radius)

        # Show the result
        cv.imshow("Frame ",frame)
        cv.imshow("Game",bird_eye_frame) 

        key = cv.waitKey(100)
        if key == 27:
            recorder.show()
            break

if __name__ == "__main__":
    main()
