import cv2 as cv
import numpy as np
import random as rng

video_capture = cv.VideoCapture("source/vid1.avi")
# Read the first frame
success, frame = video_capture.read()
# Set the width and height for bird'eye view
width,heigth = 400,500

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

while success:
    success, frame = video_capture.read()
    frame_original = frame.copy()

    # Using transformation matrix convert frame into "Bird'Eye View"
    bird_eye_frame = cv.warpPerspective(frame,transformation_matrix,(width,heigth))

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

    # Get the center of each circular contour and combine close ones
    contour_centers = get_contours_centers(circle_contours)
    contour_centers_combined = combine_close_contours(contour_centers) 
    radius = get_contour_radius(circle_contours[0])

    # Visualize the current result 
    for cent in contour_centers:
        cv.circle(bird_eye_frame, (cent[0], cent[1]), 4, (0, 255, 0), -1)
        if cent in contour_centers_combined:  
            text = is_checker_white(bird_eye_frame_gray,cent) == 1 and "White" or "Black"
            cv.putText(bird_eye_frame,text,(cent[0]-10,cent[1]-20),1,1,(0,255,0),2)
            cv.circle(bird_eye_frame, (cent[0], cent[1]), radius, (0, 0, 255), 3)  # Adjust color and radius as needed                  

    # Print Matrices
    print(f"Detected all circle contours count:{len(circle_contours)}")
    print(f"Filtered circle contours count:{len(contour_centers_combined)}")

    # Show the result
    cv.imshow("Chessboard Bird Eye", bird_eye_frame)

    key = cv.waitKey(300)
    if key == 27:
        break