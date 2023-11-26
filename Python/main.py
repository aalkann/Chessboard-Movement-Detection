import cv2 as cv
import numpy as np
import random as rng

def is_contour_circle(contour, epsilon_factor=0.02, circularity_threshold=0.5,min_area = 300)-> bool:
    """ Checks if the detected contour is circle or not """
    epsilon = epsilon_factor * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    if len(approx) >= 8:  
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        return circularity > circularity_threshold and area > min_area

    return False

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

def is_checker_white(frame_gray,point)->int:
    """ Classify checkers as black(0) or white(1) """
    return frame_gray[point[1],point[0]]>180 and 1 or 0


# Set available camera                          
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