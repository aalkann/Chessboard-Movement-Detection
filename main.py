import cv2 as cv
import numpy as np
import random as rng

video_capture = cv.VideoCapture("source/vid1.avi")
success, frame = video_capture.read()
width,heigth = 400,500

# Detect Chessboard Coordinates
points = np.array([[91, 2], [9, 261], [472, 259], [376, 6]],dtype=np.float32)  # For now use static point
new_points = np.array([[0, 0], [0, heigth], [width, heigth], [width, 0]],dtype=np.float32)

# Calculate the transformation matrix
transformation_matrix = cv.getPerspectiveTransform(points,new_points)
print(transformation_matrix)

while success:
    success, frame = video_capture.read()
    frame_original = frame.copy()

    # Apply transformation matrix into the orignal image
    bird_eye_frame = cv.warpPerspective(frame,transformation_matrix,(width,heigth))
    bird_eye_frame_gray = cv.cvtColor(bird_eye_frame,cv.COLOR_BGR2GRAY)
    bird_eye_frame_hsv = cv.cvtColor(bird_eye_frame,cv.COLOR_BGR2HSV)

    # White Color
    low = np.array([0, 0, 170])
    high = np.array([180, 85, 255])
    mask = cv.inRange(bird_eye_frame_hsv, low, high)
    result = cv.bitwise_and(bird_eye_frame, bird_eye_frame, mask=mask)

    cv.imshow("Chessboard Bird Eye", bird_eye_frame)
    cv.imshow("Chessboard Bird Eye Gray", bird_eye_frame_gray)
    cv.imshow("White", result)
    cv.imshow("Chessboard", frame_original)

    if cv.waitKey(300) == 27:
        break
