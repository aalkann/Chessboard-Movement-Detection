import cv2 as cv
import numpy as np


def is_game_board_contour(contour,threshold=0.9):
    global display_size
    frame_size = display_size[0]*display_size[1]
    # Get the bounding rectangle around the contour
    _, _, w, h = cv.boundingRect(contour)

    epsilon = 0.02 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    area = cv.contourArea(approx)
    area = area < 1 and 1 or area
    sqaure_ratio = area/frame_size

    # Calculate the aspect ratio
    aspect_ratio = float(w) / h

    aspect_ratio = aspect_ratio == 1 and 0.95 or aspect_ratio
    # The aspect ratio of a square is 1, so you can measure similarity based on the difference
    similarity = 1 / abs(aspect_ratio - 1)

    return similarity >= threshold and len(approx) <= 4 and sqaure_ratio > 0.1

video_capture = cv.VideoCapture("source/vid2.mp4")

success , frame = video_capture.read()  # Change the path to your image

if frame is None:
    print("Error: Couldn't read the image.")
    exit()
display_size = (800, 600)
frame = cv.resize(frame, display_size)

def GetGameboardCoordinates(frame,shift=3):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, binaryinv = cv.threshold(gray, 70, 255, cv.THRESH_BINARY_INV)    
    dilated = cv.dilate(binaryinv,(5,5),iterations=2)
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    square_contour = [x for x in contours if is_game_board_contour(x)][0]
    frame_contours = frame.copy()
    cv.drawContours(frame_contours, square_contour, -1, (0, 255, 0), 2)
    epsilon = 0.02 * cv.arcLength(square_contour, True)
    approx = cv.approxPolyDP(square_contour, epsilon, True)    
    
    # Display the frame with contours
    cv.imshow("Contours", frame_contours)
    # cv.imshow("binaryinv", binaryinv)
    cv.imshow("dilated", dilated)
    # cv.imshow("edges", edges)
    cv.waitKey(0)

    # Initialize extreme points
    top_left = (frame.shape[1], frame.shape[0])
    top_right = (0, frame.shape[0])
    bottom_left = (frame.shape[1], 0)
    bottom_right = (0, 0)

    # Find extreme points
    for point in approx:
        if point[0][0] + point[0][1] < top_left[0] + top_left[1]:
            top_left = tuple(point[0])
        if point[0][0] - point[0][1] > top_right[0] - top_right[1]:
            top_right = tuple(point[0])
        if point[0][0] - point[0][1] < bottom_left[0] - bottom_left[1]:
            bottom_left = tuple(point[0])
        if point[0][0] + point[0][1] > bottom_right[0] + bottom_right[1]:
            bottom_right = tuple(point[0])
    
    top_left = tuple([top_left[0]-shift,top_left[1]- shift])
    top_right = tuple([top_right[0]+shift,top_right[1]- shift])
    bottom_left = tuple([bottom_left[0]-shift,bottom_left[1]+ shift])
    bottom_right = tuple([bottom_right[0]+shift,bottom_right[1]+ shift])

    # Define the corners as a NumPy array
    corners = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.float32)
    return corners

width,heigth = 500,500
new_points = np.array([[0, 0], [0, heigth], [width, heigth], [width, 0]],dtype=np.float32)
transformation_matrix = cv.getPerspectiveTransform(GetGameboardCoordinates(frame),new_points)
bird_eye_frame = cv.warpPerspective(frame,transformation_matrix,(width,heigth))
cv.imshow("Chessboard Bird Eye", bird_eye_frame)
cv.waitKey(0)
cv.destroyAllWindows()

