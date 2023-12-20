import cv2 as cv
import numpy as np
frame = cv.imread("/home/eray/Desktop/opencv/codes/dama7.jpeg")  # Change the path to your image

if frame is None:
    print("Error: Couldn't read the image.")
    exit()
display_size = (800, 600)
frame = cv.resize(frame, display_size)
def findCorners(frame):
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Apply binary inverse thresholding
    _, binaryinv = cv.threshold(gray, 32, 255, cv.THRESH_BINARY_INV)

    # Apply GaussianBlur to reduce noise
    blurred = cv.GaussianBlur(binaryinv, (5, 5), 0)
	
    # Use Canny edge detector to find edges
    edges = cv.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    frame_contours = frame.copy()
    cv.drawContours(frame_contours, contours, -1, (0, 255, 0), 2)

    # Display the frame with contours
    cv.imshow("Contours", frame_contours)
    cv.waitKey(0)

    # Initialize extreme points
    top_left = (frame.shape[1], frame.shape[0])
    top_right = (0, frame.shape[0])
    bottom_left = (frame.shape[1], 0)
    bottom_right = (0, 0)

    # Find extreme points
    for contour in contours:
        for point in contour:
            if point[0][0] + point[0][1] < top_left[0] + top_left[1]:
                top_left = tuple(point[0])
            if point[0][0] - point[0][1] > top_right[0] - top_right[1]:
                top_right = tuple(point[0])
            if point[0][0] - point[0][1] < bottom_left[0] - bottom_left[1]:
                bottom_left = tuple(point[0])
            if point[0][0] + point[0][1] > bottom_right[0] + bottom_right[1]:
                bottom_right = tuple(point[0])

    # Define the corners as a NumPy array
    corners = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.float32)

    return corners

width,heigth = 500,500
new_points = np.array([[0, 0], [0, heigth], [width, heigth], [width, 0]],dtype=np.float32)
transformation_matrix = cv.getPerspectiveTransform(findCorners(frame),new_points)
bird_eye_frame = cv.warpPerspective(frame,transformation_matrix,(width,heigth))
cv.imshow("Chessboard Bird Eye", bird_eye_frame)
cv.waitKey(0)
cv.destroyAllWindows()

