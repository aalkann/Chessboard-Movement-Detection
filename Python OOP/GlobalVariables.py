import numpy as np

# Set the width and height for bird'eye view
width,heigth = 500,500
# Parameters for checkers positioning and locationing
square_x_scaler = width//8
square_x_bias = square_x_scaler//2
square_y_scaler = heigth//8
square_y_bias = square_y_scaler//2
radius = (width//20)
max_area =  (np.pi * radius ** 2) + (np.pi * radius ** 2)/2


