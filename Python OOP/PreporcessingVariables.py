import numpy as np

guassian_blur_kernel_size = 7  # Gaussian Blur Kernel Size
threshold1_canny = 60  # Threshold value 1 for Canny Edge Detecter
threshold2_canny = 120 # Threshold value 2 for Canny Edge Detecter
kernel_morp = np.ones((3,3), np.uint8) # Morphological operation kernel