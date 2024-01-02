#pragma once

#include <opencv2/opencv.hpp>

int gaussianBlurKernelSize = 7;  // Gaussian Blur Kernel Size
int threshold1Canny = 60;  // Threshold value 1 for Canny Edge Detector
int threshold2Canny = 120; // Threshold value 2 for Canny Edge Detector
cv::Mat kernelMorp = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // Morphological operation kernel
