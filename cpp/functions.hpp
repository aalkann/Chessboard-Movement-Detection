#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
std::string point_to_string(cv::Point p){
    string temp = "x: " + std::to_string(p.x) + " y: " + std::to_string(p.y);
    return temp;
}
std::string to_string(Player player) {
    return (player == Player::White) ? "WHITE" : "BLACK";
}
bool is_contour_circle(vector<Point> contour, double epsilon_factor=0.02, double circularity_threshold=0.4, int min_area=300) {
    double epsilon = epsilon_factor * arcLength(contour, true);
    vector<Point> approx;
    approxPolyDP(contour, approx, epsilon, true);
    if (approx.size() >= 6) {
        double area = contourArea(contour);
        double perimeter = arcLength(contour, true);
        double circularity = 4 * CV_PI * (area / (perimeter * perimeter));
        return circularity > circularity_threshold && area > min_area && area < max_area;
    }
    return false;
}

bool is_hand_inside(bool hand_on_screen, Mat current_frame, Mat old_frame, int thresh=60, int sensitivity=6) {
    Mat diff;
    absdiff(old_frame, current_frame, diff);
    threshold(diff, diff, thresh, 255, THRESH_BINARY);
    int different_pixels_number = countNonZero(diff);
    if(different_pixels_number == 0) different_pixels_number = 1;
    double result = diff.total() / different_pixels_number;
    if (result <= sensitivity && !hand_on_screen) {
        return true;
    }
    else if (result >= sensitivity && hand_on_screen) {
        return false;
    }
    else {
        return hand_on_screen;
    }
}

cv::Point get_square_position(cv::Point square_index) {
    int x_position = square_index.x * square_x_scaler + square_x_bias;
    int y_position = square_index.y * square_y_scaler + square_y_bias;
    cv::Point p(x_position,y_position);
    return p;
}

int get_contour_radius(vector<Point> contour) {
    double area = contourArea(contour);
    double perimeter = arcLength(contour, true);
    int radius = static_cast<int>((2 * area) / perimeter);
    return radius;
}

void visualize_hand_tracking(Mat old_frame, Mat current_frame) {
    Mat diff;
    absdiff(old_frame, current_frame, diff);
    threshold(diff, diff, 60, 255, THRESH_BINARY);
    imshow("Frame Diff", diff);
}



