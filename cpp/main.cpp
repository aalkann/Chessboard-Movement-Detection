#include <opencv2/opencv.hpp>
#include <iostream>
#include "variables.hpp"
#include "preprocessingVariables.hpp"
#include "GameMovementRecorder.hpp"
#include "functions.hpp"
#include "checkers.hpp"
#include "IndexDetector.hpp"

using namespace cv;
using namespace std;

int main() {
    VideoCapture video_capture("/source/vid1.avi");
    
    Mat frame;
    bool success = video_capture.read(frame);
    
    Mat current_full_frame_gray;
    cvtColor(frame, current_full_frame_gray, COLOR_BGR2GRAY);
    Mat old_full_frame_gray;
    current_full_frame_gray.copyTo(old_full_frame_gray);
    bool hand_on_screen = false;
    bool hand_on_screen_previous = false;
    int frame_count_while_hand_out = 1;
    
    Checkers manager;
    IndexDetector index_detector;
    GameMovementRecorder recorder;
    
    Point2f points[4] = {{113, 14}, {51, 248}, {437, 246}, {357, 17}};
    Point2f new_points[4] = {{0, 0}, {0, height}, {width, height}, {width, 0}};
    
    Mat transformation_matrix = getPerspectiveTransform(points, new_points);
    
    double maxVal,minVal;
    cv::Point minLoc, maxLoc;
    while (success) {
        success = video_capture.read(frame);
        
        cvtColor(frame, current_full_frame_gray, COLOR_BGR2GRAY);
        
        Mat bird_eye_frame;
        warpPerspective(frame, bird_eye_frame, transformation_matrix, Size(width, height));
        
        hand_on_screen_previous = hand_on_screen;
        hand_on_screen = is_hand_inside(hand_on_screen, current_full_frame_gray, old_full_frame_gray);
        
        if (hand_on_screen == true) {
            frame_count_while_hand_out = 0;
        } else if (frame_count_while_hand_out == 0 && hand_on_screen == false && hand_on_screen_previous == true) {
            frame_count_while_hand_out += 1;
        }
        
        if (frame_count_while_hand_out != 0 && hand_on_screen == false) {
            Mat bird_eye_frame_gray,bird_eye_frame_gray_temp;
            cvtColor(bird_eye_frame, bird_eye_frame_gray, COLOR_BGR2GRAY);
            bird_eye_frame_gray.copyTo(bird_eye_frame_gray_temp);
            Mat bird_eye_frame_gray_gamma_corrected;
	        if (!bird_eye_frame_gray_temp.empty()) {
                double minVal, maxVal;
                cv::Point minLoc, maxLoc;

                // Find the minimum and maximum values in the input matrix
                cv::minMaxLoc(bird_eye_frame_gray_temp, &minVal, &maxVal, &minLoc, &maxLoc);

                // Ensure the input matrix is of the correct data type
                if (bird_eye_frame_gray_temp.type() != CV_32F && bird_eye_frame_gray_temp.type() != CV_64F) {
                    bird_eye_frame_gray_temp.convertTo(bird_eye_frame_gray_temp, CV_32F); // or CV_64F
                }

                // Apply the power operation and store the result
                cv::pow(bird_eye_frame_gray_temp / maxVal, 0.7, bird_eye_frame_gray_gamma_corrected);

                // Convert the resulting matrix to 8-bit unsigned integer format
                bird_eye_frame_gray_gamma_corrected.convertTo(bird_eye_frame_gray_gamma_corrected, CV_8U, 255);
            } else {
                // Handle the case where the input matrix is empty
                std::cerr << "Error: Empty input matrix." << std::endl;
                // Additional error handling if needed
            }
            
            Mat bird_eye_frame_blur;
            GaussianBlur(bird_eye_frame_gray_gamma_corrected, bird_eye_frame_blur, Size(gaussianBlurKernelSize, gaussianBlurKernelSize), 0);
            
            Mat edges;
            Canny(bird_eye_frame_blur, edges, threshold1Canny, threshold2Canny);
            
            Mat dilated_edges;
            dilate(edges, dilated_edges, getStructuringElement(MORPH_RECT, Size(kernelMorp.rows, kernelMorp.cols)), Point(-1, -1), 2);
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(dilated_edges, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
            vector<vector<Point>> circle_contours;
            for (int i = 0; i < contours.size(); i++) {
                if (is_contour_circle(contours[i])) {
                    circle_contours.push_back(contours[i]);
                }
            }
            
            index_detector.set_index(circle_contours, bird_eye_frame_gray);
            
            float radius = get_contour_radius(circle_contours[0]);
            
            frame_count_while_hand_out += 1;
        }
        
        if (frame_count_while_hand_out == index_detector.nGet() + 1 && hand_on_screen == false) {
            frame_count_while_hand_out = 0;
            
            index_detector.calculate_indexes();
            
            manager.set_square_white_index(index_detector.white_result_indexGet());
            manager.set_square_black_index(index_detector.black_result_indexGet());
            
            tuple<Player,cv::Point,cv::Point,std::vector<cv::Point>> record = manager.find_different_index();
            
            if (std::get<0>(record) != Player::Null) {
                movementRecord r;
                r.p = std::get<0>(record);
                r.old_player = std::get<1>(record);
                r.new_player = std::get<2>(record);
                r.defeats = std::get<3>(record);
                recorder.record(r);
            }
        }
        
        visualize_hand_tracking(old_full_frame_gray, current_full_frame_gray);
        current_full_frame_gray.copyTo(old_full_frame_gray);
        manager.apply_changes(bird_eye_frame, radius);
        
        imshow("Frame ", frame);
        imshow("Game", bird_eye_frame);
        
        int key = waitKey(100);
        if (key == 27) {
            recorder.show();
            break;
        }
    }
    
    return 0;
}



