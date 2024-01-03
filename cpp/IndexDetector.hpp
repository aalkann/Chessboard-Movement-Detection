#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "functions.hpp"
struct PointComparator {
    bool operator()(const cv::Point& lhs, const cv::Point& rhs) const {
        if (lhs.y != rhs.y) {
            return lhs.y < rhs.y;
        }
        return lhs.x < rhs.x;
    }
};
class IndexDetector {
private:
    std::vector<std::vector<cv::Point>> white_lastn_square_indexes;
    std::vector<cv::Point> white_result_index;
    std::vector<std::vector<cv::Point>> black_lastn_square_indexes;
    std::vector<cv::Point> black_result_index;
    int n = 4;
    int index = 0;

public:
	std::vector<cv::Point> white_result_indexGet(){
		return white_result_index;
	}
	std::vector<cv::Point> black_result_indexGet(){
		return black_result_index;
	}
	int nGet(){
		return n;
	}
    void set_index(std::vector<std::vector<cv::Point>> contours, cv::Mat bird_eye_frame_gray) {
        std::vector<std::vector<int>> contour_centers = get_contours_centers(contours);
        std::vector<std::vector<int>> contour_centers_combined = combine_close_contours(contour_centers);
        std::vector<cv::Point> white_checker_index;
        std::vector<cv::Point> black_checker_index;
        for (auto cent : contour_centers_combined) {
            cv::Point checker_index = get_square_index_from_center(cent);
            bool is_white = is_checker_white(bird_eye_frame_gray, cent);
            if (is_white) {
                white_checker_index.push_back(checker_index);
            } else {
                black_checker_index.push_back(checker_index);
            }
        }
        if (white_lastn_square_indexes.size() != n) {
            white_lastn_square_indexes.push_back(white_checker_index);
            black_lastn_square_indexes.push_back(black_checker_index);
        } else {
            white_lastn_square_indexes[index] = white_checker_index;
            black_lastn_square_indexes[index] = black_checker_index;
            int result = (index + 1) % n;
            index = result;
        }
    }

    cv::Point get_square_index_from_center(std::vector<int> center) {
        int square_x_scaler, square_y_scaler;
        cv::Point p(center[0] / square_x_scaler,center[1] / square_x_scaler);
        return p;
    }

    bool is_checker_white(cv::Mat frame_gray, std::vector<int> point) {
        return frame_gray.at<uchar>(point[1], point[0]) > 180;
    }

    std::vector<std::vector<int>> combine_close_contours(std::vector<std::vector<int>> centers, int threshold_distance = 25) {
        int length = centers.size();
        int distance = 0;
        std::vector<std::vector<int>> result;
        for (int i = 0; i < length; i++) {
            bool add_flag = true;
            distance = 0;
            for (int j = i + 1; j < length; j++) {
                distance = std::sqrt(std::pow(centers[i][0] - centers[j][0], 2) + std::pow(centers[i][1] - centers[j][1], 2));
                if (distance < threshold_distance) {
                    add_flag = false;
                    break;
                }
            }
            if (add_flag) {
                result.push_back({centers[i][0], centers[i][1]});
            }
        }
        return result;
    }

    std::vector<std::vector<int>> get_contours_centers(std::vector<std::vector<cv::Point>> contours) {
        std::vector<std::vector<int>> contour_centers;
        for (auto contour : contours) {
            cv::Moments M = cv::moments(contour);
            if (M.m00 != 0) {
                int cX = M.m10 / M.m00;
                int cY = M.m01 / M.m00;
                contour_centers.push_back({cX, cY});
            }
        }
        return contour_centers;
    }

    std::vector<std::vector<cv::Point>> combine_indexes() {
        std::set<cv::Point,PointComparator> combined_white_index;
        for (auto x : white_lastn_square_indexes) {
            for (auto i : x) {
                combined_white_index.insert(i);
            }
        }
        std::set<cv::Point,PointComparator> combined_black_index;
        for (auto x : black_lastn_square_indexes) {
            for (auto i : x) {
                combined_black_index.insert(i);
            }
        }
        std::vector<cv::Point> white_index(combined_white_index.begin(), combined_white_index.end());
        std::vector<cv::Point> black_index(combined_black_index.begin(), combined_black_index.end());
        return {white_index, black_index};
    }

    void calculate_indexes() {
        std::vector<std::vector<cv::Point>> indexes = combine_indexes();
        white_result_index = indexes[0];
        black_result_index = indexes[1];
    }

    void visualize_all_checkers(int radius, cv::Mat bird_eye_frame) {
        std::vector<cv::Point> white_centers;
        for (auto x : white_result_index) {
            white_centers.push_back(get_square_position(x));
        }
        for (auto cent : white_centers) {
            cv::putText(bird_eye_frame, "White", cv::Point(cent.x - 10, cent.y - 20), 1, 1, cv::Scalar(0, 255, 0), 2);
            cv::circle(bird_eye_frame, cv::Point(cent.x, cent.y), radius, cv::Scalar(0, 0, 255), 3);
        }
        std::vector<cv::Point> black_centers;
        for (auto x : black_result_index) {
            black_centers.push_back(get_square_position(x));
        }
        for (auto cent : black_centers) {
            cv::putText(bird_eye_frame, "Black", cv::Point(cent.x - 10, cent.y - 20), 1, 1, cv::Scalar(0, 255, 0), 2);
            cv::circle(bird_eye_frame, cv::Point(cent.x, cent.y), radius, cv::Scalar(0, 0, 255), 3);
        }
    }
};



