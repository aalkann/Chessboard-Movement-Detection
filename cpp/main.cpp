#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

enum class Movement {
    White = 1,
    WhiteWithBlack = 2,
    Black = 3,
    BlackWithWhite = 4,
    Wrong = 5
};

enum class Player {
    White = 1,
    Black = 2
};

class GameMovementRecorder {
private:
    std::vector<std::tuple<Player, cv::Point, cv::Point, std::vector<cv::Point>>> records;

public:
    void record(Player player, const cv::Point& old_player, const cv::Point& new_player, const std::vector<cv::Point>& defeateds) {
        records.push_back(std::make_tuple(player, old_player, new_player, defeateds));
    }

    void show() {
        for (const auto& movement : records) {
            std::cout << std::get<0>(movement) << " " << std::get<1>(movement) << " " << std::get<2>(movement) << " ";
            std::cout << "Defeated: ";
            for (const auto& defeated : std::get<3>(movement)) {
                std::cout << defeated << " ";
            }
            std::cout << std::endl;
        }
    }
};
Player get_player(Movement movement);
GameMovementRecorder recorder;
cv::Mat bird_eye_frame;
cv::Mat bird_eye_frame_gray;
int width = 500;
int height = 500;
int radius = width / 20;
int square_x_scaler = width / 8;
int square_x_bias = square_x_scaler / 2;
int square_y_scaler = height / 8;
int square_y_bias = square_y_scaler / 2;
class Checkers {
private:
    std::vector<cv::Point> square_index_with_white_checkers_old;
    std::vector<cv::Point> square_index_with_white_checkers_new;
    std::vector<cv::Point> square_index_with_black_checkers_old;
    std::vector<cv::Point> square_index_with_black_checkers_new;
    Movement movement;
    cv::Point movement_old_index;
    cv::Point movement_new_index;
    std::vector<cv::Point> defeated_indexes;

    std::vector<cv::Point> get_indexes_changes(const std::vector<cv::Point>& old_indexes, const std::vector<cv::Point>& new_indexes) {
        std::vector<cv::Point> result;
        for (const auto& old : old_indexes) {
            bool flag = false;
            for (const auto& new_inx : new_indexes) {
                if (old == new_inx) {
                    flag = true;
                    break;
                }
            }
            if (!flag) {
                result.push_back(old);
            }
        }
        return result;
    }

    Movement what_happaned(int nnw, int now, int nnb, int nob) {
        if (nnw == 1 && nnb == 0) {
            if (nob > 0) {
                return Movement::WhiteWithBlack;
            }
            return Movement::White;
        }
        else if (nnb == 1 && nnw == 0) {
            if (now > 0) {
                return Movement::BlackWithWhite;
            }
            return Movement::Black;
        }
        return Movement::Wrong;
    }

    cv::Point get_closest_index(const std::vector<cv::Point>& indexes, const cv::Point& point) {
        if (indexes.empty()) {
            return cv::Point();
        }
        cv::Point result = indexes[0];
        int distance_min = std::abs(indexes[0].x - point.x) + std::abs(indexes[0].y - point.y);
        for (const auto& index : indexes) {
            int distance = std::abs(index.x - point.x) + std::abs(index.y - point.y);
            if (distance < distance_min) {
                result = index;
                distance_min = distance;
            }
        }
        return result;
    }

public:
    void find_different_white_index() {
        std::vector<cv::Point> old_sqaure_white_indexes = get_indexes_changes(square_index_with_white_checkers_old, square_index_with_white_checkers_new);
        std::vector<cv::Point> old_sqaure_black_indexes = get_indexes_changes(square_index_with_black_checkers_old, square_index_with_black_checkers_new);

        std::vector<cv::Point> new_sqaure_white_indexes = get_indexes_changes(square_index_with_white_checkers_new, square_index_with_white_checkers_old);
        std::vector<cv::Point> new_sqaure_black_indexes = get_indexes_changes(square_index_with_black_checkers_new, square_index_with_black_checkers_old);

        int number_new_white = new_sqaure_white_indexes.size();
        int number_old_white = old_sqaure_white_indexes.size();
        int number_new_black = new_sqaure_black_indexes.size();
        int number_old_black = old_sqaure_black_indexes.size();

        movement = what_happaned(number_new_white, number_old_white, number_new_black, number_old_black);
        std::cout << "Movement: " << static_cast<int>(movement) << std::endl;

        Player player = get_player(movement);
        defeated_indexes.clear();

        switch (movement) {
            case Movement::White:
                movement_old_index = get_closest_index(old_sqaure_white_indexes, new_sqaure_white_indexes[0]);
                movement_new_index = new_sqaure_white_indexes[0];
                std::cout << "White New: ";
                for (const auto& index : new_sqaure_white_indexes) {
                    std::cout << "(" << index.x << ", " << index.y << ") ";
                }
                std::cout << std::endl;
                std::cout << "White Old: ";
                for (const auto& index : old_sqaure_white_indexes) {
                    std::cout << "(" << index.x << ", " << index.y << ") ";
                }
                std::cout << std::endl;
                break;
            case Movement::WhiteWithBlack:
                movement_old_index = get_closest_index(old_sqaure_white_indexes, new_sqaure_white_indexes[0]);
                movement_new_index = new_sqaure_white_indexes[0];
                defeated_indexes = old_sqaure_black_indexes;
                std::cout << "White New: ";
                for (const auto& index : new_sqaure_white_indexes) {
                    std::cout << "(" << index.x << ", " << index.y << ") ";
                }
                std::cout << std::endl;
                std::cout << "White Old: ";
                for (const auto& index : old_sqaure_white_indexes) {
                    std::cout << "(" << index.x << ", " << index.y << ") ";
                }
                std::cout << std::endl;
                std::cout << "Defeated Blacks: ";
                for (const auto& index : old_sqaure_black_indexes) {
                    std::cout << "(" << index.x << ", " << index.y << ") ";
                }
                std::cout << std::endl;
                break;
            case Movement::Black:
                movement_old_index = get_closest_index(old_sqaure_black_indexes, new_sqaure_black_indexes[0]);
                movement_new_index = new_sqaure_black_indexes[0];
                std::cout << "Black New: ";
                for (const auto& index : new_sqaure_black_indexes) {
                    std::cout << "(" << index.x << ", " << index.y << ") ";
                }
                std::cout << std::endl;
                std::cout << "Black Old: ";
                for (const auto& index : old_sqaure_black_indexes) {
                    std::cout << "(" << index.x << ", " << index.y << ") ";
                }
                std::cout << std::endl;
                break;
            case Movement::BlackWithWhite:
                movement_old_index = get_closest_index(old_sqaure_black_indexes, new_sqaure_black_indexes[0]);
                movement_new_index = new_sqaure_black_indexes[0];
                defeated_indexes = old_sqaure_white_indexes;
                std::cout << "Black New: ";
                for (const auto& index : new_sqaure_black_indexes) {
                    std::cout << "(" << index.x << ", " << index.y << ") ";
                }
                std::cout << std::endl;
                std::cout << "Black Old: ";
                for (const auto& index : old_sqaure_black_indexes) {
                    std::cout << "(" << index.x << ", " << index.y << ") ";
                }
                std::cout << std::endl;
                std::cout << "Defeated Whites: ";
                for (const auto& index : old_sqaure_white_indexes) {
                    std::cout << "(" << index.x << ", " << index.y << ") ";
                }
                std::cout << std::endl;
                break;
            default:
                movement_old_index = cv::Point();
                movement_new_index = cv::Point();
                break;
        }

        if (movement != Movement::Wrong) {
            recorder.record(player, movement_old_index, movement_new_index, defeated_indexes);
        }

        std::cout << std::endl;
    }

    void set_square_white_index(const std::vector<cv::Point>& new_index) {
        square_index_with_white_checkers_old = square_index_with_white_checkers_new;
        square_index_with_white_checkers_new = new_index;
    }

    void set_square_black_index(const std::vector<cv::Point>& new_index) {
        square_index_with_black_checkers_old = square_index_with_black_checkers_new;
        square_index_with_black_checkers_new = new_index;
    }

    void visualize_changes() {
        cv::circle(bird_eye_frame, movement_new_index, radius, cv::Scalar(0, 0, 255), 3);
    }
};

bool is_contour_circle(const std::vector<cv::Point>& contour, double epsilon_factor = 0.02, double circularity_threshold = 0.4, int min_area = 300) {
    double epsilon = epsilon_factor * cv::arcLength(contour, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, epsilon, true);
    if (approx.size() >= 6) {
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        double circularity = 4 * M_PI * (area / (perimeter * perimeter));
        return circularity > circularity_threshold && area > min_area;
    }
    return false;
}

bool is_hand_inside(const cv::Mat& current_frame, int threshold = 60, int sensitivity = 6) {
    static bool hand_on_screen = false;
    static cv::Mat old_frame;
    cv::Mat diff;
    cv::absdiff(old_frame, current_frame, diff);
    cv::threshold(diff, diff, threshold, 255, cv::THRESH_BINARY);
    int different_pixels_number = cv::sum(diff)[0];
    double result = diff.size().area() / different_pixels_number;
    if (result <= sensitivity && !hand_on_screen) {
        hand_on_screen = true;
    }
    else if (result >= sensitivity && hand_on_screen) {
        hand_on_screen = false;
    }
    return hand_on_screen;
}

void set_old_frame_for_hand_detection(const cv::Mat& current_frame, cv::Mat& old_frame) {
    old_frame = current_frame.clone();
}

cv::Point get_square_position(const cv::Point& square_index, int square_x_scaler, int square_y_scaler, int square_x_bias, int square_y_bias) {
    int x_position = square_index.x * square_x_scaler + square_x_bias;
    int y_position = square_index.y * square_y_scaler + square_y_bias;
    return cv::Point(x_position, y_position);
}

Player get_player(Movement movement) {
    switch (movement) {
        case Movement::White:
        case Movement::WhiteWithBlack:
            return Player::White;
        case Movement::Black:
        case Movement::BlackWithWhite:
            return Player::Black;
        default:
            return Player::White;
    }
}

std::vector<cv::Point> get_contours_centers(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<cv::Point> contour_centers;
    for (const auto& contour : contours) {
        cv::Moments M = cv::moments(contour);
        if (M.m00 != 0) {
            int cX = static_cast<int>(M.m10 / M.m00);
            int cY = static_cast<int>(M.m01 / M.m00);
            contour_centers.push_back(cv::Point(cX, cY));
        }
    }
    return contour_centers;
}

int get_contour_radius(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    int radius = static_cast<int>((2 * area) / perimeter);
    return radius;
}

std::vector<cv::Point> combine_close_contours(const std::vector<cv::Point>& centers, int threshold_distance = 25) {
    std::vector<cv::Point> result;
    for (size_t i = 0; i < centers.size(); ++i) {
        bool add_flag = true;
        double distance = 0;
        for (size_t j = i + 1; j < centers.size(); ++j) {
            distance = std::sqrt(std::pow(centers[i].x - centers[j].x, 2) + std::pow(centers[i].y - centers[j].y, 2));
            if (distance < threshold_distance) {
                add_flag = false;
                break;
            }
        }
        if (add_flag) {
            result.push_back(centers[i]);
        }
    }
    return result;
}

bool is_checker_white(const cv::Mat& frame_gray, const cv::Point& point) {
    return frame_gray.at<uchar>(point.y, point.x) > 180;
}

cv::Point get_square_index_from_contour(const std::vector<cv::Point>& contour, int square_width, int square_height) {
    cv::Point contour_center = get_contours_centers({ contour })[0];
    return cv::Point(contour_center.x / square_width, contour_center.y / square_height);
}

cv::Point get_square_index_from_center(const cv::Point& center, int x_scaler, int y_scaler) {
    return cv::Point(center.x / x_scaler, center.y / y_scaler);
}

class IndexDetector {
private:
    std::vector<std::vector<cv::Point>> white_lastn_square_indexes;
    std::vector<std::vector<cv::Point>> black_lastn_square_indexes;
    std::vector<cv::Point> white_result_index;
    std::vector<cv::Point> black_result_index;
    int n;
    int index;

    void set_index(const std::vector<std::vector<cv::Point>>& contours) {
        std::vector<cv::Point> contour_centers = get_contours_centers(contours);
        std::vector<cv::Point> contour_centers_combined = combine_close_contours(contour_centers);
        std::vector<cv::Point> white_checker_index;
        std::vector<cv::Point> black_checker_index;
        for (const auto& cent : contour_centers_combined) {
            cv::Point checker_index = get_square_index_from_center(cent, square_x_scaler, square_y_scaler);
            bool is_white = is_checker_white(bird_eye_frame_gray, cent);
            if (is_white) {
                white_checker_index.push_back(checker_index);
            }
            else {
                black_checker_index.push_back(checker_index);
            }
        }
        if (white_lastn_square_indexes.size() != n) {
            white_lastn_square_indexes.push_back(white_checker_index);
            black_lastn_square_indexes.push_back(black_checker_index);
        }
        else {
            white_lastn_square_indexes[index] = white_checker_index;
            black_lastn_square_indexes[index] = black_checker_index;
            index = (index + 1) % n;
        }
    }

    void combine_indexes() {
        std::set<cv::Point> combined_white_index;
        for (const auto& x : white_lastn_square_indexes) {
            combined_white_index.insert(x.begin(), x.end());
        }
        std::set<cv::Point> combined_black_index;
        for (const auto& x : black_lastn_square_indexes) {
            combined_black_index.insert(x.begin(), x.end());
        }
        white_result_index.assign(combined_white_index.begin(), combined_white_index.end());
        black_result_index.assign(combined_black_index.begin(), combined_black_index.end());
    }

public:
    IndexDetector(int n) : n(n), index(0) {}

    std::pair<std::vector<cv::Point>, std::vector<cv::Point>> get_indexes(const std::vector<std::vector<cv::Point>>& contours) {
        set_index(contours);
        combine_indexes();
        return { white_result_index, black_result_index };
    }

    void visualize(int radius) {
        for (const auto& cent : white_result_index) {
            cv::putText(bird_eye_frame, "White", cv::Point(cent.x - 10, cent.y - 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            cv::circle(bird_eye_frame, cent, radius, cv::Scalar(0, 0, 255), 3);
        }
        for (const auto& cent : black_result_index) {
            cv::putText(bird_eye_frame, "Black", cv::Point(cent.x - 10, cent.y - 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            cv::circle(bird_eye_frame, cent, radius, cv::Scalar(0, 0, 255), 3);
        }
    }
};



std::vector<cv::Point2f> findCorners(const cv::Mat& frame) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat binaryinv;
    cv::threshold(gray, binaryinv, 27, 255, cv::THRESH_BINARY_INV);

    cv::Mat blurred;
    cv::GaussianBlur(binaryinv, blurred, cv::Size(5, 5), 0);

    cv::Mat edges;
    cv::Canny(blurred, edges, 50, 150);

    cv::Mat dilated_edges;
    cv::dilate(edges, dilated_edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 2);
    cv::Mat eroded_edges;
    cv::erode(dilated_edges, eroded_edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(eroded_edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Point top_left(frame.cols, frame.rows);
    cv::Point top_right(0, frame.rows);
    cv::Point bottom_left(frame.cols, 0);
    cv::Point bottom_right(0, 0);

    for (const auto& contour : contours) {
        for (const auto& point : contour) {
            if (point.x + point.y < top_left.x + top_left.y) {
                top_left = point;
            }
            if (point.x - point.y > top_right.x - top_right.y) {
                top_right = point;
            }
            if (point.x - point.y < bottom_left.x - bottom_left.y) {
                bottom_left = point;
            }
            if (point.x + point.y > bottom_right.x + bottom_right.y) {
                bottom_right = point;
            }
        }
    }

    std::vector<cv::Point2f> corners = { top_left, bottom_left, bottom_right, top_right };
    return corners;
}

int main() {
    cv::VideoCapture video_capture("/home/eray/Desktop/Chessboard-Movement-Detection-main/source/chess.mp4");

    cv::Mat frame;
    video_capture.read(frame);

    

    Checkers manager;
    IndexDetector index_detector(4);

    int b_k = 7;
    int threshold1 = 60;
    int threshold2 = 120;
    cv::Mat kernel_morp = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    std::vector<cv::Point2f> points = { { 113, 14 }, { 51, 248 }, { 437, 246 }, { 357, 17 } };
    std::vector<cv::Point2f> new_points = { { 0, 0 }, { 0, height }, { width, height }, { width, 0 } };

    cv::Mat transformation_matrix = cv::getPerspectiveTransform(findCorners(frame), new_points);
    int frame_count = 1;
    cv::Mat old_frame;
    bool hand_on_screen = false;
    bool hand_on_screen_previous = false;

    while (video_capture.read(frame)) {
        cv::Mat full_view_frame_gray;
        cv::cvtColor(frame, full_view_frame_gray, cv::COLOR_BGR2GRAY);
        cv::warpPerspective(frame, bird_eye_frame, transformation_matrix, cv::Size(width, height));

        hand_on_screen_previous = hand_on_screen;
        hand_on_screen = is_hand_inside(full_view_frame_gray);
        if (hand_on_screen) {
            frame_count = 0;
        }
        else if (frame_count == 0 && !hand_on_screen && hand_on_screen_previous) {
            frame_count += 1;
        }

        if (frame_count != 0 && !hand_on_screen) {
            cv::cvtColor(bird_eye_frame, bird_eye_frame_gray, cv::COLOR_BGR2GRAY);

            cv::Mat bird_eye_frame_gray_gamma_corrected;
            cv::pow(bird_eye_frame_gray / 255.0, 0.7, bird_eye_frame_gray_gamma_corrected);
            bird_eye_frame_gray_gamma_corrected *= 255;
            bird_eye_frame_gray_gamma_corrected.convertTo(bird_eye_frame_gray_gamma_corrected, CV_8U);

            cv::Mat bird_eye_frame_blur;
            cv::GaussianBlur(bird_eye_frame_gray_gamma_corrected, bird_eye_frame_blur, cv::Size(b_k, b_k), 0);

            cv::Mat edges;
            cv::Canny(bird_eye_frame_blur, edges, threshold1, threshold2);

            cv::Mat dilated_edges;
            cv::dilate(edges, dilated_edges, kernel_morp, cv::Point(-1, -1), 2);
            cv::Mat eroded_edges;
            cv::erode(dilated_edges, eroded_edges, kernel_morp, cv::Point(-1, -1), 1);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(eroded_edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
            std::vector<std::vector<cv::Point>> circle_contours;
            for (const auto& contour : contours) {
                if (is_contour_circle(contour)) {
                    circle_contours.push_back(contour);
                }
            }
            std::pair<std::vector<cv::Point>, std::vector<cv::Point>> indexes = index_detector.get_indexes(circle_contours);
            radius = get_contour_radius(circle_contours[0]);
            frame_count += 1;
        }

        if (frame_count == index_detector.n + 1 && !hand_on_screen) {
            frame_count = 0;
            index_detector.get_indexes();
            manager.set_square_white_index(index_detector.white_result_index);
            manager.set_square_black_index(index_detector.black_result_index);
            manager.find_different_white_index();
            set_old_frame_for_hand_detection(full_view_frame_gray, old_frame);
        }

        cv::Mat diff;
        cv::absdiff(old_frame, full_view_frame_gray, diff);
        cv::threshold(diff, diff, 60, 255, cv::THRESH_BINARY);
        cv::imshow("Frame Diff", diff);

        manager.visualize_changes();

        cv::imshow("Frame", frame);
        cv::imshow("Chessboard Bird Eye", bird_eye_frame);
        int key = cv::waitKey(200);
        if (key == 27) {
            recorder.show();
            break;
        }
    }

    return 0;
}


