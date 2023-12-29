#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
int max_area = 1000; // Define max_area if not declared globally
bool hand_on_screen = false;
Mat old_frame;

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

class Checkers {
public:
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
            for (const auto& new_point : new_indexes) {
                if (old == new_point) {
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

    Movement what_happened(int nnw, int now, int nnb, int nob) {
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

    void find_different_white_index() {
        // Eskide olup da yenide olmayan 
        auto old_square_white_indexes = get_indexes_changes(square_index_with_white_checkers_old, square_index_with_white_checkers_new);
        auto old_square_black_indexes = get_indexes_changes(square_index_with_black_checkers_old, square_index_with_black_checkers_new);

        // Yenide olup da eskide olmayan 
        auto new_square_white_indexes = get_indexes_changes(square_index_with_white_checkers_new, square_index_with_white_checkers_old);
        auto new_square_black_indexes = get_indexes_changes(square_index_with_black_checkers_new, square_index_with_black_checkers_old);

        int number_new_white = new_square_white_indexes.size();
        int number_old_white = old_square_white_indexes.size();
        int number_new_black = new_square_black_indexes.size();
        int number_old_black = old_square_black_indexes.size();

        movement = what_happened(number_new_white, number_old_white, number_new_black, number_old_black);
        std::cout << "Movement: " << static_cast<int>(movement) << std::endl;

        Player player = get_player(movement);
        defeated_indexes.clear();

        switch (movement) {
            case Movement::White:
                movement_old_index = get_closest_index(old_square_white_indexes, new_square_white_indexes[0]);
                movement_new_index = new_square_white_indexes[0];
                std::cout << "White New: " << new_square_white_indexes[0] << std::endl;
                std::cout << "White Old: " << old_square_white_indexes[0] << std::endl;
                break;
            case Movement::WhiteWithBlack:
                movement_old_index = get_closest_index(old_square_white_indexes, new_square_white_indexes[0]);
                movement_new_index = new_square_white_indexes[0];
                defeated_indexes = old_square_black_indexes;
                std::cout << "White New: " << new_square_white_indexes[0] << std::endl;
                std::cout << "White Old: " << old_square_white_indexes[0] << std::endl;
                std::cout << "Defeated Blacks: ";
                for (const auto& index : old_square_black_indexes) {
                    std::cout << index << " ";
                }
                std::cout << std::endl;
                break;
            case Movement::Black:
                movement_old_index = get_closest_index(old_square_black_indexes, new_square_black_indexes[0]);
                movement_new_index = new_square_black_indexes[0];
                std::cout << "Black New: " << new_square_black_indexes[0] << std::endl;
                std::cout << "Black Old: " << old_square_black_indexes[0] << std::endl;
                break;
            case Movement::BlackWithWhite:
                movement_old_index = get_closest_index(old_square_black_indexes, new_square_black_indexes[0]);
                movement_new_index = new_square_black_indexes[0];
                defeated_indexes = old_square_white_indexes;
                std::cout << "Black New: " << new_square_black_indexes[0] << std::endl;
                std::cout << "Black Old: " << old_square_black_indexes[0] << std::endl;
                std::cout << "Defeated Whites: ";
                for (const auto& index : old_square_white_indexes) {
                    std::cout << index << " ";
                }
                std::cout << std::endl;
                break;
            default:
                movement_old_index = cv::Point(-1, -1);
                movement_new_index = cv::Point(-1, -1);
        }

        if (movement != Movement::Wrong) {
            // recorder.record(player, movement_old_index, movement_new_index, defeated_indexes);
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

    cv::Point get_closest_index(const std::vector<cv::Point>& indexes, const cv::Point& point) {
        if (indexes.empty()) {
            return cv::Point(-1, -1);
        }

        cv::Point result = indexes[0];
        double distance_min = std::abs(indexes[0].x - point.x) + std::abs(indexes[0].y - point.y);

        for (const auto& index : indexes) {
            double distance = std::abs(index.x - point.x) + std::abs(index.y - point.y);
            if (distance < distance_min) {
                result = index;
                distance_min = distance;
            }
        }
        return result;
    }

    void visualize_changes() {
        if (movement_new_index == cv::Point(-1, -1) || movement_old_index == cv::Point(-1, -1)) {
            return;
        }

        std::vector<int> mover_old_position = get_square_position(movement_old_index);
        std::vector<int> mover_new_position = get_square_position(movement_new_index);

        cv::putText(bird_eye_frame, "Old " + std::to_string(movement_old_index), cv::Point(mover_old_position[0] - 10, mover_old_position[1] - 20), 1, 1, cv::Scalar(255, 0, 0), 2);
        cv::circle(bird_eye_frame, cv::Point(mover_old_position[0], mover_old_position[1]), radius, cv::Scalar(255, 0, 0), 3);

        cv::putText(bird_eye_frame, "New " + std::to_string(movement_new_index), cv::Point(mover_new_position[0] - 10, mover_new_position[1] - 20), 1, 1, cv::Scalar(0, 255, 0), 2);
        cv::circle(bird_eye_frame, cv::Point(mover_new_position[0], mover_new_position[1]), radius, cv::Scalar(0, 255, 0), 3);

        cv::arrowedLine(bird_eye_frame, cv::Point(mover_old_position[0], mover_old_position[1]), cv::Point(mover_new_position[0], mover_new_position[1]), cv::Scalar(255, 255, 0), 2);

        if (!defeated_indexes.empty()) {
            for (size_t i = 0; i < defeated_indexes.size(); ++i) {
                std::vector<int> position = get_square_position(defeated_indexes[i]);
                cv::putText(bird_eye_frame, "Defeated " + std::to_string(defeated_indexes[i]), cv::Point(position[0] - 10, position[1] - 20), 1, 1, cv::Scalar(0, 0, 255), 2);
                cv::circle(bird_eye_frame, cv::Point(position[0], position[1]), radius, cv::Scalar(0, 0, 255), 3);
            }
        }
    }
    bool is_contour_circle(const vector<Point>& contour, double epsilon_factor = 0.02, double circularity_threshold = 0.4, int min_area = 300) {
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

    void is_hand_inside(const Mat& current_frame, int threshold = 60, int sensitivity = 6) {
        Mat diff;
        absdiff(old_frame, current_frame, diff);
        threshold(diff, diff, threshold, 255, THRESH_BINARY);

        int different_pixels_number = countNonZero(diff);
        double result = static_cast<double>(diff.total()) / different_pixels_number;

        if (result <= sensitivity && !hand_on_screen) {
            hand_on_screen = true;
            // cout << "Hand in" << endl;
        } else if (result >= sensitivity && hand_on_screen) {
            hand_on_screen = false;
            // cout << "Hand out" << endl;
        }
    }

    void set_old_frame_for_hand_detection(const Mat& current_frame) {
        old_frame = current_frame.clone();
    }

    vector<int> get_square_position(const vector<int>& square_index, int square_x_scaler, int square_y_scaler, int square_x_bias, int square_y_bias) {
        int x_position = square_index[0] * square_x_scaler + square_x_bias;
        int y_position = square_index[1] * square_y_scaler + square_y_bias;
        // cout << "(" << x_position << ", " << y_position << ")" << endl;
        // Do something with the position if needed
    }

    enum Movement { White, WhiteWithBlack, Black, BlackWithWhite };
    enum Player { White, Black };

    Player get_player(Movement movement) {
        switch (movement) {
            case Movement::White:
            case Movement::WhiteWithBlack:
                return Player::White;
            case Movement::Black:
            case Movement::BlackWithWhite:
                return Player::Black;
        }
    }

    vector<Point> get_contours_centers(const vector<vector<Point>>& contours) {
        vector<Point> contour_centers;
        for (const auto& contour : contours) {
            Moments M = moments(contour);
            if (M.m00 != 0) {
                int cX = static_cast<int>(M.m10 / M.m00);
                int cY = static_cast<int>(M.m01 / M.m00);
                contour_centers.push_back(Point(cX, cY));
            }
        }
        return contour_centers;
    }

    int get_contour_radius(const vector<Point>& contour) {
        double area = contourArea(contour);
        double perimeter = arcLength(contour, true);
        return static_cast<int>((2 * area) / perimeter);
    }

    vector<Point> combine_close_contours(const vector<Point>& centers, int threshold_distance = 25) {
        vector<Point> result;
        for (size_t i = 0; i < centers.size(); ++i) {
            bool add_flag = true;
            double distance = 0;
            for (size_t j = i + 1; j < centers.size(); ++j) {
                distance = norm(centers[i] - centers[j]);
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

    bool is_checker_white(const Mat& frame_gray, const Point& point) {
        return frame_gray.at<uchar>(point.y, point.x) > 180;
    }

    Point get_square_index_from_contour(const vector<Point>& contour, int square_width, int square_height) {
        Point contour_center = get_contours_centers({contour})[0];
        return Point(contour_center.x / square_width, contour_center.y / square_height);
    }

    Point get_square_index_from_center(const Point& center, int x_scaler, int y_scaler) {
        return Point(center.x / x_scaler, center.y / y_scaler);
    }
};

class IndexDetector {
    public:
        static vector<vector<Point>> white_lastn_square_indexes;
        static vector<vector<Point>> white_result_index;
        static vector<vector<Point>> black_lastn_square_indexes;
        static vector<vector<Point>> black_result_index;
        static int n;
        static int index;

        void set_index(const vector<vector<Point>>& contours) {
            vector<Point> contour_centers = get_contours_centers(contours);
            vector<Point> contour_centers_combined = combine_close_contours(contour_centers);

            vector<Point> white_checker_index;
            vector<Point> black_checker_index;

            for (const auto& cent : contour_centers_combined) {
                Point checker_index = get_square_index_from_center(cent, square_x_scaler, square_y_scaler);
                bool is_white = is_checker_white(bird_eye_frame_gray, cent);

                if (is_white)
                    white_checker_index.push_back(checker_index);
                else
                    black_checker_index.push_back(checker_index);
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

        pair<vector<Point>, vector<Point>> combine_indexes() {
            set<Point> combined_white_index;
            for (const auto& x : white_lastn_square_indexes) {
                combined_white_index.insert(x.begin(), x.end());
            }

            set<Point> combined_black_index;
            for (const auto& x : black_lastn_square_indexes) {
                combined_black_index.insert(x.begin(), x.end());
            }

            return {vector<Point>(combined_white_index.begin(), combined_white_index.end()),
                    vector<Point>(combined_black_index.begin(), combined_black_index.end())};
        }

        pair<vector<Point>, vector<Point>> get_indexes() {
            tie(white_result_index, black_result_index) = combine_indexes();
            return {white_result_index, black_result_index};
        }

        void visualize(int radius) {
            visualize_color("White", white_result_index, Scalar(0, 255, 0), radius);
            visualize_color("Black", black_result_index, Scalar(0, 0, 255), radius);
        }

    private:
        void visualize_color(const String& color, const vector<vector<Point>>& indexes, const Scalar& text_color, int radius) {
            for (const auto& cent : indexes) {
                Point position = get_square_position(cent[0]);
                putText(bird_eye_frame, color, Point(position.x - 10, position.y - 20), 1, 1, text_color, 2);
                circle(bird_eye_frame, position, radius, Scalar(0, 255, 0), 3);
            }
        }
};

vector<vector<Point>> IndexDetector::white_lastn_square_indexes;
vector<vector<Point>> IndexDetector::white_result_index;
vector<vector<Point>> IndexDetector::black_lastn_square_indexes;
vector<vector<Point>> IndexDetector::black_result_index;
int IndexDetector::n = 4;
int IndexDetector::index = 0;
class GameMovementRecorder {
public:
    std::vector<std::vector<int>> records;

    void record(Player player, Player old_player, Player new_player, std::vector<int> defeateds) {
        records.push_back({ static_cast<int>(player), static_cast<int>(old_player), static_cast<int>(new_player) });
        records.back().insert(records.back().end(), defeateds.begin(), defeateds.end());
    }

    void show() {
        for (const auto& movement : records) {
            for (int value : movement) {
                std::cout << value << ' ';
            }
            std::cout << std::endl;
        }
    }
};
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class Checkers {
    // Add Checkers class implementation here
};

class IndexDetector {
    // Add IndexDetector class implementation here
};

class GameMovementRecorder {
    // Add GameMovementRecorder class implementation here
};

// Function to find corners
vector<Point2f> findCorners(Mat& frame) {
    // Convert the frame to grayscale
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // Apply binary inverse thresholding
    Mat binaryinv;
    threshold(gray, binaryinv, 27, 255, THRESH_BINARY_INV);

    // Apply GaussianBlur to reduce noise
    Mat blurred;
    GaussianBlur(binaryinv, blurred, Size(5, 5), 0);

    // Use Canny edge detector to find edges
    Mat edges;
    Canny(blurred, edges, 50, 150);

    // Find contours
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Draw contours on the original frame
    Mat frame_contours = frame.clone();
    drawContours(frame_contours, contours, -1, Scalar(0, 255, 0), 2);

    // Display the frame with contours
    imshow("Contours", frame_contours);
    waitKey(0);

    // Initialize extreme points
    Point2f top_left(frame.cols, frame.rows);
    Point2f top_right(0, frame.rows);
    Point2f bottom_left(frame.cols, 0);
    Point2f bottom_right(0, 0);

    // Find extreme points
    for (const auto& contour : contours) {
        for (const auto& point : contour) {
            if (point.x + point.y < top_left.x + top_left.y)
                top_left = point;
            if (point.x - point.y > top_right.x - top_right.y)
                top_right = point;
            if (point.x - point.y < bottom_left.x - bottom_left.y)
                bottom_left = point;
            if (point.x + point.y > bottom_right.x + bottom_right.y)
                bottom_right = point;
        }
    }

    // Define the corners as a vector of Point2f
    vector<Point2f> corners = {top_left, bottom_left, bottom_right, top_right};

    return corners;
}

int main() {
    // Set available camera
    VideoCapture video_capture("/home/eray/Desktop/Chessboard-Movement-Detection-main/source/chess.mp4");

    // Read the first frame
    Mat frame;
    bool success = video_capture.read(frame);

    // Set the width and height for bird's-eye view
    int width = 500, height = 500;

    // Parameters for checkers positioning and locationing
    int square_x_scaler = width / 8;
    int square_x_bias = square_x_scaler / 2;
    int square_y_scaler = height / 8;
    int square_y_bias = square_y_scaler / 2;

    // Checker Manager
    Checkers manager;
    IndexDetector index_detector;
    GameMovementRecorder recorder;

    // Parameters for preprocessing
    int b_k = 7;  // Gaussian Blur Kernel Size
    int threshold1 = 60;  // Threshold value 1 for Canny Edge Detector
    int threshold2 = 120; // Threshold value 2 for Canny Edge Detector
    Mat kernel_morp = getStructuringElement(MORPH_RECT, Size(3, 3)); // Morphological operation kernel

    // Detect Chessboard Coordinates
    // For now, use static points
    vector<Point2f> points = {{113, 14}, {51, 248}, {437, 246}, {357, 17}};
    vector<Point2f> new_points = {{0, 0}, {0, height}, {width, height}, {width, 0}};

    // Calculate the transformation matrix
    Mat transformation_matrix = getPerspectiveTransform(findCorners(frame), new_points);

    int frame_count = 1;
    int radius = width / 20;
    double max_area = (CV_PI * radius * radius) + (CV_PI * radius * radius) / 2.0;
    Mat old_frame;
    cvtColor(frame, old_frame, COLOR_BGR2GRAY);
    bool hand_on_screen = false;
    bool hand_on_screen_previous = false;
    while (success) {
        success = video_capture.read(frame);
        frame_original = frame.clone();
        cv::cvtColor(frame, full_view_frame_gray, cv::COLOR_BGR2GRAY);

        // Using transformation matrix convert frame into "Bird'Eye View"
        cv::warpPerspective(frame, bird_eye_frame, transformation_matrix, cv::Size(width, height));

        // Check if hand is in
        hand_on_screen_previous = hand_on_screen;
        is_hand_inside(full_view_frame_gray);

        if (hand_on_screen == true) {
            frame_count = 0;
        } else if (frame_count == 0 && hand_on_screen == false && hand_on_screen_previous == true) {
            frame_count += 1;
        }

        // Hand went out
        if (frame_count != 0 && hand_on_screen == false) {
            // Make frame gray scale
            cv::cvtColor(bird_eye_frame, bird_eye_frame_gray, cv::COLOR_BGR2GRAY);
            // Apply gamma correction to reduce the shadow effect
            bird_eye_frame_gray_gamma_corrected = pow(bird_eye_frame_gray / 255.0, 0.7) * 255.0;
            bird_eye_frame_gray_gamma_corrected.convertTo(bird_eye_frame_gray_gamma_corrected, CV_8U);
            // Blur the frame for making Canny algorithm more precise
            cv::GaussianBlur(bird_eye_frame_gray_gamma_corrected, bird_eye_frame_blur, cv::Size(b_k, b_k), 0);
            // Detect Edges
            cv::Canny(bird_eye_frame_blur, edges, threshold1, threshold2);
            // Apply Morphological operations to fix broken edges
            cv::dilate(edges, dilated_edges, kernel_morp, cv::Point(-1, -1), 2);
            cv::erode(dilated_edges, eroded_edges, kernel_morp, cv::Point(-1, -1), 1);
            // Find Circular Contours
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(eroded_edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
            std::vector<std::vector<cv::Point>> circle_contours;
            for (const auto& contour : contours) {
                if (is_contour_circle(contour)) {
                    circle_contours.push_back(contour);
                }
            }

            index_detector.set_index(circle_contours);
            radius = get_contour_radius(circle_contours[0]);

            frame_count += 1;
        }

        // Display changes
        if (frame_count == index_detector.n + 1 && hand_on_screen == false) {
            frame_count = 0;
            auto [white_result_index, black_result_index] = index_detector.get_indexes();
            manager.set_square_white_index(white_result_index);
            manager.set_square_black_index(black_result_index);
            manager.find_different_white_index();
            set_old_frame_for_hand_detection(full_view_frame_gray);
        }

        // For hand detection visualization
        cv::absdiff(old_frame, full_view_frame_gray, diff);
        cv::threshold(diff, diff, 60, 255, cv::THRESH_BINARY);
        cv::imshow("Frame Diff", diff);

        // Show all detected checkers
        // index_detector.visualize(radius);
        // Show changes
        manager.visualize_changes();

        // Show the result
        cv::imshow("Frame ", frame);
        cv::imshow("Chessboard Bird Eye", bird_eye_frame);

        int key = cv::waitKey(200);
        if (key == 27) {
            recorder.show();
            break;
        }
    }
    return 0;
}
