#pragma once

#include <opencv2/opencv.hpp>
#include "functions.hpp"
#include "enums.hpp"
using namespace cv;

class Checkers {
private:
    std::vector<Point> square_index_with_white_checkers_old;
    std::vector<Point> square_index_with_white_checkers_new;
    std::vector<Point> square_index_with_black_checkers_old;
    std::vector<Point> square_index_with_black_checkers_new;
    Movement movement;
    Point movement_old_index;
    Point movement_new_index;
    std::vector<Point> defeated_indexes;

public:
    std::vector<Point> get_indexes_changes(std::vector<Point> old_indexes, std::vector<Point> new_indexes) {
        std::vector<Point> result;
        for (auto old : old_indexes) {
            bool flag = false;
            for (auto new_inx : new_indexes) {
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

    tuple<Player,cv::Point,cv::Point,std::vector<cv::Point>> find_different_index() {
        std::vector<Point> old_sqaure_white_indexes = get_indexes_changes(square_index_with_white_checkers_old, square_index_with_white_checkers_new);
        std::vector<Point> old_sqaure_black_indexes = get_indexes_changes(square_index_with_black_checkers_old, square_index_with_black_checkers_new);

        std::vector<Point> new_sqaure_white_indexes = get_indexes_changes(square_index_with_white_checkers_new, square_index_with_white_checkers_old);
        std::vector<Point> new_sqaure_black_indexes = get_indexes_changes(square_index_with_black_checkers_new, square_index_with_black_checkers_old);

        int number_new_white = new_sqaure_white_indexes.size();
        int number_old_white = old_sqaure_white_indexes.size();
        int number_new_black = new_sqaure_black_indexes.size();
        int number_old_black = old_sqaure_black_indexes.size();

        movement = what_happaned(number_new_white, number_old_white, number_new_black, number_old_black);

        Player player = get_player(movement);
        defeated_indexes.clear();

        switch (movement) {
            case Movement::White:
                movement_old_index = get_closest_index(old_sqaure_white_indexes, new_sqaure_white_indexes[0]);
                movement_new_index = new_sqaure_white_indexes[0];
                break;
            case Movement::WhiteWithBlack:
                movement_old_index = get_closest_index(old_sqaure_white_indexes, new_sqaure_white_indexes[0]);
                movement_new_index = new_sqaure_white_indexes[0];
                defeated_indexes = old_sqaure_black_indexes;
                break;
            case Movement::Black:
                movement_old_index = get_closest_index(old_sqaure_black_indexes, new_sqaure_black_indexes[0]);
                movement_new_index = new_sqaure_black_indexes[0];
                break;
            case Movement::BlackWithWhite:
                movement_old_index = get_closest_index(old_sqaure_black_indexes, new_sqaure_black_indexes[0]);
                movement_new_index = new_sqaure_black_indexes[0];
                defeated_indexes = old_sqaure_white_indexes;
                break;
            default:
                movement_old_index = Point();
                movement_new_index = Point();
                break;
        }

        if (movement != Movement::Wrong) {
            return std::make_tuple(player, movement_old_index, movement_new_index, defeated_indexes);
        }
        return std::make_tuple(Player::Null,cv::Point(0,0),cv::Point(0,0),vector<cv::Point>());
    }

    Player get_player(Movement movement) {
        switch (movement) {
            case Movement::White:
            case Movement::WhiteWithBlack:
                return Player::White;
            case Movement::Black:
            case Movement::BlackWithWhite:
                return Player::Black;
        }
        return Player::Null;
    }

    void set_square_white_index(std::vector<Point> new_index) {
        square_index_with_white_checkers_old = square_index_with_white_checkers_new;
        square_index_with_white_checkers_new = new_index;
    }

    void set_square_black_index(std::vector<Point> new_index) {
        square_index_with_black_checkers_old = square_index_with_black_checkers_new;
        square_index_with_black_checkers_new = new_index;
    }

    Point get_closest_index(std::vector<Point> indexes, Point point) {
        if (indexes.empty()) {
            return Point();
        }
        Point result = indexes[0];
        int distance_min = abs(indexes[0].x - point.x) + abs(indexes[0].y - point.y);
        for (auto index : indexes) {
            int distance = abs(index.x - point.x) + abs(index.y - point.y);
            if (distance < distance_min) {
                result = index;
                distance_min = distance;
            }
        }
        return result;
    }

    void apply_changes(Mat bird_eye_frame, int radius) {
        if (movement_new_index == Point() || movement_old_index == Point()) {
            return;
        }
        Point mover_old_position = get_square_position(movement_old_index);
        Point mover_new_position = get_square_position(movement_new_index);

        putText(bird_eye_frame, "Old " + point_to_string(movement_old_index), Point(mover_old_position.x - 10, mover_old_position.y - 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        circle(bird_eye_frame, mover_old_position, radius, Scalar(255, 0, 0), 3);

        putText(bird_eye_frame, "New " + point_to_string(movement_new_index), Point(mover_new_position.x - 10, mover_new_position.y - 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        circle(bird_eye_frame, mover_new_position, radius, Scalar(0, 255, 0), 3);

        arrowedLine(bird_eye_frame, mover_old_position, mover_new_position, Scalar(255, 255, 0), 2);

        if (!defeated_indexes.empty()) {
            std::vector<Point> defeated_positions;
            for (auto cent : defeated_indexes) {
                defeated_positions.push_back(get_square_position(cent));
            }
            for (int i = 0; i < defeated_indexes.size(); i++) {
                putText(bird_eye_frame, "Defeated " + point_to_string(defeated_indexes[i]), Point(defeated_positions[i].x - 10, defeated_positions[i].y - 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
                circle(bird_eye_frame, defeated_positions[i], radius, Scalar(0, 0, 255), 3);
            }
        }
    }
};
