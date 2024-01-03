#pragma once

#include <iostream>
#include <vector>
#include "enums.hpp"
#include "functions.hpp"

struct movementRecord{
    Player p;
    cv::Point old_player;
    cv::Point new_player;
    std::vector<cv::Point> defeats;
};
class GameMovementRecorder {
    std::vector<movementRecord> records;
public:
    void record(movementRecord rec) {
        records.push_back(rec);
    }
    void show() {
        for (auto movement : records) {
            std::cout << "Player: "<<to_string(movement.p)<<" Old location: " <<movement.old_player<<" New location: "<<movement.new_player<<" defeated pieces locations: ";
            for(auto x:movement.defeats){
                cout<<x<<" ";
            }
            std::cout << std::endl;
        }
    }
};

