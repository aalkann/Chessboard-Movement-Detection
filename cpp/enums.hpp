#pragma once
enum class Player {
    White = 1, // White move without black defeated
    Black = 2,  // Black move without white defeated
    Null = 3
};

enum class Movement {
    White = 1,          // White move without black defeated
    WhiteWithBlack,     // White move with black defeated
    Black,              // Black move without white defeated
    BlackWithWhite,     // Black move with white defeated
    Wrong               // Wrong movement
};
