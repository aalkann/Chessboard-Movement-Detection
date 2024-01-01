from enum import Enum

class Movement(Enum):
    White = 1 # White move without black defeated
    WhiteWithBlack = 2 # White move with black defeated
    Black = 3 # Black move without white defeated
    BlackWithWhite = 4 # Black move with white defeated
    Wrong = 5 # Wrong movement