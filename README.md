# Chessboard Movement Detection

### Purpose

Creating a system that track the game and show the movements and records at the end of the match.

### Techniques

Only image processing techniques were used to create the system.

There isn't any deep learning model or machine learning algorithm.

We can simply list the processing steps as below
- Convert the board image into Bird'eye view
- Check if hand is on the board
- Find checker contours
- Find the position of each checker on the board
- Seperate white and black checkers
- Compare previously saved state with the current state and find the differences
- Save the changes (Player, Checkter Old Position, Checker New Position, Defeated Rival Checkers)


### Visual Results

Original Video Frame:

![alt text](/visuals/image.png)

Bird'eye View Frame:

![alt text](/visuals/image-1.png)

When the player makes a move

![alt text](/visuals/image-2.png)

![alt text](/visuals/image-3.png)

When player defeat a rival checker

![alt text](/visuals/image-4.png)