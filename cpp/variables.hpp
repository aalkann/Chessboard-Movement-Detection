#pragma once

#include <cmath>

int width = 500;
int height = 500;

int square_x_scaler = width / 8;
int square_x_bias = square_x_scaler / 2;
int square_y_scaler = height / 8;
int square_y_bias = square_y_scaler / 2;
int radius = width / 20;
double max_area = M_PI * pow(radius, 2) + (M_PI * pow(radius, 2)) / 2;



