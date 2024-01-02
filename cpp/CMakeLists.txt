cmake_minimum_required(VERSION 3.0)
project(checkerboard)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Add source files
set(SOURCE_FILES
    main.cpp
    checkers.hpp
    enums.hpp
    functions.hpp
    GameMovementRecorder.hpp
    IndexDetector.hpp
    preprocessingVariables.hpp
    variables.hpp
)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add executable
add_executable(CheckerboardDetection ${SOURCE_FILES})

# Set include directories
target_include_directories(CheckerboardDetection PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link against OpenCV libraries
target_link_libraries(CheckerboardDetection PRIVATE ${OpenCV_LIBS})

