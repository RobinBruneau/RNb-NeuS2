cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
project(preprocess)
SET( CMAKE_CXX_FLAGS_RELEASE  "")
SET( CMAKE_CXX_FLAGS  "-O3 -DNDEBUG -std=c++20 -march=native -fpic -ftree-vectorize")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


# Add your source files
add_executable(gen_images_light_60_opti gen_images_light_60_opti.cpp)

# Link libraries
target_link_libraries(gen_images_light_60_opti ${OpenCV_LIBS})
