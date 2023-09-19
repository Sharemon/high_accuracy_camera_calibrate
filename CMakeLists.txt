cmake_minimum_required(VERSION 3.0.0)
project(high_accuracy_corner_detector VERSION 0.1.0)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

file(GLOB_RECURSE cpp_srcs src/*.cpp)
file(GLOB_RECURSE cpp_apps app/*.cpp)

add_executable(high_accuracy_corner_detector ${cpp_srcs} ${cpp_apps})
target_link_libraries(high_accuracy_corner_detector ${OpenCV_LIBS})
