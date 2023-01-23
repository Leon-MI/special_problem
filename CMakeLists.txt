cmake_minimum_required(VERSION 3.15)
project( Pipeline )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Pipeline pipeline.cpp )
target_link_libraries( Pipeline ${OpenCV_LIBS} )

set (CMAKE_CXX_FLAGS "-O3 -march=native -Wall -Wextra -Wpedantic -Werror")
