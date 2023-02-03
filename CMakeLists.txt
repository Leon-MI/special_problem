# set (CMAKE_CXX_FLAGS "-O3 -march=native -Wall -Wextra -Wpedantic -Werror")
project(Pipeline)

# compiler options, set c++11 (requires cmake 3.2 or higher)
cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 11) 

###############################################################################
## find libraries #############################################################
###############################################################################

# FIND LIBRARIES
find_package(OpenCV REQUIRED PATHS "~/local_storage/leon/opencv_install")
find_package(CUDA REQUIRED)

# INCLUDE HEADERS
include_directories(${OpenCV_INCLUDE_DIRS})

# COMPILER FLAGS AND DEFINITIONS
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_86,code=sm_86)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)

# SPECIFY TARGETS
cuda_add_executable(Pipeline pipeline.cu)

target_link_libraries(Pipeline ${OpenCV_LIBS})
