# special_problem

## Setup
cmake .
make

Image should be located in "images/frame00000_raw.png"

## Opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.7.0 opencv
mv opencv_contrib-4.7.0 opencv_contrib

# Create build directory and switch into it
mkdir opencv_install
cd opencv && mkdir -p build && cd build
# Configure

cmake -D WITH_OPENCL=OFF \
-D WITH_OPENGL=OFF \
-D WITH_OPENCL=OFF \
-D WITH_OPENCL_SVM=OFF \
-D WITH_PROTOBUF=OFF \
-D WITH_TBB=ON \
-D WITH_VULKAN=OFF \
-D WITH_QT=OFF \
-D BUILD_WITH_DEBUG_INFO=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_EXAMPLES=OFF \
-D BUILD_PROTOBUF=OFF \
-D PROTOBUF_UPDATE_FILES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_opencv_python2=OFF \
-D BUILD_opencv_python3=OFF \
-D WITH_PYTHON=OFF \
-D BUILD_opencv_java=OFF=OFF \
-D WITH_JAVA=OFF \
-D WITH_GTK=OFF \
-D WITH_TEST=OFF \
-D CMAKE_INSTALL_PREFIX=~/local_storage/leon/opencv_install \
-D CPU_BASELINE_DISABLE=SSE3 \
-D CPU_BASELINE_REQUIRE=SSE2 \
-D OPENCV_EXTRA_MODULES_PATH='~/local_storage/leon/opencv_contrib/modules/cudaarithm;~/local_storage/leon/opencv_contrib/modules/cudabgsegm;~/local_storage/leon/opencv_contrib/modules/cudacodec;~/local_storage/leon/opencv_contrib/modules/cudafeatures2d;~/local_storage/leon/opencv_contrib/modules/cudafilters;~/local_storage/leon/opencv_contrib/modules/cudaimgproc;~/local_storage/leon/opencv_contrib/modules/cudalegacy;~/local_storage/leon/opencv_contrib/modules/cudaobjdetect;~/local_storage/leon/opencv_contrib/modules/cudaoptflow;~/local_storage/leon/opencv_contrib/modules/cudastereo;~/local_storage/leon/opencv_contrib/modules/cudawarping;~/local_storage/leon/opencv_contrib/modules/cudev' \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_GENERATE_SETUPVARS=OFF \
-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
-D WITH_CUDA=ON \
-D WITH_CUDNN=OFF \
-D CUDA_ARCH_BIN='86-real' \
-D CUDA_ARCH_PTX='86-real' \
..