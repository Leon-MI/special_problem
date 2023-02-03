#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <cuda_runtime.h>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace cv::cuda;

#define ROWS 2048
#define COLS 2448
#define ROWS2 1024
#define COLS2 1224

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void my_split(const cuda::PtrStepSzb dev_img, cuda::PtrStepSzb bggr0, cuda::PtrStepSzb bggr45, cuda::PtrStepSzb bggr90, cuda::PtrStepSzb bggr135) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= ROWS || j >= COLS) {
        return;
    }
    if (i%2 == 0) {
        if (j%2 == 0)
            bggr90(i/2, j/2) = dev_img(i, j);
        else
            bggr45(i/2, (j-1)/2) = dev_img(i, j);
    } else {
        if (j%2 == 0)
            bggr135((i-1)/2, j/2) = dev_img(i, j);
        else
            bggr0((i-1)/2, (j-1)/2) = dev_img(i, j);
    }
}

void demosaicing(const GpuMat& input, const GpuMat& output) {
    cuda::cvtColor(input, output, COLOR_BayerBG2BGR);
    CHECK_LAST_CUDA_ERROR();
}

void rgb2mono(const GpuMat& input, const GpuMat& output) {
    cuda::cvtColor(input, output, COLOR_BGR2GRAY);
    CHECK_LAST_CUDA_ERROR();
}

__global__ void compute_stokes(
    const cuda::PtrStepSzb mono0,
    const cuda::PtrStepSzb mono45,
    const cuda::PtrStepSzb mono90,
    const cuda::PtrStepSzb mono135,
    short3 * output,
    int oStep
    ) {
    // https://stackoverflow.com/questions/46389154/gpumat-accessing-2-channel-float-data-in-custom-kernel
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= ROWS2 || j >= COLS2)
        return;
    /* Compute linear index from 3D indices */
    const int tidOut = i * oStep + j;

    const unsigned char m0 = mono0(i, j);
    const unsigned char m90 = mono90(i, j);
    
    output[tidOut].x = m0 + m90;
    output[tidOut].y = m0 - m90;
    output[tidOut].z = mono45(i, j) - mono135(i, j);
}

__global__ void compute_dolp(const short3 * stokes, int iStep, cuda::PtrStepSz<double> output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= ROWS2 || j >= COLS2)
        return;
    /* Compute linear index from 3D indices */
    const int tidIn = i * iStep + j;
    
    const short3 s = stokes[tidIn];
    if (s.x == (short) 0)
        output(i, j) = 0;
    else
        output(i, j) = sqrt((double)((s.y*s.y) + (s.z*s.z))) / s.x;
}


__global__ void compute_aolp(const short3 * stokes, int iStep, cuda::PtrStepSz<double> output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= ROWS2 || j >= COLS2)
        return;
    /* Compute linear index from 3D indices */
    const int tidIn = i * iStep + j;
    
    const short3 s = stokes[tidIn];
    if (s.z == (short)0) {
        output(i, j) = 0.0;
    } else {
        const double sy= (double)s.y;
        const double sz = (double)s.z;
        const double angle = (double)1/2 * atan2(sy, sz);
        output(i, j) = angle + (double)CV_PI/2;
    }
}

__global__ void false_coloring(const cuda::PtrStepSz<double> aolp, cuda::PtrStepSz<double> dolp, unsigned char* output, int oStep) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= ROWS2 || j >= COLS2)
        return;
    /* Compute linear index from 3D indices */
    const int tidOut = i*oStep + 3*j;

    const double a = aolp(i, j);
    const double d = dolp(i, j) * 255;

    output[tidOut] = (unsigned char) (179 * fmod(a, CV_PI) / CV_PI);
    output[tidOut + 1] = (unsigned char) 255;
    output[tidOut + 2] = (unsigned char) min(max((double)0, d), (double)255);
}

void save_img(const char * name, const Mat & M) {
    std::string out_name = name;
    out_name = "images/" + out_name + ".png";
    std::vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    imwrite(out_name, M, compression_params);
}

void save_text(const char * name, const Mat & M) {
    std::string out_name = name;
    out_name = "images/" + out_name + ".txt";
    std::ofstream myfile (out_name);
    cv::Ptr<cv::Formatter> formatMat=Formatter::get(cv::Formatter::FMT_DEFAULT);
    formatMat->set64fPrecision(3);
    formatMat->set32fPrecision(3);

    if (myfile.is_open())
    {
        myfile << formatMat->format(M);
        myfile.close();
    }
    else std::cout << "Unable to open file";
}

int main()
{
    cuda::setDevice(0);
    // Read and upload img to gpu
    Mat img_raw = imread("images/frame00000_raw.png", IMREAD_GRAYSCALE);
    // Mat img_raw = (Mat_<unsigned char>(4,4) << 90, 45, 90, 45, 135, 1, 135, 1, 90, 45, 90, 45, 135, 1, 135, 1);
    std::cout << "Raw img rows: " << img_raw.rows << "cols: " << img_raw.cols << " type=" << img_raw.type() << std::endl;
    
    GpuMat dev_img;
    dev_img.upload(img_raw);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Split
    cudaEventRecord(start, 0);
    GpuMat dev_bggr0(ROWS2, COLS2, img_raw.type());
    GpuMat dev_bggr45(ROWS2, COLS2, img_raw.type());
    GpuMat dev_bggr90(ROWS2, COLS2, img_raw.type());
    GpuMat dev_bggr135(ROWS2, COLS2, img_raw.type());

    // IMG Size is 2048, 2448, so ideally 5 013 504 threads but can't do that
    dim3 blocks(64, 77);
    dim3 threads(32, 32);
    my_split<<<blocks, threads>>>(dev_img, dev_bggr0, dev_bggr45, dev_bggr90, dev_bggr135);
    CHECK_LAST_CUDA_ERROR();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Split: %f ms\n", elapsedTime);
    Mat bggr;
    dev_bggr0.download(bggr);
    save_img("bggr0", bggr);
    dev_bggr45.download(bggr);
    save_img("bggr45", bggr);
    dev_bggr90.download(bggr);
    save_img("bggr90", bggr);
    dev_bggr135.download(bggr);
    save_img("bggr135", bggr);

    // Debayer
    cudaEventRecord(start, 0);
    GpuMat dev_rgb0(ROWS2, COLS2, CV_8UC3);
    GpuMat dev_rgb45(ROWS2, COLS2, CV_8UC3);
    GpuMat dev_rgb90(ROWS2, COLS2, CV_8UC3);
    GpuMat dev_rgb135(ROWS2, COLS2, CV_8UC3);
    demosaicing(dev_bggr0, dev_rgb0);
    demosaicing(dev_bggr45, dev_rgb45);
    demosaicing(dev_bggr90, dev_rgb90);
    demosaicing(dev_bggr135, dev_rgb135);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Debayer: %f ms\n", elapsedTime);

    // RGB to MONO
    cudaEventRecord(start, 0);
    GpuMat dev_mono0(ROWS2, COLS2, CV_8UC1);
    GpuMat dev_mono45(ROWS2, COLS2, CV_8UC1);
    GpuMat dev_mono90(ROWS2, COLS2, CV_8UC1);
    GpuMat dev_mono135(ROWS2, COLS2, CV_8UC1);
    rgb2mono(dev_rgb0, dev_mono0);
    rgb2mono(dev_rgb45, dev_mono45);
    rgb2mono(dev_rgb90, dev_mono90);
    rgb2mono(dev_rgb135, dev_mono135);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("RGB to Mono: %f ms\n", elapsedTime);
    Mat mono;
    dev_mono0.download(mono);
    save_img("mono0", mono);
    dev_mono45.download(mono);
    save_img("mono45", mono);
    dev_mono90.download(mono);
    save_img("mono90", mono);
    dev_mono135.download(mono);
    save_img("mono135", mono);

    // Stokes
    cudaEventRecord(start, 0);
    blocks.x=32;
    blocks.y=39;
    GpuMat dev_stokes(ROWS2, COLS2, CV_16SC3);
    int sdev_stokes = std::ceil((float)dev_stokes.step / sizeof(short3));
    compute_stokes<<<blocks, threads>>>(dev_mono0, dev_mono45, dev_mono90,
        dev_mono135, reinterpret_cast<short3*>(dev_stokes.data), sdev_stokes);
    CHECK_LAST_CUDA_ERROR();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Stokes: %f ms\n", elapsedTime);
    Mat stokes;
    dev_stokes.download(stokes);
    save_img("stokes", stokes);
    save_text("stokes", stokes);

    // Dolp
    cudaEventRecord(start, 0);
    GpuMat dev_dolp(ROWS2, COLS2, CV_64FC1);
    compute_dolp<<<blocks, threads>>>(reinterpret_cast<short3*>(dev_stokes.data), sdev_stokes, dev_dolp);
    CHECK_LAST_CUDA_ERROR();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Dolp: %f ms\n", elapsedTime);
    Mat dolp;
    dev_dolp.download(dolp);
    normalize(dolp,  dolp, 0, 255, NORM_MINMAX);
    save_img("dolp", dolp);
    save_text("dolp", dolp);

    // Aolp
    cudaEventRecord(start, 0);
    GpuMat dev_aolp(ROWS2, COLS2, CV_64FC1);
    compute_aolp<<<blocks, threads>>>(reinterpret_cast<short3*>(dev_stokes.data), sdev_stokes, dev_aolp);
    CHECK_LAST_CUDA_ERROR();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Aolp: %f ms\n", elapsedTime);
    Mat aolp;
    dev_aolp.download(aolp);
    normalize(aolp,  aolp, 0, 255, NORM_MINMAX);
    save_img("aolp", aolp);
    save_text("aolp", aolp);

    // False coloring
    cudaEventRecord(start, 0);
    GpuMat dev_hsv(ROWS2, COLS2, CV_8UC3);
    GpuMat dev_colored(ROWS2, COLS2, CV_8UC3);
    int sdev_hsv = dev_hsv.step;
    // int sdev_hsv = std::ceil((float)dev_hsv.step / sizeof(uchar3));
    false_coloring<<<blocks, threads>>>(dev_aolp, dev_dolp,
        reinterpret_cast<unsigned char*>(dev_hsv.data), sdev_hsv
    );
    CHECK_LAST_CUDA_ERROR();
    cuda::cvtColor(dev_hsv, dev_colored, COLOR_HSV2RGB);
    CHECK_LAST_CUDA_ERROR();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("False coloring: %f ms\n", elapsedTime);
    
    Mat hsv;
    dev_hsv.download(hsv);
    save_img("hsv", hsv);

    Mat colored;
    dev_colored.download(colored);
    save_img("colored", colored);
    return 0;
}
