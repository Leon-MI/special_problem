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


void benchmark_indiv(const Mat & img_raw) {
    const int n = 50;
    // float t_split [n] = {0.0f};
    // float t_debayer_mono [n] = {0.0f} ;
    // float t_stokes [n] = {0.0f};
    // float t_aolp_dolp [n] = {0.0f};
    // float t_hsv_rgb [n] = {0.0f};
    float t_upload = 0.0f;
    float t_split = 0.0f;
    float t_debayer_mono = 0.0f;
    float t_stokes = 0.0f;
    float t_aolp_dolp = 0.0f;
    float t_hsv_rgb = 0.0f;
    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    GpuMat dev_bggr0(ROWS2, COLS2, img_raw.type());
    GpuMat dev_bggr45(ROWS2, COLS2, img_raw.type());
    GpuMat dev_bggr90(ROWS2, COLS2, img_raw.type());
    GpuMat dev_bggr135(ROWS2, COLS2, img_raw.type());
    GpuMat dev_rgb0(ROWS2, COLS2, CV_8UC3);
    GpuMat dev_rgb45(ROWS2, COLS2, CV_8UC3);
    GpuMat dev_rgb90(ROWS2, COLS2, CV_8UC3);
    GpuMat dev_rgb135(ROWS2, COLS2, CV_8UC3);
    GpuMat dev_mono0(ROWS2, COLS2, CV_8UC1);
    GpuMat dev_mono45(ROWS2, COLS2, CV_8UC1);
    GpuMat dev_mono90(ROWS2, COLS2, CV_8UC1);
    GpuMat dev_mono135(ROWS2, COLS2, CV_8UC1);
    GpuMat dev_stokes(ROWS2, COLS2, CV_16SC3);
    GpuMat dev_dolp(ROWS2, COLS2, CV_64FC1);
    GpuMat dev_aolp(ROWS2, COLS2, CV_64FC1);
    GpuMat dev_hsv(ROWS2, COLS2, CV_8UC3);
    GpuMat dev_colored(ROWS2, COLS2, CV_8UC3);

    for (int i = 0; i < n; i++) {
        // Upload img
        cudaEventRecord(start, 0);
        GpuMat dev_img;
        dev_img.upload(img_raw);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        t_upload += elapsed_time;

        // Split
        dim3 blocks(64, 77);
        dim3 threads(32, 32);
        cudaEventRecord(start, 0);
        my_split<<<blocks, threads>>>(dev_img, dev_bggr0, dev_bggr45, dev_bggr90, dev_bggr135);
        Mat bggr0, bggr45, bggr90, bggr135;
        dev_bggr0.download(bggr0);
        dev_bggr45.download(bggr45);
        dev_bggr90.download(bggr90);
        dev_bggr135.download(bggr135);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        t_split += elapsed_time;
        bggr0.release();bggr45.release();bggr90.release();bggr135.release();

        // Debayer + RGB To mono
        blocks.x=32;
        blocks.y=39;
        Mat rgb0, rgb45, rgb90, rgb135;
        Mat mono0, mono45, mono90, mono135;

        cuda::Stream s0, s45, s90, s135;
        cudaEventRecord(start, 0);
        cuda::cvtColor(dev_bggr0, dev_rgb0, COLOR_BayerBG2BGR, 0, s0);
        cuda::cvtColor(dev_bggr45, dev_rgb45, COLOR_BayerBG2BGR, 0, s45);
        cuda::cvtColor(dev_bggr90, dev_rgb90, COLOR_BayerBG2BGR, 0, s90);
        cuda::cvtColor(dev_bggr135, dev_rgb135, COLOR_BayerBG2BGR, 0, s135);
        dev_rgb0.download(rgb0, s0);
        dev_rgb45.download(rgb45, s45);
        dev_rgb90.download(rgb90, s90);
        dev_rgb135.download(rgb135, s135);

        cuda::cvtColor(dev_rgb0, dev_mono0, COLOR_BGR2GRAY, 0, s0);
        cuda::cvtColor(dev_rgb45, dev_mono45, COLOR_BGR2GRAY, 0, s45);
        cuda::cvtColor(dev_rgb90, dev_mono90, COLOR_BGR2GRAY, 0, s90);
        cuda::cvtColor(dev_rgb135, dev_mono135, COLOR_BGR2GRAY, 0, s135);
        dev_mono0.download(mono0, s0);
        dev_mono45.download(mono45, s45);
        dev_mono90.download(mono90, s90);
        dev_mono135.download(mono135, s135);

        s0.waitForCompletion();
        s45.waitForCompletion();
        s90.waitForCompletion();
        s135.waitForCompletion();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        t_debayer_mono += elapsed_time;

        rgb0.release(); rgb45.release(); rgb90.release(); rgb135.release();
        mono0.release(); mono45.release(); mono90.release(); mono135.release();

        // Stokes

        Mat stokes;
        cudaEventRecord(start, 0);
        int sdev_stokes = std::ceil((float)dev_stokes.step / sizeof(short3));
        compute_stokes<<<blocks, threads>>>(dev_mono0, dev_mono45, dev_mono90,
            dev_mono135, reinterpret_cast<short3*>(dev_stokes.data), sdev_stokes);
        dev_stokes.download(stokes);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        t_stokes += elapsed_time;
        save_img("stokes", stokes);
        stokes.release();

        // Dolp + Aolp
        cudaStream_t sdolp; 
        cudaStreamCreate(&sdolp);

        Mat dolp;
        cudaStream_t saolp; 
        cudaStreamCreate(&saolp);
        Mat aolp;
    
        cudaEventRecord(start, 0);
        compute_dolp<<<blocks, threads, 0, sdolp>>>(reinterpret_cast<short3*>(dev_stokes.data), sdev_stokes, dev_dolp);
        compute_aolp<<<blocks, threads, 0, saolp>>>(reinterpret_cast<short3*>(dev_stokes.data), sdev_stokes, dev_aolp);

        cudaStreamDestroy(sdolp);
        cudaStreamDestroy(saolp);
        dev_dolp.download(dolp);
        dev_aolp.download(aolp);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        t_aolp_dolp += elapsed_time;
        dolp.release(); aolp.release();

        // False coloring
        Mat colored;
        cudaEventRecord(start, 0);
        int sdev_hsv = dev_hsv.step;
        false_coloring<<<blocks, threads>>>(dev_aolp, dev_dolp,
            reinterpret_cast<unsigned char*>(dev_hsv.data), sdev_hsv
        );

        cuda::cvtColor(dev_hsv, dev_colored, COLOR_HSV2RGB);
        dev_colored.download(colored);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        t_hsv_rgb += elapsed_time;

        save_img("colored", colored);
        colored.release();

        dev_img.release();
    }
    dev_bggr0.release();
    dev_bggr45.release();
    dev_bggr90.release();
    dev_bggr135.release();
    dev_rgb0.release();
    dev_rgb45.release();
    dev_rgb90.release();
    dev_rgb135.release();
    dev_mono0.release();
    dev_mono45.release();
    dev_mono90.release();
    dev_mono135.release();
    dev_stokes.release();
    dev_dolp.release();
    dev_aolp.release();
    dev_hsv.release();
    dev_colored.release();
    // for (int i = 0; i < n;  i++) {
    //     std::cout << t_split[n] << " ";
    // }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "Turtlebot, mean over " << n << " runs" << std::endl;
    std::cout << "Host/Device Memory, streams on debayer/mono & aolp/dolp" << std::endl << std::endl;
    std::cout << "Upload: " << t_upload / n << "ms" << std::endl;
    std::cout << "Split: " << t_split/n << "ms" << std::endl;
    std::cout << "Debayer + mono: " << t_debayer_mono / n << "ms" << std::endl;
    std::cout << "Stokes: " << t_stokes / n << "ms" << std::endl;
    std::cout << "Aolp + Dolp: " << t_aolp_dolp / n << "ms" << std::endl;
    std::cout << "False coloring: " << t_hsv_rgb / n << "ms" << std::endl;
    std::cout << "Total: " << (t_upload + t_split + t_debayer_mono + t_stokes + t_aolp_dolp + t_hsv_rgb) / n << "ms" << std::endl; 
}

int main()
{
    cuda::setDevice(0);
    // Read and upload img to gpu
    Mat img_raw = imread("images/frame00000_raw.png", IMREAD_GRAYSCALE);

    benchmark_indiv(img_raw);
    img_raw.release();
    return 0;
}
