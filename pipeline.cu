#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaimgproc.hpp>
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

#define ALLOC_TYPE AllocType::splitted

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


    if (i >= ROWS || j >= (COLS / 4 + 1)) {
        return;
    }
    j = j*2;
    int jj = j*2;
    if (i%2 == 0) {
        bggr90(i/2, j) = dev_img(i, jj);
        bggr90(i/2, j+1) = dev_img(i, jj+2);
        bggr45(i/2, j) = dev_img(i, jj+1);
        bggr45(i/2, j+1) = dev_img(i, jj+3);
    } else {
        bggr135((i-1)/2, j) = dev_img(i, jj);
        bggr135((i-1)/2, j+1) = dev_img(i, jj+2);
        bggr0((i-1)/2, j) = dev_img(i, jj+1);
        bggr0((i-1)/2, j+1) = dev_img(i, jj+3);
    }
}


inline __device__ short3 compute_stokes(
    const int i,
    const int j,
    const cuda::PtrStepSzb mono0,
    const cuda::PtrStepSzb mono45,
    const cuda::PtrStepSzb mono90,
    const cuda::PtrStepSzb mono135,
    cuda::PtrStep<short3>  stokes
    ) {
    const unsigned char m0 = mono0(i, j);
    const unsigned char m90 = mono90(i, j);
    
    short3 v_stokes;
    v_stokes.x = m0 + m90;
    v_stokes.y = m0 - m90;
    v_stokes.z = mono45(i, j) - mono135(i, j);

    stokes(i, j) = v_stokes;
    return v_stokes;
}

inline __device__ float compute_dolp(const int i, const int j, const short3 & v_stokes, cuda::PtrStepSz<float> dolp) {
    float v_dolp;
    if (v_stokes.x == (short)0)
        v_dolp = 0;
    else
        v_dolp = sqrtf((v_stokes.y*v_stokes.y) + (v_stokes.z*v_stokes.z)) / v_stokes.x;
    dolp(i, j) = v_dolp;
    return v_dolp;
}

inline __device__ float compute_aolp(const int i, const int j, const short3 & v_stokes, cuda::PtrStepSz<float> aolp) {
    float v_aolp;
    if (v_stokes.z == (short)0) {
        v_aolp = 0.0;
    } else {
        const float sy = (float)v_stokes.y;
        const float sz = (float)v_stokes.z;
        const float angle = ((float)1/2) * atan2f(sy, sz);
        v_aolp = angle + (float)CV_PI/2;
    }
    aolp(i, j) = v_aolp;
    return v_aolp;
}

inline __device__ uchar3 aolp_dolp_2_hsv(const int i, const int j, const float v_dolp, const float v_aolp, cuda::PtrStep<uchar3> hsv) {
    uchar3 v_hsv;

    v_hsv.x = (unsigned char) (179 * fmodf(v_aolp, CV_PI) / CV_PI);
    v_hsv.y = (unsigned char) 255;
    v_hsv.z = (unsigned char) min(max((float)0, v_dolp*255), (float)255);

    hsv(i, j) = v_hsv;
    return v_hsv;
}


__global__ void pipeline_end(
    const cuda::PtrStepSzb mono0,
    const cuda::PtrStepSzb mono45,
    const cuda::PtrStepSzb mono90,
    const cuda::PtrStepSzb mono135,
    cuda::PtrStep<short3>  stokes,
    cuda::PtrStepSz<float> dolp,
    cuda::PtrStepSz<float> aolp,
    cuda::PtrStep<uchar3>  hsv
    ) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i > (ROWS2-1) || j > (COLS2-1))
        return;
    
    const short3 v_stokes = compute_stokes(
        i, j,
        mono0, mono45, mono90, mono135, stokes
    );

    const float v_dolp = compute_dolp(i, j, v_stokes, dolp);
    const float v_aolp = compute_aolp(i, j, v_stokes, aolp);

    // False coloring
    const uchar3 v_hsv = aolp_dolp_2_hsv(i, j, v_dolp, v_aolp, hsv);
}



inline size_t imageFormatSize(size_t width, size_t height, int format)
{
    size_t s = sizeof(uchar3);
    if (format == CV_8UC1)
        s = sizeof(uchar1);
    else if (format == CV_32FC3)
        s = sizeof(float3);
    else if (format == CV_32FC1)
        s = sizeof(float);
    else if (format == CV_64FC1)
        s = sizeof(double);
    else if (format == CV_32SC3)
        s = sizeof(int3);
    else if (format == CV_16SC3)
        s = sizeof(short3);

	return width * height * s;
}

enum AllocType {
    splitted,
    shared,
    mapped, 
    unified
};

class MMat {
    private:
        AllocType alloc_type;
        cuda::HostMem hostMem;
    public:
        Mat cpuMat;
        GpuMat gpuMat;

        MMat(int rows, int cols, int img_type, AllocType alloc_type_) {
            alloc_type = alloc_type_;
            if (alloc_type == AllocType::splitted) {
                gpuMat = GpuMat(rows, cols, img_type);
            } else if (alloc_type == AllocType::shared) {
                hostMem = cuda::HostMem(rows, cols, img_type, cuda::HostMem::SHARED );
                CHECK_LAST_CUDA_ERROR();
                cpuMat = hostMem.createMatHeader();
                gpuMat = hostMem.createGpuMatHeader();
            } else if (alloc_type == AllocType::mapped) {
                void *cpu_ptr, *gpu_ptr;
                size_t size = imageFormatSize(cols, rows, img_type);
                cudaHostAlloc(&cpu_ptr, size, cudaHostAllocMapped);
                CHECK_LAST_CUDA_ERROR();
                cudaHostGetDevicePointer(&gpu_ptr, cpu_ptr, 0);
                CHECK_LAST_CUDA_ERROR();

                gpuMat = GpuMat(rows, cols, img_type, gpu_ptr);
                cpuMat = Mat(rows, cols, img_type, cpu_ptr);
            } else if (alloc_type == AllocType::unified) {
                void* unified_ptr;
                size_t size = imageFormatSize(cols, rows, img_type);
                cudaMallocManaged(&unified_ptr, size);
                CHECK_LAST_CUDA_ERROR();

                cpuMat = Mat(rows, cols, img_type, unified_ptr);
                gpuMat = GpuMat(rows, cols, img_type, unified_ptr);
            }
        }

        ~MMat() {
            switch (alloc_type) {
                case AllocType::splitted:
                    // TODO Check if cpuMat is initialized
                    cpuMat.release();
                    gpuMat.release();
                    break;
                case AllocType::shared:
                case AllocType::mapped:
                case AllocType::unified:
                    cpuMat.release();
                    break;
                default:
                    break;
            }
        }

        void download() {
            if (alloc_type == AllocType::splitted)
                gpuMat.download(cpuMat);
        }

        void download(cuda::Stream & s) {
            if (alloc_type == AllocType::splitted)
                gpuMat.download(cpuMat, s);
        }

        void save_img(const char * name) {
            download();
            std::string out_name = name;
            out_name = "images/" + out_name + ".png";
            std::vector<int> compression_params;
            compression_params.push_back(IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);
            imwrite(out_name, cpuMat, compression_params);
        }

        MMat(const MMat&) = delete;
        // "copy assignment operator"
        MMat& operator= (const MMat&) = delete;  //  MMat p6; p6 = p1;
        // "move constructor"
        MMat(MMat&&) = delete;                   //  MMat p7{ std::move(p2) };
        // "move assignment operator"
        MMat& operator= (MMat&&) = delete;  
};



void benchmark_indiv(const GpuMat & dev_img_raw) {
    int n = 51;
    float t_upload = 0.0f;
    float t_split = 0.0f;
    float t_debayer_mono = 0.0f;
    float t_stokes = 0.0f;
    float t_aolp_dolp = 0.0f;
    float t_hsv_rgb = 0.0f;
    float t_pipeline_end = 0.0f;
    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    MMat bggr0 (ROWS2, COLS2, CV_8UC1, ALLOC_TYPE );
    MMat bggr45 (ROWS2, COLS2, CV_8UC1, ALLOC_TYPE );
    MMat bggr90 (ROWS2, COLS2, CV_8UC1, ALLOC_TYPE );
    MMat bggr135 (ROWS2, COLS2, CV_8UC1, ALLOC_TYPE );

    MMat rgb0 (ROWS2, COLS2, CV_8UC3, ALLOC_TYPE);
    MMat rgb45 (ROWS2, COLS2, CV_8UC3, ALLOC_TYPE);
    MMat rgb90 (ROWS2, COLS2, CV_8UC3, ALLOC_TYPE);
    MMat rgb135 (ROWS2, COLS2, CV_8UC3, ALLOC_TYPE);
    MMat mono0(ROWS2, COLS2, CV_8UC1, ALLOC_TYPE);
    MMat mono45(ROWS2, COLS2, CV_8UC1, ALLOC_TYPE);
    MMat mono90(ROWS2, COLS2, CV_8UC1, ALLOC_TYPE);
    MMat mono135(ROWS2, COLS2, CV_8UC1, ALLOC_TYPE);

    MMat m_stokes(ROWS2, COLS2, CV_16SC3, ALLOC_TYPE);

    MMat dolp(ROWS2, COLS2, CV_32FC1, ALLOC_TYPE);
    MMat aolp(ROWS2, COLS2, CV_32FC1, ALLOC_TYPE);

    MMat hsv(ROWS2, COLS2, CV_8UC3, ALLOC_TYPE);
    MMat colored (ROWS2, COLS2, CV_8UC3, ALLOC_TYPE);

    CHECK_LAST_CUDA_ERROR();

    for (int i = 0; i < n; i++) {
        if (i == 1) {
            std::cout << "First round done" << std::endl;
            // Kernel have been compiled, reset timers
            t_upload = 0.0f;
            t_split = 0.0f;
            t_debayer_mono = 0.0f;
            t_stokes = 0.0f;
            t_aolp_dolp = 0.0f;
            t_hsv_rgb = 0.0f;
            t_pipeline_end = 0.0f;
        }
        // Split
        dim3 blocks(64, 20);
        dim3 threads(32, 32);
        cudaEventRecord(start, 0);
        my_split<<<blocks, threads>>>(dev_img_raw, bggr0.gpuMat, bggr45.gpuMat, bggr90.gpuMat, bggr135.gpuMat);
        CHECK_LAST_CUDA_ERROR();

        bggr0.download(); bggr45.download(); bggr90.download(); bggr135.download();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        t_split += elapsed_time;

        // Debayer + RGB To mono
        blocks.x=32;
        blocks.y=39;

        cuda::Stream s0, s45, s90, s135;
        cudaEventRecord(start, 0);
        cuda::cvtColor(bggr0.gpuMat, rgb0.gpuMat, COLOR_BayerBG2BGR, 0, s0);
        cuda::cvtColor(bggr45.gpuMat, rgb45.gpuMat, COLOR_BayerBG2BGR, 0, s45);
        cuda::cvtColor(bggr90.gpuMat, rgb90.gpuMat, COLOR_BayerBG2BGR, 0, s90);
        cuda::cvtColor(bggr135.gpuMat, rgb135.gpuMat, COLOR_BayerBG2BGR, 0, s135);
        rgb0.download(s0); rgb45.download(s45); rgb90.download(s90); rgb135.download(s135);

        cuda::cvtColor(rgb0.gpuMat, mono0.gpuMat, COLOR_BGR2GRAY, 0, s0);
        cuda::cvtColor(rgb45.gpuMat, mono45.gpuMat, COLOR_BGR2GRAY, 0, s45);
        cuda::cvtColor(rgb90.gpuMat, mono90.gpuMat, COLOR_BGR2GRAY, 0, s90);
        cuda::cvtColor(rgb135.gpuMat, mono135.gpuMat, COLOR_BGR2GRAY, 0, s135);
        mono0.download(s0); mono45.download(s45); mono90.download(s90); mono135.download(s135);

        s0.waitForCompletion();
        s45.waitForCompletion();
        s90.waitForCompletion();
        s135.waitForCompletion();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        t_debayer_mono += elapsed_time;

        // End of pipeline: stokes + aolp + dolp + false coloring
        cudaEventRecord(start, 0);
        pipeline_end<<<blocks, threads>>>(mono0.gpuMat, mono45.gpuMat, mono90.gpuMat, mono135.gpuMat, m_stokes.gpuMat, dolp.gpuMat, aolp.gpuMat, hsv.gpuMat);

        m_stokes.download();
        dolp.download();
        aolp.download();

        cuda::cvtColor(hsv.gpuMat, colored.gpuMat, COLOR_HSV2RGB);
        colored.download();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        t_pipeline_end += elapsed_time;
    }

    colored.save_img("colored");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    n -= 1;
    std::cout << "Xavier, mean over " << n << " runs" << std::endl;
    std::cout << "AllocType::splitted, streams on debayer/mono" << std::endl; 
    std::cout << "Stokes: CV_16SC3, AOLP/DOLP: CV_32FC1" << std::endl; 
    std::cout << "Commit :" << std::endl << std::endl;
    std::cout << "Upload: " << t_upload / n << "ms" << std::endl;
    std::cout << "Split: " << t_split/n << "ms" << std::endl;
    std::cout << "Debayer + mono: " << t_debayer_mono / n << "ms" << std::endl;
    std::cout << "Stokes + Aolp + Dolp + False coloring: " << t_pipeline_end / n << "ms" << std::endl;
    std::cout << "Total: " << (t_upload + t_split + t_debayer_mono + t_pipeline_end) / n << "ms" << std::endl; 
}

int main()
{
    // cuda::setDevice(0);
    // cudaSetDeviceFlags(cudaDeviceMapHost);
    // Read and upload img to gpu
    Mat img_raw = imread("images/frame00000_raw.png", IMREAD_GRAYSCALE);
    GpuMat dev_img_raw;
    dev_img_raw.upload(img_raw);

    benchmark_indiv(dev_img_raw);

    dev_img_raw.release();
    img_raw.release();
    return 0;
}
