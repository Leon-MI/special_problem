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



inline size_t imageFormatSize(size_t width, size_t height,  int format)
{
    size_t depth = sizeof(uchar3) * 8;
    if (format == CV_8UC1)
        depth = sizeof(uchar3);
    // else if (format == CV_16SC3)
    // std::cout << "sizeof : " << depth  << " Format: " << depth << std::endl;

	return (width * height * depth) / 8;
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
    public:
        Mat cpuMat;
        GpuMat gpuMat;

        MMat(int rows, int cols, int img_type, AllocType alloc_type_) {
            alloc_type = alloc_type_;
            if (alloc_type == AllocType::splitted) {
                gpuMat = GpuMat(rows, cols, img_type);
            } else if (alloc_type == AllocType::shared) {
                cuda::HostMem hostMem (rows, cols, img_type, cuda::HostMem::SHARED );
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

        void save_img(const char * name) {
            download();
            std::string out_name = name;
            out_name = "images/" + out_name + ".png";
            std::vector<int> compression_params;
            compression_params.push_back(IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);
            imwrite(out_name, cpuMat, compression_params);
        }
};




void benchmark_indiv(const GpuMat & dev_img_raw) {
    const int n = 50;
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

    MMat bggr0 (ROWS2, COLS2, dev_img_raw.type(), AllocType::splitted );
    MMat bggr45 (ROWS2, COLS2, dev_img_raw.type(), AllocType::splitted );
    MMat bggr90 (ROWS2, COLS2, dev_img_raw.type(), AllocType::splitted );
    MMat bggr135 (ROWS2, COLS2, dev_img_raw.type(), AllocType::splitted );
    
    MMat rgb0 (ROWS2, COLS2, CV_8UC3, AllocType::splitted);
    MMat rgb45 (ROWS2, COLS2, CV_8UC3, AllocType::splitted);
    MMat rgb90 (ROWS2, COLS2, CV_8UC3, AllocType::splitted);
    MMat rgb135 (ROWS2, COLS2, CV_8UC3, AllocType::splitted);
    MMat mono0(ROWS2, COLS2, CV_8UC1, AllocType::splitted);
    MMat mono45(ROWS2, COLS2, CV_8UC1, AllocType::splitted);
    MMat mono90(ROWS2, COLS2, CV_8UC1, AllocType::splitted);
    MMat mono135(ROWS2, COLS2, CV_8UC1, AllocType::splitted);


    GpuMat dev_stokes(ROWS2, COLS2, CV_16SC3);
    GpuMat dev_dolp(ROWS2, COLS2, CV_64FC1);
    GpuMat dev_aolp(ROWS2, COLS2, CV_64FC1);
    GpuMat dev_hsv(ROWS2, COLS2, CV_8UC3);

    MMat colored (ROWS2, COLS2, CV_8UC3, AllocType::splitted);

    Mat stokes;
    Mat dolp;
    Mat aolp;
    Mat hsv;

    for (int i = 0; i < n; i++) {
        // Split
        dim3 blocks(64, 77);
        dim3 threads(32, 32);
        cudaEventRecord(start, 0);
        my_split<<<blocks, threads>>>(dev_img_raw, bggr0.gpuMat, bggr45.gpuMat, bggr90.gpuMat, bggr135.gpuMat);
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

        cuda::cvtColor(rgb0.gpuMat, mono0.gpuMat, COLOR_BGR2GRAY, 0, s0);
        cuda::cvtColor(rgb45.gpuMat, mono45.gpuMat, COLOR_BGR2GRAY, 0, s45);
        cuda::cvtColor(rgb90.gpuMat, mono90.gpuMat, COLOR_BGR2GRAY, 0, s90);
        cuda::cvtColor(rgb135.gpuMat, mono135.gpuMat, COLOR_BGR2GRAY, 0, s135);

        s0.waitForCompletion();
        s45.waitForCompletion();
        s90.waitForCompletion();
        s135.waitForCompletion();

        rgb0.download(); rgb45.download(); rgb90.download(); rgb135.download();
        mono0.download(); mono45.download(); mono90.download(); mono135.download();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        t_debayer_mono += elapsed_time;

        // Stokes
        cudaEventRecord(start, 0);
        int sdev_stokes = std::ceil((float)dev_stokes.step / sizeof(short3));
        compute_stokes<<<blocks, threads>>>(mono0.gpuMat, mono45.gpuMat, mono90.gpuMat, mono135.gpuMat,
             reinterpret_cast<short3*>(dev_stokes.data), sdev_stokes);
        dev_stokes.download(stokes);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        t_stokes += elapsed_time;

        // Dolp + Aolp
        cudaStream_t sdolp, saolp; 
        cudaStreamCreate(&sdolp);
        cudaStreamCreate(&saolp);
    
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

        // False coloring
        cudaEventRecord(start, 0);
        int sdev_hsv = dev_hsv.step;
        false_coloring<<<blocks, threads>>>(dev_aolp, dev_dolp,
            reinterpret_cast<unsigned char*>(dev_hsv.data), sdev_hsv
        );

        cuda::cvtColor(dev_hsv, colored.gpuMat, COLOR_HSV2RGB);
        colored.download();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        t_hsv_rgb += elapsed_time;

    }
    stokes.release();
    dolp.release();
    aolp.release();
    hsv.release();

    rgb0.save_img("rgb0");
    rgb45.save_img("rgb45");
    rgb90.save_img("rgb90");
    rgb135.save_img("rgb135");
    mono0.save_img("mono0");
    mono45.save_img("mono45");
    mono90.save_img("mono90");
    mono135.save_img("mono135");

    colored.save_img("colored");


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "Turtlebot, mean over " << n << " runs" << std::endl;
    std::cout << "Host/Device Memory, streams on debayer/mono & aolp/dolp" << std::endl; 
    std::cout << "Stokes: CV_16SC3, AOLP/DOLP CV_64FC1" << std::endl; 
    std::cout << "Commit : " << std::endl << std::endl;
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
