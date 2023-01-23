#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

using namespace cv;

void split(const Mat& img_raw, Mat& bggr0, Mat& bggr45, Mat& bggr90, Mat& bggr135) {
    for (int i = 0; i < img_raw.rows; i++) {
        for (int j=0; j <img_raw.cols; j+=2) {
            if (i%2 == 0) {
                bggr90.at<uchar>(i/2, j/2) = img_raw.at<uchar>(i, j);
                bggr45.at<uchar>(i/2, j/2) = img_raw.at<uchar>(i, j+1);
            } else {
                bggr135.at<uchar>((i-1)/2, j/2) = img_raw.at<uchar>(i, j);
                bggr0.at<uchar>((i-1)/2, j/2) = img_raw.at<uchar>(i, j+1);
            }
        }
    }
}

void demosaicing(const Mat& input, const Mat& output) {
    cvtColor(input, output, COLOR_BayerBG2BGR);
}

void rgb2mono(const Mat& input, const Mat& output) {
    cvtColor(input, output, COLOR_BGR2GRAY);
}


void compute_stokes(Mat& mono0, Mat& mono45, Mat& mono90, Mat& mono135, Mat& output) {
    for (int i = 0; i < output.rows; i++) {
        for (int j=0; j < output.cols; j++) {
            output.at<cv::Vec3s>(i, j)[0] = mono0.at<uchar>(i, j) + mono90.at<uchar>(i, j);
            output.at<cv::Vec3s>(i, j)[1] = mono0.at<uchar>(i, j) - mono90.at<uchar>(i, j);
            output.at<cv::Vec3s>(i, j)[2] = mono45.at<uchar>(i, j) - mono135.at<uchar>(i, j);
        }
    }
}

void compute_dolp(const Mat& stokes, Mat& output) {
    for (int i = 0; i < stokes.rows; i++) {
        for(int j = 0; j < stokes.cols; j++) {
            short s0 = stokes.at<cv::Vec3s>(i, j)[0];
            if (s0 == (short)0) {
                output.at<double>(i, j) = 0;
            } else {
                short s1 = stokes.at<cv::Vec3s>(i, j)[1];
                short s2 = stokes.at<cv::Vec3s>(i, j)[2];
                output.at<double>(i, j) = std::sqrt(std::pow(s1, 2) + std::pow(s2, 2))/s0;
            }
        }
    }
}

void compute_aolp(const Mat& stokes, Mat& output) {
    for (int i = 0; i < stokes.rows; i++) {
        for(int j = 0; j < stokes.cols; j++) {
            short s2 = stokes.at<cv::Vec3s>(i, j)[2];
            if (s2 == (short)0) {
                output.at<double>(i, j) = 0;
            } else {
                short s1 = stokes.at<cv::Vec3s>(i, j)[1];
                double div = (double)s1 / s2;
                output.at<double>(i, j) = (double)1/2 * std::atan(div);
            }
        }
    }
}

void false_coloring(const Mat & aolp, unsigned char intensity, const Mat & dolp, Mat & output) {
    Mat hsv(aolp.size(), CV_8UC3);
    // Hue -> 0, Saturation -> 1, value -> 2
    for (int i = 0; i < dolp.rows; i++) {
        for(int j = 0; j < dolp.cols; j++) {
            double a = aolp.at<double>(i, j);
            double d = dolp.at<double>(i, j)*255;
            hsv.at<Vec3b>(i, j)[0] = (unsigned char) (179 * std::fmod(a, CV_PI) / CV_PI);
            hsv.at<Vec3b>(i, j)[1] = intensity;
            if (d<0)
                hsv.at<Vec3b>(i, j)[2] = 0;
            else if (d > 255)
                hsv.at<Vec3b>(i, j)[2] = 255;
            else
                hsv.at<Vec3b>(i, j)[2] = (unsigned char) d;
        }
    }
    cvtColor(hsv, output, COLOR_HSV2RGB);
}

void save_img(const char * name, const Mat & M) {
    std::string out_name = name;
    out_name = "images/" + out_name + ".png";
    std::vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    imwrite(out_name, M, compression_params);
}

int main()
{
    Mat img_raw = imread("images/frame00000_raw.png", IMREAD_GRAYSCALE);
 
    std::cout << "img_raw: " << img_raw.size() << std::endl;

    TickMeter tm;

    // Split
    tm.start();
    Mat bggr0(img_raw.size()/2, img_raw.type());
    Mat bggr45(img_raw.size()/2, img_raw.type());
    Mat bggr90(img_raw.size()/2, img_raw.type());
    Mat bggr135(img_raw.size()/2, img_raw.type());
    // std::cout << "bggr0: (" << bggr0.rows << ", " << bggr0.cols << ")" << std::endl;
    split(img_raw, bggr0, bggr45, bggr90, bggr135);


    // Debayer
    Mat rgb0(bggr0.size(), CV_8UC3);
    Mat rgb45(bggr45.size(), CV_8UC3);
    Mat rgb90(bggr90.size(), CV_8UC3);
    Mat rgb135(bggr135.size(), CV_8UC3);
    demosaicing(bggr0, rgb0);
    demosaicing(bggr45, rgb45);
    demosaicing(bggr90, rgb90);
    demosaicing(bggr135, rgb135);
    tm.stop();
    std::cout << "Split + debayer: " << tm.getTimeSec() << "sec" << std::endl;

    tm.reset();
    tm.start();
    Mat mono0(rgb0.size(), CV_8UC1);
    Mat mono45(rgb45.size(), CV_8UC1);
    Mat mono90(rgb90.size(), CV_8UC1);
    Mat mono135(rgb135.size(), CV_8UC1);
    rgb2mono(rgb0, mono0);
    rgb2mono(rgb45, mono45);
    rgb2mono(rgb90, mono90);
    rgb2mono(rgb135, mono135);
    tm.stop();
    std::cout << "grayscale: " << tm.getTimeSec() << "sec" << std::endl;

    tm.reset();
    tm.start();
    Mat stokes(mono0.size(), CV_16SC3);
    compute_stokes(mono0, mono45, mono90, mono135, stokes);
    tm.stop();
    std::cout << "stokes: " << tm.getTimeSec() << "sec" << std::endl;

    tm.reset();
    tm.start();
    Mat dolp(stokes.size(), CV_64FC1);
    compute_dolp(stokes, dolp);
    tm.stop();
    std::cout << "dolp: " << tm.getTimeSec() << "sec" << std::endl;

    tm.reset();
    tm.start();
    Mat aolp(stokes.size(), CV_64FC1);
    compute_aolp(stokes, aolp);
    tm.stop();
    std::cout << "aolp: " << tm.getTimeSec() << "sec" << std::endl;

    tm.reset();
    tm.start();
    Mat colored(aolp.size(), CV_8UC3);
    false_coloring(aolp, 255, dolp, colored);
    tm.stop();
    std::cout << "coloring: " << tm.getTimeSec() << "sec" << std::endl;


    save_img("out", colored);

    return 0;
}
