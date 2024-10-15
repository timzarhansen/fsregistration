//
// Created by tim-linux on 26.03.22.
//

//
// Created by jurobotics on 13.09.21.
//
// /home/tim-external/dataFolder/StPereDataset/lowNoise52/scanNumber_0/00_ForShow.jpg /home/tim-external/dataFolder/StPereDataset/lowNoise52/scanNumber_1/00_ForShow.jpg
// /home/tim-external/dataFolder/ValentinBunkerData/noNoise305_52/scanNumber_0/00_ForShow.jpg  /home/tim-external/dataFolder/ValentinBunkerData/noNoise305_52/scanNumber_1/00_ForShow.jpg
#include "generalHelpfulTools.h"
//#include "slamToolsRos.h"
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <filesystem>
#include "softRegistrationClass.h"
//#include <pcl/point_types.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>


void convertMatToDoubleArray(cv::Mat inputImg, double voxelData[]) {

    std::vector<uchar> array;
    if (inputImg.isContinuous()) {
        // array.assign(mat.datastart, mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
        array.assign(inputImg.data, inputImg.data + inputImg.total() * inputImg.channels());
    } else {
        for (int i = 0; i < inputImg.rows; ++i) {
            array.insert(array.end(), inputImg.ptr<uchar>(i),
                         inputImg.ptr<uchar>(i) + inputImg.cols * inputImg.channels());
        }
    }

    for (int i = 0; i < array.size(); i++) {
        voxelData[i] = array[i];
//        std::cout << voxelData[i] <<std::endl;
//        std::cout << "i: "<< i <<std::endl;
    }

}


int getVoxelIndex(float x, float y, float z, float voxelSize, int N) {
    int voxelX = static_cast<int>(x / voxelSize);
    int voxelY = static_cast<int>(y / voxelSize);
    int voxelZ = static_cast<int>(z / voxelSize);
    return voxelX + voxelY * N + voxelZ * N * N;
}


void process3Dimage(const std::string& filename, float gridSideLength, float voxelSize, int N) {
    std::ifstream file(filename);
    std::string line;
    bool headerEnded = false;

    while (std::getline(file, line)) {
        if (line == "end_header") {
            headerEnded = true;
            continue;
        }

        if (headerEnded) {
            std::istringstream iss(line);
            float x, y, z, confidence, intensity;
            if (!(iss >> x >> y >> z >> confidence >> intensity)) { break; } // Error

            x += gridSideLength / 2;
            y += gridSideLength / 2;
            z += gridSideLength / 2;

            std::vector<int> voxelGrid(N * N * N, 0);
            int index = getVoxelIndex(x, y, z, voxelSize, N);
            if (index >= 0 && index < voxelGrid.size()) {
                voxelGrid[index] = 1;
            }
        }
    }
}

int main(int argc, char **argv) {
    // input needs to be two scans as voxelData



    std::string current_exec_name = argv[0]; // Name of the current exec program
    std::vector<std::string> all_args;

    if (argc > 2) {
        //std::cout << "temp1" << std::endl;
        all_args.assign(argv + 1, argv + argc);
        //std::cout << "12"<< all_args[1]<<std::endl;
    } else {
        std::cout << "no arguments given" << std::endl;
        exit(-1);
    }
    std::cout << all_args[0] << std::endl;

    cv::Mat img1 = cv::imread(
            all_args[0],
            cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(
            all_args[1],
            cv::IMREAD_GRAYSCALE);
//   cv::imshow("Display window", img1);
//    int k = cv::waitKey(0);
//    cv::Mat img1 = cv::imread("/home/ws/matlab/registrationFourier/FMT/firstImage.jpg", cv::IMREAD_GRAYSCALE);
//    cv::Mat img2 = cv::imread("/home/ws/matlab/registrationFourier/FMT/secondImage.jpg", cv::IMREAD_GRAYSCALE);
    int dimensionScan = img1.rows;
    std::cout << "image size: " << dimensionScan << std::endl;
    double *voxelData1;
    double *voxelData2;
    voxelData1 = (double *) malloc(sizeof(double) * dimensionScan * dimensionScan);
    voxelData2 = (double *) malloc(sizeof(double) * dimensionScan * dimensionScan);

    convertMatToDoubleArray(img1, voxelData1);
    convertMatToDoubleArray(img2, voxelData2);

    softRegistrationClass scanRegistrationObject(img1.rows, img1.rows / 2, img1.rows / 2, img1.rows / 2 - 1);

    double fitnessX;
    double fitnessY;
    Eigen::Matrix4d ourInitialGuess = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d covarianceMatrixResult;
    Eigen::Matrix4d estimatedTransformation = scanRegistrationObject.registrationOfTwoVoxelsSOFFTFast(voxelData1,
                                                                                                      voxelData2,
                                                                                                      ourInitialGuess,
                                                                                                      covarianceMatrixResult,
                                                                                                      true, true,
                                                                                                      1,
                                                                                                      false,
                                                                                                      true);
//    Eigen::Matrix4d tmpMatrix4d = estimatedTransformation.inverse();
//    estimatedTransformation = tmpMatrix4d;

//    estimatedTransformation(1, 3) = estimatedTransformation(1, 3)-5;
//    estimatedTransformation(0, 3) = estimatedTransformation(0, 3)-2;
    cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << estimatedTransformation(0,0),
            estimatedTransformation(0,1),
            estimatedTransformation(0, 3),
            estimatedTransformation(1,0),
            estimatedTransformation(1,1),
            estimatedTransformation(1, 3));






    cv::Mat magTMP1(dimensionScan, dimensionScan, CV_64F, voxelData1);
    //add gaussian blur
    //            cv::imwrite("/home/tim-external/Documents/imreg_fmt/firstImage.jpg", magTMP1);

    cv::Mat magTMP2(dimensionScan, dimensionScan, CV_64F, voxelData2);

    std::cout << trans_mat << std::endl;
    warpAffine(magTMP2, magTMP2, trans_mat, magTMP2.size());
//            convertMatToDoubleArray(img1, voxelData1);
//            convertMatToDoubleArray(img2, voxelData2);

    std::ofstream myFile1, myFile2;
    myFile1.open("/home/ws/matlab/registrationFourier/csvFiles/resultVoxel1.csv");
    myFile2.open("/home/ws/matlab/registrationFourier/csvFiles/resultVoxel2.csv");
    for (int i = 0; i < dimensionScan; i++) {
        for (int j = 0; j < dimensionScan; j++) {
            myFile1 << voxelData1[j + dimensionScan * i]; // real part
            myFile1 << "\n";
            myFile2 << voxelData2[j + dimensionScan * i]; // imaginary part
            myFile2 << "\n";
        }
    }
    myFile1.close();
    myFile2.close();







//    Eigen::Matrix4d estimatedTransformation = scanRegistrationObject.sofftRegistration(*scan1,*scan2,fitnessX,fitnessY,-100,true);


    return (0);
}