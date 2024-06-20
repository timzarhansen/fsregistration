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
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/common/norms.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "softRegistrationClass3D.h"

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


void process3Dimage(const std::string &filename, float gridSideLength, float voxelSize, int N) {
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


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud <pcl::PointXYZ>);
    pcl::PLYReader Reader;
    Reader.read("/home/tim-external/ros2_underlay/src/fs2D/exampleData/dragon_recon/dragon_vrip.ply", *cloud);

    pcl::ConvexHull <pcl::PointXYZ> cHull;
    pcl::PointCloud <pcl::PointXYZ> cHull_points;
    cHull.setInputCloud(cloud);
    cHull.reconstruct(cHull_points);
    double maximumDistance = 0;
    for (int i = 0; i < cloud->points.size(); i++) {
        Eigen::Vector3d tmpPoint;
        tmpPoint.x() = cloud->points[i].x;
        tmpPoint.y() = cloud->points[i].y;
        tmpPoint.z() = cloud->points[i].z;
        if (maximumDistance < tmpPoint.norm()) {
            maximumDistance = tmpPoint.norm();
        }

    }
//    std::cout << maximumDistance << std::endl;
    double sizeVoxelOneDirection = 2 * maximumDistance * 1.4;
    int N = 128;
    double *voxelData1;
    double *voxelData2;
    voxelData1 = (double *) malloc(sizeof(double) * N * N * N);
    voxelData2 = (double *) malloc(sizeof(double) * N * N * N);
    for (int i = 0; i < N * N * N; i++) {
        voxelData1[i] = 0;
        voxelData2[i] = 0;
    }

    int whichObject = 0;// 0 = dragon ; 1 = double dragon ; 2 = cube

    switch (whichObject) {
        case 0:
            ///////////////////////// Dragon
            for (int i = 0; i < cloud->points.size(); i++) {
                Eigen::Vector4d currentVector(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1);
                currentVector = generalHelpfulTools::getTransformationMatrixFromRPY(0, 0, 0) * currentVector;
                int xIndex = N / 2 + currentVector.x() * N / sizeVoxelOneDirection;
                int yIndex = N / 2 + currentVector.y() * N / sizeVoxelOneDirection;
                int zIndex = N / 2 + currentVector.z() * N / sizeVoxelOneDirection;
                voxelData1[generalHelpfulTools::index3D(xIndex, yIndex, zIndex, N)] = 1;
            }

            for (int i = 0; i < cloud->points.size(); i++) {
                Eigen::Vector4d currentVector(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1);
                Eigen::Vector4d shiftVector(10, 5, -14, 0);// in pixel
                currentVector += shiftVector * sizeVoxelOneDirection / N;
                currentVector =
                        generalHelpfulTools::getTransformationMatrixFromRPY(40.0 / 180.0 * M_PI, -30.0 / 180.0 * M_PI,
                                                                            10.0 / 180.0 * M_PI) * currentVector;

                int xIndex = N / 2 + currentVector.x() * N / sizeVoxelOneDirection;
                int yIndex = N / 2 + currentVector.y() * N / sizeVoxelOneDirection;
                int zIndex = N / 2 + currentVector.z() * N / sizeVoxelOneDirection;
                voxelData2[generalHelpfulTools::index3D(xIndex, yIndex, zIndex, N)] = 1;
            }

            break;
        case 1:
            ///////////////////////// Dragon Mirrored
            for (int i = 0; i < cloud->points.size(); i++) {
                Eigen::Vector4d currentVector(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1);
                currentVector = generalHelpfulTools::getTransformationMatrixFromRPY(0, 0, 0) * currentVector;
                int xIndex = N / 2 + currentVector.x() * N / sizeVoxelOneDirection;
                int yIndex = N / 2 + currentVector.y() * N / sizeVoxelOneDirection;
                int zIndex = N / 2 + currentVector.z() * N / sizeVoxelOneDirection;
                voxelData1[zIndex + N * yIndex + N * N * xIndex] = 1;
            }

            for (int i = 0; i < cloud->points.size(); i++) {
                Eigen::Vector4d currentVector(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1);
                currentVector = generalHelpfulTools::getTransformationMatrixFromRPY(M_PI, 0, 0) * currentVector;
                int xIndex = N / 2 + currentVector.x() * N / sizeVoxelOneDirection;
                int yIndex = N / 2 + currentVector.y() * N / sizeVoxelOneDirection;
                int zIndex = N / 2 + currentVector.z() * N / sizeVoxelOneDirection;
                voxelData1[zIndex + N * yIndex + N * N * xIndex] = 1;
            }


            for (int i = 0; i < cloud->points.size(); i++) {


                Eigen::Vector4d currentVector(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1);
                currentVector =
                        generalHelpfulTools::getTransformationMatrixFromRPY(0.0 / 180.0 * M_PI, 20.0 / 180.0 * M_PI,
                                                                            00.0 / 180.0 * M_PI) * currentVector;

                int xIndex = N / 2 + currentVector.x() * N / sizeVoxelOneDirection;
                int yIndex = N / 2 + currentVector.y() * N / sizeVoxelOneDirection;
                int zIndex = N / 2 + currentVector.z() * N / sizeVoxelOneDirection;
                voxelData2[zIndex + N * yIndex + N * N * xIndex] = 1;
            }

            for (int i = 0; i < cloud->points.size(); i++) {


                Eigen::Vector4d currentVector(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1);
                currentVector =
                        generalHelpfulTools::getTransformationMatrixFromRPY(180.0 / 180.0 * M_PI, 20.0 / 180.0 * M_PI,
                                                                            0.0 / 180.0 * M_PI) * currentVector;


                int xIndex = N / 2 + currentVector.x() * N / sizeVoxelOneDirection;
                int yIndex = N / 2 + currentVector.y() * N / sizeVoxelOneDirection;
                int zIndex = N / 2 + currentVector.z() * N / sizeVoxelOneDirection;
                voxelData2[zIndex + N * yIndex + N * N * xIndex] = 1;
            }
            break;
        case 2:
            /////////////////////// CUUUUUUUUUUUUUUUUUBEEEEEEEEEEEEEEEEE
            int cubeSize = 30;

            for (int i = -cubeSize; i < cubeSize; i++) {
                for (int j = -cubeSize; j < cubeSize; j++) {
                    for (int k = -cubeSize; k < cubeSize; k++) {

                        int xIndex = N / 2 + i;
                        int yIndex = N / 2 + j;
                        int zIndex = N / 2 + k;
                        voxelData1[zIndex + N * yIndex + N * N * xIndex] = 1;
                    }
                }
            }
            for (int i = -cubeSize; i < cubeSize; i++) {
                for (int j = -cubeSize; j < cubeSize; j++) {
                    for (int k = -cubeSize; k < cubeSize; k++) {
                        Eigen::Vector4d currentVector(i, j, k, 1);
                        currentVector = generalHelpfulTools::getTransformationMatrixFromRPY(0.0 / 180.0 * M_PI,
                                                                                            0.0 / 180.0 * M_PI,
                                                                                            0.0 / 180.0 * M_PI) *                                        currentVector;


                        int xIndex = N / 2 + currentVector.x();
                        int yIndex = N / 2 + currentVector.y();
                        int zIndex = N / 2 + currentVector.z();

                        voxelData2[zIndex + N * yIndex + N * N * xIndex] = 1;
                    }
                }
            }
            break;
    }

    softRegistrationClass3D registrationObject(N, N / 2, N / 2, N / 2 - 1);

    registrationObject.sofftRegistrationVoxel3DListOfPossibleRotations(voxelData1, voxelData2, true, true, true,
                                                                       false);


    return (0);
}
