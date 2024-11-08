//
// Created by tim-external on 27.10.23.
//
#include <iostream>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "fsregistration/srv/request_list_potential_solution3_d.hpp"
#include "fsregistration/srv/request_one_potential_solution3_d.hpp"
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include "cv_bridge/cv_bridge.h"
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "generalHelpfulTools.h"
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/common/norms.h>
#include "fsregistration/srv/detail/request_list_potential_solution3_d__struct.hpp"


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


fsregistration::srv::RequestListPotentialSolution3D::Request
createAllPotentialSolutionRequest(double* sonar1, double* sonar2, double sizeOfVoxel,
                                  double potentialForNecessaryPeak,int N) {

    fsregistration::srv::RequestListPotentialSolution3D::Request request;
    for(int i  = 0  ; i<N*N*N;i++){
        request.sonar_scan_1.push_back(sonar1[i]);
        request.sonar_scan_2.push_back(sonar2[i]);
    }
    request.timing_computation_duration = true;
    request.debug=false;
    request.use_clahe=true;
    request.dimension_size=N;
    request.size_of_voxel=sizeOfVoxel;
    // request.potential_for_necessary_peak = potentialForNecessaryPeak;

    return request;
}

int main(int argc, char **argv) {
    // input needs to be two scans as voxelData
    rclcpp::init(argc, argv);


    std::string current_exec_name = argv[0]; // Name of the current exec program
    std::vector<std::string> all_args;

    if (argc > 0) {
        //std::cout << "temp1" << std::endl;
        all_args.assign(argv + 1, argv + argc);
        //std::cout << "12"<< all_args[1]<<std::endl;
    } else {
        std::cout << "no arguments given" << std::endl;
        exit(-1);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud <pcl::PointXYZ>);
    pcl::PLYReader Reader;
    Reader.read("exampleData/dragon_recon/dragon_vrip.ply", *cloud);

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
    int N = 64;
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
                Eigen::Vector4d shiftVector(8, 3, -11, 0);// in pixel
//                Eigen::Vector4d shiftVector(0, 0, 0, 0);// in pixel

                currentVector += shiftVector * sizeVoxelOneDirection / N;
//                currentVector =
//                        generalHelpfulTools::getTransformationMatrixFromRPY(10.0 / 180.0 * M_PI, -0.0 / 180.0 * M_PI,
//                                                                            0.0 / 180.0 * M_PI) * currentVector;
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
                                                                                            0.0 / 180.0 * M_PI) *currentVector;


                        int xIndex = N / 2 + currentVector.x();
                        int yIndex = N / 2 + currentVector.y();
                        int zIndex = N / 2 + currentVector.z();

                        voxelData2[zIndex + N * yIndex + N * N * xIndex] = 1;
                    }
                }
            }
            break;
    }




    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("client_registration");


    rclcpp::Client<fsregistration::srv::RequestListPotentialSolution3D>::SharedPtr client2 =
            node->create_client<fsregistration::srv::RequestListPotentialSolution3D>("fs3D/registration/all_solutions");


    // waiting until clients are connecting
    while (!client2->wait_for_service(std::chrono::duration<float>(0.1))) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
            return 0;
        }
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "service not available, waiting again...");
    }


////////////////////////////////////////////////////////////////////////////////////////////////// NOW ALL SOLUTIONS //////////////////////////////////////////////////////////////////////////////////////////////////



    auto request2 = std::make_shared<fsregistration::srv::RequestListPotentialSolution3D::Request>(
            createAllPotentialSolutionRequest(voxelData1, voxelData2, 1, 0.1,N));


    auto future2 = client2->async_send_request(request2);

    if (rclcpp::spin_until_future_complete(node, future2) ==
        rclcpp::FutureReturnCode::SUCCESS) {
        // Wait for the result.
        try {
            auto response2 = future2.get();
            tf2::Quaternion orientation;
            tf2::Vector3 position;
            for (int i = 0; i < response2->list_potential_solutions.size(); i++) {

                tf2::convert(response2->list_potential_solutions[i].resulting_transformation.orientation, orientation);
                tf2::convert(response2->list_potential_solutions[i].resulting_transformation.position, position);
                Eigen::Matrix4d resultingTransformation = generalHelpfulTools::getTransformationMatrixTF2(position,
                                                                                                          orientation);
                std::cout << resultingTransformation << std::endl;
                std::cout << response2->list_potential_solutions[i].persistent_transformation_peak_value << std::endl;
                std::cout << response2->list_potential_solutions[i].rotation_peak_height << std::endl;
                std::cout << response2->list_potential_solutions[i].transformation_peak_height << std::endl;

            }
        }
        catch (const std::exception &e) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Service call failed.");
        }
    }
    std::cout << "second call done" << std::endl;
    free(voxelData1);
    free(voxelData2);
    exit(161);
}





