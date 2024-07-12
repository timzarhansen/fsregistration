//
// Created by tim-external on 27.10.23.
//
#include <iostream>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "fsregistration/srv/request_list_potential_solution.hpp"
#include "fsregistration/srv/request_one_potential_solution.hpp"
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include "cv_bridge/cv_bridge.h"
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "generalHelpfulTools.h"


fsregistration::srv::RequestOnePotentialSolution::Request
createOnePotentialSolutionRequest(cv::Mat sonar1, cv::Mat sonar2, Eigen::Matrix4d initialGuess, double sizeOfPixel,
                                  double potentialForNecessaryPeak) {


    cv_bridge::CvImagePtr cv_ptr;

    sensor_msgs::msg::Image::SharedPtr sonar_msg_1 =
            cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", sonar1)
                    .toImageMsg();
    sensor_msgs::msg::Image::SharedPtr sonar_msg_2 =
            cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", sonar2)
                    .toImageMsg();
    geometry_msgs::msg::Pose poseMsg;

    Eigen::Quaterniond quaternionInitGuess;
    Eigen::Vector3d translationInitGuess;
    generalHelpfulTools::splitTransformationMatrixToQuadAndTrans(translationInitGuess, quaternionInitGuess,
                                                                 initialGuess);

    poseMsg.position.x = translationInitGuess.x();
    poseMsg.position.y = translationInitGuess.y();
    poseMsg.position.z = translationInitGuess.z();

    poseMsg.orientation.x = quaternionInitGuess.x();
    poseMsg.orientation.y = quaternionInitGuess.y();
    poseMsg.orientation.z = quaternionInitGuess.z();
    poseMsg.orientation.w = quaternionInitGuess.w();


    fsregistration::srv::RequestOnePotentialSolution::Request request;
    request.sonar_scan_1 = *sonar_msg_1;
    request.sonar_scan_2 = *sonar_msg_2;
    request.size_of_pixel = 1;
    request.potential_for_necessary_peak = 0.1;
    request.initial_guess = poseMsg;
    return request;
}

fsregistration::srv::RequestListPotentialSolution::Request
createAllPotentialSolutionRequest(cv::Mat sonar1, cv::Mat sonar2, double sizeOfPixel,
                                  double potentialForNecessaryPeak) {


    cv_bridge::CvImagePtr cv_ptr;

    sensor_msgs::msg::Image::SharedPtr sonar_msg_1 =
            cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", sonar1)
                    .toImageMsg();
    sensor_msgs::msg::Image::SharedPtr sonar_msg_2 =
            cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", sonar2)
                    .toImageMsg();


    fsregistration::srv::RequestListPotentialSolution::Request request;
    request.sonar_scan_1 = *sonar_msg_1;
    request.sonar_scan_2 = *sonar_msg_2;
    request.size_of_pixel = 1;
    request.potential_for_necessary_peak = 0.1;

    return request;
}

int main(int argc, char **argv) {
    // input needs to be two scans as voxelData
    rclcpp::init(argc, argv);


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


    cv::Mat img1 = cv::imread(
            all_args[0],
            cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(
            all_args[1],
            cv::IMREAD_GRAYSCALE);


    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("client_registration");

    rclcpp::Client<fsregistration::srv::RequestOnePotentialSolution>::SharedPtr onePotentialClient =
            node->create_client<fsregistration::srv::RequestOnePotentialSolution>("fsregistration/registration/one_solution");

    rclcpp::Client<fsregistration::srv::RequestListPotentialSolution>::SharedPtr client2 =
            node->create_client<fsregistration::srv::RequestListPotentialSolution>("fsregistration/registration/all_solutions");


    // waiting until clients are connecting
    while (!onePotentialClient->wait_for_service(std::chrono::duration<float>(0.1))) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
            return 0;
        }
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "service not available, waiting again...");
    }
    while (!client2->wait_for_service(std::chrono::duration<float>(0.1))) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
            return 0;
        }
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "service not available, waiting again...");
    }


    auto request = std::make_shared<fsregistration::srv::RequestOnePotentialSolution::Request>(
            createOnePotentialSolutionRequest(img1, img2, Eigen::Matrix4d::Identity(), 1, 0.1));

//    auto request = std::make_shared<fsregistration::srv::RequestOnePotentialSolution::Request>();




    auto future = onePotentialClient->async_send_request(request);

    if (rclcpp::spin_until_future_complete(node, future) ==
        rclcpp::FutureReturnCode::SUCCESS) {
        // Wait for the result.
        try {
            auto response = future.get();
            tf2::Quaternion orientation;
            tf2::Vector3 position;
            tf2::convert(response->potential_solution.resulting_transformation.orientation, orientation);
            tf2::convert(response->potential_solution.resulting_transformation.position, position);

            Eigen::Matrix4d resultingTransformation = generalHelpfulTools::getTransformationMatrixTF2(position,
                                                                                                      orientation);
            std::cout << resultingTransformation << std::endl;

        }
        catch (const std::exception &e) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Service call failed.");
        }
    }
    std::cout << "first call done" << std::endl;

////////////////////////////////////////////////////////////////////////////////////////////////// NOW ALL SOLUTIONS //////////////////////////////////////////////////////////////////////////////////////////////////



    auto request2 = std::make_shared<fsregistration::srv::RequestListPotentialSolution::Request>(
            createAllPotentialSolutionRequest(img1, img2, 1, 0.1));


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
}





