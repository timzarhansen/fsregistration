//
// Created by tim-external on 25.10.23.
//



//
// Created by jurobotics on 13.09.21.

//#include "ekfDVL.h"
#include "rclcpp/rclcpp.hpp"
#include "fs2d/srv/request_list_potential_solution.hpp"
#include "fs2d/srv/request_one_potential_solution.hpp"
#include "softRegistrationClass.h"
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "cv_bridge/cv_bridge.h"

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

class ROSClassRegistrationNode : public rclcpp::Node {
public:
    ROSClassRegistrationNode(int sizeImage) : Node("fs2dregistrationnode"),
                                              scanRegistrationObject(sizeImage, sizeImage / 2, sizeImage / 2,
                                                                     sizeImage / 2 - 1) {
        // for now 256 could be 32/64/128/256/512 More gets complicated to compute
        this->dimensionOfImages = sizeImage;
        this->serviceOnePotentialSolution = this->create_service<fs2d::srv::RequestOnePotentialSolution>(
                "fs2d/registration/one_solution",
                std::bind(&ROSClassRegistrationNode::sendSingleSolutionCallback,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2));
        this->servicelistPotentialSolutions = this->create_service<fs2d::srv::RequestListPotentialSolution>(
                "fs2d/registration/all_solutions",
                std::bind(&ROSClassRegistrationNode::sendAllSolutionsCallback,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2));

    }

private:

    rclcpp::Service<fs2d::srv::RequestOnePotentialSolution>::SharedPtr serviceOnePotentialSolution;
    rclcpp::Service<fs2d::srv::RequestListPotentialSolution>::SharedPtr servicelistPotentialSolutions;

    std::mutex registrationMutex;
    softRegistrationClass scanRegistrationObject;
    int dimensionOfImages;

    bool sendSingleSolutionCallback(const std::shared_ptr<fs2d::srv::RequestOnePotentialSolution::Request> req,
                                    std::shared_ptr<fs2d::srv::RequestOnePotentialSolution::Response> res) {
        std::cout << "getting registration for image:" << std::endl;

        cv_bridge::CvImagePtr cv_ptr1;
        cv_bridge::CvImagePtr cv_ptr2;
        cv::Mat sonarImage1;
        cv::Mat sonarImage2;
        try {
            cv_ptr1 = cv_bridge::toCvCopy(req->sonar_scan_1, req->sonar_scan_1.encoding);
            cv::Mat(cv_ptr1->image).convertTo(sonarImage1, CV_8UC1);
        }
        catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "couldnt convert sonar image 1: %s", e.what());
            return false;
        }

        try {
            cv_ptr2 = cv_bridge::toCvCopy(req->sonar_scan_2, req->sonar_scan_2.encoding);
            cv::Mat(cv_ptr2->image).convertTo(sonarImage2, CV_8UC1);
        }
        catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "couldnt convert sonar image 2: %s", e.what());
            return false;
        }


        tf2::Quaternion orientation;
        tf2::Vector3 position;
        tf2::convert(req->initial_guess.orientation, orientation);
        tf2::convert(req->initial_guess.position, position);


        Eigen::Matrix4d initialGuess = generalHelpfulTools::getTransformationMatrixTF2(position, orientation);
        double *voxelData1;
        double *voxelData2;
        voxelData1 = (double *) malloc(sizeof(double) * this->dimensionOfImages * this->dimensionOfImages);
        voxelData2 = (double *) malloc(sizeof(double) * this->dimensionOfImages * this->dimensionOfImages);

        convertMatToDoubleArray(sonarImage1, voxelData1);
        convertMatToDoubleArray(sonarImage2, voxelData2);

        Eigen::Matrix3d covarianceMatrixResult;
        this->registrationMutex.lock();

        //calculate the registration
        Eigen::Matrix4d resultingRegistrationTransformation = scanRegistrationObject.registrationOfTwoVoxelsSOFFTFast(
                voxelData1,
                voxelData2,
                initialGuess,
                covarianceMatrixResult,
                true, true,
                req->size_of_pixel,
                false,
                false,req->potential_for_necessary_peak);

        this->registrationMutex.unlock();


        generalHelpfulTools::getTF2FromTransformationMatrix(position, orientation, resultingRegistrationTransformation);
        //set result in res
        geometry_msgs::msg::Pose resultingPose;

//        tf2::convert(orientation, resultingPose.orientation);
        resultingPose.position.x = position.x();
        resultingPose.position.y = position.y();
        resultingPose.position.z = position.z();

        resultingPose.orientation.x = orientation.x();
        resultingPose.orientation.y = orientation.y();
        resultingPose.orientation.z = orientation.z();
        resultingPose.orientation.w = orientation.w();

        //tf2::convert(position, resultingPose.position);
        res->potential_solution.rotation_covariance = -1;
        res->potential_solution.resulting_transformation = resultingPose;
        std::cout << resultingRegistrationTransformation << std::endl;
        std::cout << "done registration for image:" << std::endl;
        return true;
    }

    bool sendAllSolutionsCallback(const std::shared_ptr<fs2d::srv::RequestListPotentialSolution::Request> req,
                                  std::shared_ptr<fs2d::srv::RequestListPotentialSolution::Response> res) {
        std::cout << "starting all solution callback" << std::endl;
        cv_bridge::CvImagePtr cv_ptr1;
        cv_bridge::CvImagePtr cv_ptr2;
        cv::Mat sonarImage1;
        cv::Mat sonarImage2;
        try {
            cv_ptr1 = cv_bridge::toCvCopy(req->sonar_scan_1, req->sonar_scan_1.encoding);
            cv::Mat(cv_ptr1->image).convertTo(sonarImage1, CV_8UC1);
        }
        catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "couldnt convert sonar image 1: %s", e.what());
            return false;
        }

        try {
            cv_ptr2 = cv_bridge::toCvCopy(req->sonar_scan_2, req->sonar_scan_2.encoding);
            cv::Mat(cv_ptr2->image).convertTo(sonarImage2, CV_8UC1);
        }
        catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "couldnt convert sonar image 2: %s", e.what());
            return false;
        }


        double *voxelData1;
        double *voxelData2;
        voxelData1 = (double *) malloc(sizeof(double) * this->dimensionOfImages * this->dimensionOfImages);
        voxelData2 = (double *) malloc(sizeof(double) * this->dimensionOfImages * this->dimensionOfImages);

        convertMatToDoubleArray(sonarImage1, voxelData1);
        convertMatToDoubleArray(sonarImage2, voxelData2);

        Eigen::Matrix3d covarianceMatrixResult;
        this->registrationMutex.lock();

        //calculate the registration
        std::vector<transformationPeak> listPotentialSolutions = scanRegistrationObject.registrationOfTwoVoxelsSOFFTAllSoluations(
                voxelData1,
                voxelData2,
                req->size_of_pixel,
                false, false,req->potential_for_necessary_peak);
        this->registrationMutex.unlock();
        std::cout << "req->potential_for_necessary_peak" << std::endl;
        std::cout << req->potential_for_necessary_peak << std::endl;
        for (int i = 0; i < listPotentialSolutions.size(); i++) {
            Eigen::Quaterniond orientationEigen;
            orientationEigen = generalHelpfulTools::getQuaternionFromRPY(0, 0,
                                                                         listPotentialSolutions[i].potentialRotation.angle);
            tf2::Quaternion orientation(orientationEigen.x(), orientationEigen.y(), orientationEigen.z(),
                                        orientationEigen.w());


            for (int j = 0; j < listPotentialSolutions[i].potentialTranslations.size(); j++) {


                tf2::Vector3 position(listPotentialSolutions[i].potentialTranslations[j].translationSI.x(),
                                      listPotentialSolutions[i].potentialTranslations[j].translationSI.y(), 0);

//                generalHelpfulTools::getTF2FromTransformationMatrix(position, orientation, resultingRegistrationTransformation);
                //set result in res
                geometry_msgs::msg::Pose resultingPose;

                tf2::convert(orientation, resultingPose.orientation);
                resultingPose.orientation.x = orientation.x();
                resultingPose.orientation.y = orientation.y();
                resultingPose.orientation.z = orientation.z();
                resultingPose.orientation.w = orientation.w();


                resultingPose.position.x = position.x();
                resultingPose.position.y = position.y();
                resultingPose.position.z = position.z();

                fs2d::msg::PotentialSolution potentialSolutionMSG;
                //tf2::convert(position, resultingPose.position);


                potentialSolutionMSG.resulting_transformation = resultingPose;
                potentialSolutionMSG.transformation_peak_height = listPotentialSolutions[i].potentialTranslations[j].peakHeight;
                potentialSolutionMSG.rotation_peak_height = listPotentialSolutions[i].potentialRotation.peakCorrelation;
                potentialSolutionMSG.persistent_transformation_peak_value = listPotentialSolutions[i].potentialTranslations[j].persistenceValue;

                potentialSolutionMSG.translation_covariance;//2x2 matrix
                potentialSolutionMSG.rotation_covariance = 0.1;//currently missing

                res->list_potential_solutions.push_back(potentialSolutionMSG);
            }
        }


        return true;
    }

};


int main(int argc, char **argv) {

    rclcpp::init(argc, argv);
    //For now we Assume that it is always a 256 image.
    rclcpp::spin(std::make_shared<ROSClassRegistrationNode>(256));


    rclcpp::shutdown();


    return (0);
}
















