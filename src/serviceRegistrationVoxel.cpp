//
// Created by tim-external on 25.10.23.
//



//
// Created by jurobotics on 13.09.21.

//#include "ekfDVL.h"
#include "rclcpp/rclcpp.hpp"
#include "fsregistration/srv/request_list_potential_solution3_d.hpp"
#include "fsregistration/srv/request_one_potential_solution3_d.hpp"
#include "softRegistrationClass3D.h"
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "cv_bridge/cv_bridge.h"
#include "fsregistration/srv/detail/request_list_potential_solution3_d__struct.hpp"

//#define DEBUG_MODE false

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
    ROSClassRegistrationNode() : Node("fs3dregistrationnode") {

        // for now 256 could be 32/64/128/256/512 More gets complicated to compute
        this->potentialVoxelSizes = std::vector<int>{16, 32, 64, 128, 256};
//        this->dimensionOfImages = sizeImage;
//        this->serviceOnePotentialSolution = this->create_service<fsregistration::srv::RequestOnePotentialSolution3D>(
//                "fsregistration/registration/one_solution",
//                std::bind(&ROSClassRegistrationNode::sendSingleSolutionCallback,
//                          this,
//                          std::placeholders::_1,
//                          std::placeholders::_2));
        this->servicelistPotentialSolutions = this->create_service<fsregistration::srv::RequestListPotentialSolution3D>(
                "fs3D/registration/all_solutions",
                std::bind(&ROSClassRegistrationNode::sendAllSolutionsCallback,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2));

    }

private:

//    rclcpp::Service<fsregistration::srv::RequestOnePotentialSolution3D>::SharedPtr serviceOnePotentialSolution;
    rclcpp::Service<fsregistration::srv::RequestListPotentialSolution3D>::SharedPtr servicelistPotentialSolutions;
    std::vector<int> potentialVoxelSizes;
    std::mutex registrationMutex;
//    softRegistrationClass scanRegistrationObject;
//    int dimensionOfImages;
    std::vector<softRegistrationClass3D *> softRegistrationObjectList;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    int getIndexOfRegistration(int sizeOfTheVoxel) {
        if (std::find(this->potentialVoxelSizes.begin(), this->potentialVoxelSizes.end(), sizeOfTheVoxel) ==
            this->potentialVoxelSizes.end()) {
            std::cout << sizeOfTheVoxel << std::endl;
            std::cout << "Wrong size of Voxels " << std::endl;
            return -1;
        }

        if (softRegistrationObjectList.empty()) {
            //Create correct image size
            this->softRegistrationObjectList.push_back(
                    new softRegistrationClass3D(sizeOfTheVoxel, sizeOfTheVoxel / 2, sizeOfTheVoxel / 2,
                                                sizeOfTheVoxel / 2 - 1));
        }
        bool alreadyHaveCorrectRegistration = false;
        int positionOfCorrect = 0;
        for (int i = 0; i < this->softRegistrationObjectList.size(); i++) {
            if (this->softRegistrationObjectList[i]->getSizeOfRegistration() == sizeOfTheVoxel) {
                positionOfCorrect = i;
                alreadyHaveCorrectRegistration = true;
                break;
            }
        }

        if (!alreadyHaveCorrectRegistration) {
            //Create correct image size
            this->softRegistrationObjectList.push_back(
                    new softRegistrationClass3D(sizeOfTheVoxel, sizeOfTheVoxel / 2, sizeOfTheVoxel / 2,
                                                sizeOfTheVoxel / 2 - 1));

            positionOfCorrect = this->softRegistrationObjectList.size() - 1;
        }
        return positionOfCorrect;
    }

//    bool
//    sendSingleSolutionCallback(const std::shared_ptr<fsregistration::srv::RequestOnePotentialSolution3D::Request> req,
//                               std::shared_ptr<fsregistration::srv::RequestOnePotentialSolution3D::Response> res) {
//
//        int sizeOfTheVoxel = req->size_voxel;
//
//        int positionOfCorrectRegistration = this->getIndexOfRegistration(sizeOfTheVoxel);
//        if (positionOfCorrectRegistration < 0) {
//            return false;
//        }
//
//
//        std::cout << "getting registration for image: " << std::endl;
//
//        tf2::Quaternion orientation;
//        tf2::Vector3 position;
//        tf2::convert(req->initial_guess.orientation, orientation);
//        tf2::convert(req->initial_guess.position, position);
//
//
//        Eigen::Matrix4d initialGuess = generalHelpfulTools::getTransformationMatrixTF2(position, orientation);
//        double *voxelData1;
//        double *voxelData2;
//        voxelData1 = (double *) malloc(sizeof(double) * sizeOfTheVoxel * sizeOfTheVoxel);
//        voxelData2 = (double *) malloc(sizeof(double) * sizeOfTheVoxel * sizeOfTheVoxel);
//
//        for (int i = 0; i < sizeOfTheVoxel * sizeOfTheVoxel; i++) {
//            voxelData1[i] = req->sonar_scan_1[i];
//            voxelData2[i] = req->sonar_scan_2[i];
//        }
//
////        cv::Mat magTMP1(this->dimensionOfImages, this->dimensionOfImages, CV_64F, voxelData1);
////        cv::Mat magTMP2(this->dimensionOfImages, this->dimensionOfImages, CV_64F, voxelData2);
////        cv::imshow("sonar1", magTMP1);
////        cv::imshow("sonar2", magTMP2);
////        int k = cv::waitKey(0); // Wait for a keystroke in the window
//
//
//
////        convertMatToDoubleArray(sonarImage1, voxelData1);
////        convertMatToDoubleArray(sonarImage2, voxelData2);
//        // computation time measurement
//        Eigen::Matrix3d covarianceMatrixResult;
//        this->registrationMutex.lock();
//        std::chrono::steady_clock::time_point begin;
//
//        begin = std::chrono::steady_clock::now();
//
//
//
//
//        //calculate the registration
//        Eigen::Matrix4d resultingRegistrationTransformation = softRegistrationObjectList[positionOfCorrectRegistration]->registrationOfTwoVoxelsSOFFTFast(
//                voxelData1,
//                voxelData2,
//                initialGuess,
//                covarianceMatrixResult,
//                true, true,
//                req->size_of_voxel,
//                false,
//                DEBUG_MODE, req->potential_for_necessary_peak);
//        std::chrono::steady_clock::time_point end;
//        end = std::chrono::steady_clock::now();
//
//        double timeToCalculate = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
//        this->registrationMutex.unlock();
//        free(voxelData1);
//        free(voxelData2);
//
//        generalHelpfulTools::getTF2FromTransformationMatrix(position, orientation, resultingRegistrationTransformation);
//        //set result in res
//        geometry_msgs::msg::Pose resultingPose;
//
////        tf2::convert(orientation, resultingPose.orientation);
//        resultingPose.position.x = position.x();
//        resultingPose.position.y = position.y();
//        resultingPose.position.z = position.z();
//
//        resultingPose.orientation.x = orientation.x();
//        resultingPose.orientation.y = orientation.y();
//        resultingPose.orientation.z = orientation.z();
//        resultingPose.orientation.w = orientation.w();
//        Eigen::Matrix2d covarianceTranslation = covarianceMatrixResult.block<2, 2>(0, 0);
//        //tf2::convert(position, resultingPose.position);
////        res->potential_solution.rotation_covariance = covarianceMatrixResult(2, 2);
//        res->potential_solution.rotation_covariance[0] = 0.1;//currently missing
//        res->potential_solution.rotation_covariance[1] = 0.1;//currently missing
//        res->potential_solution.rotation_covariance[2] = 0.1;//currently missing
//        res->potential_solution.resulting_transformation = resultingPose;
//        res->potential_solution.translation_covariance[0] = covarianceTranslation(0);
//        res->potential_solution.translation_covariance[1] = covarianceTranslation(1);
//        res->potential_solution.translation_covariance[2] = covarianceTranslation(2);
//        res->potential_solution.translation_covariance[3] = covarianceTranslation(3);
//        res->potential_solution.time_to_calculate = timeToCalculate;
//        //printing the results
//        std::cout << initialGuess << std::endl;
//        std::cout << resultingRegistrationTransformation << std::endl;
//        std::cout << "done registration for image:" << std::endl;
//
//        return true;
//    }

    bool
    sendAllSolutionsCallback(const std::shared_ptr<fsregistration::srv::RequestListPotentialSolution3D::Request> req,
                             std::shared_ptr<fsregistration::srv::RequestListPotentialSolution3D::Response> res) {
        std::cout << "starting all solution callback" << std::endl;
        int dimensionSize = req->dimension_size;

        int positionOfCorrectRegistration = this->getIndexOfRegistration(dimensionSize);
        if (positionOfCorrectRegistration < 0) {
            return false;
        }


        double *voxelData1;
        double *voxelData2;
        voxelData1 = (double *) malloc(sizeof(double) * dimensionSize * dimensionSize * dimensionSize);
        voxelData2 = (double *) malloc(sizeof(double) * dimensionSize * dimensionSize * dimensionSize);

//        convertMatToDoubleArray(sonarImage1, voxelData1);
//        convertMatToDoubleArray(sonarImage2, voxelData2);
        for (int i = 0; i < dimensionSize * dimensionSize * dimensionSize; i++) {
            voxelData1[i] = req->sonar_scan_1[i];
            voxelData2[i] = req->sonar_scan_2[i];
        }
        Eigen::Matrix3d covarianceMatrixResult;
        this->registrationMutex.lock();
        if (req->timing_computation_duration) {
            begin = std::chrono::steady_clock::now();
        }




        //calculate the registration
        std::vector<transformationPeakfs3D> listPotentialSolutions = softRegistrationObjectList[positionOfCorrectRegistration]->sofftRegistrationVoxel3DListOfPossibleTransformations(
                voxelData1,
                voxelData2, req->debug, req->use_clahe, req->timing_computation_duration, req->size_of_voxel);
        double timeToCalculate = -1;
        if (req->timing_computation_duration) {
            end = std::chrono::steady_clock::now();
            timeToCalculate = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        }

        this->registrationMutex.unlock();
        free(voxelData1);
        free(voxelData2);

        std::cout << "req->potential_for_necessary_peak" << std::endl;
        std::cout << req->potential_for_necessary_peak << std::endl;
        for (int i = 0; i < listPotentialSolutions.size(); i++) {
//            Eigen::Quaterniond orientationEigen();
//            orientationEigen = generalHelpfulTools::getQuaternionFromRPY(0, 0,
//                                                                         );
            tf2::Quaternion orientation(listPotentialSolutions[i].potentialRotation.x,
                                        listPotentialSolutions[i].potentialRotation.y,
                                        listPotentialSolutions[i].potentialRotation.z,
                                        listPotentialSolutions[i].potentialRotation.w);


            for (int j = 0; j < listPotentialSolutions[i].potentialTranslations.size(); j++) {


                tf2::Vector3 position(listPotentialSolutions[i].potentialTranslations[j].xTranslation,
                                      listPotentialSolutions[i].potentialTranslations[j].yTranslation,
                                      listPotentialSolutions[i].potentialTranslations[j].zTranslation);

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

                fsregistration::msg::PotentialSolution3D potentialSolutionMSG;
                //tf2::convert(position, resultingPose.position);


                potentialSolutionMSG.resulting_transformation = resultingPose;
                potentialSolutionMSG.transformation_peak_height = listPotentialSolutions[i].potentialTranslations[j].levelPotential;
                potentialSolutionMSG.rotation_peak_height = listPotentialSolutions[i].potentialRotation.correlationHeight;
                potentialSolutionMSG.persistent_transformation_peak_value = listPotentialSolutions[i].potentialTranslations[j].persistence;

                potentialSolutionMSG.translation_covariance;//2x2 matrix
                potentialSolutionMSG.rotation_covariance[0] = 0.1;//currently missing
                potentialSolutionMSG.rotation_covariance[1] = 0.1;//currently missing
                potentialSolutionMSG.rotation_covariance[2] = 0.1;//currently missing
                potentialSolutionMSG.time_to_calculate = timeToCalculate;
                res->list_potential_solutions.push_back(potentialSolutionMSG);
            }
        }


        return true;
    }

};


int main(int argc, char **argv) {

    rclcpp::init(argc, argv);
    //For now we Assume that it is always a 256 image.
    rclcpp::spin(std::make_shared<ROSClassRegistrationNode>());


    rclcpp::shutdown();


    return (0);
}















