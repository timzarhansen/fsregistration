//
// Created by tim-external on 25.10.23.
//



//
// Created by jurobotics on 13.09.21.

//#include "ekfDVL.h"
#include "rclcpp/rclcpp.hpp"
#include "fsregistration/srv/request_list_potential_solution2_d.hpp"
#include "fsregistration/srv/request_one_potential_solution2_d.hpp"
#include "softRegistrationClass.h"
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "cv_bridge/cv_bridge.h"

#define DEBUG_MODE false
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
    ROSClassRegistrationNode() : Node("fsregistrationregistrationnode") {

        // for now 256 could be 32/64/128/256/512 More gets complicated to compute
        this->potentialVoxelSizes = std::vector<int>{32, 64, 128, 256, 512 };
//        this->dimensionOfImages = sizeImage;
        this->serviceOnePotentialSolution = this->create_service<fsregistration::srv::RequestOnePotentialSolution2D>(
                "fs2d/registration/one_solution",
                std::bind(&ROSClassRegistrationNode::sendSingleSolutionCallback,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2));
        this->servicelistPotentialSolutions = this->create_service<fsregistration::srv::RequestListPotentialSolution2D>(
                "fs2d/registration/all_solutions",
                std::bind(&ROSClassRegistrationNode::sendAllSolutionsCallback,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2));

    }

private:

    rclcpp::Service<fsregistration::srv::RequestOnePotentialSolution2D>::SharedPtr serviceOnePotentialSolution;
    rclcpp::Service<fsregistration::srv::RequestListPotentialSolution2D>::SharedPtr servicelistPotentialSolutions;
    std::vector<int> potentialVoxelSizes;
    std::mutex registrationMutex;
//    softRegistrationClass scanRegistrationObject;
//    int dimensionOfImages;
    std::vector<softRegistrationClass *> softRegistrationObjectList;

    int getIndexOfRegistration(int sizeOfTheImage){
        if (std::find(this->potentialVoxelSizes.begin(), this->potentialVoxelSizes.end(), sizeOfTheImage) == this->potentialVoxelSizes.end() ){
            std::cout << sizeOfTheImage << std::endl;
            std::cout << "Wrong size of image " << std::endl;
            return -1;
        }

        if(softRegistrationObjectList.empty()){
            //Create correct image size
            this->softRegistrationObjectList.push_back(
                    new softRegistrationClass(sizeOfTheImage, sizeOfTheImage / 2, sizeOfTheImage / 2, sizeOfTheImage / 2 - 1));
        }
        bool alreadyHaveCorrectRegistration = false;
        int positionOfCorrect = 0;
        for(int i  = 0 ; i<this->softRegistrationObjectList.size();i++){
            if(this->softRegistrationObjectList[i]->getSizeOfRegistration() == sizeOfTheImage){
                positionOfCorrect = i;
                alreadyHaveCorrectRegistration = true;
                break;
            }
        }

        if(!alreadyHaveCorrectRegistration){
            //Create correct image size
            this->softRegistrationObjectList.push_back(
                    new softRegistrationClass(sizeOfTheImage, sizeOfTheImage / 2, sizeOfTheImage / 2, sizeOfTheImage / 2 - 1));

            positionOfCorrect = this->softRegistrationObjectList.size()-1;
        }
        return positionOfCorrect;
    }

    bool sendSingleSolutionCallback(const std::shared_ptr<fsregistration::srv::RequestOnePotentialSolution2D::Request> req,
                                    std::shared_ptr<fsregistration::srv::RequestOnePotentialSolution2D::Response> res) {

        int sizeOfTheImage = req->size_image;

        int positionOfCorrectRegistration = this->getIndexOfRegistration(sizeOfTheImage);
        if(positionOfCorrectRegistration <0){
            return false;
        }



        std::cout << "getting registration for image: " << std::endl;

        tf2::Quaternion orientation;
        tf2::Vector3 position;
        tf2::convert(req->initial_guess.orientation, orientation);
        tf2::convert(req->initial_guess.position, position);


        Eigen::Matrix4d initialGuess = generalHelpfulTools::getTransformationMatrixTF2(position, orientation);
        double *voxelData1;
        double *voxelData2;
        voxelData1 = (double *) malloc(sizeof(double) * sizeOfTheImage*sizeOfTheImage);
        voxelData2 = (double *) malloc(sizeof(double) * sizeOfTheImage*sizeOfTheImage);

        for(int i  = 0 ; i<sizeOfTheImage*sizeOfTheImage ; i++){
            voxelData1[i] = req->sonar_scan_1[i];
            voxelData2[i] = req->sonar_scan_2[i];
        }

//        cv::Mat magTMP1(this->dimensionOfImages, this->dimensionOfImages, CV_64F, voxelData1);
//        cv::Mat magTMP2(this->dimensionOfImages, this->dimensionOfImages, CV_64F, voxelData2);
//        cv::imshow("sonar1", magTMP1);
//        cv::imshow("sonar2", magTMP2);
//        int k = cv::waitKey(0); // Wait for a keystroke in the window



//        convertMatToDoubleArray(sonarImage1, voxelData1);
//        convertMatToDoubleArray(sonarImage2, voxelData2);
        // computation time measurement
        Eigen::Matrix3d covarianceMatrixResult;
        this->registrationMutex.lock();
        std::chrono::steady_clock::time_point begin;

        begin = std::chrono::steady_clock::now();




        //calculate the registration
        Eigen::Matrix4d resultingRegistrationTransformation = softRegistrationObjectList[positionOfCorrectRegistration]->registrationOfTwoVoxelsSOFFTFast(
                voxelData1,
                voxelData2,
                initialGuess,
                covarianceMatrixResult,
                true, true,
                req->size_of_pixel,
                false,
                DEBUG_MODE,req->potential_for_necessary_peak);
        std::chrono::steady_clock::time_point end;
        end = std::chrono::steady_clock::now();

        double timeToCalculate = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        this->registrationMutex.unlock();
        free(voxelData1);
        free(voxelData2);

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
        Eigen::Matrix2d covarianceTranslation = covarianceMatrixResult.block<2,2>(0,0);
        //tf2::convert(position, resultingPose.position);
        res->potential_solution.rotation_covariance = covarianceMatrixResult(2, 2);
        res->potential_solution.resulting_transformation = resultingPose;
        res->potential_solution.translation_covariance[0] = covarianceTranslation(0);
        res->potential_solution.translation_covariance[1] = covarianceTranslation(1);
        res->potential_solution.translation_covariance[2] = covarianceTranslation(2);
        res->potential_solution.translation_covariance[3] = covarianceTranslation(3);
        res->potential_solution.time_to_calculate = timeToCalculate;
        //printing the results
        std::cout << initialGuess << std::endl;
        std::cout << resultingRegistrationTransformation << std::endl;
        std::cout << "done registration for image:" << std::endl;

        return true;
    }

    bool sendAllSolutionsCallback(const std::shared_ptr<fsregistration::srv::RequestListPotentialSolution2D::Request> req,
                                  std::shared_ptr<fsregistration::srv::RequestListPotentialSolution2D::Response> res) {
        std::cout << "starting all solution callback" << std::endl;
        int sizeOfTheImage = req->size_image;

        int positionOfCorrectRegistration = this->getIndexOfRegistration(sizeOfTheImage);
        if(positionOfCorrectRegistration <0){
            return false;
        }


        double *voxelData1;
        double *voxelData2;
        voxelData1 = (double *) malloc(sizeof(double) * sizeOfTheImage*sizeOfTheImage);
        voxelData2 = (double *) malloc(sizeof(double) * sizeOfTheImage*sizeOfTheImage);

//        convertMatToDoubleArray(sonarImage1, voxelData1);
//        convertMatToDoubleArray(sonarImage2, voxelData2);
        for(int i  = 0 ; i< sizeOfTheImage*sizeOfTheImage ; i++){
            voxelData1[i] = req->sonar_scan_1[i];
            voxelData2[i] = req->sonar_scan_2[i];
        }
        Eigen::Matrix3d covarianceMatrixResult;
        this->registrationMutex.lock();
        std::chrono::steady_clock::time_point begin;

        begin = std::chrono::steady_clock::now();

        //calculate the registration
        std::vector<transformationPeakfs2D> listPotentialSolutions = softRegistrationObjectList[positionOfCorrectRegistration]->registrationOfTwoVoxelsSOFFTAllSoluations(
                voxelData1,
                voxelData2,
                req->size_of_pixel,
                false, DEBUG_MODE,req->potential_for_necessary_peak);

        std::chrono::steady_clock::time_point end;
        end = std::chrono::steady_clock::now();

        double timeToCalculate = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        this->registrationMutex.unlock();
        free(voxelData1);
        free(voxelData2);

        std::cout << "req->potential_for_necessary_peak" << std::endl;
        std::cout << req->potential_for_necessary_peak << std::endl;
        std::cout << "NumberOfSolutions Rotation:" << std::endl;
        std::cout << listPotentialSolutions.size() << std::endl;
        for (int i = 0; i < listPotentialSolutions.size(); i++) {
            Eigen::Quaterniond orientationEigen;
            orientationEigen = generalHelpfulTools::getQuaternionFromRPY(0, 0,
                                                                         listPotentialSolutions[i].potentialRotation.angle);
            tf2::Quaternion orientation(orientationEigen.x(), orientationEigen.y(), orientationEigen.z(),
                                        orientationEigen.w());

            std::cout << "NumberOfSolutions Translation:" << std::endl;
            std::cout << listPotentialSolutions[i].potentialTranslations.size() << std::endl;
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

                fsregistration::msg::PotentialSolution2D potentialSolutionMSG;
                //tf2::convert(position, resultingPose.position);


                potentialSolutionMSG.resulting_transformation = resultingPose;
                potentialSolutionMSG.transformation_peak_height = listPotentialSolutions[i].potentialTranslations[j].peakHeight;
                potentialSolutionMSG.rotation_peak_height = listPotentialSolutions[i].potentialRotation.peakCorrelation;
                potentialSolutionMSG.persistent_transformation_peak_value = listPotentialSolutions[i].potentialTranslations[j].persistenceValue;

                potentialSolutionMSG.translation_covariance;//2x2 matrix
                potentialSolutionMSG.rotation_covariance = 0.1;//currently missing
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
















