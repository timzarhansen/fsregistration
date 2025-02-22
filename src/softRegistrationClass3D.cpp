//
// Created by aya on 08.12.23.
//

#include "softRegistrationClass3D.h"

//bool compareTwoAngleCorrelation3D(translationPeak3D i1, translationPeak3D i2) {
//    return (i1.angle < i2.angle);
//}

double thetaIncrement3D(double index, int bandwidth) {
    return M_PI * index / (2.0 * (bandwidth - 1));
}

double phiIncrement3D(double index, int bandwidth) {
    return M_PI * index / bandwidth;
}

//double radiusIncrement(int index, int maximum, int minimum, int numberOfIncrements) {
//    return ((double) index / (double) numberOfIncrements) * ((double) (maximum - minimum)) + (double) minimum;
//}

double angleDifference3D(double angle1, double angle2) {
    //gives angle 1 - angle 2
    return atan2(sin(angle1 - angle2), cos(angle1 - angle2));
}

double
softRegistrationClass3D::getSpectrumFromVoxelData3D(const double voxelData[], double magnitude[], double phase[],
                                                    bool gaussianBlur) {
    double *voxelDataTMP;
    voxelDataTMP = (double *) malloc(sizeof(double) * N * N * N);
    for (int i = 0; i < this->N * this->N * this->N; i++) {
        voxelDataTMP[i] = voxelData[i];
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                int index = generalHelpfulTools::index3D(i, j, k, N);
                inputSpacialData[index][0] = voxelDataTMP[index]; // real part
                inputSpacialData[index][1] = 0; // imaginary part
            }
        }
    }

    fftw_execute(planVoxelToFourier3D);


    double maximumMagnitude = 0;


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                int index = generalHelpfulTools::index3D(i, j, k, N);
                magnitude[index] = sqrt(
                    spectrumOut[index][0] * spectrumOut[index][0] + spectrumOut[index][1] * spectrumOut[index][1]);
                if (maximumMagnitude < magnitude[index]) {
                    maximumMagnitude = magnitude[index];
                }
                phase[index] = atan2(spectrumOut[index][1], spectrumOut[index][0]);
            }
        }
    }


    free(voxelDataTMP);
    return maximumMagnitude;
}

double
softRegistrationClass3D::getSpectrumFromVoxelData3DCorrelation(const double voxelData[], double magnitude[],
                                                               double phase[],
                                                               bool gaussianBlur) {
    //    double *voxelDataTMP;
    //    voxelDataTMP = (double *) malloc(sizeof(double) * N * N * N);
    //    for (int i = 0; i < this->N * this->N * this->N; i++) {
    //        voxelDataTMP[i] = voxelData[i];
    //    }

    for (int i = 0; i < this->correlationN * this->correlationN * this->correlationN; i++) {
        inputSpacialDataCorrelation[i][0] = 0;
        inputSpacialDataCorrelation[i][1] = 0;
    }

    for (int i = 0; i < this->N; i++) {
        for (int j = 0; j < this->N; j++) {
            for (int k = 0; k < this->N; k++) {
                int indexN = generalHelpfulTools::index3D(i, j, k, this->N);
                int indexNCorrelation = generalHelpfulTools::index3D(i + (int) (this->correlationN / 4),
                                                                     j + (int) (this->correlationN / 4),
                                                                     k + (int) (this->correlationN / 4),
                                                                     this->correlationN);
                inputSpacialDataCorrelation[indexNCorrelation][0] = voxelData[indexN]; // real part
                inputSpacialDataCorrelation[indexNCorrelation][1] = 0; // imaginary part
            }
        }
    }

    fftw_execute(planVoxelToFourier3DCorrelation);


    double maximumMagnitude = 0;


    for (int i = 0; i < this->correlationN; i++) {
        for (int j = 0; j < this->correlationN; j++) {
            for (int k = 0; k < this->correlationN; k++) {
                int index = generalHelpfulTools::index3D(i, j, k, this->correlationN);
                magnitude[index] = sqrt(
                    spectrumOutCorrelation[index][0] * spectrumOutCorrelation[index][0] + spectrumOutCorrelation[index][
                        1] * spectrumOutCorrelation[index][1]);
                if (maximumMagnitude < magnitude[index]) {
                    maximumMagnitude = magnitude[index];
                }
                phase[index] = atan2(spectrumOutCorrelation[index][1], spectrumOutCorrelation[index][0]);
            }
        }
    }


    //    free(voxelDataTMP);
    return maximumMagnitude;
}
transformationPeakfs3D
softRegistrationClass3D::sofftRegistrationVoxel3DOneSolution(double voxelData1Input[], double voxelData2Input[], tf2::Quaternion initGuessOrientation,
    tf2::Vector3 initGuessPosition,
                                                      bool debug, bool useClahe,
                                                      bool timeStuff , double sizeVoxel ,
                                                      double r_min ,
                                                      double r_max,
                                                      double level_potential_rotation,
                                                      double level_potential_translation,
                                                      bool set_r_manual, int normalization) {
std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    std::chrono::duration<double, std::milli> diff{};

    if (timeStuff) {
        begin = std::chrono::steady_clock::now();
    }


    double maximumScan1Magnitude = this->getSpectrumFromVoxelData3D(voxelData1Input, this->magnitude1,
                                                                    this->phase1, false);
    double maximumScan2Magnitude = this->getSpectrumFromVoxelData3D(voxelData2Input, this->magnitude2,
                                                                    this->phase2, false);

    if (timeStuff) {
        end = std::chrono::steady_clock::now();
        diff = end - begin;
        std::cout << "computation 3D Spectrum: " << diff.count() << std::endl;
        begin = std::chrono::steady_clock::now();
    }


    if (debug) {
        std::ofstream myFile1, myFile2, myFile3, myFile4, myFile5, myFile6;
        myFile1.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/magnitudeFFTW1.csv");
        myFile2.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/phaseFFTW1.csv");
        myFile3.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/voxelDataFFTW1.csv");
        myFile4.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/magnitudeFFTW2.csv");
        myFile5.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/phaseFFTW2.csv");
        myFile6.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/voxelDataFFTW2.csv");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    int index = generalHelpfulTools::index3D(i, j, k, N);
                    myFile1 << this->magnitude1[index];
                    myFile1 << "\n";
                    myFile2 << this->phase1[index];
                    myFile2 << "\n";
                    myFile3 << voxelData1Input[index];
                    myFile3 << "\n";
                    myFile4 << this->magnitude2[index];
                    myFile4 << "\n";
                    myFile5 << this->phase2[index];
                    myFile5 << "\n";
                    myFile6 << voxelData2Input[index];
                    myFile6 << "\n";
                }
            }
        }

        myFile1.close();
        myFile2.close();
        myFile3.close();
        myFile4.close();
        myFile5.close();
        myFile6.close();
    }

    // get global maximum for normalization
    double globalMaximumMagnitude;
    if (maximumScan2Magnitude < maximumScan1Magnitude) {
        globalMaximumMagnitude = maximumScan1Magnitude;
    } else {
        globalMaximumMagnitude = maximumScan2Magnitude;
    }

    //normalize and shift both spectrums
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                int index = generalHelpfulTools::index3D(i, j, k, N);

                int indexX = (N / 2 + i) % N;
                int indexY = (N / 2 + j) % N;
                int indexZ = (N / 2 + k) % N;

                int indexshift = generalHelpfulTools::index3D(indexX, indexY, indexZ, N);
                magnitude1Shifted[indexshift] =
                        magnitude1[index] / globalMaximumMagnitude;
                magnitude2Shifted[indexshift] =
                        magnitude2[index] / globalMaximumMagnitude;
            }
        }
    }


    for (int i = 0; i < N * N; i++) {
        resampledMagnitudeSO3_1[i] = 0;
        resampledMagnitudeSO3_2[i] = 0;
        resampledMagnitudeSO3_1TMP[i] = 0;
        resampledMagnitudeSO3_2TMP[i] = 0;
    }

    int bandwidth = N / 2;
    if (!set_r_manual) {
        r_min = N / 8.0;
        r_max = N / 2.0 - N / 8.0;
    }
    // config -> r_min r_max
    for (int r = int(r_min); r < r_max; r++) {
        // was N/16

        for (int i = 0; i < N * N; i++) {
            resampledMagnitudeSO3_1TMP[i] = 0;
            resampledMagnitudeSO3_2TMP[i] = 0;
        }
        double minValue = INFINITY;
        double maxValue = 0;
        for (int i = 0; i < 2 * bandwidth; i++) {
            for (int j = 0; j < 2 * bandwidth; j++) {
                double theta = thetaIncrement3D((double) i, bandwidth);
                double phi = phiIncrement3D((double) j, bandwidth);
                double radius = r;

                int xIndex = int(std::round(radius * std::sin(theta) * std::cos(phi) + N / 2.0 + 0.1));
                int yIndex = int(std::round(radius * std::sin(theta) * std::sin(phi) + N / 2.0 + 0.1));
                int zIndex = int(std::round(radius * std::cos(theta) + N / 2.0 + 0.1));


                resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)] =
                        magnitude1Shifted[generalHelpfulTools::index3D(xIndex, yIndex, zIndex, bandwidth * 2)];
                resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)] =
                        magnitude2Shifted[generalHelpfulTools::index3D(xIndex, yIndex, zIndex, bandwidth * 2)];
                if (minValue > resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)]) {
                    minValue = resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)];
                }
                if (maxValue < resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)]) {
                    maxValue = resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)];
                }
                if (minValue > resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)]) {
                    minValue = resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)];
                }
                if (maxValue < resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)]) {
                    maxValue = resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)];
                }
            }
        }
        // make signal between 0 and 255 for Clahe
        for (int i = 0; i < 2 * bandwidth; i++) {
            for (int j = 0; j < 2 * bandwidth; j++) {
                this->resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, 2 * bandwidth)] =
                        255.0 * ((this->resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, 2 * bandwidth)] -
                                  minValue) /
                                 (maxValue - minValue));
                this->resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, 2 * bandwidth)] =
                        255.0 * ((this->resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, 2 * bandwidth)] -
                                  minValue) /
                                 (maxValue - minValue));
            }
        }

        cv::Mat magTMP1(N, N, CV_64FC1, resampledMagnitudeSO3_1TMP);
        cv::Mat magTMP2(N, N, CV_64FC1, resampledMagnitudeSO3_2TMP);
        magTMP1.convertTo(magTMP1, CV_8UC1);
        magTMP2.convertTo(magTMP2, CV_8UC1);

        //        cv::imshow("b1", magTMP1);
        //        cv::imshow("b2", magTMP2);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        //        clahe->setClipLimit(1);
        if (useClahe) {
            clahe->apply(magTMP1, magTMP1);
            clahe->apply(magTMP2, magTMP2);
        }
        //        cv::imshow("a1", magTMP1);
        //        cv::imshow("a2", magTMP2);
        //        int k = cv::waitKey(0); // Wait for a keystroke in the window
        //transform signal back to 0-1
        for (int i = 0; i < 2 * bandwidth; i++) {
            for (int j = 0; j < 2 * bandwidth; j++) {
                resampledMagnitudeSO3_1[generalHelpfulTools::index2D(i, j, bandwidth * 2)] +=
                        ((double) magTMP1.data[generalHelpfulTools::index2D(i, j, bandwidth * 2)]) /
                        255.0;
                resampledMagnitudeSO3_2[generalHelpfulTools::index2D(i, j, bandwidth * 2)] +=
                        ((double) magTMP2.data[generalHelpfulTools::index2D(i, j, bandwidth * 2)]) /
                        255.0;
            }
        }
    }
    if (debug) {
        std::ofstream myFile1, myFile2;
        myFile1.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/resampledMagnitudeSO3_1.csv");
        myFile2.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/resampledMagnitudeSO3_2.csv");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                myFile1 << this->resampledMagnitudeSO3_1[generalHelpfulTools::index2D(i, j,
                    bandwidth * 2)]; // real part
                myFile1 << "\n";
                myFile2 << this->resampledMagnitudeSO3_2[generalHelpfulTools::index2D(i, j,
                    bandwidth * 2)]; // imaginary part
                myFile2 << "\n";
            }
        }
        myFile1.close();
        myFile2.close();
    }

    if (timeStuff) {
        end = std::chrono::steady_clock::now();
        diff = end - begin;
        std::cout << "computation from 3D Spectrum to SOFT descriptor: " << diff.count() << std::endl;
        begin = std::chrono::steady_clock::now();
    }


    this->sofftCorrelationObject3D.correlationOfTwoSignalsInSO3(resampledMagnitudeSO3_1, resampledMagnitudeSO3_2,
                                                                resultingCorrelationComplex);
    if (timeStuff) {
        end = std::chrono::steady_clock::now();
        diff = end - begin;
        std::cout << "rotationCorrelation: " << diff.count() << std::endl;
        begin = std::chrono::steady_clock::now();
    }

    if (debug) {
        std::ofstream myFile1, myFile2;
        myFile1.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/resultingCorrelationReal.csv");
        myFile2.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/resultingCorrelationComplex.csv");
        for (int i = 0; i < bwOut * 2; i++) {
            for (int j = 0; j < bwOut * 2; j++) {
                for (int k = 0; k < bwOut * 2; k++) {
                    int index = generalHelpfulTools::index3D(i, j, k, bwOut * 2);
                    myFile1 << this->resultingCorrelationComplex[index][0]; // real part
                    myFile1 << "\n";
                    myFile2 << this->resultingCorrelationComplex[index][1]; // imaginary part
                    myFile2 << "\n";
                }
            }
        }
        myFile1.close();
        myFile2.close();
    }

    double minimumCorrelation = INFINITY;
    double maximumCorrelation = 0;
    for (int i = 0; i < 8 * bwOut * bwOut * bwOut; i++) {
        if (minimumCorrelation > NORM(resultingCorrelationComplex[i])) {
            minimumCorrelation = NORM(resultingCorrelationComplex[i]);
        }
        if (maximumCorrelation < NORM(resultingCorrelationComplex[i])) {
            maximumCorrelation = NORM(resultingCorrelationComplex[i]);
        }
    }


    double correlationCurrent;
    std::vector<My4DPoint> listOfQuaternionCorrelation;
    for (int i = 0; i < bwOut * 2; i++) {
        for (int j = 0; j < bwOut * 2; j++) {
            for (int k = 0; k < bwOut * 2; k++) {
                correlationCurrent =
                        (NORM(resultingCorrelationComplex[generalHelpfulTools::index3D(i, j, k, 2 * bwOut)]) -
                         minimumCorrelation) /
                        (maximumCorrelation - minimumCorrelation);
                Eigen::AngleAxisd rotation_vectorz1(k * 2 * M_PI / (N), Eigen::Vector3d(0, 0, 1));
                Eigen::AngleAxisd rotation_vectory(i * M_PI / (N), Eigen::Vector3d(0, 1, 0));
                Eigen::AngleAxisd rotation_vectorz2(j * 2 * M_PI / (N), Eigen::Vector3d(0, 0, 1));

                Eigen::Matrix3d tmpMatrix3d =
                        rotation_vectorz1.toRotationMatrix().inverse() *
                        rotation_vectory.toRotationMatrix().inverse() *
                        rotation_vectorz2.toRotationMatrix().inverse();
                Eigen::Quaterniond quaternionResult(tmpMatrix3d);
                quaternionResult.normalize();

                if (quaternionResult.w() < 0) {
                    Eigen::Quaterniond tmpQuad = quaternionResult;
                    quaternionResult.w() = -tmpQuad.w();
                    quaternionResult.x() = -tmpQuad.x();
                    quaternionResult.y() = -tmpQuad.y();
                    quaternionResult.z() = -tmpQuad.z();
                }
                My4DPoint currentPoint;
                currentPoint.correlation = correlationCurrent;
                currentPoint[0] = quaternionResult.w();
                currentPoint[1] = quaternionResult.x();
                currentPoint[2] = quaternionResult.y();
                currentPoint[3] = quaternionResult.z();

                listOfQuaternionCorrelation.push_back(currentPoint);
            }
        }
    }
    if (timeStuff) {
        end = std::chrono::steady_clock::now();
        diff = end - begin;
        std::cout << "overhead until peak detection: " << diff.count() << std::endl;
        begin = std::chrono::steady_clock::now();
    }
    // config -> settings for peak detection
    std::vector<rotationPeak4D> potentialRotationsTMP = this->peakDetectionOf4DCorrelationWithKDTreeFindPeaksLibrary(
        listOfQuaternionCorrelation, level_potential_rotation);
    if (timeStuff) {
        end = std::chrono::steady_clock::now();
        diff = end - begin;
        std::cout << "peak detection: " << diff.count() << std::endl;
        begin = std::chrono::steady_clock::now();
    }



    int indexOfInterest= 0;
    double bestFit = INFINITY;
    for (int i = 0; i < potentialRotationsTMP.size(); i++) {
        Eigen::Quaterniond rotationQuat(potentialRotationsTMP[i].w, potentialRotationsTMP[i].x,
                                        potentialRotationsTMP[i].y, potentialRotationsTMP[i].z);
        Eigen::Vector3d rpyCurrentRot = generalHelpfulTools::getRollPitchYaw(rotationQuat);

        Eigen::Quaterniond initRotationQuat(initGuessOrientation.w(), initGuessOrientation.x(),initGuessOrientation.y(), initGuessOrientation.z());
        double distance = abs(generalHelpfulTools::angleDifferenceQuaternion(rotationQuat,initRotationQuat));
        if (bestFit>distance) {
            bestFit = distance;
            indexOfInterest = i;
        }
    }
    rotationPeak4D bestFittingPeak = potentialRotationsTMP[indexOfInterest];

    if (timeStuff) {
        end = std::chrono::steady_clock::now();
        diff = end - begin;
        std::cout << "plotting Solutions: " << diff.count() << std::endl;
        begin = std::chrono::steady_clock::now();
    }

    transformationPeakfs3D bestFittingSolution;
    // for (int p = 0; p < potentialRotationsTMP.size(); p++) {
        //    for (int p = 0; p < 1; p++) {

        double *voxelData2Rotated;
        voxelData2Rotated = (double *) malloc(sizeof(double) * this->N * this->N * this->N);
        for (int i = 0; i < this->N * this->N * this->N; i++) {
            voxelData2Rotated[i] = 0;
        }
        Eigen::Quaterniond currentRotation(bestFittingPeak.w, bestFittingPeak.x,
                                           bestFittingPeak.y, bestFittingPeak.z);
        //        Eigen::Quaterniond currentRotation(0.99999, 0.022,
        //                                           0, 0);
        //        currentRotation.normalize();

        std::cout << currentRotation << std::endl;

        for (int i = 0; i < this->N; i++) {
            for (int j = 0; j < this->N; j++) {
                for (int k = 0; k < this->N; k++) {
                    int index = generalHelpfulTools::index3D(i, j, k, this->N);
                    //                    if(index<0){
                    //                        std::cout << "huhu2" << std::endl;
                    //                    }
                    int xCoordinate = i - this->N / 2;
                    int yCoordinate = j - this->N / 2;
                    int zCoordinate = k - this->N / 2;


                    Eigen::Vector3d newCoordinate(xCoordinate, yCoordinate, zCoordinate);
                    //                    std::cout << newCoordinate << std::endl;
                    Eigen::Vector3d lookUpVector = currentRotation * newCoordinate;
                    //                    std::cout << lookUpVector << std::endl;
                    //                    if(index> this->N*this->N*this->N/2){
                    //                        std::cout << "lookUpVector" << std::endl;
                    //                    }
                    double occupancyValue = getPixelValueInterpolated(lookUpVector, voxelData2Input, this->N);
                    voxelData2Rotated[index] = occupancyValue;
                }
            }
        }

        this->getSpectrumFromVoxelData3DCorrelation(voxelData1Input,
                                                    this->magnitude1Correlation,
                                                    this->phase1Correlation,
                                                    false);
        this->getSpectrumFromVoxelData3DCorrelation(voxelData2Rotated,
                                                    this->magnitude2Correlation,
                                                    this->phase2Correlation,
                                                    false);

        if (debug) {
            std::ofstream myFile1, myFile2, myFile3;
            myFile1.open(
                "/home/tim-external/matlab/registrationFourier/3D/csvFiles/magnitudeFFTW2Rotated.csv");
            myFile2.open(
                "/home/tim-external/matlab/registrationFourier/3D/csvFiles/phaseFFTW2Rotated.csv");

            for (int i = 0; i < this->correlationN; i++) {
                for (int j = 0; j < this->correlationN; j++) {
                    for (int k = 0; k < this->correlationN; k++) {
                        int index = generalHelpfulTools::index3D(i, j, k, this->correlationN);
                        myFile1 << this->magnitude2Correlation[index];
                        myFile1 << "\n";
                        myFile2 << this->phase2Correlation[index];
                        myFile2 << "\n";
                    }
                }
            }
            myFile1.close();
            myFile2.close();

            myFile3.open(
                "/home/tim-external/matlab/registrationFourier/3D/csvFiles/voxelDataFFTW2Rotated.csv");
            for (int i = 0; i < this->N; i++) {
                for (int j = 0; j < this->N; j++) {
                    for (int k = 0; k < this->N; k++) {
                        int index = generalHelpfulTools::index3D(i, j, k, this->N);
                        myFile3 << voxelData2Rotated[index];
                        myFile3 << "\n";
                    }
                }
            }
            myFile3.close();
        }

        //calculate correlation of spectrums
        for (int i = 0; i < this->correlationN; i++) {
            for (int j = 0; j < this->correlationN; j++) {
                for (int k = 0; k < this->correlationN; k++) {
                    int indexX = i;
                    int indexY = j;
                    int indexZ = k;
                    int index = generalHelpfulTools::index3D(indexX, indexY, indexZ, this->correlationN);

                    //                    if(index<0){
                    //                        std::cout << "thats a problem" << std::endl;
                    //                    }
                    //calculate the spectrum back
                    std::complex<double> tmpComplex1 =
                            magnitude1Correlation[index] *
                            std::exp(std::complex<double>(0, phase1Correlation[index]));
                    std::complex<double> tmpComplex2 =
                            magnitude2Correlation[index] *
                            std::exp(std::complex<double>(0, phase2Correlation[index]));
                    std::complex<double> resultComplex = ((tmpComplex1) * conj(tmpComplex2)); // cross correlation
                    resultingPhaseDiff3DCorrelation[index][0] = resultComplex.real();
                    resultingPhaseDiff3DCorrelation[index][1] = resultComplex.imag();
                }
            }
        }
        // back fft
        fftw_execute(planFourierToVoxel3DCorrelation);
        // fftshift and calc magnitude
        double maximumCorrelationTranslation = 0;
        double minimumCorrelationTranslation = INFINITY;

        for (int i = 0; i < this->correlationN; i++) {
            for (int j = 0; j < this->correlationN; j++) {
                for (int k = 0; k < this->correlationN; k++) {
                    int indexX = ((this->correlationN / 2) + i + this->correlationN) % this->correlationN;
                    // changed j and i here
                    int indexY = ((this->correlationN / 2) + j + this->correlationN) % this->correlationN;
                    int indexZ = ((this->correlationN / 2) + k + this->correlationN) % this->correlationN;
                    int indexShifted = generalHelpfulTools::index3D(indexX, indexY, indexZ, this->correlationN);
                    int index = generalHelpfulTools::index3D(i, j, k, this->correlationN);
                    //maybe without sqrt, but for now thats fine
                    //                    double normalizationFactorForCorrelation =
                    //                            1 / this->normalizationFactorCalculation(indexX, indexY, indexZ,this->correlationN);
                    //                    normalizationFactorForCorrelation = sqrt(normalizationFactorForCorrelation);
                    double normalizationFactorForCorrelation;
                    switch (normalization) {
                        case 0:
                            // code block
                            normalizationFactorForCorrelation = 1;
                            break;
                        case 1:
                            normalizationFactorForCorrelation =
                                    1 / normalizationFactorCalculation(indexX, indexY, indexZ, this->correlationN);
                        // normalizationFactorForCorrelation = sqrt(normalizationFactorForCorrelation);
                            break;
                        default:
                            std::cout << "normalization has to be 0,1 but was: " << normalization << std::endl;
                            exit(-1);
                    }


                    resultingCorrelationDouble[indexShifted] =
                            normalizationFactorForCorrelation *
                            NORM(resultingShiftPeaks3DCorrelation[index]); // magnitude;

                    if (maximumCorrelationTranslation < resultingCorrelationDouble[indexShifted]) {
                        maximumCorrelationTranslation = resultingCorrelationDouble[indexShifted];
                    }
                    if (minimumCorrelationTranslation > resultingCorrelationDouble[indexShifted]) {
                        minimumCorrelationTranslation = resultingCorrelationDouble[indexShifted];
                    }
                }
            }
        }

        for (int i = 0; i < this->correlationN * this->correlationN * this->correlationN; i++) {
            resultingCorrelationDouble[i] = (resultingCorrelationDouble[i] - minimumCorrelationTranslation) /
                                            (maximumCorrelationTranslation - minimumCorrelationTranslation);
        }

        if (debug) {
            std::ofstream myFile10;
            myFile10.open(
                "/home/tim-external/matlab/registrationFourier/3D/csvFiles/resultingCorrelationShift.csv");
            for (int i = 0; i < this->correlationN; i++) {
                for (int j = 0; j < this->correlationN; j++) {
                    for (int k = 0; k < this->correlationN; k++) {
                        myFile10 << resultingCorrelationDouble[
                            generalHelpfulTools::index3D(i, j, k, this->correlationN)];
                        myFile10 << "\n";
                    }
                }
            }
            myFile10.close();
        }
        // config -> settings for peak detection
        std::vector<translationPeak3D> resulting3DPeakList = peakDetectionOf3DCorrelationFindPeaksLibrary(
            resultingCorrelationDouble, this->correlationN, sizeVoxel, level_potential_translation);
        transformationPeakfs3D tmpSolution;
        tmpSolution.potentialRotation = bestFittingPeak;
    // @TODO compute best FIT based on estimation
        int indexBestFittingSolution= 0;
        double score=INFINITY;
        Eigen::Vector3d initGuessPositionEigen(initGuessPosition.x(), initGuessPosition.y(), initGuessPosition.z());

        for (int i = 0; i < resulting3DPeakList.size(); i++) {
            // sub Pixel Computation Here:

            resulting3DPeakList[i].correlationHeight = resulting3DPeakList[i].correlationHeight *
                                           (maximumCorrelationTranslation - minimumCorrelationTranslation) +
                                           minimumCorrelationTranslation;

            Eigen::Vector3d peakEstimationPosition(resulting3DPeakList[i].xTranslation,resulting3DPeakList[i].yTranslation,resulting3DPeakList[i].zTranslation);

            double distanceInitVsEst = (peakEstimationPosition-initGuessPositionEigen).norm();

            if (distanceInitVsEst < score) {
                indexBestFittingSolution = i;
                score = distanceInitVsEst;

            }
        }

        tmpSolution.potentialTranslations.push_back(resulting3DPeakList[indexBestFittingSolution]);

        free(voxelData2Rotated);


        if (timeStuff) {
            end = std::chrono::steady_clock::now();
            diff = end - begin;
            std::cout << "computing one Solution: " << diff.count() << std::endl;
            begin = std::chrono::steady_clock::now();
        }

    return tmpSolution;
}


std::vector<transformationPeakfs3D>
softRegistrationClass3D::sofftRegistrationVoxel3DListOfPossibleTransformations(double voxelData1Input[],
                                                                               double voxelData2Input[], bool debug,
                                                                               bool useClahe, bool timeStuff,
                                                                               double sizeVoxel, double r_min,
                                                                               double r_max,
                                                                               double level_potential_rotation,
                                                                               double level_potential_translation,
                                                                               bool set_r_manual, int normalization) {
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    std::chrono::duration<double, std::milli> diff{};

    if (timeStuff) {
        begin = std::chrono::steady_clock::now();
    }


    double maximumScan1Magnitude = this->getSpectrumFromVoxelData3D(voxelData1Input, this->magnitude1,
                                                                    this->phase1, false);
    double maximumScan2Magnitude = this->getSpectrumFromVoxelData3D(voxelData2Input, this->magnitude2,
                                                                    this->phase2, false);

    if (timeStuff) {
        end = std::chrono::steady_clock::now();
        diff = end - begin;
        std::cout << "computation 3D Spectrum: " << diff.count() << std::endl;
        begin = std::chrono::steady_clock::now();
    }


    if (debug) {
        std::ofstream myFile1, myFile2, myFile3, myFile4, myFile5, myFile6;
        myFile1.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/magnitudeFFTW1.csv");
        myFile2.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/phaseFFTW1.csv");
        myFile3.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/voxelDataFFTW1.csv");
        myFile4.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/magnitudeFFTW2.csv");
        myFile5.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/phaseFFTW2.csv");
        myFile6.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/voxelDataFFTW2.csv");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    int index = generalHelpfulTools::index3D(i, j, k, N);
                    myFile1 << this->magnitude1[index];
                    myFile1 << "\n";
                    myFile2 << this->phase1[index];
                    myFile2 << "\n";
                    myFile3 << voxelData1Input[index];
                    myFile3 << "\n";
                    myFile4 << this->magnitude2[index];
                    myFile4 << "\n";
                    myFile5 << this->phase2[index];
                    myFile5 << "\n";
                    myFile6 << voxelData2Input[index];
                    myFile6 << "\n";
                }
            }
        }

        myFile1.close();
        myFile2.close();
        myFile3.close();
        myFile4.close();
        myFile5.close();
        myFile6.close();
    }

    // get global maximum for normalization
    double globalMaximumMagnitude;
    if (maximumScan2Magnitude < maximumScan1Magnitude) {
        globalMaximumMagnitude = maximumScan1Magnitude;
    } else {
        globalMaximumMagnitude = maximumScan2Magnitude;
    }

    //normalize and shift both spectrums
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                int index = generalHelpfulTools::index3D(i, j, k, N);

                int indexX = (N / 2 + i) % N;
                int indexY = (N / 2 + j) % N;
                int indexZ = (N / 2 + k) % N;

                int indexshift = generalHelpfulTools::index3D(indexX, indexY, indexZ, N);
                magnitude1Shifted[indexshift] =
                        magnitude1[index] / globalMaximumMagnitude;
                magnitude2Shifted[indexshift] =
                        magnitude2[index] / globalMaximumMagnitude;
            }
        }
    }


    for (int i = 0; i < N * N; i++) {
        resampledMagnitudeSO3_1[i] = 0;
        resampledMagnitudeSO3_2[i] = 0;
        resampledMagnitudeSO3_1TMP[i] = 0;
        resampledMagnitudeSO3_2TMP[i] = 0;
    }

    int bandwidth = N / 2;
    if (!set_r_manual) {
        r_min = N / 8.0;
        r_max = N / 2.0 - N / 8.0;
    }
    // config -> r_min r_max
    for (int r = int(r_min); r < r_max; r++) {
        // was N/16

        for (int i = 0; i < N * N; i++) {
            resampledMagnitudeSO3_1TMP[i] = 0;
            resampledMagnitudeSO3_2TMP[i] = 0;
        }
        double minValue = INFINITY;
        double maxValue = 0;
        for (int i = 0; i < 2 * bandwidth; i++) {
            for (int j = 0; j < 2 * bandwidth; j++) {
                double theta = thetaIncrement3D((double) i, bandwidth);
                double phi = phiIncrement3D((double) j, bandwidth);
                double radius = r;

                int xIndex = int(std::round(radius * std::sin(theta) * std::cos(phi) + N / 2.0 + 0.1));
                int yIndex = int(std::round(radius * std::sin(theta) * std::sin(phi) + N / 2.0 + 0.1));
                int zIndex = int(std::round(radius * std::cos(theta) + N / 2.0 + 0.1));


                resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)] =
                        magnitude1Shifted[generalHelpfulTools::index3D(xIndex, yIndex, zIndex, bandwidth * 2)];
                resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)] =
                        magnitude2Shifted[generalHelpfulTools::index3D(xIndex, yIndex, zIndex, bandwidth * 2)];
                if (minValue > resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)]) {
                    minValue = resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)];
                }
                if (maxValue < resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)]) {
                    maxValue = resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)];
                }
                if (minValue > resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)]) {
                    minValue = resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)];
                }
                if (maxValue < resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)]) {
                    maxValue = resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, bandwidth * 2)];
                }
            }
        }
        // make signal between 0 and 255 for Clahe
        for (int i = 0; i < 2 * bandwidth; i++) {
            for (int j = 0; j < 2 * bandwidth; j++) {
                this->resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, 2 * bandwidth)] =
                        255.0 * ((this->resampledMagnitudeSO3_1TMP[generalHelpfulTools::index2D(i, j, 2 * bandwidth)] -
                                  minValue) /
                                 (maxValue - minValue));
                this->resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, 2 * bandwidth)] =
                        255.0 * ((this->resampledMagnitudeSO3_2TMP[generalHelpfulTools::index2D(i, j, 2 * bandwidth)] -
                                  minValue) /
                                 (maxValue - minValue));
            }
        }

        cv::Mat magTMP1(N, N, CV_64FC1, resampledMagnitudeSO3_1TMP);
        cv::Mat magTMP2(N, N, CV_64FC1, resampledMagnitudeSO3_2TMP);
        magTMP1.convertTo(magTMP1, CV_8UC1);
        magTMP2.convertTo(magTMP2, CV_8UC1);

        //        cv::imshow("b1", magTMP1);
        //        cv::imshow("b2", magTMP2);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        //        clahe->setClipLimit(1);
        if (useClahe) {
            clahe->apply(magTMP1, magTMP1);
            clahe->apply(magTMP2, magTMP2);
        }
        //        cv::imshow("a1", magTMP1);
        //        cv::imshow("a2", magTMP2);
        //        int k = cv::waitKey(0); // Wait for a keystroke in the window
        //transform signal back to 0-1
        for (int i = 0; i < 2 * bandwidth; i++) {
            for (int j = 0; j < 2 * bandwidth; j++) {
                resampledMagnitudeSO3_1[generalHelpfulTools::index2D(i, j, bandwidth * 2)] +=
                        ((double) magTMP1.data[generalHelpfulTools::index2D(i, j, bandwidth * 2)]) /
                        255.0;
                resampledMagnitudeSO3_2[generalHelpfulTools::index2D(i, j, bandwidth * 2)] +=
                        ((double) magTMP2.data[generalHelpfulTools::index2D(i, j, bandwidth * 2)]) /
                        255.0;
            }
        }
    }
    if (debug) {
        std::ofstream myFile1, myFile2;
        myFile1.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/resampledMagnitudeSO3_1.csv");
        myFile2.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/resampledMagnitudeSO3_2.csv");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                myFile1 << this->resampledMagnitudeSO3_1[generalHelpfulTools::index2D(i, j,
                    bandwidth * 2)]; // real part
                myFile1 << "\n";
                myFile2 << this->resampledMagnitudeSO3_2[generalHelpfulTools::index2D(i, j,
                    bandwidth * 2)]; // imaginary part
                myFile2 << "\n";
            }
        }
        myFile1.close();
        myFile2.close();
    }

    if (timeStuff) {
        end = std::chrono::steady_clock::now();
        diff = end - begin;
        std::cout << "computation from 3D Spectrum to SOFT descriptor: " << diff.count() << std::endl;
        begin = std::chrono::steady_clock::now();
    }


    this->sofftCorrelationObject3D.correlationOfTwoSignalsInSO3(resampledMagnitudeSO3_1, resampledMagnitudeSO3_2,
                                                                resultingCorrelationComplex);
    if (timeStuff) {
        end = std::chrono::steady_clock::now();
        diff = end - begin;
        std::cout << "rotationCorrelation: " << diff.count() << std::endl;
        begin = std::chrono::steady_clock::now();
    }

    if (debug) {
        std::ofstream myFile1, myFile2;
        myFile1.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/resultingCorrelationReal.csv");
        myFile2.open(
            "/home/tim-external/matlab/registrationFourier/3D/csvFiles/resultingCorrelationComplex.csv");
        for (int i = 0; i < bwOut * 2; i++) {
            for (int j = 0; j < bwOut * 2; j++) {
                for (int k = 0; k < bwOut * 2; k++) {
                    int index = generalHelpfulTools::index3D(i, j, k, bwOut * 2);
                    myFile1 << this->resultingCorrelationComplex[index][0]; // real part
                    myFile1 << "\n";
                    myFile2 << this->resultingCorrelationComplex[index][1]; // imaginary part
                    myFile2 << "\n";
                }
            }
        }
        myFile1.close();
        myFile2.close();
    }

    double minimumCorrelation = INFINITY;
    double maximumCorrelation = 0;
    for (int i = 0; i < 8 * bwOut * bwOut * bwOut; i++) {
        if (minimumCorrelation > NORM(resultingCorrelationComplex[i])) {
            minimumCorrelation = NORM(resultingCorrelationComplex[i]);
        }
        if (maximumCorrelation < NORM(resultingCorrelationComplex[i])) {
            maximumCorrelation = NORM(resultingCorrelationComplex[i]);
        }
    }


    double correlationCurrent;
    std::vector<My4DPoint> listOfQuaternionCorrelation;
    for (int i = 0; i < bwOut * 2; i++) {
        for (int j = 0; j < bwOut * 2; j++) {
            for (int k = 0; k < bwOut * 2; k++) {
                correlationCurrent =
                        (NORM(resultingCorrelationComplex[generalHelpfulTools::index3D(i, j, k, 2 * bwOut)]) -
                         minimumCorrelation) /
                        (maximumCorrelation - minimumCorrelation);
                Eigen::AngleAxisd rotation_vectorz1(k * 2 * M_PI / (N), Eigen::Vector3d(0, 0, 1));
                Eigen::AngleAxisd rotation_vectory(i * M_PI / (N), Eigen::Vector3d(0, 1, 0));
                Eigen::AngleAxisd rotation_vectorz2(j * 2 * M_PI / (N), Eigen::Vector3d(0, 0, 1));

                Eigen::Matrix3d tmpMatrix3d =
                        rotation_vectorz1.toRotationMatrix().inverse() *
                        rotation_vectory.toRotationMatrix().inverse() *
                        rotation_vectorz2.toRotationMatrix().inverse();
                Eigen::Quaterniond quaternionResult(tmpMatrix3d);
                quaternionResult.normalize();

                if (quaternionResult.w() < 0) {
                    Eigen::Quaterniond tmpQuad = quaternionResult;
                    quaternionResult.w() = -tmpQuad.w();
                    quaternionResult.x() = -tmpQuad.x();
                    quaternionResult.y() = -tmpQuad.y();
                    quaternionResult.z() = -tmpQuad.z();
                }
                My4DPoint currentPoint;
                currentPoint.correlation = correlationCurrent;
                currentPoint[0] = quaternionResult.w();
                currentPoint[1] = quaternionResult.x();
                currentPoint[2] = quaternionResult.y();
                currentPoint[3] = quaternionResult.z();

                listOfQuaternionCorrelation.push_back(currentPoint);
            }
        }
    }
    if (timeStuff) {
        end = std::chrono::steady_clock::now();
        diff = end - begin;
        std::cout << "overhead until peak detection: " << diff.count() << std::endl;
        begin = std::chrono::steady_clock::now();
    }
    // config -> settings for peak detection
    std::vector<rotationPeak4D> potentialRotationsTMP = this->peakDetectionOf4DCorrelationWithKDTreeFindPeaksLibrary(
        listOfQuaternionCorrelation, level_potential_rotation);
    if (timeStuff) {
        end = std::chrono::steady_clock::now();
        diff = end - begin;
        std::cout << "peak detection: " << diff.count() << std::endl;
        begin = std::chrono::steady_clock::now();
    }
    for (int i = 0; i < potentialRotationsTMP.size(); i++) {
        Eigen::Quaterniond rotationQuat(potentialRotationsTMP[i].w, potentialRotationsTMP[i].x,
                                        potentialRotationsTMP[i].y, potentialRotationsTMP[i].z);
        Eigen::Vector3d rpyCurrentRot = generalHelpfulTools::getRollPitchYaw(rotationQuat);

        std::cout << i << " , " << potentialRotationsTMP[i].levelPotential << " , "
                << potentialRotationsTMP[i].correlationHeight
                << " , " << potentialRotationsTMP[i].persistence << " , " << rpyCurrentRot[0] * 180 / M_PI << " , "
                << rpyCurrentRot[1] * 180 / M_PI << " , " << rpyCurrentRot[2] * 180 / M_PI << " , "
                << potentialRotationsTMP[i].x << " , " << potentialRotationsTMP[i].y << " , "
                << potentialRotationsTMP[i].z << " , " << potentialRotationsTMP[i].w << std::endl;
    }
    if (timeStuff) {
        end = std::chrono::steady_clock::now();
        diff = end - begin;
        std::cout << "plotting Solutions: " << diff.count() << std::endl;
        begin = std::chrono::steady_clock::now();
    }
    std::vector<transformationPeakfs3D> allSolutions;
    for (int p = 0; p < potentialRotationsTMP.size(); p++) {
        //    for (int p = 0; p < 1; p++) {

        double *voxelData2Rotated;
        voxelData2Rotated = (double *) malloc(sizeof(double) * this->N * this->N * this->N);
        for (int i = 0; i < this->N * this->N * this->N; i++) {
            voxelData2Rotated[i] = 0;
        }
        Eigen::Quaterniond currentRotation(potentialRotationsTMP[p].w, potentialRotationsTMP[p].x,
                                           potentialRotationsTMP[p].y, potentialRotationsTMP[p].z);
        //        Eigen::Quaterniond currentRotation(0.99999, 0.022,
        //                                           0, 0);
        //        currentRotation.normalize();

        std::cout << currentRotation << std::endl;

        for (int i = 0; i < this->N; i++) {
            for (int j = 0; j < this->N; j++) {
                for (int k = 0; k < this->N; k++) {
                    int index = generalHelpfulTools::index3D(i, j, k, this->N);
                    //                    if(index<0){
                    //                        std::cout << "huhu2" << std::endl;
                    //                    }
                    int xCoordinate = i - this->N / 2;
                    int yCoordinate = j - this->N / 2;
                    int zCoordinate = k - this->N / 2;


                    Eigen::Vector3d newCoordinate(xCoordinate, yCoordinate, zCoordinate);
                    //                    std::cout << newCoordinate << std::endl;
                    Eigen::Vector3d lookUpVector = currentRotation * newCoordinate;
                    //                    std::cout << lookUpVector << std::endl;
                    //                    if(index> this->N*this->N*this->N/2){
                    //                        std::cout << "lookUpVector" << std::endl;
                    //                    }
                    double occupancyValue = getPixelValueInterpolated(lookUpVector, voxelData2Input, this->N);
                    voxelData2Rotated[index] = occupancyValue;
                }
            }
        }

        this->getSpectrumFromVoxelData3DCorrelation(voxelData1Input,
                                                    this->magnitude1Correlation,
                                                    this->phase1Correlation,
                                                    false);
        this->getSpectrumFromVoxelData3DCorrelation(voxelData2Rotated,
                                                    this->magnitude2Correlation,
                                                    this->phase2Correlation,
                                                    false);

        if (debug) {
            std::ofstream myFile1, myFile2, myFile3;
            myFile1.open(
                "/home/tim-external/matlab/registrationFourier/3D/csvFiles/magnitudeFFTW2Rotated" +
                std::to_string(p) + ".csv");
            myFile2.open(
                "/home/tim-external/matlab/registrationFourier/3D/csvFiles/phaseFFTW2Rotated" +
                std::to_string(p) + ".csv");

            for (int i = 0; i < this->correlationN; i++) {
                for (int j = 0; j < this->correlationN; j++) {
                    for (int k = 0; k < this->correlationN; k++) {
                        int index = generalHelpfulTools::index3D(i, j, k, this->correlationN);
                        myFile1 << this->magnitude2Correlation[index];
                        myFile1 << "\n";
                        myFile2 << this->phase2Correlation[index];
                        myFile2 << "\n";
                    }
                }
            }
            myFile1.close();
            myFile2.close();

            myFile3.open(
                "/home/tim-external/matlab/registrationFourier/3D/csvFiles/voxelDataFFTW2Rotated" +
                std::to_string(p) + ".csv");
            for (int i = 0; i < this->N; i++) {
                for (int j = 0; j < this->N; j++) {
                    for (int k = 0; k < this->N; k++) {
                        int index = generalHelpfulTools::index3D(i, j, k, this->N);
                        myFile3 << voxelData2Rotated[index];
                        myFile3 << "\n";
                    }
                }
            }
            myFile3.close();
        }

        //calculate correlation of spectrums
        for (int i = 0; i < this->correlationN; i++) {
            for (int j = 0; j < this->correlationN; j++) {
                for (int k = 0; k < this->correlationN; k++) {
                    int indexX = i;
                    int indexY = j;
                    int indexZ = k;
                    int index = generalHelpfulTools::index3D(indexX, indexY, indexZ, this->correlationN);

                    //                    if(index<0){
                    //                        std::cout << "thats a problem" << std::endl;
                    //                    }
                    //calculate the spectrum back
                    std::complex<double> tmpComplex1 =
                            magnitude1Correlation[index] *
                            std::exp(std::complex<double>(0, phase1Correlation[index]));
                    std::complex<double> tmpComplex2 =
                            magnitude2Correlation[index] *
                            std::exp(std::complex<double>(0, phase2Correlation[index]));
                    std::complex<double> resultComplex = ((tmpComplex1) * conj(tmpComplex2)); // cross correlation
                    resultingPhaseDiff3DCorrelation[index][0] = resultComplex.real();
                    resultingPhaseDiff3DCorrelation[index][1] = resultComplex.imag();
                }
            }
        }
        // back fft
        fftw_execute(planFourierToVoxel3DCorrelation);
        // fftshift and calc magnitude
        double maximumCorrelationTranslation = 0;
        double minimumCorrelationTranslation = INFINITY;

        for (int i = 0; i < this->correlationN; i++) {
            for (int j = 0; j < this->correlationN; j++) {
                for (int k = 0; k < this->correlationN; k++) {
                    int indexX = ((this->correlationN / 2) + i + this->correlationN) % this->correlationN;
                    // changed j and i here
                    int indexY = ((this->correlationN / 2) + j + this->correlationN) % this->correlationN;
                    int indexZ = ((this->correlationN / 2) + k + this->correlationN) % this->correlationN;
                    int indexShifted = generalHelpfulTools::index3D(indexX, indexY, indexZ, this->correlationN);
                    int index = generalHelpfulTools::index3D(i, j, k, this->correlationN);
                    //maybe without sqrt, but for now thats fine
                    //                    double normalizationFactorForCorrelation =
                    //                            1 / this->normalizationFactorCalculation(indexX, indexY, indexZ,this->correlationN);
                    //                    normalizationFactorForCorrelation = sqrt(normalizationFactorForCorrelation);
                    double normalizationFactorForCorrelation;
                    switch (normalization) {
                        case 0:
                            // code block
                            normalizationFactorForCorrelation = 1;
                            break;
                        case 1:
                            normalizationFactorForCorrelation =
                                    1 / normalizationFactorCalculation(indexX, indexY, indexZ, this->correlationN);
                        // normalizationFactorForCorrelation = sqrt(normalizationFactorForCorrelation);
                            break;
                        default:
                            std::cout << "normalization has to be 0,1 but was: " << normalization << std::endl;
                            exit(-1);
                    }


                    resultingCorrelationDouble[indexShifted] =
                            normalizationFactorForCorrelation *
                            NORM(resultingShiftPeaks3DCorrelation[index]); // magnitude;

                    if (maximumCorrelationTranslation < resultingCorrelationDouble[indexShifted]) {
                        maximumCorrelationTranslation = resultingCorrelationDouble[indexShifted];
                    }
                    if (minimumCorrelationTranslation > resultingCorrelationDouble[indexShifted]) {
                        minimumCorrelationTranslation = resultingCorrelationDouble[indexShifted];
                    }
                }
            }
        }

        for (int i = 0; i < this->correlationN * this->correlationN * this->correlationN; i++) {
            resultingCorrelationDouble[i] = (resultingCorrelationDouble[i] - minimumCorrelationTranslation) /
                                            (maximumCorrelationTranslation - minimumCorrelationTranslation);
        }

        if (debug) {
            std::ofstream myFile10;
            myFile10.open(
                "/home/tim-external/matlab/registrationFourier/3D/csvFiles/resultingCorrelationShift" +
                std::to_string(p) + ".csv");
            for (int i = 0; i < this->correlationN; i++) {
                for (int j = 0; j < this->correlationN; j++) {
                    for (int k = 0; k < this->correlationN; k++) {
                        myFile10 << resultingCorrelationDouble[
                            generalHelpfulTools::index3D(i, j, k, this->correlationN)];
                        myFile10 << "\n";
                    }
                }
            }
            myFile10.close();
        }
        // config -> settings for peak detection
        std::vector<translationPeak3D> resulting3DPeakList = peakDetectionOf3DCorrelationFindPeaksLibrary(
            resultingCorrelationDouble, this->correlationN, sizeVoxel, level_potential_translation);
        transformationPeakfs3D tmpSolution;
        tmpSolution.potentialRotation = potentialRotationsTMP[p];
        for (int i = 0; i < resulting3DPeakList.size(); i++) {
            // sub Pixel Computation Here:



            resulting3DPeakList[i].correlationHeight = resulting3DPeakList[i].correlationHeight *
                                                       (maximumCorrelationTranslation - minimumCorrelationTranslation) +
                                                       minimumCorrelationTranslation;

            std::cout << p << " , " << i << " , " << resulting3DPeakList[i].levelPotential << " , "
                    << resulting3DPeakList[i].correlationHeight << " , " << resulting3DPeakList[i].persistence
                    << " , " << resulting3DPeakList[i].xTranslation << " , "
                    << resulting3DPeakList[i].yTranslation
                    << " , " << resulting3DPeakList[i].zTranslation << std::endl;

            tmpSolution.potentialTranslations.push_back(resulting3DPeakList[i]);
        }


        free(voxelData2Rotated);


        allSolutions.push_back(tmpSolution);

        if (timeStuff) {
            end = std::chrono::steady_clock::now();
            diff = end - begin;
            std::cout << "computing one Solution " << p << " :" << diff.count() << std::endl;
            begin = std::chrono::steady_clock::now();
        }
    }
    return allSolutions;
}

//int getIndexOfData(int x, int y, int z, int N_input) {
//    if (x < 0 || x > N_input) {
//        return -1;
//    }
//    if (y < 0 || y > N_input) {
//        return -1;
//    }
//    if (z < 0 || z > N_input) {
//        return -1;
//    }
//    return z + N_input * y + N_input * N_input * x;
//}

double softRegistrationClass3D::normalizationFactorCalculation(int x, int y, int z, int currentN) {
    double tmpCalculation = 0;
    //    tmpCalculation = abs(1.0/((x-this->N/2)*(y-this->N/2)));
    //    tmpCalculation = this->N * this->N * (this->N - (x + 1) + 1);
    //    tmpCalculation = tmpCalculation * (this->N - (y + 1) + 1);
    if (x < ceil(currentN / 2)) {
        tmpCalculation = (x + 1);
    } else {
        tmpCalculation = (currentN - x);
    }

    if (y < ceil(currentN / 2)) {
        tmpCalculation = tmpCalculation * (y + 1);
    } else {
        tmpCalculation = tmpCalculation * (currentN - y);
    }

    if (z < ceil(currentN / 2)) {
        tmpCalculation = tmpCalculation * (z + 1);
    } else {
        tmpCalculation = tmpCalculation * (currentN - z);
    }

    return (tmpCalculation);
}

double
softRegistrationClass3D::getPixelValueInterpolated(Eigen::Vector3d positionVector, const double *volumeData,
                                                   int dimensionN) {
    //    int index = positionVector.x() + N * positionVector.y() + N * N * positionVector.z();
    //    std::cout << "test1" << std::endl;
    int xDown = floor(positionVector.x());
    int xUp = ceil(positionVector.x());

    int yDown = floor(positionVector.y());
    int yUp = ceil(positionVector.y());

    int zDown = floor(positionVector.z());
    int zUp = ceil(positionVector.z());

    int xDownCorrected = xDown + dimensionN / 2;
    int xUpCorrected = xUp + dimensionN / 2;

    int yDownCorrected = yDown + dimensionN / 2;
    int yUpCorrected = yUp + dimensionN / 2;

    int zDownCorrected = zDown + dimensionN / 2;
    int zUpCorrected = zUp + dimensionN / 2;


    double dist1 = (positionVector - Eigen::Vector3d(xDown, yDown, zDown)).norm();
    double dist2 = (positionVector - Eigen::Vector3d(xDown, yDown, zUp)).norm();
    double dist3 = (positionVector - Eigen::Vector3d(xDown, yUp, zDown)).norm();
    double dist4 = (positionVector - Eigen::Vector3d(xDown, yUp, zUp)).norm();
    double dist5 = (positionVector - Eigen::Vector3d(xUp, yDown, zDown)).norm();
    double dist6 = (positionVector - Eigen::Vector3d(xUp, yDown, zUp)).norm();
    double dist7 = (positionVector - Eigen::Vector3d(xUp, yUp, zDown)).norm();
    double dist8 = (positionVector - Eigen::Vector3d(xUp, yUp, zUp)).norm();


    //    double fullLength = dist1 + dist2 + dist3 + dist4 + dist5 + dist6 + dist7 + dist8;


    int index1 = generalHelpfulTools::index3D(xDownCorrected, yDownCorrected, zDownCorrected, dimensionN);
    int index2 = generalHelpfulTools::index3D(xDownCorrected, yDownCorrected, zUpCorrected, dimensionN);
    int index3 = generalHelpfulTools::index3D(xDownCorrected, yUpCorrected, zDownCorrected, dimensionN);
    int index4 = generalHelpfulTools::index3D(xDownCorrected, yUpCorrected, zUpCorrected, dimensionN);
    int index5 = generalHelpfulTools::index3D(xUpCorrected, yDownCorrected, zDownCorrected, dimensionN);
    int index6 = generalHelpfulTools::index3D(xUpCorrected, yDownCorrected, zUpCorrected, dimensionN);
    int index7 = generalHelpfulTools::index3D(xUpCorrected, yUpCorrected, zDownCorrected, dimensionN);
    int index8 = generalHelpfulTools::index3D(xUpCorrected, yUpCorrected, zUpCorrected, dimensionN);


    double correlationValue1;
    double correlationValue2;
    double correlationValue3;
    double correlationValue4;
    double correlationValue5;
    double correlationValue6;
    double correlationValue7;
    double correlationValue8;
    //    std::cout << "test2" << std::endl;
    if (index1 == -1) {
        correlationValue1 = 0;
    } else {
        correlationValue1 = volumeData[index1];
    }
    if (index2 == -1) {
        correlationValue2 = 0;
    } else {
        correlationValue2 = volumeData[index2];
    }
    if (index3 == -1) {
        correlationValue3 = 0;
    } else {
        correlationValue3 = volumeData[index3];
    }
    if (index4 == -1) {
        correlationValue4 = 0;
    } else {
        correlationValue4 = volumeData[index4];
    }
    if (index5 == -1) {
        correlationValue5 = 0;
    } else {
        correlationValue5 = volumeData[index5];
    }
    if (index6 == -1) {
        correlationValue6 = 0;
    } else {
        correlationValue6 = volumeData[index6];
    }
    if (index7 == -1) {
        correlationValue7 = 0;
    } else {
        correlationValue7 = volumeData[index7];
    }
    if (index8 == -1) {
        correlationValue8 = 0;
    } else {
        correlationValue8 = volumeData[index8];
    }

    double e = 1.0 / 10000000.0;
    if (dist1 < e) {
        return correlationValue1;
    }
    if (dist2 < e) {
        return correlationValue2;
    }
    if (dist3 < e) {
        return correlationValue3;
    }
    if (dist4 < e) {
        return correlationValue4;
    }
    if (dist5 < e) {
        return correlationValue5;
    }
    if (dist6 < e) {
        return correlationValue6;
    }
    if (dist7 < e) {
        return correlationValue7;
    }
    if (dist8 < e) {
        return correlationValue8;
    }

    //    std::cout << "test3" << std::endl;
    // inverse weighted sum
    double correlationValue = (correlationValue1 / dist1 + correlationValue2 / dist2 + correlationValue3 / dist3 +
                               correlationValue4 / dist4 + correlationValue5 / dist5 + correlationValue6 / dist6 +
                               correlationValue7 / dist7 + correlationValue8 / dist8) /
                              (1.0 / dist1 + 1.0 / dist2 + 1.0 / dist3 + 1.0 / dist4 + 1.0 / dist5 + 1.0 / dist6 +
                               1.0 / dist7 +
                               1.0 / dist8);
    if (correlationValue > 1.1) {
        std::cout << "huhu" << std::endl;
    }
    return correlationValue;
}


std::vector<translationPeak3D>
softRegistrationClass3D::peakDetectionOf3DCorrelationFindPeaksLibraryFromFFTW_COMPLEX(fftw_complex *inputcorrelation,
    double cellSize) const {
    double *current3DCorrelation;
    current3DCorrelation = (double *) malloc(
        sizeof(double) * this->N * this->N * this->N);

    double maxValue = 0;
    double minValue = INFINITY;
    //copy data
    for (int j = 0; j < this->N; j++) {
        for (int i = 0; i < this->N; i++) {
            for (int k = 0; k < this->N; k++) {
                double currentCorrelation =
                        (inputcorrelation[generalHelpfulTools::index3D(i, j, k, this->N)][0]) *
                        (inputcorrelation[generalHelpfulTools::index3D(i, j, k, this->N)][0]) +
                        (inputcorrelation[generalHelpfulTools::index3D(i, j, k, this->N)][1]) *
                        (inputcorrelation[generalHelpfulTools::index3D(i, j, k, this->N)][1]);
                current3DCorrelation[generalHelpfulTools::index3D(i, j, k, this->N)] = currentCorrelation;
                if (current3DCorrelation[generalHelpfulTools::index3D(i, j, k, this->N)] >
                    maxValue) {
                    maxValue = current3DCorrelation[generalHelpfulTools::index3D(i, j, k, this->N)];
                }
                if (current3DCorrelation[generalHelpfulTools::index3D(i, j, k, this->N)] <
                    minValue) {
                    minValue = current3DCorrelation[generalHelpfulTools::index3D(i, j, k, this->N)];
                }
            }
        }
    }
    //normalize data
    for (int j = 0; j < this->N; j++) {
        for (int i = 0; i < this->N; i++) {
            for (int k = 0; k < this->N; k++) {
                current3DCorrelation[generalHelpfulTools::index3D(i, j, k, this->N)] =
                        (current3DCorrelation[generalHelpfulTools::index3D(i, j, k, this->N)] - minValue) /
                        (maxValue - minValue);
            }
        }
    }
    cv::Mat magTMP1(this->N, this->N, CV_64F, current3DCorrelation);
    //    cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);

    //    cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
    //                                            cv::Size(ceil(0.02 * this->N), ceil(0.02 * this->N)));
    //    cv::morphologyEx(magTMP1, magTMP1, cv::MORPH_TOPHAT, element);

    size_t ourSize = this->N;
    findpeaks::volume_t<double> volume = {
        ourSize, ourSize, ourSize,
        current3DCorrelation
    };

    std::vector<findpeaks::peak_3d<double> > peaks = findpeaks::persistance3d(volume);
    std::vector<translationPeak3D> tmpTranslations;
    std::cout << peaks.size() << std::endl;
    int numberOfPeaks = 0;
    for (const auto &p: peaks) {
        //calculation of level, that is a potential translation
        double levelPotential = p.persistence * sqrt(p.birth_level) *
                                Eigen::Vector3d((double) ((int) p.birth_position.x - (int) p.death_position.x),
                                                (double) ((int) p.birth_position.y - (int) p.death_position.y),
                                                (double) ((int) p.birth_position.z - (int) p.death_position.z)).norm() /
                                this->N / 1.73205080757;

        std::cout << p.persistence << std::endl;
        std::cout << Eigen::Vector3d((double) ((int) p.birth_position.x - (int) p.death_position.x),
                                     (double) ((int) p.birth_position.y - (int) p.death_position.y),
                                     (double) ((int) p.birth_position.z - (int) p.death_position.z)).norm()
                << std::endl;
        std::cout << sqrt(p.birth_level) << std::endl;


        translationPeak3D tmpTranslationPeak{};
        tmpTranslationPeak.xTranslation = double(p.birth_position.x);
        tmpTranslationPeak.yTranslation = double(p.birth_position.y);
        tmpTranslationPeak.zTranslation = double(p.birth_position.z);
        tmpTranslationPeak.persistence = double(p.persistence);
        tmpTranslationPeak.correlationHeight = current3DCorrelation[generalHelpfulTools::index3D(
            int(p.birth_position.x),
            int(p.birth_position.y),
            int(p.birth_position.z),
            this->N)];
        tmpTranslationPeak.levelPotential = levelPotential;
        if (levelPotential > 0.1) {
            tmpTranslations.push_back(tmpTranslationPeak);
            std::cout << "test" << std::endl;
            numberOfPeaks++;
        }
    }
    free(current3DCorrelation);
    return (tmpTranslations);
}

std::vector<translationPeak3D>
softRegistrationClass3D::peakDetectionOf3DCorrelationFindPeaksLibrary(const double *inputcorrelation, int dimensionN,
                                                                      double cellSize,
                                                                      double level_potential_translation) const {
    double *current3DCorrelation;
    current3DCorrelation = (double *) malloc(
        sizeof(double) * dimensionN * dimensionN * dimensionN);

    //    double maxValue = 0;
    //    double minValue = INFINITY;
    //copy data
    for (int i = 0; i < dimensionN; i++) {
        for (int j = 0; j < dimensionN; j++) {
            for (int k = 0; k < dimensionN; k++) {
                int index = generalHelpfulTools::index3D(i, j, k, dimensionN);
                double currentCorrelation = inputcorrelation[index];
                current3DCorrelation[index] = currentCorrelation;
            }
        }
    }


    size_t ourSize = dimensionN;
    findpeaks::volume_t<double> volume = {
        ourSize, ourSize, ourSize,
        current3DCorrelation
    };

    std::vector<findpeaks::peak_3d<double> > peaks = findpeaks::persistance3d(volume);
    std::vector<translationPeak3D> tmpTranslations;
    int numberOfPeaks = 0;
    for (const auto &p: peaks) {
        //calculation of level, that is a potential translation
        double currentPersistence;
        double levelPotential;
        if (p.persistence == INFINITY) {
            currentPersistence = 1;
            levelPotential = 1;
        } else {
            currentPersistence = p.persistence;
            levelPotential = currentPersistence * sqrt(p.birth_level) *
                             Eigen::Vector3d((double) ((int) p.birth_position.x - (int) p.death_position.x),
                                             (double) ((int) p.birth_position.y - (int) p.death_position.y),
                                             (double) ((int) p.birth_position.z - (int) p.death_position.z)).norm() /
                             this->N;
        }


        Eigen::Vector3d subPixelPeak = subPixelComputation(inputcorrelation, dimensionN,
                                                     p.birth_position.x,
                                                     p.birth_position.y,
                                                     p.birth_position.z);




        translationPeak3D tmpTranslationPeak{};
        tmpTranslationPeak.xTranslation =
        -(double) ((double) subPixelPeak[0] - (double) (dimensionN - 1.0) / 2.0) * cellSize;
        tmpTranslationPeak.yTranslation =
                -(double) ((double) subPixelPeak[1] - (double) (dimensionN - 1.0) / 2.0) * cellSize;
        tmpTranslationPeak.zTranslation =
                -(double) ((double) subPixelPeak[2] - (double) (dimensionN - 1.0) / 2.0) * cellSize;
        // tmpTranslationPeak.xTranslation =
        //         -(double) ((double) p.birth_position.x - (double) (dimensionN - 1.0) / 2.0) * cellSize;
        // tmpTranslationPeak.yTranslation =
        //         -(double) ((double) p.birth_position.y - (double) (dimensionN - 1.0) / 2.0) * cellSize;
        // tmpTranslationPeak.zTranslation =
        //         -(double) ((double) p.birth_position.z - (double) (dimensionN - 1.0) / 2.0) * cellSize;
        tmpTranslationPeak.persistence = currentPersistence;
        tmpTranslationPeak.correlationHeight = current3DCorrelation[generalHelpfulTools::index3D(
            int(p.birth_position.x),
            int(p.birth_position.y),
            int(p.birth_position.z),
            dimensionN)];
        tmpTranslationPeak.levelPotential = levelPotential;
        if (levelPotential > level_potential_translation) {
            tmpTranslations.push_back(tmpTranslationPeak);
            numberOfPeaks++;
        }
    }
    free(current3DCorrelation);
    return (tmpTranslations);
}

std::vector<rotationPeak4D>
softRegistrationClass3D::peakDetectionOf4DCorrelationFindPeaksLibrary(const double *inputcorrelation, long dimensionN,
                                                                      double cellSize) {
    double *current4DCorrelation;
    current4DCorrelation = (double *) malloc(
        sizeof(double) * dimensionN * dimensionN * dimensionN * dimensionN);

    double maxValue = 0;
    double minValue = INFINITY;
    //copy data
    for (int j = 0; j < dimensionN; j++) {
        for (int i = 0; i < dimensionN; i++) {
            for (int k = 0; k < dimensionN; k++) {
                for (int l = 0; l < dimensionN; l++) {
                    double currentCorrelation = inputcorrelation[j + dimensionN * i +
                                                                 k * dimensionN * dimensionN +
                                                                 l * dimensionN * dimensionN * dimensionN];
                    current4DCorrelation[j + dimensionN * i + k * dimensionN * dimensionN +
                                         l * dimensionN * dimensionN * dimensionN] = currentCorrelation;
                    if (currentCorrelation > maxValue) {
                        maxValue = currentCorrelation;
                    }
                    if (currentCorrelation < minValue) {
                        minValue = currentCorrelation;
                    }
                }
            }
        }
    }
    //normalize data
    for (int j = 0; j < dimensionN; j++) {
        for (int i = 0; i < dimensionN; i++) {
            for (int k = 0; k < dimensionN; k++) {
                for (int l = 0; l < dimensionN; l++) {
                    current4DCorrelation[j + dimensionN * i + k * dimensionN * dimensionN +
                                         l * dimensionN * dimensionN * dimensionN] =
                            (current4DCorrelation[j + dimensionN * i + k * dimensionN * dimensionN +
                                                  l * dimensionN * dimensionN * dimensionN] - minValue) /
                            (maxValue - minValue);
                }
            }
        }
    }
    //    cv::Mat magTMP1(dimensionN, dimensionN, CV_64F, current3DCorrelation);
    //    cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);

    //    cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
    //                                            cv::Size(ceil(0.02 * this->N), ceil(0.02 * this->N)));
    //    cv::morphologyEx(magTMP1, magTMP1, cv::MORPH_TOPHAT, element);

    size_t ourSize = dimensionN;
    findpeaks::volume4D_t<double> volume = {
        ourSize, ourSize, ourSize, ourSize,
        current4DCorrelation
    };

    std::vector<findpeaks::peak_4d<double> > peaks = findpeaks::persistance4d(volume);
    std::vector<rotationPeak4D> tmpRotations;
    std::cout << peaks.size() << std::endl;
    int numberOfPeaks = 0;
    for (const auto &p: peaks) {
        //calculation of level, that is a potential translation
        double levelPotential = p.persistence * sqrt(p.birth_level) *
                                Eigen::Vector4d((double) ((int) p.birth_position.x - (int) p.death_position.x),
                                                (double) ((int) p.birth_position.y - (int) p.death_position.y),
                                                (double) ((int) p.birth_position.z - (int) p.death_position.z),
                                                (double) ((int) p.birth_position.w - (int) p.death_position.w)).norm() /
                                double(dimensionN) / 1.73205080757;

        //        std::cout << p.persistence << std::endl;
        //        std::cout << Eigen::Vector4d((double) ((int) p.birth_position.x - (int) p.death_position.x),
        //                                     (double) ((int) p.birth_position.y - (int) p.death_position.y),
        //                                     (double) ((int) p.birth_position.z - (int) p.death_position.z),
        //                                     (double) ((int) p.birth_position.w - (int) p.death_position.w)).norm()
        //                  << std::endl;
        //        std::cout << sqrt(p.birth_level) << std::endl;


        rotationPeak4D tmpTranslationPeak{};
        tmpTranslationPeak.x = double(p.birth_position.x);
        tmpTranslationPeak.y = double(p.birth_position.y);
        tmpTranslationPeak.z = double(p.birth_position.z);
        tmpTranslationPeak.w = double(p.birth_position.w);
        tmpTranslationPeak.persistence = p.persistence;
        tmpTranslationPeak.correlationHeight = current4DCorrelation[p.birth_position.w +
                                                                    dimensionN * p.birth_position.z +
                                                                    p.birth_position.y * dimensionN *
                                                                    dimensionN +
                                                                    p.birth_position.x * dimensionN * dimensionN *
                                                                    dimensionN];
        tmpTranslationPeak.levelPotential = levelPotential;
        if (levelPotential > 0.01) {
            tmpRotations.push_back(tmpTranslationPeak);
            //            std::cout << "test" << std::endl;
            numberOfPeaks++;
        }
    }
    free(current4DCorrelation);
    return (tmpRotations);
}

std::vector<rotationPeak4D>
softRegistrationClass3D::peakDetectionOf4DCorrelationWithKDTreeFindPeaksLibrary(
    std::vector<My4DPoint> listOfQuaternionCorrelation, double level_potential_rotation) {
    double *current1DCorrelation;
    current1DCorrelation = (double *) malloc(
        sizeof(double) * listOfQuaternionCorrelation.size());

    for (int i = 0; i < listOfQuaternionCorrelation.size(); i++) {
        current1DCorrelation[i] = listOfQuaternionCorrelation[i].correlation;
    }


    findpeaks::oneDimensionalList_t<double> inputCorrelations = {
        listOfQuaternionCorrelation.size(),
        current1DCorrelation
    };
    //    kdt::KDTree<My4DPoint> kdtree(listOfQuaternionCorrelation);

    std::vector<findpeaks::peak_1d<double> > peaks = findpeaks::persistanceQuaternionsKDTree(inputCorrelations,
        this->lookupTableForCorrelations);
    std::vector<rotationPeak4D> tmpRotations;

    int numberOfPeaks = 0;
    for (const auto &p: peaks) {
        My4DPoint birthPositionPoint = listOfQuaternionCorrelation[p.birth_position.x];
        // My4DPoint deathPositionPoint = listOfQuaternionCorrelation[p.death_position.x];
        double currentPersistence;
        if (p.persistence == INFINITY) {
            currentPersistence = 1;
        } else {
            currentPersistence = p.persistence;
        }
        double levelPotential = currentPersistence * currentPersistence * p.birth_level * p.birth_level;

        rotationPeak4D tmpTranslationPeak{};
        tmpTranslationPeak.x = birthPositionPoint[1];
        tmpTranslationPeak.y = birthPositionPoint[2];
        tmpTranslationPeak.z = birthPositionPoint[3];
        tmpTranslationPeak.w = birthPositionPoint[0];
        tmpTranslationPeak.persistence = p.persistence;
        tmpTranslationPeak.correlationHeight = birthPositionPoint.correlation;
        tmpTranslationPeak.levelPotential = levelPotential;
        if (levelPotential > level_potential_rotation) {
            tmpRotations.push_back(tmpTranslationPeak);
            //            std::cout << "test" << std::endl;
            numberOfPeaks++;
        }
    }
    free(current1DCorrelation);
    return (tmpRotations);
}

int softRegistrationClass3D::getSizeOfRegistration() const {
    return this->N;
}




Eigen::Vector3d softRegistrationClass3D::subPixelComputation(const double *inputcorrelation, int dimensionN,
                                                               double xPosition, double yPosition, double zPosition) const{

    int nSubPixel = 3;
    std::vector<Eigen::Vector3d> listOfPoints;
    // listOfPoints.push_back(Eigen::Vector3d(xPosition, yPosition, zPosition));
    listOfPoints.push_back(Eigen::Vector3d(xPosition + nSubPixel, yPosition, zPosition));
    listOfPoints.push_back(Eigen::Vector3d(xPosition - nSubPixel, yPosition, zPosition));
    listOfPoints.push_back(Eigen::Vector3d(xPosition, yPosition + nSubPixel, zPosition));
    listOfPoints.push_back(Eigen::Vector3d(xPosition, yPosition - nSubPixel, zPosition));
    listOfPoints.push_back(Eigen::Vector3d(xPosition, yPosition, zPosition + nSubPixel));
    listOfPoints.push_back(Eigen::Vector3d(xPosition, yPosition, zPosition - nSubPixel));
    for (int xPos = -(nSubPixel-1); xPos <= (nSubPixel-1); ++xPos) {
        for (int yPos = -(nSubPixel-1); yPos <= (nSubPixel-1); ++yPos) {
            for (int zPos = -(nSubPixel-1); zPos <=(nSubPixel-1); ++zPos) {
                listOfPoints.push_back(Eigen::Vector3d(xPosition+xPos, yPosition+yPos, zPosition+zPos));
            }
        }
    }

    double totalWeight = 0;
    double totalXPos = 0;
    double totalYPos = 0;
    double totalZPos = 0;
    for (const Eigen::Vector3d &point: listOfPoints) {
        double weight = inputcorrelation[generalHelpfulTools::index3D(int(point[0]), int(point[1]), int(point[2]),
                                                                      dimensionN)];
        totalWeight += weight;
        totalXPos += point[0] * weight;
        totalYPos += point[1] * weight;
        totalZPos += point[2] * weight;
    }

    double centerX = totalXPos / totalWeight;
    double centerY = totalYPos / totalWeight;
    double centerZ = totalZPos / totalWeight;


    // translationPeak3D tmpTranslationPeak{};

    return Eigen::Vector3d(centerX, centerY, centerZ);
}
