//
// Created by aya on 08.12.23.
//

#include "softRegistrationClass3D.h"

#define DEBUG_RESULTS_3D "/home/tim-external/volumeROS/src/fsregistration/debug_results/3d/data/"

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
                                                        bool benchmark, double sizeVoxel ,
                                                        double r_min ,
                                                        double r_max,
                                                        double level_potential_rotation,
                                                        double level_potential_translation,
                                                        bool set_r_manual, int normalization,
                                                        bool useSimpleRotationPeak, bool useSimpleTranslationPeak) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    if (benchmark) {
        std::cerr << "\n=== BENCHMARK: 3D One Solution ===" << std::endl;
    }

    double spectrumTime = 0, softDescriptorTime = 0, rotationCorrelationTime = 0;
    double overheadTime = 0, peakDetectionTime = 0, plottingTime = 0;
    double voxelRotationTime = 0, fft1Time = 0, fft2Time = 0;
    double correlationTime = 0, ifftTime = 0, fftshiftTime = 0;
    double translationPeakDetectionTime = 0, totalTransTime = 0;

    auto spectrumStart = std::chrono::high_resolution_clock::now();


    double maximumScan1Magnitude = this->getSpectrumFromVoxelData3D(voxelData1Input, this->magnitude1,
                                                                    this->phase1, false);
    double maximumScan2Magnitude = this->getSpectrumFromVoxelData3D(voxelData2Input, this->magnitude2,
                                                                    this->phase2, false);

    auto spectrumEnd = std::chrono::high_resolution_clock::now();
    if (benchmark) {
        spectrumTime = std::chrono::duration<double, std::milli>(spectrumEnd - spectrumStart).count();
    }


    if (debug) {
        std::ofstream myFile1, myFile2, myFile3, myFile4, myFile5, myFile6;
        myFile1.open(
            DEBUG_RESULTS_3D "magnitudeFFTW1.csv");
        myFile2.open(
            DEBUG_RESULTS_3D "phaseFFTW1.csv");
        myFile3.open(
            DEBUG_RESULTS_3D "voxelDataFFTW1.csv");
        myFile4.open(
            DEBUG_RESULTS_3D "magnitudeFFTW2.csv");
        myFile5.open(
            DEBUG_RESULTS_3D "phaseFFTW2.csv");
        myFile6.open(
            DEBUG_RESULTS_3D "voxelDataFFTW2.csv");
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
            DEBUG_RESULTS_3D "resampledMagnitudeSO3_1.csv");
        myFile2.open(
            DEBUG_RESULTS_3D "resampledMagnitudeSO3_2.csv");
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

    if (benchmark) {
        auto softDescriptorEnd = std::chrono::high_resolution_clock::now();
        softDescriptorTime = std::chrono::duration<double, std::milli>(softDescriptorEnd - spectrumEnd).count();
    }


    auto rotationCorrelationStart = std::chrono::high_resolution_clock::now();
    this->sofftCorrelationObject3D.correlationOfTwoSignalsInSO3(resampledMagnitudeSO3_1, resampledMagnitudeSO3_2,
                                                                 resultingCorrelationComplex);
    if (benchmark) {
        auto rotationCorrelationEnd = std::chrono::high_resolution_clock::now();
        rotationCorrelationTime = std::chrono::duration<double, std::milli>(rotationCorrelationEnd - rotationCorrelationStart).count();
    }

    if (debug) {
        std::ofstream myFile1, myFile2;
        myFile1.open(
            DEBUG_RESULTS_3D "resultingCorrelationReal.csv");
        myFile2.open(
            DEBUG_RESULTS_3D "resultingCorrelationComplex.csv");
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

    auto overheadStart = std::chrono::high_resolution_clock::now();

    std::vector<rotationPeak4D> potentialRotationsTMP;
    if (useSimpleRotationPeak) {
        auto peakDetectionStart = std::chrono::high_resolution_clock::now();
        potentialRotationsTMP = peakDetectionOf4DCorrelationSimpleMaxRaw(
            resultingCorrelationComplex, bwOut, N);
        if (benchmark) {
            auto peakDetectionEnd = std::chrono::high_resolution_clock::now();
            peakDetectionTime = std::chrono::duration<double, std::milli>(peakDetectionEnd - peakDetectionStart).count();
        }
    } else {
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

        if (benchmark) {
            auto overheadEnd = std::chrono::high_resolution_clock::now();
            overheadTime = std::chrono::duration<double, std::milli>(overheadEnd - overheadStart).count();
        }

        auto peakDetectionStart = std::chrono::high_resolution_clock::now();
        potentialRotationsTMP = this->peakDetectionOf4DCorrelationWithKDTreeFindPeaksLibrary(
            listOfQuaternionCorrelation, level_potential_rotation);
        if (benchmark) {
            auto peakDetectionEnd = std::chrono::high_resolution_clock::now();
            peakDetectionTime = std::chrono::duration<double, std::milli>(peakDetectionEnd - peakDetectionStart).count();
        }
    }
    if (useSimpleRotationPeak && benchmark) {
        auto overheadEnd = std::chrono::high_resolution_clock::now();
        overheadTime = std::chrono::duration<double, std::milli>(overheadEnd - overheadStart).count();
    }



    auto plottingStart = std::chrono::high_resolution_clock::now();
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

    if (benchmark) {
        auto plottingEnd = std::chrono::high_resolution_clock::now();
        plottingTime = std::chrono::duration<double, std::milli>(plottingEnd - plottingStart).count();
    }

    auto totalTransStart = std::chrono::high_resolution_clock::now();

    transformationPeakfs3D bestFittingSolution;

    auto voxelRotationStart = std::chrono::high_resolution_clock::now();

    double *voxelData2Rotated;
    voxelData2Rotated = (double *) malloc(sizeof(double) * this->N * this->N * this->N);
    for (int i = 0; i < this->N * this->N * this->N; i++) {
        voxelData2Rotated[i] = 0;
    }
   Eigen::Quaterniond currentRotation(bestFittingPeak.w, bestFittingPeak.x,
                                        bestFittingPeak.y, bestFittingPeak.z);
    Eigen::Matrix3d R = currentRotation.toRotationMatrix();

    for (int i = 0; i < this->N; i++) {
        double xi = i - this->N / 2;
        for (int j = 0; j < this->N; j++) {
            double yj = j - this->N / 2;
            for (int k = 0; k < this->N; k++) {
                double zk = k - this->N / 2;

                double lx = R(0,0)*xi + R(0,1)*yj + R(0,2)*zk;
                double ly = R(1,0)*xi + R(1,1)*yj + R(1,2)*zk;
                double lz = R(2,0)*xi + R(2,1)*yj + R(2,2)*zk;

                int index = generalHelpfulTools::index3D(i, j, k, this->N);
                double occupancyValue = getPixelValueInterpolated(Eigen::Vector3d(lx, ly, lz), voxelData2Input, this->N);
                voxelData2Rotated[index] = occupancyValue;
            }
        }
    }

    if (benchmark) {
        auto voxelRotationEnd = std::chrono::high_resolution_clock::now();
        voxelRotationTime = std::chrono::duration<double, std::milli>(voxelRotationEnd - voxelRotationStart).count();
    }

    auto fft1Start = std::chrono::high_resolution_clock::now();
    this->getSpectrumFromVoxelData3DCorrelation(voxelData1Input,
                                                this->magnitude1Correlation,
                                                this->phase1Correlation,
                                                false);
    if (benchmark) {
        auto fft1End = std::chrono::high_resolution_clock::now();
        fft1Time = std::chrono::duration<double, std::milli>(fft1End - fft1Start).count();
    }

    auto fft2Start = std::chrono::high_resolution_clock::now();
    this->getSpectrumFromVoxelData3DCorrelation(voxelData2Rotated,
                                                this->magnitude2Correlation,
                                                this->phase2Correlation,
                                                false);
    if (benchmark) {
        auto fft2End = std::chrono::high_resolution_clock::now();
        fft2Time = std::chrono::duration<double, std::milli>(fft2End - fft2Start).count();
    }

    if (debug) {
        std::ofstream myFile1, myFile2, myFile3;
        myFile1.open(
            DEBUG_RESULTS_3D "magnitudeFFTW2Rotated.csv");
        myFile2.open(
            DEBUG_RESULTS_3D "phaseFFTW2Rotated.csv");

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
            DEBUG_RESULTS_3D "voxelDataFFTW2Rotated.csv");
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

    auto correlationStart = std::chrono::high_resolution_clock::now();
    //calculate correlation of spectrums
    for (int i = 0; i < this->correlationN; i++) {
        for (int j = 0; j < this->correlationN; j++) {
            for (int k = 0; k < this->correlationN; k++) {
                int indexX = i;
                int indexY = j;
                int indexZ = k;
                int index = generalHelpfulTools::index3D(indexX, indexY, indexZ, this->correlationN);

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
    if (benchmark) {
        auto correlationEnd = std::chrono::high_resolution_clock::now();
        correlationTime = std::chrono::duration<double, std::milli>(correlationEnd - correlationStart).count();
    }

    auto ifftStart = std::chrono::high_resolution_clock::now();
    // back fft
    fftw_execute(planFourierToVoxel3DCorrelation);
    if (benchmark) {
        auto ifftEnd = std::chrono::high_resolution_clock::now();
        ifftTime = std::chrono::duration<double, std::milli>(ifftEnd - ifftStart).count();
    }

    auto fftshiftStart = std::chrono::high_resolution_clock::now();
    // fftshift and calc magnitude
    double maximumCorrelationTranslation = 0;
    double minimumCorrelationTranslation = INFINITY;

    for (int i = 0; i < this->correlationN; i++) {
        for (int j = 0; j < this->correlationN; j++) {
            for (int k = 0; k < this->correlationN; k++) {
                int indexX = ((this->correlationN / 2) + i + this->correlationN) % this->correlationN;
                int indexY = ((this->correlationN / 2) + j + this->correlationN) % this->correlationN;
                int indexZ = ((this->correlationN / 2) + k + this->correlationN) % this->correlationN;
                int indexShifted = generalHelpfulTools::index3D(indexX, indexY, indexZ, this->correlationN);
                int index = generalHelpfulTools::index3D(i, j, k, this->correlationN);
                double normalizationFactorForCorrelation;
                switch (normalization) {
                    case 0:
                        normalizationFactorForCorrelation = 1;
                        break;
                    case 1:
                        normalizationFactorForCorrelation =
                                1 / normalizationFactorCalculation(indexX, indexY, indexZ, this->correlationN);
                        break;
                    case 2:
                        normalizationFactorForCorrelation =
                                1 / normalizationFactorCalculation(indexX, indexY, indexZ, this->correlationN);
                        normalizationFactorForCorrelation = normalizationFactorForCorrelation*normalizationFactorForCorrelation;
                        break;
                    default:
                        std::cout << "normalization has to be 0,1 but was: " << normalization << std::endl;
                        exit(-1);
                }

                resultingCorrelationDouble[indexShifted] =
                        normalizationFactorForCorrelation *
                        NORM(resultingShiftPeaks3DCorrelation[index]);

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
        if (benchmark) {
            auto fftshiftEnd = std::chrono::high_resolution_clock::now();
            fftshiftTime = std::chrono::duration<double, std::milli>(fftshiftEnd - fftshiftStart).count();
        }

    if (debug) {
        std::ofstream myFile10;
        myFile10.open(
            DEBUG_RESULTS_3D "resultingCorrelationShift.csv");
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

    auto translationPeakDetectionStart = std::chrono::high_resolution_clock::now();
    std::vector<translationPeak3D> resulting3DPeakList;
    if (useSimpleTranslationPeak) {
        resulting3DPeakList = peakDetectionOf3DCorrelationSimpleMax(
            resultingCorrelationDouble, this->correlationN, sizeVoxel);
    } else {
        resulting3DPeakList = peakDetectionOf3DCorrelationFindPeaksLibrary(
            resultingCorrelationDouble, this->correlationN, sizeVoxel, level_potential_translation);
    }
    if (benchmark) {
        auto translationPeakDetectionEnd = std::chrono::high_resolution_clock::now();
        translationPeakDetectionTime = std::chrono::duration<double, std::milli>(translationPeakDetectionEnd - translationPeakDetectionStart).count();
    }
    std::cout << "number of solutions Translation: " << resulting3DPeakList.size()<< std::endl;
    transformationPeakfs3D tmpSolution;
    tmpSolution.potentialRotation = bestFittingPeak;

    int indexBestFittingSolution= 0;
    double score=INFINITY;
    Eigen::Vector3d initGuessPositionEigen(initGuessPosition.x(), initGuessPosition.y(), initGuessPosition.z());

    for (int i = 0; i < resulting3DPeakList.size(); i++) {
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

    if (benchmark) {
        auto totalTransEnd = std::chrono::high_resolution_clock::now();
        totalTransTime = std::chrono::duration<double, std::milli>(totalTransEnd - totalTransStart).count();
    }

    if (benchmark) {
        auto totalEnd = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

        int numTransPeaks = tmpSolution.potentialTranslations.size();
        std::cerr << "  --- 3D One Solution Summary ---" << std::endl;
        std::cerr << "    Rotation peaks found:           1" << std::endl;
        std::cerr << "    Translation peaks found:        " << numTransPeaks << std::endl;
        std::cerr << "    3D Spectrum (FFT):              " << spectrumTime << " ms" << std::endl;
        std::cerr << "    SOFT descriptor projection:     " << softDescriptorTime << " ms" << std::endl;
        std::cerr << "    SOFT correlation (FFT):         " << rotationCorrelationTime << " ms" << std::endl;
        std::cerr << "    Quaternion preparation:         " << overheadTime << " ms" << std::endl;
        std::cerr << "    Rotation peak detection:        " << peakDetectionTime << " ms" << std::endl;
        std::cerr << "    Rotation selection:             " << plottingTime << " ms" << std::endl;
        std::cerr << "    --- Translation breakdown (1 solution) ---" << std::endl;
        std::cerr << "      Voxel rotation:               " << voxelRotationTime << " ms" << std::endl;
        std::cerr << "      FFT1:                         " << fft1Time << " ms" << std::endl;
        std::cerr << "      FFT2:                         " << fft2Time << " ms" << std::endl;
        std::cerr << "      Complex correlation:          " << correlationTime << " ms" << std::endl;
        std::cerr << "      IFFT:                         " << ifftTime << " ms" << std::endl;
        std::cerr << "      fftshift + magnitude:         " << fftshiftTime << " ms" << std::endl;
        std::cerr << "      Peak detection:               " << translationPeakDetectionTime << " ms" << std::endl;
        std::cerr << "    Total translation:              " << totalTransTime << " ms" << std::endl;
        std::cerr << "    Total time:                     " << totalTime << " ms" << std::endl;
    }

    return tmpSolution;
}


std::vector<transformationPeakfs3D>
softRegistrationClass3D::sofftRegistrationVoxel3DListOfPossibleTransformations(double voxelData1Input[],
                                                                                 double voxelData2Input[], bool debug,
                                                                                 bool useClahe, bool benchmark,
                                                                                 double sizeVoxel, double r_min,
                                                                                 double r_max,
                                                                                 double level_potential_rotation,
                                                                                 double level_potential_translation,
                                                                                 bool set_r_manual, int normalization,
                                                                                 bool useSimpleRotationPeak, bool useSimpleTranslationPeak,
                                                                                 BenchmarkTimings3D* timings) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    double spectrumTime = 0, softDescriptorTime = 0, rotationCorrelationTime = 0;
    double overheadTime = 0, peakDetectionTime = 0, plottingTime = 0;
    double totalAllTransTime = 0;
    double transVoxelRotationTime = 0, transFft1Time = 0, transFft2Time = 0;
    double transCorrelationTime = 0, transIfftTime = 0, transFftshiftTime = 0;
    double transPeakDetectionTime = 0;
    std::vector<double> transPerSolutionTimes;

    auto spectrumStart = std::chrono::high_resolution_clock::now();


    double maximumScan1Magnitude = this->getSpectrumFromVoxelData3D(voxelData1Input, this->magnitude1,
                                                                    this->phase1, false);
    double maximumScan2Magnitude = this->getSpectrumFromVoxelData3D(voxelData2Input, this->magnitude2,
                                                                    this->phase2, false);

    auto spectrumEnd = std::chrono::high_resolution_clock::now();
    if (benchmark || timings) {
        spectrumTime = std::chrono::duration<double, std::milli>(spectrumEnd - spectrumStart).count();
    }


    if (debug) {
        std::ofstream myFile1, myFile2, myFile3, myFile4, myFile5, myFile6;
        myFile1.open(
            DEBUG_RESULTS_3D "magnitudeFFTW1.csv");
        myFile2.open(
            DEBUG_RESULTS_3D "phaseFFTW1.csv");
        myFile3.open(
            DEBUG_RESULTS_3D "voxelDataFFTW1.csv");
        myFile4.open(
            DEBUG_RESULTS_3D "magnitudeFFTW2.csv");
        myFile5.open(
            DEBUG_RESULTS_3D "phaseFFTW2.csv");
        myFile6.open(
            DEBUG_RESULTS_3D "voxelDataFFTW2.csv");
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

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        if (useClahe) {
            clahe->apply(magTMP1, magTMP1);
            clahe->apply(magTMP2, magTMP2);
        }
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
            DEBUG_RESULTS_3D "resampledMagnitudeSO3_1.csv");
        myFile2.open(
            DEBUG_RESULTS_3D "resampledMagnitudeSO3_2.csv");
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

    if (benchmark || timings) {
        auto softDescriptorEnd = std::chrono::high_resolution_clock::now();
        softDescriptorTime = std::chrono::duration<double, std::milli>(softDescriptorEnd - spectrumEnd).count();
    }


    auto rotationCorrelationStart = std::chrono::high_resolution_clock::now();
    this->sofftCorrelationObject3D.correlationOfTwoSignalsInSO3(resampledMagnitudeSO3_1, resampledMagnitudeSO3_2,
                                                                 resultingCorrelationComplex);
    if (benchmark || timings) {
        auto rotationCorrelationEnd = std::chrono::high_resolution_clock::now();
        rotationCorrelationTime = std::chrono::duration<double, std::milli>(rotationCorrelationEnd - rotationCorrelationStart).count();
    }

    if (debug) {
        std::ofstream myFile1, myFile2;
        myFile1.open(
            DEBUG_RESULTS_3D "resultingCorrelationReal.csv");
        myFile2.open(
            DEBUG_RESULTS_3D "resultingCorrelationComplex.csv");
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

    auto overheadStart = std::chrono::high_resolution_clock::now();

    std::vector<rotationPeak4D> potentialRotationsTMP;
    if (useSimpleRotationPeak) {
        auto peakDetectionStart = std::chrono::high_resolution_clock::now();
        potentialRotationsTMP = peakDetectionOf4DCorrelationSimpleMaxRaw(
            resultingCorrelationComplex, bwOut, N);
        if (benchmark || timings) {
            auto peakDetectionEnd = std::chrono::high_resolution_clock::now();
            peakDetectionTime = std::chrono::duration<double, std::milli>(peakDetectionEnd - peakDetectionStart).count();
        }
    } else {
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

        if (benchmark || timings) {
            auto overheadEnd = std::chrono::high_resolution_clock::now();
            overheadTime = std::chrono::duration<double, std::milli>(overheadEnd - overheadStart).count();
        }

        auto peakDetectionStart = std::chrono::high_resolution_clock::now();
        potentialRotationsTMP = this->peakDetectionOf4DCorrelationWithKDTreeFindPeaksLibrary(
            listOfQuaternionCorrelation, level_potential_rotation);
        if (benchmark || timings) {
            auto peakDetectionEnd = std::chrono::high_resolution_clock::now();
            peakDetectionTime = std::chrono::duration<double, std::milli>(peakDetectionEnd - peakDetectionStart).count();
        }
    }
    if (useSimpleRotationPeak && benchmark) {
        auto overheadEnd = std::chrono::high_resolution_clock::now();
        overheadTime = std::chrono::duration<double, std::milli>(overheadEnd - overheadStart).count();
    }

    auto plottingStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < potentialRotationsTMP.size(); i++) {
        Eigen::Quaterniond rotationQuat(potentialRotationsTMP[i].w, potentialRotationsTMP[i].x,
                                        potentialRotationsTMP[i].y, potentialRotationsTMP[i].z);
        Eigen::Vector3d rpyCurrentRot = generalHelpfulTools::getRollPitchYaw(rotationQuat);
        if (debug) {
                std::cout << i << " , " << potentialRotationsTMP[i].levelPotential << " , "
            << potentialRotationsTMP[i].correlationHeight
            << " , " << potentialRotationsTMP[i].persistence << " , " << rpyCurrentRot[0] * 180 / M_PI << " , "
            << rpyCurrentRot[1] * 180 / M_PI << " , " << rpyCurrentRot[2] * 180 / M_PI << " , "
            << potentialRotationsTMP[i].x << " , " << potentialRotationsTMP[i].y << " , "
            << potentialRotationsTMP[i].z << " , " << potentialRotationsTMP[i].w << std::endl;
        }
    }
    if (benchmark || timings) {
        auto plottingEnd = std::chrono::high_resolution_clock::now();
        plottingTime = std::chrono::duration<double, std::milli>(plottingEnd - plottingStart).count();
    }

    auto totalAllTransStart = std::chrono::high_resolution_clock::now();
    std::vector<transformationPeakfs3D> allSolutions;
    for (int p = 0; p < potentialRotationsTMP.size(); p++) {
        auto solStart = std::chrono::high_resolution_clock::now();

        double *voxelData2Rotated;
        voxelData2Rotated = (double *) malloc(sizeof(double) * this->N * this->N * this->N);
        for (int i = 0; i < this->N * this->N * this->N; i++) {
            voxelData2Rotated[i] = 0;
        }
       Eigen::Quaterniond currentRotation(potentialRotationsTMP[p].w, potentialRotationsTMP[p].x,
                                            potentialRotationsTMP[p].y, potentialRotationsTMP[p].z);
        Eigen::Matrix3d R = currentRotation.toRotationMatrix();
        if (debug) {
            std::cout << currentRotation << std::endl;
        }

        auto voxelRotStart = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < this->N; i++) {
            double xi = i - this->N / 2;
            for (int j = 0; j < this->N; j++) {
                double yj = j - this->N / 2;
                for (int k = 0; k < this->N; k++) {
                    double zk = k - this->N / 2;

                    double lx = R(0,0)*xi + R(0,1)*yj + R(0,2)*zk;
                    double ly = R(1,0)*xi + R(1,1)*yj + R(1,2)*zk;
                    double lz = R(2,0)*xi + R(2,1)*yj + R(2,2)*zk;

                    int index = generalHelpfulTools::index3D(i, j, k, this->N);
                    double occupancyValue = getPixelValueInterpolated(Eigen::Vector3d(lx, ly, lz), voxelData2Input, this->N);
                    voxelData2Rotated[index] = occupancyValue;
                }
            }
        }

        if (benchmark || timings) {
            auto voxelRotEnd = std::chrono::high_resolution_clock::now();
            transVoxelRotationTime += std::chrono::duration<double, std::milli>(voxelRotEnd - voxelRotStart).count();
        }

        auto fft1Start = std::chrono::high_resolution_clock::now();
       this->getSpectrumFromVoxelData3DCorrelation(voxelData1Input,
                                                     this->magnitude1Correlation,
                                                     this->phase1Correlation,
                                                     false);
        if (benchmark || timings) {
            auto fft1End = std::chrono::high_resolution_clock::now();
            transFft1Time += std::chrono::duration<double, std::milli>(fft1End - fft1Start).count();
            fft1Start = std::chrono::high_resolution_clock::now();
        }
        this->getSpectrumFromVoxelData3DCorrelation(voxelData2Rotated,
                                                     this->magnitude2Correlation,
                                                     this->phase2Correlation,
                                                     false);
        if (benchmark || timings) {
            auto fft2End = std::chrono::high_resolution_clock::now();
            transFft2Time += std::chrono::duration<double, std::milli>(fft2End - fft1Start).count();
        }

        if (debug) {
            std::ofstream myFile1, myFile2, myFile3;
            myFile1.open(
                DEBUG_RESULTS_3D "magnitudeFFTW2Rotated" +
                std::to_string(p) + ".csv");
            myFile2.open(
                DEBUG_RESULTS_3D "phaseFFTW2Rotated" +
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
                DEBUG_RESULTS_3D "voxelDataFFTW2Rotated" +
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
        auto corrStart = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < this->correlationN; i++) {
            for (int j = 0; j < this->correlationN; j++) {
                for (int k = 0; k < this->correlationN; k++) {
                    int indexX = i;
                    int indexY = j;
                    int indexZ = k;
                    int index = generalHelpfulTools::index3D(indexX, indexY, indexZ, this->correlationN);

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
        if (benchmark || timings) {
            auto corrEnd = std::chrono::high_resolution_clock::now();
            transCorrelationTime += std::chrono::duration<double, std::milli>(corrEnd - corrStart).count();
        }

        auto ifftStart = std::chrono::high_resolution_clock::now();
        // back fft
        fftw_execute(planFourierToVoxel3DCorrelation);
        if (benchmark || timings) {
            auto ifftEnd = std::chrono::high_resolution_clock::now();
            transIfftTime += std::chrono::duration<double, std::milli>(ifftEnd - ifftStart).count();
        }

        auto fftshiftStart = std::chrono::high_resolution_clock::now();
        // fftshift and calc magnitude
        double maximumCorrelationTranslation = 0;
        double minimumCorrelationTranslation = INFINITY;

        for (int i = 0; i < this->correlationN; i++) {
            for (int j = 0; j < this->correlationN; j++) {
                for (int k = 0; k < this->correlationN; k++) {
                    int indexX = ((this->correlationN / 2) + i + this->correlationN) % this->correlationN;
                    int indexY = ((this->correlationN / 2) + j + this->correlationN) % this->correlationN;
                    int indexZ = ((this->correlationN / 2) + k + this->correlationN) % this->correlationN;
                    int indexShifted = generalHelpfulTools::index3D(indexX, indexY, indexZ, this->correlationN);
                    int index = generalHelpfulTools::index3D(i, j, k, this->correlationN);
                    double normalizationFactorForCorrelation;
                    switch (normalization) {
                        case 0:
                            normalizationFactorForCorrelation = 1;
                            break;
                        case 1:
                            normalizationFactorForCorrelation =
                                    1 / normalizationFactorCalculation(indexX, indexY, indexZ, this->correlationN);
                            break;
                        case 2:
                            normalizationFactorForCorrelation =
                                    1 / normalizationFactorCalculation(indexX, indexY, indexZ, this->correlationN);
                            normalizationFactorForCorrelation = normalizationFactorForCorrelation*normalizationFactorForCorrelation;
                            break;
                        default:
                            std::cout << "normalization has to be 0,1 but was: " << normalization << std::endl;
                            exit(-1);
                    }

                    resultingCorrelationDouble[indexShifted] =
                            normalizationFactorForCorrelation *
                            NORM(resultingShiftPeaks3DCorrelation[index]);

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

        if (benchmark || timings) {
            auto fftshiftEnd = std::chrono::high_resolution_clock::now();
            transFftshiftTime += std::chrono::duration<double, std::milli>(fftshiftEnd - fftshiftStart).count();
        }

        if (debug) {
            std::ofstream myFile10;
            myFile10.open(
                DEBUG_RESULTS_3D "resultingCorrelationShift" +
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
        auto peakDetStart = std::chrono::high_resolution_clock::now();
        std::vector<translationPeak3D> resulting3DPeakList;
        if (useSimpleTranslationPeak) {
            resulting3DPeakList = peakDetectionOf3DCorrelationSimpleMax(
                resultingCorrelationDouble, this->correlationN, sizeVoxel);
        } else {
            resulting3DPeakList = peakDetectionOf3DCorrelationFindPeaksLibrary(
                resultingCorrelationDouble, this->correlationN, sizeVoxel, level_potential_translation);
        }
        if (benchmark || timings) {
            auto peakDetEnd = std::chrono::high_resolution_clock::now();
            transPeakDetectionTime += std::chrono::duration<double, std::milli>(peakDetEnd - peakDetStart).count();
        }
        transformationPeakfs3D tmpSolution;
        tmpSolution.potentialRotation = potentialRotationsTMP[p];
        for (int i = 0; i < resulting3DPeakList.size(); i++) {
            resulting3DPeakList[i].correlationHeight = resulting3DPeakList[i].correlationHeight *
                                                       (maximumCorrelationTranslation - minimumCorrelationTranslation) +
                                                       minimumCorrelationTranslation;
            if (debug) {
                std::cout << p << " , " << i << " , " << resulting3DPeakList[i].levelPotential << " , "
                << resulting3DPeakList[i].correlationHeight << " , " << resulting3DPeakList[i].persistence
                << " , " << resulting3DPeakList[i].xTranslation << " , "
                << resulting3DPeakList[i].yTranslation
                << " , " << resulting3DPeakList[i].zTranslation << std::endl;
            }


            tmpSolution.potentialTranslations.push_back(resulting3DPeakList[i]);
        }


        free(voxelData2Rotated);


        allSolutions.push_back(tmpSolution);

        if (benchmark || timings) {
            auto transPerSolutionEnd = std::chrono::high_resolution_clock::now();
            double transPerSolutionTime = std::chrono::duration<double, std::milli>(transPerSolutionEnd - solStart).count();
            transPerSolutionTimes.push_back(transPerSolutionTime);
            if (benchmark && !timings) {
                std::cerr << "  Translation solution [" << p << "]: " << transPerSolutionTime << " ms" << std::endl;
            }
        }
    }

    if (benchmark || timings) {
        auto totalAllTransEnd = std::chrono::high_resolution_clock::now();
        totalAllTransTime = std::chrono::duration<double, std::milli>(totalAllTransEnd - totalAllTransStart).count();

        auto totalEnd = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
        int numSolutions = potentialRotationsTMP.size();

        int totalTransPeaks = 0;
        for (const auto& sol : allSolutions) {
            totalTransPeaks += sol.potentialTranslations.size();
        }

        if (timings) {
            timings->spectrumTime = spectrumTime;
            timings->softDescriptorTime = softDescriptorTime;
            timings->rotationCorrelationTime = rotationCorrelationTime;
            timings->overheadTime = overheadTime;
            timings->peakDetectionTime = peakDetectionTime;
            timings->plottingTime = plottingTime;
            timings->totalAllTransTime = totalAllTransTime;
            timings->transVoxelRotationTime = transVoxelRotationTime;
            timings->transFft1Time = transFft1Time;
            timings->transFft2Time = transFft2Time;
            timings->transCorrelationTime = transCorrelationTime;
            timings->transIfftTime = transIfftTime;
            timings->transFftshiftTime = transFftshiftTime;
            timings->transPeakDetectionTime = transPeakDetectionTime;
            timings->transPerSolutionTimes = transPerSolutionTimes;
            timings->numSolutions = numSolutions;
            timings->totalTransPeaks = totalTransPeaks;
            timings->totalTime = totalTime;
        }

        if (benchmark && !timings) {
            std::cerr << "  --- 3D All Solutions Summary ---" << std::endl;
            std::cerr << "    Rotation peaks found:           " << numSolutions << std::endl;
            std::cerr << "    Translation peaks found:        " << totalTransPeaks << std::endl;
            std::cerr << "    3D Spectrum (FFT):              " << spectrumTime << " ms" << std::endl;
            std::cerr << "    SOFT descriptor projection:     " << softDescriptorTime << " ms" << std::endl;
            std::cerr << "    SOFT correlation (FFT):         " << rotationCorrelationTime << " ms" << std::endl;
            std::cerr << "    Quaternion preparation:         " << overheadTime << " ms" << std::endl;
            std::cerr << "    Rotation peak detection:        " << peakDetectionTime << " ms" << std::endl;
            std::cerr << "    Solution printing:              " << plottingTime << " ms" << std::endl;
            std::cerr << "    --- Translation breakdown (" << numSolutions << " solutions) ---" << std::endl;
            if (numSolutions > 0) {
                double perSol = 1.0 / numSolutions;
                std::cerr << "      Voxel rotation:             " << transVoxelRotationTime << " ms (" << (transVoxelRotationTime * perSol) << " ms/sol)" << std::endl;
                std::cerr << "      FFT1:                       " << transFft1Time << " ms (" << (transFft1Time * perSol) << " ms/sol)" << std::endl;
                std::cerr << "      FFT2:                       " << transFft2Time << " ms (" << (transFft2Time * perSol) << " ms/sol)" << std::endl;
                std::cerr << "      Complex correlation:        " << transCorrelationTime << " ms (" << (transCorrelationTime * perSol) << " ms/sol)" << std::endl;
                std::cerr << "      IFFT:                       " << transIfftTime << " ms (" << (transIfftTime * perSol) << " ms/sol)" << std::endl;
                std::cerr << "      fftshift + magnitude:       " << transFftshiftTime << " ms (" << (transFftshiftTime * perSol) << " ms/sol)" << std::endl;
                std::cerr << "      Peak detection:             " << transPeakDetectionTime << " ms (" << (transPeakDetectionTime * perSol) << " ms/sol)" << std::endl;
            }
            std::cerr << "    Total translation:            " << totalAllTransTime << " ms" << std::endl;
            std::cerr << "    Total time:                   " << totalTime << " ms" << std::endl;
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
    int xDown = floor(positionVector.x());
    int yDown = floor(positionVector.y());
    int zDown = floor(positionVector.z());

    double dx = positionVector.x() - xDown;
    double dy = positionVector.y() - yDown;
    double dz = positionVector.z() - zDown;

    int xUp = xDown + 1;
    int yUp = yDown + 1;
    int zUp = zDown + 1;

    int off = dimensionN / 2;
    int idx000 = generalHelpfulTools::index3D(xDown + off, yDown + off, zDown + off, dimensionN);
    int idx100 = generalHelpfulTools::index3D(xUp   + off, yDown + off, zDown + off, dimensionN);
    int idx010 = generalHelpfulTools::index3D(xDown + off, yUp   + off, zDown + off, dimensionN);
    int idx110 = generalHelpfulTools::index3D(xUp   + off, yUp   + off, zDown + off, dimensionN);
    int idx001 = generalHelpfulTools::index3D(xDown + off, yDown + off, zUp   + off, dimensionN);
    int idx101 = generalHelpfulTools::index3D(xUp   + off, yDown + off, zUp   + off, dimensionN);
    int idx011 = generalHelpfulTools::index3D(xDown + off, yUp   + off, zUp   + off, dimensionN);
    int idx111 = generalHelpfulTools::index3D(xUp   + off, yUp   + off, zUp   + off, dimensionN);

    auto v = [&](int idx) -> double {
        return (idx == -1) ? 0.0 : volumeData[idx];
    };

    double v000 = v(idx000), v100 = v(idx100), v010 = v(idx010), v110 = v(idx110);
    double v001 = v(idx001), v101 = v(idx101), v011 = v(idx011), v111 = v(idx111);

    double v00 = (1.0 - dx) * v000 + dx * v100;
    double v10 = (1.0 - dx) * v010 + dx * v110;
    double v01 = (1.0 - dx) * v001 + dx * v101;
    double v11 = (1.0 - dx) * v011 + dx * v111;

    double v0 = (1.0 - dy) * v00 + dy * v10;
    double v1 = (1.0 - dy) * v01 + dy * v11;

    return (1.0 - dz) * v0 + dz * v1;
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
        double levelPotential = p.persistence * sqrt(p.birth_level) *
                                Eigen::Vector4d((double) ((int) p.birth_position.x - (int) p.death_position.x),
                                                (double) ((int) p.birth_position.y - (int) p.death_position.y),
                                                (double) ((int) p.birth_position.z - (int) p.death_position.z),
                                                (double) ((int) p.birth_position.w - (int) p.death_position.w)).norm() /
                                double(dimensionN) / 1.73205080757;


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

    std::vector<findpeaks::peak_1d<double> > peaks = findpeaks::persistanceQuaternionsKDTree(inputCorrelations,
        this->lookupTableForCorrelations);
    std::vector<rotationPeak4D> tmpRotations;

    int numberOfPeaks = 0;
    for (const auto &p: peaks) {
        My4DPoint birthPositionPoint = listOfQuaternionCorrelation[p.birth_position.x];
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
            numberOfPeaks++;
        }
    }
    free(current1DCorrelation);
    return (tmpRotations);
}

std::vector<rotationPeak4D>
softRegistrationClass3D::peakDetectionOf4DCorrelationSimpleMax(std::vector<My4DPoint> listOfQuaternionCorrelation) {
    std::vector<rotationPeak4D> tmpRotations;
    if (listOfQuaternionCorrelation.empty()) {
        return tmpRotations;
    }

    size_t maxIndex = 0;
    double maxCorrelation = listOfQuaternionCorrelation[0].correlation;
    for (size_t i = 1; i < listOfQuaternionCorrelation.size(); i++) {
        if (listOfQuaternionCorrelation[i].correlation > maxCorrelation) {
            maxCorrelation = listOfQuaternionCorrelation[i].correlation;
            maxIndex = i;
        }
    }

    My4DPoint bestPoint = listOfQuaternionCorrelation[maxIndex];
    rotationPeak4D peak{};
    peak.x = bestPoint[1];
    peak.y = bestPoint[2];
    peak.z = bestPoint[3];
    peak.w = bestPoint[0];
    peak.persistence = 1.0;
    peak.correlationHeight = bestPoint.correlation;
    peak.levelPotential = 1.0;
    tmpRotations.push_back(peak);
    return tmpRotations;
}

std::vector<rotationPeak4D>
softRegistrationClass3D::peakDetectionOf4DCorrelationSimpleMaxRaw(fftw_complex* resultingCorrelationComplex, int bwOut, int N) {
    std::vector<rotationPeak4D> tmpRotations;
    if (!resultingCorrelationComplex || bwOut <= 0) {
        return tmpRotations;
    }

    size_t maxIndex = 0;
    double maxCorrelation = NORM(resultingCorrelationComplex[0]);
    size_t totalSize = (size_t)bwOut * 2 * bwOut * 2 * bwOut * 2;
    for (size_t i = 1; i < totalSize; i++) {
        double currentCorr = NORM(resultingCorrelationComplex[i]);
        if (currentCorr > maxCorrelation) {
            maxCorrelation = currentCorr;
            maxIndex = i;
        }
    }

    int k = maxIndex / ((bwOut * 2) * (bwOut * 2));
    int j = (maxIndex / (bwOut * 2)) % (bwOut * 2);
    int i = maxIndex % (bwOut * 2);

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
        quaternionResult.w() = -quaternionResult.w();
        quaternionResult.x() = -quaternionResult.x();
        quaternionResult.y() = -quaternionResult.y();
        quaternionResult.z() = -quaternionResult.z();
    }

    rotationPeak4D peak{};
    peak.x = quaternionResult.x();
    peak.y = quaternionResult.y();
    peak.z = quaternionResult.z();
    peak.w = quaternionResult.w();
    peak.persistence = 1.0;
    peak.correlationHeight = maxCorrelation;
    peak.levelPotential = 1.0;
    tmpRotations.push_back(peak);
    return tmpRotations;
}

std::vector<translationPeak3D>
softRegistrationClass3D::peakDetectionOf3DCorrelationSimpleMax(const double *inputcorrelation, int dimensionN, double cellSize) const {
    std::vector<translationPeak3D> tmpTranslations;
    if (!inputcorrelation || dimensionN <= 0) {
        return tmpTranslations;
    }

    size_t maxIndex = 0;
    double maxValue = inputcorrelation[0];
    size_t totalSize = (size_t)dimensionN * dimensionN * dimensionN;
    for (size_t i = 1; i < totalSize; i++) {
        if (inputcorrelation[i] > maxValue) {
            maxValue = inputcorrelation[i];
            maxIndex = i;
        }
    }

    int maxX = maxIndex / (dimensionN * dimensionN);
    int maxY = (maxIndex / dimensionN) % dimensionN;
    int maxZ = maxIndex % dimensionN;

    Eigen::Vector3d subPixelPeak = subPixelComputation(inputcorrelation, dimensionN,
        (double)maxX, (double)maxY, (double)maxZ);

    translationPeak3D peak{};
    peak.xTranslation = -(double)((double)subPixelPeak[0] - (double)(dimensionN - 1.0) / 2.0) * cellSize;
    peak.yTranslation = -(double)((double)subPixelPeak[1] - (double)(dimensionN - 1.0) / 2.0) * cellSize;
    peak.zTranslation = -(double)((double)subPixelPeak[2] - (double)(dimensionN - 1.0) / 2.0) * cellSize;
    peak.persistence = 1.0;
    peak.correlationHeight = maxValue;
    peak.levelPotential = 1.0;
    tmpTranslations.push_back(peak);
    return tmpTranslations;
}

int softRegistrationClass3D::getSizeOfRegistration() const {
    return this->N;
}



Eigen::Vector3d softRegistrationClass3D::subPixelComputation(const double *inputcorrelation, int dimensionN,
                                                               double xPosition, double yPosition, double zPosition) const{

    int nSubPixel = 3;
    std::vector<Eigen::Vector3d> listOfPoints;
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


    return Eigen::Vector3d(centerX, centerY, centerZ);
}
