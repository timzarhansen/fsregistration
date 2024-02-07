//
// Created by aya on 08.12.23.
//

#include "softRegistrationClass3D.h"

//bool compareTwoAngleCorrelation3D(rotationPeak3D i1, rotationPeak3D i2) {
//    return (i1.angle < i2.angle);
//}

double thetaIncrement3D(double index, int bandwidth) {
    return M_PI * (2 * index + 1) / (4.0 * bandwidth);
}

double phiIncrement3D(double index, int bandwidth) {
    return M_PI * index / bandwidth;
}

double angleDifference3D(double angle1, double angle2) {//gives angle 1 - angle 2
    return atan2(sin(angle1 - angle2), cos(angle1 - angle2));
}

double
softRegistrationClass3D::getSpectrumFromVoxelData3D(double voxelData[], double magnitude[], double phase[],
                                                  bool gaussianBlur) {

    double *voxelDataTMP;
    voxelDataTMP = (double *) malloc(sizeof(double) * N * N * N);
    for (int i = 0; i < this->N * this->N; i++) {
        voxelDataTMP[i] = voxelData[i];
    }

//    if (gaussianBlur) {
//        cv::Mat magTMP1(this->N, this->N, CV_64F, voxelDataTMP);
//        //add gaussian blur
//        cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);
////        cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);
////        cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);
//    }



    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int index = k + j * N + i * N * N;
                inputSpacialData[index][0] = voxelDataTMP[index]; // real part
                inputSpacialData[index][1] = 0; // imaginary part
            }
        }
    }

    fftw_execute(planVoxelToFourier3D);


    double maximumMagnitude = 0;

    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int index = k + j * N + i * N * N;
                magnitude[index] = sqrt(spectrumOut[index][0] * spectrumOut[index][0] + spectrumOut[index][1] * spectrumOut[index][1]);
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


std::vector<rotationPeak3D>
softRegistrationClass3D::sofftRegistrationVoxel3DListOfPossibleRotations(double voxelData1Input[],
                                                                       double voxelData2Input[], bool debug,
                                                                       bool multipleRadii, bool useClahe,
                                                                       bool useHamming) {

    double maximumScan1Magnitude = this->getSpectrumFromVoxelData3D(voxelData1Input, this->magnitude1,
                                                                    this->phase1, false);
    double maximumScan2Magnitude = this->getSpectrumFromVoxelData3D(voxelData2Input, this->magnitude2,
                                                                    this->phase2, false);

    double globalMaximumMagnitude;
    if (maximumScan2Magnitude < maximumScan1Magnitude) {
        globalMaximumMagnitude = maximumScan1Magnitude;
    } else {
        globalMaximumMagnitude = maximumScan2Magnitude;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {

                int index = k + N * j + N * N * i;

                int indexX = (N / 2 + i) % N;
                int indexY = (N / 2 + j) % N;
                int indexZ = (N / 2 + k) % N;

                int indexshift = indexZ + N * indexY + N * N * indexX;

                magnitude1Shifted[indexshift] =
                        magnitude1[index] / globalMaximumMagnitude;
                magnitude2Shifted[indexshift] =
                        magnitude2[index] / globalMaximumMagnitude;
            }
        }
    }


    for (int i = 0; i < N * N * N; i++) {
        resampledMagnitudeSO3_1[i] = 0;
        resampledMagnitudeSO3_2[i] = 0;
        resampledMagnitudeSO3_1TMP[i] = 0;
        resampledMagnitudeSO3_2TMP[i] = 0;
    }

    int minRNumber = 1 + floor(N * 0.05);//was 4
    int maxRNumber = N / 2 - floor(N * 0.05);
    int bandwidth = N / 2;

    if(multipleRadii){
        minRNumber = maxRNumber - 1;
    }

    for (int r = minRNumber; r < maxRNumber; r++) {
        for (int i = 0; i < 2 * bandwidth; i++) {
            for (int j = 0; j < 2 * bandwidth; j++) {
                for (int k = 0; k < 2 * bandwidth; k++) {

                    double theta = thetaIncrement3D((double) i, bandwidth);
                    double phi = phiIncrement3D((double) j, bandwidth);

                    int xIndex = std::round(r * std::sin(theta) * std::cos(phi) + bandwidth) - 1;
                    int yIndex = std::round(r * std::sin(theta) * std::sin(phi) + bandwidth) - 1;
                    int zIndex = std::round(r * std::cos(theta) + bandwidth) - 1;

                    //  double hammingCoeff = 25.0/46.0-(1.0-25.0/46.0)*cos(2*M_PI*k/(2*bandwidth));
                    double hammingCoeff = 1;


                    resampledMagnitudeSO3_1TMP[k + N * j + N * N * i] +=
                            255 * magnitude1Shifted[zIndex + N * yIndex + N * N * xIndex] * hammingCoeff;
                    resampledMagnitudeSO3_2TMP[k + N * j + N * N * i] +=
                            255 * magnitude2Shifted[zIndex + N * yIndex + N * N * xIndex] * hammingCoeff;
                }
            }

        }
    }

    this->sofftCorrelationObject3D.correlationOfTwoSignalsInSO3(resampledMagnitudeSO3_1TMP, resampledMagnitudeSO3_2TMP,
                                                              resultingCorrelationComplex);

    double z1;
    double z2;
    double y;
    double maxCorrelation = 0;
    std::vector<rotationPeak3D> correlationOfAngle;
    for (int i = 0; i < N; i++) {
        for (int j = 0; i < N; i++) {
            for (int k = 0; i < N; i++) {
                z1 = j * 2.0 * M_PI / N;
                z2 = i * 2.0 * M_PI / N;
                y = M_PI * (2 * k + 1) / 2 / N;

                rotationPeak3D tmpHolding;
                tmpHolding.peakCorrelation = resultingCorrelationComplex[k + N * j + N * N * i][0]; // real part
                if (tmpHolding.peakCorrelation > maxCorrelation) {
                    maxCorrelation = tmpHolding.peakCorrelation;
                }
                // test on dataset with N and N/2 and 0   first test + n/2
                tmpHolding.theta = y;
                tmpHolding.phi = std::fmod(-(z1 + z2) + 6 * M_PI + 0.0 * M_PI / (N),
                                                   2 * M_PI);
                correlationOfAngle.push_back(tmpHolding);
            }
        }
    }
    // std::sort(correlationOfAngle.begin(), correlationOfAngle.end(), compareTwoAngleCorrelation3D);

    std::vector<float> correlationAveraged;
    std::vector<std::pair<float, float>> angleList;
    float maximumCorrelation = 0;
    float minimumCorrelation = INFINITY;

    double currentAverageTheta = correlationOfAngle[0].theta;
    double currentAveragePhi = correlationOfAngle[0].phi;
    int numberOfAngles = 1;
    double averageCorrelation = correlationOfAngle[0].peakCorrelation;

    for (int i = 1; i < correlationOfAngle.size(); i++) {

        double thetaDiff = std::abs(currentAverageTheta - correlationOfAngle[i].theta);
        double phiDiff = std::abs(currentAveragePhi - correlationOfAngle[i].phi);

        if (thetaDiff < 1.0 / N / 4.0 && phiDiff < 1.0 / N / 4.0) {
            numberOfAngles++;
            averageCorrelation += correlationOfAngle[i].peakCorrelation;
        } else {
            correlationAveraged.push_back(averageCorrelation / numberOfAngles);
            angleList.push_back({currentAverageTheta, currentAveragePhi});
            numberOfAngles = 1;
            averageCorrelation = correlationOfAngle[i].peakCorrelation;
            currentAverageTheta = correlationOfAngle[i].theta;
            currentAveragePhi = correlationOfAngle[i].phi;
            if (minimumCorrelation > correlationAveraged.back()) {
                minimumCorrelation = correlationAveraged.back();
            }
            if (maximumCorrelation < correlationAveraged.back()) {
                maximumCorrelation = correlationAveraged.back();
            }

        }
    }

    correlationAveraged.push_back((float) (averageCorrelation / numberOfAngles));

    if (minimumCorrelation > correlationAveraged.back()) {
        minimumCorrelation = correlationAveraged.back();
    }
    if (maximumCorrelation < correlationAveraged.back()) {
        maximumCorrelation = correlationAveraged.back();
    }

    angleList.push_back({currentAverageTheta, currentAveragePhi});

    for (int i = 0; i < correlationAveraged.size(); i++) {
        correlationAveraged[i] =
                (correlationAveraged[i] - minimumCorrelation) / (maximumCorrelation - minimumCorrelation);
    }


    auto minmax = std::min_element(correlationAveraged.begin(), correlationAveraged.end());
    long distanceToMinElement = std::distance(correlationAveraged.begin(), minmax);


    size_t ourSize = correlationAveraged.size();
    double *tmpDoubleArray = (double *) malloc(sizeof(double) * ourSize);
    for(int i = 0 ; i<ourSize;i++){
        tmpDoubleArray[i] = correlationAveraged[i];
    }
    findpeaks::image_t<double> image = {
            ourSize, 1,
            tmpDoubleArray
    };

    std::vector<findpeaks::peak_t<double>> peaks = findpeaks::persistance(image);

    free(tmpDoubleArray);

    std::vector<rotationPeak3D> returnVectorWithPotentialAngles;

    for (const auto& peak : peaks) {
        rotationPeak3D tmpPeak{};
        size_t peakIndex = peak.birth_position.x;

        if (peakIndex >= correlationAveraged.size()) {
            continue;
        }

        tmpPeak.theta = angleList[peakIndex].first; // Assuming first is theta
        tmpPeak.phi = angleList[peakIndex].second;

        tmpPeak.peakCorrelation = correlationAveraged[peakIndex];
        returnVectorWithPotentialAngles.push_back(tmpPeak);
    }

    return returnVectorWithPotentialAngles;

}