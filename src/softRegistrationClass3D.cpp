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
    for (int i = 0; i < this->N * this->N * this->N; i++) {
        voxelDataTMP[i] = voxelData[i];
    }

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


std::vector<rotationPeak3D>
softRegistrationClass3D::sofftRegistrationVoxel3DListOfPossibleRotations(double voxelData1Input[],
                                                                         double voxelData2Input[], bool debug,
                                                                         bool multipleRadii, bool useClahe,
                                                                         bool useHamming) {

    double maximumScan1Magnitude = this->getSpectrumFromVoxelData3D(voxelData1Input, this->magnitude1,
                                                                    this->phase1, false);
    double maximumScan2Magnitude = this->getSpectrumFromVoxelData3D(voxelData2Input, this->magnitude2,
                                                                    this->phase2, false);


    if (debug) {
        std::ofstream myFile1, myFile2, myFile3, myFile4, myFile5, myFile6;
        myFile1.open(
                "/home/tim-external/Documents/matlabTestEnvironment/registrationFourier/3D/csvFiles/magnitudeFFTW1.csv");
        myFile2.open(
                "/home/tim-external/Documents/matlabTestEnvironment/registrationFourier/3D/csvFiles/phaseFFTW1.csv");
        myFile3.open(
                "/home/tim-external/Documents/matlabTestEnvironment/registrationFourier/3D/csvFiles/voxelDataFFTW1.csv");
        myFile4.open(
                "/home/tim-external/Documents/matlabTestEnvironment/registrationFourier/3D/csvFiles/magnitudeFFTW2.csv");
        myFile5.open(
                "/home/tim-external/Documents/matlabTestEnvironment/registrationFourier/3D/csvFiles/phaseFFTW2.csv");
        myFile6.open(
                "/home/tim-external/Documents/matlabTestEnvironment/registrationFourier/3D/csvFiles/voxelDataFFTW2.csv");
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < N; k++) {
                    myFile1 << this->magnitude1[j + N * i + N * N * k]; // real part
                    myFile1 << "\n";
                    myFile2 << this->phase1[j + N * i + N * N * k]; // imaginary part
                    myFile2 << "\n";
                    myFile3 << voxelData1Input[j + N * i + N * N * k]; // imaginary part
                    myFile3 << "\n";
                    myFile4 << this->magnitude2[j + N * i + N * N * k]; // real part
                    myFile4 << "\n";
                    myFile5 << this->phase2[j + N * i + N * N * k]; // imaginary part
                    myFile5 << "\n";
                    myFile6 << voxelData2Input[j + N * i + N * N * k]; // imaginary part
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


    double globalMaximumMagnitude;
    if (maximumScan2Magnitude < maximumScan1Magnitude) {
        globalMaximumMagnitude = maximumScan1Magnitude;
    } else {
        globalMaximumMagnitude = maximumScan2Magnitude;
    }
    double minMagnitude = 20000000;
    //normalize and shift both spectrums
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {

                int index = k + N * j + N * N * i;

                int indexX = (N / 2 + i) % N;
                int indexY = (N / 2 + j) % N;
                int indexZ = (N / 2 + k) % N;

                int indexshift = indexZ + N * indexY + N * N * indexX;
                if (minMagnitude > magnitude1[index]) {
                    minMagnitude = magnitude1[index];
                }
                if (minMagnitude > magnitude2[index]) {
                    minMagnitude = magnitude2[index];
                }
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

    int minRNumber = 1 + floor(N * 0.05);//was 4
    int maxRNumber = N / 2 - floor(N * 0.05);
    int bandwidth = N / 2;

//    minRNumber = maxRNumber - 1;

    double minValue = INFINITY;
    double maxValue = 0;
    for (int r = minRNumber; r < maxRNumber; r++) {
        for (int i = 0; i < 2 * bandwidth; i++) {
            for (int j = 0; j < 2 * bandwidth; j++) {
//                for (int k = 0; k < 2 * bandwidth; k++) {

                double theta = thetaIncrement3D((double) i, bandwidth);
                double phi = phiIncrement3D((double) j, bandwidth);

                int xIndex = std::round(r * std::sin(theta) * std::cos(phi) + bandwidth) - 1;
                int yIndex = std::round(r * std::sin(theta) * std::sin(phi) + bandwidth) - 1;
                int zIndex = std::round(r * std::cos(theta) + bandwidth) - 1;

                //  double hammingCoeff = 25.0/46.0-(1.0-25.0/46.0)*cos(2*M_PI*k/(2*bandwidth));
                double hammingCoeff = 1;


                resampledMagnitudeSO3_1TMP[j + N * i] +=
                        255 * magnitude1Shifted[zIndex + N * yIndex + N * N * xIndex] * hammingCoeff;
                resampledMagnitudeSO3_2TMP[j + N * i] +=
                        255 * magnitude2Shifted[zIndex + N * yIndex + N * N * xIndex] * hammingCoeff;
//                }
                if (resampledMagnitudeSO3_1TMP[j + N * i] <= 0) {
                    std::cout << resampledMagnitudeSO3_1TMP[j + N * i] << std::endl;
                }
                if (resampledMagnitudeSO3_2TMP[j + N * i] <= 0) {
                    std::cout << resampledMagnitudeSO3_2TMP[j + N * i] << std::endl;
                }


                if (minValue > resampledMagnitudeSO3_1TMP[j + N * i]) {
                    minValue = resampledMagnitudeSO3_1TMP[j + N * i];
                }
                if (maxValue < resampledMagnitudeSO3_1TMP[j + N * i]) {
                    maxValue = resampledMagnitudeSO3_1TMP[j + N * i];
                }

                if (minValue > resampledMagnitudeSO3_2TMP[j + N * i]) {
                    minValue = resampledMagnitudeSO3_2TMP[j + N * i];
                }
                if (maxValue < resampledMagnitudeSO3_2TMP[j + N * i]) {
                    maxValue = resampledMagnitudeSO3_2TMP[j + N * i];
                }

            }
        }
    }
    // Here a CLAHE or something simular cound be done.

    for (int i = 0; i < 2 * bandwidth; i++) {
        for (int j = 0; j < 2 * bandwidth; j++) {
            this->resampledMagnitudeSO3_1TMP[j + N * i] =
                    (this->resampledMagnitudeSO3_1TMP[j + N * i] - minValue) / (maxValue - minValue);
            this->resampledMagnitudeSO3_2TMP[j + N * i] =
                    (this->resampledMagnitudeSO3_2TMP[j + N * i] - minValue) / (maxValue - minValue);

        }
    }
    if (debug) {
        std::ofstream myFile1, myFile2;
        myFile1.open(
                "/home/tim-external/Documents/matlabTestEnvironment/registrationFourier/3D/csvFiles/resampledMagnitudeSO3_1.csv");
        myFile2.open(
                "/home/tim-external/Documents/matlabTestEnvironment/registrationFourier/3D/csvFiles/resampledMagnitudeSO3_2.csv");
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < N; k++) {
                    myFile1 << this->resampledMagnitudeSO3_1TMP[j + N * i + N * N * k]; // real part
                    myFile1 << "\n";
                    myFile2 << this->resampledMagnitudeSO3_2TMP[j + N * i + N * N * k]; // imaginary part
                    myFile2 << "\n";
                }
            }
        }
        myFile1.close();
        myFile2.close();
    }



    this->sofftCorrelationObject3D.correlationOfTwoSignalsInSO3(resampledMagnitudeSO3_1TMP, resampledMagnitudeSO3_2TMP,
                                                                resultingCorrelationComplex);

    if (debug) {
        std::ofstream myFile1,myFile2;
        myFile1.open(
                "/home/tim-external/Documents/matlabTestEnvironment/registrationFourier/3D/csvFiles/resultingCorrelationReal.csv");
        myFile2.open(
                "/home/tim-external/Documents/matlabTestEnvironment/registrationFourier/3D/csvFiles/resultingCorrelationComplex.csv");
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < N; k++) {
                    myFile1 << this->resultingCorrelationComplex[j + N * i + N * N * k][0]; // real part
                    myFile1 << "\n";
                    myFile2 << this->resultingCorrelationComplex[j + N * i + N * N * k][1]; // imaginary part
                    myFile2 << "\n";
                }
            }
        }
        myFile1.close();
        myFile2.close();
    }

    std::vector<rotationPeak3D> potentialRotations = this->peakDetectionOf3DCorrelationFindPeaksLibrary(
            resultingCorrelationComplex,
            1);


    return potentialRotations;

}

std::vector<rotationPeak3D>
softRegistrationClass3D::peakDetectionOf3DCorrelationFindPeaksLibrary(fftw_complex *inputcorrelation, double cellSize) {

    double *current3DCorrelation;
    current3DCorrelation = (double *) malloc(
            sizeof(double) * this->correlationN * this->correlationN * this->correlationN);

    double maxValue = 0;
    double minValue = INFINITY;
    //copy data
    for (int j = 0; j < this->correlationN; j++) {
        for (int i = 0; i < this->correlationN; i++) {
            for (int k = 0; k < this->correlationN; k++) {


                double currentCorrelation = (inputcorrelation[j + this->correlationN * i +
                                                              k * this->correlationN * this->correlationN][0]) *
                                            (inputcorrelation[j + this->correlationN * i +
                                                              k * this->correlationN * this->correlationN][0]) +
                                            (inputcorrelation[j + this->correlationN * i +
                                                              k * this->correlationN * this->correlationN][1]) *
                                            (inputcorrelation[j + this->correlationN * i +
                                                              k * this->correlationN * this->correlationN][1]);
                current3DCorrelation[j + this->correlationN * i +
                                     k * this->correlationN * this->correlationN] = currentCorrelation;
                if (current3DCorrelation[j + this->correlationN * i + k * this->correlationN * this->correlationN] >
                    maxValue) {
                    maxValue = current3DCorrelation[j + this->correlationN * i +
                                                    k * this->correlationN * this->correlationN];
                }
                if (current3DCorrelation[j + this->correlationN * i + k * this->correlationN * this->correlationN] <
                    minValue) {
                    minValue = current3DCorrelation[j + this->correlationN * i +
                                                    k * this->correlationN * this->correlationN];
                }
            }
        }
    }
    //normalize data
    for (int j = 0; j < this->correlationN; j++) {
        for (int i = 0; i < this->correlationN; i++) {
            for (int k = 0; k < this->correlationN; k++) {
                current3DCorrelation[j + this->correlationN * i + k * this->correlationN * this->correlationN] =
                        (current3DCorrelation[j + this->correlationN * i +
                                              k * this->correlationN * this->correlationN] - minValue) /
                        (maxValue - minValue);
            }
        }
    }
    cv::Mat magTMP1(this->correlationN, this->correlationN, CV_64F, current3DCorrelation);
//    cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);

//    cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
//                                            cv::Size(ceil(0.02 * this->correlationN), ceil(0.02 * this->correlationN)));
//    cv::morphologyEx(magTMP1, magTMP1, cv::MORPH_TOPHAT, element);

    size_t ourSize = this->correlationN;
    findpeaks::volume_t<double> volume = {
            ourSize, ourSize, ourSize,
            current3DCorrelation
    };

    std::vector<findpeaks::peak_3d<double>> peaks = findpeaks::persistance3d(volume);
    std::vector<rotationPeak3D> tmpTranslations;
    std::cout << peaks.size() << std::endl;
    int numberOfPeaks = 0;
    for (const auto &p: peaks) {
        //calculation of level, that is a potential translation
        double levelPotential = p.persistence * sqrt(p.birth_level) *
                                Eigen::Vector3d((double) ((int) p.birth_position.x - (int) p.death_position.x),
                                                (double) ((int) p.birth_position.y - (int) p.death_position.y),
                                                (double) ((int) p.birth_position.z - (int) p.death_position.z)).norm() /
                                this->correlationN / 1.73205080757;

        std::cout << p.persistence << std::endl;
        std::cout << Eigen::Vector3d((double) ((int) p.birth_position.x - (int) p.death_position.x),
                                     (double) ((int) p.birth_position.y - (int) p.death_position.y),
                                     (double) ((int) p.birth_position.z - (int) p.death_position.z)).norm()
                  << std::endl;
        std::cout << sqrt(p.birth_level) << std::endl;


        rotationPeak3D tmpTranslationPeak;
        tmpTranslationPeak.z1Rotation = p.birth_position.x;
        tmpTranslationPeak.yRotation = p.birth_position.y;
        tmpTranslationPeak.z2Rotation = p.birth_position.z;
        tmpTranslationPeak.persistence = p.persistence;
        tmpTranslationPeak.correlationHeight = current3DCorrelation[p.birth_position.z +
                                                                    this->correlationN * p.birth_position.y +
                                                                    p.birth_position.x * this->correlationN *
                                                                    this->correlationN];
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


