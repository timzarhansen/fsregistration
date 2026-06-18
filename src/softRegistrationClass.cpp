//
// Created by aya on 01.03.22.
//

#include "softRegistrationClass.h"
#include <chrono>
#include <filesystem>

#define DEBUG_RESULTS_2D "/home/tim-external/ros_ws/src/fsregistration/plotting_results/2d/data/"

bool compareTwoAngleCorrelation(rotationPeakfs2D i1, rotationPeakfs2D i2) {
    return (i1.angle < i2.angle);
}

std::vector<double> linspace(double start_in, double end_in, int num_in) {
    if (num_in < 0) {
        std::cout << "number of linspace negative" << std::endl;
        exit(-1);
    }
    std::vector<double> linspaced;

    double start = start_in;
    double end = end_in;
    auto num = (double)num_in;

    if (num == 0) { return linspaced; }
    if (num == 1) {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);//stepSize

    for (int i = 0; i < num - 1; ++i) {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
    // are exactly the same as the input
    return linspaced;
}

//bool compareTwoPeaks(indexPeak i1, indexPeak i2) {
//    return (i1.height > i2.height);
//}

double thetaIncrement(double index, int bandwidth) {
    return M_PI * (1 * index + 0) / (2.0 * bandwidth);
}

double phiIncrement(double index, int bandwidth) {
    return M_PI * index / bandwidth;
}

double angleDifference(double angle1, double angle2) {//gives angle 1 - angle 2
    return atan2(sin(angle1 - angle2), cos(angle1 - angle2));
}

double
softRegistrationClass::getSpectrumFromVoxelData2D(double voxelData[], double magnitude[], double phase[],
    bool gaussianBlur) {

    double* voxelDataTMP;
    voxelDataTMP = (double*)malloc(sizeof(double) * N * N);
    for (int i = 0; i < this->N * this->N; i++) {
        voxelDataTMP[i] = voxelData[i];
    }
    if (gaussianBlur) {
        cv::Mat magTMP1(this->N, this->N, CV_64F, voxelDataTMP);
        //add gaussian blur
        cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);
        //        cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);
        //        cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);
    }



    //from voxel data to row and input for fftw
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            inputSpacialData[j + N * i][0] = voxelDataTMP[j + N * i]; // real part
            inputSpacialData[j + N * i][1] = 0; // imaginary part
        }
    }

    fftw_execute(planVoxelToFourier2D);

    //calc magnitude and phase
    double maximumMagnitude = 0;

    //get magnitude and find maximum
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            magnitude[j + N * i] = sqrt(
                spectrumOut[j + N * i][0] *
                spectrumOut[j + N * i][0] +
                spectrumOut[j + N * i][1] *
                spectrumOut[j + N * i][1]); // real part;
            if (maximumMagnitude < magnitude[j + N * i]) {
                maximumMagnitude = magnitude[j + N * i];
            }

            phase[j + N * i] = atan2(spectrumOut[j + N * i][1], spectrumOut[j + N * i][0]);

        }
    }

    free(voxelDataTMP);
    return maximumMagnitude;
}

double
softRegistrationClass::getSpectrumFromVoxelData2DCorrelation(double voxelData[], fftw_complex* complexOut,
    bool gaussianBlur, double normalizationFactor) {
    if (gaussianBlur) {
        cv::Mat magTMP1(this->correlationN, this->correlationN, CV_64F, voxelData);
        cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);
    }

    for (int i = 0; i < this->correlationN; i++) {
        inputSpacialDataCorrelation[i][0] = 0;
        inputSpacialDataCorrelation[i][1] = 0;
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            inputSpacialDataCorrelation[(j + (int)(this->correlationN / 4)) +
                this->correlationN * (i + (int)(this->correlationN / 4))][0] =
                normalizationFactor * voxelData[j + N * i];
        }
    }

    fftw_execute(planVoxelToFourier2DCorrelation);

    if (complexOut) {
        int total = this->correlationN * this->correlationN;
        for (int i = 0; i < total; i++) {
            complexOut[i][0] = spectrumOutCorrelation[i][0];
            complexOut[i][1] = spectrumOutCorrelation[i][1];
        }
    }

    return 0;
}


double
softRegistrationClass::sofftRegistrationVoxel2DRotationOnly(double voxelData1Input[], double voxelData2Input[],
     double goodGuessAlpha, double& covariance, bool debug, double level_potential_rotation, bool useDirect) {
    auto allAnglesList = this->sofftRegistrationVoxel2DListOfPossibleRotations(voxelData1Input, voxelData2Input, debug, false, true, true, nullptr, level_potential_rotation, useDirect);
    auto result = findClosestRotationAngle(allAnglesList, goodGuessAlpha);
    covariance = result.covariance;
    return result.angle;
}

rotationPeakfs2D softRegistrationClass::findClosestRotationAngle(
    const std::vector<rotationPeakfs2D>& allAnglesList, double goodGuessAlpha) {
    int indexCorrectAngle = 0;
    for (int i = 1; i < (int)allAnglesList.size(); i++) {
        if (std::abs(angleDifference(allAnglesList[indexCorrectAngle].angle, goodGuessAlpha)) >
            std::abs(angleDifference(allAnglesList[i].angle, goodGuessAlpha))) {
            indexCorrectAngle = i;
        }
    }
    return allAnglesList[indexCorrectAngle];
}

std::vector<rotationPeakfs2D> softRegistrationClass::runRotationPeakDetection(
    const RotationCorrelationResult& result,
    BenchmarkTimings2D* timings,
    double level_potential_rotation) {
    std::vector<float> correlationAveraged = result.correlationAveraged;
    std::vector<float> angleList = result.angleList;

    auto peakDetStart = std::chrono::high_resolution_clock::now();

    if (level_potential_rotation <= 0.0) {
        auto minmax = std::min_element(correlationAveraged.begin(), correlationAveraged.end());
        long distanceToMinElement = std::distance(correlationAveraged.begin(), minmax);
        std::rotate(correlationAveraged.begin(), correlationAveraged.begin() + distanceToMinElement,
            correlationAveraged.end());

        std::vector<int> out;
        PeakFinder::findPeaks(correlationAveraged, out, true, 8.0);

        std::rotate(correlationAveraged.begin(),
            correlationAveraged.begin() + correlationAveraged.size() - distanceToMinElement,
            correlationAveraged.end());
        for (int i = 0; i < (int)out.size(); ++i) {
            out[i] = out[i] + (int)distanceToMinElement;
            if (out[i] >= (int)correlationAveraged.size()) {
                out[i] = out[i] - (int)correlationAveraged.size();
            }
        }

        std::vector<rotationPeakfs2D> returnVector;
        for (int i = 0; i < (int)out.size(); i++) {
            rotationPeakfs2D tmpPeak{};
            tmpPeak.angle = angleList[out[i]];
            tmpPeak.peakCorrelation = correlationAveraged[out[i]];
            tmpPeak.covariance = 0.05;
            tmpPeak.levelPotential = tmpPeak.peakCorrelation * tmpPeak.peakCorrelation;
            returnVector.push_back(tmpPeak);
        }
        if (timings) {
            timings->rotationPeakDetectionTime = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - peakDetStart).count();
        }
        return returnVector;
    }

    size_t n = correlationAveraged.size();
    if (n == 0) {
        if (timings) {
            timings->rotationPeakDetectionTime = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - peakDetStart).count();
        }
        return {};
    }

    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(),
        [&correlationAveraged](size_t a, size_t b) {
            return correlationAveraged[a] > correlationAveraged[b];
        });

    findpeaks::union_find ds(n);
    std::map<size_t, findpeaks::peak_1d<float>> peaks;

    {
        size_t p = indices[0];
        float v = correlationAveraged[p];
        constexpr float inf = std::numeric_limits<float>::has_infinity
            ? std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::max();
        ds.add(p, -1);
        peaks[p] = {v, inf, {(int)p}, {0}};
    }

    for (int i = 1; i < (int)indices.size(); i++) {
        size_t p = indices[i];
        float v = correlationAveraged[p];

        ds.add(p, -(i + 1));

        std::vector<size_t> neighbors;
        neighbors.push_back((p + n - 1) % n);
        neighbors.push_back((p + 1) % n);

        std::vector<size_t> reps;
        for (size_t q : neighbors) {
            if (!ds.contains(q)) {
                continue;
            }
            size_t rep = ds.find_set(q);
            bool found = false;
            for (size_t r : reps) {
                if (r == rep) { found = true; break; }
            }
            if (!found) reps.push_back(rep);
        }

        if (reps.empty()) continue;

        size_t oldp = reps[0];
        ds.join(oldp, p);
        for (size_t j = 1; j < reps.size(); j++) {
            size_t setid = ds.find_set(reps[j]);
            if (peaks.count(setid) == 0) {
                float persistenceVal = correlationAveraged[setid] - v;
                if (persistenceVal < 0) persistenceVal = -persistenceVal;
                peaks[setid] = {correlationAveraged[setid], persistenceVal, {(int)setid}, {(int)p}};
            }
            ds.join(oldp, reps[j]);
        }
    }

    std::vector<rotationPeakfs2D> returnVector;
    for (const auto& [key, val] : peaks) {
        float birthLevel = val.birth_level;
        float persistence = val.persistence;
        if (persistence == std::numeric_limits<float>::infinity()) {
            persistence = 1.0f;
        }
        double levelPotential = (double)persistence * (double)persistence * (double)birthLevel * (double)birthLevel;

        if (levelPotential > level_potential_rotation) {
            rotationPeakfs2D tmpPeak{};
            tmpPeak.angle = angleList[val.birth_position.x];
            tmpPeak.peakCorrelation = birthLevel;
            tmpPeak.covariance = 0.05;
            tmpPeak.levelPotential = levelPotential;
            returnVector.push_back(tmpPeak);
        }
    }

    std::sort(returnVector.begin(), returnVector.end(),
        [](const rotationPeakfs2D& a, const rotationPeakfs2D& b) {
            return a.peakCorrelation > b.peakCorrelation;
        });

    if (timings) {
        timings->rotationPeakDetectionTime = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - peakDetStart).count();
    }
    return returnVector;
}

RotationCorrelationResult
softRegistrationClass::computeRotationCorrelation1D(double voxelData1Input[], double voxelData2Input[],
     bool useDirect, bool multipleRadii, bool useClahe,
     bool useHamming, bool debug, BenchmarkTimings2D* timings,
     std::vector<rotationPeakfs2D>* outPeaks,
     double level_potential_rotation) {
    auto spectrumStart = std::chrono::high_resolution_clock::now();
    double maximumScan1Magnitude = this->getSpectrumFromVoxelData2D(voxelData1Input, this->magnitude1,
        this->phase1, false);
    double maximumScan2Magnitude = this->getSpectrumFromVoxelData2D(voxelData2Input, this->magnitude2,
        this->phase2, false);
    auto spectrumEnd = std::chrono::high_resolution_clock::now();
    if (timings) {
        timings->spectrumTime = std::chrono::duration<double, std::milli>(spectrumEnd - spectrumStart).count();
    }

    double globalMaximumMagnitude;
    if (maximumScan2Magnitude < maximumScan1Magnitude) {
        globalMaximumMagnitude = maximumScan1Magnitude;
    }
    else {
        globalMaximumMagnitude = maximumScan2Magnitude;
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            int indexX = (N / 2 + i) % N;
            int indexY = (N / 2 + j) % N;
            magnitude1Shifted[indexY + N * indexX] = magnitude1[j + N * i] / globalMaximumMagnitude;
            magnitude2Shifted[indexY + N * indexX] = magnitude2[j + N * i] / globalMaximumMagnitude;
        }
    }

    if (debug) {
        generalHelpfulTools::ensureDirectoryExists(DEBUG_RESULTS_2D);
        std::ofstream myFile1, myFile2, myFile3, myFile4, myFile5, myFile6;
        myFile1.open(DEBUG_RESULTS_2D "magnitudeFFTW1.csv");
        myFile2.open(DEBUG_RESULTS_2D "phaseFFTW1.csv");
        myFile3.open(DEBUG_RESULTS_2D "voxelDataFFTW1.csv");
        myFile4.open(DEBUG_RESULTS_2D "magnitudeFFTW2.csv");
        myFile5.open(DEBUG_RESULTS_2D "phaseFFTW2.csv");
        myFile6.open(DEBUG_RESULTS_2D "voxelDataFFTW2.csv");
        for (int j = 0; j < this->N; j++) {
            for (int i = 0; i < this->N; i++) {
                myFile1 << magnitude1Shifted[j + this->N * i] << "\n";
                myFile2 << phase1[j + this->N * i] << "\n";
                myFile3 << voxelData1Input[j + this->N * i] << "\n";
                myFile4 << magnitude2Shifted[j + this->N * i] << "\n";
                myFile5 << phase2[j + this->N * i] << "\n";
                myFile6 << voxelData2Input[j + this->N * i] << "\n";
            }
        }
        myFile1.close();
        myFile2.close();
        myFile3.close();
        myFile4.close();
        myFile5.close();
        myFile6.close();
    }

    auto descriptorStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N * N; i++) {
        resampledMagnitudeSO3_1[i] = 0;
        resampledMagnitudeSO3_2[i] = 0;
        resampledMagnitudeSO3_1TMP[i] = 0;
        resampledMagnitudeSO3_2TMP[i] = 0;
    }

    int minRNumber = 1 + floor(N * 0.05);
    int maxRNumber = N / 2 - floor(N * 0.05);
    int bandwidth = N / 2;

    if (multipleRadii) {
        minRNumber = maxRNumber - 1;
    }

    for (int r = minRNumber; r < maxRNumber; r++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                int xIndex = std::round((double)r * xAngle[j * N + k] + bandwidth) - 1;
                int yIndex = std::round((double)r * yAngle[j * N + k] + bandwidth) - 1;
                resampledMagnitudeSO3_1TMP[k + j * bandwidth * 2] = 255 * magnitude1Shifted[yIndex + N * xIndex];
                resampledMagnitudeSO3_2TMP[k + j * bandwidth * 2] = 255 * magnitude2Shifted[yIndex + N * xIndex];
            }
        }

        cv::Mat magTMP1(N, N, CV_64FC1, resampledMagnitudeSO3_1TMP);
        cv::Mat magTMP2(N, N, CV_64FC1, resampledMagnitudeSO3_2TMP);
        if (useClahe) {
            magTMP1.convertTo(magCLAHE1, CV_8UC1);
            magTMP2.convertTo(magCLAHE2, CV_8UC1);
            clahe->apply(magCLAHE1, magCLAHE1);
            clahe->apply(magCLAHE2, magCLAHE2);
        }
        else {
            magTMP1.convertTo(magCLAHE1, CV_8UC1);
            magTMP2.convertTo(magCLAHE2, CV_8UC1);
        }

        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                resampledMagnitudeSO3_1[j + k * bandwidth * 2] = resampledMagnitudeSO3_1[j + k * bandwidth * 2] +
                    ((double)magCLAHE1.data[j + k * bandwidth * 2]) /
                    255.0 * hammingCoeffs[k];
                resampledMagnitudeSO3_2[j + k * bandwidth * 2] = resampledMagnitudeSO3_2[j + k * bandwidth * 2] +
                    ((double)magCLAHE2.data[j + k * bandwidth * 2]) /
                    255.0 * hammingCoeffs[k];
            }
        }
    }
    auto descriptorEnd = std::chrono::high_resolution_clock::now();
    if (timings) {
        timings->softDescriptorTime = std::chrono::duration<double, std::milli>(descriptorEnd - descriptorStart).count();
    }

    if (debug) {
        generalHelpfulTools::ensureDirectoryExists(DEBUG_RESULTS_2D);
        std::ofstream myFile7, myFile8;
        myFile7.open(DEBUG_RESULTS_2D "resampledVoxel1.csv");
        myFile8.open(DEBUG_RESULTS_2D "resampledVoxel2.csv");
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                myFile7 << resampledMagnitudeSO3_1[j + k * bandwidth * 2] << "\n";
                myFile8 << resampledMagnitudeSO3_2[j + k * bandwidth * 2] << "\n";
            }
        }
        myFile7.close();
        myFile8.close();
    }

    RotationCorrelationResult result;

    if (!useDirect) {
        auto corrStart = std::chrono::high_resolution_clock::now();
        this->sofftCorrelationObject.correlationOfTwoSignalsInSO3(resampledMagnitudeSO3_1, resampledMagnitudeSO3_2,
            resultingCorrelationComplex);
        auto corrEnd = std::chrono::high_resolution_clock::now();
        if (timings) {
            timings->rotationCorrelationTime = std::chrono::duration<double, std::milli>(corrEnd - corrStart).count();
        }

        if (debug) {
            generalHelpfulTools::ensureDirectoryExists(DEBUG_RESULTS_2D);
            FILE* fp;
            fp = fopen(DEBUG_RESULTS_2D "resultCorrelation3D.csv", "w");
            for (int i = 0; i < 8 * bwOut * bwOut * bwOut; i++)
                fprintf(fp, "%.16f\n", resultingCorrelationComplex[i][0]);
            fclose(fp);
        }

        auto extractStart = std::chrono::high_resolution_clock::now();
        double z1, z2;
        std::vector<rotationPeakfs2D> correlationOfAngle;
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                z1 = j * 2.0 * M_PI / N;
                z2 = i * 2.0 * M_PI / N;
                rotationPeakfs2D tmpHolding;
                tmpHolding.peakCorrelation = resultingCorrelationComplex[j + N * (i + N * 0)][0];
                tmpHolding.angle = std::fmod(-(z1 + z2) + 6 * M_PI + 0.0 * M_PI / (N), 2 * M_PI);
                correlationOfAngle.push_back(tmpHolding);
            }
        }

        std::sort(correlationOfAngle.begin(), correlationOfAngle.end(), compareTwoAngleCorrelation);

        std::vector<float> correlationAveraged, angleList;
        float maximumCorrelation = 0;
        float minimumCorrelation = INFINITY;
        double currentAverageAngle = correlationOfAngle[0].angle;
        int numberOfAngles = 1;
        double averageCorrelation = correlationOfAngle[0].peakCorrelation;

        for (size_t i = 1; i < correlationOfAngle.size(); i++) {
            if (std::abs(currentAverageAngle - correlationOfAngle[i].angle) < 1.0 / N / 4.0) {
                numberOfAngles++;
                averageCorrelation += correlationOfAngle[i].peakCorrelation;
            }
            else {
                correlationAveraged.push_back((float)(averageCorrelation / numberOfAngles));
                angleList.push_back((float)currentAverageAngle);
                numberOfAngles = 1;
                averageCorrelation = correlationOfAngle[i].peakCorrelation;
                currentAverageAngle = correlationOfAngle[i].angle;
                if (minimumCorrelation > correlationAveraged.back()) minimumCorrelation = correlationAveraged.back();
                if (maximumCorrelation < correlationAveraged.back()) maximumCorrelation = correlationAveraged.back();
            }
        }
        correlationAveraged.push_back((float)(averageCorrelation / numberOfAngles));
        if (minimumCorrelation > correlationAveraged.back()) minimumCorrelation = correlationAveraged.back();
        if (maximumCorrelation < correlationAveraged.back()) maximumCorrelation = correlationAveraged.back();

        for (size_t i = 0; i < correlationAveraged.size(); i++) {
            correlationAveraged[i] = (correlationAveraged[i] - minimumCorrelation) / (maximumCorrelation - minimumCorrelation);
        }

        angleList.push_back((float)currentAverageAngle);
        result.correlationAveraged = correlationAveraged;
        result.angleList = angleList;
        auto extractEnd = std::chrono::high_resolution_clock::now();
        if (timings) {
            timings->rotationExtractionTime = std::chrono::duration<double, std::milli>(extractEnd - extractStart).count();
        }

        if (debug) {
            generalHelpfulTools::ensureDirectoryExists(DEBUG_RESULTS_2D);
            std::ofstream myFile9;
            myFile9.open(DEBUG_RESULTS_2D "resultingCorrelation1D.csv");
            for (size_t i = 0; i < correlationAveraged.size(); i++) {
                myFile9 << correlationAveraged[i] << "\n";
            }
            myFile9.close();
        }

    }
    else {
        auto corrStart = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N * N; i++) {
            this->sofftCorrelationObject.sigR[i] = resampledMagnitudeSO3_1[i];
            this->sofftCorrelationObject.sigI[i] = 0;
        }
        FST_semi_memo(this->sofftCorrelationObject.sigR, this->sofftCorrelationObject.sigI,
            this->sofftCorrelationObject.sigCoefR, this->sofftCorrelationObject.sigCoefI,
            bwIn, this->sofftCorrelationObject.seminaive_naive_table,
            (double*)this->sofftCorrelationObject.workspace2, 0, bwIn,
            &this->sofftCorrelationObject.dctPlan, &this->sofftCorrelationObject.fftPlan,
            this->sofftCorrelationObject.weights);

        for (int i = 0; i < N * N; i++) {
            this->sofftCorrelationObject.sigR[i] = resampledMagnitudeSO3_2[i];
            this->sofftCorrelationObject.sigI[i] = 0;
        }
        FST_semi_memo(this->sofftCorrelationObject.sigR, this->sofftCorrelationObject.sigI,
            this->sofftCorrelationObject.patCoefR, this->sofftCorrelationObject.patCoefI,
            bwIn, this->sofftCorrelationObject.seminaive_naive_table,
            (double*)this->sofftCorrelationObject.workspace2, 0, bwIn,
            &this->sofftCorrelationObject.dctPlan, &this->sofftCorrelationObject.fftPlan,
            this->sofftCorrelationObject.weights);
        auto corrEnd = std::chrono::high_resolution_clock::now();
        if (timings) {
            timings->rotationCorrelationTime = std::chrono::duration<double, std::milli>(corrEnd - corrStart).count();
        }

        if (debug) {
            generalHelpfulTools::ensureDirectoryExists(DEBUG_RESULTS_2D);
            std::ofstream coef1R, coef1I, coef2R, coef2I;
            coef1R.open(DEBUG_RESULTS_2D "sigCoefR_1angle.csv");
            coef1I.open(DEBUG_RESULTS_2D "sigCoefI_1angle.csv");
            coef2R.open(DEBUG_RESULTS_2D "patCoefR_1angle.csv");
            coef2I.open(DEBUG_RESULTS_2D "patCoefI_1angle.csv");
            for (int i = 0; i < bwIn * bwIn; i++) {
                coef1R << this->sofftCorrelationObject.sigCoefR[i] << "\n";
                coef1I << this->sofftCorrelationObject.sigCoefI[i] << "\n";
                coef2R << this->sofftCorrelationObject.patCoefR[i] << "\n";
                coef2I << this->sofftCorrelationObject.patCoefI[i] << "\n";
            }
            coef1R.close();
            coef1I.close();
            coef2R.close();
            coef2I.close();
        }

        auto extractStart = std::chrono::high_resolution_clock::now();
        int nAlpha = N;
        for (int m = 0; m < 2 * bwIn; m++) {
            this->PmR[m] = 0;
            this->PmI[m] = 0;
        }

        for (int l = 0; l < bwIn; l++) {
            for (int m = -l; m <= l; m++) {
                int bigL = bwIn - 1;
                int almIdx;
                if (m >= 0) {
                    almIdx = m * (bigL + 1) - (m * (m - 1) / 2) + (l - m);
                }
                else {
                    almIdx = (bigL * (bigL + 3) / 2) + 1 +
                        ((bigL + m) * (bigL + m + 1) / 2) + (l - abs(m));
                }
                this->PmR[m + bwIn] += this->sofftCorrelationObject.sigCoefR[almIdx] *
                    this->sofftCorrelationObject.patCoefR[almIdx] +
                    this->sofftCorrelationObject.sigCoefI[almIdx] *
                    this->sofftCorrelationObject.patCoefI[almIdx];
                this->PmI[m + bwIn] += this->sofftCorrelationObject.sigCoefI[almIdx] *
                    this->sofftCorrelationObject.patCoefR[almIdx] -
                    this->sofftCorrelationObject.sigCoefR[almIdx] *
                    this->sofftCorrelationObject.patCoefI[almIdx];
            }
        }

        for (int k = 0; k < nAlpha; k++) {
            double alpha = 2.0 * M_PI * k / nAlpha;
            double corrR = 0;
            for (int mPos = 0; mPos < 2 * bwIn; mPos++) {
                int m = mPos - bwIn;
                if (m == 0) continue;
                double phase = -m * alpha;
                corrR += this->PmR[mPos] * cos(phase) - this->PmI[mPos] * sin(phase);
            }
            this->correlation1D[k] = corrR;
        }

        if (debug) {
            generalHelpfulTools::ensureDirectoryExists(DEBUG_RESULTS_2D);
            std::ofstream corrFile, PmFile;
            corrFile.open(DEBUG_RESULTS_2D "correlation1D_1angle.csv");
            PmFile.open(DEBUG_RESULTS_2D "Pm_1angle.csv");
            for (int k = 0; k < nAlpha; k++) {
                corrFile << this->correlation1D[k] << "\n";
            }
            for (int m = 0; m < 2 * bwIn; m++) {
                PmFile << this->PmR[m] << "," << this->PmI[m] << "\n";
            }
            corrFile.close();
            PmFile.close();
        }

        std::vector<float> correlationAveraged(nAlpha);
        std::vector<float> angleList(nAlpha);
        for (int k = 0; k < nAlpha; k++) {
            correlationAveraged[k] = (float)this->correlation1D[k];
            angleList[k] = (float)(2.0 * M_PI * k / nAlpha);
        }

        float maximumCorrelation = *std::max_element(correlationAveraged.begin(), correlationAveraged.end());
        float minimumCorrelation = *std::min_element(correlationAveraged.begin(), correlationAveraged.end());
        for (size_t i = 0; i < correlationAveraged.size(); i++) {
            correlationAveraged[i] = (correlationAveraged[i] - minimumCorrelation) / (maximumCorrelation - minimumCorrelation);
        }

        result.correlationAveraged = correlationAveraged;
        result.angleList = angleList;
        auto extractEnd = std::chrono::high_resolution_clock::now();
        if (timings) {
            timings->rotationExtractionTime = std::chrono::duration<double, std::milli>(extractEnd - extractStart).count();
        }
    }

    if (outPeaks) {
        *outPeaks = this->runRotationPeakDetection(result, timings, level_potential_rotation);
        if (debug) {
            generalHelpfulTools::ensureDirectoryExists(DEBUG_RESULTS_2D);

        std::ofstream peakFile;
        peakFile.open(DEBUG_RESULTS_2D "rotationPeaks.csv");
        peakFile << "angle\tpeakCorrelation\tcovariance\tlevelPotential\tindex\n";
        for (int i = 0; i < (int)outPeaks->size(); i++) {
            peakFile << (*outPeaks)[i].angle << "\t"
                << (*outPeaks)[i].peakCorrelation << "\t"
                << (*outPeaks)[i].covariance << "\t"
                << (*outPeaks)[i].levelPotential << "\t"
                << i << "\n";
        }
        peakFile.close();

        std::ofstream corrCurveFile;
        corrCurveFile.open(DEBUG_RESULTS_2D "rotationCorrelation1D.csv");
        corrCurveFile << "index\tangle\tnormalizedCorrelation\n";
        for (int i = 0; i < (int)result.correlationAveraged.size(); i++) {
            corrCurveFile << i << "\t"
                << result.angleList[i] << "\t"
                << result.correlationAveraged[i] << "\n";
        }
        corrCurveFile.close();
        }
    }

    return result;
}

std::vector<rotationPeakfs2D>
softRegistrationClass::sofftRegistrationVoxel2DListOfPossibleRotations(double voxelData1Input[],
     double voxelData2Input[], bool debug,
     bool multipleRadii, bool useClahe,
     bool useHamming,
     BenchmarkTimings2D* timings,
     double level_potential_rotation,
     bool useDirect) {

    std::vector<rotationPeakfs2D> peaks;
    auto result = computeRotationCorrelation1D(voxelData1Input, voxelData2Input,
        useDirect, multipleRadii, useClahe, useHamming, debug, timings, &peaks, level_potential_rotation);

    return peaks;
}

std::pair<std::vector<float>, std::vector<float>>
softRegistrationClass::compute1AngleCorrelationArraySO3(double voxelData1Input[], double voxelData2Input[],
    bool multipleRadii, bool useClahe, bool useHamming, bool debug) {
    auto result = computeRotationCorrelation1D(voxelData1Input, voxelData2Input,
        /*useDirect=*/false, multipleRadii, useClahe, useHamming, debug, nullptr);
    if (debug) {
        generalHelpfulTools::ensureDirectoryExists(DEBUG_RESULTS_2D);
        std::ofstream myFile;
        myFile.open(DEBUG_RESULTS_2D "correlation1D_SO3.csv");
        for (size_t i = 0; i < result.correlationAveraged.size(); i++) {
            myFile << result.correlationAveraged[i] << "\n";
        }
        myFile.close();
    }
    return { result.correlationAveraged, result.angleList };
}

std::pair<std::vector<float>, std::vector<float>>
softRegistrationClass::compute1AngleCorrelationArrayDirect(double voxelData1Input[], double voxelData2Input[],
    bool multipleRadii, bool useClahe, bool useHamming, bool debug) {
    auto result = computeRotationCorrelation1D(voxelData1Input, voxelData2Input,
        /*useDirect=*/true, multipleRadii, useClahe, useHamming, debug, nullptr);
    if (debug) {
        generalHelpfulTools::ensureDirectoryExists(DEBUG_RESULTS_2D);
        std::ofstream myFile;
        myFile.open(DEBUG_RESULTS_2D "correlation1D_Direct.csv");
        for (size_t i = 0; i < result.correlationAveraged.size(); i++) {
            myFile << result.correlationAveraged[i] << "\n";
        }
        myFile.close();
    }
    return { result.correlationAveraged, result.angleList };
}


std::vector<translationPeakfs2D>
softRegistrationClass::sofftRegistrationVoxel2DTranslationAllPossibleSolutions(double voxelData1Input[],
    double voxelData2Input[],
    double cellSize,
    double normalizationFactor,
    bool debug,
    int numberOfRotationForDebug,
    double potentialNecessaryForPeak,
    bool benchmark,
    BenchmarkTimings2D* timings) {
    //copy and normalize voxelDataInput

    auto totalTransStart = std::chrono::high_resolution_clock::now();

    double fft1Time = 0, fft2Time = 0, correlationTime = 0, ifftTime = 0;
    double fftshiftTime = 0, peakDetectionTime = 0, covarianceTime = 0;

    auto fft1Start = std::chrono::high_resolution_clock::now();
    this->getSpectrumFromVoxelData2DCorrelation(voxelData1Input, this->complexSpectrum1Correlation,
        false, normalizationFactor);
    auto fft1End = std::chrono::high_resolution_clock::now();
    if (benchmark || timings) {
        fft1Time = std::chrono::duration<double, std::milli>(fft1End - fft1Start).count();
    }

    auto fft2Start = std::chrono::high_resolution_clock::now();
    this->getSpectrumFromVoxelData2DCorrelation(voxelData2Input, this->complexSpectrum2Correlation,
        false, normalizationFactor);
    auto fft2End = std::chrono::high_resolution_clock::now();
    if (benchmark || timings) {
        fft2Time = std::chrono::duration<double, std::milli>(fft2End - fft2Start).count();
    }

    auto correlationStart = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < this->correlationN; j++) {
        for (int i = 0; i < this->correlationN; i++) {
            int idx = j + this->correlationN * i;
            double r1 = complexSpectrum1Correlation[idx][0];
            double i1 = complexSpectrum1Correlation[idx][1];
            double r2 = complexSpectrum2Correlation[idx][0];
            double i2 = complexSpectrum2Correlation[idx][1];
            resultingPhaseDiff2DCorrelation[idx][0] = r1 * r2 + i1 * i2;
            resultingPhaseDiff2DCorrelation[idx][1] = i1 * r2 - r1 * i2;
        }
    }
    auto correlationEnd = std::chrono::high_resolution_clock::now();
    if (benchmark || timings) {
        correlationTime = std::chrono::duration<double, std::milli>(correlationEnd - correlationStart).count();
    }

    auto ifftStart = std::chrono::high_resolution_clock::now();
    fftw_execute(planFourierToVoxel2DCorrelation);
    auto ifftEnd = std::chrono::high_resolution_clock::now();
    if (benchmark || timings) {
        ifftTime = std::chrono::duration<double, std::milli>(ifftEnd - ifftStart).count();
    }

    auto fftshiftStart = std::chrono::high_resolution_clock::now();
    double maximumCorrelation = 0;
    for (int j = 0; j < this->correlationN; j++) {
        for (int i = 0; i < this->correlationN; i++) {
            int indexX = (this->correlationN / 2 + i + this->correlationN) % this->correlationN;
            int indexY = (this->correlationN / 2 + j + this->correlationN) % this->correlationN;
            double normalizationFactorForCorrelation = 1 / this->normalizationFactorCalculation(indexX, indexY);
            normalizationFactorForCorrelation = sqrt(normalizationFactorForCorrelation);
            resultingCorrelationDouble[indexY + this->correlationN * indexX] = normalizationFactorForCorrelation * sqrt(
                resultingShiftPeaks2DCorrelation[j + this->correlationN * i][0] *
                resultingShiftPeaks2DCorrelation[j + this->correlationN * i][0] +
                resultingShiftPeaks2DCorrelation[j + this->correlationN * i][1] *
                resultingShiftPeaks2DCorrelation[j + this->correlationN * i][1]);
            if (maximumCorrelation < resultingCorrelationDouble[indexY + this->correlationN * indexX]) {
                maximumCorrelation = resultingCorrelationDouble[indexY + this->correlationN * indexX];
            }
        }
    }
    auto fftshiftEnd = std::chrono::high_resolution_clock::now();
    if (benchmark || timings) {
        fftshiftTime = std::chrono::duration<double, std::milli>(fftshiftEnd - fftshiftStart).count();
    }

    // Save 2D correlation surface for visualization
    if (debug) {
        std::ofstream corr2DFile;
        corr2DFile.open(DEBUG_RESULTS_2D "translationCorrelation2D_angle" + std::to_string(numberOfRotationForDebug) + ".csv");
        for (int j = 0; j < this->correlationN; j++) {
            for (int i = 0; i < this->correlationN; i++) {
                corr2DFile << resultingCorrelationDouble[j + this->correlationN * i];
                if (i < this->correlationN - 1) corr2DFile << ",";
            }
            corr2DFile << "\n";
        }
        corr2DFile.close();
    }

    auto peakDetectionStart = std::chrono::high_resolution_clock::now();
    std::vector<translationPeakfs2D> potentialTranslations = this->peakDetectionOf2DCorrelationFindPeaksLibrary(
        cellSize, potentialNecessaryForPeak, 0.05, benchmark);
    auto peakDetectionEnd = std::chrono::high_resolution_clock::now();
    if (benchmark || timings) {
        peakDetectionTime = std::chrono::duration<double, std::milli>(peakDetectionEnd - peakDetectionStart).count();
    }

    if (benchmark || timings) {
        auto totalTransEnd = std::chrono::high_resolution_clock::now();
        double totalTransTime = std::chrono::duration<double, std::milli>(totalTransEnd - totalTransStart).count();

        timings->transFft1Time += fft1Time;
        timings->transFft2Time += fft2Time;
        timings->transCorrelationTime += correlationTime;
        timings->transIfftTime += ifftTime;
        timings->transFftshiftTime += fftshiftTime;
        timings->transPeakDetectionTime += peakDetectionTime;
        timings->totalTranslationTime += totalTransTime;
    }

    return potentialTranslations;
}

Eigen::Matrix4d softRegistrationClass::registrationOfTwoVoxelsSOFFTFast(double voxelData1Input[],
     double voxelData2Input[],
     Eigen::Matrix4d& initialGuess,
     Eigen::Matrix3d& covarianceMatrix,
     bool useInitialAngle,
     bool useInitialTranslation,
     double cellSize,
     bool useGauss,
     bool debug, double potentialNecessaryForPeak, bool benchmark,
     double level_potential_rotation) {
    if (!useInitialAngle || !useInitialTranslation) {
        std::cout << "this function has to be used with initial guess = true" << std::endl;
        exit(-1);
    }

    double goodGuessAlpha = std::atan2(initialGuess(1, 0),
        initialGuess(0, 0));


    std::vector<translationPeakfs2D> listOfTranslations;
    std::vector<Eigen::Matrix4d> listOfTransformations;

    //   std::vector<double> maximumHeightPeakList;
    std::vector<rotationPeakfs2D> estimatedAngles;
    double angleCovariance;
    double angleTMP = this->sofftRegistrationVoxel2DRotationOnly(voxelData1Input, voxelData2Input, goodGuessAlpha,
        angleCovariance, debug, level_potential_rotation, false);

    rotationPeakfs2D rotationPeakTMP;
    rotationPeakTMP.angle = angleTMP;
    rotationPeakTMP.covariance = angleCovariance;
    estimatedAngles.push_back(rotationPeakTMP);



    //    std::cout << "number of possible solutions: " << estimatedAngles.size() << std::endl;

    int angleIndex = 0;
    for (auto& estimatedAngle : estimatedAngles) {

        //copy data
        for (int i = 0; i < N * N; i++) {
            this->voxelData1[i] = voxelData1Input[i];
            this->voxelData2[i] = voxelData2Input[i];
        }

        cv::Mat magTMP1(this->N, this->N, CV_64F, voxelData1);
        cv::Mat magTMP2(this->N, this->N, CV_64F, voxelData2);
        //add gaussian blur
        if (useGauss) {
            for (int i = 0; i < 2; i++) {
                cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);
                cv::GaussianBlur(magTMP2, magTMP2, cv::Size(9, 9), 0);
            }
        }

        cv::Point2f pc(magTMP1.cols / 2., magTMP1.rows / 2.);
        //positive values mean COUNTER CLOCK WISE (open cv description) threfore negative rotation
        cv::Mat r = cv::getRotationMatrix2D(pc, estimatedAngle.angle * 180.0 / M_PI, 1.0);
        cv::warpAffine(magTMP1, magTMP1, r, magTMP1.size()); // what size I should use?

        std::vector<translationPeakfs2D> potentialTranslations = this->sofftRegistrationVoxel2DTranslationAllPossibleSolutions(
            voxelData1, voxelData2,
            cellSize,
            1.0,
            debug, angleIndex, potentialNecessaryForPeak, benchmark);
        Eigen::Matrix4d estimatedRotationScans = Eigen::Matrix4d::Identity();
        Eigen::AngleAxisd rotation_vectorTMP(estimatedAngle.angle, Eigen::Vector3d(0, 0, 1));
        Eigen::Matrix3d tmpRotMatrix3d = rotation_vectorTMP.toRotationMatrix();
        estimatedRotationScans.block<3, 3>(0, 0) = tmpRotMatrix3d;
        translationPeakfs2D bestFitTranslation;
        if (useInitialTranslation) {
            double distance = 100000;
            for (auto& potentialTranslation : potentialTranslations) {
                double diffX = potentialTranslation.translationSI.x() - initialGuess(0, 3);
                double diffY = potentialTranslation.translationSI.y() - initialGuess(1, 3);
                if (distance > sqrt(diffX * diffX + diffY * diffY)) {

                    bestFitTranslation = potentialTranslation;
                    distance = sqrt(diffX * diffX + diffY * diffY);

                }
            }
        }
        else {
            double highestPeak = 0;
            int indexHighestPeak = 0;
            for (int i = 0; i < potentialTranslations.size(); i++) {
                if (potentialTranslations[i].peakHeight > highestPeak) {
                    indexHighestPeak = i;
                    highestPeak = potentialTranslations[i].peakHeight;
                }
            }
            bestFitTranslation = potentialTranslations[indexHighestPeak];
        }

        estimatedRotationScans(0, 3) = bestFitTranslation.translationSI.x();
        estimatedRotationScans(1, 3) = bestFitTranslation.translationSI.y();
        estimatedRotationScans(2, 3) = 0;

        listOfTransformations.push_back(estimatedRotationScans);
        listOfTranslations.push_back(bestFitTranslation);

        angleIndex++;
    }

    //find maximum peak:
    double highestPeak = 0;
    int indexHighestPeak = 0;
    for (int i = 0; i < listOfTransformations.size(); i++) {
        if (highestPeak < listOfTranslations[i].peakHeight) {
            highestPeak = listOfTranslations[i].peakHeight;
            indexHighestPeak = i;
        }
    }

    covarianceMatrix.block<2, 2>(0,
        0) = listOfTranslations[indexHighestPeak].covariance;
    covarianceMatrix(2, 2) = angleCovariance;

    return listOfTransformations[indexHighestPeak];//should be the transformation matrix from 1 to 2
}

std::vector<transformationPeakfs2D>
softRegistrationClass::registrationOfTwoVoxelsSOFFTAllSoluations(double voxelData1Input[],
     double voxelData2Input[],
     double cellSize,
     bool useGauss,
     bool debug, double potentialNecessaryForPeak,
     bool multipleRadii,
     bool useClahe,
     bool useHamming,
     bool useDirect,
     bool benchmark,
     BenchmarkTimings2D* timings,
     double level_potential_rotation) {

    std::vector<transformationPeakfs2D> listOfTransformations;
    std::vector<rotationPeakfs2D> estimatedAnglePeak;

    BenchmarkTimings2D localTimings;
    BenchmarkTimings2D* pTimings = (benchmark || timings) ? &localTimings : nullptr;

    auto totalStart = std::chrono::high_resolution_clock::now();

    estimatedAnglePeak = this->sofftRegistrationVoxel2DListOfPossibleRotations(voxelData1Input, voxelData2Input,
        debug, multipleRadii, useClahe,
        useHamming, pTimings, level_potential_rotation, useDirect);

    int numAngles = estimatedAnglePeak.size();
    listOfTransformations.reserve(numAngles);

    std::vector<double> voxelData1_local(this->N * this->N);
    std::vector<double> voxelData2_local(this->N * this->N);

    double totalPreprocessingTime = 0;
    double totalTranslationTime = 0;
    std::vector<double> transPerAngleTimes;

    for (int angleIndex = 0; angleIndex < numAngles; angleIndex++) {
        auto& estimatedAngle = estimatedAnglePeak[angleIndex];

        auto preprocessStart = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < this->N * this->N; i++) {
            voxelData1_local[i] = voxelData1Input[i];
            voxelData2_local[i] = voxelData2Input[i];
        }

        cv::Mat magTMP1(this->N, this->N, CV_64F, voxelData1_local.data());
        cv::Mat magTMP2(this->N, this->N, CV_64F, voxelData2_local.data());

        if (useGauss) {
            for (int i = 0; i < 2; i++) {
                cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);
                cv::GaussianBlur(magTMP2, magTMP2, cv::Size(9, 9), 0);
            }
        }

        cv::Point2f pc(magTMP1.cols / 2., magTMP1.rows / 2.);
        cv::Mat r = cv::getRotationMatrix2D(pc, estimatedAngle.angle * 180.0 / M_PI, 1.0);
        cv::warpAffine(magTMP1, magTMP1, r, magTMP1.size());
        auto preprocessEnd = std::chrono::high_resolution_clock::now();
        totalPreprocessingTime += std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count();

        auto angleStart = std::chrono::high_resolution_clock::now();
        std::vector<translationPeakfs2D> potentialTranslations =
            this->sofftRegistrationVoxel2DTranslationAllPossibleSolutions(
                voxelData1_local.data(), voxelData2_local.data(),
                cellSize, 1.0, debug, angleIndex, potentialNecessaryForPeak, benchmark, pTimings);
        auto angleEnd = std::chrono::high_resolution_clock::now();
        double angleTime = std::chrono::duration<double, std::milli>(angleEnd - angleStart).count();
        totalTranslationTime += angleTime;
        transPerAngleTimes.push_back(angleTime);

        transformationPeakfs2D transformationPeakTMP;
        transformationPeakTMP.potentialRotation = estimatedAngle;
        transformationPeakTMP.potentialTranslations = potentialTranslations;

        listOfTransformations.push_back(transformationPeakTMP);
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    int totalTransPeaks = 0;
    for (const auto& sol : listOfTransformations) {
        totalTransPeaks += sol.potentialTranslations.size();
    }

    if (benchmark || timings) {
        localTimings.numAngles = numAngles;
        localTimings.totalTransPeaks = totalTransPeaks;
        localTimings.totalTranslationTime = totalTranslationTime;
        localTimings.transPreprocessingTime = totalPreprocessingTime;
        localTimings.transPerAngleTimes = transPerAngleTimes;
        localTimings.totalTime = totalTime;

        if (timings) {
            *timings = localTimings;
        }


    }

    // save list of transformations
    if (debug) {
        generalHelpfulTools::ensureDirectoryExists(DEBUG_RESULTS_2D);

        std::ofstream myFile12;
        myFile12.open(
            DEBUG_RESULTS_2D "dataForReadIn.csv");

        // Header with metadata
        myFile12 << "N\tcorrelationN\tcellSize\tpotentialNecessaryForPeak\tnumAngles\tnumTotalSolutions\n";
        myFile12 << this->N << "\t"
            << (this->N * 2 - 1) << "\t"
            << cellSize << "\t"
            << potentialNecessaryForPeak << "\t";

        int totalSolutions = 0;
        for (const auto& sol : listOfTransformations) {
            totalSolutions += sol.potentialTranslations.size();
        }
        myFile12 << listOfTransformations.size() << "\t" << totalSolutions << "\n";

        // Per-angle breakdown
        myFile12 << "\nangleIndex\tangle\tnumTranslations\n";
        for (size_t i = 0; i < listOfTransformations.size(); i++) {
            myFile12 << i << "\t"
                << listOfTransformations[i].potentialRotation.angle << "\t"
                << listOfTransformations[i].potentialTranslations.size() << "\n";
        }
        myFile12.close();

    }
    //save every transformation in files.
    if (debug) {
        generalHelpfulTools::ensureDirectoryExists(DEBUG_RESULTS_2D);

        int numberOfTransformations = 0;
        for (auto& listOfTransformation : listOfTransformations) {
            Eigen::Matrix4d currentMatrix = Eigen::Matrix4d::Identity();
            //rotation
            currentMatrix.block<3, 3>(0, 0) = generalHelpfulTools::getQuaternionFromRPY(0, 0,
                listOfTransformation.potentialRotation.angle).toRotationMatrix();
            for (auto& potentialTranslation : listOfTransformation.potentialTranslations) {
                //translation
                currentMatrix.block<3, 1>(0, 3) = Eigen::Vector3d(potentialTranslation.translationSI.x(),
                    potentialTranslation.translationSI.y(), 0);

                std::ofstream myFile12;
                myFile12.open(
                    DEBUG_RESULTS_2D "potentialTransformation" +
                    std::to_string(numberOfTransformations) + ".csv");
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        myFile12 << currentMatrix(i, j) << " ";//number of possible rotations
                    }
                    myFile12 << "\n";
                }
                myFile12 << "\n";
                myFile12 << potentialTranslation.peakHeight;
                myFile12 << "\n";
                myFile12 << potentialTranslation.persistenceValue;
                myFile12 << "\n";

                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        myFile12 << potentialTranslation.covariance(i, j) << " ";//number of possible rotations
                    }
                    myFile12 << "\n";
                }
                myFile12.close();
                numberOfTransformations++;
            }

        }


    }
    return listOfTransformations;
}


std::vector<translationPeakfs2D>
softRegistrationClass::peakDetectionOf2DCorrelationSimpleDouble1D(double maximumCorrelation, double cellSize,
    int impactOfNoiseFactor,
    double percentageOfMaxCorrelationIgnored) {

    float impactOfNoise = std::pow(2, impactOfNoiseFactor);

    std::vector<std::vector<int>> xPeaks, yPeaks;

    for (int j = 0; j < this->correlationN; j++) {
        std::vector<float> inputYLine;
        for (int i = 0; i < this->correlationN; i++) {
            inputYLine.push_back((float)resultingCorrelationDouble[j + this->correlationN * i]);
        }
        std::vector<int> out;
        PeakFinder::findPeaks(inputYLine, out, false, impactOfNoise);
        //        for(int i = 0 ; i < out.size();i++){
        //            std::cout << out[i] << std::endl;
        //        }
        //        std::cout <<"next"<< std::endl;
        yPeaks.push_back(out);
    }

    for (int i = 0; i < this->correlationN; i++) {
        std::vector<float> inputXLine;
        for (int j = 0; j < this->correlationN; j++) {
            inputXLine.push_back((float)resultingCorrelationDouble[j + this->correlationN * i]);
        }
        std::vector<int> out;
        PeakFinder::findPeaks(inputXLine, out, false, impactOfNoise);
        //        for(int j = 0 ; j < out.size();j++){
        //            std::cout << out[j] << std::endl;
        //        }
        //        std::cout <<"next"<< std::endl;
        xPeaks.push_back(out);

    }
    std::vector<translationPeakfs2D> potentialTranslations;
    for (int j = 0; j < this->correlationN; j++) {
        for (int i = 0; i < this->correlationN; i++) {
            auto iteratorX = std::find(xPeaks[i].begin(), xPeaks[i].end(), j);
            if (iteratorX != xPeaks[i].end()) {
                auto iteratorY = std::find(yPeaks[j].begin(), yPeaks[j].end(), i);
                if (iteratorY != yPeaks[j].end()) {
                    //                    std::cout << "found Peak:" << std::endl;
                    //                    std::cout << *iteratorX << std::endl;
                    //                    std::cout << *iteratorY  << std::endl;

                    //                    std::cout << i << std::endl;
                    //                    std::cout << j  << std::endl;
                    if (maximumCorrelation * percentageOfMaxCorrelationIgnored <
                        resultingCorrelationDouble[j + this->correlationN * i]) {
                        translationPeakfs2D tmpTranslationPeak;
                        tmpTranslationPeak.translationSI.x() = -((i - (int)(this->correlationN / 2.0)) * cellSize);
                        tmpTranslationPeak.translationSI.y() = -((j - (int)(this->correlationN / 2.0)) * cellSize);
                        tmpTranslationPeak.translationVoxel.x() = i;
                        tmpTranslationPeak.translationVoxel.y() = j;
                        tmpTranslationPeak.peakHeight = resultingCorrelationDouble[j + this->correlationN * i];
                        potentialTranslations.push_back(tmpTranslationPeak);
                    }

                }
            }
        }
    }
    return potentialTranslations;
}









bool softRegistrationClass::isPeak(cv::Mat mx[], std::vector<cv::Point>& conn_points) {
    cv::Point poi_point = conn_points.back();
    int row = poi_point.y;
    int col = poi_point.x;
    float poi_val = mx[0].at<float>(poi_point);
    bool isPeakEle = true;
    for (int mask_row = row - 1; mask_row <= row + 1; mask_row++) {
        for (int mask_col = col - 1; mask_col <= col + 1; mask_col++) {
            if (mask_row == row && mask_col == col) {
                continue;
            }
            float conn_pt_val = mx[0].at<float>(mask_row, mask_col);
            if (poi_val < conn_pt_val) {
                isPeakEle = false;
                break;
            }
            if (poi_val == conn_pt_val) {
                int Peak_status = mx[1].at<uchar>(mask_row, mask_col);
                if (Peak_status == 0) {
                    isPeakEle = false;
                    break;
                }
                else if (Peak_status == 1) {
                    isPeakEle = true;
                    break;
                }
                else {
                    cv::Point p(mask_col, mask_row);
                    std::vector<cv::Point>::iterator it;
                    it = std::find(conn_points.begin(), conn_points.end(), p);
                    if (it == conn_points.end()) {
                        conn_points.push_back(p);
                        isPeakEle = isPeakEle && isPeak(mx, conn_points);
                    }
                }
            }
        }
        if (isPeakEle == false) {
            break;
        }
    }
    return isPeakEle;
}

cv::Mat softRegistrationClass::imregionalmax(cv::Mat& src) {
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar::all(-1));
    cv::Mat mx_ch1(padded.rows, padded.cols, CV_8UC1, cv::Scalar(
        2)); //Peak elements will be represented by 1, others by 0, initializing Mat with 2 for differentiation
    cv::Mat mx[2] = { padded, mx_ch1 }; //mx[0] is padded image, mx[1] is regional maxima matrix
    int mx_rows = mx[0].rows;
    int mx_cols = mx[0].cols;
    cv::Mat dest(mx[0].size(), CV_8UC1);

    //Check each pixel for local max
    for (int row = 1; row < mx_rows - 1; row++) {
        for (int col = 1; col < mx_cols - 1; col++) {
            std::vector<cv::Point> conn_points; //this vector holds all connected points for candidate pixel
            cv::Point p(col, row);
            conn_points.push_back(p);
            bool isPartOfPeak = isPeak(mx, conn_points);
            if (isPartOfPeak) {
                mx[1].at<uchar>(row, col) = 1;
            }
            else {
                mx[1].at<uchar>(row, col) = 0;
            }
        }
    }
    dest = mx[1](cv::Rect(1, 1, src.cols, src.rows));
    return dest;
}

double softRegistrationClass::normalizationFactorCalculation(int x, int y) {

    double tmpCalculation = 0;
    //    tmpCalculation = abs(1.0/((x-this->correlationN/2)*(y-this->correlationN/2)));
    //    tmpCalculation = this->correlationN * this->correlationN * (this->correlationN - (x + 1) + 1);
    //    tmpCalculation = tmpCalculation * (this->correlationN - (y + 1) + 1);
    if (x < ceil(this->correlationN / 2)) {
        tmpCalculation = (x + 1);
    }
    else {
        tmpCalculation = (this->correlationN - x);
    }

    if (y < ceil(this->correlationN / 2)) {
        tmpCalculation = tmpCalculation * (y + 1);
    }
    else {
        tmpCalculation = tmpCalculation * (this->correlationN - y);
    }

    return (tmpCalculation);
}

std::vector<translationPeakfs2D> softRegistrationClass::peakDetectionOf2DCorrelationFindPeaksLibrary(double cellSize,
    double potentialNecessaryForPeak,
    double ignoreSidesPercentage,
    bool benchmark) {

    double* current2DCorrelation;
    current2DCorrelation = (double*)malloc(sizeof(double) * this->correlationN * this->correlationN);

    double maxValue = 0;
    for (int j = 0; j < this->correlationN; j++) {
        for (int i = 0; i < this->correlationN; i++) {
            current2DCorrelation[j + this->correlationN * i] = this->resultingCorrelationDouble[j + this->correlationN * i];
            if (current2DCorrelation[j + this->correlationN * i] > maxValue) {
                maxValue = current2DCorrelation[j + this->correlationN * i];
            }
        }
    }
    for (int j = 0; j < this->correlationN; j++) {
        for (int i = 0; i < this->correlationN; i++) {
            current2DCorrelation[j + this->correlationN * i] = current2DCorrelation[j + this->correlationN * i] / maxValue;
            this->resultingCorrelationDouble[j + this->correlationN * i] = this->resultingCorrelationDouble[j + this->correlationN * i] / maxValue;
        }
    }
    size_t ourSize = this->correlationN;
    findpeaks::image_t<double> image = { ourSize, ourSize, this->resultingCorrelationDouble };
    std::vector<findpeaks::peak_t<double>> peaks = findpeaks::persistance(image);
    std::vector<translationPeakfs2D> tmpTranslations;
    for (const auto& p : peaks) {
        double levelPotential = p.persistence * sqrt(p.birth_level) *
            Eigen::Vector2d((double)((int)p.birth_position.x - (int)p.death_position.x),
                (double)((int)p.birth_position.y - (int)p.death_position.y)).norm() *
            511.0 / this->correlationN;
        bool inInterestingArea = true;
        if ((int)p.birth_position.x<ignoreSidesPercentage * this->correlationN || (int)p.birth_position.x>(
            1 - ignoreSidesPercentage) * this->correlationN ||
            (int)p.birth_position.y<ignoreSidesPercentage * this->correlationN || (int)p.birth_position.y>(
                1 - ignoreSidesPercentage) * this->correlationN) {
            inInterestingArea = false;
        }
        if (p.birth_level > 0.0 && levelPotential > potentialNecessaryForPeak && inInterestingArea) {
            translationPeakfs2D tmpTranslationPeak;
            tmpTranslationPeak.translationSI.x() = -(((int)p.birth_position.x - (int)(this->correlationN / 2.0)) *
                cellSize);
            tmpTranslationPeak.translationSI.y() = -(((int)p.birth_position.y - (int)(this->correlationN / 2.0)) *
                cellSize);
            tmpTranslationPeak.translationVoxel.x() = (int)p.birth_position.x;
            tmpTranslationPeak.translationVoxel.y() = (int)p.birth_position.y;
            tmpTranslationPeak.peakHeight = resultingCorrelationDouble[p.birth_position.y +
                this->correlationN * p.birth_position.x] *
                maxValue;
            tmpTranslationPeak.persistenceValue = levelPotential;
            tmpTranslations.push_back(tmpTranslationPeak);
        }
    }
    free(current2DCorrelation);
    return tmpTranslations;
}

int softRegistrationClass::getSizeOfRegistration() {
    return this->N;
}

