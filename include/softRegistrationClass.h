//
// Created by tim-linux on 01.03.22.
//

#ifndef FSREGISTRATION_softRegistrationClass_H
#define FSREGISTRATION_softRegistrationClass_H

//#include "softRegistrationClass.h"
#include "softCorrelationClass.h"
#include "PeakFinder.h"
#include "generalHelpfulTools.h"
//#include "slamToolsRos.h"

//#include <pcl/io/pcd_io.h>
//#include <pcl/io/ply_io.h>
//#include <pcl/common/transforms.h>
//#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core.hpp>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Min_sphere_of_spheres_d.h>
#include <CGAL/Min_sphere_of_points_d_traits_2.h>
#include <CGAL/Random.h>

#include <iostream>
#include <fstream>
#include <findpeaks/mask.hpp>
#include <findpeaks/persistence.hpp>
#include "fftw3.h"

struct rotationPeakfs2D {
    double angle;
    double peakCorrelation;
    double covariance;
    double levelPotential;
};

struct translationPeakfs2D {
    Eigen::Vector2d translationSI;
    Eigen::Vector2i translationVoxel;
    double peakHeight;
    double persistenceValue;
    Eigen::Matrix2d covariance;
};

struct transformationPeakfs2D {
    std::vector<translationPeakfs2D> potentialTranslations;
    rotationPeakfs2D potentialRotation;
};

struct BenchmarkTimings2D {
    double spectrumTime = 0;
    double softDescriptorTime = 0;
    double rotationCorrelationTime = 0;
    double rotationExtractionTime = 0;
    double rotationPeakDetectionTime = 0;
    double totalTranslationTime = 0;
    double transFft1Time = 0;
    double transFft2Time = 0;
    double transCorrelationTime = 0;
    double transIfftTime = 0;
    double transFftshiftTime = 0;
    double transPeakDetectionTime = 0;
    double transPreprocessingTime = 0;
    double freqRotationPhaseTime = 0;
    std::vector<double> transPerAngleTimes;
    int numAngles = 0;
    int totalTransPeaks = 0;
    double totalTime = 0;
};

struct RotationCorrelationResult {
    std::vector<float> correlationAveraged;
    std::vector<float> angleList;
};

class softRegistrationClass {
public:
    softRegistrationClass(int N, int bwOut, int bwIn, int degLim) : sofftCorrelationObject(N, bwOut, bwIn,
                                                                                                degLim) {
        this->N = N;
        this->correlationN = N * 2 - 1;
        this->bwOut = bwOut;
        this->bwIn = bwIn;
        this->degLim = degLim;
        this->resultingCorrelationDouble = (double *) malloc(sizeof(double) * this->correlationN * this->correlationN);
        this->resultingCorrelationComplex = fftw_alloc_complex(8 * bwOut * bwOut * bwOut);
//        (fftw_complex *) fftw_malloc(
//                sizeof(fftw_complex) * (8 * bwOut * bwOut * bwOut));
        this->resultingPhaseDiff2D = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N);
        this->resultingPhaseDiff2DCorrelation = (fftw_complex *) fftw_malloc(
                sizeof(fftw_complex) * this->correlationN * this->correlationN);
        this->resultingShiftPeaks2D = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N);
        this->resultingShiftPeaks2DCorrelation = (fftw_complex *) fftw_malloc(
                sizeof(fftw_complex) * this->correlationN * this->correlationN);

        this->magnitude1Shifted = (double *) malloc(sizeof(double) * N * N * N);
        this->magnitude2Shifted = (double *) malloc(sizeof(double) * N * N * N);
        this->voxelData1 = (double *) malloc(sizeof(double) * N * N * N);
        this->voxelData2 = (double *) malloc(sizeof(double) * N * N * N);
//        this->spectrum1 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N * N);
//        this->spectrum2 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N * N);
        this->spectrumOut = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N * N);
        this->spectrumOutCorrelation = (fftw_complex *) fftw_malloc(
                sizeof(fftw_complex) * this->correlationN * this->correlationN * this->correlationN);
        this->phase1 = (double *) malloc(sizeof(double) * N * N * N);
        this->phase2 = (double *) malloc(sizeof(double) * N * N * N);
        this->magnitude1 = (double *) malloc(sizeof(double) * N * N * N);
        this->magnitude2 = (double *) malloc(sizeof(double) * N * N * N);
        this->complexSpectrum1Correlation = (fftw_complex *) fftw_malloc(
                sizeof(fftw_complex) * this->correlationN * this->correlationN * this->correlationN);
        this->complexSpectrum2Correlation = (fftw_complex *) fftw_malloc(
                sizeof(fftw_complex) * this->correlationN * this->correlationN * this->correlationN);

        resampledMagnitudeSO3_1 = (double *) malloc(sizeof(double) * N * N);
        resampledMagnitudeSO3_2 = (double *) malloc(sizeof(double) * N * N);
        resampledMagnitudeSO3_1TMP = (double *) malloc(sizeof(double) * N * N);
        resampledMagnitudeSO3_2TMP = (double *) malloc(sizeof(double) * N * N);
        inputSpacialData = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N * N);
        inputSpacialDataCorrelation = (fftw_complex *) fftw_malloc(
                sizeof(fftw_complex) * this->correlationN * this->correlationN * this->correlationN);
//        planToFourierVoxel = fftw_plan_dft_3d(N, N, N, resultingPhaseDiff2D,
//                                              resultingShiftPeaks2D, FFTW_BACKWARD, FFTW_ESTIMATE);
        planFourierToVoxel2D = fftw_plan_dft_2d(N, N, resultingPhaseDiff2D,
                                                resultingShiftPeaks2D, FFTW_BACKWARD, FFTW_ESTIMATE);
        planFourierToVoxel2DCorrelation = fftw_plan_dft_2d(this->correlationN, this->correlationN,
                                                           resultingPhaseDiff2DCorrelation,
                                                           resultingShiftPeaks2DCorrelation, FFTW_BACKWARD,
                                                           FFTW_ESTIMATE);
//        correlation2DResult = (double *) malloc(sizeof(double) * N * N);


        planVoxelToFourier3D = fftw_plan_dft_3d(N, N, N, inputSpacialData,
                                                spectrumOut, FFTW_FORWARD, FFTW_ESTIMATE);
        planVoxelToFourier2D = fftw_plan_dft_2d(N, N, inputSpacialData,
                                                spectrumOut, FFTW_FORWARD, FFTW_ESTIMATE);
       planVoxelToFourier2DCorrelation = fftw_plan_dft_2d(this->correlationN, this->correlationN,
                                                            inputSpacialDataCorrelation,
                                                            spectrumOutCorrelation, FFTW_FORWARD, FFTW_ESTIMATE);
        this->correlation1D = (double *) malloc(sizeof(double) * N);
        this->PmR = (double *) malloc(sizeof(double) * (2 * bwIn));
        this->PmI = (double *) malloc(sizeof(double) * (2 * bwIn));

        // Precompute spherical projection lookup tables
        int bandwidth = N / 2;
        hammingCoeffs = (double*) malloc(sizeof(double) * N);
        xAngle = (double*) malloc(sizeof(double) * N * N);
        yAngle = (double*) malloc(sizeof(double) * N * N);

        double* sinThetaLocal = (double*) malloc(sizeof(double) * N);
        double* cosPhiLocal = (double*) malloc(sizeof(double) * N);
        double* sinPhiLocal = (double*) malloc(sizeof(double) * N);

        for (int j = 0; j < N; j++) {
            sinThetaLocal[j] = std::sin(M_PI * j / (double) N);
        }
        for (int k = 0; k < N; k++) {
            cosPhiLocal[k] = std::cos(M_PI * k / (double) bandwidth);
            sinPhiLocal[k] = std::sin(M_PI * k / (double) bandwidth);
            hammingCoeffs[k] = 25.0/46.0 - (1.0 - 25.0/46.0) * std::cos(2.0 * M_PI * k / (double) N);
        }
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                xAngle[j * N + k] = sinThetaLocal[j] * cosPhiLocal[k];
                yAngle[j * N + k] = sinThetaLocal[j] * sinPhiLocal[k];
            }
        }

        free(sinThetaLocal);
        free(cosPhiLocal);
        free(sinPhiLocal);

        // CLAHE + reusable Mats
        clahe = cv::createCLAHE();
        clahe->setClipLimit(3);
        magCLAHE1 = cv::Mat(N, N, CV_8UC1);
        magCLAHE2 = cv::Mat(N, N, CV_8UC1);
    }

        ~softRegistrationClass() {
        sofftCorrelationObject.~softCorrelationClass();

        free(hammingCoeffs);
        free(xAngle);
        free(yAngle);
    }


    double
    getSpectrumFromVoxelData2D(double voxelData[], double magnitude[], double phase[], bool gaussianBlur = false);

//    Eigen::Matrix4d
//    registrationOfTwoVoxel2D(double voxelData1[], double voxelData2[], double &fitnessX, double &fitnessY,
//                             double goodGuessAlpha, bool debug);


    double
   sofftRegistrationVoxel2DRotationOnly(double voxelData1Input[], double voxelData2Input[], double goodGuessAlpha, double &covariance,
                                          bool debug = false, double level_potential_rotation = 0.0, bool useDirect = false);

    std::vector<rotationPeakfs2D>
    sofftRegistrationVoxel2DListOfPossibleRotations(double voxelData1Input[], double voxelData2Input[],
                                                    bool debug = false, bool multipleRadii = false,
                                                    bool useClahe = true, bool useHamming = true,
                                                    BenchmarkTimings2D* timings = nullptr,
                                                    double level_potential_rotation = 0.1,
                                                    bool useDirect = false);

//    Eigen::Vector2d sofftRegistrationVoxel2DTranslation(double voxelData1Input[],
//                                                        double voxelData2Input[],
//                                                        double &fitnessX, double &fitnessY, double cellSize,
//                                                        Eigen::Vector3d initialGuess, bool useInitialGuess,
//                                                        double &heightMaximumPeak, bool debug = false);

  Eigen::Matrix4d registrationOfTwoVoxelsSOFFTFast(double voxelData1Input[],
                                                       double voxelData2Input[],
                                                       Eigen::Matrix4d &initialGuess,Eigen::Matrix3d &covarianceMatrix,
                                                       bool useInitialAngle, bool useInitialTranslation,
                                                       double cellSize,
                                                       bool useGauss,
                                                       bool debug = false,
                                                       double potentialNecessaryForPeak = 0.1,
                                                       bool benchmark = false,
                                                       double level_potential_rotation = 0.1,
                                                       int normalization = 1);

std::vector<transformationPeakfs2D> registrationOfTwoVoxelsSOFFTAllSoluations(double voxelData1Input[],
                                                                                      double voxelData2Input[],
                                                                                      double cellSize,
                                                                                      bool useGauss,
                                                                                      bool debug = false,
                                                                                      double potentialNecessaryForPeak = 0.1,
                                                                                      bool multipleRadii = false,
                                                                                      bool useClahe = true,
                                                                                      bool useHamming = true,
                                                                                      bool useDirect = true,
                                                                                      bool benchmark = false,
                                                                                      BenchmarkTimings2D* timings = nullptr,
                                                                                      double level_potential_rotation = 0.0,
                                                                                      int normalization = 1,
                                                                                      bool usePhaseCorrelation = false);

  double getSpectrumFromVoxelData2DCorrelation(double voxelData[], fftw_complex *complexOut,
                                                  bool gaussianBlur, double normalizationFactor);

 std::vector<translationPeakfs2D> sofftRegistrationVoxel2DTranslationAllPossibleSolutions(double voxelData1Input[],
                                                                                                double voxelData2Input[],
                                                                                                double cellSize,
                                                                                                double normalizationFactor,
                                                                                                bool debug = false,
                                                                                                int numberOfRotationForDebug = 0,
                                                                                                double potentialNecessaryForPeak = 0.1,
                                                                                                bool benchmark = false,
                                                                                                 BenchmarkTimings2D* timings = nullptr,
                                                                                                 int normalization = 1,
                                                                                                 bool usePhaseCorrelation = false);


    std::pair<std::vector<float>, std::vector<float>>
    compute1AngleCorrelationArraySO3(double voxelData1Input[], double voxelData2Input[],
                                      bool multipleRadii = false,
                                      bool useClahe = true, bool useHamming = true,
                                      bool debug = false);

    std::pair<std::vector<float>, std::vector<float>>
    compute1AngleCorrelationArrayDirect(double voxelData1Input[], double voxelData2Input[],
                                         bool multipleRadii = false,
                                         bool useClahe = true, bool useHamming = true,
                                         bool debug = false);

    std::vector<translationPeakfs2D>
    peakDetectionOf2DCorrelationSimpleDouble1D(double maximumCorrelation, double cellSize, int impactOfNoiseFactor = 2,
                                               double percentageOfMaxCorrelationIgnored = 0.10);

    bool isPeak(cv::Mat mx[], std::vector<cv::Point> &conn_points);

    cv::Mat imregionalmax(cv::Mat &src);

    double normalizationFactorCalculation(int x, int y);

    cv::Mat opencv_imextendedmax(cv::Mat &inputMatrix, double hParam);

    void imextendedmax_imreconstruct(cv::Mat g, cv::Mat f, cv::Mat &dest);

   std::vector<translationPeakfs2D>
    peakDetectionOf2DCorrelationFindPeaksLibrary(double cellSize, double potentialNecessaryForPeak = 0.1,
                                                  double ignoreSidesPercentage = 0.05, bool benchmark = false);

    int getSizeOfRegistration();


private://here everything is created. malloc is done in the constructor




    int N, correlationN;//describes the size of the overall voxel system + correlation N
    int bwOut, bwIn, degLim;
    double *voxelData1;
    double *voxelData2;
//    fftw_complex *spectrum1;
//    fftw_complex *spectrum2;
    fftw_complex *spectrumOut;
    fftw_complex *spectrumOutCorrelation;
    double *magnitude1;
    double *magnitude2;
    double *phase1;
    double *phase2;
    fftw_complex *complexSpectrum1Correlation;
    fftw_complex *complexSpectrum2Correlation;

    double *magnitude1Shifted;
    double *magnitude2Shifted;
    double *resampledMagnitudeSO3_1;
    double *resampledMagnitudeSO3_2;
    double *resampledMagnitudeSO3_1TMP;
    double *resampledMagnitudeSO3_2TMP;
    softCorrelationClass sofftCorrelationObject;
    fftw_complex *resultingCorrelationComplex;
    fftw_complex *resultingPhaseDiff2D;
    fftw_complex *resultingPhaseDiff2DCorrelation;
    fftw_complex *resultingShiftPeaks2D;
    fftw_complex *resultingShiftPeaks2DCorrelation;
    double *resultingCorrelationDouble;
//    fftw_plan planToFourierVoxel;
//    double *correlation2DResult;
    fftw_complex *inputSpacialData;
    fftw_complex *inputSpacialDataCorrelation;
    fftw_plan planVoxelToFourier3D;
    fftw_plan planVoxelToFourier2D;
    fftw_plan planVoxelToFourier2DCorrelation;
    fftw_plan planFourierToVoxel2D;
    fftw_plan planFourierToVoxel2DCorrelation;
    double *correlation1D;
    double *PmR;
    double *PmI;

    // Precomputed spherical projection lookup tables
    double* hammingCoeffs;
    double* xAngle;
    double* yAngle;

    // Reusable OpenCV objects
    cv::Ptr<cv::CLAHE> clahe;
    cv::Mat magCLAHE1;
    cv::Mat magCLAHE2;

   RotationCorrelationResult computeRotationCorrelation1D(double voxelData1Input[], double voxelData2Input[],
                                                            bool useDirect, bool multipleRadii, bool useClahe,
                                                            bool useHamming, bool debug, BenchmarkTimings2D* timings,
                                                            std::vector<rotationPeakfs2D>* outPeaks = nullptr,
                                                            double level_potential_rotation = 0.1);

    rotationPeakfs2D findClosestRotationAngle(const std::vector<rotationPeakfs2D>& allAnglesList, double goodGuessAlpha);

    std::vector<rotationPeakfs2D> runRotationPeakDetection(const RotationCorrelationResult& result,
                                                           BenchmarkTimings2D* timings,
                                                           double level_potential_rotation = 0.1);

};


#endif //UNDERWATERSLAM_softRegistrationClass_H
