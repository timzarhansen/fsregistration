//
// Created by aya on 08.12.23.
//

#ifndef FSREGISTRATION_SOFTREGISTRATIONCLASS3D_H
#define FSREGISTRATION_SOFTREGISTRATIONCLASS3D_H

#include "softRegistrationClass3D.h"
#include "softCorrelationClass3D.h"

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
// #include "../../../../../../opt/ros/humble/include/rclcpp/rclcpp/any_subscription_callback.hpp"

struct translationPeak3D {
    double xTranslation;
    double yTranslation;
    double zTranslation;
    double persistence;
    double levelPotential;
    double correlationHeight;
    double globalCorrelationHeight;
};

struct rotationPeak4D {
    double x;
    double y;
    double z;
    double w;
    double persistence;
    double levelPotential;
    double correlationHeight;
};

//struct translationPeak3D {
//    Eigen::Quaterniond rotation;
//    double peakCorrelation;
//    double covariance;
//};

//struct translationPeak3D {
//    Eigen::Vector3d translationSI;
//    Eigen::Vector3i translationVoxel;
//    double peakHeight;
//    double persistenceValue;
//    double levelPotentialValue;
//    Eigen::Matrix3d covariance;
//};

struct transformationPeakfs3D {
    std::vector<translationPeak3D> potentialTranslations;
    rotationPeak4D potentialRotation;
};

struct BenchmarkTimings3D {
    double spectrumTime;
    double softDescriptorTime;
    double rotationCorrelationTime;
    double overheadTime;
    double peakDetectionTime;
    double plottingTime;
    double totalAllTransTime;
    double transVoxelRotationTime;
    double transFft1Time;
    double transFft2Time;
    double transCorrelationTime;
    double transIfftTime;
    double transFftshiftTime;
    double transPeakDetectionTime;
    std::vector<double> transPerSolutionTimes;
    int numSolutions;
    int totalTransPeaks;
    double totalTime;
};

class softRegistrationClass3D {
public:
    softRegistrationClass3D(int N, int bwOut, int bwIn, int degLim) : sofftCorrelationObject3D(N, bwOut, bwIn,
        degLim) {
        this->N = N;
        this->correlationN = N * 2 - 1;
        //        this->N = N * 2 - 1;
        //        this->N = N;

        this->bwOut = bwOut;
        this->bwIn = bwIn;
        this->degLim = degLim;
        this->resultingCorrelationDouble = (double *) malloc(
            sizeof(double) * this->correlationN * this->correlationN * this->correlationN);
        this->resultingCorrelationComplex = fftw_alloc_complex(8 * bwOut * bwOut * bwOut);

        this->resultingPhaseDiff3D = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N * N);
        this->resultingPhaseDiff3DCorrelation = (fftw_complex *) fftw_malloc(
            sizeof(fftw_complex) * this->correlationN * this->correlationN * this->correlationN);
        this->resultingShiftPeaks3D = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N * N);
        this->resultingShiftPeaks3DCorrelation = (fftw_complex *) fftw_malloc(
            sizeof(fftw_complex) * this->correlationN * this->correlationN * this->correlationN);

        this->magnitude1Shifted = (double *) malloc(sizeof(double) * N * N * N);
        this->magnitude2Shifted = (double *) malloc(sizeof(double) * N * N * N);
        this->voxelData1 = (double *) malloc(sizeof(double) * N * N * N);
        this->voxelData2 = (double *) malloc(sizeof(double) * N * N * N);

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


        planFourierToVoxel3D = fftw_plan_dft_3d(N, N, N, resultingPhaseDiff3D,
                                                resultingShiftPeaks3D, FFTW_BACKWARD, FFTW_ESTIMATE);
        planFourierToVoxel3DCorrelation = fftw_plan_dft_3d(this->correlationN, this->correlationN, this->correlationN,
                                                           resultingPhaseDiff3DCorrelation,
                                                           resultingShiftPeaks3DCorrelation, FFTW_BACKWARD,
                                                           FFTW_ESTIMATE);

        planVoxelToFourier3D = fftw_plan_dft_3d(N, N, N, inputSpacialData,
                                                spectrumOut, FFTW_FORWARD, FFTW_ESTIMATE);
        planVoxelToFourier3DCorrelation = fftw_plan_dft_3d(this->correlationN, this->correlationN, this->correlationN,
                                                           inputSpacialDataCorrelation,
                                                           spectrumOutCorrelation, FFTW_FORWARD, FFTW_ESTIMATE);
        // Precompute quaternion tables for direct z-y-z formula
        int S = bwOut * 2;
        quatTableSizeBeta = S;
        quatTableSizeSum = 4 * bwOut;
        quatTableDiffOffset = 2 * bwOut - 1;

        quatCosBeta2 = (double*) malloc(sizeof(double) * S);
        quatSinBeta2 = (double*) malloc(sizeof(double) * S);
        quatCosSum = (double*) malloc(sizeof(double) * quatTableSizeSum);
        quatSinSum = (double*) malloc(sizeof(double) * quatTableSizeSum);
        quatCosDiff = (double*) malloc(sizeof(double) * quatTableSizeSum);
        quatSinDiff = (double*) malloc(sizeof(double) * quatTableSizeSum);

        for (int i = 0; i < S; i++) {
            double beta2 = i * M_PI / (2.0 * N);
            quatCosBeta2[i] = std::cos(beta2);
            quatSinBeta2[i] = std::sin(beta2);
        }
        for (int s = 0; s < quatTableSizeSum; s++) {
            double sa = s * M_PI / N;
            quatCosSum[s] = std::cos(sa);
            quatSinSum[s] = std::sin(sa);
        }
        for (int d = -(2 * bwOut - 1); d <= 2 * bwOut - 1; d++) {
            double da = d * M_PI / N;
            quatCosDiff[d + quatTableDiffOffset] = std::cos(da);
            quatSinDiff[d + quatTableDiffOffset] = std::sin(da);
        }

        // Precompute KD tree using direct quaternion formula
        std::vector<My4DPoint> listOfQuaternionCorrelation;
        listOfQuaternionCorrelation.reserve(S * S * S);
        for (int i = 0; i < S; i++) {
            double cb = quatCosBeta2[i];
            double sb = quatSinBeta2[i];
            for (int j = 0; j < S; j++) {
                for (int k = 0; k < S; k++) {
                    double qw =  cb * quatCosSum[j + k];
                    double qx =  sb * quatSinDiff[j - k + quatTableDiffOffset];
                    double qy = -sb * quatCosDiff[j - k + quatTableDiffOffset];
                    double qz = -cb * quatSinSum[j + k];
                    if (qw < 0) {
                        qw = -qw;
                        qx = -qx;
                        qy = -qy;
                        qz = -qz;
                    }
                    listOfQuaternionCorrelation.emplace_back(qw, qx, qy, qz, 0.0);
                }
            }
        }

        kdt::KDTree<My4DPoint> *rotationKDTree = new kdt::KDTree<My4DPoint>(listOfQuaternionCorrelation);

        // use KD tree for computation of all the different things. Create a lookUpDataset for it.
        double radiusOfKDTree; // 128: 0.025 64: 0.05 32: 0.1 16: 0.2 256: 0.0125 512: 0.00625
        switch (this->N) {
            case 16:
                radiusOfKDTree = 0.2;
                break;
            case 32:
                radiusOfKDTree = 0.1;
                break;
            case 64:
                radiusOfKDTree = 0.05; //was 0.05
                break;
            case 128:
                radiusOfKDTree = 0.025;
                break;
            case 256:
                radiusOfKDTree = 0.0125;
                break;
            case 512:
                radiusOfKDTree = 0.00625;
                break;
        }
        //        My4DPoint testQuere = rotationKDTree->getPoint(0);
        //        std::vector<int> ni23 = rotationKDTree->radiusSearch(testQuere, radiusOfKDTree);
        //
        //        std::cout << ni23.size() << std::endl;


        //        std::vector<std::vector<int>> lookupTableForCorrelations;
        for (int i = 0; i < listOfQuaternionCorrelation.size(); i++) {
            My4DPoint quere = rotationKDTree->getPoint(i);
            std::vector<int> ni = rotationKDTree->radiusSearch(quere, radiusOfKDTree);
            if (abs(quere[0]) < 0.01) {
                My4DPoint quere2;
                quere2.correlation = quere.correlation;
                quere2[0] = quere[0];
                quere2[1] = -quere[1];
                quere2[2] = -quere[2];
                quere2[3] = -quere[3];
                std::vector<int> ni2 = rotationKDTree->radiusSearch(quere2, radiusOfKDTree);
                ni.insert(ni.end(), ni2.begin(), ni2.end());
            }
            //            std::cout << ni.size() << std::endl;
            this->lookupTableForCorrelations.push_back(ni);
        }

        // Precompute spherical projection lookup tables for 3D descriptor projection
        int bandwidth = N / 2;
        sinTheta3D = (double*) malloc(sizeof(double) * N);
        cosTheta3D = (double*) malloc(sizeof(double) * N);
        sinPhi3D = (double*) malloc(sizeof(double) * N);
        cosPhi3D = (double*) malloc(sizeof(double) * N);
        xAngle3D = (double*) malloc(sizeof(double) * N * N);
        yAngle3D = (double*) malloc(sizeof(double) * N * N);
        zAngle3D = (double*) malloc(sizeof(double) * N * N);

        for (int i = 0; i < N; i++) {
            double theta = M_PI * i / (2.0 * (bandwidth - 1));
            sinTheta3D[i] = std::sin(theta);
            cosTheta3D[i] = std::cos(theta);
        }
        for (int j = 0; j < N; j++) {
            double phi = M_PI * j / bandwidth;
            sinPhi3D[j] = std::sin(phi);
            cosPhi3D[j] = std::cos(phi);
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                xAngle3D[i * N + j] = sinTheta3D[i] * cosPhi3D[j];
                yAngle3D[i * N + j] = sinTheta3D[i] * sinPhi3D[j];
                zAngle3D[i * N + j] = cosTheta3D[i];
            }
        }

        // Reusable OpenCV objects
        clahe3D = cv::createCLAHE();
        magCLAHE1_3D = cv::Mat(N, N, CV_8UC1);
        magCLAHE2_3D = cv::Mat(N, N, CV_8UC1);
    }

    ~softRegistrationClass3D() {
        sofftCorrelationObject3D.~softCorrelationClass3D();
        free(sinTheta3D);
        free(cosTheta3D);
        free(sinPhi3D);
        free(cosPhi3D);
        free(xAngle3D);
        free(yAngle3D);
        free(zAngle3D);
        free(quatCosBeta2);
        free(quatSinBeta2);
        free(quatCosSum);
        free(quatSinSum);
        free(quatCosDiff);
        free(quatSinDiff);


        //        free(this->resultingCorrelationDouble);
        //        fftw_free(this->resultingCorrelationComplex);
        //        fftw_free(this->resultingPhaseDiff2D );
        //        fftw_free(this->resultingShiftPeaks2D);
        //        fftw_free(this->magnitude1Shifted );
        //        fftw_free(this->magnitude2Shifted );
        //        fftw_free(this->voxelData1 );
        //        fftw_free(this->voxelData2 );
        //        fftw_free(this->spectrumOut );
        //        fftw_free(this->phase1);
        //        fftw_free(this->phase2);
        //        fftw_free(this->magnitude1 );
        //        fftw_free(this->magnitude2 );
        //        fftw_free(resampledMagnitudeSO3_1 );
        //        fftw_free(resampledMagnitudeSO3_2);
        //        fftw_free(resampledMagnitudeSO3_1TMP);
        //        fftw_free(resampledMagnitudeSO3_2TMP );
        //        fftw_free(inputSpacialData);
        //        fftw_destroy_plan(planFourierToVoxel2D);
        //        fftw_destroy_plan(planVoxelToFourier3D);
        //        fftw_destroy_plan(planVoxelToFourier2D);
    }


    double
    getSpectrumFromVoxelData3D(const double voxelData[], double magnitude[], double phase[], bool gaussianBlur = false);

   double
    getSpectrumFromVoxelData3DCorrelation(const double voxelData[], fftw_complex *complexOut,
                                           bool gaussianBlur = false);

    double
    sofftRegistrationVoxel2DRotationOnly(double voxelData1Input[], double voxelData2Input[], double goodGuessAlpha,
                                         double &covariance,
                                         bool debug = false);

   std::vector<transformationPeakfs3D>
     sofftRegistrationVoxel3DListOfPossibleTransformations(double voxelData1Input[], double voxelData2Input[],
                                                             bool debug = false, bool useClahe = true,
                                                             bool benchmark = false, double sizeVoxel = 1,
                                                             double r_min = 0.0,
                                                             double r_max = 0.0,
                                                             double level_potential_rotation = 0.01,
                                                             double level_potential_translation = 0.1,
                                                             bool set_r_manual = false, int normalization = 0,
                                                             bool useSimpleRotationPeak = false, bool useSimpleTranslationPeak = false,
                                                             BenchmarkTimings3D* timings = nullptr);
    transformationPeakfs3D
    sofftRegistrationVoxel3DOneSolution(double voxelData1Input[], double voxelData2Input[], tf2::Quaternion initGuessOrientation,tf2::Vector3 initGuessPosition,
                                                            bool debug = false, bool useClahe = true,
                                                            bool benchmark = false, double sizeVoxel = 1,
                                                            double r_min = 0.0,
                                                            double r_max = 0.0,
                                                            double level_potential_rotation = 0.01,
                                                            double level_potential_translation = 0.1,
                                                            bool set_r_manual = false, int normalization = 0,
                                                            bool useSimpleRotationPeak = false, bool useSimpleTranslationPeak = false);

    bool isPeak(cv::Mat mx[], std::vector<cv::Point> &conn_points);

    cv::Mat imregionalmax(cv::Mat &src);

    static double normalizationFactorCalculation(int x, int y, int z, int currentN);

    cv::Mat opencv_imextendedmax(cv::Mat &inputMatrix, double hParam);

    void imextendedmax_imreconstruct(cv::Mat g, cv::Mat f, cv::Mat &dest);

    std::vector<translationPeak3D>
    peakDetectionOf3DCorrelationFindPeaksLibraryFromFFTW_COMPLEX(fftw_complex *inputcorrelation, double cellSize) const;

    std::vector<translationPeak3D>
    peakDetectionOf3DCorrelationFindPeaksLibrary(const double *inputcorrelation, int dimensionN, double cellSize,double level_potential_translation=0.01) const;

    static std::vector<rotationPeak4D>
    peakDetectionOf4DCorrelationFindPeaksLibrary(const double *inputcorrelation, long dimensionN, double cellSize);

    std::vector<rotationPeak4D>
    peakDetectionOf4DCorrelationWithKDTreeFindPeaksLibrary(std::vector<My4DPoint> listOfQuaternionCorrelation,double level_potential_rotation=0.01);

    static std::vector<rotationPeak4D>
    peakDetectionOf4DCorrelationSimpleMax(std::vector<My4DPoint> listOfQuaternionCorrelation);

    static std::vector<rotationPeak4D>
    peakDetectionOf4DCorrelationSimpleMaxRaw(fftw_complex* resultingCorrelationComplex, int bwOut, int N);

    std::vector<translationPeak3D>
    peakDetectionOf3DCorrelationSimpleMax(const double *inputcorrelation, int dimensionN, double cellSize) const;

    static double getPixelValueInterpolated(Eigen::Vector3d positionVector, const double *volumeData, int dimensionN);

    int getSizeOfRegistration() const;
    Eigen::Vector3d subPixelComputation(const double *inputcorrelation, int dimensionN,double xPosition,double yPosition,double zPosition) const;
    //    int index3D(int x, int y, int z,int NInput);
private: //here everything is created. malloc is done in the constructor


    int N; // correlationN;//describes the size of the overall voxel system + correlation N
    int correlationN;
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
    softCorrelationClass3D sofftCorrelationObject3D;
    fftw_complex *resultingCorrelationComplex;
    fftw_complex *resultingPhaseDiff3D;
    fftw_complex *resultingPhaseDiff3DCorrelation;
    fftw_complex *resultingShiftPeaks3D;
    fftw_complex *resultingShiftPeaks3DCorrelation;
    double *resultingCorrelationDouble;
    fftw_complex *inputSpacialData;
    fftw_complex *inputSpacialDataCorrelation;
    fftw_plan planVoxelToFourier3D;
    fftw_plan planVoxelToFourier3DCorrelation;
    fftw_plan planFourierToVoxel3D;
    fftw_plan planFourierToVoxel3DCorrelation;
    //    kdt::KDTree<My4DPoint> *rotationKDTree;
    std::vector<std::vector<int> > lookupTableForCorrelations;

    // Precomputed spherical projection lookup tables
    double* sinTheta3D;
    double* cosTheta3D;
    double* sinPhi3D;
    double* cosPhi3D;
    double* xAngle3D;
    double* yAngle3D;
    double* zAngle3D;

    // Reusable OpenCV objects
    cv::Ptr<cv::CLAHE> clahe3D;
    cv::Mat magCLAHE1_3D;
    cv::Mat magCLAHE2_3D;

    // Precomputed quaternion tables for direct z-y-z formula
    int quatTableSizeBeta;
    int quatTableSizeSum;
    int quatTableDiffOffset;
    double* quatCosBeta2;
    double* quatSinBeta2;
    double* quatCosSum;
    double* quatSinSum;
    double* quatCosDiff;
    double* quatSinDiff;
};

#endif //FSREGISTRATION_SOFTREGISTRATIONCLASS3D_H
