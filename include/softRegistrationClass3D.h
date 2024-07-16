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

struct translationPeak3D {
    double xTranslation;
    double yTranslation;
    double zTranslation;
    double persistence;
    double levelPotential;
    double correlationHeight;
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


class softRegistrationClass3D {
public:
    softRegistrationClass3D(int N, int bwOut, int bwIn, int degLim) : sofftCorrelationObject3D(N, bwOut, bwIn,
                                                                                               degLim) {
        this->N = N;
//        this->N = N * 2 - 1;
//        this->N = N;

        this->bwOut = bwOut;
        this->bwIn = bwIn;
        this->degLim = degLim;
        this->resultingCorrelationDouble = (double *) malloc(
                sizeof(double) * this->N * this->N * this->N);
        this->resultingCorrelationComplex = fftw_alloc_complex(8 * bwOut * bwOut * bwOut);

        this->resultingPhaseDiff3D = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N * N);
        this->resultingPhaseDiff3DCorrelation = (fftw_complex *) fftw_malloc(
                sizeof(fftw_complex) * this->N * this->N * this->N);
        this->resultingShiftPeaks3D = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N * N);
        this->resultingShiftPeaks3DCorrelation = (fftw_complex *) fftw_malloc(
                sizeof(fftw_complex) * this->N * this->N * this->N);

        this->magnitude1Shifted = (double *) malloc(sizeof(double) * N * N * N);
        this->magnitude2Shifted = (double *) malloc(sizeof(double) * N * N * N);
        this->voxelData1 = (double *) malloc(sizeof(double) * N * N * N);
        this->voxelData2 = (double *) malloc(sizeof(double) * N * N * N);

        this->spectrumOut = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N * N);
        this->spectrumOutCorrelation = (fftw_complex *) fftw_malloc(
                sizeof(fftw_complex) * this->N * this->N * this->N);
        this->phase1 = (double *) malloc(sizeof(double) * N * N * N);
        this->phase2 = (double *) malloc(sizeof(double) * N * N * N);
        this->magnitude1 = (double *) malloc(sizeof(double) * N * N * N);
        this->magnitude2 = (double *) malloc(sizeof(double) * N * N * N);
        this->phase1Correlation = (double *) malloc(
                sizeof(double) * this->N * this->N * this->N);
        this->phase2Correlation = (double *) malloc(
                sizeof(double) * this->N * this->N * this->N);
        this->magnitude1Correlation = (double *) malloc(
                sizeof(double) * this->N * this->N * this->N);
        this->magnitude2Correlation = (double *) malloc(
                sizeof(double) * this->N * this->N * this->N);
        resampledMagnitudeSO3_1 = (double *) malloc(sizeof(double) * N * N);
        resampledMagnitudeSO3_2 = (double *) malloc(sizeof(double) * N * N);
        resampledMagnitudeSO3_1TMP = (double *) malloc(sizeof(double) * N * N);
        resampledMagnitudeSO3_2TMP = (double *) malloc(sizeof(double) * N * N);
        inputSpacialData = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N * N);
        inputSpacialDataCorrelation = (fftw_complex *) fftw_malloc(
                sizeof(fftw_complex) * this->N * this->N * this->N);


        planFourierToVoxel3D = fftw_plan_dft_3d(N, N, N, resultingPhaseDiff3D,
                                                resultingShiftPeaks3D, FFTW_BACKWARD, FFTW_ESTIMATE);
        planFourierToVoxel3DCorrelation = fftw_plan_dft_3d(this->N, this->N, this->N,
                                                           resultingPhaseDiff3DCorrelation,
                                                           resultingShiftPeaks3DCorrelation, FFTW_BACKWARD,
                                                           FFTW_ESTIMATE);

        planVoxelToFourier3D = fftw_plan_dft_3d(N, N, N, inputSpacialData,
                                                spectrumOut, FFTW_FORWARD, FFTW_ESTIMATE);
        planVoxelToFourier3DCorrelation = fftw_plan_dft_3d(this->N, this->N, this->N,
                                                           inputSpacialDataCorrelation,
                                                           spectrumOutCorrelation, FFTW_FORWARD, FFTW_ESTIMATE);
        //precalculating KD tree for faster neighbour calculation
        std::vector<My4DPoint> listOfQuaternionCorrelation;
        for (int i = 0; i < bwOut * 2; i++) {
            for (int j = 0; j < bwOut * 2; j++) {
                for (int k = 0; k < bwOut * 2; k++) {

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
                    currentPoint.correlation = 0;
                    currentPoint[0] = quaternionResult.w();
                    currentPoint[1] = quaternionResult.x();
                    currentPoint[2] = quaternionResult.y();
                    currentPoint[3] = quaternionResult.z();

                    listOfQuaternionCorrelation.push_back(currentPoint);
                }
            }
        }

        kdt::KDTree<My4DPoint> *rotationKDTree = new kdt::KDTree<My4DPoint>(listOfQuaternionCorrelation);

        // use KD tree for computation of all the different things. Create a lookUpDataset for it.
        double radiusOfKDTree;// 128: 0.025 64: 0.05 32: 0.1 16: 0.2 256: 0.0125 512: 0.00625
        switch (this->N) {
            case 16:
                radiusOfKDTree = 0.2;
                break;
            case 32:
                radiusOfKDTree = 0.1;
                break;
            case 64:
                radiusOfKDTree = 0.05;
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
    }

    ~softRegistrationClass3D() {
        sofftCorrelationObject3D.~softCorrelationClass3D();


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
    getSpectrumFromVoxelData3D(double voxelData[], double magnitude[], double phase[], bool gaussianBlur = false);


    double
    sofftRegistrationVoxel2DRotationOnly(double voxelData1Input[], double voxelData2Input[], double goodGuessAlpha,
                                         double &covariance,
                                         bool debug = false);

    std::vector<transformationPeakfs3D>
    sofftRegistrationVoxel3DListOfPossibleTransformations(double voxelData1Input[], double voxelData2Input[],
                                                          bool debug = false, bool useClahe = true,
                                                          bool timeStuff = false);


    bool isPeak(cv::Mat mx[], std::vector<cv::Point> &conn_points);

    cv::Mat imregionalmax(cv::Mat &src);

    double normalizationFactorCalculation(int x, int y, int z);

    cv::Mat opencv_imextendedmax(cv::Mat &inputMatrix, double hParam);

    void imextendedmax_imreconstruct(cv::Mat g, cv::Mat f, cv::Mat &dest);

    std::vector<translationPeak3D>
    peakDetectionOf3DCorrelationFindPeaksLibraryFromFFTW_COMPLEX(fftw_complex *inputcorrelation, double cellSize);

    std::vector<translationPeak3D>
    peakDetectionOf3DCorrelationFindPeaksLibrary(double *inputcorrelation, int dimensionN, double cellSize);

    std::vector<rotationPeak4D>
    peakDetectionOf4DCorrelationFindPeaksLibrary(double *inputcorrelation, long dimensionN, double cellSize);

    std::vector<rotationPeak4D>
    peakDetectionOf4DCorrelationWithKDTreeFindPeaksLibrary(std::vector<My4DPoint> listOfQuaternionCorrelation);

    double getPixelValueInterpolated(Eigen::Vector3d positionVector, double *volumeData, int dimensionN);

    int getSizeOfRegistration();
//    int index3D(int x, int y, int z,int NInput);
private://here everything is created. malloc is done in the constructor




    int N;// correlationN;//describes the size of the overall voxel system + correlation N
    int bwOut, bwIn, degLim;
    double *voxelData1;
    double *voxelData2;
//    fftw_complex *spectrum1;
//    fftw_complex *spectrum2;
    fftw_complex *spectrumOut;
    fftw_complex *spectrumOutCorrelation;
    double *magnitude1;
    double *magnitude2;
    double *magnitude1Correlation;
    double *magnitude2Correlation;
    double *phase1;
    double *phase2;
    double *phase1Correlation;
    double *phase2Correlation;
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
    std::vector<std::vector<int>> lookupTableForCorrelations;
};

#endif //FSREGISTRATION_SOFTREGISTRATIONCLASS3D_H
