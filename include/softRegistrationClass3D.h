//
// Created by aya on 08.12.23.
//

#ifndef FS2D_SOFTREGISTRATIONCLASS3D_H
#define FS2D_SOFTREGISTRATIONCLASS3D_H

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

struct rotationPeak3D {
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

class softRegistrationClass3D {
public:
    softRegistrationClass3D(int N, int bwOut, int bwIn, int degLim) : sofftCorrelationObject3D(N, bwOut, bwIn,
                                                                                               degLim) {
        this->N = N;
//        this->correlationN = N * 2 - 1;
        this->correlationN = N ;

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
        this->phase1Correlation = (double *) malloc(
                sizeof(double) * this->correlationN * this->correlationN * this->correlationN);
        this->phase2Correlation = (double *) malloc(
                sizeof(double) * this->correlationN * this->correlationN * this->correlationN);
        this->magnitude1Correlation = (double *) malloc(
                sizeof(double) * this->correlationN * this->correlationN * this->correlationN);
        this->magnitude2Correlation = (double *) malloc(
                sizeof(double) * this->correlationN * this->correlationN * this->correlationN);
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

    std::vector<rotationPeak3D>
    sofftRegistrationVoxel3DListOfPossibleRotations(double voxelData1Input[], double voxelData2Input[],
                                                    bool debug = false, bool multipleRadii = false,
                                                    bool useClahe = true, bool useHamming = true);


    bool isPeak(cv::Mat mx[], std::vector<cv::Point> &conn_points);

    cv::Mat imregionalmax(cv::Mat &src);

    double normalizationFactorCalculation(int x, int y,int z);

    cv::Mat opencv_imextendedmax(cv::Mat &inputMatrix, double hParam);

    void imextendedmax_imreconstruct(cv::Mat g, cv::Mat f, cv::Mat &dest);

    std::vector<rotationPeak3D> peakDetectionOf3DCorrelationFindPeaksLibraryFromFFTW_COMPLEX(fftw_complex* inputcorrelation,double cellSize);
    std::vector<rotationPeak3D> peakDetectionOf3DCorrelationFindPeaksLibrary(double* inputcorrelation, int dimensionN, double cellSize);
    std::vector<rotationPeak4D> peakDetectionOf4DCorrelationFindPeaksLibrary(double* inputcorrelation, long dimensionN, double cellSize);
    std::vector<rotationPeak4D> peakDetectionOf4DCorrelationWithKDTreeFindPeaksLibrary(std::vector<My4DPoint> listOfQuaternionCorrelation);
    double getPixelValueInterpolated(Eigen::Vector3d positionVector,double *volumeData);
//    int index3D(int x, int y, int z,int NInput);
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
};

#endif //FS2D_SOFTREGISTRATIONCLASS3D_H
