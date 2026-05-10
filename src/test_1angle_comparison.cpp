//
// Test script to compare old SO(3) method vs new 1-angle method
//

#include "softRegistrationClass.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>

double angleDifference(double angle1, double angle2) {
    return atan2(sin(angle1 - angle2), cos(angle1 - angle2));
}

int main(int argc, char** argv) {
    std::string img1Path = "/workspaces/opencodeTestProject/fsregistration/exampleData/voxelScan1.jpg";
    std::string img2Path = "/workspaces/opencodeTestProject/fsregistration/exampleData/voxelScan2.jpg";
    
    if (argc > 2) {
        img1Path = argv[1];
        img2Path = argv[2];
    }
    
    std::cout << "=== 1-Angle Correlation Comparison Test ===" << std::endl;
    std::cout << "Loading images..." << std::endl;
    std::cout << "  Image 1: " << img1Path << std::endl;
    std::cout << "  Image 2: " << img2Path << std::endl;
    
    cv::Mat img1 = cv::imread(img1Path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(img2Path, cv::IMREAD_GRAYSCALE);
    
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load images!" << std::endl;
        return 1;
    }
    
    std::cout << "  Image 1 size: " << img1.cols << "x" << img1.rows << std::endl;
    std::cout << "  Image 2 size: " << img2.cols << "x" << img2.rows << std::endl;
    
    int N = 128;
    int bwOut = N / 2;
    int bwIn = N / 2;
    int degLim = bwOut - 1;
    
    std::cout << "\nParameters: N=" << N << ", bwOut=" << bwOut 
              << ", bwIn=" << bwIn << ", degLim=" << degLim << std::endl;
    
    double* voxelData1 = (double*)malloc(N * N * sizeof(double));
    double* voxelData2 = (double*)malloc(N * N * sizeof(double));
    
    cv::Mat img1Resized, img2Resized;
    cv::resize(img1, img1Resized, cv::Size(N, N));
    cv::resize(img2, img2Resized, cv::Size(N, N));
    
    for (int i = 0; i < N * N; i++) {
        voxelData1[i] = (double)img1Resized.ptr<uchar>(i / N)[i % N];
        voxelData2[i] = (double)img2Resized.ptr<uchar>(i / N)[i % N];
    }
    
    std::cout << "\nCreating softRegistrationClass..." << std::endl;
    softRegistrationClass registrar(N, bwOut, bwIn, degLim);
    
    std::cout << "\n1. Testing OLD method (full SO(3) correlation)..." << std::endl;
    auto startOld = std::chrono::steady_clock::now();
    std::vector<rotationPeakfs2D> peaksOld = registrar.sofftRegistrationVoxel2DListOfPossibleRotations(
        voxelData1, voxelData2, false, false, true, true);
    auto endOld = std::chrono::steady_clock::now();
    double timeOld = std::chrono::duration<double>(endOld - startOld).count();
    
    std::cout << "   OLD method completed in " << (timeOld * 1000) << " ms" << std::endl;
    std::cout << "   Found " << peaksOld.size() << " peaks" << std::endl;
    
    if (peaksOld.size() > 0) {
        std::cout << "   Top 3 peaks (OLD):" << std::endl;
        for (size_t i = 0; i < std::min(peaksOld.size(), (size_t)3); i++) {
            std::cout << "     [" << i << "] angle=" << peaksOld[i].angle * 180.0 / M_PI << "°"
                      << ", correlation=" << peaksOld[i].peakCorrelation << std::endl;
        }
    }
    
    std::cout << "\n2. Testing NEW method (1-angle correlation)..." << std::endl;
    auto startNew = std::chrono::steady_clock::now();
    std::vector<rotationPeakfs2D> peaksNew = registrar.sofftRegistrationVoxel2DListOfPossibleRotations1Angle(
        voxelData1, voxelData2, false, false, true, true);
    auto endNew = std::chrono::steady_clock::now();
    double timeNew = std::chrono::duration<double>(endNew - startNew).count();
    
    std::cout << "   NEW method completed in " << (timeNew * 1000) << " ms" << std::endl;
    std::cout << "   Found " << peaksNew.size() << " peaks" << std::endl;
    
    if (peaksNew.size() > 0) {
        std::cout << "   Top 3 peaks (NEW):" << std::endl;
        for (size_t i = 0; i < std::min(peaksNew.size(), (size_t)3); i++) {
            std::cout << "     [" << i << "] angle=" << peaksNew[i].angle * 180.0 / M_PI << "°"
                      << ", correlation=" << peaksNew[i].peakCorrelation << std::endl;
        }
    }
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Speedup: " << (timeOld / timeNew) << "x" << std::endl;
    
    if (peaksOld.size() > 0 && peaksNew.size() > 0) {
        double bestAngleOld = peaksOld[0].angle;
        double bestAngleNew = peaksNew[0].angle;
        double angleDiff = std::abs(angleDifference(bestAngleOld, bestAngleNew));
        double angleDiffDeg = angleDiff * 180.0 / M_PI;
        
        std::cout << "Best angle (OLD): " << bestAngleOld * 180.0 / M_PI << "°" << std::endl;
        std::cout << "Best angle (NEW): " << bestAngleNew * 180.0 / M_PI << "°" << std::endl;
        std::cout << "Angle difference: " << angleDiffDeg << "°" << std::endl;
        
        double toleranceDeg = 2.8;
        if (angleDiffDeg < toleranceDeg) {
            std::cout << "\n✓ TEST PASSED (angle difference < " << toleranceDeg << "°)" << std::endl;
        } else {
            std::cout << "\n✗ TEST FAILED (angle difference >= " << toleranceDeg << "°)" << std::endl;
            std::cout << "\nNote: This may indicate:" << std::endl;
            std::cout << "  - Different correlation computation" << std::endl;
            std::cout << "  - Peak detection differences" << std::endl;
            std::cout << "  - Need to review the new method implementation" << std::endl;
        }
    } else {
        std::cout << "\n⚠ WARNING: Could not compare angles (missing peaks)" << std::endl;
    }
    
    free(voxelData1);
    free(voxelData2);
    
    return 0;
}
