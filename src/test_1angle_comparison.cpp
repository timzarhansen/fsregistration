//
// Test script to compare old SO(3) method vs new 1-angle method
//

#include "softRegistrationClass.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

#define DEBUG_RESULTS_2D "/home/tim-external/volumeROS/src/fsregistration/debug_results/2d/"
#include <opencv4/opencv2/imgproc.hpp>
#include <fstream>
#include <random>
#include <algorithm>

// Forward declaration - defined in softRegistrationClass.cpp
extern double angleDifference(double angle1, double angle2);

void saveCorrelationToCSV(const std::vector<float>& correlation, const std::vector<float>& angles, const std::string& filename) {
    std::ofstream file(filename);
    for (size_t i = 0; i < correlation.size(); i++) {
        file << angles[i] * 180.0 / M_PI << "," << correlation[i] << "\n";
    }
    file.close();
}

int main(int argc, char** argv) {
    std::string img1Path = "/home/tim-external/volumeROS/src/fsregistration/exampleData/voxelScan1.jpg";
    std::string img2Path = "/home/tim-external/volumeROS/src/fsregistration/exampleData/voxelScan2.jpg";
    
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
    
    int N = 512;
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

    std::cout << "\n1. Computing SO3 method correlation array (full SO(3))..." << std::endl;
    auto startOld = std::chrono::steady_clock::now();
    auto [corrOld, anglesOld] = registrar.compute1AngleCorrelationArraySO3(
        voxelData1, voxelData2, false, true, true, false);
    auto endOld = std::chrono::steady_clock::now();
    double timeOld = std::chrono::duration<double>(endOld - startOld).count();

    std::cout << "   SO3 method completed in " << (timeOld * 1000) << " ms" << std::endl;
    std::cout << "   Correlation array size: " << corrOld.size() << " points" << std::endl;

    std::cout << "\n2. Computing Direct method correlation array (1-angle direct)..." << std::endl;
    auto startNew = std::chrono::steady_clock::now();
    auto [corrNew, anglesNew] = registrar.compute1AngleCorrelationArrayDirect(
        voxelData1, voxelData2, false, true, true, false);
    auto endNew = std::chrono::steady_clock::now();
    double timeNew = std::chrono::duration<double>(endNew - startNew).count();

    std::cout << "   NEW method completed in " << (timeNew * 1000) << " ms" << std::endl;
    std::cout << "   Correlation array size: " << corrNew.size() << " points" << std::endl;

    // Save correlation arrays to CSV for plotting
    saveCorrelationToCSV(corrOld, anglesOld, std::string(DEBUG_RESULTS_2D) + "correlation_OLD.csv");
    saveCorrelationToCSV(corrNew, anglesNew, std::string(DEBUG_RESULTS_2D) + "correlation_NEW.csv");
    std::cout << "\n   Saved correlation arrays to CSV files for plotting" << std::endl;

    std::cout << "\n=== Correlation Array Comparison ===" << std::endl;
    std::cout << "Speedup: " << (timeOld / timeNew) << "x" << std::endl;

    // Compare 10 random points with fixed seed for reproducibility
    std::mt19937 rng(42);
    std::uniform_int_distribution<> distrib(0, static_cast<int>(corrOld.size()) - 1);

    std::cout << "\nComparing 10 random points:" << std::endl;

    double maxDiff = 0;
    double sumDiff = 0;
    double sumSqDiff = 0;

    for (int trial = 0; trial < 10; trial++) {
        int idx = distrib(rng);
        double diff = std::abs(corrOld[idx] - corrNew[idx]);
        maxDiff = std::max(maxDiff, diff);
        sumDiff += diff;
        sumSqDiff += diff * diff;

        std::cout << "  Point " << (trial + 1) << " (angle="
                  << anglesOld[idx] * 180.0 / M_PI << "°): "
                  << "OLD=" << corrOld[idx] << ", NEW=" << corrNew[idx]
                  << ", diff=" << diff << std::endl;
    }

    double meanDiff = sumDiff / 10;
    double rmseDiff = std::sqrt(sumSqDiff / 10);

    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << "Maximum difference: " << maxDiff << std::endl;
    std::cout << "Mean absolute difference: " << meanDiff << std::endl;
    std::cout << "RMSE: " << rmseDiff << std::endl;

    // Test pass/fail based on maximum difference
    double tolerance = 0.01;
    std::cout << "\n=== Test Result ===" << std::endl;
    if (maxDiff < tolerance) {
        std::cout << "✓ TEST PASSED (max diff < " << tolerance << ")" << std::endl;
    } else {
        std::cout << "✗ TEST FAILED (max diff >= " << tolerance << ")" << std::endl;
    }

    free(voxelData1);
    free(voxelData2);

    return 0;
}
