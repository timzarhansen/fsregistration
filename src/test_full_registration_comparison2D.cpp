//
// Test full 2D registration with OLD (full SO(3)) vs NEW (1-angle direct) methods
//

#include "softRegistrationClass.h"
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

// Forward declaration
extern double angleDifference(double angle1, double angle2);

struct TestResult {
    std::string name;
    double totalTimeMs;
    int numRotationPeaks;
    int numTranslationPeaks;
    Eigen::Matrix4d bestTransformation;
};

void printTransformationMatrix(const std::string& label, const Eigen::Matrix4d& transformation, double timeMs) {
    std::cout << "\n" << label << std::endl;

    // Extract rotation angle (around Z axis for 2D)
    double rotationAngle = std::atan2(transformation(1, 0), transformation(0, 0));
    double rotationAngleDeg = rotationAngle * 180.0 / M_PI;

    // Extract translation
    double transX = transformation(0, 3);
    double transY = transformation(1, 3);

    std::cout << "  Rotation: " << rotationAngle << " rad (" << rotationAngleDeg << "°)" << std::endl;
    std::cout << "  Translation: (" << transX << ", " << transY << ") pixels" << std::endl;
    std::cout << "  Total time: " << timeMs << " ms" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Full 2D Registration Comparison Test ===" << std::endl;
    
    // Image paths (hardcoded as requested)
    std::string img1Path = "/home/tim-external/volumeROS/src/fsregistration/exampleData/voxelScan1.jpg";
    std::string img2Path = "/home/tim-external/volumeROS/src/fsregistration/exampleData/voxelScan2.jpg";
    
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
    
    // Parameters
    int N = 256;
    int bwOut = N / 2;
    int bwIn = N / 2;
    int degLim = bwOut - 1;
    double cellSize = 0.005;  // 5mm
    bool useGauss = true;
    double potentialNecessaryForPeak = 0.1;
    
    std::cout << "\nParameters: N=" << N << ", bwOut=" << bwOut 
              << ", bwIn=" << bwIn << ", degLim=" << degLim << std::endl;
    
    // Resize images to N×N and convert to voxel data
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
    
    // Initial guess: Identity matrix
    Eigen::Matrix4d initialGuess = Eigen::Matrix4d::Identity();
    std::cout << "Initial guess: Identity matrix" << std::endl;
    
    // ========================================
    // Method 1: OLD (Full SO(3))
    // ========================================
    std::cout << "\n\n--- OLD Method (Full SO(3)) ---" << std::endl;

    auto startTotalOld = std::chrono::steady_clock::now();

    std::vector<transformationPeakfs2D> allTransformationsOld = registrar.registrationOfTwoVoxelsSO3(
        voxelData1, voxelData2, cellSize, useGauss, false, potentialNecessaryForPeak, false, true, true, true);

    auto endTotalOld = std::chrono::steady_clock::now();
    double totalTimeOld = std::chrono::duration<double, std::milli>(endTotalOld - startTotalOld).count();

    // Count peaks
    int numRotPeaksOld = allTransformationsOld.size();
    int numTransPeaksOld = 0;
    for (const auto& sol : allTransformationsOld) {
        numTransPeaksOld += sol.potentialTranslations.size();
    }

    // Find best transformation (closest to initial guess)
    transformationPeakfs2D bestTransformationOld;
    double bestDistanceOld = 100000;
    double initialRotation = std::atan2(initialGuess(1, 0), initialGuess(0, 0));
    for (auto& trans : allTransformationsOld) {
        double rotationDiff = std::abs(angleDifference(trans.potentialRotation.angle, initialRotation));
        for (auto& translation : trans.potentialTranslations) {
            double diffX = translation.translationSI.x() - initialGuess(0, 3);
            double diffY = translation.translationSI.y() - initialGuess(1, 3);
            double translationDistance = std::sqrt(diffX * diffX + diffY * diffY);
            double totalDistance = rotationDiff * 100 + translationDistance;
            if (totalDistance < bestDistanceOld) {
                bestTransformationOld = trans;
                bestDistanceOld = totalDistance;
            }
        }
    }

    // Build transformation matrix
    Eigen::Matrix4d resultOld = Eigen::Matrix4d::Identity();
    Eigen::AngleAxisd rotation_vectorOld(bestTransformationOld.potentialRotation.angle, Eigen::Vector3d(0, 0, 1));
    Eigen::Matrix3d rotMatrixOld = rotation_vectorOld.toRotationMatrix();
    resultOld.block<3, 3>(0, 0) = rotMatrixOld;
    resultOld(0, 3) = bestTransformationOld.potentialTranslations[0].translationSI.x();
    resultOld(1, 3) = bestTransformationOld.potentialTranslations[0].translationSI.y();

    std::cout << "\n  Rotation peaks found:    " << numRotPeaksOld << std::endl;
    std::cout << "  Translation peaks found: " << numTransPeaksOld << std::endl;

    printTransformationMatrix("Result:", resultOld, totalTimeOld);

    // ========================================
    // Method 2: NEW (1-Angle Direct)
    // ========================================
    std::cout << "\n\n--- NEW Method (1-Angle Direct) ---" << std::endl;

    auto startTotalNew = std::chrono::steady_clock::now();

    std::vector<transformationPeakfs2D> allTransformationsNew = registrar.registrationOfTwoVoxelsDirect(
        voxelData1, voxelData2, cellSize, useGauss, false, potentialNecessaryForPeak, false, true, true, true);

    auto endTotalNew = std::chrono::steady_clock::now();
    double totalTimeNew = std::chrono::duration<double, std::milli>(endTotalNew - startTotalNew).count();

    // Count peaks
    int numRotPeaksNew = allTransformationsNew.size();
    int numTransPeaksNew = 0;
    for (const auto& sol : allTransformationsNew) {
        numTransPeaksNew += sol.potentialTranslations.size();
    }

    // Find best transformation (closest to initial guess)
    transformationPeakfs2D bestTransformationNew;
    double bestDistanceNew = 100000;
    for (auto& trans : allTransformationsNew) {
        double rotationDiff = std::abs(angleDifference(trans.potentialRotation.angle, initialRotation));
        for (auto& translation : trans.potentialTranslations) {
            double diffX = translation.translationSI.x() - initialGuess(0, 3);
            double diffY = translation.translationSI.y() - initialGuess(1, 3);
            double translationDistance = std::sqrt(diffX * diffX + diffY * diffY);
            double totalDistance = rotationDiff * 100 + translationDistance;
            if (totalDistance < bestDistanceNew) {
                bestTransformationNew = trans;
                bestDistanceNew = totalDistance;
            }
        }
    }

    // Build transformation matrix
    Eigen::Matrix4d resultNew = Eigen::Matrix4d::Identity();
    Eigen::AngleAxisd rotation_vectorNew(bestTransformationNew.potentialRotation.angle, Eigen::Vector3d(0, 0, 1));
    Eigen::Matrix3d rotMatrixNew = rotation_vectorNew.toRotationMatrix();
    resultNew.block<3, 3>(0, 0) = rotMatrixNew;
    resultNew(0, 3) = bestTransformationNew.potentialTranslations[0].translationSI.x();
    resultNew(1, 3) = bestTransformationNew.potentialTranslations[0].translationSI.y();

    std::cout << "\n  Rotation peaks found:    " << numRotPeaksNew << std::endl;
    std::cout << "  Translation peaks found: " << numTransPeaksNew << std::endl;

    printTransformationMatrix("Result:", resultNew, totalTimeNew);
    
    // ========================================
    // Summary table
    // ========================================
    std::cout << "\n\n================================================================" << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << "================================================================" << std::endl;

    std::vector<TestResult> results;
    {
        TestResult r;
        r.name = "Full SO(3)";
        r.totalTimeMs = totalTimeOld;
        r.numRotationPeaks = numRotPeaksOld;
        r.numTranslationPeaks = numTransPeaksOld;
        r.bestTransformation = resultOld;
        results.push_back(r);
    }
    {
        TestResult r;
        r.name = "1-Angle Direct";
        r.totalTimeMs = totalTimeNew;
        r.numRotationPeaks = numRotPeaksNew;
        r.numTranslationPeaks = numTransPeaksNew;
        r.bestTransformation = resultNew;
        results.push_back(r);
    }

    std::cout << std::left << std::setw(28) << "Test"
              << std::right << std::setw(12) << "Time ms"
              << std::setw(10) << "Rot"
              << std::setw(10) << "Trans"
              << std::setw(14) << "Speedup" << std::endl;
    std::cout << std::string(74, '-') << std::endl;

    double baselineTime = results[0].totalTimeMs;
    for (const auto& r : results) {
        double speedup = (baselineTime > 0) ? (baselineTime / r.totalTimeMs) : 0;
        std::cout << std::left << std::setw(28) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(1) << r.totalTimeMs
                  << std::setw(10) << r.numRotationPeaks
                  << std::setw(10) << r.numTranslationPeaks
                  << std::setw(14) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    std::cout << std::string(74, '-') << std::endl;

    // ========================================
    // Comparison
    // ========================================
    std::cout << "\n\n=== Comparison ===" << std::endl;

    // Timing comparison
    std::cout << "\nTiming:" << std::endl;
    std::cout << "  Total: OLD=" << std::fixed << std::setprecision(1) << totalTimeOld << " ms, NEW="
              << std::fixed << std::setprecision(1) << totalTimeNew << " ms, speedup="
              << std::fixed << std::setprecision(2) << (totalTimeOld / totalTimeNew) << "x" << std::endl;
    
    // Result comparison
    std::cout << "\nResult comparison:" << std::endl;
    
    double rotationDiff = std::abs(angleDifference(bestTransformationOld.potentialRotation.angle, 
                                                    bestTransformationNew.potentialRotation.angle));
    double rotationDiffDeg = rotationDiff * 180.0 / M_PI;
    std::cout << "  Rotation difference: " << rotationDiff << " rad (" << rotationDiffDeg << "°)" << std::endl;
    
    double transDiffX = std::abs(bestTransformationOld.potentialTranslations[0].translationSI.x() - 
                                  bestTransformationNew.potentialTranslations[0].translationSI.x());
    double transDiffY = std::abs(bestTransformationOld.potentialTranslations[0].translationSI.y() - 
                                  bestTransformationNew.potentialTranslations[0].translationSI.y());
    std::cout << "  Translation difference: (" << transDiffX << ", " << transDiffY << ") pixels" << std::endl;
    
    // Pass/fail
    double rotationTolerance = 0.01;  // 0.57°
    double translationTolerance = 1.0;  // 1 pixel
    
    std::cout << "\n=== Test Result ===" << std::endl;
    if (rotationDiff < rotationTolerance && transDiffX < translationTolerance && transDiffY < translationTolerance) {
        std::cout << "✓ TEST PASSED (rotation diff < " << rotationTolerance 
                  << " rad, translation diff < " << translationTolerance << " pixels)" << std::endl;
    } else {
        std::cout << "✗ TEST FAILED" << std::endl;
        if (rotationDiff >= rotationTolerance) {
            std::cout << "  - Rotation difference too large: " << rotationDiff << " rad" << std::endl;
        }
        if (transDiffX >= translationTolerance || transDiffY >= translationTolerance) {
            std::cout << "  - Translation difference too large" << std::endl;
        }
    }
    
    // Cleanup
    free(voxelData1);
    free(voxelData2);
    
    return 0;
}
