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

// Forward declaration
extern double angleDifference(double angle1, double angle2);

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
    
    // Rotation detection
    double goodGuessAlpha = std::atan2(initialGuess(1, 0), initialGuess(0, 0));
    double angleCovarianceOld;
    
    auto startRotationOld = std::chrono::steady_clock::now();
    double rotationAngleOld = registrar.sofftRegistrationVoxel2DRotationOnly(
        voxelData1, voxelData2, goodGuessAlpha, angleCovarianceOld, false);
    auto endRotationOld = std::chrono::steady_clock::now();
    double rotationTimeOld = std::chrono::duration<double, std::milli>(endRotationOld - startRotationOld).count();
    
    std::cout << "Rotation detection time: " << rotationTimeOld << " ms" << std::endl;
    std::cout << "Detected rotation: " << rotationAngleOld << " rad (" 
              << rotationAngleOld * 180.0 / M_PI << "°)" << std::endl;
    
    // Translation detection (after rotating image 1)
    auto startTranslationOld = std::chrono::steady_clock::now();
    
    // Copy and rotate image 1
    double* voxelData1Rotated = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++) {
        voxelData1Rotated[i] = voxelData1[i];
    }
    
    cv::Mat magTMP1(N, N, CV_64F, voxelData1Rotated);
    cv::Mat magTMP2(N, N, CV_64F, voxelData2);
    
    if (useGauss) {
        for (int i = 0; i < 2; i++) {
            cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);
            cv::GaussianBlur(magTMP2, magTMP2, cv::Size(9, 9), 0);
        }
    }
    
    cv::Point2f pc(magTMP1.cols / 2., magTMP1.rows / 2.);
    cv::Mat r = cv::getRotationMatrix2D(pc, rotationAngleOld * 180.0 / M_PI, 1.0);
    cv::warpAffine(magTMP1, magTMP1, r, magTMP1.size());
    
    std::vector<translationPeakfs2D> translationsOld = registrar.sofftRegistrationVoxel2DTranslationAllPossibleSolutions(
        voxelData1Rotated, voxelData2, cellSize, 1.0, false, 0, potentialNecessaryForPeak);
    
    auto endTranslationOld = std::chrono::steady_clock::now();
    double translationTimeOld = std::chrono::duration<double, std::milli>(endTranslationOld - startTranslationOld).count();
    
    std::cout << "Translation detection time: " << translationTimeOld << " ms" << std::endl;
    
    // Find best translation (closest to initial guess)
    translationPeakfs2D bestTranslationOld;
    double bestDistance = 100000;
    for (auto& trans : translationsOld) {
        double diffX = trans.translationSI.x() - initialGuess(0, 3);
        double diffY = trans.translationSI.y() - initialGuess(1, 3);
        double distance = std::sqrt(diffX * diffX + diffY * diffY);
        if (distance < bestDistance) {
            bestTranslationOld = trans;
            bestDistance = distance;
        }
    }
    
    auto endTotalOld = std::chrono::steady_clock::now();
    double totalTimeOld = std::chrono::duration<double, std::milli>(endTotalOld - startTotalOld).count();
    
    // Build transformation matrix
    Eigen::Matrix4d resultOld = Eigen::Matrix4d::Identity();
    Eigen::AngleAxisd rotation_vectorOld(rotationAngleOld, Eigen::Vector3d(0, 0, 1));
    Eigen::Matrix3d rotMatrixOld = rotation_vectorOld.toRotationMatrix();
    resultOld.block<3, 3>(0, 0) = rotMatrixOld;
    resultOld(0, 3) = bestTranslationOld.translationSI.x();
    resultOld(1, 3) = bestTranslationOld.translationSI.y();
    
    printTransformationMatrix("Result:", resultOld, totalTimeOld);
    
    free(voxelData1Rotated);
    
    // ========================================
    // Method 2: NEW (1-Angle Direct)
    // ========================================
    std::cout << "\n\n--- NEW Method (1-Angle Direct) ---" << std::endl;
    
    auto startTotalNew = std::chrono::steady_clock::now();
    
    // Rotation detection
    double angleCovarianceNew;
    
    auto startRotationNew = std::chrono::steady_clock::now();
    double rotationAngleNew = registrar.sofftRegistrationVoxel2DRotationOnlyWithMethod(
        voxelData1, voxelData2, goodGuessAlpha, angleCovarianceNew, true, false);
    auto endRotationNew = std::chrono::steady_clock::now();
    double rotationTimeNew = std::chrono::duration<double, std::milli>(endRotationNew - startRotationNew).count();
    
    std::cout << "Rotation detection time: " << rotationTimeNew << " ms" << std::endl;
    std::cout << "Detected rotation: " << rotationAngleNew << " rad (" 
              << rotationAngleNew * 180.0 / M_PI << "°)" << std::endl;
    
    // Translation detection (after rotating image 1)
    auto startTranslationNew = std::chrono::steady_clock::now();
    
    // Copy and rotate image 1
    voxelData1Rotated = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++) {
        voxelData1Rotated[i] = voxelData1[i];
    }
    
    magTMP1 = cv::Mat(N, N, CV_64F, voxelData1Rotated);
    magTMP2 = cv::Mat(N, N, CV_64F, voxelData2);
    
    if (useGauss) {
        for (int i = 0; i < 2; i++) {
            cv::GaussianBlur(magTMP1, magTMP1, cv::Size(9, 9), 0);
            cv::GaussianBlur(magTMP2, magTMP2, cv::Size(9, 9), 0);
        }
    }
    
    pc = cv::Point2f(magTMP1.cols / 2., magTMP1.rows / 2.);
    r = cv::getRotationMatrix2D(pc, rotationAngleNew * 180.0 / M_PI, 1.0);
    cv::warpAffine(magTMP1, magTMP1, r, magTMP1.size());
    
    std::vector<translationPeakfs2D> translationsNew = registrar.sofftRegistrationVoxel2DTranslationAllPossibleSolutions(
        voxelData1Rotated, voxelData2, cellSize, 1.0, false, 0, potentialNecessaryForPeak);
    
    auto endTranslationNew = std::chrono::steady_clock::now();
    double translationTimeNew = std::chrono::duration<double, std::milli>(endTranslationNew - startTranslationNew).count();
    
    std::cout << "Translation detection time: " << translationTimeNew << " ms" << std::endl;
    
    // Find best translation (closest to initial guess)
    translationPeakfs2D bestTranslationNew;
    bestDistance = 100000;
    for (auto& trans : translationsNew) {
        double diffX = trans.translationSI.x() - initialGuess(0, 3);
        double diffY = trans.translationSI.y() - initialGuess(1, 3);
        double distance = std::sqrt(diffX * diffX + diffY * diffY);
        if (distance < bestDistance) {
            bestTranslationNew = trans;
            bestDistance = distance;
        }
    }
    
    auto endTotalNew = std::chrono::steady_clock::now();
    double totalTimeNew = std::chrono::duration<double, std::milli>(endTotalNew - startTotalNew).count();
    
    // Build transformation matrix
    Eigen::Matrix4d resultNew = Eigen::Matrix4d::Identity();
    Eigen::AngleAxisd rotation_vectorNew(rotationAngleNew, Eigen::Vector3d(0, 0, 1));
    Eigen::Matrix3d rotMatrixNew = rotation_vectorNew.toRotationMatrix();
    resultNew.block<3, 3>(0, 0) = rotMatrixNew;
    resultNew(0, 3) = bestTranslationNew.translationSI.x();
    resultNew(1, 3) = bestTranslationNew.translationSI.y();
    
    printTransformationMatrix("Result:", resultNew, totalTimeNew);
    
    free(voxelData1Rotated);
    
    // ========================================
    // Comparison
    // ========================================
    std::cout << "\n\n=== Comparison ===" << std::endl;
    
    // Timing comparison
    std::cout << "\nTiming breakdown:" << std::endl;
    std::cout << "  Rotation detection: OLD=" << rotationTimeOld << " ms, NEW=" << rotationTimeNew << " ms, speedup=" 
              << (rotationTimeOld / rotationTimeNew) << "x" << std::endl;
    std::cout << "  Translation detection: OLD=" << translationTimeOld << " ms, NEW=" << translationTimeNew << " ms, speedup=" 
              << (translationTimeOld / translationTimeNew) << "x" << std::endl;
    std::cout << "  Total: OLD=" << totalTimeOld << " ms, NEW=" << totalTimeNew << " ms, speedup=" 
              << (totalTimeOld / totalTimeNew) << "x" << std::endl;
    
    // Result comparison
    std::cout << "\nResult comparison:" << std::endl;
    
    double rotationDiff = std::abs(angleDifference(rotationAngleOld, rotationAngleNew));
    double rotationDiffDeg = rotationDiff * 180.0 / M_PI;
    std::cout << "  Rotation difference: " << rotationDiff << " rad (" << rotationDiffDeg << "°)" << std::endl;
    
    double transDiffX = std::abs(bestTranslationOld.translationSI.x() - bestTranslationNew.translationSI.x());
    double transDiffY = std::abs(bestTranslationOld.translationSI.y() - bestTranslationNew.translationSI.y());
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
