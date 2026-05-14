//
// Test full 3D registration: Speed comparison of peak detection strategies
//

#define ENABLE_DEBUG 0

#include "softRegistrationClass3D.h"
#include "generalHelpfulTools.h"
#include <chrono>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/norms.h>

// ============================================================
//  CONFIGURATION — change these to quickly adjust the test
// ============================================================

// --- Grid / voxel parameters ---
#define CONFIG_N                        64
#define CONFIG_CELL_SIZE                1.0
#define CONFIG_VOXEL_SIZE_MULTIPLIER    1.4

// --- Bandwidth / spherical harmonic parameters ---
#define CONFIG_BW_OUT                   (CONFIG_N / 2)
#define CONFIG_BW_IN                    (CONFIG_N / 2)
#define CONFIG_DEG_LIM                  (CONFIG_BW_OUT - 1)

// --- Ground truth transformation (degrees / voxels) ---
#define CONFIG_GT_ROLL_DEG              10.0
#define CONFIG_GT_PITCH_DEG             -10.0
#define CONFIG_GT_YAW_DEG               10.0
#define CONFIG_GT_TRANS_X               8.0
#define CONFIG_GT_TRANS_Y               3.0
#define CONFIG_GT_TRANS_Z               -11.0

// --- Peak detection thresholds ---
#define CONFIG_LEVEL_POTENTIAL_ROTATION 0.01
#define CONFIG_LEVEL_POTENTIAL_TRANS    0.1

// --- Accuracy evaluation tolerances ---
#define CONFIG_ROT_TOLERANCE_DEG        5.0
#define CONFIG_TRANS_TOLERANCE_VOXELS   5.0

// --- Data path ---
#define CONFIG_PLY_PATH                 "/home/tim-external/volumeROS/src/fsregistration/exampleData/dragon_recon/dragon_vrip.ply"

// ============================================================

struct TestConfig {
    std::string name;
    bool useSimpleRotationPeak;
    bool useSimpleTranslationPeak;
};

struct TestResult {
    std::string name;
    double totalTimeMs;
    int numRotationPeaks;
    int numTranslationPeaks;
    Eigen::Matrix4d bestTransformation;
};

#if ENABLE_DEBUG
struct DebugConfig {
    bool printAllSolutions;
    bool printRotationPeaks;
    bool printTranslationPeaks;
    bool enableDebugFlag;
};

void debugPrintSolutions(const std::vector<transformationPeakfs3D>& solutions, const DebugConfig& dbg) {
    if (!dbg.printAllSolutions && !dbg.printRotationPeaks && !dbg.printTranslationPeaks) return;

    std::cout << "\n--- DEBUG: Solution Details ---" << std::endl;

    for (size_t i = 0; i < solutions.size(); i++) {
        const auto& sol = solutions[i];

        if (dbg.printRotationPeaks) {
            Eigen::Quaterniond quat(sol.potentialRotation.w, sol.potentialRotation.x,
                                    sol.potentialRotation.y, sol.potentialRotation.z);
            Eigen::Vector3d rpy = generalHelpfulTools::getRollPitchYaw(quat);
            std::cout << "  [Rot " << i << "] q=(" << sol.potentialRotation.w << ", "
                      << sol.potentialRotation.x << ", " << sol.potentialRotation.y << ", "
                      << sol.potentialRotation.z << ") | corr=" << sol.potentialRotation.correlationHeight
                      << " | persist=" << sol.potentialRotation.persistence
                      << " | RPY=(" << rpy.x() * 180.0 / M_PI << ", "
                      << rpy.y() * 180.0 / M_PI << ", " << rpy.z() * 180.0 / M_PI << "°)" << std::endl;
        }

        if (dbg.printTranslationPeaks) {
            for (size_t j = 0; j < sol.potentialTranslations.size(); j++) {
                const auto& t = sol.potentialTranslations[j];
                std::cout << "    [Trans " << i << "." << j << "] (" << t.xTranslation << ", "
                          << t.yTranslation << ", " << t.zTranslation << ") | corr=" << t.correlationHeight
                          << " | persist=" << t.persistence << " | level=" << t.levelPotential << std::endl;
            }
        }

        if (dbg.printAllSolutions) {
            Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
            Eigen::Quaterniond quat(sol.potentialRotation.w, sol.potentialRotation.x,
                                    sol.potentialRotation.y, sol.potentialRotation.z);
            quat.normalize();
            mat.block<3, 3>(0, 0) = quat.toRotationMatrix();
            if (!sol.potentialTranslations.empty()) {
                mat(0, 3) = sol.potentialTranslations[0].xTranslation;
                mat(1, 3) = sol.potentialTranslations[0].yTranslation;
                mat(2, 3) = sol.potentialTranslations[0].zTranslation;
            }
            Eigen::Vector3d rpy = generalHelpfulTools::getRollPitchYaw(quat);
            std::cout << "  [Sol " << i << "] RPY=(" << rpy.x() * 180.0 / M_PI << ", "
                      << rpy.y() * 180.0 / M_PI << ", " << rpy.z() * 180.0 / M_PI
                      << "°) | Trans=(" << mat(0, 3) << ", " << mat(1, 3) << ", " << mat(2, 3)
                      << ") | RotCorr=" << sol.potentialRotation.correlationHeight << std::endl;
        }
    }
    std::cout << "--- END DEBUG ---" << std::endl;
}
#endif

void printTransformation3D(const std::string& label, const Eigen::Matrix4d& transformation, double timeMs) {
    std::cout << "\n" << label << std::endl;

    Eigen::Quaterniond quat(transformation.block<3, 3>(0, 0));
    Eigen::Vector3d rpy = generalHelpfulTools::getRollPitchYaw(quat);

    std::cout << "  Roll:  " << rpy.x() * 180.0 / M_PI << "°" << std::endl;
    std::cout << "  Pitch: " << rpy.y() * 180.0 / M_PI << "°" << std::endl;
    std::cout << "  Yaw:   " << rpy.z() * 180.0 / M_PI << "°" << std::endl;
    std::cout << "  Translation: (" << transformation(0, 3) << ", " << transformation(1, 3)
              << ", " << transformation(2, 3) << ") voxels" << std::endl;
    std::cout << "  Total time: " << timeMs << " ms" << std::endl;
}

Eigen::Matrix4d buildMatrixFromPeak(const transformationPeakfs3D& solution) {
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond quat(solution.potentialRotation.w, solution.potentialRotation.x,
                            solution.potentialRotation.y, solution.potentialRotation.z);
    quat.normalize();
    result.block<3, 3>(0, 0) = quat.toRotationMatrix();
    result(0, 3) = solution.potentialTranslations[0].xTranslation;
    result(1, 3) = solution.potentialTranslations[0].yTranslation;
    result(2, 3) = solution.potentialTranslations[0].zTranslation;
    return result;
}

#if ENABLE_DEBUG
TestResult runTest(softRegistrationClass3D& registrar,
                   double* voxelData1, double* voxelData2,
                   const TestConfig& config, double cellSize,
                   const DebugConfig& debugConfig) {
#else
TestResult runTest(softRegistrationClass3D& registrar,
                   double* voxelData1, double* voxelData2,
                   const TestConfig& config, double cellSize) {
#endif
    std::cout << "\n\n================================================================" << std::endl;
    std::cout << "Test: " << config.name << std::endl;
    std::cout << "  Rotation peak: " << (config.useSimpleRotationPeak ? "SimpleMax" : "Persistence")
              << " | Translation peak: " << (config.useSimpleTranslationPeak ? "SimpleMax" : "Persistence")
              << std::endl;
    std::cout << "================================================================" << std::endl;

    auto start = std::chrono::steady_clock::now();

    BenchmarkTimings3D timings;
#if ENABLE_DEBUG
    std::vector<transformationPeakfs3D> solutions = registrar.sofftRegistrationVoxel3DListOfPossibleTransformations(
        voxelData1, voxelData2, debugConfig.enableDebugFlag, true, true, cellSize,
        0.0, 0.0, CONFIG_LEVEL_POTENTIAL_ROTATION, CONFIG_LEVEL_POTENTIAL_TRANS, false, 1,
        config.useSimpleRotationPeak, config.useSimpleTranslationPeak, &timings);
#else
    std::vector<transformationPeakfs3D> solutions = registrar.sofftRegistrationVoxel3DListOfPossibleTransformations(
        voxelData1, voxelData2, false, true, true, cellSize,
        0.0, 0.0, CONFIG_LEVEL_POTENTIAL_ROTATION, CONFIG_LEVEL_POTENTIAL_TRANS, false, 1,
        config.useSimpleRotationPeak, config.useSimpleTranslationPeak, &timings);
#endif

    auto end = std::chrono::steady_clock::now();
    double totalTimeMs = std::chrono::duration<double, std::milli>(end - start).count();

    int numRotationPeaks = timings.numSolutions;
    int numTranslationPeaks = timings.totalTransPeaks;

    // Print per-solution translation times
    for (size_t i = 0; i < timings.transPerSolutionTimes.size(); i++) {
        std::cout << "  Translation solution [" << i << "]: " << std::fixed << std::setprecision(3)
                  << timings.transPerSolutionTimes[i] << " ms" << std::endl;
    }

    // Print benchmark summary
    int numSol = timings.numSolutions;
    double perSol = (numSol > 0) ? 1.0 / numSol : 0;
    std::cout << "  --- 3D All Solutions Summary ---" << std::endl;
    std::cout << "    Rotation peaks found:           " << numSol << std::endl;
    std::cout << "    Translation peaks found:        " << timings.totalTransPeaks << std::endl;
    std::cout << "    3D Spectrum (FFT):              " << std::fixed << std::setprecision(3)
              << timings.spectrumTime << " ms" << std::endl;
    std::cout << "    SOFT descriptor projection:     " << std::fixed << std::setprecision(4)
              << timings.softDescriptorTime << " ms" << std::endl;
    std::cout << "    SOFT correlation (FFT):         " << std::fixed << std::setprecision(5)
              << timings.rotationCorrelationTime << " ms" << std::endl;
    std::cout << "    Quaternion preparation:         " << std::fixed << std::setprecision(2)
              << timings.overheadTime << " ms" << std::endl;
    std::cout << "    Rotation peak detection:        " << std::fixed << std::setprecision(2)
              << timings.peakDetectionTime << " ms" << std::endl;
    std::cout << "    Solution printing:              " << std::fixed << std::setprecision(6)
              << timings.plottingTime << " ms" << std::endl;
    std::cout << "    --- Translation breakdown (" << numSol << " solutions) ---" << std::endl;
    if (numSol > 0) {
        std::cout << "      Voxel rotation:             " << std::fixed << std::setprecision(3)
                  << timings.transVoxelRotationTime << " ms (" << std::setprecision(4)
                  << (timings.transVoxelRotationTime * perSol) << " ms/sol)" << std::endl;
        std::cout << "      FFT1:                       " << std::fixed << std::setprecision(3)
                  << timings.transFft1Time << " ms (" << std::setprecision(3)
                  << (timings.transFft1Time * perSol) << " ms/sol)" << std::endl;
        std::cout << "      FFT2:                       " << std::fixed << std::setprecision(3)
                  << timings.transFft2Time << " ms (" << std::setprecision(3)
                  << (timings.transFft2Time * perSol) << " ms/sol)" << std::endl;
        std::cout << "      Complex correlation:        " << std::fixed << std::setprecision(3)
                  << timings.transCorrelationTime << " ms (" << std::setprecision(3)
                  << (timings.transCorrelationTime * perSol) << " ms/sol)" << std::endl;
        std::cout << "      IFFT:                       " << std::fixed << std::setprecision(3)
                  << timings.transIfftTime << " ms (" << std::setprecision(4)
                  << (timings.transIfftTime * perSol) << " ms/sol)" << std::endl;
        std::cout << "      fftshift + magnitude:       " << std::fixed << std::setprecision(3)
                  << timings.transFftshiftTime << " ms (" << std::setprecision(4)
                  << (timings.transFftshiftTime * perSol) << " ms/sol)" << std::endl;
        std::cout << "      Peak detection:             " << std::fixed << std::setprecision(1)
                  << timings.transPeakDetectionTime << " ms (" << std::setprecision(4)
                  << (timings.transPeakDetectionTime * perSol) << " ms/sol)" << std::endl;
    }
    std::cout << "    Total translation:            " << std::fixed << std::setprecision(1)
              << timings.totalAllTransTime << " ms" << std::endl;
    std::cout << "    Total time:                   " << std::fixed << std::setprecision(1)
              << timings.totalTime << " ms" << std::endl;

    // Find best solution by rotation correlation
    transformationPeakfs3D bestSolution;
    double bestCorrelation = -1;
    for (const auto& sol : solutions) {
        if (sol.potentialRotation.correlationHeight > bestCorrelation) {
            bestCorrelation = sol.potentialRotation.correlationHeight;
            bestSolution = sol;
        }
    }

    Eigen::Matrix4d bestMatrix = buildMatrixFromPeak(bestSolution);

    std::cout << "\n  Rotation peaks found:    " << numRotationPeaks << std::endl;
    std::cout << "  Translation peaks found: " << numTranslationPeaks << std::endl;
    std::cout << "  Wall time:               " << std::fixed << std::setprecision(1) << totalTimeMs << " ms" << std::endl;

    printTransformation3D("  Best solution (by rotation correlation):", bestMatrix, totalTimeMs);

#if ENABLE_DEBUG
    debugPrintSolutions(solutions, debugConfig);
#endif

    TestResult result;
    result.name = config.name;
    result.totalTimeMs = totalTimeMs;
    result.numRotationPeaks = numRotationPeaks;
    result.numTranslationPeaks = numTranslationPeaks;
    result.bestTransformation = bestMatrix;
    return result;
}

void printSummaryTable(const std::vector<TestResult>& results) {
    std::cout << "\n\n================================================================" << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << "================================================================" << std::endl;

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
}

int main(int argc, char** argv) {
    std::cout << "=== Full 3D Registration Speed Comparison ===" << std::endl;

#if ENABLE_DEBUG
    DebugConfig debugConfig = {true, true, true, true};
    std::cout << "[DEBUG MODE ENABLED]" << std::endl;
#endif

    std::string plyPath = CONFIG_PLY_PATH;
    std::cout << "Loading point cloud: " << plyPath << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PLYReader reader;
    if (reader.read(plyPath, *cloud) == -1) {
        std::cerr << "Error: Could not load PLY file!" << std::endl;
        return 1;
    }
    std::cout << "  Points loaded: " << cloud->points.size() << std::endl;

    double maximumDistance = 0;
    for (const auto& pt : cloud->points) {
        Eigen::Vector3d tmp(pt.x, pt.y, pt.z);
        double norm = tmp.norm();
        if (norm > maximumDistance) {
            maximumDistance = norm;
        }
    }

    int N = CONFIG_N;
    double sizeVoxelOneDirection = 2 * maximumDistance * CONFIG_VOXEL_SIZE_MULTIPLIER;
    double cellSize = CONFIG_CELL_SIZE;

    std::cout << "Parameters: N=" << N << ", maxDistance=" << maximumDistance
              << ", gridSide=" << sizeVoxelOneDirection << std::endl;

    double* voxelData1 = (double*)malloc(sizeof(double) * N * N * N);
    double* voxelData2 = (double*)malloc(sizeof(double) * N * N * N);
    for (int i = 0; i < N * N * N; i++) {
        voxelData1[i] = 0;
        voxelData2[i] = 0;
    }

    // Ground truth transformation: RPY(CONFIG_GT_ROLL_DEG°, CONFIG_GT_PITCH_DEG°, CONFIG_GT_YAW_DEG°) + translation(CONFIG_GT_TRANS_X, CONFIG_GT_TRANS_Y, CONFIG_GT_TRANS_Z) voxels
    double gtRoll = CONFIG_GT_ROLL_DEG / 180.0 * M_PI;
    double gtPitch = CONFIG_GT_PITCH_DEG / 180.0 * M_PI;
    double gtYaw = CONFIG_GT_YAW_DEG / 180.0 * M_PI;
    double gtTransX = CONFIG_GT_TRANS_X;
    double gtTransY = CONFIG_GT_TRANS_Y;
    double gtTransZ = CONFIG_GT_TRANS_Z;

    Eigen::Matrix4d gtMatrix = generalHelpfulTools::getTransformationMatrixFromRPY(gtRoll, gtPitch, gtYaw);
    gtMatrix(0, 3) = gtTransX;
    gtMatrix(1, 3) = gtTransY;
    gtMatrix(2, 3) = gtTransZ;

    Eigen::Quaterniond gtQuat(gtMatrix.block<3, 3>(0, 0));
    Eigen::Vector3d gtRpy = generalHelpfulTools::getRollPitchYaw(gtQuat);

    std::cout << "\nGround truth transformation:" << std::endl;
    std::cout << "  Roll:  " << gtRpy.x() * 180.0 / M_PI << "°" << std::endl;
    std::cout << "  Pitch: " << gtRpy.y() * 180.0 / M_PI << "°" << std::endl;
    std::cout << "  Yaw:   " << gtRpy.z() * 180.0 / M_PI << "°" << std::endl;
    std::cout << "  Translation: (" << gtTransX << ", " << gtTransY << ", " << gtTransZ << ") voxels" << std::endl;

    // Fill voxelData1 (original)
    for (const auto& pt : cloud->points) {
        Eigen::Vector4d currentVector(pt.x, pt.y, pt.z, 1);
        currentVector = generalHelpfulTools::getTransformationMatrixFromRPY(0, 0, 0) * currentVector;
        int xIndex = N / 2 + currentVector.x() * N / sizeVoxelOneDirection;
        int yIndex = N / 2 + currentVector.y() * N / sizeVoxelOneDirection;
        int zIndex = N / 2 + currentVector.z() * N / sizeVoxelOneDirection;
        voxelData1[generalHelpfulTools::index3D(xIndex, yIndex, zIndex, N)] = 1;
    }

    // Fill voxelData2 (transformed)
    for (const auto& pt : cloud->points) {
        Eigen::Vector4d currentVector(pt.x, pt.y, pt.z, 1);
        Eigen::Vector4d shiftVector(gtTransX, gtTransY, gtTransZ, 0);
        currentVector += shiftVector * sizeVoxelOneDirection / N;
        currentVector = generalHelpfulTools::getTransformationMatrixFromRPY(gtRoll, gtPitch, gtYaw) * currentVector;
        int xIndex = N / 2 + currentVector.x() * N / sizeVoxelOneDirection;
        int yIndex = N / 2 + currentVector.y() * N / sizeVoxelOneDirection;
        int zIndex = N / 2 + currentVector.z() * N / sizeVoxelOneDirection;
        voxelData2[generalHelpfulTools::index3D(xIndex, yIndex, zIndex, N)] = 1;
    }

    int bwOut = CONFIG_BW_OUT;
    int bwIn = CONFIG_BW_IN;
    int degLim = CONFIG_DEG_LIM;

    std::cout << "\nCreating softRegistrationClass3D..." << std::endl;
    softRegistrationClass3D registrar(N, bwOut, bwIn, degLim);

    // Define test configurations
    std::vector<TestConfig> testConfigs = {
        {"All Solutions (Persistence + Persistence)",  false, false},
        {"Multi Rot, 1 Trans (Persistence + SimpleMax)", false, true},
        {"Single Solution (SimpleMax + SimpleMax)",     true, true},
    };

    // Run tests
    std::vector<TestResult> results;
    for (const auto& config : testConfigs) {
#if ENABLE_DEBUG
        TestResult result = runTest(registrar, voxelData1, voxelData2, config, cellSize, debugConfig);
#else
        TestResult result = runTest(registrar, voxelData1, voxelData2, config, cellSize);
#endif
        results.push_back(result);
    }

    // Print summary table
    printSummaryTable(results);

    // Evaluate accuracy against ground truth
    std::cout << "\n\n================================================================" << std::endl;
    std::cout << "Accuracy vs Ground Truth" << std::endl;
    std::cout << "================================================================" << std::endl;

    double rotTolerance = CONFIG_ROT_TOLERANCE_DEG;
    double transTolerance = CONFIG_TRANS_TOLERANCE_VOXELS;

    for (const auto& r : results) {
        double rotDiff = generalHelpfulTools::angleDifferenceQuaternion(gtQuat,
                                Eigen::Quaterniond(r.bestTransformation.block<3, 3>(0, 0))) * 180.0 / M_PI;

        Eigen::Vector3d gtTrans(gtTransX, gtTransY, gtTransZ);
        Eigen::Vector3d estTrans(r.bestTransformation(0, 3), r.bestTransformation(1, 3), r.bestTransformation(2, 3));
        double transDiff = (gtTrans - estTrans).norm();

        std::cout << "\n  " << r.name << ":" << std::endl;
        std::cout << "    Rotation error:     " << std::fixed << std::setprecision(2) << rotDiff << "°" << std::endl;
        std::cout << "    Translation error:  " << std::fixed << std::setprecision(2) << transDiff << " voxels" << std::endl;

        if (rotDiff < rotTolerance && transDiff < transTolerance) {
            std::cout << "    PASSED (rot < " << rotTolerance << "°, trans < " << transTolerance << " voxels)" << std::endl;
        } else {
            std::cout << "    FAILED" << std::endl;
            if (rotDiff >= rotTolerance)
                std::cout << "      - Rotation error too large: " << rotDiff << "°" << std::endl;
            if (transDiff >= transTolerance)
                std::cout << "      - Translation error too large: " << transDiff << " voxels" << std::endl;
        }
    }

    // Compare methods against each other
    if (results.size() >= 2) {
        std::cout << "\n\n================================================================" << std::endl;
        std::cout << "Method Comparison" << std::endl;
        std::cout << "================================================================" << std::endl;

        for (size_t i = 1; i < results.size(); i++) {
            double rotDiff = generalHelpfulTools::angleDifferenceQuaternion(
                Eigen::Quaterniond(results[0].bestTransformation.block<3, 3>(0, 0)),
                Eigen::Quaterniond(results[i].bestTransformation.block<3, 3>(0, 0))) * 180.0 / M_PI;
            Eigen::Vector3d trans0(results[0].bestTransformation(0, 3), results[0].bestTransformation(1, 3), results[0].bestTransformation(2, 3));
            Eigen::Vector3d transI(results[i].bestTransformation(0, 3), results[i].bestTransformation(1, 3), results[i].bestTransformation(2, 3));
            double transDiff = (trans0 - transI).norm();

            std::cout << "\n  " << results[0].name << " vs " << results[i].name << ":" << std::endl;
            std::cout << "    Rotation difference:  " << std::fixed << std::setprecision(2) << rotDiff << "°" << std::endl;
            std::cout << "    Translation diff:   " << std::fixed << std::setprecision(2) << transDiff << " voxels" << std::endl;
        }
    }

    free(voxelData1);
    free(voxelData2);

    return 0;
}
