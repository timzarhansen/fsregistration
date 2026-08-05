//
// Test: FS2D direct 1-angle method with configurable number of angles (numAngles).
//
// Question under test: the 1D rotation correlation C(theta) = sum_m P_m exp(-im theta)
// is a trigonometric polynomial of degree B-1 (B = N/2). The image size N fixes the
// number of coefficients, but the evaluation grid (numAngles) is independent and can
// be finer than N at negligible cost. This test verifies:
//   1. consistency: finer grids agree with the N-grid at shared angles (pure interpolation)
//   2. peak refinement: does the detected peak angle change / sharpen with numAngles?
//   3. cost: timing per numAngles
//
// Usage: test_more_angles [img1] [img2] [N]
//   default: exampleData/voxelScan1.jpg voxelScan2.jpg N=256
//

#include "softRegistrationClass.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>

// returns index of the global maximum
static int argmax(const std::vector<float>& v) {
    return (int)(std::max_element(v.begin(), v.end()) - v.begin());
}

int main(int argc, char** argv) {
    std::string img1Path = "/home/tim-external/ros_ws/src/fsregistration/exampleData/voxelScan1.jpg";
    std::string img2Path = "/home/tim-external/ros_ws/src/fsregistration/exampleData/voxelScan2.jpg";
    int N = 256;

    if (argc > 2) { img1Path = argv[1]; img2Path = argv[2]; }
    if (argc > 3) { N = std::atoi(argv[3]); }

    std::cout << "=== FS2D Direct Method: numAngles test ===" << std::endl;
    std::cout << "Image 1: " << img1Path << std::endl;
    std::cout << "Image 2: " << img2Path << std::endl;
    std::cout << "N       : " << N << std::endl;

    cv::Mat img1 = cv::imread(img1Path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(img2Path, cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: could not load images!" << std::endl;
        return 1;
    }

    int bwOut = N / 2, bwIn = N / 2, degLim = bwOut - 1;
    std::cout << "bwOut=" << bwOut << ", bwIn=" << bwIn << ", degLim=" << degLim << std::endl;

    double* voxelData1 = (double*)malloc(N * N * sizeof(double));
    double* voxelData2 = (double*)malloc(N * N * sizeof(double));
    cv::Mat img1Resized, img2Resized;
    cv::resize(img1, img1Resized, cv::Size(N, N));
    cv::resize(img2, img2Resized, cv::Size(N, N));
    for (int i = 0; i < N * N; i++) {
        voxelData1[i] = (double)img1Resized.ptr<uchar>(i / N)[i % N];
        voxelData2[i] = (double)img2Resized.ptr<uchar>(i / N)[i % N];
    }

    softRegistrationClass registrar(N, bwOut, bwIn, degLim);

    // ---- reference: full SO(3) method peak ----
    std::cout << "\n--- Reference: full SO(3) method ---" << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    auto [corrSO3, anglesSO3] = registrar.compute1AngleCorrelationArraySO3(
        voxelData1, voxelData2, false, true, true, false);
    auto t1 = std::chrono::steady_clock::now();
    int pSO3 = argmax(corrSO3);
    std::cout << "  time      : " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;
    std::cout << "  peak angle: " << anglesSO3[pSO3] * 180.0 / M_PI << " deg  (corr=" << corrSO3[pSO3] << ")"
              << "  [" << corrSO3.size() << " angles]" << std::endl;

    // ---- direct method at increasing numAngles ----
    std::cout << "\n--- Direct method vs numAngles ---" << std::endl;
    std::cout << "  numAngles | time(ms) | peak angle (deg) | peak corr | max |diff| vs N-grid | peak delta vs N-grid (deg)" << std::endl;

    std::vector<float> corrRef;   // N-grid curve, kept for the interpolation check
    std::vector<float> angleRef;
    double refPeakDeg = 0.0;

    const int multipliers[] = {1, 2, 4, 8, 16};
    const int numMultipliers = sizeof(multipliers) / sizeof(multipliers[0]);

    std::vector<float> corrFinest, anglesFinest;
    int finestNumAngles = 0;
    for (int mi = 0; mi < numMultipliers; mi++) {
        int mult = multipliers[mi];
        int numAngles = N * mult;
        auto s0 = std::chrono::steady_clock::now();
        auto [corr, angles] = registrar.compute1AngleCorrelationArrayDirect(
            voxelData1, voxelData2, false, true, true, false, numAngles);
        auto s1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(s1 - s0).count();

        int pk = argmax(corr);
        double peakDeg = angles[pk] * 180.0 / M_PI;

        if (mult == 1) {
            corrRef = corr;
            angleRef = angles;
            refPeakDeg = peakDeg;
        }

        // interpolation check: at shared angles (k * numAngles/N) the finer grid must
        // agree with the N-grid (normalization shifts slightly, so use a small tol)
        double maxDiff = 0.0;
        for (int k = 0; k < N; k++) {
            int idx = k * mult;
            double d = std::abs((double)corr[idx] - (double)corrRef[k]);
            if (d > maxDiff) maxDiff = d;
        }

        double peakDelta = peakDeg - refPeakDeg;
        // wrap to [-180, 180]
        while (peakDelta > 180.0) peakDelta -= 360.0;
        while (peakDelta < -180.0) peakDelta += 360.0;

        printf("  %8d | %7.2f | %16.4f | %8.4f | %16.2e | %+12.4f\n",
               numAngles, ms, peakDeg, corr[pk], maxDiff, peakDelta);

        if (mi == numMultipliers - 1) {
            corrFinest = corr;
            anglesFinest = angles;
            finestNumAngles = numAngles;
        }
    }

    // peak-neighborhood check at the finest grid: is the peak sharp or flat?
    int pkFine = argmax(corrFinest);
    double binDeg = 360.0 / finestNumAngles;
    std::cout << "\n--- Peak neighborhood at finest grid (" << finestNumAngles << " angles, "
              << binDeg << " deg/bin) ---" << std::endl;
    for (int off = -4; off <= 4; off++) {
        int idx = (pkFine + off + finestNumAngles) % finestNumAngles;
        double deg = anglesFinest[idx] * 180.0 / M_PI;
        if (deg > 180.0) deg -= 360.0;
        std::cout << "  " << (off == 0 ? ">" : " ") << " " << deg << " deg : " << corrFinest[idx] << std::endl;
    }

    std::cout << "\n--- Summary ---" << std::endl;
    std::cout << "  peak delta = how much the finer grid shifts the detected peak vs the N-grid." << std::endl;
    std::cout << "  If delta shrinks to ~0 and max|diff| ~ 1e-4, the finer grid is pure interpolation:" << std::endl;
    std::cout << "  the N-grid quantization (2pi/N = " << 360.0 / N << " deg) was the only limit on peak localization." << std::endl;

    free(voxelData1);
    free(voxelData2);
    return 0;
}
