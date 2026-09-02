//
// Throwaway validation test for the hidden-component scan (C++ port of the
// rotation_curve_analysis.ipynb algorithm).
//
// Re-runs the last debug registration (dumped scans + parameters in
// plotting_results/2d/data) with useDirect=true and useHiddenComponentScan=true
// and prints the updated rotation candidates. The debug dumps (kernel1D.csv,
// hiddenComponentScan.csv, rotationPeaks.csv, rotationCorrelation1D.csv) can
// then be compared 1:1 against a fresh Python run of the notebook's algorithm
// on the same input dumps.
//
// Usage: test_hidden_component_scan [dataDir]
//   default dataDir = /home/tim-external/ros_ws/src/fsregistration/plotting_results/2d/data
//

#include "softRegistrationClass.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

static std::vector<double> readColumn(const std::string& path, int maxLines = -1) {
    std::vector<double> out;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "cannot open: " << path << std::endl;
        return out;
    }
    std::string line;
    int count = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double v;
        if (iss >> v) out.push_back(v);
        if (maxLines > 0 && ++count >= maxLines) break;
    }
    return out;
}

int main(int argc, char** argv) {
    std::string dataDir = "/home/tim-external/ros_ws/src/fsregistration/plotting_results/2d/data";
    bool useScan = true;
    bool multipleRadii = false, useClahe = true, useHamming = true;
    if (argc > 1) dataDir = argv[1];
    if (argc > 2) useScan = std::string(argv[2]) != "0";
    if (argc > 3) multipleRadii = std::string(argv[3]) != "0";
    if (argc > 4) useClahe = std::string(argv[4]) != "0";
    if (argc > 5) useHamming = std::string(argv[5]) != "0";
    std::cout << "=== hidden component scan validation ===" << std::endl;
    std::cout << "data dir: " << dataDir << "  useScan=" << (useScan ? 1 : 0)
              << "  multipleRadii=" << (multipleRadii ? 1 : 0)
              << " useClahe=" << (useClahe ? 1 : 0) << " useHamming=" << (useHamming ? 1 : 0) << std::endl;

    // parameters from the previous debug run (first row with 6 numeric values)
    std::vector<double> cfg;
    {
        std::ifstream f(dataDir + "/dataForReadIn.csv");
        std::string line;
        while (std::getline(f, line) && cfg.empty()) {
            std::istringstream iss(line);
            std::vector<double> row;
            double v;
            while (iss >> v) row.push_back(v);
            if (row.size() == 6) cfg = row;
        }
    }
    if (cfg.size() < 4) { std::cerr << "missing dataForReadIn.csv" << std::endl; return 1; }
    int N = (int)cfg[0];
    double cellSize = cfg[2];
    double potentialNecessaryForPeak = cfg[3];
    std::cout << "N=" << N << " cellSize=" << cellSize
              << " potentialNecessaryForPeak=" << potentialNecessaryForPeak << std::endl;

    // input scans dumped by the debug run. The dump writes line k = V[(k/N) + N*(k%N)]
    // (i.e. the transposed layout), so un-transpose on read to make repeated runs stable.
    std::vector<double> scan1Raw = readColumn(dataDir + "/voxelDataFFTW1.csv");
    std::vector<double> scan2Raw = readColumn(dataDir + "/voxelDataFFTW2.csv");
    if ((int)scan1Raw.size() != N * N || (int)scan2Raw.size() != N * N) {
        std::cerr << "bad scan dumps: " << scan1Raw.size() << " / " << scan2Raw.size() << std::endl;
        return 1;
    }
    std::vector<double> scan1(N * N), scan2(N * N);
    for (int k = 0; k < N * N; k++) {
        int idx = (k / N) + N * (k % N);  // transpose permutation
        scan1[idx] = scan1Raw[k];
        scan2[idx] = scan2Raw[k];
    }

    softRegistrationClass registrar(N, N / 2, N / 2, N / 2 - 1);

    std::cout << "\nRunning registration with useDirect=true, numAngles=4096,"
              << " useHiddenComponentScan=" << (useScan ? 1 : 0) << " ..." << std::endl;
    std::cout.flush();
    auto t0 = std::chrono::steady_clock::now();
    std::vector<transformationPeakfs2D> solutions = registrar.registrationOfTwoVoxelsSOFFTAllSoluations(
        scan1.data(), scan2.data(),
        cellSize,
        /*useGauss=*/false,
        /*debug=*/true,
        potentialNecessaryForPeak,
        /*multipleRadii=*/multipleRadii,
        /*useClahe=*/useClahe,
        /*useHamming=*/useHamming,
        /*useDirect=*/true,
        /*benchmark=*/false,
        /*timings=*/nullptr,
        /*level_potential_rotation=*/0.001,
        /*normalization=*/1,
        /*usePhaseCorrelation=*/false,
        /*numAngles=*/4096,
        /*r_min=*/0.0, /*r_max=*/0.0,
        /*useHiddenComponentScan=*/useScan);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "done in " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;

    std::cout << "\nUpdated rotation candidates (" << solutions.size() << "):" << std::endl;
    std::cout << "  angle(rad)  angle(deg)  peakCorrelation  levelPotential" << std::endl;
    for (const auto& sol : solutions) {
        const rotationPeakfs2D& r = sol.potentialRotation;
        std::cout << "  " << r.angle << "  " << (r.angle * 180.0 / M_PI)
                  << "  " << r.peakCorrelation << "  " << r.levelPotential << std::endl;
    }
    return 0;
}
