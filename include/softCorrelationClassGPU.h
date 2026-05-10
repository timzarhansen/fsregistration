//
// Created by opencode - GPU-accelerated SO(3) correlation using s2fft
//

#ifndef FSREGISTRATION_SOFTCORRELATIONCLASSGPU_H
#define FSREGISTRATION_SOFTCORRELATIONCLASSGPU_H

#include "fftw3.h"
#include <mutex>
#include <atomic>

class softCorrelationClassGPU {
public:
    // Error codes
    static constexpr int ERROR_NONE = 0;
    static constexpr int ERROR_PYTHON_INIT = 1;
    static constexpr int ERROR_IMPORT_S2FFT = 2;
    static constexpr int ERROR_INVALID_INPUT = 3;
    static constexpr int ERROR_COMPUTATION = 4;
    static constexpr int ERROR_OUTPUT_MISMATCH = 5;

    // Constructor - identical signature to softCorrelationClass
    softCorrelationClassGPU(int N, int bwOut, int bwIn, int degLim);

    // Destructor
    ~softCorrelationClassGPU();

    // Main correlation method - returns error code
    int correlationOfTwoSignalsInSO3(
        double resampledMagnitude1[],
        double resampledMagnitude2[],
        fftw_complex so3SigReturn[]
    );

    // Check if Python backend is initialized
    bool isInitialized() const;

private:
    int N;        // Grid size (2 * bwIn)
    int bwOut;    // Output bandlimit
    int bwIn;     // Input bandlimit
    int degLim;   // Degree limit for correlation

    // Static singleton for Python interpreter (shared across all instances)
    static std::atomic<bool> s_pythonInitialized;
    static std::mutex s_initMutex;

    // Internal method to ensure Python is initialized
    int ensurePythonInitialized();
};

#endif // FSREGISTRATION_SOFTCORRELATIONCLASSGPU_H
