//
// Test script to compare softCorrelationClass vs softCorrelationClassGPU
//

#include "softCorrelationClass.h"
#include "softCorrelationClassGPU.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <random>

#define MAX_ERROR 1e-6

int main() {
    // Parameters (typical values)
    int N = 64;        // Grid size = 2 * bwIn
    int bwOut = 32;    // Output bandlimit
    int bwIn = 32;     // Input bandlimit (N = 2 * bwIn)
    int degLim = 31;   // Degree limit (bwOut - 1)

    std::cout << "=== SO(3) Correlation Comparison Test ===" << std::endl;
    std::cout << "Parameters: N=" << N << ", bwOut=" << bwOut 
              << ", bwIn=" << bwIn << ", degLim=" << degLim << std::endl;

    // Allocate input arrays
    double* signal1 = (double*)malloc(N * N * sizeof(double));
    double* signal2 = (double*)malloc(N * N * sizeof(double));

    // Allocate output arrays
    fftw_complex* output_cpu = (fftw_complex*)fftw_malloc(
        sizeof(fftw_complex) * 8 * bwOut * bwOut * bwOut);
    fftw_complex* output_gpu = (fftw_complex*)fftw_malloc(
        sizeof(fftw_complex) * 8 * bwOut * bwOut * bwOut);

    // Generate test signals (random data)
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < N * N; i++) {
        signal1[i] = dis(gen);
        signal2[i] = dis(gen);
    }

    std::cout << "\n1. Testing softCorrelationClass (CPU/FFTW)..." << std::endl;
    softCorrelationClass cpuCorrelator(N, bwOut, bwIn, degLim);
    cpuCorrelator.correlationOfTwoSignalsInSO3(signal1, signal2, output_cpu);
    std::cout << "   CPU computation completed." << std::endl;

    std::cout << "\n2. Testing softCorrelationClassGPU (s2fft/JAX)..." << std::endl;
    softCorrelationClassGPU gpuCorrelator(N, bwOut, bwIn, degLim);
    
    int gpuError = gpuCorrelator.correlationOfTwoSignalsInSO3(signal1, signal2, output_gpu);
    
    if (gpuError != softCorrelationClassGPU::ERROR_NONE) {
        std::cerr << "   GPU computation failed with error code: " << gpuError << std::endl;
        std::cerr << "   Error codes: 0=none, 1=python_init, 2=import_s2fft, "
                  << "3=invalid_input, 4=computation, 5=output_mismatch" << std::endl;
        free(signal1);
        free(signal2);
        fftw_free(output_cpu);
        fftw_free(output_gpu);
        return 1;
    }
    std::cout << "   GPU computation completed." << std::endl;

    // Compare results
    std::cout << "\n3. Comparing results..." << std::endl;
    
    size_t outputSize = 8 * bwOut * bwOut * bwOut;
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    size_t maxErrorIdx = 0;
    
    for (size_t i = 0; i < outputSize; i++) {
        double realCpu = output_cpu[i][0];
        double imagCpu = output_cpu[i][1];
        double realGpu = output_gpu[i][0];
        double imagGpu = output_gpu[i][1];
        
        double absError = std::sqrt(
            std::pow(realGpu - realCpu, 2) + std::pow(imagGpu - imagCpu, 2));
        
        double magnitude = std::sqrt(realCpu * realCpu + imagCpu * imagCpu);
        double relError = (magnitude > 1e-12) ? absError / magnitude : absError;
        
        if (absError > maxAbsError) {
            maxAbsError = absError;
            maxErrorIdx = i;
        }
        if (relError > maxRelError) {
            maxRelError = relError;
        }
    }

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Output size: " << outputSize << " complex values" << std::endl;
    std::cout << "Max absolute error: " << maxAbsError << std::endl;
    std::cout << "Max relative error: " << maxRelError << std::endl;
    std::cout << "Max error at index: " << maxErrorIdx << std::endl;

    // Sample values comparison
    std::cout << "\nSample values (first 5):" << std::endl;
    for (size_t i = 0; i < 5; i++) {
        std::cout << "  [" << i << "] CPU: (" << output_cpu[i][0] << ", " 
                  << output_cpu[i][1] << ")  GPU: (" << output_gpu[i][0] << ", " 
                  << output_gpu[i][1] << ")" << std::endl;
    }

    // Pass/fail
    if (maxRelError < MAX_ERROR) {
        std::cout << "\n✓ TEST PASSED (error < " << MAX_ERROR << ")" << std::endl;
        int result = 0;
        free(signal1);
        free(signal2);
        fftw_free(output_cpu);
        fftw_free(output_gpu);
        return result;
    } else {
        std::cout << "\n✗ TEST FAILED (error >= " << MAX_ERROR << ")" << std::endl;
        std::cout << "\nNote: This may be expected due to:" << std::endl;
        std::cout << "  - Different normalization conventions between soft20 and s2fft" << std::endl;
        std::cout << "  - Grid size mismatch (soft20 uses N×N, s2fft DH uses 2L×(2L-1))" << std::endl;
        std::cout << "  - Numerical precision differences" << std::endl;
        free(signal1);
        free(signal2);
        fftw_free(output_cpu);
        fftw_free(output_gpu);
        return 1;
    }
}
