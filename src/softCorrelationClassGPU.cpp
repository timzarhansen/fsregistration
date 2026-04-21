//
// Created by opencode - GPU-accelerated SO(3) correlation using s2fft
//

#include "softCorrelationClassGPU.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cstring>

namespace py = pybind11;

// Static member definitions
std::atomic<bool> softCorrelationClassGPU::s_pythonInitialized(false);
std::mutex softCorrelationClassGPU::s_initMutex;

softCorrelationClassGPU::softCorrelationClassGPU(int N, int bwOut, int bwIn, int degLim)
    : N(N), bwOut(bwOut), bwIn(bwIn), degLim(degLim) {
    // Python will be initialized on first use via ensurePythonInitialized()
}

softCorrelationClassGPU::~softCorrelationClassGPU() {
    // Don't finalize interpreter - other instances may still be using it
    // The interpreter will be cleaned up when the program exits
}

int softCorrelationClassGPU::ensurePythonInitialized() {
    // Check if already initialized (lock-free)
    if (s_pythonInitialized.load()) {
        return ERROR_NONE;
    }

    // Acquire lock for initialization
    std::lock_guard<std::mutex> lock(s_initMutex);

    // Double-check after acquiring lock
    if (s_pythonInitialized.load()) {
        return ERROR_NONE;
    }

    try {
        // Initialize Python interpreter
        py::initialize_interpreter();

        // Get PYTHONPATH from environment and add to sys.path
        py::module_ sys = py::module_::import("sys");
        const char* pythonPath = std::getenv("PYTHONPATH");
        if (pythonPath != nullptr) {
            // Split PYTHONPATH by ':' and add each directory
            std::string pathStr(pythonPath);
            size_t start = 0;
            size_t end = pathStr.find(':');
            while (end != std::string::npos) {
                std::string dir = pathStr.substr(start, end - start);
                if (!dir.empty()) {
                    sys.attr("path").attr("insert")(0, dir);
                }
                start = end + 1;
                end = pathStr.find(':', start);
            }
            // Add the last directory
            if (start < pathStr.length()) {
                sys.attr("path").attr("insert")(0, pathStr.substr(start));
            }
        }

        // Import and configure JAX for 64-bit precision
        py::module_ jax = py::module_::import("jax");
        jax.attr("config").attr("update")("jax_enable_x64", true);

        // Import s2fft to verify it's available
        py::module_::import("s2fft");

        // Mark as initialized
        s_pythonInitialized.store(true);
        return ERROR_NONE;

    } catch (const py::error_already_set& e) {
        std::cerr << "[softCorrelationClassGPU] Python initialization failed: " << e.what() << std::endl;
        return ERROR_PYTHON_INIT;
    } catch (const std::exception& e) {
        std::cerr << "[softCorrelationClassGPU] Exception during Python init: " << e.what() << std::endl;
        return ERROR_PYTHON_INIT;
    }
}

int softCorrelationClassGPU::correlationOfTwoSignalsInSO3(
    double resampledMagnitude1[],
    double resampledMagnitude2[],
    fftw_complex so3SigReturn[]) {

    // Ensure Python is initialized
    int initError = ensurePythonInitialized();
    if (initError != ERROR_NONE) {
        return initError;
    }

    try {
        // Acquire GIL for Python calls
        py::gil_scoped_acquire acquire;

        // Import the backend module
        py::module_ backend = py::module_::import("softCorrelation_gpu_backend");

        // Create a correlator instance with our parameters
        py::object correlator = backend.attr("create_correlator")(N, bwOut, bwIn, degLim);

        // Prepare input arrays as numpy arrays
        py::array_t<double> arr1({N, N});
        py::array_t<double> arr2({N, N});

        // Copy input data to numpy arrays
        std::memcpy(arr1.mutable_data(), resampledMagnitude1, N * N * sizeof(double));
        std::memcpy(arr2.mutable_data(), resampledMagnitude2, N * N * sizeof(double));

        // Call the correlate method
        py::tuple result = correlator.attr("correlate")(arr1, arr2).cast<py::tuple>();

        // Extract the result tuple: (success: bool, output: ndarray, error_msg: str)
        bool success = result[0].cast<bool>();
        if (!success) {
            std::string errorMsg = result[2].cast<std::string>();
            std::cerr << "[softCorrelationClassGPU] Python computation error: " << errorMsg << std::endl;
            return ERROR_COMPUTATION;
        }

        // Get the output array
        py::object output_obj = result[1];
        py::array_t<double> output = output_obj.cast<py::array_t<double>>();
        py::buffer_info buf = output.request();

        // Verify output size
        // s2fft with DH sampling produces: (2*N_azim-1) * (2*bwOut) * (2*bwOut-1) complex values
        // where N_azim = degLim + 1
        // Python returns interleaved [real, imag] as 1D array of doubles
        size_t n_azim = degLim + 1;
        size_t s2fft_output_size = (2 * n_azim - 1) * (2 * bwOut) * (2 * bwOut - 1);
        size_t expectedNumDoubles = s2fft_output_size * 2;  // 2 doubles per complex
        
        // buf.size is the number of elements (not bytes) in pybind11
        if (buf.size != expectedNumDoubles) {
            std::cerr << "[softCorrelationClassGPU] Output size mismatch: expected " 
                      << expectedNumDoubles << " doubles (" << s2fft_output_size << " complex), "
                      << "got " << buf.size << " doubles" << std::endl;
            return ERROR_OUTPUT_MISMATCH;
        }

        // Copy output to so3SigReturn (fftw_complex is typically double[2])
        // buf.size is number of elements, buf.itemsize is bytes per element
        std::memcpy(so3SigReturn, buf.ptr, buf.size * buf.itemsize);

        return ERROR_NONE;

    } catch (const py::error_already_set& e) {
        std::cerr << "[softCorrelationClassGPU] Python error during correlation: " << e.what() << std::endl;
        return ERROR_COMPUTATION;
    } catch (const std::exception& e) {
        std::cerr << "[softCorrelationClassGPU] Exception during correlation: " << e.what() << std::endl;
        return ERROR_COMPUTATION;
    }
}

bool softCorrelationClassGPU::isInitialized() const {
    return s_pythonInitialized.load();
}
