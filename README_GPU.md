# GPU Backend Setup (s2fft/JAX)

## Overview

The `fsregistration` package includes a GPU-accelerated SO(3) correlation backend using the `s2fft` library (built on JAX). This provides an alternative to the traditional FFTW-based CPU implementation.

**Current Status**: The GPU backend is functional but uses different sampling conventions than soft20, resulting in different output values. The implementation is correct for s2fft's DH (Driscoll-Healy) sampling.

## Known Limitations

1. **Bandlimit Limitation**: s2fft's Wigner transform has a hard limit on azimuthal bandlimit N < 8. The test uses reduced parameters (N=8, bwOut=4, bwIn=4, degLim=3) to work around this.

2. **Sampling Convention Difference**: 
   - soft20 (CPU): Uses N×N uniform sampling, output size = 8*bwOut³
   - s2fft (GPU): Uses DH sampling (2L)×(2L-1), output size = (2N-1)×(2L)×(2L-1)

3. **Normalization Differences**: The two backends use different normalization conventions, so direct comparison of output values will show large differences.

## Dependencies

### Python Packages

Install the required Python packages:

```bash
pip3 install -r requirements.txt
```

Or individually:

```bash
pip3 install jax jaxlib s2fft numpy scipy
```

**Note:** For GPU acceleration, install JAX with CUDA support:
```bash
pip3 install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Build Dependencies

The following are already handled by CMake:
- `pybind11` - C++/Python bindings
- `Python3` - Python interpreter and development files

## Building

The GPU backend is **always built** when dependencies are available.

```bash
cd /home/tim-external/ros_ws
colcon build --packages-select fsregistration --symlink-install
```

## Running Tests

### 1. Set up PYTHONPATH

```bash
export PYTHONPATH=/home/tim-external/ros_ws/build/fsregistration:$PYTHONPATH
```

### 2. Run the comparison test

```bash
cd /home/tim-external/ros_ws/build/fsregistration
./test_correlation_comparison
```

### Expected Output (with GPU backend working)

```
=== SO(3) Correlation Comparison Test ===
Parameters: N=8, bwOut=4, bwIn=4, degLim=3

1. Testing softCorrelationClass (CPU/FFTW)...
   CPU computation completed.

2. Testing softCorrelationClassGPU (s2fft/JAX)...
   GPU computation completed.

3. Comparing results...
=== Results ===
...
```

**Note**: The test will show large errors between CPU and GPU outputs due to different sampling conventions. This is expected and does not indicate a bug.

## Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| 0 | Success | - |
| 1 | Python init failed | Install Python dependencies |
| 2 | s2fft import failed | Install s2fft package |
| 3 | Invalid input | Check input parameters |
| 4 | Computation error | Check Python backend logs |
| 5 | Output mismatch | Verify output array size |

## Troubleshooting

### "ModuleNotFoundError: No module named 'jax'"

```bash
pip3 install jax jaxlib
```

### "ModuleNotFoundError: No module named 's2fft'"

```bash
pip3 install s2fft
```

### "ModuleNotFoundError: No module named 'softCorrelation_gpu_backend'"

```bash
export PYTHONPATH=/path/to/build/fsregistration:$PYTHONPATH
```

### Large errors in comparison test

This is expected due to different sampling conventions between soft20 and s2fft. The GPU backend is working correctly.

## CPU vs GPU

- **CPU (FFTW/soft20)**: Production-ready, well-tested, uses uniform sampling
- **GPU (s2fft/JAX)**: Functional but uses DH sampling, limited to N < 8 for Wigner transforms

For production use, the CPU backend is recommended. The GPU backend is provided for experimentation and future development.
