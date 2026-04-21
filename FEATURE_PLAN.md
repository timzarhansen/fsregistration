# SO(3) Correlation High-Resolution Enhancement Plan

## Overview

This document outlines a phased approach to enable high-resolution SO(3) correlation for production use cases requiring:
- Harmonic bandlimit: bwOut > 128
- Azimuthal bandlimit: N > 64
- Angular accuracy sufficient for 3D rotation peak detection (Euler angles in zyz convention)

## Current State

### CPU Backend (soft20/FFTW)
- ✅ Production-ready and well-tested
- ✅ Supports required resolutions (tested up to bwOut=256)
- ❌ Single-threaded in critical paths
- ❌ Performance bottleneck: `Inverse_SO3_Naive_fftw` with O(bwOut²) independent 1D FFTs

### GPU Backend (s2fft/JAX)
- ✅ Functional implementation exists
- ❌ **Hard limitation**: Wigner transform N < 8 (cannot support production requirements)
- ❌ Different sampling convention (DH vs uniform)
- ❌ Not suitable for production use

## Requirements Analysis

### Target Parameters
| Parameter | Current Test | Production Target | Notes |
|-----------|--------------|-------------------|-------|
| bwOut | 4 | 128-256 | Harmonic bandlimit |
| N (azimuthal) | 8 | 64-128 | Azimuthal bandlimit |
| bwIn | 4 | 64-128 | Input bandlimit |
| degLim | 3 | 32-64 | Degree limit |

### Memory Requirements (estimated)
- bwOut=64: ~300 MB
- bwOut=128: ~2.4 GB
- bwOut=256: ~19 GB

### Computational Complexity
- `Inverse_SO3_Naive_fftw`: O(bwOut² × N × log(N)) with bwOut² independent 1D FFTs
- Each (m1, m2) coefficient pair is independent → **embarrassingly parallel**

## Implementation Options

### Option 1: Multi-Core CPU Parallelization (OpenMP)

**Estimated Effort**: 1-2 weeks

**Benefits**:
- Immediate performance improvement (15-25x on 32-core CPU)
- Minimal code changes (add OpenMP pragmas to existing loops)
- No new dependencies or hardware requirements
- Maintains soft20's proven numerical accuracy

**Implementation**:
```cpp
// In soft20/src/lib1/soft_fftw.c, function Inverse_SO3_Naive_fftw
#pragma omp parallel for collapse(2)
for (int m1 = -bwOut; m1 <= bwOut; m1++) {
    for (int m2 = -bwOut; m2 <= bwOut; m2++) {
        // Each (m1, m2) pair is independent
        // Perform 1D FFT for this coefficient pair
    }
}
```

**Tradeoffs**:
- Scales with core count (diminishing returns beyond 32-64 cores)
- Memory bandwidth bound at high parallelism
- Still limited by CPU cache hierarchy

---

### Option 2: CUDA Acceleration with cuFFT

**Estimated Effort**: 3-5 weeks

**Benefits**:
- Highest performance (10-30x speedup on A100 GPU)
- Scales to very high resolutions (bwOut > 256)
- Future-proof for GPU-based pipelines
- cuFFT highly optimized for 1D/2D/3D FFTs

**Implementation Strategy**:

#### Phase 2a: Abstraction Layer (1-2 weeks)
Create FFT backend abstraction to support multiple implementations:

```cpp
// include/fft_planner.h
class FFTPlanner {
public:
    virtual ~FFTPlanner() = default;
    virtual void plan1DFFT(void* data, int n, FFTDirection dir) = 0;
    virtual void execute1DFFT() = 0;
    virtual void planMany1DFFTs(void* data, int n, int count, FFTDirection dir) = 0;
    // ... other FFT types
};

class FFTWPlanner : public FFTPlanner { /* existing FFTW code */ };
class CUFFTPlanner : public FFTPlanner { /* new CUDA code */ };
```

#### Phase 2b: CUDA Backend (2-3 weeks)
- Implement `CUFFTPlanner` using cuFFT API
- Use `cufftPlanMany` for batched 1D FFTs (matches soft20's pattern)
- Handle GPU memory management (cudaMalloc, cudaMemcpy)
- Integrate with soft20's existing workflow

**Key CUDA Components**:
- `cufftPlanMany` for O(bwOut²) independent 1D FFTs
- `cudaMemcpyAsync` with streams for overlap
- Shared memory optimization for small transforms

**Tradeoffs**:
- Requires NVIDIA GPU (compute capability ≥ 7.0 recommended)
- Added complexity with GPU memory management
- CPU-GPU data transfer overhead for small problems
- Break-even point: bwOut ≥ 64 for meaningful speedup

---

### Option 3: Hybrid Approach (Recommended)

**Estimated Effort**: 4-6 weeks total

**Strategy**: Implement both OpenMP and CUDA, with runtime selection:

```cpp
enum class FFTBackend {
    AUTO,      // Auto-select based on problem size and hardware
    OPENMP,    // Multi-core CPU
    CUDA       // GPU acceleration
};

// Runtime selection logic
FFTBackend selectBackend(int bwOut, int N) {
    if (cudaAvailable && bwOut >= 64) return CUDA;
    if (omp_get_max_threads() > 1) return OPENMP;
    return OPENMP; // default to single-threaded
}
```

**Benefits**:
- Best of both worlds: CPU for small problems, GPU for large
- No forced hardware requirements
- Gradual migration path (OpenMP first, then add CUDA)
- Production-ready at each phase

---

## Recommended Phased Implementation

### Phase 1: OpenMP Parallelization (Weeks 1-2)

**Goals**:
- Add OpenMP pragmas to `Inverse_SO3_Naive_fftw`
- Benchmark performance on multi-core CPU
- Validate numerical accuracy unchanged

**Files to modify**:
- `src/soft20/src/lib1/soft_fftw.c` (main parallelization)
- `src/soft20/CMakeLists.txt` (add OpenMP flags)
- Optional: `src/soft20/src/lib1/so3_correlate_fftw.c` (additional loops)

**Success Criteria**:
- 15-25x speedup on 32-core CPU for bwOut=128
- Identical numerical output to single-threaded version
- Memory usage within 2x of single-threaded

---

### Phase 2: CUDA Abstraction Layer (Weeks 3-4)

**Goals**:
- Design FFTPlanner interface
- Wrap existing FFTW code as FFTWPlanner
- Implement CUFFTPlanner for core operations
- Add backend selection logic

**New Files**:
- `include/fft_planner.h` (abstraction interface)
- `src/fftw_planner.cpp` (FFTW backend)
- `src/cufft_planner.cu` (CUDA backend)
- `src/backend_selector.cpp` (runtime selection)

**Success Criteria**:
- FFTWPlanner produces identical results to current implementation
- CUFFTPlanner compiles and runs on test GPU
- Backend selection works correctly

---

### Phase 3: CUDA Integration & Optimization (Weeks 5-6)

**Goals**:
- Replace soft20 FFT calls with FFTPlanner interface
- Optimize GPU memory transfers (pinned memory, streams)
- Benchmark and tune for target parameters
- Add comprehensive tests

**Success Criteria**:
- 10-30x speedup vs single-threaded CPU for bwOut ≥ 128
- Numerical accuracy within 1e-6 relative error
- Memory efficient (≤ 20 GB for bwOut=256)

---

## Alternative: External GPU Library Integration

If in-house CUDA development is not feasible, consider:

### Option A: TensorFlow/PyTorch FFT
- Use existing GPU FFT implementations
- Python-based workflow (similar to current s2fft backend)
- **Limitation**: Still constrained by Wigner transform N < 8 issue

### Option B: cuSPARSE + Custom Wigner
- Use cuSPARSE for sparse matrix operations
- Implement Wigner-d recursion on GPU
- **Effort**: 4-6 weeks, high complexity

### Option C: Third-party SO(3) GPU Library
- Research existing GPU-accelerated SO(3) libraries
- **Status**: No widely-adopted solution found

---

## Decision Points

### 1. Immediate Needs vs Long-term Goals

**Question**: Do you need production-ready performance in < 2 weeks?

- **Yes**: Implement OpenMP parallelization only (Phase 1)
- **No**: Proceed with full hybrid approach (Phases 1-3)

### 2. Hardware Availability

**Question**: Do you have access to NVIDIA GPUs (compute capability ≥ 7.0)?

- **Yes**: CUDA acceleration is viable
- **No**: Focus on OpenMP only

### 3. Team Expertise

**Question**: Does your team have CUDA development experience?

- **Yes**: Proceed with CUDA implementation
- **No**: Consider OpenMP first, then evaluate CUDA needs

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OpenMP scaling issues | Low | Medium | Profile and tune parallel regions |
| CUDA memory limits | Medium | High | Implement memory pooling, streaming |
| Numerical accuracy regression | Low | Critical | Comprehensive test suite |
| cuFFT API changes | Low | Low | Abstract cuFFT calls behind interface |
| Team bandwidth | Medium | High | Phased delivery, OpenMP first |

---

## Success Metrics

### Performance
- bwOut=64, N=32: < 1 second (currently ~15 seconds)
- bwOut=128, N=64: < 5 seconds (currently ~2 minutes)
- bwOut=256, N=128: < 30 seconds (currently ~15 minutes)

### Accuracy
- Relative error < 1e-6 vs single-precision reference
- Peak location error < 0.1° for Euler angles

### Usability
- No API changes for existing code
- Backend selection automatic or single-enum parameter
- Build system handles CUDA dependency gracefully

---

## Next Steps

1. **Decision**: Which implementation path to pursue?
   - [ ] OpenMP only (1-2 weeks)
   - [ ] Hybrid OpenMP + CUDA (4-6 weeks)
   - [ ] CUDA only (3-5 weeks)
   - [ ] Re-evaluate requirements

2. **Resource Allocation**: 
   - Developer time estimate: 160-240 hours for full implementation
   - Hardware: NVIDIA GPU (A100 recommended for testing)

3. **Timeline**:
   - Phase 1 (OpenMP): 2 weeks
   - Phase 2-3 (CUDA): 4 weeks additional
   - Total: 6 weeks for complete hybrid solution

---

## Appendix: Technical Details

### soft20 FFT Call Patterns

```c
// Inverse_SO3_Naive_fftw: O(bwOut²) independent 1D FFTs
for (m1 = -bwOut to bwOut) {
    for (m2 = -bwOut to bwOut) {
        // 1D FFT of length N (azimuthal dimension)
        fftw_plan_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
    }
}
```

### cuFFT Equivalent

```cuda
// cufftPlanMany for batched 1D FFTs
cufftPlanMany(&plan, 1, &n, 
              d_in, n, n, 1, 1,  // input stride
              d_out, n, n, 1, 1, // output stride
              CUFFT_C2C, batch); // batch = bwOut²
```

### Memory Layout Considerations

- soft20 uses interleaved (m1, m2) coefficient storage
- CUDA prefers contiguous memory for coalesced access
- May require data reorganization for optimal GPU performance

---

## References

- soft20 documentation: https://github.com/Zarbokk/soft20
- cuFFT documentation: https://docs.nvidia.com/cuda/cufft/
- OpenMP specification: https://www.openmp.org/specifications/
- SO(3) harmonic analysis: Wigner, E. P. (1959). Group Theory
