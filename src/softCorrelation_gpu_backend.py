"""
GPU-accelerated SO(3) correlation backend using s2fft.

This module provides a Python backend for the softCorrelationClassGPU C++ class,
using the s2fft library (JAX-based) for GPU-accelerated spherical harmonic
and Wigner transforms.
"""

import jax
import jax.numpy as jnp
import numpy as np
import s2fft

# Enable 64-bit precision (critical for numerical accuracy matching soft20)
jax.config.update("jax_enable_x64", True)


class SO3CorrelationBackend:
    """
    SO(3) correlation backend using s2fft.
    
    This class replicates the functionality of softCorrelationClass from soft20,
    but uses s2fft (JAX-based) for GPU acceleration.
    """

    def __init__(self, N, bwOut, bwIn, degLim):
        """
        Initialize the correlation backend.
        
        Args:
            N: Grid size (should be 2 * bwIn for Driscoll-Healy sampling)
            bwOut: Output bandlimit for SO(3) transform
            bwIn: Input bandlimit for S^2 transform
            degLim: Degree limit for correlation (max degree to compute)
        """
        self.N = N
        self.bwOut = bwOut
        self.bwIn = bwIn
        self.degLim = degLim

    def correlate(self, signal1, signal2):
        """
        Compute SO(3) correlation of two signals on the sphere.
        
        This method replicates the correlationOfTwoSignalsInSO3 method from
        softCorrelationClass, using s2fft for the transforms.
        
        Args:
            signal1: np.ndarray of shape (N, N) - first signal on Driscoll-Healy grid
            signal2: np.ndarray of shape (N, N) - second signal on Driscoll-Healy grid
            
        Returns:
            tuple: (success: bool, output: ndarray or None, error_msg: str)
                   output shape: (8*bwOut^3, 2) with [real, imag] pairs
        """
        try:
            L = self.bwIn
            
            # Reshape inputs to Driscoll-Healy grid format
            # soft20 uses DH sampling: (2L) x (2L-1) grid
            signal1 = np.array(signal1, dtype=np.float64).reshape(2 * L, 2 * L - 1)
            signal2 = np.array(signal2, dtype=np.float64).reshape(2 * L, 2 * L - 1)

            # Step 1: Forward spherical harmonic transform for both signals
            # Using Driscoll-Healy sampling to match soft20
            flm_sig = s2fft.forward(signal1, L, sampling="dh", method="jax", reality=True)
            flm_pat = s2fft.forward(signal2, L, sampling="dh", method="jax", reality=True)

            # Step 2: Combine S^2 coefficients into SO(3) Wigner coefficients
            # This replicates so3CombineCoef_fftw from soft20
            flmn = self._combine_coefficients(flm_sig, flm_pat)

            # Step 3: Inverse SO(3) transform to get correlation
            N_azim = self.degLim + 1  # azimuthal bandlimit
            f_so3 = s2fft.wigner.inverse(
                flmn, self.bwOut, N_azim,
                sampling="dh", method="jax", reality=False
            )

            # Step 4: Reshape output to match soft20 format (8*bwOut^3 complex values)
            output = self._reshape_to_soft20_format(f_so3)

            return True, output, ""

        except Exception as e:
            return False, None, str(e)

    def _combine_coefficients(self, flm_sig, flm_pat):
        """
        Combine S^2 coefficients into SO(3) Wigner coefficients.
        
        This method replicates the logic from soft20's so3CombineCoef_fftw function.
        The correlation in SO(3) is computed by combining the spherical harmonic
        coefficients of two signals using Wigner-D matrix properties.
        
        Args:
            flm_sig: Spherical harmonic coefficients of signal, shape (L, 2L-1)
            flm_pat: Spherical harmonic coefficients of pattern, shape (L, 2L-1)
            
        Returns:
            flmn: Wigner coefficients, shape (2N-1, bwOut, 2*bwOut-1)
        """
        N = self.degLim + 1
        L = self.bwIn

        # Output shape: (2N-1, bwOut, 2*bwOut-1)
        # Indexing: flmn[n + N - 1, l, m + bwOut - 1]
        flmn = np.zeros((2 * N - 1, self.bwOut, 2 * self.bwOut - 1), dtype=np.complex128)

        for el in range(self.degLim + 1):
            # Wigner normalization factor from soft20
            wig_norm = 2.0 * np.pi * np.sqrt(2.0 / (2.0 * el + 1.0))

            for m1 in range(-el, el + 1):
                # Get signal coefficient at (el, -m1)
                # s2fft indexing: flm[el, L - 1 + m] for order m
                sig_coef = flm_sig[el, L - 1 + (-m1)]

                # Initialize fudge factor (alternating sign)
                fudge = -1 if ((m1 + el) % 2) else 1

                for m2 in range(-el, el + 1):
                    # Get pattern coefficient at (el, -m2) and conjugate it
                    pat_coef = np.conj(flm_pat[el, L - 1 + (-m2)])

                    # Combine with normalization and fudge factor
                    combined = wig_norm * sig_coef * pat_coef * fudge

                    # Store in flmn array
                    # s2fft Wigner indexing: flmn[n, l, m] where n is azimuthal order
                    flmn[N - 1 + m2, el, self.bwOut - 1 + m1] = combined

                    # Flip fudge factor for next iteration
                    fudge *= -1

        return flmn

    def _reshape_to_soft20_format(self, f_so3):
        """
        Reshape s2fft output to match soft20's flat array format.
        
        soft20 outputs a flat array of 8*bwOut^3 complex values.
        s2fft with DH sampling outputs shape (n_gamma, n_beta, n_alpha) where:
            - n_gamma = 2N - 1 (azimuthal samples)
            - n_beta = 2L (latitudinal samples for DH)
            - n_alpha = 2L - 1 (longitudinal samples for DH)
        
        We flatten and pad/truncate to match soft20's expected size.
        
        Args:
            f_so3: SO(3) correlation from s2fft, shape varies by sampling
            
        Returns:
            output: np.ndarray of shape (8*bwOut^3, 2) with [real, imag] pairs
        """
        target_size = 8 * self.bwOut * self.bwOut * self.bwOut

        # Create output array: (target_size, 2) for [real, imag]
        output = np.zeros((target_size, 2), dtype=np.float64)

        # Flatten s2fft output
        flat_data = f_so3.flatten()
        current_size = flat_data.size

        if current_size <= target_size:
            # Pad with zeros
            output[:current_size, 0] = np.real(flat_data)
            output[:current_size, 1] = np.imag(flat_data)
        else:
            # Truncate (shouldn't normally happen with proper parameters)
            output[:, 0] = np.real(flat_data[:target_size])
            output[:, 1] = np.imag(flat_data[:target_size])

        return output


def create_correlator(N, bwOut, bwIn, degLim):
    """
    Factory function to create a SO3CorrelationBackend instance.
    
    This function is called from C++ via pybind11.
    
    Args:
        N: Grid size
        bwOut: Output bandlimit
        bwIn: Input bandlimit
        degLim: Degree limit
        
    Returns:
        SO3CorrelationBackend instance
    """
    return SO3CorrelationBackend(N, bwOut, bwIn, degLim)
