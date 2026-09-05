"""
transforms.py  --  YOUR CODE GOES HERE.

The shared transform core used by BOTH tasks. Write it once; bigmul.py
(Task A) and image_conv.py (Task B) import it.

Nothing in this file may call numpy.fft, scipy.fft, numpy.convolve,
scipy.signal, or any other library routine that performs a Fourier
transform, a convolution or a correlation for you. NumPy is for array
arithmetic only.

A quick self-test you should run before touching either application:

    import numpy as np
    from transforms import DFTAnalyzer, FFTTransformer
    x = np.random.randn(64) + 1j * np.random.randn(64)
    d, f = DFTAnalyzer(), FFTTransformer()
    assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9
    assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9
"""

import numpy as np


def next_power_of_two(n):
    """
    Return the smallest power of two that is >= ``n`` (and at least 1).

    Both tasks need this to choose a transform length for the radix-2 FFT.
    """
    # TODO: implement this function
    p = 1
    while p < n:   
        p *= 2
    return p


class DFTAnalyzer:
    """
    The Discrete Fourier Transform, computed straight from its definition.

        Analysis:   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
        Synthesis:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)

    How you write it is up to you -- a literal double loop, a precomputed
    table of twiddle factors indexed by (k*n) % N, or a NumPy expression --
    as long as it computes these sums directly and is not secretly an FFT.
    """

    name = "dft"

    def transform(self, x):
        """
        Forward DFT.

        Parameters
        ----------
        x : 1D array_like, length N (real or complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
        """
        # TODO: implement this method
        N = len(x)

        X = np.zeros(N, dtype= np.complex128)
        n = np.arange(N)

        for k in range(N):
            w_kn_N = np.exp(-2j * np.pi * k * n/N)

            to_sum = x * w_kn_N

            X[k] = np.sum(to_sum)

        return X 

    def inverse(self, spectrum):
        """
        Inverse DFT, including the 1/N factor.

        Parameters
        ----------
        spectrum : 1D array_like, length N (complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
            Do NOT discard the imaginary part here -- the caller decides when
            it is safe to take .real.
        """
        # TODO: implement this method
        N = len(spectrum)

        x = np.zeros(N, dtype= np.complex128)
        k = np.arange(N)

        for n in range(N):
            w_kn_N = np.exp(2j * np.pi * k * n/N)

            to_sum = spectrum * w_kn_N

            x[n] = np.sum(to_sum)/N

        return x 


class FFTTransformer(DFTAnalyzer):
    """
    Radix-2 decimation-in-time (Cooley-Tukey) FFT, in O(N log N).

    It inherits from DFTAnalyzer so that both applications can treat the two
    interchangeably: they call ``engine.transform(...)`` and
    ``engine.inverse(...)`` without caring which engine they hold.

    Requirements:
      * Recursive or iterative (with bit-reversal permutation) -- your choice.
      * N must be a power of two; raise ValueError for any other length.
        The caller is responsible for zero-padding up to next_power_of_two.
      * The inverse must reuse the same butterfly machinery (conjugated
        twiddles, or conjugate-transform-conjugate), not a second copy of it.
      * Twiddle factors for a stage are computed once per stage, never once
        per butterfly.
    """

    name = "fft"

    def transform(self, x):
        """Forward FFT. Same contract as DFTAnalyzer.transform."""
        # TODO: implement this method
        N = len(x)

        if N & (N-1) !=0 or N == 0:
            raise ValueError("N must be a power of 2")

        no_bits = int(np.log2(N))
        result = np.zeros(N, dtype=np.complex128)

        i = np.arange(N)
        reverse_index = np.zeros(N, dtype=np.int32)
        temp = i.copy()
        
        for bit in range(no_bits):
            reverse_index = (reverse_index << 1) | (temp & 1)
            temp >>= 1
            
        result = x[reverse_index].astype(np.complex128)

        stages = int(np.log2(N))

        for s in range(1,stages+1):
            M = 2 ** s
            half_M = M // 2

            k_array = np.arange(half_M)
            w_array = np.exp(-2j*np.pi*k_array/M)

            result_reshaped = result.reshape(-1, M)
            
            g = result_reshaped[:, :half_M].copy()
            h = w_array * result_reshaped[:, half_M:]
            
            result_reshaped[:, :half_M] = g + h
            result_reshaped[:, half_M:] = g - h
        return result 


    def inverse(self, spectrum):
        """Inverse FFT, including the 1/N factor."""
        # TODO: implement this method
        N = len(spectrum)
        spec_conjugate = np.conj(spectrum)
        res = np.conj( self.transform(spec_conjugate))/N
        return res

# ---------------------------------------------------------------------------
# BONUS (optional) -- arbitrary-length FFT.
#
# Delete this class if you are not attempting the bonus. If you do attempt it,
# run both tasks with --engine arbitrary and leave those output directories in
# your submission as the evidence.
# ---------------------------------------------------------------------------
class ArbitraryLengthFFT(FFTTransformer):
    """
    Bonus: an O(N log N) transform for ANY length N, not just powers of two.

    Bluestein's chirp-z algorithm is the usual route: rewrite the DFT as a
    convolution of two chirp sequences, and evaluate that convolution with a
    radix-2 FFT of length >= 2N-1. A mixed-radix Cooley-Tukey that factorises
    N is equally acceptable.

    With this engine, Task A no longer has to pad the digit arrays up to a
    power of two, and Task B no longer has to pad the image up to one.
    """

    name = "arbitrary"

    def transform(self, x):
        # TODO (bonus): implement this method
        raise NotImplementedError("Bonus: implement ArbitraryLengthFFT.transform")

    def inverse(self, spectrum):
        # TODO (bonus): implement this method
        raise NotImplementedError("Bonus: implement ArbitraryLengthFFT.inverse")

