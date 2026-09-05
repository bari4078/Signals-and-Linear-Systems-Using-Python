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


def find_1d_shift(a, b, engine):
    corr = engine.inverse(engine.transform(a) * np.conj(engine.transform(b))).real
    return int(np.argmax(corr))

def realign_images(original, shifted, row_idx, col_idx):
    engine = FFTTransformer()

    h_shift = find_1d_shift(original[row_idx, :], shifted[row_idx, :], engine)
    v_shift = find_1d_shift(original[:, col_idx], shifted[:, col_idx], engine)

    realigned = np.roll(shifted, shift=(-v_shift, -h_shift), axis=(0, 1))
    return realigned

def weighted_polynomial_multiply(P, Q, W):
    engine = FFTTransformer()

    Pw = [p * w for p, w in zip(P, W)]        # weight each coefficient of P

    lin_len = len(Pw) + len(Q) - 1             # true linear-convolution length
    N = next_power_of_two(lin_len)             # FFT needs a power-of-two length

    Pw_pad = np.pad(Pw, (0, N - len(Pw)))
    Q_pad  = np.pad(Q,  (0, N - len(Q)))

    spectrum = engine.transform(Pw_pad) * engine.transform(Q_pad)
    conv = engine.inverse(spectrum).real
    conv = np.round(conv).astype(int)[:lin_len]   # crop off the padding junk

    return conv[::-1].tolist()
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def linear_convolve(a, b):
    """Always-correct linear convolution using FFT, regardless of input length."""
    engine = FFTTransformer()
    result_len = len(a) + len(b) - 1
    N = next_power_of_two(result_len)
    Ap = np.pad(np.asarray(a, dtype=float), (0, N - len(a)))
    Bp = np.pad(np.asarray(b, dtype=float), (0, N - len(b)))
    raw = engine.inverse(engine.transform(Ap) * engine.transform(Bp)).real
    return raw[:result_len]




def remove_periodic_noise(img, exclude_radius=3, num_peaks=4):
    engine = FFTTransformer()
    rows_t = np.array([engine.transform(row) for row in img])
    F = np.array([engine.transform(col) for col in rows_t.T]).T

    mag = np.abs(F).copy()
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    mag[cy - exclude_radius:cy + exclude_radius, cx - exclude_radius:cx + exclude_radius] = 0  # protect DC region

    flat_idx = np.argsort(mag.ravel())[::-1][:num_peaks]
    peak_coords = np.unravel_index(flat_idx, mag.shape)
    F[peak_coords] = 0   # notch out the interference frequencies

    rows_i = np.array([engine.inverse(row) for row in F])
    cleaned = np.array([engine.inverse(col) for col in rows_i.T]).T.real
    return cleaned


def find_pattern(signal, template, threshold_ratio=0.9):
    engine = FFTTransformer()
    N = next_power_of_two(len(signal) + len(template))
    S = np.pad(signal, (0, N - len(signal)))
    T = np.pad(template, (0, N - len(template)))

    corr = engine.inverse(engine.transform(S) * np.conj(engine.transform(T))).real
    best = corr.max()
    matches = np.where(corr >= threshold_ratio * best)[0]
    return matches.tolist()





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

    '''
    arbitrary er jonno bluesteins algorithm korbo, first-> dft k convolution e convert
    using algebra, tarpor convolve erpor 
    '''

    def transform(self, x):
        # TODO (bonus): implement this method
        N = len(x)

        if N == 0 or (N & (N-1) == 0):
            return super().transform(x)

        M = next_power_of_two(2*N-1)

        n = np.arange(N)

        chirp = np.exp(1j*np.pi*(n**2)/N)

        A = np.zeros(M, dtype=np.complex128)
        #for loop diye a b ready na kore ebhabe korle faster
        A[:N] = x * np.conj(chirp)


        B = np.zeros(M, dtype=np.complex128)

        B[:N] = chirp 
        B[M-N+1:] = chirp[1:][::-1]

        A_fft = self.transform(A)
        B_fft = self.transform(B)

        C_fft = A_fft*B_fft

        C = super().inverse(C_fft)

        result = C[:N] *np.conj(chirp)

        return result

    def inverse(self, spectrum):
        # TODO (bonus): implement this method
        return super().inverse(spectrum)
