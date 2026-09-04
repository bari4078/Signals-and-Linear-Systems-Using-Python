# CSE220 – 25 Practice Problems (Very Easy → Extremely Hard)

These are **original practice problems**, built from the patterns seen across your past online tests (polynomial/integer multiplication, convolution, cross-correlation, deconvolution, 2D image processing). They are not guaranteed to appear — they're here to stretch every direction the real test could come from.

All problems assume the same toolkit from the first guide:

```python
from transforms import FFTTransformer, DFTAnalyzer, next_power_of_two
engine = FFTTransformer()
X = engine.transform(x)      # time domain -> frequency domain
x = engine.inverse(X)        # frequency domain -> time domain (complex; take .real when appropriate)
```

Problems are grouped into 5 tiers of 5. Each entry has a short problem statement, the key idea, and a working code snippet.

---

## Tier 1 — Warm-Up (Very Easy)

### 1. Manual DFT Sanity Check
**Problem:** For `x = [1, 2, 3, 4]`, compute the DFT and confirm that `X[0]` always equals the sum of all samples (the "DC" / average-energy term).

**Key idea:** `X[0] = Σ x[n] * e^0 = Σ x[n]`. No exponential needed for k=0 — it's just a sum.

```python
import numpy as np
from transforms import DFTAnalyzer

x = np.array([1, 2, 3, 4], dtype=complex)
X = DFTAnalyzer().transform(x)
print(X)
print("X[0] == sum(x)?", np.isclose(X[0], x.sum()))
```

---

### 2. Parseval's Energy Check
**Problem:** For a random signal, confirm that total energy is preserved between domains: `sum(|x|^2) == (1/N) * sum(|X|^2)`.

**Key idea:** The DFT doesn't create or destroy energy, it just redistributes it into frequency bins — useful as a correctness check for your own FFT implementation.

```python
import numpy as np
from transforms import FFTTransformer

x = np.random.randn(8) + 1j * np.random.randn(8)
X = FFTTransformer().transform(x)

lhs = np.sum(np.abs(x) ** 2)
rhs = np.sum(np.abs(X) ** 2) / len(x)
print(np.isclose(lhs, rhs))   # True
```

---

### 3. Conjugate Symmetry of Real Signals
**Problem:** For a *real* input signal, show that `X[N-k]` is the complex conjugate of `X[k]`.

**Key idea:** Real signals have "mirrored" spectra — this is why real-signal FFT libraries only bother returning half the frequencies.

```python
import numpy as np
from transforms import FFTTransformer

x = np.array([4, 1, 7, 2, 9, 3, 6, 5], dtype=float)
X = FFTTransformer().transform(x)
N = len(x)

k = 3
print(np.allclose(X[N - k], np.conj(X[k])))   # True
```

---

### 4. Tiny Linear Convolution via FFT
**Problem:** Convolve `[1, 2, 3]` and `[0, 1, 0.5]` using FFT and confirm it matches direct convolution.

**Key idea:** Pad to `len(a)+len(b)-1` (then to a power of two), multiply spectra, inverse-transform, crop.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def fft_convolve(a, b):
    engine = FFTTransformer()
    lin_len = len(a) + len(b) - 1
    N = next_power_of_two(lin_len)
    A = np.pad(a, (0, N - len(a)))
    B = np.pad(b, (0, N - len(b)))
    conv = engine.inverse(engine.transform(A) * engine.transform(B)).real
    return np.round(conv[:lin_len], 6)

print(fft_convolve([1, 2, 3], [0, 1, 0.5]))
print(np.convolve([1, 2, 3], [0, 1, 0.5]))   # should match
```

---

### 5. Padding Demo — Circular Wraparound vs. Linear Result
**Problem:** Convolve two length-4 arrays **without** padding (pure circular convolution at N=4) and **with** proper padding, and show the outputs differ.

**Key idea:** Skipping padding lets the "tail" of the result wrap around and add into the "head" — this is exactly the aliasing bug the assignment warns about.

```python
import numpy as np
from transforms import FFTTransformer

a = np.array([1, 2, 3, 4], dtype=float)
b = np.array([1, 1, 1, 1], dtype=float)
engine = FFTTransformer()

circular = engine.inverse(engine.transform(a) * engine.transform(b)).real
print("circular (wrong, wraps around):", np.round(circular, 4))
print("true linear conv:", np.convolve(a, b))
```

---

## Tier 2 — Easy

### 6. Plain Polynomial Multiplication (no weights)
**Problem:** Multiply two polynomials given as ascending-power coefficient arrays using FFT-based convolution — the base case before you add any weighting twist.

**Key idea:** Exactly the padding + multiply + crop recipe from Problem 4, no extra steps.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def polynomial_multiply(P, Q):
    engine = FFTTransformer()
    lin_len = len(P) + len(Q) - 1
    N = next_power_of_two(lin_len)
    Pp = np.pad(P, (0, N - len(P)))
    Qp = np.pad(Q, (0, N - len(Q)))
    conv = engine.inverse(engine.transform(Pp) * engine.transform(Qp)).real
    return np.round(conv[:lin_len]).astype(int)

print(polynomial_multiply([1, 2], [3, 4]))   # (1+2x)(3+4x) = 3 + 10x + 8x^2 -> [3,10,8]
```

---

### 7. Detecting an Integer Shift Between Two Signals
**Problem:** `b` is `a` circularly shifted by an unknown amount. Find the shift using cross-correlation.

**Key idea:** `argmax(IFFT(FFT(a) * conj(FFT(b))))` gives the shift directly.

```python
import numpy as np
from transforms import FFTTransformer

a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
b = np.roll(a, 3)   # shifted by 3

engine = FFTTransformer()
corr = engine.inverse(engine.transform(a) * np.conj(engine.transform(b))).real
shift = int(np.argmax(corr))
print("detected shift:", shift)   # 3
```

---

### 8. Low-Pass Noise Filtering
**Problem:** A signal has a smooth underlying shape plus high-frequency noise. Clean it by zeroing out the high-frequency bins and inverse-transforming.

**Key idea:** Low frequencies (near index 0 and N) carry the slow-changing "shape"; high frequencies (near N/2) carry noise/sharp jitter. Zeroing the middle of the spectrum smooths the signal.

```python
import numpy as np
from transforms import FFTTransformer

N = 64
t = np.linspace(0, 2 * np.pi, N, endpoint=False)
clean = np.sin(t)
noisy = clean + 0.3 * np.random.randn(N)

engine = FFTTransformer()
X = engine.transform(noisy)

cutoff = 5                      # keep only the lowest `cutoff` frequencies (and their mirrors)
X[cutoff:N - cutoff] = 0        # zero out the middle (high-frequency) bins

filtered = engine.inverse(X).real
```

---

### 9. Moving-Average Filter via FFT
**Problem:** Smooth a signal by convolving it with a box kernel (e.g. `[1/5, 1/5, 1/5, 1/5, 1/5]`) using FFT-based convolution instead of a sliding-window loop.

**Key idea:** A moving average IS a convolution — with a kernel of equal weights.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def fft_convolve(a, b):
    engine = FFTTransformer()
    lin_len = len(a) + len(b) - 1
    N = next_power_of_two(lin_len)
    A = np.pad(a, (0, N - len(a)))
    B = np.pad(b, (0, N - len(b)))
    return engine.inverse(engine.transform(A) * engine.transform(B)).real[:lin_len]

signal = np.array([1, 3, 2, 8, 5, 6, 9, 4], dtype=float)
kernel = np.ones(3) / 3          # 3-point moving average
smoothed = fft_convolve(signal, kernel)
```

---

### 10. Autocorrelation — Finding a Signal's Period
**Problem:** Given a repeating (periodic) but noisy signal, find its period by autocorrelating it with itself and locating the first strong peak after lag 0.

**Key idea:** Autocorrelation = cross-correlation of a signal with itself. Peaks appear at lags equal to the period (and its multiples).

```python
import numpy as np
from transforms import FFTTransformer

N = 64
t = np.arange(N)
period = 8
signal = np.sin(2 * np.pi * t / period) + 0.1 * np.random.randn(N)

engine = FFTTransformer()
X = engine.transform(signal)
autocorr = engine.inverse(X * np.conj(X)).real

# ignore lag 0 (always the biggest peak — a signal always matches itself perfectly)
detected_period = int(np.argmax(autocorr[1:N // 2]) + 1)
print("detected period:", detected_period)
```

---

## Tier 3 — Medium

### 11. Generalized Weighted Polynomial Multiply (mismatched lengths)
**Problem:** Same idea as the weighted-multiply online test, but now `W` might be a *different length* than `P` (e.g. only weights the first few coefficients, rest default to 1).

**Key idea:** Pad the shorter of `P`/`W` with 1s (neutral for multiplication) before the elementwise weighting step, then proceed as before.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def weighted_polynomial_multiply(P, Q, W):
    engine = FFTTransformer()
    if len(W) < len(P):
        W = list(W) + [1] * (len(P) - len(W))   # unweighted coefficients default to x1

    Pw = [p * w for p, w in zip(P, W)]
    lin_len = len(Pw) + len(Q) - 1
    N = next_power_of_two(lin_len)
    Pp = np.pad(Pw, (0, N - len(Pw)))
    Qp = np.pad(Q, (0, N - len(Q)))
    conv = engine.inverse(engine.transform(Pp) * engine.transform(Qp)).real
    return np.round(conv[:lin_len]).astype(int)[::-1].tolist()
```

---

### 12. Reusable "Correct Linear Convolution" Utility
**Problem:** Write one general-purpose `linear_convolve(a, b)` function that always pads correctly and never leaks circular wraparound — something you can reuse across every other problem instead of re-deriving the padding math each time.

**Key idea:** Wrap the padding + crop logic once, cleanly, with clear variable names.

```python
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
```

---

### 13. Robust Cross-Correlation Alignment (with noise)
**Problem:** Two sensor readings of the same event are both noisy and one is shifted relative to the other. Find the best alignment offset even though the correlation peak isn't a perfect spike.

**Key idea:** Same formula as Problem 7, but you must use `np.abs(corr)` (or just `.real` with `argmax`) since noise can create small negative dips — take the single global maximum, not the first local bump.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def best_alignment(a, b):
    engine = FFTTransformer()
    N = next_power_of_two(max(len(a), len(b)))
    Ap = np.pad(a, (0, N - len(a)))
    Bp = np.pad(b, (0, N - len(b)))
    corr = engine.inverse(engine.transform(Ap) * np.conj(engine.transform(Bp))).real
    return int(np.argmax(corr))
```

---

### 14. Basic 2D Image Blur (box kernel, linear convolution)
**Problem:** Blur a grayscale image with a small averaging kernel (e.g. 3×3 of all `1/9`) using the 2D FFT, producing a properly cropped (non-wraparound) result the same size as the input.

**Key idea:** 2D convolution = pad both dimensions to `image_size + kernel_size - 1` (then to powers of two), transform, multiply, inverse-transform, then crop the output back to the original image size, centered on the kernel's origin offset.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def blur_image(img, kernel):
    H, W = img.shape
    kh, kw = kernel.shape
    Ph, Pw = next_power_of_two(H + kh - 1), next_power_of_two(W + kw - 1)

    img_p = np.zeros((Ph, Pw)); img_p[:H, :W] = img
    ker_p = np.zeros((Ph, Pw)); ker_p[:kh, :kw] = kernel

    engine = FFTTransformer()
    # 2D transform = transform every row, then every column (separable)
    IMG = np.array([engine.transform(row) for row in img_p])
    IMG = np.array([engine.transform(col) for col in IMG.T]).T
    KER = np.array([engine.transform(row) for row in ker_p])
    KER = np.array([engine.transform(col) for col in KER.T]).T

    prod = IMG * KER
    out = np.array([engine.inverse(row) for row in prod])
    out = np.array([engine.inverse(col) for col in out.T]).T.real

    r0, c0 = kh // 2, kw // 2
    return out[r0:r0 + H, c0:c0 + W]
```

---

### 15. Dominant Frequency Detection
**Problem:** Given a noisy signal made of one strong sine wave, find its frequency (in cycles-per-signal-length) by locating the tallest non-DC peak in the magnitude spectrum.

**Key idea:** `np.abs(X)` gives the magnitude spectrum; the DC bin (index 0) must be excluded since it's always huge and irrelevant to "how fast the wave oscillates."

```python
import numpy as np
from transforms import FFTTransformer

N = 128
t = np.arange(N)
signal = 3 * np.sin(2 * np.pi * 10 * t / N) + 0.5 * np.random.randn(N)

X = FFTTransformer().transform(signal)
magnitude = np.abs(X)
magnitude[0] = 0                       # ignore the DC term
dominant_freq = int(np.argmax(magnitude[:N // 2]))   # only look at first half (real-signal symmetry)
print("dominant frequency bin:", dominant_freq)   # ~10
```

---

## Tier 4 — Hard

### 16. Noisy 2D Shift Detection Using a Multi-Row Average
**Problem:** Two images are shifted relative to each other, but noise makes a single-row cross-correlation unreliable. Improve robustness by averaging the correlation result across several rows before picking the peak.

**Key idea:** Correlate each row-pair separately, then **average the correlation curves** (not the images) before taking `argmax` — noise tends to cancel out across rows, but the true shift peak reinforces.

```python
import numpy as np
from transforms import FFTTransformer

def robust_horizontal_shift(original, shifted, n_rows_to_sample=10):
    engine = FFTTransformer()
    H, W = original.shape
    rows = np.random.choice(H, size=n_rows_to_sample, replace=False)

    total_corr = np.zeros(W)
    for r in rows:
        a, b = original[r].astype(float), shifted[r].astype(float)
        corr = engine.inverse(engine.transform(a) * np.conj(engine.transform(b))).real
        total_corr += corr

    return int(np.argmax(total_corr))
```

---

### 17. Image Deconvolution (Sharpening a Known Blur)
**Problem:** An image was blurred with a *known* kernel. Undo the blur by dividing spectra, but naive division blows up wherever the kernel's spectrum is near zero — fix it with a small stabilizing constant.

**Key idea:** `H = FFT(kernel)` has some frequencies close to 0; dividing by them amplifies tiny noise into huge spikes. Add a small `epsilon` to the denominator (or only divide where `|H|` is above a threshold) to keep it stable.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def deconvolve_1d(blurred, kernel, epsilon=1e-3):
    engine = FFTTransformer()
    N = next_power_of_two(len(blurred) + len(kernel) - 1)
    B = np.pad(blurred, (0, N - len(blurred)))
    K = np.pad(kernel, (0, N - len(kernel)))

    B_F = engine.transform(B)
    K_F = engine.transform(K)

    # naive division B_F / K_F blows up near-zero bins -- stabilize it:
    K_F_safe = np.where(np.abs(K_F) < epsilon, epsilon, K_F)
    recovered_F = B_F / K_F_safe

    return engine.inverse(recovered_F).real
```

---

### 18. Big-Integer Multiplication with Digit "Limb" Packing
**Problem:** For very large numbers (thousands of digits), a one-digit-per-FFT-slot array gets huge. Pack multiple decimal digits into each slot (a "limb", e.g. base `10^4`) to shrink the transform length, then handle multi-digit carries.

**Key idea:** Instead of `BASE = 10`, use `BASE = 10**k` (e.g. `k=4`). Each limb can now be up to `BASE-1`, so after convolution a single value can be much larger than 10 — the carry math is the same `%`/`//` idea, just with a bigger base.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

BASE_DIGITS = 4
BASE = 10 ** BASE_DIGITS

def to_limbs(text):
    text = text[::-1]
    return [int(text[i:i + BASE_DIGITS][::-1]) for i in range(0, len(text), BASE_DIGITS)]

def multiply_big_numbers_packed(a_str, b_str):
    engine = FFTTransformer()
    A, B = to_limbs(a_str), to_limbs(b_str)

    lin_len = len(A) + len(B) - 1
    N = next_power_of_two(lin_len)
    Ap = np.pad(A, (0, N - len(A)))
    Bp = np.pad(B, (0, N - len(B)))

    conv = engine.inverse(engine.transform(Ap) * engine.transform(Bp)).real
    conv = np.round(conv).astype(np.int64)[:lin_len]

    result, carry = [], 0
    for c in conv:
        c += carry
        result.append(c % BASE)
        carry = c // BASE
    while carry:
        result.append(carry % BASE)
        carry //= BASE

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    digits = str(result[-1]) + "".join(f"{limb:0{BASE_DIGITS}d}" for limb in reversed(result[:-1]))
    return digits
```

---

### 19. Frequency-Domain Notch Filter (Remove Periodic Interference)
**Problem:** An image has a repeating stripe pattern (interference) overlaid on it. Remove it by finding the bright off-center peaks in the 2D magnitude spectrum (the interference's own frequency) and zeroing them out before inverse-transforming.

**Key idea:** A periodic pattern shows up as a small number of sharp, isolated peaks in the frequency domain (away from the DC term), unlike a natural image whose energy is spread out. Zeroing just those bins removes the pattern with minimal damage to the rest of the image.

```python
import numpy as np
from transforms import FFTTransformer

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
```

---

### 20. FFT-Based Template Matching (Pattern Search)
**Problem:** Given a long 1D signal and a short "template" pattern, find every position in the signal where the template best matches (e.g. all positions where the match strength is above a threshold).

**Key idea:** Cross-correlation naturally scores "how well does the template line up here" at every position at once — much faster than sliding a comparison window manually.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def find_pattern(signal, template, threshold_ratio=0.9):
    engine = FFTTransformer()
    N = next_power_of_two(len(signal) + len(template))
    S = np.pad(signal, (0, N - len(signal)))
    T = np.pad(template, (0, N - len(template)))

    corr = engine.inverse(engine.transform(S) * np.conj(engine.transform(T))).real
    best = corr.max()
    matches = np.where(corr >= threshold_ratio * best)[0]
    return matches.tolist()
```

---

## Tier 5 — Extremely Hard

### 21. Wiener Deconvolution (Noise-Aware Restoration)
**Problem:** An image was blurred with a known kernel **and** has additive noise. Plain division (Problem 17) amplifies the noise badly. Restore it using the Wiener filter formula, which balances sharpening against noise amplification.

**Key idea:** Instead of `1/H`, use `H* / (|H|^2 + K)` where `H*` is the complex conjugate of the kernel's spectrum and `K` is a small constant representing the noise-to-signal ratio. As `K → 0` this becomes plain deconvolution; larger `K` trades sharpness for stability.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def wiener_deconvolve(blurred, kernel, K=0.01):
    engine = FFTTransformer()
    N = next_power_of_two(len(blurred) + len(kernel) - 1)
    B = np.pad(blurred, (0, N - len(blurred)))
    Kern = np.pad(kernel, (0, N - len(kernel)))

    B_F = engine.transform(B)
    H_F = engine.transform(Kern)

    wiener_filter = np.conj(H_F) / (np.abs(H_F) ** 2 + K)
    recovered_F = B_F * wiener_filter

    return engine.inverse(recovered_F).real
```

---

### 22. Arbitrary-Length FFT via Bluestein's Algorithm
**Problem:** Your `FFTTransformer` only works when `N` is a power of two. Support **any** length `N` (e.g. `N=1000`) in `O(N log N)` by rewriting the DFT as a convolution and running it through the power-of-two FFT you already have.

**Key idea:** The trick rewrites `x[n] * e^{-2πi kn/N}` using the identity `kn = (k² + n² - (k-n)²)/2`, which turns the whole sum into a convolution of two "chirp" sequences — something your existing power-of-two `FFTTransformer` can compute via `linear_convolve` (Problem 12), even though `N` itself isn't a power of two.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def bluestein_fft(x):
    N = len(x)
    n = np.arange(N)
    chirp = np.exp(-1j * np.pi * n ** 2 / N)     # the "chirp" sequence

    a = x * chirp                                 # signal, pre-multiplied by chirp
    b = np.concatenate([np.conj(chirp), np.zeros(1), np.conj(chirp[1:][::-1])])  # inverse chirp, both directions

    M = next_power_of_two(2 * N - 1)
    A = np.pad(a, (0, M - N))
    B = np.pad(b, (0, M - len(b)))

    engine = FFTTransformer()
    conv = engine.inverse(engine.transform(A) * engine.transform(B))
    return conv[N - 1:2 * N - 1] * chirp           # multiply by chirp once more, then trim to N
```

---

### 23. Streaming Convolution with Overlap-Add
**Problem:** Convolve a very long signal (too long to comfortably transform all at once) with a small fixed kernel, processing it in fixed-size chunks instead — without changing the final result.

**Key idea (overlap-add):** Split the long signal into blocks, linearly convolve each block with the kernel separately (each block's result is slightly longer than the block, due to the kernel's "tail"), then add the overlapping tails of consecutive blocks together when stitching the output back into one long signal.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def overlap_add_convolve(signal, kernel, block_size=256):
    engine = FFTTransformer()
    k_len = len(kernel)
    out_len = len(signal) + k_len - 1
    output = np.zeros(out_len)

    N = next_power_of_two(block_size + k_len - 1)
    K_pad = np.pad(kernel, (0, N - k_len))
    K_F = engine.transform(K_pad)

    for start in range(0, len(signal), block_size):
        block = signal[start:start + block_size]
        block_pad = np.pad(block, (0, N - len(block)))
        conv_block = engine.inverse(engine.transform(block_pad) * K_F).real
        end = start + len(block) + k_len - 1
        output[start:end] += conv_block[:end - start]   # "add" the overlapping tail

    return output
```

---

### 24. Blind Kernel Estimation
**Problem:** You have an original signal and the result of convolving it with an *unknown* kernel. Estimate that unknown kernel (instead of the usual "recover the original" direction), then verify your estimate by reconvolving and comparing.

**Key idea:** The deconvolution formula works either direction — solve for whichever spectrum you don't have: `H_F = Y_F / X_F` (instead of the usual `X_F = Y_F / H_F`), with the same near-zero stabilization trick as Problem 17.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def estimate_kernel(original, convolved, kernel_len, epsilon=1e-3):
    engine = FFTTransformer()
    N = next_power_of_two(len(convolved))
    X = np.pad(original, (0, N - len(original)))
    Y = np.pad(convolved, (0, N - len(convolved)))

    X_F = engine.transform(X)
    Y_F = engine.transform(Y)
    X_F_safe = np.where(np.abs(X_F) < epsilon, epsilon, X_F)

    H_F = Y_F / X_F_safe
    kernel_est = engine.inverse(H_F).real
    return kernel_est[:kernel_len]
```

---

### 25. Full Pipeline: Multi-Stage Image Forensics
**Problem:** An image has been damaged in three ways, in this order: (1) additive noise, (2) each row circularly shifted by a random unknown amount (rolling-shutter style), (3) every row except one "key row" circularly convolved with that key row (BUET-logo style encryption). Recover the original image, undoing the stages in reverse order.

**Key idea:** This is a combination of Problems 9 (denoise), 16 (shift detection), and 21 (deconvolution) chained together — the skill being tested is **recognizing which tool undoes which corruption**, and doing it in the correct reverse order.

```python
import numpy as np
from transforms import FFTTransformer

def full_recovery_pipeline(damaged_img, low_pass_cutoff=40):
    engine = FFTTransformer()
    H, W = damaged_img.shape

    # Stage 1 (undo noise): light low-pass filter, row by row
    denoised = np.zeros_like(damaged_img, dtype=float)
    for r in range(H):
        X = engine.transform(damaged_img[r].astype(float))
        X[low_pass_cutoff:W - low_pass_cutoff] = 0
        denoised[r] = engine.inverse(X).real

    # Stage 2 (undo the key-row convolution): find the key row, deconvolve every other row
    key_row_idx = int(np.argmin(denoised[:, 0]))
    key_row = denoised[key_row_idx]
    Key_F = engine.transform(key_row)
    Key_F_safe = np.where(np.abs(Key_F) < 1e-3, 1e-3, Key_F)

    unconvolved = np.zeros_like(denoised)
    unconvolved[key_row_idx] = key_row
    for r in range(H):
        if r == key_row_idx:
            continue
        Row_F = engine.transform(denoised[r])
        unconvolved[r] = engine.inverse(Row_F / Key_F_safe).real

    # Stage 3 (undo the per-row rolling-shutter shift): correlate against the key row's
    # own neighbours or a known-clean reference row, then np.roll each row back.
    reference = unconvolved[key_row_idx]
    recovered = np.zeros_like(unconvolved)
    for r in range(H):
        row = unconvolved[r]
        corr = engine.inverse(engine.transform(reference) * np.conj(engine.transform(row))).real
        shift = int(np.argmax(corr))
        recovered[r] = np.roll(row, -shift)

    return np.clip(np.round(recovered), 0, 255).astype(np.uint8)
```

---

## How to Use This List

- **Tiers 1–2:** should feel automatic — if they don't, revisit Sections 0–2 of the first guide before moving on.
- **Tiers 3–4:** this is roughly where a 30–40 minute online test tends to sit.
- **Tier 5:** unlikely to appear whole, but each one teaches a technique (Wiener filtering, notch filtering, overlap-add, blind estimation) that could show up as a *sub-step* inside a medium/hard problem — worth recognizing even if you never have to write it from scratch under time pressure.
