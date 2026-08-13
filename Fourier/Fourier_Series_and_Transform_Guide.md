# Fourier Series & Fourier Transform — Theory + Solved Problem Guide

This document has two parts:

- **Part A — Theory**: Fourier Series (FS) and Continuous Fourier Transform (CFT), explained simply, with the math and small examples.
- **Part B — Solved Problems**: every coding problem found in your assignment/test file, each with (1) what theory it's testing, (2) the plan, and (3) a working code snippet (no `np.fft`, everything via `np.trapezoid`, OOP style — matching your course's rules).

---

# PART A — THEORY

## 1. The Big Picture

Almost every signal — a sound wave, an image row, a voltage — can be broken down into simple sine/cosine waves of different frequencies, amplitudes, and phases. Two tools do this:

| Tool | Used for | Output |
|---|---|---|
| **Fourier Series (FS)** | **Periodic** signals (repeat forever with period `T`) | A **discrete** set of frequencies: `0, ±f0, ±2f0, ...` |
| **Fourier Transform (FT/CFT)** | **Any** signal, periodic or not | A **continuous** spectrum `X(f)` for every real frequency `f` |

Think of FS as a special case: if you let the period `T` of a signal go to infinity, the discrete FS frequencies get infinitely close together and merge into the continuous FT spectrum.

---

## 2. Fourier Series (FS)

### 2.1 Trigonometric form (intuition)

If `x(t)` is periodic with period `T`, you can write it as a sum of a constant + cosines + sines:

```
x(t) = a0 + Σ [ an*cos(n*ω0*t) + bn*sin(n*ω0*t) ],   ω0 = 2π/T
```

`n = 1, 2, 3, ...` are the **harmonics**. `an`, `bn` tell you "how much" of each frequency is present.

### 2.2 Complex exponential form (what you'll actually code)

Using Euler's identity `e^(jθ) = cos θ + j sin θ`, the same thing is written more compactly:

```
x(t) = Σ (n = -∞ to ∞)  cn * e^(j n ω0 t)
```

where the complex coefficient `cn` is found by:

```
cn = (1/T) * ∫[0 to T]  x(t) * e^(-j n ω0 t)  dt
```

**Key ideas:**
- `cn` is a complex number. `|cn|` = strength of that frequency, `arg(cn)` = its phase.
- Unlike the sine/cosine form, **negative `n` is not redundant here** — `c(-n)` and `cn` are complex conjugates for real signals, and both are needed to reconstruct the signal correctly (drop them and the reconstruction gets visibly wrong).
- To numerically compute `cn` without `np.fft`, just do the integral directly with `np.trapezoid(x(t) * np.exp(-1j*n*omega*t), t)`.

### 2.3 Why "epicycles"?

Each term `cn * e^(j n ω0 t)` is a **vector rotating at constant speed** `n*ω0`, with fixed length `|cn|` and starting angle `arg(cn)`. Put these rotating vectors tip-to-tail and the tip traces out the signal. Add more harmonics (`n` up to some `N`) and the drawing gets sharper — this is the classic "Fourier draws a picture" animation.

### 2.4 Simple example

A square wave (period `T=2`, amplitude 1) has FS coefficients `cn = 0` for even `n`, and `cn = 2/(jnπ)` for odd `n`. Adding more odd harmonics makes the approximation look more and more like a square wave (with ripples near the edges — this is the **Gibbs phenomenon**).

---

## 3. Continuous Fourier Transform (CFT)

### 3.1 Definition

For a (not necessarily periodic) signal `x(t)`:

```
Forward:  X(f) = ∫(-∞ to ∞)  x(t) * e^(-j 2π f t)  dt
Inverse:  x(t) = ∫(-∞ to ∞)  X(f) * e^( j 2π f t)  df
```

- `X(f)` is complex: `|X(f)|` = **magnitude spectrum** (how much of frequency `f` is present), `∠X(f)` = **phase spectrum** (its phase offset).
- In practice, the integral is over the finite range where your signal was sampled (e.g. `t ∈ [-5, 5]`), not truly `±∞` — that's an approximation, which is fine as long as the signal is negligible outside that range.

### 3.2 Computing it numerically (no `np.fft` allowed)

Since you can't use `np.fft`, you compute `X(f)` **by definition**, one frequency at a time:

```python
import numpy as np

def cft_at_f(t, x, f):
    kernel = np.exp(-1j * 2 * np.pi * f * t)
    real = np.trapezoid(np.real(x * kernel), t)
    imag = np.trapezoid(np.imag(x * kernel), t)
    return real + 1j * imag
```

Loop this over an array of frequencies `f` to get the full spectrum `X(f)`. This is slow (`O(N_t * N_f)`) but that's expected — it's what the "no FFT" rule wants.

### 3.3 Core CFT Properties (this is what almost every problem in your file tests)

Let `x(t) ↔ X(f)` mean "x(t) has CFT X(f)". `f0`, `t0`, `a` are constants.

**(a) Linearity**
```
A*x(t) + B*y(t)  ↔  A*X(f) + B*Y(f)
```

**(b) Time Shift** — shifting a signal in time does **not** change its magnitude spectrum, only adds a linear phase:
```
x(t - t0)  ↔  X(f) * e^(-j 2π f t0)
```
So `|Shifted spectrum| = |X(f)|` and `∠(Shifted spectrum) = ∠X(f) - 2π f t0`.

**(c) Time Scaling** — compressing time by `a` stretches the spectrum by `a` (and scales its height):
```
x(a*t)  ↔  (1/|a|) * X(f/a)
```
Compress the signal (large `a`) → spectrum spreads out (wider bandwidth). This is the time-frequency "trade-off".

**(d) Frequency Shift (Modulation)** — multiplying by a complex exponential in time shifts the spectrum:
```
x(t) * e^(j 2π f0 t)  ↔  X(f - f0)
```
This is exactly the "phase shift by `2π f0 t`" language your problems use — it's really a **frequency-domain shift**, achieved by multiplying the time-domain signal by `e^(j2πf0t)`.

**(e) Combined shift + scale** (used in your first problem): if
```
y(t) = x(a*t) * e^(j 2π f0 t)      [scale then modulate]
```
then combining (c) and (d):
```
Y(f) = (1/|a|) * X( (f - f0) / a )
```
This is exactly the identity your assignment asks you to verify numerically: `|Y(f)| = 1/|a| * |X((f-f0)/a)|` and matching phase.

**(f) Differentiation** — a derivative in time becomes multiplication by `j2πf` in frequency:
```
dx/dt  ↔  j 2π f X(f)
d²x/dt²  ↔  (j 2π f)² X(f) = -(2πf)² X(f)
```
This is why high frequencies "dominate" a derivative — differentiating amplifies high frequencies.

### 3.4 Why properties matter for "noise removal" problems

Noise (especially random, fine-grained noise) tends to live at **high frequencies**, while the main structure of a signal/image (smooth variation) lives at **low frequencies**. So: take the FT → zero out (or shrink) the high-frequency bins → inverse FT → you get a denoised version. This is a **low-pass filter**. The opposite (zeroing low frequencies, keeping high) is a **high-pass filter**, used for **edge detection**, since edges are fast (high-frequency) intensity changes.

---

## 4. 2D Continuous Fourier Transform (for images)

An image `I(x,y)` is a 2D continuous signal. Its 2D CFT is:

```
F(u,v) = ∫∫ I(x,y) * e^(-j 2π (u x + v y))  dx dy
```

Using Euler's identity this splits into real/imaginary parts, each a normal (real-valued) double integral — computable with `np.trapezoid`.

**Separability trick (important for speed):** `e^(-j2π(ux+vy)) = e^(-j2πux) * e^(-j2πvy)`, so the double integral over `(x,y)` for every `(u,v)` can be done as **two sequential 1D integrals**: first integrate over `x` for every row and every candidate `u` (giving an intermediate array), then integrate that result over `y` for every candidate `v`. This turns an `O(N^4)` computation into `O(N^3)` — the only way to make this tractable without `np.fft`.

**Low frequencies** `(u,v) ≈ (0,0)` = smooth/flat regions (most of an image's energy). **High frequencies** (far from origin) = edges, texture, noise. A **high-pass filter** zeroes out a disk of radius `cutoff` around the origin of `F(u,v)`; inverse-transforming what's left leaves mostly edges.

---

# PART B — SOLVED PROBLEM GUIDE

All problems below share a common toolkit. Build these once, reuse everywhere.

## 0. Shared Toolkit (OOP framework)

```python
import numpy as np
import matplotlib.pyplot as plt


class SignalGenerator:
    """Generates various 1D signals over a given time axis."""

    def __init__(self, t):
        self.t = t

    def square(self, freq=1, duty=0.5):
        return np.sign(np.sin(2 * np.pi * freq * self.t))

    def triangle(self, freq=1):
        # 2/pi * arcsin(sin(...)) gives a triangle wave in [-1, 1]
        return (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * freq * self.t))

    def gaussian(self, a=1.0):
        return np.exp(-a * self.t ** 2)

    def time_shift(self, x, t0):
        """Return x(t - t0) evaluated on self.t, via interpolation
        (NOT manual index shifting)."""
        return np.interp(self.t - t0, self.t, x, left=0, right=0)

    def time_scale(self, x, a):
        """Return x(a*t) evaluated on self.t, via interpolation."""
        return np.interp(a * self.t, self.t, x, left=0, right=0)

    def modulate(self, x, f0):
        """Multiply by e^(j*2*pi*f0*t) -> frequency shift by f0."""
        return x * np.exp(1j * 2 * np.pi * f0 * self.t)


class CFTAnalyzer:
    """Computes the Continuous Fourier Transform of a sampled signal
    by direct numerical integration (no np.fft)."""

    def __init__(self, t, x):
        self.t = t
        self.x = x

    def transform(self, f_axis):
        X = np.zeros(len(f_axis), dtype=complex)
        for i, f in enumerate(f_axis):
            kernel = np.exp(-1j * 2 * np.pi * f * self.t)
            real = np.trapezoid(np.real(self.x * kernel), self.t)
            imag = np.trapezoid(np.imag(self.x * kernel), self.t)
            X[i] = real + 1j * imag
        return X

    @staticmethod
    def magnitude(X):
        return np.abs(X)

    @staticmethod
    def phase(X):
        return np.angle(X)

    @staticmethod
    def mse(a, b):
        return np.mean((a - b) ** 2)
```

Everything below reuses `SignalGenerator` and `CFTAnalyzer`.

---

## Problem 1 — Combined Time-Shift + Time-Scale Property (`x(t) = Square + Triangle`)

**What's being tested:** Property (e) from the theory section — modulation (frequency shift by `f0`) combined with time scaling by `a`.

**Setup:** `x(t) = Square(t) + Triangle(t)`. Build:
```
y(t) = x(a*t) * e^(j 2π f0 t)      with a = 10, f0 = 10
```
Theory predicts: `Y(f) = (1/|a|) * X((f - f0)/a)`.

```python
t = np.linspace(-5, 5, 4000)
gen = SignalGenerator(t)

x = gen.square(freq=1) + gen.triangle(freq=1)

a, f0 = 10, 10
x_scaled = gen.time_scale(x, a)          # x(a*t)
y = gen.modulate(x_scaled.astype(complex), f0)   # x(a*t) * e^(j2*pi*f0*t)

f_axis = np.linspace(-10, 10, 1000)
cft = CFTAnalyzer(t, x.astype(complex))
X = cft.transform(f_axis)

cft_y = CFTAnalyzer(t, y)
Y = cft_y.transform(f_axis)

# Theoretical prediction: (1/|a|) * X((f - f0)/a)
X_pred = CFTAnalyzer(t, x.astype(complex)).transform((f_axis - f0) / a) / abs(a)

mse_mag = CFTAnalyzer.mse(np.abs(Y), np.abs(X_pred))
mse_phase = CFTAnalyzer.mse(np.angle(Y), np.angle(X_pred))
print("MSE magnitude:", mse_mag)
print("MSE phase:", mse_phase)

plt.figure()
plt.plot(f_axis, np.abs(Y), label="|Y(f)|")
plt.plot(f_axis, np.abs(X_pred), '--', label="(1/|a|)|X((f-f0)/a)|")
plt.legend(); plt.xlabel("f"); plt.title("Magnitude comparison")
plt.show()
```

**Interpretation:** small `mse_mag` and `mse_phase` confirm the scaling+shift property numerically — compressing in time spreads/shrinks the spectrum (scaling), and multiplying by `e^(j2πf0t)` slides it over by `f0` (frequency shift).

---

## Problem 2 — Time-Shift Property with a Gaussian Signal

**What's being tested:** Property (b), pure time shift.

```
x(t) = gaussian(a)
y(t) = x(t - t0)
```
Prediction: `Y(f) = X(f) * e^(-j 2π f t0)`, so `|Y(f)| = |X(f)|` and `∠Y(f) = ∠X(f) - 2π f t0`.

```python
t = np.linspace(-5, 5, 2000)
gen = SignalGenerator(t)
x = gen.gaussian(a=2.0).astype(complex)

t0 = 1.5
y = gen.time_shift(x, t0)

f_axis = np.linspace(-10, 10, 1000)
X = CFTAnalyzer(t, x).transform(f_axis)
Y = CFTAnalyzer(t, y).transform(f_axis)

# Theoretical prediction
X_pred_phase = np.angle(X) - 2 * np.pi * f_axis * t0
mag_mse = CFTAnalyzer.mse(np.abs(Y), np.abs(X))
phase_mse = CFTAnalyzer.mse(np.angle(Y), X_pred_phase)
print("Magnitude MSE:", mag_mse)   # should be ~0: shifting doesn't change |X(f)|
print("Phase MSE:", phase_mse)

plt.figure()
plt.plot(f_axis, np.abs(X), label="|X(f)|")
plt.plot(f_axis, np.abs(Y), '--', label="|Y(f)|")
plt.legend(); plt.title("Magnitude is unchanged by time shift")
plt.show()
```

**Interpretation:** magnitude curves overlap exactly (shift doesn't change "how much" of each frequency exists); phase differs by the linear term `-2πf t0`, confirmed by the low MSE.

---

## Problem 3 — Differentiation Property (1st, 2nd, 3rd derivative)

**What's being tested:** Property (f). `y1 = x'(t)`, `y2 = x''(t)`, `y3 = x'''(t)`. Prediction: `Yk(f) = (j2πf)^k * X(f)`.

```python
def numerical_derivative(x, t, order=1):
    d = x.copy()
    for _ in range(order):
        d = np.gradient(d, t)
    return d

t = np.linspace(-5, 5, 3000)
gen = SignalGenerator(t)
x = gen.gaussian(a=1.0).astype(complex)

f_axis = np.linspace(-10, 10, 1000)
X = CFTAnalyzer(t, x).transform(f_axis)

for order in (1, 2, 3):
    y = numerical_derivative(x, t, order=order).astype(complex)
    Y = CFTAnalyzer(t, y).transform(f_axis)

    predicted = ((1j * 2 * np.pi * f_axis) ** order) * X

    mse_mag = CFTAnalyzer.mse(np.abs(Y), np.abs(predicted))
    mse_phase = CFTAnalyzer.mse(np.angle(Y), np.angle(predicted))
    print(f"Order {order}: mag MSE={mse_mag:.3e}, phase MSE={mse_phase:.3e}")
```

**Interpretation:** each derivative multiplies the spectrum by another factor of `j2πf`, so higher derivatives boost high frequencies more — matching the low MSE against the theoretical `(j2πf)^order * X(f)`.

---

## Problem 4 — Fourier Series "Epicycles" (redrawing an SVG shape)

**What's being tested:** Complex exponential FS coefficients (Section 2.2), computed for an arbitrary sampled 2D path (given as complex `z(t) = x(t) + j*y(t)`), then reconstructed as a partial sum of rotating vectors.

```python
class FourierEpicycles:
    def __init__(self, t, signal, n_harmonics):
        self.t = t
        self.signal = signal          # complex z(t)
        self.n_harmonics = n_harmonics
        self.T = t[-1] - t[0]
        self.omega = 2 * np.pi / self.T
        self.coeffs = {}              # n -> c_n

    def calculate_cn(self, n):
        kernel = np.exp(-1j * n * self.omega * self.t)
        integrand = self.signal * kernel
        real = np.trapezoid(np.real(integrand), self.t)
        imag = np.trapezoid(np.imag(integrand), self.t)
        return (real + 1j * imag) / self.T

    def calculate_all_coefficients(self):
        for n in range(-self.n_harmonics, self.n_harmonics + 1):
            self.coeffs[n] = self.calculate_cn(n)

    def approximate(self, t):
        t = np.atleast_1d(t)
        result = np.zeros(len(t), dtype=complex)
        for n, cn in self.coeffs.items():
            result += cn * np.exp(1j * n * self.omega * t)
        return result if len(result) > 1 else result[0]
```

Usage (matches the assignment's `main` block, which is already provided):

```python
from svgutils import load_svg_path

t, z = load_svg_path("svgs/heart.svg", num_points=1000)
fs = FourierEpicycles(t, z, n_harmonics=150)
fs.calculate_all_coefficients()

recon = fs.approximate(t)   # complex array -> plot recon.real vs recon.imag
plt.plot(z.real, z.imag, label="Original")
plt.plot(recon.real, recon.imag, '--', label="Reconstruction (N=150)")
plt.axis('equal'); plt.legend(); plt.show()
```

**Interpretation:** each `cn` is one rotating vector (epicycle) of speed `n*ω`. Summing all of them for `n = -N..N` at each instant `t` reconstructs the shape; more harmonics `N` → sharper corners, closer match. Negative `n` are required — dropping them visibly distorts the shape (it won't close/trace correctly).

---

## Problem 5 — 2D CFT Edge Detection (`pikachu.png` → edge map)

**What's being tested:** Section 4 (2D CFT), separable numerical integration, and high-pass filtering.

```python
class CFT2D:
    def __init__(self, I, x, y):
        self.I = I
        self.x = x
        self.y = y
        # Frequency axes spanning the Nyquist range implied by pixel spacing
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        self.u = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx))  # axis values only
        self.v = np.fft.fftshift(np.fft.fftfreq(len(y), d=dy))  # (no FFT used for the transform itself)

    def compute_cft(self):
        Nu, Nv = len(self.u), len(self.v)
        # Stage 1: integrate over x for every row, every u  -> shape (rows, Nu)
        cos_ux = np.cos(2 * np.pi * np.outer(self.x, self.u))   # (Nx, Nu)
        sin_ux = np.sin(2 * np.pi * np.outer(self.x, self.u))
        stage1_re = np.trapezoid(self.I[:, :, None] * cos_ux[None, :, :], self.x, axis=1)  # (Ny, Nu)
        stage1_im = np.trapezoid(-self.I[:, :, None] * sin_ux[None, :, :], self.x, axis=1)

        # Stage 2: integrate the stage-1 result over y for every v -> shape (Nv, Nu)
        cos_vy = np.cos(2 * np.pi * np.outer(self.y, self.v))   # (Ny, Nv)
        sin_vy = np.sin(2 * np.pi * np.outer(self.y, self.v))

        real = (np.trapezoid(stage1_re[:, :, None] * cos_vy[:, None, :], self.y, axis=0)
                - np.trapezoid(stage1_im[:, :, None] * sin_vy[:, None, :], self.y, axis=0))
        imag = (np.trapezoid(stage1_re[:, :, None] * (-sin_vy[:, None, :]), self.y, axis=0)
                - np.trapezoid(stage1_im[:, :, None] * cos_vy[:, None, :], self.y, axis=0))
        return real.T, imag.T   # shape (Nv, Nu)

    def plot_magnitude(self):
        real, imag = self.compute_cft()
        mag = np.sqrt(real ** 2 + imag ** 2)
        plt.imshow(np.log1p(mag), cmap='gray')
        plt.title("Log magnitude spectrum")
        plt.show()


class InverseCFT2D:
    def __init__(self, real, imag, u, v, x, y):
        self.real, self.imag = real, imag
        self.u, self.v, self.x, self.y = u, v, x, y

    def reconstruct(self):
        # Same separable two-stage trapezoidal integration, run in reverse
        cos_uv_x = np.cos(2 * np.pi * np.outer(self.u, self.x))
        sin_uv_x = np.sin(2 * np.pi * np.outer(self.u, self.x))
        stage1_re = np.trapezoid(self.real[:, :, None] * cos_uv_x[:, None, :]
                                  - self.imag[:, :, None] * sin_uv_x[:, None, :], self.u, axis=0)
        stage1_im = np.trapezoid(self.real[:, :, None] * sin_uv_x[:, None, :]
                                  + self.imag[:, :, None] * cos_uv_x[:, None, :], self.u, axis=0)

        cos_v_y = np.cos(2 * np.pi * np.outer(self.v, self.y))
        sin_v_y = np.sin(2 * np.pi * np.outer(self.v, self.y))
        out_re = np.trapezoid(stage1_re[:, :, None] * cos_v_y.T[None, :, :]
                               - stage1_im[:, :, None] * sin_v_y.T[None, :, :], self.v, axis=1)
        return out_re  # real-valued reconstructed image (possibly negative)
```

`FrequencyFilter.high_pass(real, imag, cutoff)` (given) zeroes a central disk of radius `cutoff` (pixel-index units) in the shifted spectrum before you call `InverseCFT2D.reconstruct()`.

**Interpretation:** the low frequencies near the origin carry the smooth brightness of the image; zeroing them (high-pass) and inverse-transforming leaves the fast-changing content — edges — behind, per Section 4.

---

## Problem 6 — Denoising a Secret Letter (row-wise FT low-pass filter)

**What's being tested:** Same idea as edge detection but inverted — a **low-pass** filter to remove noise instead of a high-pass filter to find edges. Hint says "apply FT row by row."

```python
def denoise_image(image, cutoff_fraction=0.1):
    """image: 2D real array. Denoise each row independently using a
    1D CFT (computed manually) and a low-pass cutoff."""
    rows, cols = image.shape
    x = np.arange(cols)
    f_axis = np.fft.fftshift(np.fft.fftfreq(cols))  # frequency axis values only

    denoised = np.zeros_like(image, dtype=float)
    for r in range(rows):
        row = image[r, :].astype(complex)

        # Forward CFT of this row (manual integration)
        cft = CFTAnalyzer(x, row)
        F = cft.transform(f_axis)

        # Low-pass: zero out frequencies with |f| beyond cutoff
        cutoff = cutoff_fraction * f_axis.max()
        F_filtered = F.copy()
        F_filtered[np.abs(f_axis) > cutoff] = 0

        # Inverse CFT (manual): x(t) = ∫ X(f) e^(j2*pi*f*t) df
        recon = np.zeros(cols, dtype=complex)
        for i, xi in enumerate(x):
            kernel = np.exp(1j * 2 * np.pi * f_axis * xi)
            recon[i] = np.trapezoid(F_filtered * kernel, f_axis)

        denoised[r, :] = np.real(recon)

    return denoised
```

**Interpretation:** noise usually appears as fast, small, random fluctuations — high spatial frequency. Zeroing frequencies beyond a chosen `cutoff` and reconstructing keeps the slow-varying "letter shape" while discarding most of the noise. Tune `cutoff_fraction` by eye until the letter is recognizable (the assignment explicitly says it doesn't need to be perfectly clean).

---

## Problem 7 — Decomposing `f(t) = 2sin(14πt) − sin(2πt)(4sin(2πt)sin(14πt) − 1)` via FT

**What's being tested:** using the CFT's magnitude spectrum to **identify unknown frequencies** hidden inside a product/sum of sinusoids, then reconstructing the signal as a sum of pure sinusoids found from the peaks.

**Plan:**
1. Sample `f(t)` over a reasonable range with fine resolution.
2. Compute its CFT magnitude spectrum via direct integration.
3. Find the frequencies where `|F(f)|` peaks — those are the "hidden" sinusoid frequencies (a product of sines expands, via trig identities, into a sum of sines/cosines at *sum and difference* frequencies).
4. Reconstruct `f(t)` as a sum of unit-amplitude, zero-phase sinusoids at exactly those found frequencies, and confirm it matches the original.

```python
t = np.linspace(-2, 2, 4000)
f_t = 2*np.sin(14*np.pi*t) - np.sin(2*np.pi*t)*(4*np.sin(2*np.pi*t)*np.sin(14*np.pi*t) - 1)

plt.plot(t, f_t); plt.title("Original f(t)"); plt.show()

f_axis = np.linspace(-15, 15, 3000)
X = CFTAnalyzer(t, f_t.astype(complex)).transform(f_axis)
mag = np.abs(X)

plt.plot(f_axis, mag); plt.title("|F(f)| — peaks show hidden frequencies")
plt.show()

# Pick out peaks (positive-frequency side, since real signal -> symmetric spectrum)
from scipy.signal import find_peaks
peaks, _ = find_peaks(mag[f_axis >= 0], height=0.1 * mag.max())
found_freqs = f_axis[f_axis >= 0][peaks]
print("Frequencies found (Hz, i.e. cycles per unit t):", found_freqs)

# Reconstruct as sum of unit-amplitude sinusoids at those frequencies
recon = np.zeros_like(t)
for fr in found_freqs:
    recon += np.sin(2 * np.pi * fr * t)   # amplitude 1, phase 0 as stated in the problem

plt.plot(t, f_t, label="Original")
plt.plot(t, recon, '--', label="Reconstruction from FT peaks")
plt.legend(); plt.show()
```

**Interpretation:** expanding `sin(A)*sin(B)` with the product-to-sum identity `sin(A)sin(B) = ½[cos(A-B) − cos(A+B)]` shows the product term hides components at `14π ± 2π` (i.e. frequencies `8` and `6` in Hz after dividing by `2π`), in addition to the standalone `2sin(14πt)` (frequency `7`) and `sin(2πt)` (frequency `1`) terms. The CFT peaks confirm this without doing the trig algebra by hand — that's the whole point of using the transform.

---

# Quick Reference — Properties Cheat Sheet

| Property | Time domain | Frequency domain |
|---|---|---|
| Linearity | `A x(t) + B y(t)` | `A X(f) + B Y(f)` |
| Time shift | `x(t - t0)` | `X(f) e^(-j2πf t0)` |
| Time scaling | `x(a t)` | `(1/|a|) X(f/a)` |
| Frequency shift | `x(t) e^(j2πf0 t)` | `X(f - f0)` |
| Differentiation | `dx/dt` | `j2πf X(f)` |
| n-th derivative | `d^n x/dt^n` | `(j2πf)^n X(f)` |

**General numerical-CFT rules used throughout (per your assignment constraints):**
- Never use `np.fft` or any built-in FT routine — compute integrals directly.
- Always use `np.trapezoid` (or `np.trapz`) for every integral.
- Wrap everything in classes (`SignalGenerator`, `CFTAnalyzer`, `FourierEpicycles`, `CFT2D`, `InverseCFT2D`, ...) — no free-floating manual array manipulation.
- Verify every property numerically with an MSE between the measured and theoretically-predicted magnitude/phase; small MSE = correct implementation.
