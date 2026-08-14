# 15 Practice Problems — Built on the FS Epicycles + 2D-CFT Offline Assignment

These problems extend and stress-test the two pieces of code you already wrote:

- **Task 1 code** — `FourierEpicycles` (`fs_redrawer.py`, using `svg_utils.py` / `epicycle_animation.py`)
- **Task 2 code** — `ContinuousImage`, `CFT2D`, `FrequencyFilter`, `InverseCFT2D` (`cft_edge_detector.py`)

Every problem below is scoped like your 30–40 minute online lab tests: a short theory ask + a numerical verification + an MSE/plot check. They're ordered **easy → hard**. Problems 1–7 mostly touch Task 1 (Fourier Series), 8–14 mostly touch Task 2 (2D-CFT), and 15 fuses both.

> Assumed available in every snippet: your already-implemented `FourierEpicycles`, `CFT2D`, `InverseCFT2D`, `ContinuousImage`, `FrequencyFilter`, `load_svg_path`, `save_outputs` — imported exactly as in your files.

---

## Problem 1 (Easy) — How many harmonics are "enough"?

**Concept:** Fourier Series convergence — the partial sum `f̂_N(t)` approaches `f(t)` as `N → ∞`.

**Theory:** Each extra harmonic pair `(cn, c-n)` adds a finer wiggle to the reconstruction. The reconstruction error (measured as mean squared error between `f̂_N(t)` and the true `z(t)`) should shrink as `N` grows, but with diminishing returns — corners (like a star's points) need more harmonics than smooth curves (like a circle).

**Task:** For `circle.svg` and `star.svg`, compute reconstruction MSE for `N ∈ {5, 10, 25, 50, 100, 150}` and plot MSE vs `N` (log scale) for both. Confirm the star needs more harmonics for the same MSE.

```python
import numpy as np
import matplotlib.pyplot as plt
from svg_utils import load_svg_path
from fs_redrawer import FourierEpicycles

def reconstruction_mse(svg_path, N):
    t, z = load_svg_path(svg_path, num_points=1000)
    fs = FourierEpicycles(t, z, n_harmonics=N)
    fs.calculate_all_coefficients()
    z_hat = fs.approximate(t)
    return np.mean(np.abs(z - z_hat) ** 2)

Ns = [5, 10, 25, 50, 100, 150]
for shape in ["svgs/circle.svg", "svgs/star.svg"]:
    mses = [reconstruction_mse(shape, N) for N in Ns]
    plt.plot(Ns, mses, marker='o', label=shape)

plt.yscale('log'); plt.xlabel("N harmonics"); plt.ylabel("MSE (log scale)")
plt.legend(); plt.title("Reconstruction error vs. number of harmonics")
plt.show()
```

**Interpretation:** the star's sharp corners are high-frequency content — more harmonics are needed to resolve them, so its MSE curve sits above the circle's at every `N`.

---

## Problem 2 (Easy) — Verify `cn` against a shape you can compute by hand

**Concept:** Sanity-checking `calculate_cn` against a known closed-form answer.

**Theory:** `circle.svg` is (approximately) a pure circle traced at constant speed: `z(t) = R e^(jt)`. Its Fourier Series has **only one nonzero coefficient**: `c1 = R`, and `cn = 0` for all `n ≠ 1` (a circle traced at the fundamental frequency is already a single rotating vector).

**Task:** Load `circle.svg`, compute all `cn` for `n = -5..5`, and confirm `|c1|` dominates while the rest are near zero.

```python
t, z = load_svg_path("svgs/circle.svg", num_points=1000)
fs = FourierEpicycles(t, z, n_harmonics=5)
fs.calculate_all_coefficients()

for n in range(-5, 6):
    print(f"n={n:+d}  |c_n|={abs(fs.coeffs[n]):.4f}")
```

**Interpretation:** `|c1|` should be close to the circle's radius (~1, since `svg_utils` scales shapes into `[-1,1]`), and every other `|cn|` should be near zero (small nonzero values come only from the polyline/arc-length approximation, not the true circle).

---

## Problem 3 (Easy) — Verify the inverse 2D-CFT is a perfect round trip

**Concept:** `compute_cft()` followed immediately by `InverseCFT2D.reconstruct()` (no filtering) should return (approximately) the original image — this is the discrete analogue of "Fourier transform then inverse transform = identity."

**Theory:** From Eq. (3)-(5) in the assignment: `I(x,y) = ∫∫ F(u,v) e^{j2π(ux+vy)} du dv` where `F(u,v)` is exactly the CFT of `I`. If both stages of your separable integration are correct, doing forward then inverse with no filter should reproduce `I(x,y)` up to small numerical (trapezoidal-integration) error.

```python
from cft_edge_detector import ContinuousImage, CFT2D, InverseCFT2D
import numpy as np

img = ContinuousImage("pikachu.png")
cft = CFT2D(img)
real, imag = cft.compute_cft()

icft = InverseCFT2D(real, imag, cft.u, cft.v, img.x, img.y)
recon = icft.reconstruct()

mse = np.mean((recon - img.image) ** 2)
print("Round-trip MSE:", mse)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img.image, cmap='gray'); ax[0].set_title("Original")
ax[1].imshow(recon, cmap='gray'); ax[1].set_title("Reconstructed (no filter)")
plt.show()
```

**Interpretation:** a small MSE confirms your `compute_cft`/`reconstruct` pair is mathematically consistent — this is the single best debugging check before trusting any filtered result.

---

## Problem 4 (Easy) — Cutoff radius sweep

**Concept:** How the high-pass cutoff trades off "how much detail survives" vs "how much low-frequency content is removed."

**Theory:** `FrequencyFilter.high_pass` zeroes a disk of radius `cutoff` around the spectrum's center. A larger `cutoff` removes more low-*and-medium* frequency content, leaving only the very sharpest edges; a small `cutoff` keeps more of the image's shading along with the edges.

**Task:** Run the full edge-detection pipeline for `cutoff ∈ {2, 8, 15, 30, 60}` and display all five outputs in a grid.

```python
from cft_edge_detector import ContinuousImage, CFT2D, FrequencyFilter, InverseCFT2D
import matplotlib.pyplot as plt
import numpy as np

img = ContinuousImage("pikachu.png")
cft = CFT2D(img)
real, imag = cft.compute_cft()
filt = FrequencyFilter()

cutoffs = [2, 8, 15, 30, 60]
fig, axes = plt.subplots(1, len(cutoffs), figsize=(15, 3))
for ax, c in zip(axes, cutoffs):
    real_f, imag_f = filt.high_pass(real, imag, c)
    edges = InverseCFT2D(real_f, imag_f, cft.u, cft.v, img.x, img.y).reconstruct()
    edge_map = np.abs(edges); edge_map = edge_map / edge_map.max()
    ax.imshow(1 - edge_map, cmap='gray'); ax.set_title(f"cutoff={c}"); ax.axis('off')
plt.show()
```

**Interpretation:** too small a cutoff leaves shading noise in the "edge map"; too large a cutoff erases faint but real edges — there's a sweet spot, which is why the assignment default is 15.

---

## Problem 5 (Medium) — Time-shifting the epicycle drawing

**Concept:** The Fourier Series time-shift property, applied to a rotating-vector (epicycle) signal.

**Theory:** If `f(t) ↔ cn`, then shifting the start point of the trace, `g(t) = f(t - t0)`, has coefficients
```
c'n = cn * e^(-j n ω t0)
```
i.e. only the **phase** of each coefficient changes (`|c'n| = |cn|`); the magnitude spectrum (and hence the drawn shape) is identical, only rotated in "starting angle," not in geometry — because a rotation of the parametrization doesn't move the actual curve.

**Task:** Build `g(t) = f(t - t0)` for the heart shape via interpolation (like your earlier `SignalGenerator.time_shift`), compute its Fourier Series independently, and confirm `|c'n| ≈ |cn|` for every `n`.

```python
t, z = load_svg_path("svgs/heart.svg", num_points=1000)

t0 = 1.0
z_shifted = np.interp(
    (t - t0) % (2 * np.pi), t, z.real
) + 1j * np.interp((t - t0) % (2 * np.pi), t, z.imag)

fs_orig = FourierEpicycles(t, z, n_harmonics=30); fs_orig.calculate_all_coefficients()
fs_shift = FourierEpicycles(t, z_shifted, n_harmonics=30); fs_shift.calculate_all_coefficients()

mags_orig = np.array([abs(fs_orig.coeffs[n]) for n in range(-30, 31)])
mags_shift = np.array([abs(fs_shift.coeffs[n]) for n in range(-30, 31)])
print("Magnitude MSE (should be small):", np.mean((mags_orig - mags_shift) ** 2))

# Phase check against theory: c'_n should equal c_n * exp(-j n omega t0)
omega = fs_orig.omega
predicted_phase_shift = {n: np.angle(fs_orig.coeffs[n] * np.exp(-1j*n*omega*t0)) for n in range(-30, 31)}
actual_phase = {n: np.angle(fs_shift.coeffs[n]) for n in range(-30, 31)}
phase_err = np.mean([ (predicted_phase_shift[n] - actual_phase[n])**2 for n in range(-30, 31) ])
print("Phase MSE:", phase_err)
```

**Interpretation:** near-zero magnitude MSE confirms shifting the trace's starting point doesn't change the shape's "energy per harmonic"; the phase check confirms the exact `e^(-jnωt0)` rule.

---

## Problem 6 (Medium) — Low-pass filtering an epicycle drawing (harmonic truncation as a filter)

**Concept:** Using fewer harmonics is literally a **low-pass filter** on the shape — dropping high `|n|` terms removes fine (high-frequency) detail from the traced curve.

**Theory:** The full series has terms for all integer `n`; restricting to `|n| ≤ N` (which is exactly what `n_harmonics` does) is mathematically identical to zeroing out `cn` for `|n| > N` — a brick-wall low-pass filter in the "harmonic index" domain, the FS analogue of Task 2's spatial-frequency high-pass filter.

**Task:** Compute coefficients once with a large `N_max = 150`. Then reconstruct with only `|n| ≤ N` for `N ∈ {2, 5, 15, 50, 150}` by **masking**, not recomputing, and observe increasing shape fidelity.

```python
t, z = load_svg_path("svgs/star.svg", num_points=1000)
fs = FourierEpicycles(t, z, n_harmonics=150)
fs.calculate_all_coefficients()

t_dense = np.linspace(0, fs.T, 2000, endpoint=False)
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, N in zip(axes, [2, 5, 15, 50, 150]):
    z_hat = sum(fs.coeffs[n] * np.exp(1j*n*fs.omega*t_dense) for n in range(-N, N+1))
    ax.plot(z.real, z.imag, color='0.7')
    ax.plot(z_hat.real, z_hat.imag, color='crimson')
    ax.set_title(f"N={N}"); ax.axis('equal'); ax.axis('off')
plt.show()
```

**Interpretation:** small `N` gives a blurry/rounded silhouette (low frequencies = coarse shape); the star's points (high-frequency detail) only appear once enough high harmonics are included — directly parallel to Task 2's low-pass-vs-high-pass image behavior.

---

## Problem 7 (Medium) — Verify Parseval's theorem for the Fourier Series

**Concept:** Energy conservation between time domain and frequency domain.

**Theory:** For a periodic `f(t)` with FS coefficients `cn`:
```
(1/T) * ∫[0,T] |f(t)|² dt  =  Σ (n=-∞ to ∞) |cn|²
```
i.e. the average power of the signal equals the sum of the power carried by each harmonic. This is the discrete FS version of "total signal energy is preserved by the transform."

**Task:** Compute both sides for the heart shape with `N = 150` and check they roughly agree.

```python
t, z = load_svg_path("svgs/heart.svg", num_points=1000)
fs = FourierEpicycles(t, z, n_harmonics=150)
fs.calculate_all_coefficients()

lhs = (1 / fs.T) * np.trapezoid(np.abs(z) ** 2, t)
rhs = sum(abs(c) ** 2 for c in fs.coeffs.values())

print("LHS (time-domain average power):", lhs)
print("RHS (sum of |c_n|^2):", rhs)
print("Relative error:", abs(lhs - rhs) / lhs)
```

**Interpretation:** the two sides should match closely (small relative error, mostly from truncating `n` at ±150 instead of ±∞, plus trapezoidal-integration error) — confirming no "energy" is lost or invented by your `calculate_cn`.

---

## Problem 8 (Medium) — Symmetry of the 2D CFT for a real image

**Concept:** Real-signal conjugate symmetry, extended to 2D.

**Theory:** For any **real-valued** `I(x,y)`, its CFT satisfies `F(-u,-v) = conj(F(u,v))`. In terms of your `real`/`imag` arrays (indexed on a grid centered at `(cx, cy)`), this means `real[cx+i, cy+j] ≈ real[cx-i, cy-j]` and `imag[cx+i, cy+j] ≈ -imag[cx-i, cy-j]`.

**Task:** Compute `compute_cft()` on `pikachu.png` and numerically verify this symmetry.

```python
img = ContinuousImage("pikachu.png")
cft = CFT2D(img)
real, imag = cft.compute_cft()

rows, cols = real.shape
cx, cy = rows // 2, cols // 2

real_flipped = np.flip(real)
imag_flipped = np.flip(imag)

real_err = np.mean((real - real_flipped) ** 2)
imag_err = np.mean((imag + imag_flipped) ** 2)   # note the + , since imag should be antisymmetric
print("Real-part symmetry MSE:", real_err)
print("Imag-part antisymmetry MSE:", imag_err)
```

**Interpretation:** small errors on both confirm conjugate symmetry — a useful sanity check that catches bugs in how `u`/`v` are built or indexed (an off-by-one in the frequency grid often shows up here first).

---

## Problem 9 (Medium) — Build your own low-pass filter (denoising / blurring), the mirror image of `high_pass`

**Concept:** The complement of Task 2's given filter — instead of keeping only high frequencies (edges), keep only low frequencies (smooth content) to blur/denoise.

**Theory:** Where `high_pass` zeroes `F(u,v)` **inside** radius `cutoff`, a low-pass filter zeroes `F(u,v)` **outside** radius `cutoff`. This suppresses fast-varying detail (noise, texture, edges) and keeps only slow brightness variation — blurring the image, exactly the opposite effect of Problem 4/edge detection.

```python
def low_pass(real, imag, cutoff):
    rows, cols = real.shape
    cx, cy = rows // 2, cols // 2
    real_f, imag_f = real.copy(), imag.copy()
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - cx) ** 2 + (j - cy) ** 2) > cutoff:
                real_f[i, j] = 0
                imag_f[i, j] = 0
    return real_f, imag_f

img = ContinuousImage("pikachu.png")
cft = CFT2D(img)
real, imag = cft.compute_cft()

real_f, imag_f = low_pass(real, imag, cutoff=15)
blurred = InverseCFT2D(real_f, imag_f, cft.u, cft.v, img.x, img.y).reconstruct()

plt.imshow(blurred, cmap='gray'); plt.title("Low-pass (blurred) result"); plt.axis('off')
plt.show()
```

**Interpretation:** the output looks like a smoothed/blurred version of Pikachu — confirming that low frequencies alone carry an image's overall shape and shading, while high frequencies carry the sharp detail (edges, fur texture).

---

## Problem 10 (Medium) — Band-pass filtering: isolate a ring of frequencies

**Concept:** Combining a low-pass and a high-pass filter (i.e. "keep only a ring of frequencies") to see what a *specific band* of spatial frequency looks like.

**Theory:** A band-pass filter keeps `F(u,v)` only where `r_in ≤ √(u²+v²) ≤ r_out`, discarding both the very-low (flat shading) and very-high (fine noise/texture) frequencies. It isolates "medium-scale" structures — typically the coarse outlines of features rather than fine texture or flat backgrounds.

```python
def band_pass(real, imag, r_in, r_out):
    rows, cols = real.shape
    cx, cy = rows // 2, cols // 2
    real_f, imag_f = real.copy(), imag.copy()
    for i in range(rows):
        for j in range(cols):
            r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
            if not (r_in <= r <= r_out):
                real_f[i, j] = 0
                imag_f[i, j] = 0
    return real_f, imag_f

img = ContinuousImage("pikachu.png")
cft = CFT2D(img)
real, imag = cft.compute_cft()

real_f, imag_f = band_pass(real, imag, r_in=5, r_out=25)
band = InverseCFT2D(real_f, imag_f, cft.u, cft.v, img.x, img.y).reconstruct()

plt.imshow(np.abs(band), cmap='gray'); plt.title("Band-pass (r=5..25)"); plt.axis('off')
plt.show()
```

**Interpretation:** compare this to the pure high-pass edge map (Problem 4) — band-pass typically shows the major outlines cleanly without the faint fine-texture ringing that a pure high-pass sometimes leaves in.

---

## Problem 11 (Hard) — Modulation property on the epicycle signal (frequency-shifting a drawing)

**Concept:** FS modulation — multiplying the sampled signal by a rotating exponential before computing coefficients shifts every harmonic index by a constant integer `k`.

**Theory:** If `f(t) ↔ cn`, then for integer `k`:
```
g(t) = f(t) * e^(j k ω t)   ↔   c'n = c(n-k)
```
i.e. the whole coefficient spectrum shifts by `k` positions. Since `e^(jkωt)` with integer `k` is itself periodic with the same period `T`, `g(t)` is still a valid periodic signal you can feed straight back into `FourierEpicycles`.

**Task:** Modulate the heart signal by `k = 3`, recompute coefficients independently, and confirm `c'n ≈ c(n-k)`.

```python
t, z = load_svg_path("svgs/heart.svg", num_points=1000)
fs = FourierEpicycles(t, z, n_harmonics=30)
fs.calculate_all_coefficients()

k = 3
omega = fs.omega
z_mod = z * np.exp(1j * k * omega * t)
fs_mod = FourierEpicycles(t, z_mod, n_harmonics=30)
fs_mod.calculate_all_coefficients()

errs = []
for n in range(-25, 26):           # keep within range so (n-k) is also computed
    predicted = fs.coeffs.get(n - k, 0)
    actual = fs_mod.coeffs.get(n, 0)
    errs.append(abs(predicted - actual) ** 2)

print("Modulation-property MSE:", np.mean(errs))
```

**Interpretation:** low MSE confirms the shift theoretically predicted; visually, `g(t)`'s epicycle drawing traces the *same shape* as `f(t)` (multiplying by a unit-magnitude rotating exponential doesn't change `|g(t)|` or the geometric path — only reindexes which epicycle spins at which speed).

---

## Problem 12 (Hard) — Connect FS coefficients to the CFT of one period

**Concept:** The Fourier Series is a *sampled* version of the Continuous Fourier Transform, evaluated only at the harmonic frequencies.

**Theory:** If you take one period of `f(t)` (from `0` to `T`) and treat it as a finite-duration aperiodic signal, its CFT is
```
X(f) = ∫[0,T] f(t) e^(-j2πft) dt
```
Comparing this to `cn = (1/T) ∫[0,T] f(t) e^(-jnωt) dt` with `ω = 2π/T` — note `nωt = 2π(n/T)t` — you get exactly:
```
cn = (1/T) * X(f = n/T)
```
The FS coefficients are just the one-period CFT, **sampled** at integer multiples of `1/T`.

**Task:** Implement a small direct-integration CFT (no `np.fft`, matching assignment rules) and verify this identity numerically for the heart shape.

```python
def cft_at_f(t, signal, f):
    kernel = np.exp(-1j * 2 * np.pi * f * t)
    return np.trapezoid(signal * kernel, t)

t, z = load_svg_path("svgs/heart.svg", num_points=1000)
fs = FourierEpicycles(t, z, n_harmonics=20)
fs.calculate_all_coefficients()

T = fs.T
errs = []
for n in range(-20, 21):
    X_over_T = cft_at_f(t, z, n / T) / T
    errs.append(abs(fs.coeffs[n] - X_over_T) ** 2)

print("cn vs (1/T)X(n/T) MSE:", np.mean(errs))
```

**Interpretation:** very small MSE confirms the FS-as-sampled-CFT relationship — a deep conceptual bridge between the two halves of this assignment (Task 1 is really a special, periodic case of Task 2's continuous transform).

---

## Problem 13 (Hard) — Brute-force `O(N⁴)` check of your separable `O(N³)` implementation

**Concept:** Verifying the separability optimization (the assignment's core performance trick) is mathematically exact, not just fast.

**Theory:** The separable two-stage integration is algebraically identical to the direct double integral in Eq. (3)-(4) — it's a reordering of summation/integration, not an approximation. On a small enough grid you can afford to brute-force the direct `O(N⁴)` version and confirm both give the same answer.

```python
def brute_force_cft(I, x, y, u, v):
    """O(N^4) direct double integral -- only run on a small (e.g. 24x24) image!"""
    Nu, Nv = len(u), len(v)
    real = np.zeros((Nv, Nu))
    imag = np.zeros((Nv, Nu))
    for vi, vv in enumerate(v):
        for ui, uu in enumerate(u):
            integrand_re = np.zeros_like(I)
            integrand_im = np.zeros_like(I)
            for yi in range(len(y)):
                for xi in range(len(x)):
                    phase = 2*np.pi*(uu*x[xi] + vv*y[yi])
                    integrand_re[yi, xi] = I[yi, xi] * np.cos(phase)
                    integrand_im[yi, xi] = -I[yi, xi] * np.sin(phase)
            real[vi, ui] = np.trapezoid(np.trapezoid(integrand_re, x, axis=1), y)
            imag[vi, ui] = np.trapezoid(np.trapezoid(integrand_im, x, axis=1), y)
    return real, imag

# Use a tiny downsampled test image so O(N^4) finishes quickly
img = ContinuousImage("pikachu.png")
small = img.image[::20, ::20]           # e.g. down to ~24x24
x_small = img.x[::20]
y_small = img.y[::20]

class TinyImg:
    def __init__(self, image, x, y):
        self.image, self.x, self.y = image, x, y

cft_small = CFT2D(TinyImg(small, x_small, y_small))
real_fast, imag_fast = cft_small.compute_cft()
real_slow, imag_slow = brute_force_cft(small, x_small, y_small, cft_small.u, cft_small.v)

print("Real MSE (separable vs brute-force):", np.mean((real_fast - real_slow) ** 2))
print("Imag MSE (separable vs brute-force):", np.mean((imag_fast - imag_slow) ** 2))
```

**Interpretation:** near-zero MSE proves the separable trick your `compute_cft` uses is exact — the speedup from `O(N⁴)` to `O(N³)` costs nothing in accuracy.

---

## Problem 14 (Hard) — Directional edge detection: elliptical / anisotropic high-pass filter

**Concept:** Generalizing the given (isotropic, circular) `high_pass` filter to an **anisotropic** one, to detect edges in a preferred orientation.

**Theory:** The given `high_pass` zeroes a disk (equal cutoff in every direction), so it detects edges of *all* orientations equally. If instead you zero an **ellipse** — e.g. `(u/a)² + (v/b)² ≤ 1` with `a ≠ b` — you suppress low frequencies more aggressively along one axis than the other, which (because horizontal spatial frequency `u` corresponds to vertical edges, and vice versa) makes the result more sensitive to edges of a particular orientation.

```python
def elliptical_high_pass(real, imag, a, b):
    rows, cols = real.shape
    cx, cy = rows // 2, cols // 2
    real_f, imag_f = real.copy(), imag.copy()
    for i in range(rows):
        for j in range(cols):
            if ((j - cy) / a) ** 2 + ((i - cx) / b) ** 2 <= 1:
                real_f[i, j] = 0
                imag_f[i, j] = 0
    return real_f, imag_f

img = ContinuousImage("pikachu.png")
cft = CFT2D(img)
real, imag = cft.compute_cft()

# a small, b large -> suppress a wide vertical band of low horizontal frequency
# -> emphasizes vertical edges (fast change along x)
real_f, imag_f = elliptical_high_pass(real, imag, a=8, b=30)
vertical_edges = InverseCFT2D(real_f, imag_f, cft.u, cft.v, img.x, img.y).reconstruct()

plt.imshow(1 - np.abs(vertical_edges)/np.abs(vertical_edges).max(), cmap='gray')
plt.title("Anisotropic high-pass (vertical-edge emphasis)")
plt.axis('off'); plt.show()
```

**Interpretation:** comparing this against the isotropic result from Problem 4 shows edges running in one direction pop out more strongly, since the filter no longer treats every orientation symmetrically — this is the frequency-domain analogue of directional kernels like Sobel-x/Sobel-y, but derived purely from filtering `F(u,v)`, with no gradient/kernel computation anywhere.

---

## Problem 15 (Hard, capstone) — From pixels to epicycles: draw an image's edge map with Fourier Series

**Concept:** Fuse Task 2 (2D-CFT edge detection) and Task 1 (FS epicycles) into one pipeline: extract an edge/outline from a raster image using the CFT-based edge detector, turn the strongest contour into an ordered closed path, and redraw *that* path as an epicycle animation — exactly the same object your `FourierEpicycles` class already handles for hand-drawn SVGs.

**Theory:** `FourierEpicycles` only needs a closed, periodic, complex-valued path `z(t) = x(t) + jy(t)`. `svg_utils.load_svg_path` produces one from vector SVG data — but nothing in `FourierEpicycles` cares *where* the path came from. If we can extract an ordered boundary/contour from the CFT edge map (a raster image) and re-parametrize it by arc length the same way `svg_utils` does, it becomes a valid input to the exact same class.

```python
import numpy as np
from cft_edge_detector import ContinuousImage, CFT2D, FrequencyFilter, InverseCFT2D
from fs_redrawer import FourierEpicycles
from epicycle_animation import save_outputs
from skimage import measure   # only for contour extraction -- separate from the FT pipeline itself

# --- Step 1: get an edge map using YOUR Task 2 pipeline ---
img = ContinuousImage("pikachu.png")
cft = CFT2D(img)
real, imag = cft.compute_cft()
real_f, imag_f = FrequencyFilter().high_pass(real, imag, cutoff=15)
edges = InverseCFT2D(real_f, imag_f, cft.u, cft.v, img.x, img.y).reconstruct()
edge_map = np.abs(edges); edge_map = edge_map / edge_map.max()

# --- Step 2: extract the single longest closed contour from the edge map ---
contours = measure.find_contours(edge_map, level=0.3)
longest = max(contours, key=len)          # (row, col) points, in pixel index units

# --- Step 3: turn it into a centered, unit-scaled, arc-length-parametrized z(t),
#     exactly like svg_utils.load_svg_path does for SVGs ---
y_px, x_px = longest[:, 0], longest[:, 1]
x_c = x_px - x_px.mean()
y_c = -(y_px - y_px.mean())               # flip, same convention as svg_utils
scale = max(np.abs(x_c).max(), np.abs(y_c).max())
x_c, y_c = x_c / scale, y_c / scale

seg_len = np.hypot(np.diff(x_c, append=x_c[0]), np.diff(y_c, append=y_c[0]))
arc = np.concatenate(([0.0], np.cumsum(seg_len)))
total_len = arc[-1]

num_points = 1000
t = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
target_arc = t / (2 * np.pi) * total_len
x_r = np.interp(target_arc, arc, np.append(x_c, x_c[0]))
y_r = np.interp(target_arc, arc, np.append(y_c, y_c[0]))
z = x_r + 1j * y_r
z[-1] = z[0]

# --- Step 4: feed straight into YOUR Task 1 class, unmodified ---
fs = FourierEpicycles(t, z, n_harmonics=150)
fs.calculate_all_coefficients()
save_outputs(fs, z, "pikachu_edge_comparison.png", "pikachu_edge_epicycles.gif")
```

**Interpretation:** this is the strongest possible test that both parts of your assignment are *correct implementations of the same underlying mathematical object* (a closed, complex, periodic signal) rather than two unrelated pieces of code — the CFT edge detector supplies the path, and the exact same `FourierEpicycles` class that draws a heart or a star now draws Pikachu's outline with rotating vectors.

---

# Summary Table

| # | Difficulty | Builds on | Property tested |
|---|---|---|---|
| 1 | Easy | Task 1 | FS convergence vs. N |
| 2 | Easy | Task 1 | `cn` correctness on a known shape |
| 3 | Easy | Task 2 | Forward+inverse round trip |
| 4 | Easy | Task 2 | Effect of `cutoff` |
| 5 | Medium | Task 1 | Time-shift property |
| 6 | Medium | Task 1 | Harmonic truncation = low-pass filter |
| 7 | Medium | Task 1 | Parseval's theorem |
| 8 | Medium | Task 2 | Conjugate symmetry of real-image CFT |
| 9 | Medium | Task 2 | Custom low-pass filter (blur) |
| 10 | Medium | Task 2 | Custom band-pass filter |
| 11 | Hard | Task 1 | Modulation property |
| 12 | Hard | Task 1 + 2 | FS coefficients = sampled CFT |
| 13 | Hard | Task 2 | Separable vs. brute-force correctness |
| 14 | Hard | Task 2 | Anisotropic (directional) filtering |
| 15 | Hard | Task 1 + 2 | End-to-end fusion of both pipelines |
