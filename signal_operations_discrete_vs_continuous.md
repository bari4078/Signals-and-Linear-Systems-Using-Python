# Signal Operations, Explained Simply — Discrete vs. Continuous

For every operation below you get: **what it means in plain English**, then the **discrete
version** (array of samples, index-based), then the **continuous version** (array of samples
taken from a smooth function, time-based). The two are the same *idea* — continuous just uses
real time values (`t`) instead of integer indices (`n`), so it needs interpolation for anything
that lands "between" samples.

---

## 0. The Two Signal Setups

**Discrete** — a signal `x[n]` stored as an array covering `n = -8 ... 8`. Array index and time
index are related by `array_index = n + 8`.

```python
import numpy as np
import matplotlib.pyplot as plt

INF = 8
def init_signal():
    return np.zeros(2 * INF + 1)

signal = init_signal()
signal[INF] = 1        # x[0] = 1
signal[INF + 1] = 0.5  # x[1] = 0.5
signal[INF - 1] = 2    # x[-1] = 2
```

**Continuous** — a signal `x(t)` sampled very finely over a time range, stored as an array of
(t, value) pairs. Because it's *sampled*, "continuous" here just means "high-resolution":

```python
T_MIN, T_MAX, N = -4.0, 4.0, 4001
t = np.linspace(T_MIN, T_MAX, N)          # 4001 evenly-spaced time points

def x_of_t(t):
    return np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 1.5 * t)

x = x_of_t(t)
```

Keep both of these snippets handy — every function below plugs into one or the other.

---

## 1. Time Shift — `x[n - n0]` or `x(t - t0)`

**Plain English:** slide the whole signal left or right along the time axis. Shifting right
(delay) means things happen *later*; shifting left (advance) means things happen *earlier*.

### Discrete

Since we only have integer positions, shifting means physically moving values to new array
slots and filling the gap with zero.

```python
def time_shift_signal(x: np.ndarray, k: int) -> np.ndarray:
    """Returns x[n - k].  k > 0 shifts right (delay), k < 0 shifts left (advance)."""
    if k == 0:
        return x.copy()
    res = np.zeros_like(x)
    if k > 0:
        res[k:] = x[:-k]      # move everything k slots to the right
    else:
        res[:k] = x[-k:]      # move everything |k| slots to the left
    return res
```

```python
plot(time_shift_signal(signal, 2))    # x[n-2]: delayed by 2
plot(time_shift_signal(signal, -2))   # x[n+2]: advanced by 2
```

### Continuous

Since `t` is a real number, shifting is just: evaluate the *same function* at `t - t0` instead
of `t`. No array-juggling needed — just call your signal function with shifted time.

```python
def time_shift_continuous(t: np.ndarray, t0: float) -> np.ndarray:
    """Returns x(t - t0) by re-evaluating x_of_t at shifted times."""
    return x_of_t(t - t0)
```

```python
y = time_shift_continuous(t, 1.0)   # x(t - 1): delayed by 1 second
plt.plot(t, x_of_t(t), label="x(t)")
plt.plot(t, y, label="x(t-1)")
plt.legend()
plt.show()
```

> **Why the discrete version can't just do `x_of_t(t - t0)`:** discrete signals usually aren't
> given as a formula, only as sampled array values — so you must move data around inside the
> array instead of recomputing it.

---

## 2. Compression (Downscaling) — `x[kn]` or `x(kt)`

**Plain English:** speed the signal up / squeeze it. You keep only every `k`-th sample, so the
signal appears to play `k` times faster and looks "compressed" horizontally.

### Discrete

For every output time `n`, look up the input at position `k*n`. If `k*n` falls outside the
known range, that output is 0 (we don't know it).

```python
def time_scale_signal(x: np.ndarray, k: int) -> np.ndarray:
    """Returns x[k*n] — keeps only every k-th sample (compression)."""
    res = np.zeros_like(x)
    n = np.arange(-INF, INF + 1)
    valid_mask = (k * n >= -INF) & (k * n <= INF)
    res[n[valid_mask] + INF] = x[(k * n)[valid_mask] + INF]
    return res
```

```python
plot(time_scale_signal(signal, 2))   # x[2n]: compressed, plays twice as fast
```

### Continuous

Same idea, but since we have a real formula, compression is just: evaluate the function at
`k*t` instead of `t`.

```python
def compress_continuous(t: np.ndarray, k: float) -> np.ndarray:
    """Returns x(k*t) — compression when k > 1."""
    return x_of_t(k * t)
```

```python
y = compress_continuous(t, 2)   # x(2t): compressed
```

---

## 3. Stretching (Upsampling) — `x[n/k]` or `x(t/k)`

**Plain English:** slow the signal down / spread it out. New "in-between" time slots appear
that didn't exist before — you must decide what value to put there.

### Discrete — Option A: fill gaps with zero

```python
def time_scale_stretch(x: np.ndarray, k: int) -> np.ndarray:
    """Returns x[n/k], but only at integer n/k positions; everything else is 0."""
    res = np.zeros_like(x)
    for n in range(-INF, INF + 1):
        scaled_n = k * n
        if -INF <= scaled_n <= INF:
            res[n + INF] = x[scaled_n + INF]
    return res
```

### Discrete — Option B: fill gaps by interpolating (see Section 4)

This is the "smart" version — instead of zeros in the gaps, you average the two nearest known
samples. Covered fully in the next section.

### Continuous

Because a continuous signal has a real formula, stretching needs **no interpolation at all** —
just evaluate at `t/k`:

```python
def stretch_continuous(t: np.ndarray, k: float) -> np.ndarray:
    """Returns x(t/k) — stretching when k > 1."""
    return x_of_t(t / k)
```

```python
y = stretch_continuous(t, 2)   # x(t/2): stretched, plays half speed
```

> **Key takeaway:** for continuous signals given as a formula, compression/stretching is
> trivial (just change what you plug into the formula). Interpolation only becomes necessary
> when your continuous signal is itself just a big array of *pre-computed* samples and you need
> a value at a time point that wasn't originally sampled — see Section 4.2.

---

## 4. Interpolation — Filling In the Gaps

**Plain English:** when you need a signal's value at a point you don't have data for, estimate
it using the closest known neighbors. The simplest estimate is the **average of the sample
just before and the sample just after**.

### 4.1 Discrete interpolation (used inside stretching)

```python
def time_scale_interpolate(x: np.ndarray, k: int) -> np.ndarray:
    """Returns x[n/k], filling gaps with the average of the two nearest samples."""
    n = np.arange(-INF, INF + 1)
    m1 = np.floor(n / k).astype(int)   # nearest known sample to the left
    m2 = np.ceil(n / k).astype(int)    # nearest known sample to the right

    idx1 = np.clip(INF + m1, 0, 2 * INF)
    idx2 = np.clip(INF + m2, 0, 2 * INF)

    val1 = np.where((m1 >= -INF) & (m1 <= INF), x[idx1], 0.0)
    val2 = np.where((m2 >= -INF) & (m2 <= INF), x[idx2], 0.0)

    return 0.5 * (val1 + val2)    # if m1 == m2 (exact sample), this just returns that value
```

```python
plot(time_scale_interpolate(signal, 3))   # x[n/3], smoothly filled in
```

**Why floor and ceil?** `n/k` is often a fraction (e.g. `n/k = -1/3`). `floor` rounds it down to
the nearest whole sample below, `ceil` rounds it up to the nearest whole sample above. Average
those two and you have your estimate. If `n/k` is already a whole number, `floor` and `ceil`
give the *same* index, so the average is just that exact value (no error introduced).

### 4.2 Continuous interpolation (used when you only have discrete samples of a continuous
signal, and need a value at an arbitrary time)

```python
def interpolate_signal(t_original, x_original, t_query):
    """Estimate x(t_query) by averaging the two nearest sampled points."""
    dt = t_original[1] - t_original[0]                     # spacing between samples
    exact_indices = (t_query - t_original[0]) / dt          # how many steps in (can be fractional)
    left_indices = np.floor(exact_indices).astype(int)
    right_indices = np.ceil(exact_indices).astype(int)

    max_index = len(x_original) - 1
    left_indices = np.clip(left_indices, 0, max_index)
    right_indices = np.clip(right_indices, 0, max_index)

    return 0.5 * (x_original[left_indices] + x_original[right_indices])
```

Used for time sub-scaling when you must stay inside a fixed sample array instead of just
calling the original formula:

```python
def time_scale_via_interpolation(t, x, k):
    """Returns y(t) = x(t/k) using interpolation instead of recomputing x_of_t."""
    t_query = t / k
    return interpolate_signal(t, x, t_query)
```

> ⚠️ **Common mistake:** the return line must **add** the two neighbor values
> (`val1 + val2`) then multiply by 0.5. Using subtraction instead of addition is a frequent
> copy-paste bug that silently gives wrong answers.

---

## 5. Time Reversal — `x[-n]` or `x(-t)`

**Plain English:** play the signal backwards — flip it left-to-right around time 0.

### Discrete

Since index `n` and index `-n` sit at mirror positions around the center of the array,
reversal is just flipping the whole array.

```python
def time_reverse_signal(x: np.ndarray) -> np.ndarray:
    return x[::-1]
```

### Continuous

Two equivalent ways: flip the sample array (if you only have samples), or re-evaluate the
formula at `-t` (if you have the formula).

```python
def time_reverse(x: np.ndarray) -> np.ndarray:
    return x[::-1]                 # same trick, works because t is symmetric around 0

# or, if you have the formula:
x_reversed = x_of_t(-t)
```

Both give the same result **only when your `t` array is symmetric around 0** (e.g.
`linspace(-4, 4, N)`), because flipping the array then matches flipping the time values.

---

## 6. Odd/Even Decomposition — `xe[n]`, `xo[n]`

**Plain English:** any signal can be broken into two pieces —
- an **even** piece that looks identical when reversed (`xe[n] = xe[-n]`), and
- an **odd** piece that becomes its own negative when reversed (`xo[n] = -xo[-n]`).

Add them back together and you get the original signal exactly.

**Formulas** (identical for discrete and continuous — just swap `n` for `t`):

```
xe = 0.5 * (x + x_reversed)
xo = 0.5 * (x - x_reversed)
```

### Discrete

```python
def odd_even_decomposition(x: np.ndarray):
    xr = time_reverse_signal(x)     # this is x[-n]
    xe = 0.5 * (x + xr)             # even component
    xo = 0.5 * (x - xr)             # odd component
    return xo, xe
```

```python
odd_signal, even_signal = odd_even_decomposition(signal)
plot(odd_signal, title="Odd component")
plot(even_signal, title="Even component")
```

### Continuous

```python
def even_odd_decompose(x: np.ndarray):
    xr = time_reverse(x)            # this is x(-t)
    xe = 0.5 * (x + xr)
    xo = 0.5 * (x - xr)
    return xe, xo
```

```python
xe, xo = even_odd_decompose(x)
plt.plot(t, x,  label="x(t)")
plt.plot(t, xe, label="xe(t)")
plt.plot(t, xo, label="xo(t)")
plt.legend()
plt.show()
```

**Sanity check you can run yourself:** `xe + xo` should equal the original `x` (up to tiny
floating-point rounding) — if it doesn't, one of your signs is wrong.

---

## 7. Sinusoid Time Shift vs. Phase Change

**Plain English:** for a wave `x[n] = A·cos(Ω₀n + φ)`, delaying it in *time* and shifting its
*phase* can look identical — but only for a very specific phase value tied to that time shift.

```python
def sinusoid(n, A, Omega0, phi):
    return A * np.cos(Omega0 * n + phi)

def time_shift_sinusoid(n, A, Omega0, phi, n0):
    """Delays the wave by n0 samples."""
    return A * np.cos(Omega0 * (n - n0) + phi)

def phase_change_sinusoid(n, A, Omega0, phi, phi0):
    """Adds phi0 to the wave's phase — no time delay, just an angle offset."""
    return A * np.cos(Omega0 * n + phi + phi0)
```

**The bridge between them:** a time delay of `n0` samples is *exactly* the same as a phase
change of:

```python
phi0_equiv = -Omega0 * n0
```

Verify it numerically with mean squared error (should come out ~0):

```python
def mse(a, b):
    return float(np.mean((a - b) ** 2))

x_time  = time_shift_sinusoid(n, A, Omega0, phi, n0=2)
x_phase = phase_change_sinusoid(n, A, Omega0, phi, phi0=-Omega0 * 2)
print(mse(x_time, x_phase))   # ~0.0 confirms they're the same signal
```

---

## 8. Side-by-Side Cheat Sheet

| Operation | Discrete formula | Continuous formula | Discrete needs... | Continuous needs... |
|---|---|---|---|---|
| Time shift | `x[n-k]` | `x(t-t0)` | move array slots, zero-fill | re-evaluate at `t-t0` |
| Compress | `x[kn]` | `x(kt)` | keep every k-th sample | re-evaluate at `k*t` |
| Stretch | `x[n/k]` | `x(t/k)` | interpolate the gaps | re-evaluate at `t/k` (or interpolate if only samples exist) |
| Reverse | `x[-n]` | `x(-t)` | `x[::-1]` | `x[::-1]` (needs symmetric t-range) |
| Even/Odd | `0.5*(x±x[-n])` | `0.5*(x±x(-t))` | reverse then add/subtract | reverse then add/subtract |
| Interpolation | average of `floor(n/k)` and `ceil(n/k)` samples | average of nearest samples around `t_query` | `np.floor`/`np.ceil` + index math | same, but using time spacing `dt` |

**The one-sentence version:** discrete signals live in a fixed array, so every operation is
really "figure out which array slot to read from / write to" — continuous signals live as a
formula (or dense samples), so most operations are just "plug a different time value into the
same function," and interpolation only shows up when you're stuck working from samples alone.
