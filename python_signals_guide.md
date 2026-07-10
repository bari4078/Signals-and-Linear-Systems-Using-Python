# Python for Signals Labs — Everything You Need

This guide assumes **zero prior Python knowledge**. It covers the exact NumPy and Matplotlib
tools you need for discrete/continuous signal assessments, and walks through every function
from your lab specs (time shift, time scale, interpolation, time reversal, even/odd
decomposition, sinusoids) with working code.

---

## 1. The Basics You Actually Need

### 1.1 What is a NumPy array?
A NumPy array is just a list of numbers that Python can do math on *all at once* (no loops
needed). You always start a file with:

```python
import numpy as np
import matplotlib.pyplot as plt
```

### 1.2 Creating arrays

```python
np.zeros(5)                # [0. 0. 0. 0. 0.]
np.zeros_like(x)           # array of zeros, same shape/dtype as x
np.arange(-8, 9)           # [-8 -7 -6 ... 0 ... 7 8]   (stop is EXCLUSIVE, so use 9 to include 8)
np.linspace(-4, 4, 4001)   # 4001 evenly spaced points from -4 to 4 (inclusive both ends)
```

- `arange(start, stop, step)` — like counting; `stop` is never included.
- `linspace(start, stop, num)` — like ruler marks; `num` points, both ends included. Used for
  continuous-time signals sampled finely (e.g. `N = 4001` points).

### 1.3 Indexing & slicing (this is 90% of what you need)

```python
x[0]        # first element
x[-1]       # last element
x[2:5]      # elements at index 2,3,4 (stop excluded)
x[:5]       # first 5 elements
x[5:]       # everything from index 5 onward
x[::-1]     # REVERSE the whole array
x[2:]       # drop the first 2 elements
x[:-2]      # drop the last 2 elements
```

You can also **assign** to a slice:

```python
res = np.zeros_like(x)
res[2:] = x[:-2]     # shift stuff into res starting at index 2
```

### 1.4 Vectorized math (no loops!)

```python
x + y          # element-wise add
x - y
2 * x
np.cos(x)      # cosine of every element
np.abs(x)      # absolute value of every element
np.floor(x)    # round DOWN, e.g. floor(1.7) = 1
np.ceil(x)     # round UP,   e.g. ceil(1.2) = 2
```

### 1.5 Boolean masks — the key NumPy trick

You can compare an array to get `True`/`False` for every element, then use that to select or
filter:

```python
n = np.arange(-8, 9)
mask = (n >= -3) & (n <= 3)     # True where n is between -3 and 3
n[mask]                         # only the values that satisfy the mask
```

`&` = "and", `|` = "or" (use these, NOT `and`/`or`, with arrays).

### 1.6 `np.where` — vectorized if/else

```python
np.where(condition, value_if_true, value_if_false)
```

Example: give 0 wherever an index is out of range, otherwise take the real value:

```python
val = np.where((m >= -INF) & (m <= INF), x[idx], 0.0)
```

### 1.7 `np.clip` — force indices to stay in range (avoids crashes)

```python
np.clip(indices, 0, max_index)   # anything below 0 becomes 0, above max becomes max
```

This is essential when you compute an index like `n + INF` that might accidentally fall
outside the array bounds.

### 1.8 Type casting

```python
m.astype(int)     # convert floats to integers (needed before using something as an index)
```

---

## 2. Matplotlib — Plotting Syntax

### 2.1 Discrete signals → use `stem` (lollipop plot, since x[n] only exists at integers)

```python
plt.figure(figsize=(8, 3))
plt.stem(np.arange(-8, 9), signal)      # x-axis values, y-axis values
plt.xticks(np.arange(-8, 9, 1))         # show every integer tick
plt.ylim(-1, 4)                         # y-axis range
plt.title("x[n]")
plt.xlabel("n (Time Index)")
plt.ylabel("x[n]")
plt.grid(True)
plt.savefig("x[n].png")                 # save to file (optional)
plt.show()                              # display the plot
```

### 2.2 Continuous signals → use `plot` (smooth line)

```python
plt.figure()
plt.plot(t, x, label="x(t)")
plt.plot(t, y, label="y(t)", color="red", linestyle="--", linewidth=1.5)
plt.title("My Signal")
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()          # shows the labels you gave each line
plt.show()
```

### 2.3 Multiple subplots (side-by-side or stacked figures)

```python
fig, ax = plt.subplots(figsize=(9, 4))
markerline, stemlines, baseline = ax.stem(n, x, label="original")
baseline.set_visible(False)   # hides the horizontal line at y=0, cleaner look
ax.legend()
fig.tight_layout()
```

Cheat sheet of common calls:

| Call | Purpose |
|---|---|
| `plt.figure(figsize=(w,h))` | start a new figure with given size |
| `plt.plot(x, y)` | line plot (continuous signals) |
| `plt.stem(x, y)` | stem/lollipop plot (discrete signals) |
| `plt.title(...)` | set title |
| `plt.xlabel/ylabel(...)` | axis labels |
| `plt.legend()` | show legend (needs `label=` in each plot call) |
| `plt.grid(True)` | add gridlines |
| `plt.xticks(values)` | control which x tick marks are shown |
| `plt.ylim(low, high)` | set y-axis range |
| `plt.savefig("name.png")` | save figure to a file |
| `plt.show()` | display all figures |

---

## 3. How Discrete Signals Are Represented

Every assessment uses the same trick: a signal x[n] that "really" runs from -∞ to ∞ is stored
in a finite array covering just `n = -8` to `n = 8` (17 values). Anything outside that range is
treated as 0.

```python
INF = 8
def init_signal():
    return np.zeros(2 * INF + 1)     # 17 zeros
```

**The index mapping is the single most important idea:** array index `i` corresponds to time
index `n = i - INF`, so time index `n` lives at array index `n + INF`.

```python
signal = init_signal()
signal[INF] = 1          # this sets x[0] = 1
signal[INF + 1] = 0.5    # this sets x[1] = 0.5
signal[INF - 2] = 0.5    # this sets x[-2] = 0.5
```

Keep this mapping (`array_index = n + INF`) in mind for every function below — it's how you
convert between "the math index n" and "the actual Python array position."

---

## 4. Time Shift: `x[n - k]`

**Idea:** shifting right by `k` means every value moves `k` slots to the right (and zeros fill
in on the side it moved away from). Shifting by a negative `k` moves left.

```python
def time_shift_signal(x: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return x.copy()
    res = np.zeros_like(x)
    if k > 0:
        res[k:] = x[:-k]     # push values right, drop the last k, zero-fill the front
    else:
        res[:k] = x[-k:]     # push values left,  drop the first |k|, zero-fill the back
    return res
```

Example from the lab (shift `x` by 2 → `x[n-2]`, shift by -2 → `x[n+2]`):

```python
time_shift_signal(signal, 2)    # x[n-2]  (shifted right)
time_shift_signal(signal, -2)   # x[n+2]  (shifted left)
time_shift_signal(signal, 0)    # unchanged
```

---

## 5. Time Scale (downsampling): `x[kn]`

**Idea:** you only keep samples of `x` at positions that are multiples of `k`. For every valid
time index `n`, look up `x` at position `k*n`.

```python
def time_scale_signal(x: np.ndarray, k: int) -> np.ndarray:
    res = np.zeros_like(x)
    n = np.arange(-INF, INF + 1)
    valid_mask = (k * n >= -INF) & (k * n <= INF)   # keep only in-range lookups
    result_indices = n[valid_mask] + INF             # convert n -> array index
    x_indices = (k * n)[valid_mask] + INF            # convert k*n -> array index
    res[result_indices] = x[x_indices]
    return res
```

Equivalent explicit-loop version (easier to read first, but no bonus marks since it's not
vectorized):

```python
def time_scale_signal_loop(x: np.ndarray, k: int) -> np.ndarray:
    res = np.zeros_like(x)
    for n in range(-INF, INF + 1):
        scaled_n = k * n
        if -INF <= scaled_n <= INF:
            res[n + INF] = x[scaled_n + INF]
    return res
```

Example: `time_scale_signal(signal, 2)` gives `x[2n]` (compresses the signal — you see every
other original sample).

---

## 6. Time Scale (upsampling / stretching): `x[n/k]`

**Idea:** the opposite direction — you're stretching the signal, so new "in-between" time
slots appear. Task asks for two versions:

### 6.1 Plain version — fill in-between slots with 0

```python
def time_scale_signal_stretch(x: np.ndarray, k: int, downsample: bool = True) -> np.ndarray:
    res = np.zeros_like(x)
    if k == 0:
        return res
    for n in range(-INF, INF + 1):
        scaled_n = k * n
        if -INF <= scaled_n <= INF:
            res[n + INF] = x[scaled_n + INF]
    return res
```

### 6.2 With interpolation — fill in-between slots with the average of neighbors

**Idea:** for `x[n/k]`, when `n/k` isn't a whole number, `floor(n/k)` and `ceil(n/k)` give you
the two original samples surrounding the missing spot; average them.

```python
def time_scale_interpolate(x: np.ndarray, k: int) -> np.ndarray:
    n = np.arange(-INF, INF + 1)
    m1 = np.floor(n / k).astype(int)     # neighbor to the left
    m2 = np.ceil(n / k).astype(int)      # neighbor to the right

    idx1 = np.clip(INF + m1, 0, 2 * INF)   # convert to array index, keep in bounds
    idx2 = np.clip(INF + m2, 0, 2 * INF)

    val1 = np.where((m1 >= -INF) & (m1 <= INF), x[idx1], 0.0)
    val2 = np.where((m2 >= -INF) & (m2 <= INF), x[idx2], 0.0)

    return 0.5 * (val1 + val2)   # if m1 == m2 (exact sample), this just returns that value
```

Example: `-1` maps to `n/k = -1/3`, whose neighbors are `floor = -1`... conceptually the
missing sample at `-1` becomes the average of the two nearest known samples of `x`.

---

## 7. Time Reversal: `x[-n]`

**Idea:** reversing the array reverses time, because index `n` and index `-n` are mirror
positions around the center (`INF`).

```python
def time_reverse_signal(x: np.ndarray) -> np.ndarray:
    return x[::-1]
```

That's it — `[::-1]` reverses any array. Reversing twice gets you back the original:
`time_reverse_signal(time_reverse_signal(x))` equals `x`.

---

## 8. Odd/Even Decomposition

**Formulas:** any signal can be split into an even part `xe[n] = 0.5*(x[n] + x[-n])` and an odd
part `xo[n] = 0.5*(x[n] - x[-n])`. Note `xe[n] = xe[-n]` (mirror-symmetric) and
`xo[n] = -xo[-n]` (anti-symmetric) — that's literally what "even" and "odd" mean here.

```python
def odd_even_decomposition(x: np.ndarray):
    xr = time_reverse_signal(x)   # this is x[-n]
    xe = 0.5 * (x + xr)           # even component
    xo = 0.5 * (x - xr)           # odd component
    return xo, xe                # returning 2 values from a function — see note below
```

**Note on returning multiple values:** in Python, `return a, b` returns both at once, and you
unpack them like this:

```python
odd_signal, even_signal = odd_even_decomposition(signal)
```

---

## 9. Continuous-Time Version: Interpolation & Sub-scaling

When the signal is continuous (`x(t)` sampled very finely, e.g. 4001 points from -4 to 4
seconds) rather than a `[-8, 8]` integer array, the same shifting/scaling ideas apply but you
work with a time array `t` instead of integer indices.

```python
T_MIN, T_MAX, N = -4.0, 4.0, 4001
t = np.linspace(T_MIN, T_MAX, N)

def x_of_t(t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 1.5 * t)
```

### 9.1 Interpolating at arbitrary query times

**Idea:** given a desired time `t_query` that falls *between* two known samples, find the
nearest sample on each side and average them.

```python
def interpolate_signal(t_original, x_original, t_query):
    dt = t_original[1] - t_original[0]                        # spacing between samples
    exact_indices = (t_query - t_original[0]) / dt            # "how many steps in" (may be fractional)
    left_indices = np.floor(exact_indices).astype(int)
    right_indices = np.ceil(exact_indices).astype(int)

    max_index = len(x_original) - 1
    left_indices = np.clip(left_indices, 0, max_index)
    right_indices = np.clip(right_indices, 0, max_index)

    return 0.5 * (x_original[left_indices] + x_original[right_indices])
```

> ⚠️ A common bug: writing `0.5 * (x_original[left] - x_original[right])` — that's a
> **subtraction** and gives the wrong answer. It must be a **sum**, since you're averaging.

### 9.2 Time sub-scaling: `y(t) = x(t/k)`

```python
def time_scale(t, x, k):
    t_query = t / k
    return interpolate_signal(t, x, t_query)
```

### 9.3 Plotting original vs. transformed signal together

```python
def plot_pair(t, x, y, title):
    plt.figure(figsize=(10, 6))
    plt.plot(t, x, label="Original Signal x(t)", color="blue", linewidth=1.5)
    plt.plot(t, y, label="Sub-scaled Signal y(t)", color="red", linestyle="--", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Time (t)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

k = 2
y = time_scale(t, x_of_t(t), k)
plot_pair(t, x_of_t(t), y, title=f"Time Sub-scaling: y(t) = x(t/{k})")
plt.show()
```

### 9.4 Time reversal & even/odd for continuous signals (same idea, array flip)

```python
def time_reverse(x: np.ndarray) -> np.ndarray:
    return x[::-1]     # works the same way as the discrete case

def even_odd_decompose(x: np.ndarray):
    xr = time_reverse(x)
    xe = 0.5 * (x + xr)
    xo = 0.5 * (x - xr)
    return xe, xo
```

---

## 10. Discrete-Time Sinusoids: Time Shift vs. Phase Change

Signal: `x[n] = A * cos(Ω₀n + φ)`

```python
def sinusoid(n, A, Omega0, phi):
    return A * np.cos(Omega0 * n + phi)

def time_shift_sinusoid(n, A, Omega0, phi, n0):
    # shifting time by n0 replaces n with (n - n0)
    return A * np.cos(Omega0 * (n - n0) + phi)

def phase_change_sinusoid(n, A, Omega0, phi, phi0):
    # phase change just adds an extra angle offset
    return A * np.cos(Omega0 * n + phi + phi0)
```

**Key relationship:** a time shift of `n0` samples is mathematically identical to a phase
change of `phi0 = -Omega0 * n0`. That's why:

```python
n0 = 2
phi0_equiv = -Omega0 * n0     # the phase change that exactly reproduces the time shift
```

**Comparing two signals numerically — mean squared error (MSE):**

```python
def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))
```

MSE close to 0 means the two signals are essentially identical — this is how you *prove*
(rather than eyeball) that a time shift matches a phase change.

**Searching for the best matching integer shift for an arbitrary phase change:**

```python
k_min, k_max = -12, 12
best_k, best_err = None, None
for k in range(k_min, k_max + 1):
    x_time_k = time_shift_sinusoid(n, A, Omega0, phi, k)
    e = mse(x_time_k, x_phase)
    if best_err is None or e < best_err:
        best_err = e
        best_k = k
```

This is one of the few places a plain Python `for` loop is fine — you're looping over
*candidate integers* to test, not over array elements.

---

## 11. Quick Debugging Checklist

- **`IndexError`** → you probably forgot to `+ INF` (or the equivalent offset) when converting
  a time index to an array index, or an index went out of bounds — use `np.clip`.
- **Wrong shift direction** → remember `x[n-k]` with positive `k` shifts the signal to the
  *right* (delays it); double check the sign of `k`.
- **All zeros in output** → check your boolean mask isn't accidentally excluding everything
  (e.g. comparing `n` when you meant `k*n`).
- **`and`/`or` errors on arrays** → use `&`/`|` for element-wise boolean operations on NumPy
  arrays, not the Python keywords `and`/`or`.
- **Off-by-one in `arange`** → `np.arange(-8, 8)` stops at 7, not 8. Use `np.arange(-8, 9)` to
  include 8.
- **Interpolation looks wrong** → check you're **adding** the two neighbor values then
  multiplying by 0.5, not subtracting.

---

## 12. Minimal Template You Can Reuse

```python
import numpy as np
import matplotlib.pyplot as plt

INF = 8

def init_signal():
    return np.zeros(2 * INF + 1)

def plot(signal, title=None, y_range=(-1, 3), figsize=(8, 3),
         x_label='n (Time Index)', y_label='x[n]', saveTo=None):
    plt.figure(figsize=figsize)
    plt.xticks(np.arange(-INF, INF + 1, 1))
    y_range = (y_range[0], max(np.max(signal), y_range[1]) + 1)
    plt.ylim(*y_range)
    plt.stem(np.arange(-INF, INF + 1, 1), signal)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if saveTo is not None:
        plt.savefig(saveTo)
    plt.show()

# --- build your test signal ---
signal = init_signal()
signal[INF] = 1
signal[INF + 1] = 0.5
signal[INF - 1] = 2
signal[INF + 2] = 1
signal[INF - 2] = 0.5

# --- call whichever function you just implemented ---
plot(signal, title='Original Signal (x[n])')
```

Copy this skeleton into any new assessment, then just implement the one or two functions the
spec asks for using the patterns above.
