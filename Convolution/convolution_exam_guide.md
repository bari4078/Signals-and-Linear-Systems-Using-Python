# CSE220 Convolution — Exam & Viva Survival Guide

Everything here is built and tested against your actual assignment spec and the
past-paper questions in your file. All numeric examples below were run in Python
and match the expected outputs exactly.

**Companion file:** `signal_lti_toolkit.py` — one file with every class/function
below, ready to paste. Grab whichever section you need.

---

## 0. How to use this in a time-restricted exam

Every past paper follows the same shape: *"you already built Signal/LTI in
Offline 1 — now bend it to do X."* So your winning strategy is always:

1. Figure out **what plays the role of x[n]** (the input signal).
2. Figure out **what plays the role of h[n]** (the impulse response) — this is
   almost always the "trick" of the question.
3. Build both as `DiscreteSignal` objects, run them through `LTISystem`, read
   off the output.

If you only remember one formula, remember this reframing of convolution:

> **y[n] = Σₘ h[m]·x[n − m]** — "the output now is a weighted mix of the
> current input and past inputs; h[m] is how much weight the input from
> *m steps ago* gets."

That single sentence solves the moving-average, weighted-average, and
exponential-smoothing style questions instantly (Section 5.1–5.2).

---

## 1. The core idea, explained simply

**A signal** `x[n]` is just a list of numbers indexed by an integer time `n`.
Outside the list, the value is 0.

**An impulse response** `h[n]` is the signal you'd get out of a system if you
fed it a single spike (1 at n=0, 0 everywhere else). It's the system's
"fingerprint" — for an LTI (linear, time-invariant) system, knowing `h[n]`
means you can predict the output for *any* input.

**Convolution** is the rule that combines them:

```
y[n] = Σ (over all k)  x[k] · h[n − k]
```

There are two equivalent ways to think about computing it — your assignment
required both, and they must give identical answers:

- **Superposition (method A):** Every nonzero sample `x[k]` "fires" a copy of
  `h[n]`, scaled by `x[k]` and shifted so it starts at `k`. Add all those
  copies together. This is literally "the system's response to each input
  spike, added up" — linearity in action.
- **Sliding sum (method B):** For each output time `n`, directly compute
  `Σ x[k]·h[n−k]` by walking over every nonzero input sample.

### Worked example by hand

`x = [1, 0, 2]` starting at n=0, `h = [1, 1]` starting at n=0.

**Superposition:** x[0]=1 fires `1·h[n]` at n=0,1. x[2]=2 fires `2·h[n−2]` at
n=2,3.
```
n:        0    1    2    3
1·h[n]:   1    1    0    0
2·h[n-2]: 0    0    2    2
sum:      1    1    2    2
```

**Sliding sum**, same answer, computed one `n` at a time:
```
y[0] = x[0]h[0]                     = 1·1             = 1
y[1] = x[0]h[1] + x[1]h[0]          = 1·1 + 0·1        = 1
y[2] = x[1]h[1] + x[2]h[0]          = 0·1 + 2·1        = 2
y[3] = x[2]h[1]                     = 2·1              = 2
```
Both give `y = [1, 1, 2, 2]`, starting at n=0. ✅ (Verified in Python too.)

### The output-range rule

If `x` lives on `[nx_min, nx_max]` and `h` lives on `[nh_min, nh_max]`, the
full output lives on:
```
[nx_min + nh_min,  nx_max + nh_max]
```
Full output length = `len(x) + len(h) − 1`.

**Important distinction for "window" problems** (moving average, smoothing):
the *full* convolution includes edge effects where the window hangs off the
end of the data. The *valid* region — where the window is completely inside
your data — has length `len(x) − len(h) + 1`. When a question asks for
"m − n + 1 outputs" (moving average, exponential smoothing questions), they
want the **valid** region only. See Section 5.1.

---

## 2. Convolution properties (good viva answers)

- **Commutative:** `x * h = h * x` — doesn't matter which one you call the
  "input" and which the "impulse response."
- **Associative:** `(x * h1) * h2 = x * (h1 * h2)` — this is *why* cascading
  two systems equals one system with a combined impulse response (Section 5.5).
- **Distributive over addition:** `x * (h1 + h2) = x*h1 + x*h2` — this is
  *why* parallel systems add their impulse responses (Section 5.5).
- **Shift:** if `y = x * h`, then shifting `x` by `k` shifts `y` by `k`.
- **Linearity of LTI systems:** scale the input → output scales the same way;
  add two inputs → outputs add. This is exactly what `SuperSignal` /
  `output_super` uses (Section 4).
- **Identity element:** convolving with `h[0]=1` (all else 0) returns `x`
  unchanged — a "do nothing" system.
- **Impulse response fully characterizes an LTI system** — this is the
  entire premise of the assignment.

---

## 3. Your toolkit: `DiscreteSignal` and `LTISystem`

This is the cleaned-up, **tested** version of your classes — copy this in
first, every time.

```python
class DiscreteSignal:
    """A finite signal x[n] living on integer times start_time..end_time.
    Anything outside that range is treated as 0."""

    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.values = [0.0] * (end_time - start_time + 1)

    def times(self):
        return range(self.start_time, self.end_time + 1)

    def get_value_at_time(self, t):
        if t < self.start_time or t > self.end_time:
            return 0.0
        return self.values[t - self.start_time]

    def set_value_at_time(self, t, value):
        if t < self.start_time or t > self.end_time:
            raise ValueError("Time out of range")
        self.values[t - self.start_time] = value

    def shift(self, k):
        """Returns x[n - k]. k > 0 slides right, k < 0 slides left."""
        shifted = DiscreteSignal(self.start_time + k, self.end_time + k)
        shifted.values = list(self.values)
        return shifted

    def add(self, other):
        """self + other, over the union of both time ranges."""
        new_start = min(self.start_time, other.start_time)
        new_end = max(self.end_time, other.end_time)
        res = DiscreteSignal(new_start, new_end)
        for t in res.times():
            res.set_value_at_time(t, self.get_value_at_time(t) + other.get_value_at_time(t))
        return res

    def multiply(self, scalar):
        """self * scalar (a number), same time range."""
        res = DiscreteSignal(self.start_time, self.end_time)
        for t in self.times():
            res.set_value_at_time(t, self.get_value_at_time(t) * scalar)
        return res

    def nonzero_samples(self, tolerance=1e-12):
        """[(t, value), ...] for every sample that isn't (basically) zero."""
        return [(t, v) for t in self.times() if abs(v := self.get_value_at_time(t)) > tolerance]

    def plot(self, title, save_path=None, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        t = list(self.times())
        ax.stem(t, self.values)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.35)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        return ax


def make_signal(start_time, end_time, values):
    """Quick constructor: make_signal(0, 2, [1, 0, 2]) -> x[0]=1, x[1]=0, x[2]=2."""
    s = DiscreteSignal(start_time, end_time)
    s.values = list(values)
    return s


class LTISystem:
    """A discrete-time LTI system, fully described by its impulse response h[n]."""

    def __init__(self, impulse_response):
        self.impulse_response = impulse_response

    def output_range(self, input_signal):
        start = input_signal.start_time + self.impulse_response.start_time
        end = input_signal.end_time + self.impulse_response.end_time
        return start, end

    # ---- Method A: superposition ----
    def get_response_components(self, input_signal):
        comps = []
        for k, x_k in input_signal.nonzero_samples():
            component = self.impulse_response.shift(k).multiply(x_k)
            comps.append((k, component))
        return comps

    def output_by_superposition(self, input_signal):
        start, end = self.output_range(input_signal)
        res = DiscreteSignal(start, end)
        for k, component in self.get_response_components(input_signal):
            res = res.add(component)
        return res

    # ---- Method B: direct sliding convolution sum ----
    def get_contributions_at_time(self, input_signal, n):
        contributions = []
        for k, x_k in input_signal.nonzero_samples():
            h_val = self.impulse_response.get_value_at_time(n - k)
            if abs(h_val) > 1e-12:
                contributions.append((k, x_k, h_val, x_k * h_val))
        return contributions

    def output_at_time(self, input_signal, n):
        return sum(c[3] for c in self.get_contributions_at_time(input_signal, n))

    def output(self, input_signal):
        start, end = self.output_range(input_signal)
        res = DiscreteSignal(start, end)
        for n in res.times():
            res.set_value_at_time(n, self.output_at_time(input_signal, n))
        return res
```

**Plain-English recap of every method**, since a viva may ask "what does
this line do?":

| Method | In one sentence |
|---|---|
| `get_value_at_time(t)` | Look up `x[t]`; return 0 if `t` is outside the stored range. |
| `set_value_at_time(t, v)` | Store a value at time `t`. |
| `shift(k)` | Slide the whole signal so it starts `k` steps later (delay). |
| `add(other)` | Add two signals sample-by-sample, even if their ranges differ. |
| `multiply(scalar)` | Scale every sample by a number. |
| `nonzero_samples()` | List the "interesting" (non-zero) samples — used so we don't waste time on zeros. |
| `output_range` | Where the convolution result can possibly be nonzero. |
| `get_response_components` | For each nonzero `x[k]`, build the shifted-and-scaled copy of `h` it contributes. |
| `output_by_superposition` | Add up all those copies = the full output (Method A). |
| `get_contributions_at_time(x, n)` | For one output time `n`, list every `x[k]·h[n−k]` term. |
| `output_at_time(x, n)` | Add up those terms = one output sample (Method B). |
| `output(x)` | Repeat `output_at_time` for every `n` in range = the full output (Method B). |

---

## 4. Python syntax you actually need here

- **Classes:** `self` is just "this particular object." `self.values = [...]`
  stores data on the object; `def method(self, arg):` — always list `self`
  first.
- **f-strings:** `f"{value:.2f}"` → 2 decimal places. `f"{n:4d}"` → integer
  padded to width 4.
- **List comprehension:** `[x*2 for x in values]`, with a filter:
  `[x for x in values if x > 0]`.
- **`range(a, b)`** stops *before* `b` — `range(0, 5)` gives `0,1,2,3,4`. Your
  `times()` method already does `range(start, end+1)` so it's inclusive.
- **`enumerate(lst)`** → `(index, value)` pairs. **`zip(a, b)`** → pairs items
  from two lists together.
- **Reading console input (online-exam style):**
  ```python
  prices = list(map(int, input("Stock Prices: ").split()))
  n = int(input("Window size: "))
  alpha = float(input("Alpha: "))
  ```
- **Reading a file (offline-style, two lines: range then values):**
  ```python
  with open(path) as f:
      start, end = map(int, f.readline().split())
      values = list(map(float, f.readline().split()))
  ```
- **NumPy you're allowed to use** (just not `np.convolve`):
  ```python
  import numpy as np
  arr = np.array(values, dtype=float)
  np.allclose(a, b, atol=1e-6)     # compare two lists/arrays for "close enough"
  np.zeros(5); np.arange(5)
  ```
- **Matplotlib stem plot:** `ax.stem(x_values, y_values)` — the classic
  "lollipop" plot for discrete signals.
- **Matplotlib image (for the grayscale color-block plot):**
  `ax.imshow(image_array, aspect="auto")`.
- **Walrus operator** (used above): `if abs(v := expr) > tol` assigns `v`
  *and* tests it in one line. Skip it if it confuses you — a plain two-line
  version works identically.

---

## 5. Problem playbook — likely exam variations

### 5.1 Moving average (unweighted & weighted)

**Concept:** a moving average of window size `n` is convolution with a
constant impulse response. Using the `y[n] = Σ h[m]x[n−m]` framing:
`h[0]` = weight on the *most recent* sample in the window.

- **Unweighted:** every sample in the window counts equally →
  `h[m] = 1/n` for `m = 0..n-1`.
- **Weighted** (most recent day gets the biggest weight): weights
  `n, n-1, ..., 1` normalized to sum to 1, with the biggest weight at `h[0]`
  (current sample) — because `h[0]` always multiplies the *current* input.

Only the **valid** region (window fully inside the data) is wanted, which has
`len(prices) - n + 1` values — matches the "m − n + 1 outputs" phrasing you'll
see in the spec.

```python
def unweighted_moving_average(prices, n):
    x = make_signal(0, len(prices) - 1, prices)
    h = make_signal(0, n - 1, [1.0 / n] * n)
    y = LTISystem(h).output(x)
    return [y.get_value_at_time(t) for t in range(n - 1, len(prices))]


def weighted_moving_average(prices, n):
    x = make_signal(0, len(prices) - 1, prices)
    total = n * (n + 1) / 2
    h_vals = [(n - m) / total for m in range(n)]   # h[0] = n/total (current sample, biggest)
    h = make_signal(0, n - 1, h_vals)
    y = LTISystem(h).output(x)
    return [y.get_value_at_time(t) for t in range(n - 1, len(prices))]
```

Tested against the spec's sample I/O: `unweighted_moving_average([1..8], 4)`
→ `[2.5, 3.5, 4.5, 5.5, 6.5]` ✅, `weighted_moving_average([1..8], 4)` →
`[3.0, 4.0, 5.0, 6.0, 7.0]` ✅.

### 5.2 Exponential smoothing

**Concept:** same idea, but the weights decay geometrically instead of
linearly: `h[m] = α·(1−α)^m` for `m = 0..n-1` (`h[0]=α` is the biggest,
applied to the current/most-recent sample).

```python
def exponential_smoothing(prices, n, alpha):
    x = make_signal(0, len(prices) - 1, prices)
    h_vals = [alpha * (1 - alpha) ** m for m in range(n)]
    h = make_signal(0, n - 1, h_vals)
    y = LTISystem(h).output(x)
    return [y.get_value_at_time(t) for t in range(n - 1, len(prices))]
```

Tested: `exponential_smoothing([10,11,12,9,10,13,15,16,17,18], 3, 0.8)` →
`[11.68, 9.47, 9.82, 12.29, 14.40, 15.62, 16.64, 17.63]` ✅ matches the spec
exactly.

### 5.3 Polynomial multiplication

**Concept:** multiplying two polynomials is *exactly* convolution of their
coefficient lists, provided the coefficients are ordered so that **index =
power of x** (ascending). Coefficients are usually given highest-power-first,
so reverse them first, convolve, then reverse the answer back.

```python
def poly_multiply(coeffs1_desc, coeffs2_desc):
    a_asc = list(reversed(coeffs1_desc))
    b_asc = list(reversed(coeffs2_desc))
    xa = make_signal(0, len(a_asc) - 1, a_asc)
    xb = make_signal(0, len(b_asc) - 1, b_asc)
    y = LTISystem(xb).output(xa)      # order doesn't matter (convolution is commutative)
    return list(reversed(y.values))    # back to highest-power-first
```

Tested: `poly_multiply([3,-2,1], [2,0,-3,1])` → `[6,-4,-7,9,-5,1]`, i.e.
`6x⁵−4x⁴−7x³+9x²−5x+1` ✅ — matches the spec's Case 1 exactly. Output degree
is `d1+d2`, i.e. `len(result) = len(poly1)+len(poly2)-1`.

### 5.4 Superposition of multiple signals (`SuperSignal`)

**Concept:** LTI systems are linear — scale an input, the output scales the
same way; add two inputs, the outputs add. So a combination like
`x(n) = 2·x1(n) − x2(n)` can be handled either by (a) building the combined
signal directly with `.multiply()`/`.add()`, or (b) running each component
through the system separately and combining the *outputs* the same way. Both
give identical results — this is exactly what the "verify" step in your
Offline assignment is checking, one level up.

```python
class SuperSignal:
    """Holds several (coefficient, DiscreteSignal) pairs, e.g. x = 2*x1 - x2."""
    def __init__(self):
        self.components = []

    def add(self, signal, coefficient=1.0):
        self.components.append((coefficient, signal))


def output_super(system, super_signal):
    """y = sum_i coefficient_i * system.output(signal_i)."""
    total = None
    for coeff, sig in super_signal.components:
        contribution = system.output(sig).multiply(coeff)
        total = contribution if total is None else total.add(contribution)
    return total
```

Usage:
```python
x1 = make_signal(0, 0, [1.0])          # x1[0] = 1
x2 = make_signal(2, 2, [1.0])          # x2[2] = 1
ss = SuperSignal()
ss.add(x1, 2.0)
ss.add(x2, -1.0)                        # represents x(n) = 2*x1(n) - x2(n)

h = make_signal(0, 1, [1.0, 0.5])
system = LTISystem(h)
y = output_super(system, ss)
```
Tested: matches building `x = x1.multiply(2.0).add(x2.multiply(-1.0))` and
calling `system.output(x)` directly, exactly ✅.

### 5.5 Combining LTI systems (series / parallel)

**Concept:** you don't need a picture memorized — you need two rules,
which follow directly from the convolution properties in Section 2:

- **Series / cascade** (signal passes through h1, *then* h2): equivalent to
  ONE system whose impulse response is `h1 * h2` (convolve them together).
- **Parallel** (signal passes through h1 and h2 independently, outputs
  summed): equivalent to ONE system whose impulse response is `h1 + h2`.

```python
# series: apply h1 then h2, block by block
y_block = LTISystem(h2).output(LTISystem(h1).output(x))

# series: the same thing with ONE combined impulse response
h_series = LTISystem(h1).output(h2)          # convolve h1 with h2
y_combined = LTISystem(h_series).output(x)
# y_block and y_combined come out identical

# parallel: apply h1 and h2 separately, add the outputs
y_parallel_block = LTISystem(h1).output(x).add(LTISystem(h2).output(x))

# parallel: the same thing with ONE combined impulse response
h_parallel = h1.add(h2)
y_parallel_combined = LTISystem(h_parallel).output(x)
```

Both tested and verified identical to the block-by-block version. If a
question shows a mixed diagram (some blocks in parallel, then in series with
another), just apply the two rules in whatever order the diagram shows —
convolution is associative, so it doesn't matter which piece you combine
first.

### 5.6 Step response ↔ impulse response

**Concept:** the step response `s[n]` is the system's output when you feed it
a "staircase" input (1 forever, starting at n=0) instead of a single spike.
Two useful identities:

- `h[n] = s[n] − s[n−1]` (recover the impulse response from the step
  response — a step is a running sum of impulses, so differencing undoes
  that).
- `y[n] = (Δx * s)[n]`, where `Δx[n] = x[n] − x[n−1]` — you can compute a
  system's output using *only* the step response, by first differencing the
  input.

```python
def first_difference(sig):
    """sig[n] - sig[n-1], using only shift/multiply/add (assumes sig[-1]=0)."""
    return sig.add(sig.shift(1).multiply(-1))

def impulse_from_step_response(step_response):
    return first_difference(step_response)          # h[n] = s[n] - s[n-1]

def output_using_step_response(x, step_response):
    dx = first_difference(x)                         # delta_x[n] = x[n] - x[n-1]
    return LTISystem(step_response).output(dx)        # (delta_x * s)
```

Tested: `output_using_step_response(x, s)` comes out **identical** to
`LTISystem(impulse_from_step_response(s)).output(x)` — i.e. `(Δx * s) = (x * h)`
✅, exactly the verification the past paper asks for.

---

## 6. Plotting cheat sheet

**Stem plot** (the standard "lollipop" plot for a discrete signal):
```python
def plot_signal(signal, title):
    fig, ax = plt.subplots()
    ax.stem(list(signal.times()), signal.values)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("n")
    plt.show()          # or plt.savefig("path.png")
```

**Grayscale color-block plot** (each value normalized to 0–255, shown as one
row of adjacent blocks):
```python
def plot_color_blocks(signal, title, ax):
    values = np.array(signal.values, dtype=float)
    lo, hi = values.min(), values.max()
    if np.isclose(lo, hi):
        gray = np.full(values.shape, 128, dtype=np.uint8)
    else:
        gray = np.round(255 * (values - lo) / (hi - lo)).astype(np.uint8)
    image = np.stack([gray, gray, gray], axis=-1).reshape(1, len(gray), 3)
    ax.imshow(image, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks(range(len(gray)))
    ax.set_xticklabels(list(signal.times()))
```

Three subplots stacked (x, h, y) or two (x, y) — just make one `Axes` per
signal:
```python
fig, axes = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
x.plot("Input x[n]", ax=axes[0])
h.plot("Impulse response h[n]", ax=axes[1])
y.plot("Output y[n]", ax=axes[2])
```

---

## 7. Common bugs / gotchas checklist

- **`set_value_at_time` raises an error if `t` is outside the signal's
  stored range.** Build the signal with the *correct* start/end first, don't
  try to write outside it.
- **Off-by-one on window ranges.** A window of size `n` starting at 0 covers
  indices `0` to `n-1` (i.e. `make_signal(0, n-1, ...)`), not `0` to `n`.
- **Valid vs full convolution.** If your output list is longer than expected
  by `n-1` extra values on each side, you forgot to slice to the "valid"
  region (Section 5.1).
- **Weight direction flipped.** If a weighted-average answer comes out
  backwards, you likely put the biggest weight at `h[n-1]` instead of `h[0]`
  — remember `h[0]` always multiplies the *current* sample.
- **`nonzero_samples()` tolerance.** Comparing floats with `==` after
  convolution can fail due to floating-point error; use `abs(a-b) < 1e-9` or
  `np.allclose(a, b, atol=1e-6)` instead.
- **Never use `numpy.convolve`, `scipy.signal.convolve`, or
  `scipy.signal.fftconvolve`** — explicitly banned everywhere in the spec,
  even for "just checking your answer."
- **Class/variable names may differ between papers** (`Signal` vs
  `DiscreteSignal`, `LTI_System` vs `LTISystem`, `INF` vs explicit start/end
  times). Read the provided template's constructor signature first — some
  past papers use a single `INF` bound (`Signal(INF)` meaning range
  `-INF..INF`) instead of explicit `start_time, end_time`. Adapt the
  constructor, keep the same internal logic.

---

## 8. One-page formula sheet (viva-ready)

```
Convolution:              y[n] = Σ_k x[k]·h[n-k]  =  Σ_m h[m]·x[n-m]
Output range:              [nx_min+nh_min,  nx_max+nh_max]
Full length:                len(x) + len(h) - 1
Valid (full-overlap) length:  len(x) - len(h) + 1        (window problems)

Moving average (window n):        h[m] = 1/n,                m = 0..n-1
Weighted MA (recent = biggest):   h[m] = (n-m)/(n(n+1)/2),    m = 0..n-1
Exponential smoothing:            h[m] = alpha*(1-alpha)^m,   m = 0..n-1
Polynomial multiply:              convolve ascending-power coefficient lists
First difference:                 d[n] = x[n] - x[n-1]
Step -> impulse response:         h[n] = s[n] - s[n-1]
Output via step response:         y[n] = (delta_x * s)[n]

Series (cascade) systems:         h_combined = h1 * h2   (convolve)
Parallel systems:                 h_combined = h1 + h2   (add)

LTI properties: linear, time-invariant, fully described by h[n].
Convolution properties: commutative, associative, distributive over +, shift-equivariant.
```
