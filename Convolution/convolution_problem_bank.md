# CSE220 Convolution — Full Problem Bank (Easy → Extremely Difficult)

Every problem below is drawn from your Offline assignment spec or an actual
past online quiz (Jan 2024 A/B/C, July 2025 A/B/C) — these are the exact
shapes of task that have appeared, so treat this as your "could reappear"
checklist. All code snippets were run and verified to produce correct output.

**Assumes you've already pasted the base classes** `DiscreteSignal`,
`LTISystem`, `make_signal` (from `signal_lti_toolkit.py`, given earlier —
paste Section 1 of that file first, every time). Each snippet below only
shows the extra bit needed for that specific problem.

## Contents
- Tier 1 — Easy: single-method warm-ups (1–7)
- Tier 2 — Medium: full convolution & built-in filters (8–14)
- Tier 3 — Hard: turning a word problem into an impulse response (15–18)
- Tier 4 — Very Hard: beyond one signal/one system (19–21)
- Tier 5 — Extremely Difficult: multi-system & full pipelines (22–25)

---

## Tier 1 — Easy

### 1. Shift a signal
**Problem:** Given `x[n]` over `0..2` with values `[1,2,3]`, produce `x[n-2]`.

```python
x = make_signal(0, 2, [1, 2, 3])
y = x.shift(2)
print(y.start_time, y.end_time, y.values)   # 2 4 [1, 2, 3]
```
**Explanation:** `shift(k)` just moves the start/end times by `k` and keeps
the values in the same order — a delay of `k` steps.

---

### 2. Scalar-multiply a signal
**Problem:** Given `x[1] = 4`, find `0.5·x`.

```python
x = make_signal(0, 2, [2, 4, 6])
y = x.multiply(0.5)
print(y.values)   # [1.0, 2.0, 3.0]
```
**Explanation:** Multiplies every stored sample by the scalar; range stays
the same.

---

### 3. Add two signals with different time ranges
**Problem:** `a` lives on `0..2` = `[1,2,3]`, `b` lives on `-1..1` =
`[10,20,30]`. Find `a + b`.

```python
a = make_signal(0, 2, [1, 2, 3])
b = make_signal(-1, 1, [10, 20, 30])
c = a.add(b)
print(c.start_time, c.end_time, c.values)   # -1 2 [10.0, 21, 32, 3.0]
```
**Explanation:** The result spans the **union** of both ranges; wherever
only one signal has a value, the missing one is treated as 0.

---

### 4. Determine the convolution output range automatically
**Problem:** `x` lives on `[nx_min, nx_max]`, `h` lives on `[nh_min, nh_max]`.
Find where `y = x*h` can be nonzero, without hardcoding numbers.

```python
def output_range(x, h):
    return x.start_time + h.start_time, x.end_time + h.end_time

x = make_signal(0, 4, [0]*5)
h = make_signal(0, 2, [0]*3)
print(output_range(x, h))   # (0, 6)
```
**Explanation:** Add the two start times for the new start, add the two end
times for the new end — this is `LTISystem.output_range` in your toolkit.

---

### 5. Identity system
**Problem:** Show that a system with `h[0]=1` (all else 0) returns the input
unchanged.

```python
x = make_signal(-1, 2, [3.0, -1.0, 2.0, 5.0])
h_identity = make_signal(0, 0, [1.0])
y = LTISystem(h_identity).output(x)
print(y.values == x.values, y.start_time == x.start_time)   # True True
```
**Explanation:** Convolving with a single spike of height 1 just reproduces
the original signal — the "do nothing" system.

---

### 6. First difference of a signal
**Problem:** Compute `Δx[n] = x[n] - x[n-1]` using only `shift`/`multiply`/`add`
(no numpy diff).

```python
def first_difference(sig):
    return sig.add(sig.shift(1).multiply(-1))

x = make_signal(0, 9, list(range(1, 11)))   # 1..10
print(first_difference(x).values)
# [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -10.0]
```
**Explanation:** `x[n-1]` is `x` shifted right by 1; negate it and add to
`x[n]`. The last value looks odd only because the shifted copy runs one step
further right than the original — that's expected edge behaviour.

---

### 7. Compute a single output sample directly
**Problem:** For `x=[1,0,2]` (starts at 0) and `h=[1,1]` (starts at 0), find
`y[2]` without computing the whole output signal.

```python
x = make_signal(0, 2, [1, 0, 2])
h = make_signal(0, 1, [1, 1])
system = LTISystem(h)
print(system.output_at_time(x, 2))   # 2.0
```
**Explanation:** This is one term of the convolution sum,
`Σ x[k]·h[n-k]`, evaluated at a single `n` — useful when a viva question
just wants one output value explained, not the whole signal.

---

## Tier 2 — Medium

### 8. Full convolution by superposition
**Problem:** Compute the whole output of `x=[1,0,2]` (0-indexed) through
`h=[1,1]` (0-indexed) by adding scaled/shifted copies of `h`.

```python
x = make_signal(0, 2, [1, 0, 2])
h = make_signal(0, 1, [1, 1])
y = LTISystem(h).output_by_superposition(x)
print(y.start_time, y.values)   # 0 [1.0, 1.0, 2.0, 2.0]
```
**Explanation:** Every nonzero `x[k]` "fires" a copy of `h` scaled by `x[k]`
and shifted to start at `k`; summing all the copies gives the same answer as
direct convolution.

---

### 9. Full convolution by direct sliding sum
**Problem:** Same as above, but by evaluating the convolution sum at every
output index instead of superposing components.

```python
y = LTISystem(h).output(x)
print(y.start_time, y.values)   # 0 [1.0, 1.0, 2.0, 2.0]
```
**Explanation:** `output()` calls `output_at_time` for every `n` in the
output range — mechanically different from Method A, mathematically
identical result.

---

### 10. Verify the two convolution methods agree
**Problem:** Confirm `output_by_superposition` and `output` give the same
signal (required for every processed impulse response in the assignment).

```python
def max_absolute_difference(a, b):
    start = min(a.start_time, b.start_time)
    end = max(a.end_time, b.end_time)
    return max(abs(a.get_value_at_time(t) - b.get_value_at_time(t)) for t in range(start, end + 1))

y_sup = LTISystem(h).output_by_superposition(x)
y_conv = LTISystem(h).output(x)
print(max_absolute_difference(y_sup, y_conv))   # 0.0 (or ~1e-16)
```
**Explanation:** Compare sample-by-sample over the combined range; the
maximum difference should be zero up to floating-point rounding.

---

### 11. n-point moving average built-in filter
**Problem:** Apply the 3-, 5-, or 7-point moving-average impulse response
(as required by the assignment's built-in mode).

```python
def impulse_moving_average(length):
    return make_signal(0, length - 1, [1.0 / length] * length)

x = make_signal(0, 9, list(range(1, 11)))
y5 = LTISystem(impulse_moving_average(5)).output(x)
print(y5.start_time, y5.end_time)   # 0 13
print([round(v, 2) for v in y5.values])
# [0.2, 0.6, 1.2, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.8, 5.4, 3.8, 2.0]
```
**Explanation:** All taps equal `1/length` — every sample in the window
counts the same. Swap in `length=3` or `length=7` for the other two
built-ins.

---

### 12. Fixed weighted-smoothing built-in filter
**Problem:** Apply the assignment's built-in weighted smoothing:
`h[0]=0.5, h[1]=0.3, h[2]=0.2`.

```python
h = make_signal(0, 2, [0.5, 0.3, 0.2])
y = LTISystem(h).output(x)
print([round(v, 2) for v in y.values][:5])   # [0.5, 1.3, 2.3, 3.3, 4.3]
```
**Explanation:** Same convolution mechanics as any other impulse response —
this one just has fixed, unequal taps instead of equal ones.

---

### 13. Read a custom impulse response + input signal from a text file
**Problem:** Parse the assignment's input-file format:
```
input_start input_end
x[input_start] ... x[input_end]
custom
impulse_start impulse_end
h[impulse_start] ... h[impulse_end]
```

```python
def parse_signal(lines, i):
    start, end = map(int, lines[i].split())
    values = list(map(float, lines[i + 1].split()))
    return make_signal(start, end, values), i + 2

with open("inputs/1.txt") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

x, i = parse_signal(lines, 0)
mode = lines[i].lower(); i += 1
if mode in ("custom",):
    h, i = parse_signal(lines, i)
```
**Explanation:** Each signal block is always "range line, then values line" —
write one small parser for that shape and reuse it for both `x` and `h`.

---

### 14. Visualize x, h, y (stem plot + grayscale color blocks)
**Problem:** Produce the two required figure types for a processed impulse
response.

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
x.plot("Input x[n]", ax=axes[0])
h.plot("Impulse response h[n]", ax=axes[1])
y.plot("Output y[n]", ax=axes[2])
plt.savefig("plot/convolution.png", dpi=150)

def plot_color_blocks(signal, title, ax):
    v = np.array(signal.values, dtype=float)
    lo, hi = v.min(), v.max()
    gray = (np.full(v.shape, 128) if np.isclose(lo, hi)
            else np.round(255 * (v - lo) / (hi - lo))).astype(np.uint8)
    ax.imshow(np.stack([gray]*3, axis=-1).reshape(1, len(gray), 3), aspect="auto")
    ax.set_title(title); ax.set_yticks([])

fig, axes = plt.subplots(2, 1, figsize=(8, 3))
plot_color_blocks(x, "x[n]", axes[0])
plot_color_blocks(y, "y[n]", axes[1])
plt.savefig("color/convolution.png", dpi=150)
```
**Explanation:** Stem plot = one subplot per signal. Color-block plot =
normalize each signal to 0–255 **separately**, then show as a 1-row image.

---

## Tier 3 — Hard (turning a word problem into an impulse response)

### 15. Unweighted moving average of stock prices
**Problem:** Given `m` stock prices and window size `n`, print the simple
moving average — `m-n+1` values, no library averaging functions allowed.

```python
def unweighted_moving_average(prices, n):
    x = make_signal(0, len(prices) - 1, prices)
    h = make_signal(0, n - 1, [1.0 / n] * n)
    y = LTISystem(h).output(x)
    return [y.get_value_at_time(t) for t in range(n - 1, len(prices))]

print(unweighted_moving_average([1,2,3,4,5,6,7,8], 4))
# [2.5, 3.5, 4.5, 5.5, 6.5]
```
**Explanation:** `h[m] = 1/n` gives equal weight to the current sample and
the `n-1` before it. Slicing to `range(n-1, len(prices))` keeps only windows
that are fully inside the data (the "valid" region).

---

### 16. Weighted moving average of stock prices
**Problem:** Same as above, but the most recent day gets weight `n`, the day
before gets `n-1`, ..., the oldest day in the window gets weight `1`
(normalized to sum to 1).

```python
def weighted_moving_average(prices, n):
    x = make_signal(0, len(prices) - 1, prices)
    total = n * (n + 1) / 2
    h_vals = [(n - m) / total for m in range(n)]   # h[0] = n/total, biggest
    h = make_signal(0, n - 1, h_vals)
    y = LTISystem(h).output(x)
    return [y.get_value_at_time(t) for t in range(n - 1, len(prices))]

print(weighted_moving_average([1,2,3,4,5,6,7,8], 4))
# [3.0, 4.0, 5.0, 6.0, 7.0]
```
**Explanation:** `h[0]` always multiplies the *current/most-recent* sample
(from `y[n] = Σ h[m]x[n-m]`), so put the biggest weight there — not at
`h[n-1]`, which is the easy mistake.

---

### 17. Exponential smoothing of stock prices
**Problem:** Weight decays geometrically: `h[k] = α(1-α)^k` for
`k = 0..n-1`, window size `n`, decay parameter `α`.

```python
def exponential_smoothing(prices, n, alpha):
    x = make_signal(0, len(prices) - 1, prices)
    h_vals = [alpha * (1 - alpha) ** k for k in range(n)]
    h = make_signal(0, n - 1, h_vals)
    y = LTISystem(h).output(x)
    return [y.get_value_at_time(t) for t in range(n - 1, len(prices))]

print([round(v,2) for v in exponential_smoothing(
    [10,11,12,9,10,13,15,16,17,18], 3, 0.8)])
# [11.68, 9.47, 9.82, 12.29, 14.4, 15.62, 16.64, 17.63]
```
**Explanation:** Identical machinery to Problem 16, just a different weight
formula — geometric decay instead of a linear ramp.

---

### 18. Polynomial multiplication via convolution
**Problem:** Multiply two polynomials given as coefficient lists
(highest-power-first) using convolution, not library polynomial math.

```python
def poly_multiply(coeffs1_desc, coeffs2_desc):
    a_asc = list(reversed(coeffs1_desc))
    b_asc = list(reversed(coeffs2_desc))
    xa = make_signal(0, len(a_asc) - 1, a_asc)
    xb = make_signal(0, len(b_asc) - 1, b_asc)
    y = LTISystem(xb).output(xa)
    return list(reversed(y.values))

print(poly_multiply([3,-2,1], [2,0,-3,1]))
# [6.0, -4.0, -7.0, 9.0, -5.0, 1.0]  ->  6x^5 - 4x^4 - 7x^3 + 9x^2 - 5x + 1
```
**Explanation:** Multiplying `x^a` by `x^b` gives `x^(a+b)` — exactly how
convolution indices add (`k + (n-k) = n`). Reverse to ascending-power order
first so array index = power of x, convolve, then reverse the answer back.

---

## Tier 4 — Very Hard (beyond one signal / one system)

### 19. Superposition of multiple weighted input signals
**Problem:** Given `x(n) = 2·x1(n) − x2(n)` as several component signals with
coefficients, compute the system's output using linearity (don't build the
combined `x` by hand).

```python
class SuperSignal:
    def __init__(self):
        self.components = []
    def add(self, signal, coefficient=1.0):
        self.components.append((coefficient, signal))

def output_super(system, super_signal):
    total = None
    for coeff, sig in super_signal.components:
        contribution = system.output(sig).multiply(coeff)
        total = contribution if total is None else total.add(contribution)
    return total

x1 = make_signal(0, 0, [1.0])
x2 = make_signal(2, 2, [1.0])
ss = SuperSignal()
ss.add(x1, 2.0)
ss.add(x2, -1.0)

system = LTISystem(make_signal(0, 1, [1.0, 0.5]))
y = output_super(system, ss)
print(y.start_time, y.values)   # 0 [2.0, 1.0, -1.0, -0.5]
```
**Explanation:** LTI systems are linear: scale an input → output scales the
same way; add inputs → outputs add. So `system.output(2x1 - x2)` equals
`2·system.output(x1) - system.output(x2)` — verified identical either way.

---

### 20. Recover the impulse response from a step response
**Problem:** Given `s[n]` (the system's response to a unit step), find
`h[n]`.

```python
def first_difference(sig):
    return sig.add(sig.shift(1).multiply(-1))

s = make_signal(0, 4, [1.0, 1.0, 2.0, 2.0, 2.0])
h = first_difference(s)
print(h.start_time, h.end_time, h.values)
# 0 5 [1.0, 0.0, 1.0, 0.0, 0.0, -2.0]
```
**Explanation:** `h[n] = s[n] - s[n-1]`. A step response is the running sum
of the impulse response, so differencing it undoes that sum and recovers
`h[n]`.

---

### 21. Compute output using ONLY the step response
**Problem:** Without ever building `h[n]` directly, compute `y[n]` for some
input `x[n]` using only the step response `s[n]`, then verify it matches the
impulse-response method.

```python
def output_using_step_response(x, step_response):
    dx = first_difference(x)
    return LTISystem(step_response).output(dx)      # (delta_x * s)

x = make_signal(0, 3, [1.0, -1.0, 2.0, 0.5])
y_via_step = output_using_step_response(x, s)
y_via_h = LTISystem(h).output(x)
print(y_via_step.values == y_via_h.values)   # True
```
**Explanation:** Identity: `y = x*h = (Δx*u)*h = Δx*(u*h) = Δx*s`. Difference
the input first, then convolve with the step response directly — same
answer as the normal impulse-response convolution.

---

## Tier 5 — Extremely Difficult (multi-system & full pipelines)

### 22. Series (cascade) combination of two LTI systems
**Problem:** `x` passes through `h1`, then the result passes through `h2`.
Find one combined system that does the same thing in a single convolution.

```python
h1 = make_signal(0, 0, [1.0])
h2 = make_signal(1, 1, [0.5])
x = make_signal(0, 2, [1.0, 0.0, -1.0])

y_block = LTISystem(h2).output(LTISystem(h1).output(x))

h_series = LTISystem(h1).output(h2)          # convolve h1 with h2
y_combined = LTISystem(h_series).output(x)

print(y_block.values == y_combined.values)   # True
```
**Explanation:** Cascading systems is associative:
`(x*h1)*h2 = x*(h1*h2)`. Convolve the two impulse responses together to get
one equivalent system.

---

### 23. Parallel combination of two LTI systems
**Problem:** `x` passes through `h1` and `h2` independently; their outputs
are added. Find one combined system.

```python
y_parallel_block = LTISystem(h1).output(x).add(LTISystem(h2).output(x))

h_parallel = h1.add(h2)
y_parallel_combined = LTISystem(h_parallel).output(x)

print(y_parallel_block.values == y_parallel_combined.values)   # True
```
**Explanation:** Convolution distributes over addition:
`x*h1 + x*h2 = x*(h1+h2)`. Add the two impulse responses together to get one
equivalent system.

---

### 24. Mixed topology: parallel branch feeding into a series block
**Problem:** `h1` and `h2` are in parallel (outputs added), and that combined
signal then passes through `h3` in series. Verify block-by-block matches one
fully combined impulse response.

```python
h3 = make_signal(0, 1, [1.0, 1.0])

# block by block
y1 = LTISystem(h1).output(x)
y2 = LTISystem(h2).output(x)
y_final_block = LTISystem(h3).output(y1.add(y2))

# one combined impulse response: (h1 + h2) then convolved with h3
h_par = h1.add(h2)
h_combined = LTISystem(h_par).output(h3)
y_final_combined = LTISystem(h_combined).output(x)

print(y_final_block.values == y_final_combined.values)   # True
```
**Explanation:** Just apply the two rules from Problems 22/23 in whatever
order the diagram shows — convolution is associative, so build the combined
`h` from the inside out and it will always match doing it block-by-block.

---

### 25. Full file-based pipeline (compressed offline-style task)
**Problem:** Under time pressure, read an input file (custom or all 6
built-ins), run both convolution methods, verify they match, save a report
and both plot types to an output directory — the complete Offline task,
compressed.

```python
import os

def process_case(x, h, out_dir, name, description):
    system = LTISystem(h)
    y_sup = system.output_by_superposition(x)
    y_conv = system.output(x)
    max_diff = max_absolute_difference(y_sup, y_conv)

    os.makedirs(f"{out_dir}/plot", exist_ok=True)
    os.makedirs(f"{out_dir}/color", exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
    x.plot("x[n]", ax=axes[0]); h.plot("h[n]", ax=axes[1]); y_conv.plot("y[n]", ax=axes[2])
    fig.savefig(f"{out_dir}/plot/{name}.png", dpi=150)

    fig2, axes2 = plt.subplots(2, 1, figsize=(8, 3))
    plot_color_blocks(x, "x[n]", axes2[0]); plot_color_blocks(y_conv, "y[n]", axes2[1])
    fig2.savefig(f"{out_dir}/color/{name}.png", dpi=150)

    with open(f"{out_dir}/report.txt", "a") as f:
        f.write(f"{description}\nmax diff: {max_diff:.3g}\n"
                f"output range: [{y_conv.start_time},{y_conv.end_time}]\n\n")

    return y_conv

# custom case
process_case(x, h, "outputs/1", "convolution", "custom impulse response")

# built-in case: loop the six required filters in order
BUILT_INS = [
    ("builtin_1_identity", make_signal(0, 0, [1.0])),
    ("builtin_2_moving_average_3", impulse_moving_average(3)),
    ("builtin_3_moving_average_5", impulse_moving_average(5)),
    ("builtin_4_moving_average_7", impulse_moving_average(7)),
    ("builtin_5_weighted_smoothing", make_signal(0, 2, [0.5, 0.3, 0.2])),
    ("builtin_6_first_difference", make_signal(0, 1, [1.0, -1.0])),
]
for name, h_builtin in BUILT_INS:
    process_case(x, h_builtin, "outputs/2", name, name)
```
**Explanation:** This is every earlier problem wired together: parse input
→ build `x`/`h` → run both convolution methods → verify → plot both figure
types → write a report — looped once for a custom filter, six times for the
built-in bank, exactly as the offline spec requires. Everything inside
`process_case` is just Problems 8–14 called in sequence.
