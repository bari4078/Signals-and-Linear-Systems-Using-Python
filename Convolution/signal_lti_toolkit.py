"""
SIGNAL + LTI CONVOLUTION TOOLKIT  (tested and working)
=======================================================
Paste whichever parts you need. Section 1 (the two classes) is the
foundation almost every question builds on top of.
"""

import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# SECTION 1: THE TWO CORE CLASSES  (paste this first, always)
# =====================================================================

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

    # ---- Method A: superposition (add up scaled/shifted copies of h) ----
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

    # ---- Method B: direct convolution sum, one output sample at a time ----
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


# =====================================================================
# SECTION 2: SuperSignal  (superposition of several signals at once)
# =====================================================================

class SuperSignal:
    """Holds several (coefficient, DiscreteSignal) pairs, e.g. x = 2*x1 - x2."""

    def __init__(self):
        self.components = []

    def add(self, signal, coefficient=1.0):
        self.components.append((coefficient, signal))


def output_super(system, super_signal):
    """y = sum_i coefficient_i * system.output(signal_i)  (linearity of LTI systems)."""
    total = None
    for coeff, sig in super_signal.components:
        contribution = system.output(sig).multiply(coeff)
        total = contribution if total is None else total.add(contribution)
    return total


# =====================================================================
# SECTION 3: Moving average / weighted / exponential smoothing
# =====================================================================
# KEY IDEA:  y[n] = sum_m h[m] * x[n - m]
#            h[0] is the weight on the CURRENT/most-recent sample,
#            h[1] is the weight on one-step-old sample, etc.

def unweighted_moving_average(prices, n):
    """Simple average of the last n values. Returns a plain Python list."""
    x = make_signal(0, len(prices) - 1, prices)
    h = make_signal(0, n - 1, [1.0 / n] * n)
    y = LTISystem(h).output(x)
    # keep only the region where the window is fully inside the data ("valid" convolution)
    return [y.get_value_at_time(t) for t in range(n - 1, len(prices))]


def weighted_moving_average(prices, n):
    """Most recent sample gets weight n, then n-1, ..., oldest gets weight 1 (normalized)."""
    x = make_signal(0, len(prices) - 1, prices)
    total = n * (n + 1) / 2
    h_vals = [(n - m) / total for m in range(n)]     # h[0] = n/total (largest, current sample)
    h = make_signal(0, n - 1, h_vals)
    y = LTISystem(h).output(x)
    return [y.get_value_at_time(t) for t in range(n - 1, len(prices))]


def exponential_smoothing(prices, n, alpha):
    """h[m] = alpha * (1 - alpha)**m  for m = 0..n-1 (exponentially decaying weights)."""
    x = make_signal(0, len(prices) - 1, prices)
    h_vals = [alpha * (1 - alpha) ** m for m in range(n)]
    h = make_signal(0, n - 1, h_vals)
    y = LTISystem(h).output(x)
    return [y.get_value_at_time(t) for t in range(n - 1, len(prices))]


# =====================================================================
# SECTION 4: Polynomial multiplication via convolution
# =====================================================================
# Coefficients are usually given highest-power-first (e.g. 3x^2-2x+1 -> [3,-2,1]).
# Reverse to ascending-power order so index == power of x, convolve, reverse back.

def poly_multiply(coeffs1_desc, coeffs2_desc):
    a_asc = list(reversed(coeffs1_desc))
    b_asc = list(reversed(coeffs2_desc))
    xa = make_signal(0, len(a_asc) - 1, a_asc)
    xb = make_signal(0, len(b_asc) - 1, b_asc)
    y = LTISystem(xb).output(xa)          # either one can play "impulse response"
    return list(reversed(y.values))        # back to highest-power-first


# =====================================================================
# SECTION 5: Combining LTI systems (series / parallel)
# =====================================================================
# SERIES (cascade, one after another):   h_combined = conv(h1, h2) = LTISystem(h1).output(h2)
# PARALLEL (side by side, outputs added): h_combined = h1.add(h2)
#
# Example:
#   h_series   = LTISystem(h1).output(h2)          # h1 then h2
#   h_parallel = h1.add(h2)                         # h1 and h2 side by side, summed
#   y = LTISystem(h_series).output(x)               # apply the single equivalent system


# =====================================================================
# SECTION 6: Step response <-> impulse response
# =====================================================================

def first_difference(sig):
    """Returns sig[n] - sig[n-1], using only shift/multiply/add (assumes sig[-1]=0)."""
    return sig.add(sig.shift(1).multiply(-1))


def impulse_from_step_response(step_response):
    """h[n] = s[n] - s[n-1]."""
    return first_difference(step_response)


def output_using_step_response(x, step_response):
    """y[n] = (delta_x * s)[n], where delta_x[n] = x[n] - x[n-1]."""
    dx = first_difference(x)
    return LTISystem(step_response).output(dx)


# =====================================================================
# SECTION 7: Plotting helpers (stem + grayscale color-block)
# =====================================================================

def plot_color_blocks(signal, title, ax):
    """One row of grayscale blocks, each signal value normalized 0-255 on its own."""
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


if __name__ == "__main__":
    # quick self-check when you paste this file and want to confirm it runs
    x = make_signal(0, 2, [1, 0, 2])
    h = make_signal(0, 1, [1, 1])
    sysx = LTISystem(h)
    assert sysx.output(x).values == sysx.output_by_superposition(x).values
    print("Toolkit OK.")
