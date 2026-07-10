import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Time axis
# ----------------------------
T_MIN, T_MAX, N = -4.0, 4.0, 4001


def x_of_t(t: np.ndarray) -> np.ndarray:
    """
    Base signal x(t): sinusoidal signal
    """
    return (
        np.sin(2 * np.pi * 0.5 * t)
        + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    )


# ==========================================================
# ANSWER IMPLEMENTATION
# ==========================================================

def interpolate_signal(
    t_original: np.ndarray,
    x_original: np.ndarray,
    t_query: np.ndarray
) -> np.ndarray:
    """
    Interpolate using average of two neighboring samples.
    """
    dt = t_original[1] - t_original[0]
    exact_indices = (t_query - t_original[0]) / dt
    
    left_indices = np.floor(exact_indices).astype(int)
    right_indices = np.ceil(exact_indices).astype(int)

    max_index = len(x_original) - 1
    left_indices = np.clip(left_indices,0,max_index)
    right_indices = np.clip(right_indices,0,max_index)

    return 0.5 * (x_original[left_indices] - x_original[right_indices])


def time_scale(
    t: np.ndarray,
    x: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Time sub-scaling:
        y(t) = x(t / k)
    """
    t_query = t/k
    y = interpolate_signal(t,x,t_query)

    return y

def plot_pair(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    """
    Plot graphs.
    """
    plt.figure(figsize=(10,6))

    plt.plot(t,x,label = "Original Signal x(t)", color = "blue", linewidth = 1.5)
    plt.plot(t,y,label = "Sub-scaled Signal y(t)", color = "red", linestyle="--",linewidth= 1.5)
    plt.title(title)
    plt.xlabel("Time (t)")
    plt.ylabel("Amplitute")

    plt.grid(True)
    plt.legend()


# ----------------------------
# Main
# ----------------------------
def main():
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    k = 2   # sub-scaling factor
    y = time_scale(t, x, k)

    plot_pair(
        t,
        x,
        y,
        title=f"Time Sub-scaling: y(t) = x(t / {k})"
    )
    plt.show()


if __name__ == "__main__":
    main()
