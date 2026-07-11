import numpy as np

def time_scale_interpolate(x: np.ndarray, k: int, INF: int) -> np.ndarray:
    
    # 1. The original positions (xp)
    old_time = np.arange(-INF, INF + 1)
    
    # 2. The exact positions we want to look up (x)
    new_time = np.arange(-INF, INF + 1) / k
    
    # 3. Connect the dots!
    return np.interp(new_time, old_time, x, left=0.0, right=0.0)
  
