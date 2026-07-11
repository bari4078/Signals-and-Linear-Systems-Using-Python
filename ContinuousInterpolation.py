import numpy as np

def interpolate_signal(t_original, x_original, t_query):
    
    # Let numpy do all the spacing, index-finding, and boundary clipping for you
    return np.interp(t_query, t_original, x_original)
  
