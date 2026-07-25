def find_x(y, h):
    # Determine start and end times for x (since y_start = x_start + h_start)
    x_start = y.start_time - h.start_time
    x_end = y.end_time - h.end_time
    x = DiscreteSignal(x_start, x_end)
    
    # The very first value of h (the one we divide by)
    h_first = h.get_value_at_time(h.start_time)
    
    # Go through time one step at a time
    for n in x.times():
        # Start with the matching y value
        y_val = y.get_value_at_time(n + h.start_time)
        
        # Subtract the effects of the x values we ALREADY calculated
        for k in range(x_start, n):
            y_val -= x.get_value_at_time(k) * h.get_value_at_time(n + h.start_time - k)
            
        # Divide to get our new x value and save it
        x.set_value_at_time(n, y_val / h_first)
        
    return x
  
