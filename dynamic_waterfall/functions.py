# d_ext = (c / (2 * 220e6) * (-buffer_size / 2 + np.arange(buffer_size))).reshape(-1, 1)
import numpy as np

def rectangular_pulse(d_ext, a, b, edge_value=0.5):

    # Check that a and b are not equal
    if a == b:
        raise ValueError("a and b must be different values.")

    # Ensure a < b
    if a > b:
        a, b = b, a

    # Handle different ranges of x and return appropriate values
    if np.isscalar(d_ext):
        if d_ext < a or d_ext > b:
            return 0
        elif d_ext == a or d_ext == b:
            return edge_value
        else:
            return 1
    else:  # Handle the case where d_ext is an array
        result = np.zeros_like(d_ext)
        result[(d_ext > a) & (d_ext < b)] = 1
        result[(d_ext == a) | (d_ext == b)] = edge_value
        return result

# a = 3  
# b = -3  
# r = rectangular_pulse(d_ext, a, b)

