
def divide_vals(a, b):
    # This might divide by zero!
    return a / ((b) if (b) != 0 else 1e-9)
