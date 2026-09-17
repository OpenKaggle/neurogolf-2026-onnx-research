import numpy as np
from scipy import ndimage

def solver_task255(grid: np.ndarray) -> np.ndarray:
    foreground = grid > 0
    holes = ~ndimage.binary_propagation(foreground, mask=np.ones_like(foreground), structure=ndimage.generate_binary_structure(2, 2))
    out = grid.copy()
    out[holes] = 3
    return out
