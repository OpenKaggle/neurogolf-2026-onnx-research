import json
import numpy as np
from collections import deque

data = json.load(open("data/neurogolf-2026/raw/task255.json"))

def compute_green(inp):
    """
    Fill with green if:
    - Horizontally enclosed: non-zero left AND right in same row
    - NOT vertically enclosed: NOT (non-zero above AND below in any column)
    
    i.e., left & right & ~(above & below) & ~mask
    """
    h, w = inp.shape
    mask = inp > 0
    
    above = np.zeros_like(mask)
    for r in range(1, h):
        above[r] = above[r-1] | mask[r-1]
    
    below = np.zeros_like(mask)
    for r in range(h-2, -1, -1):
        below[r] = below[r+1] | mask[r+1]
    
    left = np.zeros_like(mask)
    for c in range(1, w):
        left[:, c] = left[:, c-1] | mask[:, c-1]
    
    right = np.zeros_like(mask)
    for c in range(w-2, -1, -1):
        right[:, c] = right[:, c+1] | mask[:, c+1]
    
    # Condition: horizontally enclosed but NOT vertically enclosed
    horiz_enclose = left & right
    vert_enclose = above & below
    
    fill = horiz_enclose & ~vert_enclose & ~mask
    return fill

def compute_horiz_only(inp):
    """Only horizontally enclosed (no vertical condition)"""
    h, w = inp.shape
    mask = inp > 0
    left = np.zeros_like(mask)
    for c in range(1, w):
        left[:, c] = left[:, c-1] | mask[:, c-1]
    right = np.zeros_like(mask)
    for c in range(w-2, -1, -1):
        right[:, c] = right[:, c+1] | mask[:, c+1]
    return left & right & ~mask

def compute_horiz_no_above(inp):
    """Horizontally enclosed but NOT above"""
    h, w = inp.shape
    mask = inp > 0
    above = np.zeros_like(mask)
    for r in range(1, h):
        above[r] = above[r-1] | mask[r-1]
    left = np.zeros_like(mask)
    for c in range(1, w):
        left[:, c] = left[:, c-1] | mask[:, c-1]
    right = np.zeros_like(mask)
    for c in range(w-2, -1, -1):
        right[:, c] = right[:, c+1] | mask[:, c+1]
    return left & right & ~above & ~mask

def compute_horiz_no_below(inp):
    """Horizontally enclosed but NOT below"""
    h, w = inp.shape
    mask = inp > 0
    below = np.zeros_like(mask)
    for r in range(h-2, -1, -1):
        below[r] = below[r+1] | mask[r+1]
    left = np.zeros_like(mask)
    for c in range(1, w):
        left[:, c] = left[:, c-1] | mask[:, c-1]
    right = np.zeros_like(mask)
    for c in range(w-2, -1, -1):
        right[:, c] = right[:, c+1] | mask[:, c+1]
    return left & right & ~below & ~mask

def compute_horiz_neither_vert(inp):
    """Horizontally enclosed but NEITHER above NOR below (both missing)"""
    h, w = inp.shape
    mask = inp > 0
    above = np.zeros_like(mask)
    for r in range(1, h):
        above[r] = above[r-1] | mask[r-1]
    below = np.zeros_like(mask)
    for r in range(h-2, -1, -1):
        below[r] = below[r+1] | mask[r+1]
    left = np.zeros_like(mask)
    for c in range(1, w):
        left[:, c] = left[:, c-1] | mask[:, c-1]
    right = np.zeros_like(mask)
    for c in range(w-2, -1, -1):
        right[:, c] = right[:, c+1] | mask[:, c+1]
    # Neither above nor below has any non-zero (in any column)
    return left & right & ~above & ~below & ~mask

def compute_vert_no_horiz(inp):
    """Vertically enclosed but NOT horizontally enclosed"""
    h, w = inp.shape
    mask = inp > 0
    above = np.zeros_like(mask)
    for r in range(1, h):
        above[r] = above[r-1] | mask[r-1]
    below = np.zeros_like(mask)
    for r in range(h-2, -1, -1):
        below[r] = below[r+1] | mask[r+1]
    left = np.zeros_like(mask)
    for c in range(1, w):
        left[:, c] = left[:, c-1] | mask[:, c-1]
    right = np.zeros_like(mask)
    for c in range(w-2, -1, -1):
        right[:, c] = right[:, c+1] | mask[:, c+1]
    return ~left & ~right & above & below & ~mask

def score(pred, expected):
    return np.sum((pred == 3) == (expected == 3)) / expected.size

print("=== Horizontal-enclosure-only tests ===")
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        green_exp = np.sum(expected == 3)
        print(f"\n{split} {i} (expected green: {green_exp})")
        
        for name, fn in [
            ("horiz_enclose_only", compute_horiz_only),
            ("horiz_no_above", compute_horiz_no_above),
            ("horiz_no_below", compute_horiz_no_below),
            ("horiz_no_vert_both", compute_horiz_neither_vert),
            ("horiz_not_vert_enclose", compute_green),
            ("vert_not_horiz_enclose", compute_vert_no_horiz),
        ]:
            green = fn(inp)
            out = inp.copy()
            out[green] = 3
            acc = score(out, expected)
            gp = np.sum(green)
            overlap = np.sum(green & (expected == 3))
            print(f"  {name}: acc={acc:.4f}, g={gp}, overlap={overlap}")
