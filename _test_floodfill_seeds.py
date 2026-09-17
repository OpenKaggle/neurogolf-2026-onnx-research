import json
import numpy as np
from collections import deque

data = json.load(open("data/neurogolf-2026/raw/task255.json"))

def hypothesis_floodfill_from_seeds(inp, seed_fn):
    """Flood fill (4-dir) from seed cells into background"""
    h, w = inp.shape
    seeds = seed_fn(inp)
    
    visited = np.zeros_like(inp, dtype=bool)
    q = deque()
    
    for r in range(h):
        for c in range(w):
            if seeds[r, c]:
                visited[r, c] = True
                q.append((r, c))
    
    while q:
        r, c = q.popleft()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and inp[nr, nc] == 0:
                visited[nr, nc] = True
                q.append((nr, nc))
    
    out = inp.copy()
    out[visited & (inp == 0)] = 3
    return out

def seeds_axial_4dir(inp):
    """Cells with non-zero in all 4 axial directions"""
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
    
    return above & below & left & right & ~mask

def seeds_axial_vert(inp):
    """Cells with non-zero above AND below (any column)"""
    h, w = inp.shape
    mask = inp > 0
    
    above = np.zeros_like(mask)
    for r in range(1, h):
        above[r] = above[r-1] | mask[r-1]
    
    below = np.zeros_like(mask)
    for r in range(h-2, -1, -1):
        below[r] = below[r+1] | mask[r+1]
    
    return above & below & ~mask

def seeds_axial_horiz(inp):
    """Cells with non-zero left AND right (same row)"""
    h, w = inp.shape
    mask = inp > 0
    
    left = np.zeros_like(mask)
    for c in range(1, w):
        left[:, c] = left[:, c-1] | mask[:, c-1]
    
    right = np.zeros_like(mask)
    for c in range(w-2, -1, -1):
        right[:, c] = right[:, c+1] | mask[:, c+1]
    
    return left & right & ~mask

def seeds_axial_4dir_same_col(inp):
    """Cells with non-zero above AND below in SAME column, and left AND right in same row"""
    h, w = inp.shape
    mask = inp > 0
    
    above_same = np.zeros_like(mask)
    for r in range(1, h):
        above_same[r] = mask[r-1]  # just the cell directly above
    
    below_same = np.zeros_like(mask)
    for r in range(h-2, -1, -1):
        below_same[r] = mask[r+1]  # just the cell directly below
    
    left_same = np.zeros_like(mask)
    for c in range(1, w):
        left_same[:, c] = mask[:, c-1]  # just the cell directly left
    
    right_same = np.zeros_like(mask)
    for c in range(w-2, -1, -1):
        right_same[:, c] = mask[:, c+1]  # just the cell directly right
    
    # Propagate: have non-zero somewhere above (any distance, same column)
    above_any = np.zeros_like(mask)
    for r in range(1, h):
        above_any[r] = above_any[r-1] | mask[r-1]
    
    below_any = np.zeros_like(mask)
    for r in range(h-2, -1, -1):
        below_any[r] = below_any[r+1] | mask[r+1]
    
    return above_any & below_any & left_same & right_same & ~mask

def score(pred, expected):
    return np.sum((pred == 3) == (expected == 3)) / expected.size

print("=== Flood fill from seeds ===")
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        green_exp = np.sum(expected == 3)
        print(f"\n{split} {i} (expected green: {green_exp})")
        
        for name, seeds_fn in [
            ("from_axial4dir", seeds_axial_4dir),
            ("from_axial_vert", seeds_axial_vert),
            ("from_axial_horiz", seeds_axial_horiz),
        ]:
            pred = hypothesis_floodfill_from_seeds(inp, seeds_fn)
            acc = score(pred, expected)
            gp = np.sum(pred == 3)
            ns = np.sum(seeds_fn(inp))
            print(f"  {name}: acc={acc:.4f}, g={gp}, seeds={ns}")
