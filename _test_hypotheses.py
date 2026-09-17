import json
import numpy as np

data = json.load(open("data/neurogolf-2026/raw/task255.json"))

def flood_fill_border(mask):
    """Flood fill from border 0-cells"""
    h, w = mask.shape
    result = mask.copy()
    # BFS from border 0-cells
    from collections import deque
    q = deque()
    for r in range(h):
        if result[r, 0] == 0:
            result[r, 0] = 2; q.append((r, 0))
        if result[r, w-1] == 0:
            result[r, w-1] = 2; q.append((r, w-1))
    for c in range(w):
        if result[0, c] == 0:
            result[0, c] = 2; q.append((0, c))
        if result[h-1, c] == 0:
            result[h-1, c] = 2; q.append((h-1, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and result[nr, nc] == 0:
                result[nr, nc] = 2
                q.append((nr, nc))
    # Return mask of cells that are 0 AND not reachable
    return (mask == 0) & (result != 2)

def hypothesis_flood_fill(inp):
    """Fill all 0-cells not connected to border with green"""
    filled = flood_fill_border(inp)
    out = inp.copy()
    out[filled] = 3
    return out

def hypothesis_row_gap_fill(inp):
    """For each row, fill 0-cells between leftmost and rightmost non-zero"""
    out = inp.copy()
    h, w = inp.shape
    for r in range(h):
        non_zero = np.where(inp[r] > 0)[0]
        if len(non_zero) >= 2:
            left, right = non_zero[0], non_zero[-1]
            for c in range(left, right + 1):
                if out[r, c] == 0:
                    out[r, c] = 3
    return out

def hypothesis_col_gap_fill(inp):
    """For each col, fill 0-cells between topmost and bottommost non-zero"""
    out = inp.copy()
    h, w = inp.shape
    for c in range(w):
        non_zero = np.where(inp[:, c] > 0)[0]
        if len(non_zero) >= 2:
            top, bottom = non_zero[0], non_zero[-1]
            for r in range(top, bottom + 1):
                if out[r, c] == 0:
                    out[r, c] = 3
    return out

def hypothesis_bbox_fill(inp):
    """Fill 0s in the global bounding box of all non-zero pixels"""
    out = inp.copy()
    non_zero = np.where(inp > 0)
    if len(non_zero[0]) == 0:
        return out
    rmin, rmax = non_zero[0].min(), non_zero[0].max()
    cmin, cmax = non_zero[1].min(), non_zero[1].max()
    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            if out[r, c] == 0:
                out[r, c] = 3
    return out

def hypothesis_intersection_gap(inp):
    """For each row find the largest gap between non-zero pixels.
    Green fills the intersection of the largest gaps across rows."""
    h, w = inp.shape
    # For each row, compute all gap intervals
    row_gaps = []
    for r in range(h):
        nz = np.where(inp[r] > 0)[0]
        if len(nz) >= 2:
            gaps = []
            for i in range(len(nz) - 1):
                if nz[i+1] - nz[i] > 1:
                    gaps.append((nz[i]+1, nz[i+1]-1))
            if gaps:
                row_gaps.append(gaps)
    
    if not row_gaps:
        return inp.copy()
    
    # Start with first row's gaps
    common = set()
    for g in row_gaps[0]:
        for c in range(g[0], g[1]+1):
            common.add(c)
    
    for gaps in row_gaps[1:]:
        row_set = set()
        for g in gaps:
            for c in range(g[0], g[1]+1):
                row_set.add(c)
        common = common & row_set
    
    out = inp.copy()
    for r in range(h):
        for c in common:
            if out[r, c] == 0:
                out[r, c] = 3
    return out

def hypothesis_enclosed_by_nonzero(inp):
    """Fill 0-cells that are surrounded by non-zero both above&below OR left&right"""
    out = inp.copy()
    h, w = inp.shape
    for r in range(1, h-1):
        for c in range(1, w-1):
            if inp[r, c] == 0:
                # Check if enclosed vertically (non-zero above AND below)
                above = np.any(inp[:r, c] > 0)
                below = np.any(inp[r+1:, c] > 0)
                left = np.any(inp[r, :c] > 0)
                right = np.any(inp[r, c+1:] > 0)
                if (above and below) or (left and right):
                    out[r, c] = 3
    return out

def hypothesis_horiz_enclose(inp):
    """Fill 0-cells that have non-zero to the left AND right in the same row"""
    out = inp.copy()
    h, w = inp.shape
    for r in range(h):
        nz = np.where(inp[r] > 0)[0]
        for c in range(w):
            if inp[r, c] == 0:
                has_left = np.any(nz < c)
                has_right = np.any(nz > c)
                if has_left and has_right:
                    out[r, c] = 3
    return out

def score(out, expected):
    green_pred = (out == 3)
    green_exp = (expected == 3)
    correct = np.sum(green_pred == green_exp)
    total = out.size
    return correct / total

for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        print(f"\n=== {split} {i} ===")
        for name, fn in [
            ("flood_fill", hypothesis_flood_fill),
            ("row_gap", hypothesis_row_gap_fill),
            ("col_gap", hypothesis_col_gap_fill),
            ("bbox", hypothesis_bbox_fill),
            ("intersection_gap", hypothesis_intersection_gap),
            ("enclosed_by_nonzero", hypothesis_enclosed_by_nonzero),
            ("horiz_enclose", hypothesis_horiz_enclose),
        ]:
            pred = fn(inp)
            acc = score(pred, expected)
            green_pred_count = np.sum(pred == 3)
            green_exp_count = np.sum(expected == 3)
            print(f"  {name}: acc={acc:.4f}, green_pred={green_pred_count}, green_exp={green_exp_count}")
