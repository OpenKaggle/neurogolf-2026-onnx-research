import json
import numpy as np
from collections import deque

data = json.load(open("data/neurogolf-2026/raw/task255.json"))

def dilate(mask, k=1):
    h, w = mask.shape
    result = mask.copy()
    for _ in range(k):
        dilated = result.copy()
        for r in range(h):
            for c in range(w):
                if result[r, c]:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < h and 0 <= nc < w:
                                dilated[nr, nc] = 1
        result = dilated
    return result

def erode(mask, k=1):
    h, w = mask.shape
    result = mask.copy()
    for _ in range(k):
        eroded = result.copy()
        for r in range(h):
            for c in range(w):
                if result[r, c]:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if not mask[nr, nc]:
                                    eroded[r, c] = 0
                                    break
                        else:
                            continue
                        break
        result = eroded
    return result

def fill_holes(binary):
    """Fill holes in binary mask using flood fill from border"""
    h, w = binary.shape
    visited = np.zeros_like(binary)
    q = deque()
    for r in range(h):
        for c in [0, w-1]:
            if binary[r, c] == 0 and not visited[r, c]:
                visited[r, c] = 1
                q.append((r, c))
    for c in range(w):
        for r in [0, h-1]:
            if binary[r, c] == 0 and not visited[r, c]:
                visited[r, c] = 1
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] == 0 and not visited[nr, nc]:
                visited[nr, nc] = 1
                q.append((nr, nc))
    # Holes are 0-cells not visited
    holes = (binary == 0) & (visited == 0)
    filled = binary.copy()
    filled[holes] = 1
    return filled

def hypothesis_morph_close(inp, dilate_k=3):
    """Morphological closing: dilate, fill holes, erode back"""
    mask = (inp > 0).astype(np.uint8)
    # Dilate to connect components
    dilated = dilate(mask, dilate_k)
    # Fill holes
    closed = fill_holes(dilated)
    # Erode back
    eroded = erode(closed, dilate_k)
    # Green fills 0-cells that became 1 in the closed image
    green_mask = (mask == 0) & (eroded == 1)
    out = inp.copy()
    out[green_mask] = 3
    return out

def hypothesis_gap_fill_then_dilate(inp):
    """Row gap fill THEN dilate"""
    out = inp.copy()
    h, w = inp.shape
    # First pass: fill between leftmost and rightmost non-zero per row
    for r in range(h):
        nz = np.where(inp[r] > 0)[0]
        if len(nz) >= 2:
            out[r, nz[0]+1:nz[-1]] = np.where(out[r, nz[0]+1:nz[-1]] == 0, 3, out[r, nz[0]+1:nz[-1]])
    # Dilate the green
    green = (out == 3)
    dilated = dilate(green, 1)
    out[dilated & (inp == 0)] = 3
    return out

def hypothesis_rowcol_union(inp):
    """Fill if non-zero above AND below vertically, OR left AND right horizontally"""
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
    
    # Union of vertical AND horizontal conditions
    vert = above & below
    horiz = left & right
    fill = (vert | horiz) & ~mask
    out = inp.copy()
    out[fill] = 3
    return out

def hypothesis_rowcol_and(inp):
    """Fill if (non-zero above OR left) AND (non-zero below OR right)"""
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
    
    upleft = above | left
    downright = below | right
    fill = upleft & downright & ~mask
    out = inp.copy()
    out[fill] = 3
    return out

def hypothesis_vert_and_horiz_or(inp):
    """Fill if (above AND below) AND (left OR right)"""
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
    
    fill = above & below & (left | right) & ~mask
    out = inp.copy()
    out[fill] = 3
    return out

def score(pred, expected):
    return np.sum((pred == 3) == (expected == 3)) / expected.size

print("=== Hypothesis testing ===")
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        green_exp = np.sum(expected == 3)
        print(f"\n{split} {i} (expected green: {green_exp})")
        for name, fn in [
            ("morph_close_k2", lambda x: hypothesis_morph_close(x, 2)),
            ("morph_close_k3", lambda x: hypothesis_morph_close(x, 3)),
            ("morph_close_k4", lambda x: hypothesis_morph_close(x, 4)),
            ("row_gap+dilate", hypothesis_gap_fill_then_dilate),
            ("rowcol_union", hypothesis_rowcol_union),
            ("rowcol_and", hypothesis_rowcol_and),
            ("vert_and_horiz_or", hypothesis_vert_and_horiz_or),
        ]:
            pred = fn(inp)
            acc = score(pred, expected)
            gp = np.sum(pred == 3)
            print(f"  {name}: acc={acc:.4f}, g={gp}")
