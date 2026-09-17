import json
import numpy as np
from collections import deque

data = json.load(open("data/neurogolf-2026/raw/task255.json"))

def hypothesis_axial(inp):
    """Fill if hitting non-zero in all 4 directions"""
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
    
    fill = above & below & left & right & ~mask
    out = inp.copy()
    out[fill] = 3
    return out

def hypothesis_axial_vert(inp):
    """Fill if hitting non-zero above AND below"""
    h, w = inp.shape
    mask = inp > 0
    
    above = np.zeros_like(mask)
    for r in range(1, h):
        above[r] = above[r-1] | mask[r-1]
    
    below = np.zeros_like(mask)
    for r in range(h-2, -1, -1):
        below[r] = below[r+1] | mask[r+1]
    
    fill = above & below & ~mask
    out = inp.copy()
    out[fill] = 3
    return out

def hypothesis_axial_horiz(inp):
    """Fill if hitting non-zero left AND right"""
    h, w = inp.shape
    mask = inp > 0
    
    left = np.zeros_like(mask)
    for c in range(1, w):
        left[:, c] = left[:, c-1] | mask[:, c-1]
    
    right = np.zeros_like(mask)
    for c in range(w-2, -1, -1):
        right[:, c] = right[:, c+1] | mask[:, c+1]
    
    fill = left & right & ~mask
    out = inp.copy()
    out[fill] = 3
    return out

def hypothesis_flood_fill_8dir(inp):
    """Fill 0-cells not connected to border via 8-direction connectivity"""
    h, w = inp.shape
    mask = (inp == 0).astype(np.uint8)
    q = deque()
    visited = np.zeros_like(mask)
    
    for r in range(h):
        for c in [0, w-1]:
            if mask[r, c] and not visited[r, c]:
                visited[r, c] = 1
                q.append((r, c))
    for c in range(w):
        for r in [0, h-1]:
            if mask[r, c] and not visited[r, c]:
                visited[r, c] = 1
                q.append((r, c))
    
    while q:
        r, c = q.popleft()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = 1
                    q.append((nr, nc))
    
    filled = (mask == 1) & (visited == 0)
    out = inp.copy()
    out[filled] = 3
    return out

def hypothesis_fill_gap_union_allrows(inp):
    """Fill 0-cells between leftmost and rightmost non-zero per row"""
    h, w = inp.shape
    out = inp.copy()
    for r in range(h):
        nz = np.where(inp[r] > 0)[0]
        if len(nz) >= 2:
            left, right = nz[0], nz[-1]
            out[r, left+1:right] = np.where(out[r, left+1:right] == 0, 3, out[r, left+1:right])
    return out

# Manual connected components
def get_components(mask):
    h, w = mask.shape
    labeled = np.zeros_like(mask, dtype=int)
    current = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and labeled[r, c] == 0:
                current += 1
                q = deque()
                q.append((r, c))
                labeled[r, c] = current
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labeled[nr, nc] == 0:
                            labeled[nr, nc] = current
                            q.append((nr, nc))
    return labeled, current

for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        green_exp = np.sum(expected == 3)
        print(f"\n=== {split} {i} (expected green: {green_exp}) ===")
        for name, fn in [
            ("axial_4dir", hypothesis_axial),
            ("axial_vert", hypothesis_axial_vert),
            ("axial_horiz", hypothesis_axial_horiz),
            ("flood_fill_8dir", hypothesis_flood_fill_8dir),
            ("row_gap_bothsides", hypothesis_fill_gap_union_allrows),
        ]:
            pred = fn(inp)
            acc = np.sum((pred == 3) == (expected == 3)) / expected.size
            green_pred = np.sum(pred == 3)
            print(f"  {name}: acc={acc:.4f}, green_pred={green_pred}")

# Component analysis
print("\n\n=== Connected components of non-zero ===")
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        mask = inp > 0
        labeled, n = get_components(mask)
        sizes = [np.sum(labeled == j+1) for j in range(n)]
        print(f"{split} {i}: {n} components, sizes: {sorted(sizes, reverse=True)[:5]}")

# Try: fill based on component separation
def hypothesis_component_separation(inp):
    """Find the x-coordinate that separates components, fill 0s inside"""
    h, w = inp.shape
    mask = inp > 0
    labeled, n = get_components(mask)
    
    if n < 2:
        return inp.copy()
    
    # Find the bounding boxes of the two largest components
    sizes = [(np.sum(labeled == j+1), j+1) for j in range(n)]
    sizes.sort(reverse=True)
    comp_labels = [s[1] for s in sizes[:2]]
    
    # Get bounding boxes
    boxes = []
    for lbl in comp_labels:
        ys, xs = np.where(labeled == lbl)
        boxes.append((xs.min(), xs.max(), ys.min(), ys.max()))
    
    # Leftmost component and rightmost component
    left_comp = min(boxes, key=lambda b: b[0])
    right_comp = max(boxes, key=lambda b: b[1])
    
    left_right_edge = left_comp[1]  # max x of left component
    right_left_edge = right_comp[0]  # min x of right component
    
    out = inp.copy()
    for r in range(h):
        for c in range(left_right_edge, right_left_edge + 1):
            if out[r, c] == 0:
                out[r, c] = 3
    
    return out

print("\n\n=== Component separation hypothesis ===")
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        pred = hypothesis_component_separation(inp)
        acc = np.sum((pred == 3) == (expected == 3)) / expected.size
        green_pred = np.sum(pred == 3)
        green_exp = np.sum(expected == 3)
        print(f"{split} {i}: acc={acc:.4f}, green_pred={green_pred}, green_exp={green_exp}")
