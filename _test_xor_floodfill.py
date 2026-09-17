import json
import numpy as np
from collections import deque

data = json.load(open("data/neurogolf-2026/raw/task255.json"))

def compute_seeds(inp, mode="xor"):
    h, w = inp.shape
    mask = inp > 0
    
    left_row = np.zeros_like(mask)
    for c in range(1, w):
        left_row[:, c] = left_row[:, c-1] | mask[:, c-1]
    
    right_row = np.zeros_like(mask)
    for c in range(w-2, -1, -1):
        right_row[:, c] = right_row[:, c+1] | mask[:, c+1]
    
    above_col = np.zeros_like(mask)
    for r in range(1, h):
        above_col[r] = above_col[r-1] | mask[r-1]
    
    below_col = np.zeros_like(mask)
    for r in range(h-2, -1, -1):
        below_col[r] = below_col[r+1] | mask[r+1]
    
    vert_enclose = above_col & below_col
    horiz_enclose = left_row & right_row
    
    if mode == "xor":
        seeds = (horiz_enclose ^ vert_enclose) & ~mask
    elif mode == "h_only":
        seeds = horiz_enclose & ~mask
    elif mode == "h_not_v":
        seeds = horiz_enclose & ~vert_enclose & ~mask
    elif mode == "v_not_h":
        seeds = vert_enclose & ~horiz_enclose & ~mask
    elif mode == "h_or_v":
        seeds = (horiz_enclose | vert_enclose) & ~mask
    elif mode == "h_not_v_or_v_not_h":
        seeds = ((horiz_enclose & ~vert_enclose) | (vert_enclose & ~horiz_enclose)) & ~mask
        # same as xor
    elif mode == "h_or_v_no_both":
        seeds = (horiz_enclose | vert_enclose) & ~(horiz_enclose & vert_enclose) & ~mask
        # same as xor
        
    return seeds

def flood_fill_from_seeds(inp, seeds, steps=None):
    """Flood fill 4-directionally from seeds into background (0) cells.
    If steps is None, fill until exhaustion. Otherwise, fill for 'steps' iterations."""
    h, w = inp.shape
    filled = seeds.copy()
    if steps is None:
        # Fill until exhaustion
        changed = True
        while changed:
            changed = False
            new_filled = filled.copy()
            for r in range(h):
                for c in range(w):
                    if inp[r, c] == 0 and not filled[r, c]:
                        # Check 4 neighbors
                        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < h and 0 <= nc < w and filled[nr, nc]:
                                new_filled[r, c] = True
                                changed = True
                                break
            filled = new_filled
    else:
        for _ in range(steps):
            new_filled = filled.copy()
            for r in range(h):
                for c in range(w):
                    if inp[r, c] == 0 and not filled[r, c]:
                        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < h and 0 <= nc < w and filled[nr, nc]:
                                new_filled[r, c] = True
                                break
            filled = new_filled
    
    out = inp.copy()
    out[filled] = 3
    return out

def test(inp, expected, seeds_fn, flood_steps=None):
    seeds = seeds_fn(inp)
    out = flood_fill_from_seeds(inp, seeds, flood_steps)
    acc = np.sum((out == 3) == (expected == 3)) / expected.size
    ov = np.sum((out == 3) & (expected == 3))
    fp = np.sum((out == 3) & ~(expected == 3))
    fn = np.sum(~(out == 3) & (expected == 3))
    return acc, ov, fp, fn, np.sum(seeds), np.sum(out == 3)

print("=== XOR seeds + flood fill ===")
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        ge = np.sum(expected == 3)
        print(f"\n{split} {i} (expected green: {ge}):")
        
        for mode, desc in [
            ("xor", "XOR seeds"),
            ("h_not_v", "H&~V seeds"),
            ("v_not_h", "V&~H seeds"),
            ("h_or_v", "H|V seeds"),
        ]:
            fn = lambda x, m=mode: compute_seeds(x, m)
            # No flood fill (just seeds)
            acc, ov, fp, fn_c, ns, npred = test(inp, expected, fn, 0)
            print(f"  {desc} (no flood): acc={acc:.4f}, ov={ov}, FP={fp}, FN={fn_c}, seeds={ns}, pred={npred}")
            
            # Flood fill to exhaustion
            acc, ov, fp, fn_c, ns, npred = test(inp, expected, fn, None)
            print(f"  {desc} (flood all): acc={acc:.4f}, ov={ov}, FP={fp}, FN={fn_c}, seeds={ns}, pred={npred}")

# Also try H and V both -> but NOT filling H&V
print("\n\n=== Refined: H|V but exclude H&V, then flood ===")
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        ge = np.sum(expected == 3)
        
        seeds = compute_seeds(inp, "xor")
        # XOR = H xor V = (H|V) & ~(H&V)
        
        for steps in [0, 1, 2, 3, 5, None]:
            out = flood_fill_from_seeds(inp, seeds, steps)
            acc = np.sum((out == 3) == (expected == 3)) / expected.size
            ov = np.sum((out == 3) & (expected == 3))
            fp = np.sum((out == 3) & ~(expected == 3))
            fn2 = np.sum(~(out == 3) & (expected == 3))
            npred = np.sum(out == 3)
            print(f"  {split} {i} steps={steps}: acc={acc:.4f}, ov={ov}/{ge}, FP={fp}, FN={fn2}, pred={npred}")
