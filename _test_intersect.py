import json
import numpy as np
from collections import deque

data = json.load(open("data/neurogolf-2026/raw/task255.json"))

def compute_green(inp):
    """
    Hypothesis: Fill a cell with green if BOTH:
    1. In its row, there's non-zero to the left AND right
    2. In its column, there's non-zero above AND below
    
    BUT the "left" and "right" are not any column - they must be
    the SAME row left/right. And "above" and "below" are same column.
    """
    h, w = inp.shape
    mask = inp > 0
    
    # Row-wise: left and right
    left_row = np.zeros_like(mask)
    for c in range(1, w):
        left_row[:, c] = left_row[:, c-1] | mask[:, c-1]
    
    right_row = np.zeros_like(mask)
    for c in range(w-2, -1, -1):
        right_row[:, c] = right_row[:, c+1] | mask[:, c+1]
    
    # Column-wise: above and below
    above_col = np.zeros_like(mask)
    for r in range(1, h):
        above_col[r] = above_col[r-1] | mask[r-1]
    
    below_col = np.zeros_like(mask)
    for r in range(h-2, -1, -1):
        below_col[r] = below_col[r+1] | mask[r+1]
    
    # Both conditions: row-wise enclosed AND column-wise enclosed
    both = left_row & right_row & above_col & below_col & ~mask
    return both

def compute_green_row_col_separate(inp):
    """
    Fill if row-wise enclosed OR column-wise enclosed
    """
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
    
    row_enclose = left_row & right_row
    col_enclose = above_col & below_col
    
    # XOR: exactly one of row_enclose or col_enclose, but not both
    result = (row_enclose ^ col_enclose) & ~mask
    return result

def compute_green_vert_same_column(inp):
    """
    Fill if there's non-zero above AND below in the SAME column.
    This fills vertical stripes based on column occupancy.
    """
    h, w = inp.shape
    mask = inp > 0
    
    # For each column, find topmost and bottommost non-zero
    result = np.zeros_like(mask)
    for c in range(w):
        col_nonzero = np.where(mask[:, c])[0]
        if len(col_nonzero) >= 2:
            top = col_nonzero[0]
            bottom = col_nonzero[-1]
            for r in range(top + 1, bottom):
                if not mask[r, c]:
                    result[r, c] = True
    
    return result

def compute_green_row_same_row(inp):
    """
    Fill if there's non-zero left AND right in the SAME row.
    """
    h, w = inp.shape
    mask = inp > 0
    
    result = np.zeros_like(mask)
    for r in range(h):
        nz = np.where(mask[r])[0]
        if len(nz) >= 2:
            left = nz[0]
            right = nz[-1]
            for c in range(left + 1, right):
                if not mask[r, c]:
                    result[r, c] = True
    
    return result

def score(pred, expected):
    return np.sum((pred == 3) == (expected == 3)) / expected.size

print("=== Column-row intersection/union tests ===")
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        green_exp = np.sum(expected == 3)
        print(f"\n{split} {i} (expected green: {green_exp})")
        
        for name, fn in [
            ("row_col_and (intersection)", compute_green),
            ("row_col_xor", compute_green_row_col_separate),
            ("vert_same_col_only", lambda x: compute_green_vert_same_column(x)),
            ("row_same_row_only", lambda x: compute_green_row_same_row(x)),
        ]:
            green = fn(inp)
            out = inp.copy()
            out[green] = 3
            acc = score(out, expected)
            gp = np.sum(green)
            print(f"  {name}: acc={acc:.4f}, g={gp}")

# Also try: row gap fill then column gap fill (or vice versa)
def green_row_then_col(inp):
    """First fill row gaps, then fill column gaps of result"""
    h, w = inp.shape
    
    # Step 1: row gap fill
    working = inp.copy()
    for r in range(h):
        nz = np.where(inp[r] > 0)[0]
        if len(nz) >= 2:
            for c in range(nz[0]+1, nz[-1]):
                if working[r, c] == 0:
                    working[r, c] = 3
    
    # Step 2: column gap fill (considering 3 as filled)
    mask = working > 0
    for c in range(w):
        col_nonzero = np.where(mask[:, c])[0]
        if len(col_nonzero) >= 2:
            for r in range(col_nonzero[0]+1, col_nonzero[-1]):
                if working[r, c] == 0:
                    working[r, c] = 3
    
    return working

def green_col_then_row(inp):
    """First fill column gaps, then fill row gaps"""
    h, w = inp.shape
    
    working = inp.copy()
    mask = inp > 0
    for c in range(w):
        col_nonzero = np.where(mask[:, c])[0]
        if len(col_nonzero) >= 2:
            for r in range(col_nonzero[0]+1, col_nonzero[-1]):
                if working[r, c] == 0:
                    working[r, c] = 3
    
    mask2 = working > 0
    for r in range(h):
        nz = np.where(mask2[r])[0]
        if len(nz) >= 2:
            for c in range(nz[0]+1, nz[-1]):
                if working[r, c] == 0:
                    working[r, c] = 3
    
    return working

print("\n=== Sequential fill tests ===")
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        print(f"\n{split} {i}:")
        for name, fn in [
            ("row_then_col", green_row_then_col),
            ("col_then_row", green_col_then_row),
        ]:
            out = fn(inp)
            acc = score(out, expected)
            gp = np.sum(out == 3)
            ge = np.sum(expected == 3)
            print(f"  {name}: acc={acc:.4f}, g={gp}, expected={ge}")

# Let me check what cells are DIFFERENT between actual and row-col intersection
print("\n\n=== Error analysis for row_col_and ===")
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        green_pred = compute_green(inp)
        green_actual = (expected == 3)
        
        false_pos = green_pred & ~green_actual
        false_neg = ~green_pred & green_actual
        
        print(f"\n{split} {i}: FP={np.sum(false_pos)}, FN={np.sum(false_neg)}")
        if np.sum(false_neg) > 0 and np.sum(false_neg) <= 50:
            print("  False negatives (should be green but not predicted):")
            for r, c in zip(*np.where(false_neg)):
                print(f"    ({r},{c}) : row_has_left={np.any(inp[r, :c]>0)}, row_has_right={np.any(inp[r, c+1:]>0)}, col_has_above={np.any(inp[:r, c]>0)}, col_has_below={np.any(inp[r+1:, c]>0)}")
