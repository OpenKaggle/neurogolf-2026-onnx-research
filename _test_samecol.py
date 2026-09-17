import json
import numpy as np
from collections import deque

data = json.load(open("data/neurogolf-2026/raw/task255.json"))

def compute_all(inp):
    h, w = inp.shape
    mask = inp > 0
    
    # Horizontal: non-zero left/right in SAME ROW (any column)
    left_row = np.zeros_like(mask)
    for c in range(1, w):
        left_row[:, c] = left_row[:, c-1] | mask[:, c-1]
    
    right_row = np.zeros_like(mask)
    for c in range(w-2, -1, -1):
        right_row[:, c] = right_row[:, c+1] | mask[:, c+1]
    
    # Vertical: non-zero above/below in SAME COLUMN (any row)
    above_col = np.zeros_like(mask)
    for r in range(1, h):
        above_col[r] = above_col[r-1] | mask[r-1]
    
    below_col = np.zeros_like(mask)
    for r in range(h-2, -1, -1):
        below_col[r] = below_col[r+1] | mask[r+1]
    
    # But wait - above_col and below_col check ANY row in the SAME column.
    # So above_col[r,c] = True if mask[0:r, c] has any True (non-zero at column c above row r)
    # Wait, that's not what the code does. Let me re-check.
    # above[r] = above[r-1] | mask[r-1]
    # So above[1] = above[0] | mask[0] = mask[0]
    # above[2] = above[1] | mask[1] = mask[0] | mask[1]
    # This is checking if any row BEFORE r has non-zero... but in the SAME column.
    # Because it uses element-wise |, not column-aggregated | (which would be .any()).
    # Actually wait: "above[r-1]" means the entire array above[r-1]. And mask[r-1] is the entire row.
    # So the | is element-wise: above[r][r,c] = above[r-1][r,c] | mask[r-1][c]
    # This means: above[r][r,c] checks column c only!
    
    # Hmm wait, the indexing is wrong. Let me re-examine.
    # above[r] (a 2D array) = above[r-1] (a 2D array) | mask[r-1] (a 1D array of size w)
    # In numpy, | broadcasts: mask[r-1] is a row vector, and above[r-1] is a 2D array.
    # So mask[r-1, :] | above[r-1, :, :] repeats mask[r-1] across all rows of above[r-1].
    
    # Actually wait, above[r-1] IS a 2D array of shape (h, w). And mask[r-1] is a 1D array of shape (w,).
    # The | does element-wise: above[r][i, j] = above[r-1][i, j] | mask[r-1][j]
    # But above[r-1][i, j] for i != r-1 would be some accumulated value from earlier iterations.
    # This doesn't make sense for "does column j have non-zero above row r".
    
    # STOP: The code is buggy. above[r] should be an array where above[r][r, j] tells us if column j has non-zero above.
    # But the 2D structure means above[r][i, j] for i != r would be wrong.
    
    # Let me fix: I should compute per-column, not as a 2D array.
    # Actually, let me recheck the original code:
    # above = np.zeros_like(mask)   # shape (h, w)
    # for r in range(1, h):
    #     above[r] = above[r-1] | mask[r-1]
    
    # above[r] is ROW r of the array. above[r-1] is ROW r-1 of the array.
    # mask[r-1] is ROW r-1 of the mask.
    # So above[r] = above[r-1] | mask[r-1] computes ROW r of the result.
    # above[r] is a 1D array of length w.
    # above[r][c] = above[r-1][c] | mask[r-1][c]
    # This IS checking if column c has non-zero above row r (in any of rows 0 to r-1).
    # And it's row-by-row, not column-by-column.
    
    # So above[r][c] = True if any of rows 0..r-1 in column c has non-zero.
    # That's correct! Similarly for below[r][c].
    
    return left_row, right_row, above_col, below_col

def test_condition(inp, expected, desc, condition_fn):
    """Test a condition and return overlap/FN/FP"""
    green = condition_fn(inp)
    green_expected = (expected == 3)
    overlap = np.sum(green & green_expected)
    fp = np.sum(green & ~green_expected)
    fn = np.sum(~green & green_expected)
    return overlap, fp, fn

def condition_same_col(inp):
    """Fill if horizontally enclosed AND NOT vertically enclosed in same column"""
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
    
    # Vertically enclosed: has non-zero above AND below in SAME column
    vert_enclose = above_col & below_col
    
    # Condition: horizontally enclosed AND NOT vertically enclosed
    cond = left_row & right_row & ~vert_enclose & ~mask
    return cond

def condition_vert_only(inp):
    """Fill if vertically enclosed AND NOT horizontally enclosed"""
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
    
    cond = vert_enclose & ~horiz_enclose & ~mask
    return cond

def condition_xor(inp):
    """XOR of horizontal and vertical enclosure"""
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
    
    cond = (horiz_enclose ^ vert_enclose) & ~mask
    return cond

def condition_neither(inp):
    """Neither horizontally nor vertically enclosed"""
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
    
    cond = ~vert_enclose & ~horiz_enclose & ~mask
    return cond

def condition_either(inp):
    """Either horizontally OR vertically enclosed (inclusive)"""
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
    
    cond = (horiz_enclose | vert_enclose) & ~mask
    return cond

print("=== Same-column vertical enclosure tests ===")
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        green_exp = np.sum(expected == 3)
        print(f"\n{split} {i} (expected green: {green_exp}):")
        
        tests = [
            ("H&~V (same col)", condition_same_col),
            ("V&~H (same col)", condition_vert_only),
            ("H xor V", condition_xor),
            ("~H & ~V", condition_neither),
            ("H | V", condition_either),
        ]
        
        for name, fn in tests:
            green = fn(inp)
            ov, fp, fn2 = test_condition(inp, expected, name, fn)
            gp = np.sum(green)
            acc = (np.sum(expected == 3) + np.sum(expected != 3)) - fp - fn2
            acc = (np.sum((expected == 3) == green)) / expected.size
            print(f"  {name}: acc={acc:.4f}, g={gp}, ov={ov}, FP={fp}, FN={fn2}")
