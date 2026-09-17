import json
import numpy as np

data = json.load(open("data/neurogolf-2026/raw/task255.json"))

def compute_axial4dir(inp):
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

ex = data["train"][0]
inp = np.array(ex["input"])
expected = np.array(ex["output"])

green_pred = compute_axial4dir(inp)
green_act = (expected == 3)

print(f"Predicted green cells: {np.sum(green_pred)}")
print(f"Actual green cells: {np.sum(green_act)}")
print(f"Overlap: {np.sum(green_pred & green_act)}")
print(f"FP: {np.sum(green_pred & ~green_act)}")
print(f"FN: {np.sum(~green_pred & green_act)}")

# Check specific cells
print("\n=== Row 5 check ===")
for c in range(30):
    pred = green_pred[5, c]
    act = green_act[5, c]
    inp_val = inp[5, c]
    l = np.any(inp[5, :c] > 0)
    r = np.any(inp[5, c+1:] > 0)
    a = np.any(inp[:5, c] > 0)
    b = np.any(inp[6:, c] > 0)
    if pred or act:
        print(f"  col {c}: inp={inp_val}, pred={'Y' if pred else 'N'}, act={'Y' if act else 'N'}, "
              f"L={l} R={r} A={a} B={b}")

# Let me find one cell that IS green in expected but NOT predicted
print("\n=== FN examples (green in expected but not predicted) ===")
fn_cells = np.where(~green_pred & green_act)
for idx in range(min(5, len(fn_cells[0]))):
    r, c = fn_cells[0][idx], fn_cells[1][idx]
    l = np.any(inp[r, :c] > 0)
    ri = np.any(inp[r, c+1:] > 0)
    a = np.any(inp[:r, c] > 0)
    b = np.any(inp[r+1:, c] > 0)
    print(f"  ({r},{c}): inp={inp[r,c]}, L={l} R={ri} A={a} B={b}")

# Find one cell that IS predicted but NOT green in expected
print("\n=== FP examples (predicted but not green in expected) ===")
fp_cells = np.where(green_pred & ~green_act)
for idx in range(min(5, len(fp_cells[0]))):
    r, c = fp_cells[0][idx], fp_cells[1][idx]
    l = np.any(inp[r, :c] > 0)
    ri = np.any(inp[r, c+1:] > 0)
    a = np.any(inp[:r, c] > 0)
    b = np.any(inp[r+1:, c] > 0)
    print(f"  ({r},{c}): inp={inp[r,c]}, L={l} R={ri} A={a} B={b}")
