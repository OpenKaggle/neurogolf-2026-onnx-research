import json
import numpy as np

data = json.load(open("data/neurogolf-2026/raw/task255.json"))

def classify(inp):
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
    
    return left, right, above, below, mask

for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        left, right, above, below, mask = classify(inp)
        green = (expected == 3)
        
        # Compute all categories
        counted = np.zeros_like(green, dtype=bool)
        cats = {
            "L&R&A&~B": left & right & above & ~below,
            "L&R&~A&B": left & right & ~above & below,
            "L&R&~A&~B": left & right & ~above & ~below,
            "L&~R&A&B": left & ~right & above & below,
            "~L&R&A&B": ~left & right & above & below,
            "~L&~R&A&B": ~left & ~right & above & below,
            "L&~R&A&~B": left & ~right & above & ~below,
            "~L&R&A&~B": ~left & right & above & ~below,
            "L&~R&~A&B": left & ~right & ~above & below,
            "~L&R&~A&B": ~left & right & ~above & below,
        }
        for name, cat in cats.items():
            counted = counted | (green & cat)
        
        other = green & ~counted
        n_other = np.sum(other)
        print(f"\n{split} {i}: other={n_other}")
        
        if n_other > 0 and n_other <= 30:
            for r, c in zip(*np.where(other)):
                print(f"  ({r},{c}): inp={inp[r,c]}, L={left[r,c]}, R={right[r,c]}, A={above[r,c]}, B={below[r,c]}")
        elif n_other > 30:
            print(f"  Too many to list ({n_other}), showing first 10:")
            coords = list(zip(*np.where(other)))
            for r, c in coords[:10]:
                print(f"  ({r},{c}): inp={inp[r,c]}, L={left[r,c]}, R={right[r,c]}, A={above[r,c]}, B={below[r,c]}")
            # Show pattern
            other_rows = sorted(set(r for r, c in coords[:30]))
            other_cols = sorted(set(c for r, c in coords[:30]))
            print(f"  Sample rows: {other_rows[:10]}, cols: {other_cols[:10]}")
