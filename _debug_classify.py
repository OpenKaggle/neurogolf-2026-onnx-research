import json
import numpy as np

data = json.load(open("data/neurogolf-2026/raw/task255.json"))

def classify_green_cells(inp, expected):
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
    
    green = (expected == 3)
    nz_cells = (expected > 0) & (expected != 3)
    
    categories = {
        "L&R&A&B": left & right & above & below,
        "L&R&A&~B": left & right & above & ~below,
        "L&R&~A&B": left & right & ~above & below,
        "L&R&~A&~B": left & right & ~above & ~below,
        "L&~R&A&B": left & ~right & above & below,
        "~L&R&A&B": ~left & right & above & below,
        "~L&~R&A&B": ~left & ~right & above & below,
        "L&~R&A&~B": left & ~right & above & ~below,
        "~L&R&~A&B": ~left & right & ~above & below,
        "L&~R&~A&B": left & ~right & ~above & below,
        "~L&R&A&~B": ~left & right & above & ~below,
        "other": None  # will compute as remainder
    }
    
    total = 0
    for name, cat in categories.items():
        if name == "other":
            continue
        count = np.sum(green & cat)
        total += count
        print(f"    {name}: {count}")
    
    other = np.sum(green) - total
    print(f"    other: {other}")
    
    return categories

for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        expected = np.array(ex["output"])
        print(f"\n{split} {i} (green: {np.sum(expected==3)}):")
        classify_green_cells(inp, expected)
