import json
import numpy as np

data = json.load(open("data/neurogolf-2026/raw/task255.json"))

for split in ["train"]:
    for i, ex in enumerate(data[split]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        mask_nonzero = inp > 0
        mask_green = out == 3
        
        # Find rows where green appears
        green_rows = np.any(mask_green, axis=1)
        print(f"\n=== {split} {i} ===")
        print(f"Green rows: {np.where(green_rows)[0].tolist()}")
        
        # For each green row, find leftmost and rightmost green
        for r in np.where(green_rows)[0]:
            green_cols = np.where(mask_green[r])[0]
            if len(green_cols):
                # Find nearest non-zero input on left and right of green region
                lieft_cols = np.where(inp[r] > 0)[0]
                left_of_green = lieft_cols[lieft_cols < green_cols[0]]
                right_of_green = lieft_cols[lieft_cols > green_cols[-1]]
                
                left_boundary = left_of_green[-1] if len(left_of_green) else None
                right_boundary = right_of_green[0] if len(right_of_green) else None
                print(f"  Row {r}: green cols [{green_cols[0]}-{green_cols[-1]}], "
                      f"left input boundary={left_boundary}, right input boundary={right_boundary}")

# Also check: is the green region a single connected component?
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        green = (out == 3).astype(int)
        
        # Find bounding box of green
        green_rows = np.any(green, axis=1)
        green_cols = np.any(green, axis=0)
        if np.any(green_rows):
            rmin, rmax = np.where(green_rows)[0][[0, -1]]
            cmin, cmax = np.where(green_cols)[0][[0, -1]]
            print(f"\n{split} {i}: Green bbox: rows [{rmin},{rmax}], cols [{cmin},{cmax}], "
                  f"size {(rmax-rmin+1)}x{(cmax-cmin+1)}")
            print(f"  Green filled fraction in bbox: {green[rmin:rmax+1, cmin:cmax+1].mean():.3f}")

# Check if green region corresponds to 0s in input that are enclosed by non-zero
print("\n\n=== Green corresponds to input zeros enclosed by non-zero? ===")
for split in ["train", "test"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        green_added = (out == 3) & (inp == 0)
        nongreen_same = (inp > 0) & (out == inp)
        
        total_green = np.sum(out == 3)
        all_green_from_zero = np.all((out == 3) & (inp == 0))
        print(f"{split} {i}: Green pixels added where input was 0: {np.sum(green_added)}/{total_green} "
              f"({'' if all_green_from_zero else 'NOT '}all from zeros)")
