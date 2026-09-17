import json
import numpy as np

data = json.load(open("data/neurogolf-2026/raw/task255.json"))
print("train examples:", len(data["train"]))
print("test examples:", len(data.get("test", [])))
print("arc-gen examples:", len(data.get("arc-gen", [])))

for i, ex in enumerate(data["train"]):
    inp = ex["input"]
    out = ex["output"]
    in_arr = np.array(inp)
    out_arr = np.array(out)
    print(f"\n=== Train {i} ===")
    print(f"Input shape: {in_arr.shape}, colors: {sorted(np.unique(in_arr).tolist())}")
    print(f"Output shape: {out_arr.shape}, colors: {sorted(np.unique(out_arr).tolist())}")
    print("Input:")
    for row in inp:
        print(row)
    print("Output:")
    for row in out:
        print(row)

# Also print test/arc-gen
for split in ["test", "arc-gen"]:
    for i, ex in enumerate(data.get(split, [])):
        inp = ex["input"]
        out = ex["output"]
        in_arr = np.array(inp)
        out_arr = np.array(out)
        print(f"\n=== {split} {i} ===")
        print(f"Input shape: {in_arr.shape}, colors: {sorted(np.unique(in_arr).tolist())}")
        print(f"Output shape: {out_arr.shape}, colors: {sorted(np.unique(out_arr).tolist())}")
        print("Input:")
        for row in inp:
            print(row)
        print("Output:")
        for row in out:
            print(row)
