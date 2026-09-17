from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples
from prototype_task319 import analyze_objects


TASK = 319


def surrogate_compress(arr: np.ndarray) -> np.ndarray:
    keep_rows = [0]
    for idx in range(1, arr.shape[0]):
        if not np.array_equal(arr[idx], arr[idx - 1]):
            keep_rows.append(idx)
    arr = arr[keep_rows, :]

    keep_cols = [0]
    for idx in range(1, arr.shape[1]):
        if not np.array_equal(arr[:, idx], arr[:, idx - 1]):
            keep_cols.append(idx)
    return arr[:, keep_cols]


def main() -> None:
    examples = all_examples(TASK, pathlib.Path("data/neurogolf-2026/raw"))
    total = 0
    mismatches: list[tuple[int, int, tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    feature_mismatches: list[tuple[int, int, str, float, float]] = []

    for idx, example in enumerate(examples):
        inp = np.array(example["input"], dtype=np.int8)
        for obj in analyze_objects(inp):
            total += 1
            exact = obj["compressed"]
            surrogate = surrogate_compress(obj["patch"])
            if surrogate.shape != exact.shape or not np.array_equal(surrogate, exact):
                mismatches.append((idx, obj["color"], obj["patch"].shape, exact.shape, surrogate.shape))
                continue

            exact_density = float(exact.sum() / exact.size)
            surrogate_density = float(surrogate.sum() / surrogate.size)
            if exact_density != surrogate_density:
                feature_mismatches.append((idx, obj["color"], "density", exact_density, surrogate_density))

            exact_row_var = float(np.var(exact.sum(axis=1)))
            surrogate_row_var = float(np.var(surrogate.sum(axis=1)))
            if exact_row_var != surrogate_row_var:
                feature_mismatches.append((idx, obj["color"], "row_var", exact_row_var, surrogate_row_var))

            exact_row_range = int(exact.sum(axis=1).max() - exact.sum(axis=1).min())
            surrogate_row_range = int(surrogate.sum(axis=1).max() - surrogate.sum(axis=1).min())
            if exact_row_range != surrogate_row_range:
                feature_mismatches.append((idx, obj["color"], "row_range", exact_row_range, surrogate_row_range))

    print(f"task{TASK:03d} graph-feature audit on {total} objects")
    print(f"compression mismatches: {len(mismatches)}")
    print(f"feature mismatches: {len(feature_mismatches)}")
    if mismatches:
        print("first compression mismatches", mismatches[:10])
    if feature_mismatches:
        print("first feature mismatches", feature_mismatches[:10])


if __name__ == "__main__":
    main()
