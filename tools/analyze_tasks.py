from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np


def bbox(grid: np.ndarray, ignore: int | None = 0) -> tuple[int, int, int, int] | None:
    mask = np.ones_like(grid, dtype=bool) if ignore is None else grid != ignore
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def summarize_grid(grid: list[list[int]]) -> dict:
    arr = np.array(grid, dtype=np.int64)
    counts = collections.Counter(arr.ravel().tolist())
    return {
        "shape": list(arr.shape),
        "colors": dict(sorted(counts.items())),
        "bbox_nonzero": bbox(arr, 0),
    }


def print_grid(grid: list[list[int]]) -> None:
    for row in grid:
        print("".join(str(v) for v in row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="+", type=int)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--max-examples", type=int, default=6)
    args = parser.parse_args()

    comp_dir = pathlib.Path(args.comp_dir)
    for task_id in args.tasks:
        path = comp_dir / f"task{task_id:03d}.json"
        task = json.loads(path.read_text())
        print(f"\n## task{task_id:03d}")
        for split in ["train", "test", "arc-gen"]:
            examples = task.get(split, [])
            print(f"{split}: {len(examples)}")
        all_examples = task["train"] + task["test"] + task.get("arc-gen", [])
        for index, example in enumerate(all_examples[: args.max_examples]):
            inp = example["input"]
            out = example["output"]
            print(f"\nexample {index}")
            print("input ", summarize_grid(inp))
            print("output", summarize_grid(out))
            print("in:")
            print_grid(inp)
            print("out:")
            print_grid(out)


if __name__ == "__main__":
    main()

