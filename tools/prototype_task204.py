from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples


def solve_task204(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    out = inp.copy()
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if inp[r, c] != 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            surrounded = True
            while stack:
                rr, cc = stack.pop()
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    surrounded = False
                    continue
                if inp[rr, cc] != 0:
                    if inp[rr, cc] != 1:
                        surrounded = False
                    continue
                if seen[rr, cc]:
                    continue
                seen[rr, cc] = True
                cells.append((rr, cc))
                stack.extend([(rr + 1, cc), (rr - 1, cc), (rr, cc + 1), (rr, cc - 1)])
            if not cells or not surrounded:
                continue
            rows = [x for x, _ in cells]
            cols = [y for _, y in cells]
            height = max(rows) - min(rows) + 1
            width = max(cols) - min(cols) + 1
            fill = 7 if (height % 2 == 1 or width % 2 == 1) else 2
            for rr, cc in cells:
                out[rr, cc] = fill
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    args = parser.parse_args()
    examples = all_examples(204, pathlib.Path(args.comp_dir))
    passed = 0
    for example in examples:
        inp = np.array(example["input"], dtype=np.int8)
        gt = np.array(example["output"], dtype=np.int8)
        if np.array_equal(solve_task204(inp), gt):
            passed += 1
    print(f"task204: {passed}/{len(examples)}")
    print("onnx_gate: skip (current cost ~309k; no low-cost graph built this session)")


if __name__ == "__main__":
    main()
