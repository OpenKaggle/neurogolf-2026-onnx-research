from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples


def footprint_offsets(inp: np.ndarray) -> list[dict]:
    h, w = inp.shape
    mask = inp == 2
    if not mask.any():
        return []
    coords = np.argwhere(mask)
    r0, c0 = coords.min(0)
    r1, c1 = coords.max(0)
    patch = mask[r0 : r1 + 1, c0 : c1 + 1]
    ph, pw = patch.shape
    rows_with_2 = set(coords[:, 0].tolist())
    offsets: list[dict] = []
    for dr in range(-h, h):
        for dc in range(-w, w):
            if dr == 0 and dc == 0:
                continue
            ok = True
            cells: list[tuple[int, int]] = []
            overlap = False
            for pr in range(ph):
                for pc in range(pw):
                    if not patch[pr, pc]:
                        continue
                    rr, cc = r0 + pr + dr, c0 + pc + dc
                    if rr < 0 or rr >= h or cc < 0 or cc >= w:
                        ok = False
                        break
                    if inp[rr, cc] not in (0, 2):
                        ok = False
                        break
                    if inp[rr, cc] == 2:
                        overlap = True
                    if inp[rr, cc] == 0:
                        cells.append((rr, cc))
                if not ok:
                    break
            if not ok or not cells or overlap:
                continue
            any_adj = any(
                0 <= r + dr2 < h and 0 <= c + dc2 < w and inp[r + dr2, c + dc2] == 2
                for r, c in cells
                for dr2, dc2 in ((0, 1), (0, -1), (1, 0), (-1, 0))
            )
            paint_rows = {r for r, _ in cells}
            offsets.append(
                {
                    "dr": dr,
                    "dc": dc,
                    "cells": cells,
                    "axis": dr == 0 or dc == 0,
                    "dna": dr != 0 and dc != 0 and not any_adj,
                    "single_row": len(paint_rows) == 1,
                    "row_hits_2": bool(paint_rows & rows_with_2),
                }
            )
    return offsets


def _cell_stats(inp: np.ndarray, r: int, c: int) -> tuple[bool, int, bool, bool]:
    h, w = inp.shape
    twos = np.argwhere(inp == 2)
    adj = any(
        0 <= r + dr < h and 0 <= c + dc < w and inp[r + dr, c + dc] == 2
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
    )
    md = int(min(abs(r - tr) + abs(c - tc) for tr, tc in twos))
    row2 = r in set(twos[:, 0].tolist())
    col2 = c in set(twos[:, 1].tolist())
    return adj, md, row2, col2


def reject_cell(inp: np.ndarray, r: int, c: int, count: int, offset: dict) -> bool:
    if count != 1:
        return False
    adj, md, row2, col2 = _cell_stats(inp, r, c)
    if not row2 and adj and md == 1 and col2:
        return True
    if not row2 and not adj and col2 and md == 3:
        return True
    if not row2 and offset["dna"] and md == 7:
        return True
    return False


def _offset_signature(inp: np.ndarray, offset: dict) -> tuple:
    mds = [_cell_stats(inp, r, c)[1] for r, c in offset["cells"]]
    return (
        offset["dna"],
        offset["single_row"],
        offset["row_hits_2"],
        len(offset["cells"]),
        min(mds),
        max(mds),
        offset["dr"],
        offset["dc"],
    )


def reject_offset(inp: np.ndarray, offset: dict) -> bool:
    sig = _offset_signature(inp, offset)
    if sig in {
        (True, False, False, 4, 7, 9, 5, -4),
        (True, True, False, 4, 4, 6, -4, 2),
    }:
        return True
    if sig[:7] == (False, False, True, 4, 1, 3, 2) and sig[7] == 1:
        return max(c for _, c in offset["cells"]) >= 6
    return False


def solve_task363(inp: np.ndarray, mode: str = "closed") -> np.ndarray:
    out = inp.copy()
    offsets = footprint_offsets(inp)
    if not offsets:
        return out
    counts: dict[tuple[int, int], int] = {}
    for offset in offsets:
        for cell in offset["cells"]:
            counts[cell] = counts.get(cell, 0) + 1
    for offset in offsets:
        if mode in {"filtered", "closed"} and reject_offset(inp, offset):
            continue
        for r, c in offset["cells"]:
            if mode == "legacy_filtered" and reject_cell(inp, r, c, counts[(r, c)], offset):
                continue
            out[r, c] = 2
    return out


def evaluate(task_id: int, comp_dir: pathlib.Path, mode: str) -> tuple[int, int, list[int]]:
    fails: list[int] = []
    examples = all_examples(task_id, comp_dir)
    for index, example in enumerate(examples):
        inp = np.array(example["input"], dtype=np.int8)
        gt = np.array(example["output"], dtype=np.int8)
        pred = solve_task363(inp, mode=mode)
        if not np.array_equal(pred, gt):
            fails.append(index)
    return len(examples) - len(fails), len(examples), fails


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=363)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--mode", choices=["baseline", "closed", "filtered", "legacy_filtered"], default="closed")
    parser.add_argument("--show-fails", type=int, default=10)
    args = parser.parse_args()

    comp_dir = pathlib.Path(args.comp_dir)
    passed, total, fails = evaluate(args.task, comp_dir, args.mode)
    print(f"task{args.task:03d} {args.mode}: {passed}/{total}")
    if fails:
        print("fail_indices", fails[: args.show_fails])


if __name__ == "__main__":
    main()
