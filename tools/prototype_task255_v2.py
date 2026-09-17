#!/usr/bin/env python3
"""Task255 semantic reference prototype and validation harness.

The solver is intentionally train-grounded and ID-free:
1. foreground is any non-zero cell,
2. candidate fill cells are zero cells with no 8-neighbor foreground,
3. stable horizontal/vertical corridor rectangles are recovered from long
   overlapping runs in that candidate mask,
4. selected zero cells are colored 3.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "neurogolf-2026" / "raw" / "task255.json"
REPORT_DIR = ROOT / "reports"


@dataclass(frozen=True)
class Rect:
    r0: int
    r1: int
    c0: int
    c1: int
    axis: str

    @property
    def area(self) -> int:
        return (self.r1 - self.r0 + 1) * (self.c1 - self.c0 + 1)


@dataclass
class ExampleMetric:
    solver: str
    split: str
    index: int
    exact: int
    pixel_errors: int
    false_positive_3: int
    false_negative_3: int
    target_3: int
    predicted_3: int
    rectangles: int


def load_task(path: Path = DATA_PATH) -> dict:
    return json.loads(path.read_text())


def as_array(grid: Sequence[Sequence[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def legacy_stub(grid: np.ndarray) -> np.ndarray:
    """Equivalent outcome of tools/prototype_task255.py for this task.

    The old code seeds binary propagation with all foreground cells under an
    all-true mask, so every board containing foreground propagates to all cells
    and no holes remain. All task255 examples contain foreground.
    """

    return grid.copy()


def candidate_core_mask(grid: np.ndarray) -> np.ndarray:
    """Zero cells whose 3x3 neighborhood contains no foreground."""

    h, w = grid.shape
    foreground = grid != 0
    core = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if foreground[r, c]:
                continue
            r0 = max(0, r - 1)
            r1 = min(h, r + 2)
            c0 = max(0, c - 1)
            c1 = min(w, c + 2)
            core[r, c] = not foreground[r0:r1, c0:c1].any()
    return core


def bool_runs(values: Sequence[bool], min_len: int = 1) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(values)
    while i < n:
        if not values[i]:
            i += 1
            continue
        j = i + 1
        while j < n and values[j]:
            j += 1
        if j - i >= min_len:
            runs.append((i, j - 1))
        i = j
    return runs


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def intersect(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return max(a[0], b[0]), min(a[1], b[1])


def stable_corridor_rectangles(
    core: np.ndarray, axis: str, min_major_run: int
) -> list[Rect]:
    """Group long overlapping core runs and keep their stable intersection."""

    h, w = core.shape
    count = h if axis == "h" else w
    per_line: list[list[tuple[int, int]]] = []
    for i in range(count):
        values = core[i, :] if axis == "h" else core[:, i]
        per_line.append(bool_runs(values.tolist(), min_major_run))

    active: list[dict] = []
    finished: list[dict] = []

    for line_index, runs in enumerate(per_line):
        used = [False] * len(runs)
        next_active: list[dict] = []
        for group in active:
            matches = [
                (k, run)
                for k, run in enumerate(runs)
                if not used[k] and overlaps(group["intersection"], run)
            ]
            if not matches:
                finished.append(group)
                continue
            k, run = max(
                matches,
                key=lambda item: min(group["intersection"][1], item[1][1])
                - max(group["intersection"][0], item[1][0])
                + 1,
            )
            used[k] = True
            group["end"] = line_index
            group["intersection"] = intersect(group["intersection"], run)
            next_active.append(group)

        for k, run in enumerate(runs):
            if not used[k]:
                next_active.append(
                    {"start": line_index, "end": line_index, "intersection": run}
                )
        active = next_active

    finished.extend(active)

    rects: list[Rect] = []
    for group in finished:
        a = int(group["start"])
        b = int(group["end"])
        lo, hi = group["intersection"]
        if lo > hi:
            continue
        if axis == "h":
            rects.append(Rect(a, b, lo, hi, "h"))
        else:
            rects.append(Rect(lo, hi, a, b, "v"))
    return rects


def intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def trim_one_cell_tails(rects: Sequence[Rect]) -> list[Rect]:
    """Trim weak one-cell tails where a thin rectangle crosses a broad one.

    Train examples show that stable corridor intersections define the real
    endpoint when a one-cell-wide branch has a one-cell core tail beyond a
    cross-corridor. This is a geometry rule, not a test/example ID patch.
    """

    horizontal = [rect for rect in rects if rect.axis == "h"]
    vertical = [rect for rect in rects if rect.axis == "v"]
    trimmed: list[Rect] = []

    for rect in rects:
        out = rect
        if rect.axis == "v" and rect.c0 == rect.c1:
            crossings = [
                h
                for h in horizontal
                if intervals_overlap(h.c0, h.c1, rect.c0, rect.c1)
                and intervals_overlap(h.r0, h.r1, rect.r0, rect.r1)
            ]
            if crossings:
                top = min(h.r0 for h in crossings)
                bottom = max(h.r1 for h in crossings)
                if 0 < top - out.r0 <= 1:
                    out = Rect(top, out.r1, out.c0, out.c1, out.axis)
                if 0 < out.r1 - bottom <= 1:
                    out = Rect(out.r0, bottom, out.c0, out.c1, out.axis)
        elif rect.axis == "h" and rect.r0 == rect.r1:
            crossings = [
                v
                for v in vertical
                if intervals_overlap(v.r0, v.r1, rect.r0, rect.r1)
                and intervals_overlap(v.c0, v.c1, rect.c0, rect.c1)
            ]
            if crossings:
                left = min(v.c0 for v in crossings)
                right = max(v.c1 for v in crossings)
                if 0 < left - out.c0 <= 1:
                    out = Rect(out.r0, out.r1, left, out.c1, out.axis)
                if 0 < out.c1 - right <= 1:
                    out = Rect(out.r0, out.r1, out.c0, right, out.axis)

        if out.r0 <= out.r1 and out.c0 <= out.c1:
            trimmed.append(out)
    return trimmed


def solve_with_meta(
    grid: np.ndarray, min_major_run: int | None = None
) -> tuple[np.ndarray, list[Rect], np.ndarray]:
    if min_major_run is None:
        min_major_run = max(1, min(grid.shape) // 2 - 1)

    core = candidate_core_mask(grid)
    selected = np.zeros_like(core)
    rects = stable_corridor_rectangles(core, "h", min_major_run)
    rects.extend(stable_corridor_rectangles(core, "v", min_major_run))
    rects = trim_one_cell_tails(rects)

    for rect in rects:
        selected[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] |= core[
            rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1
        ]

    out = grid.copy()
    out[selected] = 3
    return out, rects, core


def solver_task255(grid: np.ndarray) -> np.ndarray:
    out, _, _ = solve_with_meta(grid)
    return out


def core_upper_bound(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[candidate_core_mask(grid)] = 3
    return out


def metric_for(
    solver_name: str,
    split: str,
    index: int,
    pred: np.ndarray,
    target: np.ndarray,
    rectangles: int = 0,
) -> ExampleMetric:
    pred_3 = pred == 3
    target_3 = target == 3
    return ExampleMetric(
        solver=solver_name,
        split=split,
        index=index,
        exact=int(np.array_equal(pred, target)),
        pixel_errors=int(np.count_nonzero(pred != target)),
        false_positive_3=int(np.count_nonzero(pred_3 & ~target_3)),
        false_negative_3=int(np.count_nonzero(~pred_3 & target_3)),
        target_3=int(np.count_nonzero(target_3)),
        predicted_3=int(np.count_nonzero(pred_3)),
        rectangles=rectangles,
    )


def split_metrics(task: dict) -> tuple[list[ExampleMetric], list[dict]]:
    metrics: list[ExampleMetric] = []
    rect_rows: list[dict] = []

    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(task[split]):
            grid = as_array(example["input"])
            target = as_array(example["output"])

            legacy = legacy_stub(grid)
            metrics.append(metric_for("legacy_stub", split, index, legacy, target))

            core = core_upper_bound(grid)
            metrics.append(metric_for("core8_upper_bound", split, index, core, target))

            pred, rects, core_mask = solve_with_meta(grid)
            metrics.append(
                metric_for(
                    "v2_train_only_rect_corridor",
                    split,
                    index,
                    pred,
                    target,
                    rectangles=len(rects),
                )
            )
            for rid, rect in enumerate(rects):
                row = {
                    "split": split,
                    "index": index,
                    "rect_id": rid,
                    **asdict(rect),
                    "area": rect.area,
                    "core_cells_in_rect": int(
                        np.count_nonzero(
                            core_mask[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1]
                        )
                    ),
                }
                rect_rows.append(row)

    return metrics, rect_rows


def summarize_metrics(metrics: Iterable[ExampleMetric]) -> list[dict]:
    buckets: dict[tuple[str, str], list[ExampleMetric]] = {}
    for metric in metrics:
        buckets.setdefault((metric.solver, metric.split), []).append(metric)

    rows: list[dict] = []
    for (solver, split), group in sorted(buckets.items()):
        total = len(group)
        exact = sum(item.exact for item in group)
        pixel_errors = sum(item.pixel_errors for item in group)
        false_positive_3 = sum(item.false_positive_3 for item in group)
        false_negative_3 = sum(item.false_negative_3 for item in group)
        target_3 = sum(item.target_3 for item in group)
        predicted_3 = sum(item.predicted_3 for item in group)
        rows.append(
            {
                "solver": solver,
                "split": split,
                "exact": exact,
                "total": total,
                "exact_rate": exact / total if total else 0.0,
                "pixel_errors": pixel_errors,
                "false_positive_3": false_positive_3,
                "false_negative_3": false_negative_3,
                "target_3": target_3,
                "predicted_3": predicted_3,
            }
        )
    return rows


def possible_exact_min_runs(task: dict, support_indices: Sequence[int]) -> list[int]:
    exact_runs: list[int] = []
    for min_run in range(2, 31):
        ok = True
        for index in support_indices:
            example = task["train"][index]
            pred, _, _ = solve_with_meta(as_array(example["input"]), min_run)
            if not np.array_equal(pred, as_array(example["output"])):
                ok = False
                break
        if ok:
            exact_runs.append(min_run)
    return exact_runs


def color_variants(example: dict) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    grid = as_array(example["input"])
    target = as_array(example["output"])
    foreground_colors = sorted(int(v) for v in set(grid.flatten()) if v != 0)
    if len(foreground_colors) != 1:
        return
    old = foreground_colors[0]
    for new in (1, 2, 4, 5, 6, 7, 8, 9):
        if new == old:
            continue
        g = grid.copy()
        y = target.copy()
        g[g == old] = new
        y[y == old] = new
        yield f"fg_{old}_to_{new}", g, y


def translated_pair(
    grid: np.ndarray, target: np.ndarray, dr: int, dc: int
) -> tuple[np.ndarray, np.ndarray] | None:
    moving = (grid != 0) | (target == 3)
    coords = np.argwhere(moving)
    if coords.size == 0:
        return None
    shifted = coords + np.array([dr, dc])
    h, w = grid.shape
    if (
        shifted[:, 0].min() < 0
        or shifted[:, 0].max() >= h
        or shifted[:, 1].min() < 0
        or shifted[:, 1].max() >= w
    ):
        return None

    g = np.zeros_like(grid)
    y = np.zeros_like(target)
    for r in range(h):
        for c in range(w):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                g[nr, nc] = grid[r, c]
                y[nr, nc] = target[r, c]
    return g, y


def clipped_translation_pair(
    grid: np.ndarray, target: np.ndarray, dr: int, dc: int
) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    g = np.zeros_like(grid)
    y = np.zeros_like(target)
    for r in range(h):
        for c in range(w):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                g[nr, nc] = grid[r, c]
                y[nr, nc] = target[r, c]
    return g, y


def far_noise_pair(
    grid: np.ndarray, target: np.ndarray, noise_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = grid.shape
    protected = (grid != 0) | (target == 3)
    near_protected = protected.copy()
    for r in range(h):
        for c in range(w):
            if not protected[r, c]:
                continue
            r0 = max(0, r - 1)
            r1 = min(h, r + 2)
            c0 = max(0, c - 1)
            c1 = min(w, c + 2)
            near_protected[r0:r1, c0:c1] = True
    candidates = np.argwhere(~near_protected)
    if len(candidates) < noise_count:
        return None

    rng = np.random.default_rng(seed)
    chosen = candidates[rng.choice(len(candidates), size=noise_count, replace=False)]
    foreground_colors = sorted(int(v) for v in set(grid.flatten()) if v != 0)
    color = foreground_colors[0] if foreground_colors else 1
    g = grid.copy()
    y = target.copy()
    for r, c in chosen:
        g[int(r), int(c)] = color
        y[int(r), int(c)] = color
    return g, y


def pseudo_hidden_checks(task: dict) -> list[dict]:
    rows: list[dict] = []

    # Leave-one-out is deliberately fixed-geometry: no arc-gen IDs, no per-case fit.
    all_train = list(range(len(task["train"])))
    selected_min_run = 30 // 2 - 1
    for holdout in all_train:
        support = [i for i in all_train if i != holdout]
        support_exact_runs = possible_exact_min_runs(task, support)
        example = task["train"][holdout]
        pred, rects, _ = solve_with_meta(as_array(example["input"]), selected_min_run)
        target = as_array(example["output"])
        rows.append(
            {
                "check": "leave_one_out_fixed_geometry",
                "split": "train",
                "index": holdout,
                "variant": f"support={support}",
                "eligible": 1,
                "exact": int(np.array_equal(pred, target)),
                "pixel_errors": int(np.count_nonzero(pred != target)),
                "false_positive_3": int(np.count_nonzero((pred == 3) & (target != 3))),
                "false_negative_3": int(np.count_nonzero((pred != 3) & (target == 3))),
                "details": json.dumps(
                    {
                        "selected_min_run": selected_min_run,
                        "support_exact_min_runs": support_exact_runs,
                        "rectangles": [asdict(rect) for rect in rects],
                    },
                    sort_keys=True,
                ),
            }
        )

    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(task[split]):
            for name, grid, target in color_variants(example):
                pred = solver_task255(grid)
                rows.append(
                    {
                        "check": "foreground_color_permutation",
                        "split": split,
                        "index": index,
                        "variant": name,
                        "eligible": 1,
                        "exact": int(np.array_equal(pred, target)),
                        "pixel_errors": int(np.count_nonzero(pred != target)),
                        "false_positive_3": int(np.count_nonzero((pred == 3) & (target != 3))),
                        "false_negative_3": int(np.count_nonzero((pred != 3) & (target == 3))),
                        "details": "",
                    }
                )

    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(task[split]):
            grid = as_array(example["input"])
            target = as_array(example["output"])
            for dr, dc in shifts:
                pair = translated_pair(grid, target, dr, dc)
                if pair is None:
                    rows.append(
                        {
                            "check": "safe_zero_padded_translation",
                            "split": split,
                            "index": index,
                            "variant": f"dr={dr},dc={dc}",
                            "eligible": 0,
                            "exact": 0,
                            "pixel_errors": 0,
                            "false_positive_3": 0,
                            "false_negative_3": 0,
                            "details": "translation would clip foreground or target-3 cells",
                        }
                    )
                    continue
                g, y = pair
                pred = solver_task255(g)
                rows.append(
                    {
                        "check": "safe_zero_padded_translation",
                        "split": split,
                        "index": index,
                        "variant": f"dr={dr},dc={dc}",
                        "eligible": 1,
                        "exact": int(np.array_equal(pred, y)),
                        "pixel_errors": int(np.count_nonzero(pred != y)),
                        "false_positive_3": int(np.count_nonzero((pred == 3) & (y != 3))),
                        "false_negative_3": int(np.count_nonzero((pred != 3) & (y == 3))),
                        "details": "",
                    }
                )

    clipped_shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(task[split]):
            grid = as_array(example["input"])
            target = as_array(example["output"])
            for dr, dc in clipped_shifts:
                g, y = clipped_translation_pair(grid, target, dr, dc)
                pred = solver_task255(g)
                rows.append(
                    {
                        "check": "clipped_zero_padded_translation",
                        "split": split,
                        "index": index,
                        "variant": f"dr={dr},dc={dc}",
                        "eligible": 1,
                        "exact": int(np.array_equal(pred, y)),
                        "pixel_errors": int(np.count_nonzero(pred != y)),
                        "false_positive_3": int(np.count_nonzero((pred == 3) & (y != 3))),
                        "false_negative_3": int(np.count_nonzero((pred != 3) & (y == 3))),
                        "details": "input and expected output shifted together, cells outside 30x30 clipped",
                    }
                )

    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(task[split]):
            grid = as_array(example["input"])
            target = as_array(example["output"])
            for noise_count in (1, 2, 3):
                pair = far_noise_pair(
                    grid, target, noise_count, seed=255_000 + index * 17 + noise_count
                )
                if pair is None:
                    rows.append(
                        {
                            "check": "far_additive_foreground_noise",
                            "split": split,
                            "index": index,
                            "variant": f"n={noise_count}",
                            "eligible": 0,
                            "exact": 0,
                            "pixel_errors": 0,
                            "false_positive_3": 0,
                            "false_negative_3": 0,
                            "details": "not enough far-background cells",
                        }
                    )
                    continue
                g, y = pair
                pred = solver_task255(g)
                rows.append(
                    {
                        "check": "far_additive_foreground_noise",
                        "split": split,
                        "index": index,
                        "variant": f"n={noise_count}",
                        "eligible": 1,
                        "exact": int(np.array_equal(pred, y)),
                        "pixel_errors": int(np.count_nonzero(pred != y)),
                        "false_positive_3": int(np.count_nonzero((pred == 3) & (y != 3))),
                        "false_negative_3": int(np.count_nonzero((pred != 3) & (y == 3))),
                        "details": "",
                    }
                )

    return rows


def summarize_pseudo(rows: Iterable[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        buckets.setdefault((row["check"], row["split"]), []).append(row)

    summary: list[dict] = []
    for (check, split), group in sorted(buckets.items()):
        eligible = [row for row in group if row["eligible"]]
        exact = sum(int(row["exact"]) for row in eligible)
        summary.append(
            {
                "check": check,
                "split": split,
                "eligible": len(eligible),
                "total_rows": len(group),
                "exact": exact,
                "exact_rate": exact / len(eligible) if eligible else 0.0,
                "pixel_errors": sum(int(row["pixel_errors"]) for row in eligible),
                "false_positive_3": sum(int(row["false_positive_3"]) for row in eligible),
                "false_negative_3": sum(int(row["false_negative_3"]) for row in eligible),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(task: dict) -> None:
    metrics, rect_rows = split_metrics(task)
    split_summary = summarize_metrics(metrics)
    metric_rows = [asdict(row) for row in metrics]
    pseudo_rows = pseudo_hidden_checks(task)
    pseudo_summary = summarize_pseudo(pseudo_rows)

    REPORT_DIR.mkdir(exist_ok=True)
    write_csv(REPORT_DIR / "task255_v2_split_summary.csv", split_summary)
    write_csv(REPORT_DIR / "task255_v2_split_examples.csv", metric_rows)
    write_csv(REPORT_DIR / "task255_v2_rectangles.csv", rect_rows)
    write_csv(REPORT_DIR / "task255_v2_pseudo_hidden.csv", pseudo_rows)
    write_csv(REPORT_DIR / "task255_v2_pseudo_hidden_summary.csv", pseudo_summary)
    (REPORT_DIR / "task255_v2_summary.json").write_text(
        json.dumps(
            {
                "split_summary": split_summary,
                "pseudo_hidden_summary": pseudo_summary,
                "hypothesis": {
                    "name": "v2_train_only_rect_corridor",
                    "min_major_run": 14,
                    "core": "zero cell with no 8-neighbor foreground",
                    "rectangles": "long overlapping core runs grouped by axis; fill stable intersections",
                    "arc_gen_id_special_cases": 0,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (REPORT_DIR / "task255_v2_pseudo_hidden.json").write_text(
        json.dumps(
            {"rows": pseudo_rows, "summary": pseudo_summary},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--write-reports", action="store_true")
    args = parser.parse_args()

    task = load_task(args.data)
    metrics, _ = split_metrics(task)
    for row in summarize_metrics(metrics):
        print(
            f"{row['solver']:28s} {row['split']:7s} "
            f"{row['exact']:3d}/{row['total']:<3d} "
            f"pixerr={row['pixel_errors']:<5d} "
            f"fp3={row['false_positive_3']:<5d} fn3={row['false_negative_3']:<5d}"
        )
    if args.write_reports:
        write_reports(task)


if __name__ == "__main__":
    main()
