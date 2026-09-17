#!/usr/bin/env python3
"""Task366 semantic rebuild prototype.

The visible/arc-gen behavior is a two-panel object completion task:
split the grid along its longer axis, use the panel with more non-background
cells as the source, and copy source rectangles onto the sparse-anchor panel
when the anchor cells match a source color footprint.

This file is intentionally a falsification harness first. It compares the
existing label-propagation semantics with a bounded filled-rectangle rule,
records weaker hypotheses that fail, and runs explicit pseudo-hidden contracts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_PATH = ROOT / "data" / "neurogolf-2026" / "raw" / "task366.json"
REPORT_DIR = ROOT / "reports"
ANCHOR_ONNX = ROOT / "submissions" / "candidate_v4_plus7_task149_onnx" / "task366.onnx"


Grid = np.ndarray
Solver = Callable[[Grid], Grid]


@dataclass(frozen=True)
class PanelInfo:
    a: Grid
    b: Grid
    orientation: str
    half: int


@dataclass(frozen=True)
class Pattern:
    patch: Grid
    mask: Grid
    colors_used: frozenset[int]
    r0: int
    c0: int
    h: int
    w: int


@dataclass(frozen=True)
class EvalRow:
    solver: str
    split: str
    index: int
    exact: int
    same_as_label: int
    pred_shape: str
    target_shape: str
    pixel_errors: int | None


def load_task(path: Path = DATA_PATH) -> dict:
    return json.loads(path.read_text())


def all_examples(task: dict) -> list[tuple[str, int, dict]]:
    rows: list[tuple[str, int, dict]] = []
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(task.get(split, [])):
            rows.append((split, index, example))
    return rows


def arr(grid: Sequence[Sequence[int]] | Grid) -> Grid:
    return np.asarray(grid, dtype=np.int64)


def shape_s(grid: Grid) -> str:
    return f"{grid.shape[0]}x{grid.shape[1]}"


def background(grid: Grid) -> int:
    return int(np.bincount(grid.ravel(), minlength=10).argmax())


def split_panels(grid: Grid) -> PanelInfo:
    h, w = grid.shape
    if h >= w:
        half = h // 2
        return PanelInfo(grid[:half].copy(), grid[half : 2 * half].copy(), "vertical", half)
    half = w // 2
    return PanelInfo(grid[:, :half].copy(), grid[:, half : 2 * half].copy(), "horizontal", half)


def join_panels(a: Grid, b: Grid, orientation: str) -> Grid:
    if orientation == "vertical":
        return np.concatenate([a, b], axis=0)
    return np.concatenate([a, b], axis=1)


def source_dest(grid: Grid, *, force_source: str | None = None) -> tuple[Grid, int, Grid, int, str]:
    panels = split_panels(grid)
    bg_a = background(panels.a)
    bg_b = background(panels.b)
    nb_a = int((panels.a != bg_a).sum())
    nb_b = int((panels.b != bg_b).sum())

    if force_source == "a":
        src_is_a = True
    elif force_source == "b":
        src_is_a = False
    else:
        src_is_a = nb_a > nb_b

    if src_is_a:
        return panels.a, bg_a, panels.b, bg_b, "a"
    return panels.b, bg_b, panels.a, bg_a, "b"


def label_4conn(mask: Grid) -> tuple[Grid, int]:
    h, w = mask.shape
    labeled = np.zeros((h, w), dtype=np.int64)
    current = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labeled[r, c] != 0:
                continue
            current += 1
            stack = [(r, c)]
            labeled[r, c] = current
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and labeled[nr, nc] == 0
                    ):
                        labeled[nr, nc] = current
                        stack.append((nr, nc))
    return labeled, current


def label_patterns(src: Grid, src_bg: int) -> list[Pattern]:
    src_mask = src != src_bg
    labeled, n_src = label_4conn(src_mask)
    patterns: list[Pattern] = []
    for feat_id in range(1, n_src + 1):
        mask = labeled == feat_id
        coords = np.argwhere(mask)
        if len(coords) <= 1:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = src[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        colors = frozenset(int(src[r, c]) for r, c in coords)
        patterns.append(
            Pattern(
                patch,
                patch_mask.astype(bool, copy=True),
                colors,
                int(r0),
                int(c0),
                patch.shape[0],
                patch.shape[1],
            )
        )
    return patterns


def bounded_rectangle_patterns(src: Grid, src_bg: int, max_dim: int = 7) -> list[Pattern]:
    """Recover filled rectangular source objects without component propagation."""

    mask = src != src_bg
    h, w = mask.shape
    patterns: list[Pattern] = []
    used = np.zeros_like(mask, dtype=bool)

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or used[r, c]:
                continue
            if r > 0 and mask[r - 1, c]:
                continue
            if c > 0 and mask[r, c - 1]:
                continue

            best: tuple[int, int] | None = None
            for rh in range(1, max_dim + 1):
                if r + rh > h:
                    break
                for rw in range(1, max_dim + 1):
                    if rw * rh <= 1 or c + rw > w:
                        continue
                    body = mask[r : r + rh, c : c + rw]
                    if not body.all() or used[r : r + rh, c : c + rw].any():
                        continue
                    if r > 0 and mask[r - 1, c : c + rw].any():
                        continue
                    if c > 0 and mask[r : r + rh, c - 1].any():
                        continue
                    if r + rh < h and mask[r + rh, c : c + rw].any():
                        continue
                    if c + rw < w and mask[r : r + rh, c + rw].any():
                        continue
                    if best is None or rw * rh > best[0] * best[1]:
                        best = (rh, rw)

            if best is None:
                continue
            rh, rw = best
            patch = src[r : r + rh, c : c + rw].copy()
            colors = frozenset(int(x) for x in np.unique(patch))
            patterns.append(Pattern(patch, np.ones((rh, rw), dtype=bool), colors, r, c, rh, rw))
            used[r : r + rh, c : c + rw] = True

    return patterns


def place_patterns(
    dst: Grid,
    dst_bg: int,
    patterns: list[Pattern],
    *,
    use_anchor_locking: bool = True,
    sort_patterns: bool = True,
) -> Grid:
    out = dst.copy()
    anchors: dict[int, list[tuple[int, int]]] = {}
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            if int(dst[r, c]) == dst_bg:
                continue
            anchors.setdefault(int(dst[r, c]), []).append((r, c))

    work = list(patterns)
    if sort_patterns:
        anchor_colors = set(anchors)
        work.sort(
            key=lambda p: -sum(
                int(((p.patch == c) & p.mask).sum()) for c in p.colors_used & anchor_colors
            )
        )

    used_anchors: set[tuple[int, int]] = set()
    for pat in work:
        key_colors = sorted(pat.colors_used & set(anchors))
        if not key_colors:
            continue
        found = False
        for kc in key_colors:
            p_cells = list(zip(*np.where((pat.patch == kc) & pat.mask)))
            avail = [a for a in anchors.get(kc, []) if not use_anchor_locking or a not in used_anchors]
            if len(avail) < len(p_cells):
                continue
            anchor_set = set(avail)
            for anchor in avail:
                dr = anchor[0] - int(p_cells[0][0])
                dc = anchor[1] - int(p_cells[0][1])
                matched: list[tuple[int, int]] = []
                ok = True
                for pr, pc in p_cells:
                    ar, ac = int(pr) + dr, int(pc) + dc
                    if (ar, ac) in anchor_set and (
                        not use_anchor_locking or (ar, ac) not in used_anchors
                    ):
                        matched.append((ar, ac))
                    else:
                        ok = False
                        break
                if not ok:
                    continue

                for r in range(pat.h):
                    for c in range(pat.w):
                        if not pat.mask[r, c]:
                            continue
                        rr, cc = dr + r, dc + c
                        if 0 <= rr < dst.shape[0] and 0 <= cc < dst.shape[1]:
                            out[rr, cc] = int(pat.patch[r, c])
                if use_anchor_locking:
                    used_anchors.update(matched)
                found = True
                break
            if found:
                break
    return out


def solve_with_patterns(
    grid: Grid,
    pattern_fn: Callable[[Grid, int], list[Pattern]],
    *,
    force_source: str | None = None,
    use_anchor_locking: bool = True,
    sort_patterns: bool = True,
) -> Grid:
    src, src_bg, dst, dst_bg, _ = source_dest(grid, force_source=force_source)
    return place_patterns(
        dst,
        dst_bg,
        pattern_fn(src, src_bg),
        use_anchor_locking=use_anchor_locking,
        sort_patterns=sort_patterns,
    )


def solver_label(grid: Grid) -> Grid:
    return solve_with_patterns(grid, label_patterns)


def solver_bounded_rectangles(grid: Grid, max_dim: int = 7) -> Grid:
    return solve_with_patterns(
        grid,
        lambda src, src_bg: bounded_rectangle_patterns(src, src_bg, max_dim=max_dim),
    )


def solver_first_panel_source(grid: Grid) -> Grid:
    return solve_with_patterns(grid, lambda s, b: bounded_rectangle_patterns(s, b, 7), force_source="a")


def solver_no_pattern_priority(grid: Grid) -> Grid:
    return solve_with_patterns(
        grid,
        lambda s, b: bounded_rectangle_patterns(s, b, 7),
        sort_patterns=False,
    )


def solver_no_anchor_locking(grid: Grid) -> Grid:
    return solve_with_patterns(
        grid,
        lambda s, b: bounded_rectangle_patterns(s, b, 7),
        use_anchor_locking=False,
    )


def evaluate_solver(name: str, solver: Solver, task: dict, label_outputs: dict[tuple[str, int], Grid]) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for split, index, example in all_examples(task):
        target = arr(example["output"])
        pred = arr(solver(arr(example["input"])))
        same_shape = pred.shape == target.shape
        exact = bool(same_shape and np.array_equal(pred, target))
        label = label_outputs[(split, index)]
        same_label = bool(pred.shape == label.shape and np.array_equal(pred, label))
        rows.append(
            EvalRow(
                solver=name,
                split=split,
                index=index,
                exact=int(exact),
                same_as_label=int(same_label),
                pred_shape=shape_s(pred),
                target_shape=shape_s(target),
                pixel_errors=int(np.sum(pred != target)) if same_shape else None,
            )
        )
    return rows


def summarize_eval(rows: Iterable[EvalRow]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str], list[EvalRow]] = {}
    for row in rows:
        by_key.setdefault((row.solver, row.split), []).append(row)

    out: list[dict[str, object]] = []
    for (solver, split), group in sorted(by_key.items()):
        out.append(
            {
                "solver": solver,
                "split": split,
                "exact": sum(r.exact for r in group),
                "total": len(group),
                "same_as_label": sum(r.same_as_label for r in group),
                "pixel_errors": sum(r.pixel_errors or 0 for r in group),
            }
        )
    return out


def panel_shape_stats(task: dict) -> dict[str, object]:
    pattern_dims: list[tuple[int, int]] = []
    pattern_count = 0
    full_rect_violations = 0
    encodable = {"train": 0, "test": 0, "arc-gen": 0}
    totals = {"train": 0, "test": 0, "arc-gen": 0}
    source_side: dict[str, int] = {}
    for split, _, example in all_examples(task):
        g = arr(example["input"])
        out = arr(example["output"])
        totals[split] += 1
        if g.shape[0] <= 30 and g.shape[1] <= 30 and out.shape[0] <= 30 and out.shape[1] <= 30:
            encodable[split] += 1
        src, src_bg, _, _, side = source_dest(g)
        panels = split_panels(g)
        source_side[f"{panels.orientation}_{side}"] = source_side.get(f"{panels.orientation}_{side}", 0) + 1
        for pat in label_patterns(src, src_bg):
            pattern_count += 1
            pattern_dims.append((pat.h, pat.w))
            mask = pat.patch != 0
            if not mask.all():
                full_rect_violations += 1

    dims = np.asarray(pattern_dims, dtype=np.int64)
    return {
        "examples": sum(totals.values()),
        "totals": totals,
        "encodable": encodable,
        "patterns": pattern_count,
        "max_pattern_h": int(dims[:, 0].max()),
        "max_pattern_w": int(dims[:, 1].max()),
        "full_rect_violations": full_rect_violations,
        "source_side": source_side,
    }


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def task366_contracts():
    from tools import pseudo_hidden as ph

    max_shape = (40, 40)
    contracts = [
        ph.color_permutation_contract(
            {0: 9, 1: 4, 2: 8, 3: 6, 4: 1, 5: 7, 6: 3, 7: 5, 8: 2, 9: 0},
            name="task366_color_bijection",
            max_shape=max_shape,
        )
    ]

    def swap_input(grid: Grid) -> Grid:
        p = split_panels(arr(grid))
        return join_panels(p.b, p.a, p.orientation)

    contracts.append(
        ph.InvarianceContract(
            name="task366_swap_panels",
            input_transform=swap_input,
            expected_output_transform=lambda grid: arr(grid),
            max_input_shape=max_shape,
            max_output_shape=max_shape,
            description="Swapping source/destination panels should not change the aligned output.",
        )
    )

    contracts.append(
        ph.InvarianceContract(
            name="task366_transpose",
            input_transform=lambda grid: arr(grid).T,
            expected_output_transform=lambda grid: arr(grid).T,
            max_input_shape=max_shape,
            max_output_shape=max_shape,
            description="Transpose swaps vertical and horizontal panel layouts.",
        )
    )

    def mirror_panel_grid(grid: Grid) -> Grid:
        p = split_panels(arr(grid))
        return join_panels(np.fliplr(p.a), np.fliplr(p.b), p.orientation)

    contracts.append(
        ph.InvarianceContract(
            name="task366_mirror_each_panel_lr",
            input_transform=mirror_panel_grid,
            expected_output_transform=lambda grid: np.fliplr(arr(grid)),
            max_input_shape=max_shape,
            max_output_shape=max_shape,
            description="Mirror source and anchor panels in their local coordinates.",
        )
    )

    def can_shift_grid(grid: Grid, dr: int, dc: int) -> bool:
        bg = background(grid)
        coords = np.argwhere(grid != bg)
        if len(coords) == 0:
            return True
        shifted = coords + np.array([dr, dc])
        return bool(
            (shifted[:, 0] >= 0).all()
            and (shifted[:, 1] >= 0).all()
            and (shifted[:, 0] < grid.shape[0]).all()
            and (shifted[:, 1] < grid.shape[1]).all()
        )

    def shift_grid(grid: Grid, dr: int, dc: int) -> Grid:
        grid = arr(grid)
        bg = background(grid)
        out = np.full_like(grid, bg)
        coords = np.argwhere(grid != bg)
        for r, c in coords:
            out[int(r) + dr, int(c) + dc] = grid[int(r), int(c)]
        return out

    def shift_panels(grid: Grid, dr: int, dc: int) -> Grid:
        p = split_panels(arr(grid))
        return join_panels(shift_grid(p.a, dr, dc), shift_grid(p.b, dr, dc), p.orientation)

    def shift_skip(dr: int, dc: int):
        def skip(input_grid: Grid, output_grid: Grid) -> str | None:
            p = split_panels(arr(input_grid))
            if not can_shift_grid(p.a, dr, dc):
                return "panel_a_would_clip"
            if not can_shift_grid(p.b, dr, dc):
                return "panel_b_would_clip"
            if not can_shift_grid(arr(output_grid), dr, dc):
                return "output_would_clip"
            return None

        return skip

    for name, dr, dc in (
        ("task366_shift_panels_down_right", 1, 1),
        ("task366_shift_panels_up_left", -1, -1),
    ):
        contracts.append(
            ph.InvarianceContract(
                name=name,
                input_transform=lambda grid, dr=dr, dc=dc: shift_panels(arr(grid), dr, dc),
                expected_output_transform=lambda grid, dr=dr, dc=dc: shift_grid(arr(grid), dr, dc),
                max_input_shape=max_shape,
                max_output_shape=max_shape,
                skip_if=shift_skip(dr, dc),
                description="Translate non-background content inside each panel.",
            )
        )

    return contracts


def run_pseudo_hidden(task: dict) -> tuple[list[dict[str, object]], dict[str, int], str]:
    from tools import pseudo_hidden as ph

    examples = [example for _, _, example in all_examples(task)]
    contracts = task366_contracts()
    results = ph.run_contracts(
        lambda grid: solver_bounded_rectangles(arr(grid), max_dim=7),
        examples,
        contracts,
        include_original=True,
        rule_label="bounded_rectangles_max7",
    )
    rows = [asdict(r) for r in results]
    summary = ph.summarize_results(results)
    failures = ph.format_failures(results, limit=40)
    return rows, summary, failures


def onnx_anchor_comparison(task: dict, anchor_path: Path = ANCHOR_ONNX) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not anchor_path.exists():
        return rows
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - optional dependency
        return [{"status": f"onnxruntime_unavailable:{type(exc).__name__}"}]

    from tools.neurogolf_local import encode_grid

    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(anchor_path), options, providers=["CPUExecutionProvider"])

    for split, index, example in all_examples(task):
        g = arr(example["input"])
        target = arr(example["output"])
        if g.shape[0] > 30 or g.shape[1] > 30 or target.shape[0] > 30 or target.shape[1] > 30:
            rows.append(
                {
                    "split": split,
                    "index": index,
                    "encodable": 0,
                    "matches_target": "",
                    "matches_bounded_rectangles": "",
                    "input_shape": shape_s(g),
                    "output_shape": shape_s(target),
                }
            )
            continue
        output = session.run(["output"], {"input": encode_grid(example["input"])})[0] > 0.0
        target_tensor = encode_grid(example["output"]) > 0.0
        semantic_tensor = encode_grid(solver_bounded_rectangles(g, max_dim=7).tolist()) > 0.0
        rows.append(
            {
                "split": split,
                "index": index,
                "encodable": 1,
                "matches_target": int(np.array_equal(output, target_tensor)),
                "matches_bounded_rectangles": int(np.array_equal(output, semantic_tensor)),
                "input_shape": shape_s(g),
                "output_shape": shape_s(target),
            }
        )
    return rows


def write_report(
    stats: dict[str, object],
    eval_summary: list[dict[str, object]],
    pseudo_summary: dict[str, int],
    pseudo_failures: str,
    onnx_rows: list[dict[str, object]],
) -> None:
    exact_by_solver: dict[str, tuple[int, int]] = {}
    label_by_solver: dict[str, tuple[int, int]] = {}
    for row in eval_summary:
        solver = str(row["solver"])
        exact, total = exact_by_solver.get(solver, (0, 0))
        same, total_label = label_by_solver.get(solver, (0, 0))
        exact_by_solver[solver] = (exact + int(row["exact"]), total + int(row["total"]))
        label_by_solver[solver] = (
            same + int(row["same_as_label"]),
            total_label + int(row["total"]),
        )

    onnx_enc = [r for r in onnx_rows if r.get("encodable") == 1]
    onnx_match_target = sum(int(r["matches_target"]) for r in onnx_enc) if onnx_enc else 0
    onnx_match_sem = sum(int(r["matches_bounded_rectangles"]) for r in onnx_enc) if onnx_enc else 0

    lines = [
        "# Task366 Semantic V2",
        "",
        "## Data closure",
        "",
        f"- Examples: {stats['examples']} total; splits {stats['totals']}.",
        f"- Encodable by fixed 30x30 ONNX tensor: {stats['encodable']}.",
        f"- Source components recovered from the reference rule: {stats['patterns']} patterns.",
        f"- Filled-rectangle audit: max height {stats['max_pattern_h']}, "
        f"max width {stats['max_pattern_w']}, non-rectangular violations {stats['full_rect_violations']}.",
        f"- Source side distribution: {stats['source_side']}.",
        "",
        "## Hypotheses",
        "",
    ]

    for solver, (exact, total) in sorted(exact_by_solver.items()):
        same, _ = label_by_solver[solver]
        lines.append(f"- {solver}: target exact {exact}/{total}; same as label rule {same}/{total}.")

    lines.extend(
        [
            "",
            "The bounded rule is: split on the longer axis, choose the denser non-background "
            "panel as source, recover filled source rectangles up to 7x7, then place each "
            "rectangle on the sparse panel when all cells of a shared key color coincide "
            "with unused anchors of that color. Weaker variants intentionally fail, which "
            "keeps the rule from collapsing into a looser copy-anywhere story.",
            "",
            "## Pseudo-hidden",
            "",
            f"- Summary: {pseudo_summary}.",
            f"- Failures: {'none' if not pseudo_failures else pseudo_failures}",
            "",
            "Contracts used: original, arbitrary color bijection, source/destination panel swap, "
            "whole-grid transpose, per-panel horizontal mirror, and in-panel translations "
            "when they do not clip content.",
            "",
            "## Existing ONNX",
            "",
            f"- Anchor model compared on encodable rows: target {onnx_match_target}/{len(onnx_enc)}, "
            f"semantic {onnx_match_sem}/{len(onnx_enc)}.",
            "- Anchor cost reference from existing reports: 830720 cost, 713 nodes, 90138 bytes.",
            "",
            "## ONNX decision",
            "",
            "Semantic closure is strong in Python. A smaller ONNX is justified only if the "
            "bounded-rectangle rule can be compiled without reintroducing label propagation. "
            "The prototype leaves builder work to tools/build_task366_v2.py.",
            "",
        ]
    )
    (REPORT_DIR / "task366_semantic_v2.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    task = load_task(args.data)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    label_outputs = {
        (split, index): solver_label(arr(example["input"]))
        for split, index, example in all_examples(task)
    }

    solvers: list[tuple[str, Solver]] = [
        ("label_components_reference", solver_label),
        ("bounded_rectangles_max7", lambda grid: solver_bounded_rectangles(grid, max_dim=7)),
        ("bounded_rectangles_max6", lambda grid: solver_bounded_rectangles(grid, max_dim=6)),
        ("first_panel_is_source", solver_first_panel_source),
        ("no_pattern_priority", solver_no_pattern_priority),
        ("no_anchor_locking", solver_no_anchor_locking),
    ]

    eval_rows: list[EvalRow] = []
    for name, solver in solvers:
        eval_rows.extend(evaluate_solver(name, solver, task, label_outputs))

    eval_detail = [asdict(row) for row in eval_rows]
    eval_summary = summarize_eval(eval_rows)
    write_csv(REPORT_DIR / "task366_v2_equivalence.csv", eval_detail)
    write_csv(REPORT_DIR / "task366_v2_hypotheses.csv", eval_summary)

    stats = panel_shape_stats(task)
    (REPORT_DIR / "task366_v2_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))

    pseudo_rows, pseudo_summary, pseudo_failures = run_pseudo_hidden(task)
    write_csv(REPORT_DIR / "task366_v2_pseudo_hidden.csv", pseudo_rows)
    (REPORT_DIR / "task366_v2_pseudo_hidden_summary.json").write_text(
        json.dumps(pseudo_summary, indent=2, sort_keys=True)
    )
    (REPORT_DIR / "task366_v2_pseudo_hidden_failures.txt").write_text(pseudo_failures)

    onnx_rows = onnx_anchor_comparison(task)
    write_csv(REPORT_DIR / "task366_v2_existing_onnx.csv", onnx_rows)

    write_report(stats, eval_summary, pseudo_summary, pseudo_failures, onnx_rows)

    print("wrote reports/task366_semantic_v2.md")
    print(json.dumps({"eval": eval_summary, "pseudo": pseudo_summary}, indent=2))


if __name__ == "__main__":
    main()
