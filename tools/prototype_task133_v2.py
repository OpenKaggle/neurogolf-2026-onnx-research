#!/usr/bin/env python3
"""Task133 semantic rebuild prototype.

Task133 is a stencil-completion task. Every nonzero component contains one
shared anchor color and one unique payload color. The most complete component
defines a low-resolution stencil in anchor-sized tiles; every other component
is completed by stamping that same payload-tile set around its anchor block.

This file is intentionally both solver and falsification harness. It records
visible/arc-gen accuracy, weaker hypotheses, explicit pseudo-hidden contracts,
and leave-one-out checks.
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
sys.dont_write_bytecode = True

DATA_PATH = ROOT / "data" / "neurogolf-2026" / "raw" / "task133.json"
REPORT_DIR = ROOT / "reports"

Grid = np.ndarray
Solver = Callable[[Grid], Grid]


@dataclass(frozen=True)
class Component:
    label: int
    mask: Grid
    colors: frozenset[int]
    anchor_color: int
    payload_color: int
    anchor_top: int
    anchor_left: int
    anchor_h: int
    anchor_w: int
    tile_colors: dict[tuple[int, int], int]

    @property
    def payload_tiles(self) -> frozenset[tuple[int, int]]:
        return frozenset(pos for pos, color in self.tile_colors.items() if color == self.payload_color)

    @property
    def tile_count(self) -> int:
        return len(self.tile_colors)

    @property
    def payload_tile_count(self) -> int:
        return len(self.payload_tiles)

    @property
    def tile_bbox_area(self) -> int:
        coords = np.asarray(list(self.tile_colors), dtype=np.int64)
        if coords.size == 0:
            return 0
        lo = coords.min(axis=0)
        hi = coords.max(axis=0)
        return int((hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1))


@dataclass(frozen=True)
class EvalRow:
    solver: str
    split: str
    index: int
    exact: int
    pred_shape: str
    target_shape: str
    pixel_errors: int | None


@dataclass(frozen=True)
class ComponentRow:
    split: str
    index: int
    component: int
    anchor_color: int
    payload_color: int
    anchor_shape: str
    observed_tiles: str
    completed_tiles: str
    input_tile_count: int
    output_tile_count: int
    is_template: int


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


def label_4conn(mask: Grid) -> tuple[Grid, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int64)
    label = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            label += 1
            stack = [(r, c)]
            labels[r, c] = label
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = label
                        stack.append((nr, nc))
    return labels, label


def shared_anchor_color(grid: Grid) -> int | None:
    labels, n_labels = label_4conn(grid != 0)
    if n_labels == 0:
        return None
    color_sets: list[set[int]] = []
    color_components: dict[int, set[int]] = {color: set() for color in range(1, 10)}
    for label in range(1, n_labels + 1):
        colors = {int(x) for x in np.unique(grid[labels == label]) if int(x) != 0}
        if not colors:
            continue
        color_sets.append(colors)
        for color in colors:
            color_components[color].add(label)
    common = set.intersection(*color_sets) if color_sets else set()
    repeated = {color for color, comps in color_components.items() if len(comps) >= 2}
    if len(common) == 1 and repeated == common:
        return next(iter(common))
    if len(common) == 1:
        return next(iter(common))
    return None


def component_tiles(grid: Grid, labels: Grid, label: int, anchor_color: int) -> Component | None:
    mask = labels == label
    colors = {int(x) for x in np.unique(grid[mask]) if int(x) != 0}
    payloads = sorted(colors - {anchor_color})
    if len(payloads) != 1:
        return None
    payload_color = payloads[0]

    anchor_coords = np.argwhere(mask & (grid == anchor_color))
    if len(anchor_coords) == 0:
        return None
    ar0, ac0 = [int(x) for x in anchor_coords.min(axis=0)]
    ar1, ac1 = [int(x) for x in anchor_coords.max(axis=0)]
    anchor_h = ar1 - ar0 + 1
    anchor_w = ac1 - ac0 + 1
    anchor_block = grid[ar0 : ar1 + 1, ac0 : ac1 + 1]
    if not np.all(anchor_block == anchor_color):
        return None

    tile_colors: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(mask):
        tr = (int(r) - ar0) // anchor_h
        tc = (int(c) - ac0) // anchor_w
        key = (tr, tc)
        color = int(grid[int(r), int(c)])
        existing = tile_colors.get(key)
        if existing is not None and existing != color:
            return None
        tile_colors[key] = color

    return Component(
        label=label,
        mask=mask,
        colors=frozenset(colors),
        anchor_color=anchor_color,
        payload_color=payload_color,
        anchor_top=ar0,
        anchor_left=ac0,
        anchor_h=anchor_h,
        anchor_w=anchor_w,
        tile_colors=tile_colors,
    )


def extract_components(grid: Grid) -> list[Component]:
    anchor_color = shared_anchor_color(grid)
    if anchor_color is None:
        return []
    labels, n_labels = label_4conn(grid != 0)
    components: list[Component] = []
    for label in range(1, n_labels + 1):
        component = component_tiles(grid, labels, label, anchor_color)
        if component is None:
            return []
        components.append(component)
    return components


def choose_template(components: Sequence[Component]) -> Component | None:
    if not components:
        return None
    return max(
        components,
        key=lambda component: (
            component.tile_count,
            component.payload_tile_count,
            component.tile_bbox_area,
            -component.label,
        ),
    )


def solve_task133(grid: Grid) -> Grid:
    grid = arr(grid)
    components = extract_components(grid)
    template = choose_template(components)
    if template is None:
        return grid.copy()

    out = grid.copy()
    h, w = out.shape
    for component in components:
        for tr, tc in template.payload_tiles:
            r0 = component.anchor_top + tr * component.anchor_h
            c0 = component.anchor_left + tc * component.anchor_w
            r1 = r0 + component.anchor_h
            c1 = c0 + component.anchor_w
            if r0 >= h or c0 >= w or r1 <= 0 or c1 <= 0:
                continue
            rr0, rr1 = max(0, r0), min(h, r1)
            cc0, cc1 = max(0, c0), min(w, c1)
            region = out[rr0:rr1, cc0:cc1]
            region[region == 0] = component.payload_color
    return out


def solve_global_bbox_fill(grid: Grid) -> Grid:
    """Weaker hypothesis: fill each component's final bounding box with payload."""

    grid = arr(grid)
    components = extract_components(grid)
    template = choose_template(components)
    if template is None:
        return grid.copy()
    offsets = np.asarray(list(template.payload_tiles | {(0, 0)}), dtype=np.int64)
    tr0, tc0 = offsets.min(axis=0)
    tr1, tc1 = offsets.max(axis=0)

    out = grid.copy()
    h, w = out.shape
    for component in components:
        r0 = component.anchor_top + int(tr0) * component.anchor_h
        c0 = component.anchor_left + int(tc0) * component.anchor_w
        r1 = component.anchor_top + (int(tr1) + 1) * component.anchor_h
        c1 = component.anchor_left + (int(tc1) + 1) * component.anchor_w
        rr0, rr1 = max(0, r0), min(h, r1)
        cc0, cc1 = max(0, c0), min(w, c1)
        region = out[rr0:rr1, cc0:cc1]
        region[region == 0] = component.payload_color
    return out


def solve_per_component_self_copy(grid: Grid) -> Grid:
    """Weaker hypothesis: extend only the observed payload tile offsets."""

    grid = arr(grid)
    components = extract_components(grid)
    if not components:
        return grid.copy()
    out = grid.copy()
    h, w = out.shape
    for component in components:
        for tr, tc in component.payload_tiles:
            r0 = component.anchor_top + tr * component.anchor_h
            c0 = component.anchor_left + tc * component.anchor_w
            r1 = r0 + component.anchor_h
            c1 = c0 + component.anchor_w
            if r0 >= h or c0 >= w or r1 <= 0 or c1 <= 0:
                continue
            region = out[max(0, r0) : min(h, r1), max(0, c0) : min(w, c1)]
            region[region == 0] = component.payload_color
    return out


def solve_template_no_scaling(grid: Grid) -> Grid:
    """Weaker hypothesis: stamp one cell per template tile, ignoring anchor scale."""

    grid = arr(grid)
    components = extract_components(grid)
    template = choose_template(components)
    if template is None:
        return grid.copy()
    out = grid.copy()
    h, w = out.shape
    for component in components:
        for tr, tc in template.payload_tiles:
            r = component.anchor_top + tr
            c = component.anchor_left + tc
            if 0 <= r < h and 0 <= c < w and out[r, c] == 0:
                out[r, c] = component.payload_color
    return out


def evaluate_solver(name: str, solver: Solver, task: dict) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for split, index, example in all_examples(task):
        target = arr(example["output"])
        pred = solver(arr(example["input"]))
        same_shape = pred.shape == target.shape
        exact = int(same_shape and np.array_equal(pred, target))
        rows.append(
            EvalRow(
                solver=name,
                split=split,
                index=index,
                exact=exact,
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
                "exact": sum(row.exact for row in group),
                "total": len(group),
                "pixel_errors": sum(row.pixel_errors or 0 for row in group),
            }
        )
    return out


def tile_set_s(tiles: Iterable[tuple[int, int]]) -> str:
    return " ".join(f"{r}:{c}" for r, c in sorted(tiles))


def component_rows(task: dict) -> list[ComponentRow]:
    rows: list[ComponentRow] = []
    for split, index, example in all_examples(task):
        grid = arr(example["input"])
        output = arr(example["output"])
        components = extract_components(grid)
        template = choose_template(components)
        out_components = extract_components(output)
        by_payload = {component.payload_color: component for component in out_components}
        for component in components:
            completed = by_payload.get(component.payload_color, component)
            rows.append(
                ComponentRow(
                    split=split,
                    index=index,
                    component=component.label,
                    anchor_color=component.anchor_color,
                    payload_color=component.payload_color,
                    anchor_shape=f"{component.anchor_h}x{component.anchor_w}",
                    observed_tiles=tile_set_s(component.tile_colors),
                    completed_tiles=tile_set_s(completed.tile_colors),
                    input_tile_count=component.tile_count,
                    output_tile_count=completed.tile_count,
                    is_template=int(template is not None and component.label == template.label),
                )
            )
    return rows


def evidence_stats(task: dict) -> dict[str, object]:
    examples = all_examples(task)
    anchor_shapes: dict[str, int] = {}
    n_components: dict[str, int] = {}
    template_tiles: dict[str, int] = {}
    anchor_proxy_ok = 0
    payload_unique_ok = 0
    for _, _, example in examples:
        grid = arr(example["input"])
        components = extract_components(grid)
        n_components[str(len(components))] = n_components.get(str(len(components)), 0) + 1
        if components:
            anchor = components[0].anchor_color
            color_to_components: dict[int, set[int]] = {color: set() for color in range(1, 10)}
            for component in components:
                for color in component.colors:
                    color_to_components[color].add(component.label)
            repeated = {color for color, labels in color_to_components.items() if len(labels) >= 2}
            if repeated == {anchor}:
                anchor_proxy_ok += 1
            payloads = [component.payload_color for component in components]
            if len(payloads) == len(set(payloads)):
                payload_unique_ok += 1
        template = choose_template(components)
        if template is not None:
            template_tiles[tile_set_s(template.tile_colors)] = template_tiles.get(tile_set_s(template.tile_colors), 0) + 1
        for component in components:
            key = f"{component.anchor_h}x{component.anchor_w}"
            anchor_shapes[key] = anchor_shapes.get(key, 0) + 1
    return {
        "examples": len(examples),
        "train": len(task.get("train", [])),
        "test": len(task.get("test", [])),
        "arc_gen": len(task.get("arc-gen", [])),
        "component_count_distribution": n_components,
        "anchor_shape_distribution": anchor_shapes,
        "anchor_proxy_ok": anchor_proxy_ok,
        "payload_unique_ok": payload_unique_ok,
        "template_tile_patterns": template_tiles,
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


def task133_contracts():
    from tools import pseudo_hidden as ph

    max_shape = (30, 30)
    contracts: list[ph.InvarianceContract] = []
    contracts.append(
        ph.color_permutation_contract(
            {0: 0, 1: 7, 2: 5, 3: 8, 4: 2, 5: 9, 6: 3, 7: 1, 8: 6, 9: 4},
            name="task133_color_bijection",
            max_shape=max_shape,
        )
    )
    contracts.append(
        ph.InvarianceContract(
            name="task133_transpose",
            input_transform=lambda grid: arr(grid).T,
            expected_output_transform=lambda grid: arr(grid).T,
            max_input_shape=max_shape,
            max_output_shape=max_shape,
            description="Transpose preserves the tile-stencil relation.",
        )
    )
    contracts.append(
        ph.InvarianceContract(
            name="task133_mirror_lr",
            input_transform=lambda grid: np.fliplr(arr(grid)),
            expected_output_transform=lambda grid: np.fliplr(arr(grid)),
            max_input_shape=max_shape,
            max_output_shape=max_shape,
            description="Horizontal reflection mirrors the stencil offsets.",
        )
    )
    contracts.append(
        ph.InvarianceContract(
            name="task133_mirror_ud",
            input_transform=lambda grid: np.flipud(arr(grid)),
            expected_output_transform=lambda grid: np.flipud(arr(grid)),
            max_input_shape=max_shape,
            max_output_shape=max_shape,
            description="Vertical reflection mirrors the stencil offsets.",
        )
    )
    contracts.append(
        ph.InvarianceContract(
            name="task133_zero_pad_top_left",
            input_transform=lambda grid: ph.pad_zero_border(grid, top=1, left=1),
            expected_output_transform=lambda grid: ph.pad_zero_border(grid, top=1, left=1),
            max_input_shape=max_shape,
            max_output_shape=max_shape,
            description="Translation by adding a zero border should translate output.",
        )
    )
    contracts.append(
        ph.InvarianceContract(
            name="task133_zero_pad_bottom_right",
            input_transform=lambda grid: ph.pad_zero_border(grid, bottom=1, right=1),
            expected_output_transform=lambda grid: ph.pad_zero_border(grid, bottom=1, right=1),
            max_input_shape=max_shape,
            max_output_shape=max_shape,
            description="Padding at the lower/right edge keeps relative offsets unchanged.",
        )
    )
    return contracts


def run_pseudo_hidden(task: dict) -> tuple[list[dict[str, object]], dict[str, int], str]:
    from tools import pseudo_hidden as ph

    examples = [example for _, _, example in all_examples(task)]
    results = ph.run_contracts(
        lambda grid: solve_task133(arr(grid)),
        examples,
        task133_contracts(),
        include_original=True,
        rule_label="task133_stencil_v2",
    )
    rows = [asdict(result) for result in results]
    return rows, ph.summarize_results(results), ph.format_failures(results, limit=20)


def run_leave_one_out(task: dict) -> tuple[list[dict[str, object]], dict[str, int], str]:
    from tools import pseudo_hidden as ph

    examples = [example for split in ("train", "test") for example in task.get(split, [])]

    def callback(train_subset, heldout, heldout_index):
        del train_subset, heldout
        return ph.RuleSpec(
            solver=lambda grid: solve_task133(arr(grid)),
            contracts=task133_contracts(),
            label=f"loo_{heldout_index}",
        )

    results = ph.leave_one_out(callback, examples, include_original=True)
    rows = [asdict(result) for result in results]
    return rows, ph.summarize_results(results), ph.format_failures(results, limit=20)


def write_report(
    report_path: Path,
    stats: dict[str, object],
    eval_summary: list[dict[str, object]],
    pseudo_summary: dict[str, int],
    pseudo_failures: str,
    loo_summary: dict[str, int],
    loo_failures: str,
) -> None:
    anchor_cost = 196_680
    konbu_cost = 193_923
    anchor_score = 12.810666678474124
    konbu_score = 12.824783547959548
    anchor_ph = ROOT / "reports" / "task133_v2_anchor_pseudo_hidden_summary.json"
    anchor_ph_text = ""
    if anchor_ph.exists():
        try:
            anchor_ph_text = json.dumps(json.loads(anchor_ph.read_text()), sort_keys=True)
        except json.JSONDecodeError:
            anchor_ph_text = anchor_ph.read_text().strip()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Task133 Semantic Rebuild v2",
                "",
                "## Rule",
                "",
                "Each nonzero connected component contains exactly two nonzero colors: one shared anchor color and one unique payload color. "
                "The shared anchor color is also the only nonzero color that appears in more than one component. The component with the most occupied "
                "anchor-scaled tiles defines the complete stencil. Copy that stencil's payload-tile offsets onto every component, replacing the payload "
                "color and scaling every tile by that component's anchor block size. Existing nonzero cells are preserved.",
                "",
                "## Evidence",
                "",
                f"- Examples checked: {stats['examples']} ({stats['train']} train, {stats['test']} test, {stats['arc_gen']} arc-gen).",
                f"- Anchor proxy success: {stats['anchor_proxy_ok']}/{stats['examples']} examples.",
                f"- Unique payload colors: {stats['payload_unique_ok']}/{stats['examples']} examples.",
                f"- Component-count distribution: {stats['component_count_distribution']}.",
                f"- Anchor-shape distribution: {stats['anchor_shape_distribution']}.",
                "",
                "## Visible And Arc-Gen Accuracy",
                "",
                "| solver | split | exact | total | pixel_errors |",
                "|---|---:|---:|---:|---:|",
                *[
                    f"| {row['solver']} | {row['split']} | {row['exact']} | {row['total']} | {row['pixel_errors']} |"
                    for row in eval_summary
                ],
                "",
                "The weaker hypotheses fail on visible and/or arc-gen: bounding-box fill overwrites holes, self-copy never adds unseen stencil tiles, "
                "and no-scaling misses multi-cell anchor blocks.",
                "",
                "## Pseudo-Hidden Checks",
                "",
                f"- Metamorphic summary: {pseudo_summary}.",
                f"- Metamorphic failures: {pseudo_failures or 'none'}.",
                f"- Leave-one-out summary: {loo_summary}.",
                f"- Leave-one-out failures: {loo_failures or 'none'}.",
                "",
                "Contracts used: original, explicit color bijection, transpose, left-right mirror, up-down mirror, and zero-border translations where "
                "the transformed grids remain within 30x30.",
                "",
                "## ONNX Decision",
                "",
                "The Python semantics are strong, but I did not write a new ONNX builder. Existing compact anchors already pass all 267 known examples: "
                f"v4_plus9 task133 cost {anchor_cost}, score {anchor_score:.6f}; konbu17 task133 cost {konbu_cost}, score {konbu_score:.6f}. "
                "Inspecting v4_plus9 shows it is already a compact semantic graph: it finds the anchor color from repeated color adjacency, extracts "
                "a 7x7 template pattern, resizes it for anchor blocks, and paints with convolutions. It also passes the same metamorphic suite as the "
                f"Python solver. ONNX pseudo-hidden summary: {anchor_ph_text or 'not rerun in report generation'}. "
                "A fresh builder would need to beat this existing graph, and the obvious connected-component/scatter route is likely to exceed the "
                "current memory budget.",
                "",
                "## Estimated Gain",
                "",
                "Estimated gain for this branch is 0. The tempting konbu17 graph is about +0.014 local score by cost, but it fails 1293/1785 non-skipped "
                "metamorphic checks, so it is likely a visible-example lookup rather than safe semantics. A new semantic handbuild would need to beat "
                "196,680 cost while preserving the v4_plus9 metamorphic behavior.",
                "",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--reports", default=str(REPORT_DIR))
    args = parser.parse_args()

    task = load_task(Path(args.data))
    report_dir = Path(args.reports)
    report_dir.mkdir(parents=True, exist_ok=True)

    solvers: list[tuple[str, Solver]] = [
        ("stencil_v2", solve_task133),
        ("global_bbox_fill", solve_global_bbox_fill),
        ("self_copy_only", solve_per_component_self_copy),
        ("template_no_scaling", solve_template_no_scaling),
    ]
    eval_rows: list[EvalRow] = []
    for name, solver in solvers:
        eval_rows.extend(evaluate_solver(name, solver, task))
    eval_summary = summarize_eval(eval_rows)

    pseudo_rows, pseudo_summary, pseudo_failures = run_pseudo_hidden(task)
    loo_rows, loo_summary, loo_failures = run_leave_one_out(task)
    stats = evidence_stats(task)

    write_csv(report_dir / "task133_v2_eval.csv", [asdict(row) for row in eval_rows])
    write_csv(report_dir / "task133_v2_summary.csv", eval_summary)
    write_csv(report_dir / "task133_v2_components.csv", [asdict(row) for row in component_rows(task)])
    write_csv(report_dir / "task133_v2_pseudo_hidden.csv", pseudo_rows)
    write_csv(report_dir / "task133_v2_leave_one_out.csv", loo_rows)
    (report_dir / "task133_v2_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    (report_dir / "task133_v2_pseudo_hidden_summary.json").write_text(
        json.dumps({"pseudo_hidden": pseudo_summary, "leave_one_out": loo_summary}, indent=2, sort_keys=True)
    )
    write_report(
        report_dir / "task133_semantic_v2.md",
        stats,
        eval_summary,
        pseudo_summary,
        pseudo_failures,
        loo_summary,
        loo_failures,
    )

    print(json.dumps({"eval": eval_summary, "pseudo_hidden": pseudo_summary, "leave_one_out": loo_summary}, indent=2))


if __name__ == "__main__":
    main()
