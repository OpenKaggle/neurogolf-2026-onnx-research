#!/usr/bin/env python3
"""Task285 semantic rebuild prototype and falsification harness.

Semantics found from visible + arc-gen data:
1. Split each 8-connected non-zero object into a majority-color body and
   minority-color anchor cells.
2. Each anchor chooses a direction from the body bounding box.
3. Reflect the body across the chosen side/corner of its bounding box and
   paint the reflected zero cells with the anchor color.

The script intentionally records weaker hypotheses, pseudo-hidden contracts,
leave-one-out checks, and the current anchor ONNX comparison. It does not build
ONNX; the exact object-level rule is strong, but compiling connected-component
majority/bbox semantics compactly is not justified by the current graph budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools import pseudo_hidden as ph
except ImportError:  # pragma: no cover - supports direct script execution.
    import pseudo_hidden as ph


DATA_PATH = ROOT / "data" / "neurogolf-2026" / "raw" / "task285.json"
REPORT_DIR = ROOT / "reports"
ANCHOR_ONNX = ROOT / "submissions" / "candidate_v4_plus9_compiler_onnx" / "task285.onnx"

NEIGHBORS_8 = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr != 0 or dc != 0
)

Grid = np.ndarray
Solver = Callable[[Grid], Grid]


@dataclass(frozen=True)
class Component:
    cells: tuple[tuple[int, int], ...]
    counts: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class HypothesisRow:
    solver: str
    split: str
    exact: int
    total: int
    pixel_errors: int


@dataclass(frozen=True)
class StructureRow:
    split: str
    index: int
    shape: str
    components: int
    anchor_cells: int
    copied_cells: int
    added_cells: int
    unique_majority: int
    max_body_cells: int
    max_markers: int


def load_task(path: Path = DATA_PATH) -> dict:
    return json.loads(path.read_text())


def all_examples(task: dict) -> list[tuple[str, int, dict]]:
    rows: list[tuple[str, int, dict]] = []
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(task.get(split, [])):
            rows.append((split, index, example))
    return rows


def visible_examples(task: dict) -> list[dict]:
    return list(task.get("train", [])) + list(task.get("test", []))


def arr(grid: Sequence[Sequence[int]] | Grid) -> Grid:
    return np.asarray(grid, dtype=np.int64)


def bbox(cells: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def foreground_components(grid: Grid) -> list[Component]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[Component] = []

    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS_8:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and grid[nr, nc] != 0
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            counts = tuple(sorted(Counter(int(grid[x, y]) for x, y in cells).items()))
            components.append(Component(tuple(sorted(cells)), counts, bbox(cells)))
    return components


def color_components(grid: Grid) -> list[Component]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[Component] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS_8:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(
                Component(
                    tuple(sorted(cells)),
                    ((color, len(cells)),),
                    bbox(cells),
                )
            )
    return components


def unique_majority_color(component: Component) -> int | None:
    positive = [(color, count) for color, count in component.counts if color != 0]
    if not positive:
        return None
    max_count = max(count for _, count in positive)
    winners = [color for color, count in positive if count == max_count]
    if len(winners) != 1 or max_count <= 1:
        return None
    return int(winners[0])


def direction_from_bbox(
    cell: tuple[int, int], body_bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    r, c = cell
    r0, r1, c0, c1 = body_bbox
    dr = -1 if r < r0 else (1 if r > r1 else 0)
    dc = -1 if c < c0 else (1 if c > c1 else 0)
    return dr, dc


def reflected_cells(
    cells: Sequence[tuple[int, int]],
    body_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = body_bbox
    dr, dc = direction
    reflected: list[tuple[int, int]] = []
    for r, c in cells:
        rr = r if dr == 0 else (2 * r1 + 1 - r if dr > 0 else 2 * r0 - 1 - r)
        cc = c if dc == 0 else (2 * c1 + 1 - c if dc > 0 else 2 * c0 - 1 - c)
        reflected.append((int(rr), int(cc)))
    return reflected


def in_bounds(shape: tuple[int, int], cells: Iterable[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def solver_majority_reflect(grid: Grid) -> Grid:
    out = grid.copy()
    for component in foreground_components(grid):
        body_color = unique_majority_color(component)
        if body_color is None:
            continue

        body_cells = tuple((r, c) for r, c in component.cells if grid[r, c] == body_color)
        body_box = bbox(body_cells)
        marker_cells = [
            (r, c, int(grid[r, c]))
            for r, c in component.cells
            if int(grid[r, c]) != body_color
        ]

        for mr, mc, marker_color in marker_cells:
            direction = direction_from_bbox((mr, mc), body_box)
            if direction == (0, 0):
                continue
            target_cells = reflected_cells(body_cells, body_box, direction)
            if not in_bounds(grid.shape, target_cells):
                continue
            for tr, tc in target_cells:
                if out[tr, tc] == 0:
                    out[tr, tc] = marker_color
    return out


def solver_majority_no_diagonal(grid: Grid) -> Grid:
    out = grid.copy()
    for component in foreground_components(grid):
        body_color = unique_majority_color(component)
        if body_color is None:
            continue
        body_cells = tuple((r, c) for r, c in component.cells if grid[r, c] == body_color)
        body_box = bbox(body_cells)
        for mr, mc in ((r, c) for r, c in component.cells if int(grid[r, c]) != body_color):
            direction = direction_from_bbox((mr, mc), body_box)
            if direction == (0, 0) or (direction[0] != 0 and direction[1] != 0):
                continue
            target_cells = reflected_cells(body_cells, body_box, direction)
            if in_bounds(grid.shape, target_cells):
                for tr, tc in target_cells:
                    if out[tr, tc] == 0:
                        out[tr, tc] = int(grid[mr, mc])
    return out


def solver_majority_body_color(grid: Grid) -> Grid:
    out = grid.copy()
    for component in foreground_components(grid):
        body_color = unique_majority_color(component)
        if body_color is None:
            continue
        body_cells = tuple((r, c) for r, c in component.cells if grid[r, c] == body_color)
        body_box = bbox(body_cells)
        for mr, mc in ((r, c) for r, c in component.cells if int(grid[r, c]) != body_color):
            direction = direction_from_bbox((mr, mc), body_box)
            if direction == (0, 0):
                continue
            target_cells = reflected_cells(body_cells, body_box, direction)
            if in_bounds(grid.shape, target_cells):
                for tr, tc in target_cells:
                    if out[tr, tc] == 0:
                        out[tr, tc] = body_color
    return out


def solver_color_component_any_marker(grid: Grid) -> Grid:
    out = grid.copy()
    for component in color_components(grid):
        color = component.counts[0][0]
        if len(component.cells) < 2:
            continue
        for direction in (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ):
            target_cells = reflected_cells(component.cells, component.bbox, direction)
            if not in_bounds(grid.shape, target_cells):
                continue
            marker_colors = sorted(
                {
                    int(grid[r, c])
                    for r, c in target_cells
                    if grid[r, c] != 0 and int(grid[r, c]) != color
                }
            )
            for marker_color in marker_colors:
                for tr, tc in target_cells:
                    if out[tr, tc] == 0:
                        out[tr, tc] = marker_color
    return out


def solver_color_component_singleton_marker(grid: Grid) -> Grid:
    components = color_components(grid)
    component_id = -np.ones(grid.shape, dtype=np.int64)
    for index, component in enumerate(components):
        for r, c in component.cells:
            component_id[r, c] = index

    out = grid.copy()
    for component in components:
        color = component.counts[0][0]
        if len(component.cells) < 2:
            continue
        for direction in (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ):
            target_cells = reflected_cells(component.cells, component.bbox, direction)
            if not in_bounds(grid.shape, target_cells):
                continue
            markers: list[int] = []
            valid = True
            for r, c in target_cells:
                if grid[r, c] == 0 or int(grid[r, c]) == color:
                    continue
                marker_component = components[int(component_id[r, c])]
                if len(marker_component.cells) != 1:
                    valid = False
                    break
                markers.append(int(grid[r, c]))
            if not valid or not markers:
                continue
            for marker_color in sorted(set(markers)):
                for tr, tc in target_cells:
                    if out[tr, tc] == 0:
                        out[tr, tc] = marker_color
    return out


SOLVERS: dict[str, Solver] = {
    "semantic_majority_reflect": solver_majority_reflect,
    "weaker_same_color_any_marker": solver_color_component_any_marker,
    "weaker_same_color_singleton_marker": solver_color_component_singleton_marker,
    "weaker_majority_no_diagonal": solver_majority_no_diagonal,
    "weaker_majority_paints_body_color": solver_majority_body_color,
}


def evaluate_solver(task: dict, name: str, solver: Solver) -> list[HypothesisRow]:
    rows: list[HypothesisRow] = []
    for split in ("train", "test", "arc-gen"):
        exact = 0
        total = 0
        pixel_errors = 0
        for example in task.get(split, []):
            pred = solver(arr(example["input"]))
            target = arr(example["output"])
            exact += int(np.array_equal(pred, target))
            total += 1
            pixel_errors += int(np.count_nonzero(pred != target))
        rows.append(HypothesisRow(name, split, exact, total, pixel_errors))
    all_exact = sum(row.exact for row in rows)
    all_total = sum(row.total for row in rows)
    all_errors = sum(row.pixel_errors for row in rows)
    rows.append(HypothesisRow(name, "all", all_exact, all_total, all_errors))
    return rows


def structure_rows(task: dict) -> list[StructureRow]:
    rows: list[StructureRow] = []
    for split, index, example in all_examples(task):
        grid = arr(example["input"])
        target = arr(example["output"])
        added = int(np.count_nonzero((target != grid) & (grid == 0)))
        components = foreground_components(grid)
        anchor_cells = 0
        copied_cells = 0
        unique = 1
        max_body = 0
        max_markers = 0
        for component in components:
            body_color = unique_majority_color(component)
            if body_color is None:
                unique = 0
                continue
            body_cells = tuple((r, c) for r, c in component.cells if grid[r, c] == body_color)
            markers = len(component.cells) - len(body_cells)
            anchor_cells += markers
            copied_cells += markers * len(body_cells)
            max_body = max(max_body, len(body_cells))
            max_markers = max(max_markers, markers)
        rows.append(
            StructureRow(
                split=split,
                index=index,
                shape=f"{grid.shape[0]}x{grid.shape[1]}",
                components=len(components),
                anchor_cells=anchor_cells,
                copied_cells=copied_cells,
                added_cells=added,
                unique_majority=unique,
                max_body_cells=max_body,
                max_markers=max_markers,
            )
        )
    return rows


def transform_contract(name: str, transform: Callable[[Grid], Grid]) -> ph.InvarianceContract:
    return ph.InvarianceContract(
        name=name,
        input_transform=transform,
        expected_output_transform=transform,
        description=f"Apply {name} to input and output.",
    )


def shift_grid(grid: Grid, dr: int, dc: int) -> Grid:
    out = np.zeros_like(grid)
    h, w = grid.shape
    src_r0 = max(0, -dr)
    src_r1 = min(h, h - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(w, w - dc)
    dst_r0 = src_r0 + dr
    dst_r1 = src_r1 + dr
    dst_c0 = src_c0 + dc
    dst_c1 = src_c1 + dc
    out[dst_r0:dst_r1, dst_c0:dst_c1] = grid[src_r0:src_r1, src_c0:src_c1]
    return out


def shift_contract(dr: int, dc: int, name: str) -> ph.InvarianceContract:
    def skip_if(input_grid: Grid, output_grid: Grid) -> str | None:
        for label, grid in (("input", input_grid), ("output", output_grid)):
            if dr > 0 and np.any(grid[-dr:, :] != 0):
                return f"{label}_would_clip_bottom"
            if dr < 0 and np.any(grid[: -dr, :] != 0):
                return f"{label}_would_clip_top"
            if dc > 0 and np.any(grid[:, -dc:] != 0):
                return f"{label}_would_clip_right"
            if dc < 0 and np.any(grid[:, : -dc] != 0):
                return f"{label}_would_clip_left"
        return None

    return ph.InvarianceContract(
        name=name,
        input_transform=lambda grid: shift_grid(grid, dr, dc),
        expected_output_transform=lambda grid: shift_grid(grid, dr, dc),
        description=f"Translate in-frame by dr={dr}, dc={dc} when no clipping occurs.",
        skip_if=skip_if,
    )


def find_empty_slot(grid: Grid, size: int = 2, margin: int = 1) -> tuple[int, int] | None:
    h, w = grid.shape
    for r in range(0, h - size + 1):
        for c in range(0, w - size + 1):
            r0 = max(0, r - margin)
            r1 = min(h, r + size + margin)
            c0 = max(0, c - margin)
            c1 = min(w, c + size + margin)
            if np.any(grid[r0:r1, c0:c1] != 0):
                continue
            return r, c
    return None


def paste_square(grid: Grid, top: int, left: int, color: int, size: int = 2) -> Grid:
    out = grid.copy()
    out[top : top + size, left : left + size] = color
    return out


def inert_distractor_contract(name: str, color: int, size: int) -> ph.InvarianceContract:
    def shared_slot(input_grid: Grid, output_grid: Grid) -> tuple[int, int] | None:
        input_slot = find_empty_slot(input_grid, size=size)
        output_slot = find_empty_slot(output_grid, size=size)
        if input_slot == output_slot:
            return input_slot
        return None

    def skip_if(input_grid: Grid, output_grid: Grid) -> str | None:
        return None if shared_slot(input_grid, output_grid) is not None else "no_shared_empty_slot"

    def transform(grid: Grid) -> Grid:
        slot = find_empty_slot(grid, size=size)
        if slot is None:
            raise ValueError("no empty slot")
        return paste_square(grid, slot[0], slot[1], color=color, size=size)

    return ph.InvarianceContract(
        name=name,
        input_transform=transform,
        expected_output_transform=transform,
        description="Insert a disconnected single-color distractor object.",
        skip_if=skip_if,
    )


def pseudo_hidden_contracts() -> list[ph.InvarianceContract]:
    mapping_a = {0: 0, 1: 7, 2: 5, 3: 9, 4: 1, 5: 8, 6: 2, 7: 4, 8: 6, 9: 3}
    mapping_b = {0: 0, 1: 4, 2: 8, 3: 1, 4: 7, 5: 9, 6: 3, 7: 2, 8: 5, 9: 6}
    return [
        ph.color_permutation_contract(mapping_a, name="color_permutation_a"),
        ph.color_permutation_contract(mapping_b, name="color_permutation_b"),
        transform_contract("transpose", lambda grid: grid.T.copy()),
        transform_contract("mirror_lr", lambda grid: np.fliplr(grid).copy()),
        transform_contract("mirror_ud", lambda grid: np.flipud(grid).copy()),
        transform_contract("rotate_180", lambda grid: np.rot90(grid, 2).copy()),
        ph.zero_border_padding_contract(1, 0, 1, 0, expected_output="pad", name="pad_top_left"),
        ph.zero_border_padding_contract(0, 1, 0, 1, expected_output="pad", name="pad_bottom_right"),
        shift_contract(1, 1, "translate_down_right"),
        shift_contract(-1, -1, "translate_up_left"),
        inert_distractor_contract("inert_2x2_distractor", color=9, size=2),
        inert_distractor_contract("inert_singleton_distractor", color=5, size=1),
    ]


def result_dict(result: ph.CheckResult, phase: str) -> dict[str, object]:
    row = asdict(result)
    row["phase"] = phase
    return row


def write_csv(path: Path, rows: Sequence[object | dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    dict_rows = [asdict(row) if not isinstance(row, dict) else row for row in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict_rows[0]))
        writer.writeheader()
        writer.writerows(dict_rows)


def onnx_anchor_summary(anchor_path: Path, examples: list[tuple[str, int, dict]]) -> dict[str, object]:
    if not anchor_path.is_file():
        return {"available": False, "reason": "missing"}
    try:
        import onnx
        import onnxruntime as ort
        from tools.neurogolf_local import encode_grid, score_onnx
    except Exception as exc:  # pragma: no cover - environment dependent.
        return {"available": False, "reason": f"import_error:{type(exc).__name__}:{exc}"}

    flat_examples = [example for _, _, example in examples]
    score = score_onnx(anchor_path, flat_examples, 285, n_runs=0)

    session = ort.InferenceSession(str(anchor_path), providers=["CPUExecutionProvider"])
    target_exact = 0
    semantic_exact = 0
    for _, _, example in examples:
        inp = arr(example["input"])
        target = encode_grid(example["output"]) > 0.0
        semantic = encode_grid(solver_majority_reflect(inp).tolist()) > 0.0
        pred = session.run(["output"], {"input": encode_grid(example["input"])})[0] > 0.0
        target_exact += int(np.array_equal(pred, target))
        semantic_exact += int(np.array_equal(pred, semantic))

    model = onnx.load(str(anchor_path))
    return {
        "available": True,
        "path": str(anchor_path.relative_to(ROOT)),
        "target_exact": target_exact,
        "semantic_exact": semantic_exact,
        "total": len(examples),
        "cost": score.cost,
        "score": score.score,
        "memory": score.memory,
        "params": score.params,
        "nodes": len(model.graph.node),
        "file_size": anchor_path.stat().st_size,
        "error": score.error,
    }


def summarize_structure(rows: Sequence[StructureRow]) -> dict[str, object]:
    shape_counts = Counter(row.shape for row in rows)
    return {
        "examples": len(rows),
        "shape_counts": dict(sorted(shape_counts.items())),
        "components": int(sum(row.components for row in rows)),
        "anchor_cells": int(sum(row.anchor_cells for row in rows)),
        "copied_cells": int(sum(row.copied_cells for row in rows)),
        "added_cells": int(sum(row.added_cells for row in rows)),
        "unique_majority_examples": int(sum(row.unique_majority for row in rows)),
        "max_body_cells": int(max(row.max_body_cells for row in rows)),
        "max_markers": int(max(row.max_markers for row in rows)),
    }


def score_from_cost(cost: float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, cost)))


def write_markdown(
    path: Path,
    *,
    task: dict,
    hypotheses: Sequence[HypothesisRow],
    pseudo_summary: dict[str, int],
    loo_summary: dict[str, int],
    structure_summary: dict[str, object],
    anchor: dict[str, object],
) -> None:
    best = next(row for row in hypotheses if row.solver == "semantic_majority_reflect" and row.split == "all")
    current_cost = int(anchor.get("cost") or 395468)
    current_score = float(anchor.get("score") or score_from_cost(current_cost))
    competitor_score = 15.131
    competitor_cost_est = int(round(math.exp(25.0 - competitor_score)))
    semantic_compile_target = 173596
    semantic_compile_gain = score_from_cost(semantic_compile_target) - current_score
    competitor_gain = competitor_score - current_score

    lines = [
        "# Task285 Semantic V2",
        "",
        "## Data closure",
        "",
        f"- Examples: {sum(len(task.get(split, [])) for split in ('train', 'test', 'arc-gen'))} total; "
        f"splits {{'train': {len(task.get('train', []))}, 'test': {len(task.get('test', []))}, "
        f"'arc-gen': {len(task.get('arc-gen', []))}}}.",
        f"- Shapes: {structure_summary['shape_counts']}.",
        f"- All examples have unique majority-color bodies per foreground object: "
        f"{structure_summary['unique_majority_examples']}/{structure_summary['examples']}.",
        f"- Foreground objects: {structure_summary['components']}; anchor cells: "
        f"{structure_summary['anchor_cells']}; reflected body-cell placements before overlap: "
        f"{structure_summary['copied_cells']}; visible added cells after zero-only painting: "
        f"{structure_summary['added_cells']}.",
        f"- Semantic rule exact: {best.exact}/{best.total}, pixel errors {best.pixel_errors}.",
        "",
        "## Rule",
        "",
        "- Use 8-connected non-zero foreground components.",
        "- In each component, the unique majority color is the source body; minority cells are anchors.",
        "- Each anchor direction is determined by its position relative to the body bounding box.",
        "- Reflect the body over the corresponding side/corner of the body bbox and paint zero target cells with the anchor color.",
        "- Existing non-zero cells are preserved, so overlaps do not erase anchors or source bodies.",
        "",
        "## Weaker hypotheses",
        "",
    ]
    for row in hypotheses:
        if row.split == "all":
            lines.append(
                f"- {row.solver}: exact {row.exact}/{row.total}, pixel errors {row.pixel_errors}."
            )

    lines.extend(
        [
            "",
            "## Pseudo-hidden",
            "",
            f"- Contract summary: {pseudo_summary}.",
            f"- Leave-one-out visible summary: {loo_summary}.",
            "- Contracts used: original rows, two arbitrary color bijections fixing 0, transpose, left-right/up-down/180-degree symmetry, zero-border padding, in-frame translations when unclipped, and disconnected inert distractors.",
            "- These contracts directly target hidden-plausible risks: color permutation, translations/borders, reflected direction changes under symmetry, irrelevant distractors, and avoiding a rule learned from any single visible row.",
            "",
            "## Existing ONNX",
            "",
        ]
    )
    if anchor.get("available"):
        lines.extend(
            [
                f"- Anchor path: `{anchor['path']}`.",
                f"- Anchor target exact: {anchor['target_exact']}/{anchor['total']}; semantic exact: {anchor['semantic_exact']}/{anchor['total']}.",
                f"- Anchor score/cost: {float(anchor['score']):.6f}, cost {anchor['cost']}, memory {anchor['memory']}, params {anchor['params']}, nodes {anchor['nodes']}, file size {anchor['file_size']}.",
            ]
        )
    else:
        lines.append(f"- Anchor unavailable: {anchor.get('reason')}.")

    lines.extend(
        [
            "",
            "## ONNX decision",
            "",
            "- No `task285_v2` ONNX was built in this pass.",
            "- Closure is strong in Python, but an honest compile of this object-level rule needs connected-component labeling, per-component majority color, body bbox recovery, and marker-directed scatter/reflection. That is likely larger than the current 279-node, 395468-cost local-neighborhood anchor unless a new compact primitive is found.",
            f"- A merely moderate semantic compile at cost {semantic_compile_target} would score about {score_from_cost(semantic_compile_target):.3f}, gain {semantic_compile_gain:+.3f} over the current anchor.",
            f"- The reported public top score {competitor_score:.3f} implies cost about {competitor_cost_est}, gain {competitor_gain:+.3f}; matching that requires a substantially more compact encoding than the straightforward semantic graph.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--anchor-onnx", type=Path, default=ANCHOR_ONNX)
    parser.add_argument("--reports", type=Path, default=REPORT_DIR)
    parser.add_argument("--skip-onnx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_task(args.data)
    examples = all_examples(task)

    hypothesis_rows: list[HypothesisRow] = []
    for name, solver in SOLVERS.items():
        hypothesis_rows.extend(evaluate_solver(task, name, solver))

    struct_rows = structure_rows(task)

    contracts = pseudo_hidden_contracts()
    ph_examples = [example for _, _, example in examples]
    pseudo_results = ph.run_contracts(
        solver_majority_reflect,
        ph_examples,
        contracts,
        include_original=True,
        rule_label="semantic_majority_reflect",
    )

    visible = visible_examples(task)

    def build_rule(
        train_subset: list[ph.ArcExample],
        heldout: ph.ArcExample,
        heldout_index: int,
    ) -> ph.RuleSpec:
        for train_example in train_subset:
            pred = solver_majority_reflect(train_example.input_grid)
            if not np.array_equal(pred, train_example.output_grid):
                raise ValueError("semantic rule does not fit train subset")
        return ph.RuleSpec(
            solver_majority_reflect,
            contracts=contracts,
            label=f"visible_loo_{heldout_index}",
        )

    loo_results = ph.leave_one_out(
        build_rule,
        visible,
        include_original=True,
    )

    anchor = (
        {"available": False, "reason": "skipped"}
        if args.skip_onnx
        else onnx_anchor_summary(args.anchor_onnx, examples)
    )

    reports = args.reports
    write_csv(reports / "task285_v2_hypotheses.csv", hypothesis_rows)
    write_csv(reports / "task285_v2_structure.csv", struct_rows)
    write_csv(
        reports / "task285_v2_pseudo_hidden.csv",
        [result_dict(result, "pseudo_hidden") for result in pseudo_results],
    )
    write_csv(
        reports / "task285_v2_leave_one_out.csv",
        [result_dict(result, "leave_one_out") for result in loo_results],
    )

    pseudo_summary = ph.summarize_results(pseudo_results)
    loo_summary = ph.summarize_results(loo_results)
    structure_summary = summarize_structure(struct_rows)
    summary = {
        "task": 285,
        "data": str(args.data.relative_to(ROOT)),
        "hypotheses": [asdict(row) for row in hypothesis_rows if row.split == "all"],
        "pseudo_hidden_summary": pseudo_summary,
        "leave_one_out_summary": loo_summary,
        "structure_summary": structure_summary,
        "anchor": anchor,
        "onnx_built": False,
        "onnx_decision": "not_justified_without_compact_component_majority_compiler",
    }
    (reports / "task285_v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(
        reports / "task285_semantic_v2.md",
        task=task,
        hypotheses=hypothesis_rows,
        pseudo_summary=pseudo_summary,
        loo_summary=loo_summary,
        structure_summary=structure_summary,
        anchor=anchor,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
