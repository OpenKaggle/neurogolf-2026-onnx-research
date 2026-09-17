from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib
import sys
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from neurogolf_local import all_examples, encode_grid, score_onnx  # noqa: E402


TASK_ID = 25
COMP_DIR = ROOT / "data" / "neurogolf-2026" / "raw"


@dataclasses.dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    input: list[list[int]]
    output: list[list[int]]
    note: str


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    path: pathlib.Path
    source_class: str


def _as_grid(arr: np.ndarray) -> list[list[int]]:
    return arr.astype(int).tolist()


def task025_reference(grid: list[list[int]]) -> list[list[int]]:
    """Line/stray relocation reference inferred from task025 visible examples."""
    arr = np.array(grid, dtype=np.int64)
    h, w = arr.shape
    out = np.zeros_like(arr)

    for color in range(1, 10):
        row_lines = [r for r in range(h) if np.all(arr[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(arr[:, c] == color)]

        if len(row_lines) == 1 and not col_lines:
            line_r = row_lines[0]
            out[line_r, :] = color
            positions = np.argwhere(arr == color)
            for r, c in positions:
                if r == line_r:
                    continue
                target_r = line_r - 1 if r < line_r else line_r + 1
                if 0 <= target_r < h:
                    out[target_r, c] = color
        elif len(col_lines) == 1 and not row_lines:
            line_c = col_lines[0]
            out[:, line_c] = color
            positions = np.argwhere(arr == color)
            for r, c in positions:
                if c == line_c:
                    continue
                target_c = line_c - 1 if c < line_c else line_c + 1
                if 0 <= target_c < w:
                    out[r, target_c] = color
        elif row_lines or col_lines:
            # Ambiguous colors are outside the intended generated suite. Keep
            # only proven complete lines so the reference remains conservative.
            for r in row_lines:
                out[r, :] = color
            for c in col_lines:
                out[:, c] = color

    return _as_grid(out)


def _make_case(
    case_id: str,
    category: str,
    h: int,
    w: int,
    orientation: str,
    line_specs: list[tuple[int, int, list[tuple[int, int]]]],
    distractors: list[tuple[int, int, int]],
    note: str,
) -> Case:
    grid = np.zeros((h, w), dtype=np.int64)
    for color, line_pos, _strays in line_specs:
        if orientation == "horizontal":
            grid[line_pos, :] = color
        elif orientation == "vertical":
            grid[:, line_pos] = color
        else:
            raise ValueError(f"bad orientation: {orientation}")

    for color, _line_pos, strays in line_specs:
        for r, c in strays:
            if 0 <= r < h and 0 <= c < w:
                grid[r, c] = color

    for r, c, color in distractors:
        if 0 <= r < h and 0 <= c < w and grid[r, c] == 0:
            grid[r, c] = color

    in_grid = _as_grid(grid)
    return Case(case_id, category, in_grid, task025_reference(in_grid), note)


def _safe_positions(length: int, count: int) -> list[int]:
    if count == 1:
        return [length // 2]
    if count == 2:
        return [2, length - 3]
    if count == 3:
        return [2, length // 2, length - 3]
    return [2, max(5, length // 3), min(length - 6, (2 * length) // 3), length - 3]


def generated_cases() -> list[Case]:
    cases: list[Case] = []
    sizes = [(12, 12), (14, 14), (15, 22), (22, 15), (29, 14), (14, 29), (30, 30)]
    colors = [1, 2, 3, 4]

    for orientation in ("horizontal", "vertical"):
        for size_idx, (h, w) in enumerate(sizes):
            axis_len = h if orientation == "horizontal" else w
            ortho_len = w if orientation == "horizontal" else h
            for count in (1, 2, 3, 4):
                line_specs: list[tuple[int, int, list[tuple[int, int]]]] = []
                for color, line_pos in zip(colors[:count], _safe_positions(axis_len, count)):
                    left_or_above = max(0, line_pos - 2)
                    right_or_below = min(axis_len - 1, line_pos + 2)
                    adjacent_neg = max(0, line_pos - 1)
                    adjacent_pos = min(axis_len - 1, line_pos + 1)
                    coords = [
                        ((color * 3 + size_idx) % ortho_len),
                        ((color * 5 + size_idx + 1) % ortho_len),
                        ((color * 7 + size_idx + 2) % ortho_len),
                    ]
                    if orientation == "horizontal":
                        strays = [(left_or_above, coords[0]), (right_or_below, coords[1])]
                        if count == 1:
                            strays.extend([(adjacent_neg, coords[2]), (adjacent_pos, (coords[2] + 3) % ortho_len)])
                    else:
                        strays = [(coords[0], left_or_above), (coords[1], right_or_below)]
                        if count == 1:
                            strays.extend([(coords[2], adjacent_neg), ((coords[2] + 3) % ortho_len, adjacent_pos)])
                    line_specs.append((color, line_pos, strays))

                distractors = [
                    ((3 + size_idx) % h, (5 + 2 * size_idx) % w, 8),
                    ((h - 4 - size_idx) % h, (w - 5 - 2 * size_idx) % w, 9),
                ]
                cases.append(
                    _make_case(
                        f"{orientation[:1]}_{size_idx:02d}_{count}c",
                        f"{orientation}_color_count_{count}",
                        h,
                        w,
                        orientation,
                        line_specs,
                        distractors,
                        f"{orientation}, {count} active line colors, size {h}x{w}",
                    )
                )

    border_specs = [
        ("horizontal", 0, 5, [(3, 3), (9, 8)], "top_line"),
        ("horizontal", 13, 6, [(4, 5), (10, 11)], "bottom_line"),
        ("vertical", 0, 7, [(2, 4), (11, 9)], "left_line"),
        ("vertical", 13, 8, [(5, 3), (12, 10)], "right_line"),
    ]
    for idx, (orientation, pos, color, stray_orthos, label) in enumerate(border_specs):
        h = w = 14
        if orientation == "horizontal":
            strays = [(r, c) for r, c in stray_orthos]
        else:
            strays = [(r, c) for c, r in stray_orthos]
        cases.append(
            _make_case(
                f"border_{idx:02d}_{label}",
                "border_adjacency",
                h,
                w,
                orientation,
                [(color, pos, strays)],
                [(1, 1, 9), (12, 12, 1)],
                f"{label} with legal inside-side relocation",
            )
        )

    for idx, orientation in enumerate(("horizontal", "vertical")):
        h, w = 18, 18
        pos = 8
        cases.append(
            _make_case(
                f"nomove_{idx:02d}_{orientation}",
                "no_stray_and_distractor_controls",
                h,
                w,
                orientation,
                [(2 + idx, pos, [])],
                [(2, 3, 8), (15, 14, 9), (5, 11, 7)],
                "active line has no strays; all non-line singletons disappear",
            )
        )

    return cases


def _session(path: pathlib.Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _check_model(path: pathlib.Path) -> tuple[bool, str]:
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        onnx.checker.check_model(inferred, full_check=True)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc).replace("\n", " | ")


def _run_cases(session: ort.InferenceSession, cases: Iterable[Case]) -> tuple[int, int, list[dict]]:
    passed = 0
    failures: list[dict] = []
    total = 0
    for case in cases:
        total += 1
        expected = encode_grid(case.output) > 0.0
        try:
            actual = session.run(["output"], {"input": encode_grid(case.input)})[0] > 0.0
            ok = bool(np.array_equal(actual, expected))
        except Exception as exc:  # noqa: BLE001
            ok = False
            actual = None
            error = str(exc).replace("\n", " | ")
        else:
            error = ""
        if ok:
            passed += 1
        else:
            mismatch_cells = None
            if actual is not None:
                mismatch_cells = int(np.count_nonzero(actual != expected))
            failures.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "note": case.note,
                    "mismatch_cells": mismatch_cells,
                    "error": error,
                }
            )
    return passed, total, failures


def _score_dict(result) -> dict:
    return {
        "cost": result.cost,
        "score": result.score,
        "memory": result.memory,
        "params": result.params,
        "local_pass": result.local_pass,
        "local_total": result.local_total,
        "nodes": result.nodes,
        "file_size": result.file_size,
        "error": result.error,
    }


def _stageable(row: dict, anchor_score: float, anchor_cost: int | None) -> tuple[bool, str]:
    if not row["exists"]:
        return False, "missing model"
    if not row["checker_ok"]:
        return False, "checker/strict-shape failure"
    if row["visible_pass"] != row["visible_total"]:
        return False, "visible full-example miss"
    if row["pseudo_pass"] != row["pseudo_total"]:
        return False, "pseudo-hidden contract miss"
    score = row["score"]["score"]
    cost = row["score"]["cost"]
    if cost is None:
        return False, "unmeasurable cost"
    if anchor_cost is not None and cost >= anchor_cost:
        return False, "not cheaper than anchor"
    if score - anchor_score < 0.1:
        return False, "score gain below task025 probe threshold"
    if row["source_class"] == "konbu17_v36_public_family":
        return False, "public-family source still needs a release-lane single-task probe"
    return True, "passes bounded gates and is cheaper"


def _format_int(value) -> str:
    return "" if value is None else str(value)


def write_markdown(path: pathlib.Path, data: dict) -> None:
    models = data["models"]
    lines = [
        "# task025 Contract Audit - 2026-06-07",
        "",
        "Scope: bounded pseudo-hidden harness for the line/stray relocation rule. No submission bundle was built and no anchor directories, zips, or ledgers were edited.",
        "",
        "## Contract",
        "",
        "- Detect one complete same-color row or column per active nonzero color.",
        "- Keep complete line cells.",
        "- Delete unrelated singleton distractors and colors without a complete line.",
        "- Move each off-line same-color cell to the row/column immediately adjacent to its line on the same side, preserving the other coordinate.",
        "- Generated coverage varies orientation, line position, active color count, stray side, adjacent strays, border-adjacent lines, and no-stray/distractor controls.",
        "",
        "## Counts",
        "",
        f"- Visible corpus: {data['visible_total']} examples.",
        f"- Generated pseudo-hidden suite: {data['pseudo_total']} examples.",
        "",
        "| category | count |",
        "|---|---:|",
    ]
    for category, count in sorted(data["category_counts"].items()):
        lines.append(f"| {category} | {count} |")

    lines.extend(
        [
            "",
            "## Model Results",
            "",
            "| model | visible | pseudo-hidden | cost | score | delta vs anchor | checker | stageable | reason |",
            "|---|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in models:
        score = row["score"]
        delta = row["score_delta_vs_anchor"]
        delta_s = "" if delta is None else f"{delta:+.6f}"
        score_s = "" if score["score"] is None else f"{score['score']:.6f}"
        lines.append(
            "| {name} | {vp}/{vt} | {pp}/{pt} | {cost} | {score} | {delta} | {checker} | {stageable} | {reason} |".format(
                name=row["name"],
                vp=row["visible_pass"],
                vt=row["visible_total"],
                pp=row["pseudo_pass"],
                pt=row["pseudo_total"],
                cost=_format_int(score["cost"]),
                score=score_s,
                delta=delta_s,
                checker="ok" if row["checker_ok"] else "fail",
                stageable="yes" if row["stageable"] else "no",
                reason=row["stageable_reason"],
            )
        )

    lines.extend(["", "## First Pseudo-Hidden Failures", ""])
    for row in models:
        failures = row["pseudo_failures"][:8]
        if not failures:
            lines.append(f"- {row['name']}: none.")
            continue
        compact = ", ".join(f"{f['case_id']} ({f['category']}, cells={f['mismatch_cells']})" for f in failures)
        lines.append(f"- {row['name']}: {compact}.")

    stageable = [row for row in models if row["stageable"]]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Stageable candidates found: {len(stageable)}.",
            f"- Current best non-anchor pseudo-hidden performer: {data['best_non_anchor']['name']} ({data['best_non_anchor']['pseudo_pass']}/{data['best_non_anchor']['pseudo_total']}, score delta {data['best_non_anchor']['score_delta_vs_anchor']:+.6f}).",
            "- Next exact engineering move: replace the current two-reduction relocation head with a smaller typed line/stray compiler that keeps the anchor semantics, then re-run this harness. The target is a semantic ONNX below the konbu public-family cost frontier of 90K while preserving full visible and generated-contract pass counts.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=str(ROOT / "reports" / "task025_contract_audit_20260607.json"))
    parser.add_argument("--md-out", default=str(ROOT / "reports" / "task025_contract_audit_20260607.md"))
    args = parser.parse_args()

    specs = [
        ModelSpec(
            "anchor_v4_plus13_task366",
            ROOT / "submissions" / "candidate_v4_plus13_task366_onnx" / "task025.onnx",
            "lb_verified_anchor",
        ),
        ModelSpec(
            "konbu17_v36_public_family",
            ROOT / "data" / "konbu17_v36" / "task025.onnx",
            "konbu17_v36_public_family",
        ),
        ModelSpec(
            "probe_best6090_task025_semantic",
            ROOT / "submissions" / "probe_best6090_task025_semantic_onnx" / "task025.onnx",
            "older_probe",
        ),
        ModelSpec(
            "priority_semantic_v2",
            ROOT / "submissions" / "handbuilds" / "task025_priority.onnx",
            "local_semantic_prototype",
        ),
    ]

    visible = all_examples(TASK_ID, COMP_DIR)
    pseudo = generated_cases()
    category_counts = collections.Counter(case.category for case in pseudo)

    models: list[dict] = []
    for spec in specs:
        row = {
            "name": spec.name,
            "path": str(spec.path.relative_to(ROOT)),
            "source_class": spec.source_class,
            "exists": spec.path.is_file(),
        }
        if not spec.path.is_file():
            row.update(
                checker_ok=False,
                checker_error="missing",
                visible_pass=0,
                visible_total=len(visible),
                pseudo_pass=0,
                pseudo_total=len(pseudo),
                pseudo_failures=[],
                score=_score_dict(score_onnx(spec.path, visible, TASK_ID, n_runs=0)),
            )
            models.append(row)
            continue

        checker_ok, checker_error = _check_model(spec.path)
        session = _session(spec.path)
        visible_pass, visible_total, visible_failures = _run_cases(
            session,
            [Case(f"visible_{idx:03d}", "visible", ex["input"], ex["output"], "visible train/test/arc-gen") for idx, ex in enumerate(visible)],
        )
        pseudo_pass, pseudo_total, pseudo_failures = _run_cases(session, pseudo)
        row.update(
            checker_ok=checker_ok,
            checker_error=checker_error,
            visible_pass=visible_pass,
            visible_total=visible_total,
            visible_failures=visible_failures[:10],
            pseudo_pass=pseudo_pass,
            pseudo_total=pseudo_total,
            pseudo_failures=pseudo_failures,
            score=_score_dict(score_onnx(spec.path, visible, TASK_ID, n_runs=0)),
        )
        models.append(row)

    anchor = models[0]
    anchor_score = float(anchor["score"]["score"] or 0.0)
    anchor_cost = anchor["score"]["cost"]
    for row in models:
        score = row["score"]["score"]
        cost = row["score"]["cost"]
        row["score_delta_vs_anchor"] = None if score is None else float(score - anchor_score)
        row["cost_delta_vs_anchor"] = None if cost is None or anchor_cost is None else int(cost - anchor_cost)
        stageable, reason = _stageable(row, anchor_score, anchor_cost)
        row["stageable"] = stageable
        row["stageable_reason"] = reason

    non_anchor = models[1:]
    best_non_anchor = max(non_anchor, key=lambda item: (item["pseudo_pass"], item["score"]["score"] or 0.0))

    data = {
        "task_id": TASK_ID,
        "visible_total": len(visible),
        "pseudo_total": len(pseudo),
        "category_counts": dict(category_counts),
        "models": models,
        "best_non_anchor": {
            "name": best_non_anchor["name"],
            "pseudo_pass": best_non_anchor["pseudo_pass"],
            "pseudo_total": best_non_anchor["pseudo_total"],
            "score_delta_vs_anchor": best_non_anchor["score_delta_vs_anchor"],
        },
    }

    json_path = pathlib.Path(args.json_out)
    md_path = pathlib.Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, data)

    print(f"wrote {md_path.relative_to(ROOT)}")
    print(f"wrote {json_path.relative_to(ROOT)}")
    for row in models:
        print(
            "{name}: visible={vp}/{vt} pseudo={pp}/{pt} cost={cost} score={score:.6f} stageable={stageable}".format(
                name=row["name"],
                vp=row["visible_pass"],
                vt=row["visible_total"],
                pp=row["pseudo_pass"],
                pt=row["pseudo_total"],
                cost=row["score"]["cost"],
                score=row["score"]["score"],
                stageable=row["stageable"],
            )
        )


if __name__ == "__main__":
    main()
