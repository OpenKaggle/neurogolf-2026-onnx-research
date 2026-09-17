#!/usr/bin/env python3
"""Build and audit task366 compact v3.

The task366 semantic v2 closure proves two facts that are stronger than the
original graph architecture:

* source objects are filled rectangles with max side length 7, so six 3x3
  label-propagation passes are sufficient;
* the source panel is the denser non-background panel, so the graph does not
  need label histograms to decide which panel supplies rectangles.

This builder applies both rewrites to the current anchor graph, prunes dead
nodes, then runs the task-specific visible, semantic, pseudo-hidden, and scoring
checks used by the earlier task366 lane.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.neurogolf_local import all_examples as ng_all_examples
from tools.neurogolf_local import encode_grid, score_onnx
from tools.prototype_task366_v2 import (
    all_examples,
    arr,
    load_task,
    run_pseudo_hidden,
    solver_bounded_rectangles,
)


DEFAULT_ANCHOR = ROOT / "submissions" / "candidate_v4_plus12_task285_onnx" / "task366.onnx"
DEFAULT_OUT = ROOT / "submissions" / "handbuilds" / "task366_compact_v3.onnx"
DEFAULT_REPORT = ROOT / "reports" / "task366_compact_v3.md"
DEFAULT_VERIFY_JSON = ROOT / "reports" / "task366_compact_v3_verify.json"
DEFAULT_EQUIV_CSV = ROOT / "reports" / "task366_compact_v3_equivalence.csv"
BOUNDED_PASSES = 6
ANCHOR_COST = 830_720
V2_COST = 794_720


def _prune_dead_nodes(model: onnx.ModelProto, graph_name: str) -> onnx.ModelProto:
    needed = {output.name for output in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            kept.append(node)
            needed.update(input_name for input_name in node.input if input_name)
    kept.reverse()

    used_initializers = {input_name for node in kept for input_name in node.input if input_name}
    graph_inputs = {input_info.name for input_info in model.graph.input}
    graph = helper.make_graph(
        kept,
        graph_name,
        list(model.graph.input),
        list(model.graph.output),
        initializer=[
            initializer
            for initializer in model.graph.initializer
            if initializer.name in used_initializers and initializer.name not in graph_inputs
        ],
    )
    rebuilt = helper.make_model(graph, opset_imports=list(model.opset_import))
    rebuilt.ir_version = model.ir_version
    return onnx.shape_inference.infer_shapes(rebuilt, strict_mode=True)


def build_model(anchor_path: Path = DEFAULT_ANCHOR) -> onnx.ModelProto:
    model = onnx.load(str(anchor_path))

    for node in model.graph.node:
        if node.name == "lpa_lbl":
            node.input[0] = f"lpa_l{BOUNDED_PASSES}"
        elif node.name == "lpb_lbl":
            node.input[0] = f"lpb_l{BOUNDED_PASSES}"

    rewritten_nodes = []
    replaced_selector = False
    for node in model.graph.node:
        if node.name == "a_is_tmpl_b":
            rewritten_nodes.extend(
                [
                    helper.make_node(
                        "ReduceSum",
                        ["nb_a_32", "axes_all"],
                        ["nb_a_total"],
                        name="nb_a_total",
                        keepdims=1,
                    ),
                    helper.make_node(
                        "ReduceSum",
                        ["nb_b_32", "axes_all"],
                        ["nb_b_total"],
                        name="nb_b_total",
                        keepdims=1,
                    ),
                    helper.make_node(
                        "Greater",
                        ["nb_a_total", "nb_b_total"],
                        ["a_is_tmpl_b"],
                        name="a_is_tmpl_b",
                    ),
                ]
            )
            replaced_selector = True
        else:
            rewritten_nodes.append(node)

    if not replaced_selector:
        raise RuntimeError("anchor graph did not contain a_is_tmpl_b")

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    compact = _prune_dead_nodes(model, "task366_compact_v3")
    onnx.checker.check_model(compact, full_check=True)
    return compact


def shape_s(grid: np.ndarray) -> str:
    return f"{grid.shape[0]}x{grid.shape[1]}"


def onehot_equal(output: np.ndarray, grid: list[list[int]]) -> bool:
    return bool(np.array_equal(output > 0.0, encode_grid(grid) > 0.0))


def run_visible_equivalence(model_path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    task = load_task()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])

    rows: list[dict[str, object]] = []
    summary = {
        "total": 0,
        "target_exact": 0,
        "semantic_exact": 0,
        "encodable_total": 0,
        "encodable_target_exact": 0,
        "encodable_semantic_exact": 0,
        "skipped_shape": 0,
    }

    for split, index, example in all_examples(task):
        grid = arr(example["input"])
        target = arr(example["output"])
        semantic = arr(solver_bounded_rectangles(grid, max_dim=7))
        semantic_exact = bool(semantic.shape == target.shape and np.array_equal(semantic, target))
        summary["total"] += 1
        summary["semantic_exact"] += int(semantic_exact)

        encodable = grid.shape[0] <= 30 and grid.shape[1] <= 30 and target.shape[0] <= 30 and target.shape[1] <= 30
        row: dict[str, object] = {
            "split": split,
            "index": index,
            "encodable": int(encodable),
            "target_exact": "",
            "semantic_exact": int(semantic_exact),
            "onnx_semantic_exact": "",
            "input_shape": shape_s(grid),
            "target_shape": shape_s(target),
            "semantic_shape": shape_s(semantic),
        }

        if not encodable:
            summary["skipped_shape"] += 1
        else:
            output = session.run(["output"], {"input": encode_grid(example["input"])})[0]
            target_exact = onehot_equal(output, example["output"])
            onnx_semantic_exact = onehot_equal(output, semantic.tolist())
            summary["encodable_total"] += 1
            summary["encodable_target_exact"] += int(target_exact)
            summary["encodable_semantic_exact"] += int(onnx_semantic_exact)
            summary["target_exact"] += int(target_exact)
            row["target_exact"] = int(target_exact)
            row["onnx_semantic_exact"] = int(onnx_semantic_exact)

        rows.append(row)

    return rows, summary


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def score_single_task(model_path: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="task366_v3_score_") as tmp:
        tmp_dir = Path(tmp)
        shutil.copy2(model_path, tmp_dir / "task366.onnx")
        result = score_onnx(
            tmp_dir / "task366.onnx",
            ng_all_examples(366, ROOT / "data" / "neurogolf-2026" / "raw"),
            366,
            n_runs=3,
        )
    return {
        "task": result.task_id,
        "score": result.score,
        "cost": result.cost,
        "memory": result.memory,
        "params": result.params,
        "local_pass": result.local_pass,
        "local_total": result.local_total,
        "nodes": result.nodes,
        "file_size": result.file_size,
        "error": result.error,
    }


def score_from_cost(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, cost)))


def write_report(
    report_path: Path,
    out_path: Path,
    score: dict[str, object],
    equiv_summary: dict[str, int],
    pseudo_summary: dict[str, int],
    checker_ok: bool,
    shape_inference_ok: bool,
) -> None:
    cost = int(score["cost"] or 0)
    anchor_score = score_from_cost(ANCHOR_COST)
    v2_score = score_from_cost(V2_COST)
    compact_score = float(score["score"])
    stageable = (
        checker_ok
        and shape_inference_ok
        and not score["error"]
        and score["local_pass"] == score["local_total"] == 3
        and equiv_summary["encodable_target_exact"] == equiv_summary["encodable_total"]
        and equiv_summary["encodable_semantic_exact"] == equiv_summary["encodable_total"]
        and equiv_summary["semantic_exact"] == equiv_summary["total"]
        and pseudo_summary["failed"] == 0
        and cost < V2_COST
    )

    lines = [
        "# Task366 Compact V3",
        "",
        "## Build",
        "",
        "- Source anchor: `submissions/candidate_v4_plus12_task285_onnx/task366.onnx`.",
        f"- Output: `{out_path}`.",
        "- Rewrite 1: use six label-propagation passes, the closed max-7 rectangle bound from semantic v2.",
        "- Rewrite 2: replace label-histogram source selection with non-background density comparison (`sum(nb_a) > sum(nb_b)`).",
        "",
        "## Validation",
        "",
        f"- ONNX checker full_check: {'passed' if checker_ok else 'failed'}.",
        f"- Strict shape inference: {'passed' if shape_inference_ok else 'failed'}.",
        f"- Semantic Python visible/arc-gen: {equiv_summary['semantic_exact']}/{equiv_summary['total']}.",
        f"- ONNX encodable target equivalence: {equiv_summary['encodable_target_exact']}/{equiv_summary['encodable_total']}.",
        f"- ONNX encodable semantic equivalence: {equiv_summary['encodable_semantic_exact']}/{equiv_summary['encodable_total']}.",
        f"- Shape-skipped examples for fixed 30x30 ONNX input/output: {equiv_summary['skipped_shape']}.",
        f"- Pseudo-hidden contracts: {pseudo_summary['passed']} passed / {pseudo_summary['failed']} failed / {pseudo_summary['skipped']} skipped / {pseudo_summary['total']} total.",
        "",
        "## Cost",
        "",
        f"- Anchor cost: {ANCHOR_COST}, score {anchor_score:.12f}.",
        f"- V2 cost: {V2_COST}, score {v2_score:.12f}.",
        f"- Compact v3 cost: {cost}, score {compact_score:.12f}.",
        f"- Delta vs anchor: cost {cost - ANCHOR_COST:+d}, score {compact_score - anchor_score:+.12f}.",
        f"- Delta vs v2: cost {cost - V2_COST:+d}, score {compact_score - v2_score:+.12f}.",
        f"- Nodes/file: {score['nodes']} nodes, {score['file_size']} bytes.",
        f"- Memory/params: {score['memory']} memory, {score['params']} params.",
        "",
        "## Decision",
        "",
        f"- Stageable: {'yes' if stageable else 'no'}.",
    ]
    if not stageable:
        lines.append("- Reason: one or more exactness, pseudo-hidden, checker, or cost gates did not pass.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verify-json", type=Path, default=DEFAULT_VERIFY_JSON)
    parser.add_argument("--equiv-csv", type=Path, default=DEFAULT_EQUIV_CSV)
    args = parser.parse_args()

    model = build_model(args.anchor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.out))

    checker_ok = True
    shape_inference_ok = True
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        checker_ok = False
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception:
        shape_inference_ok = False

    equiv_rows, equiv_summary = run_visible_equivalence(args.out)
    write_csv(args.equiv_csv, equiv_rows)

    _, pseudo_summary, failures = run_pseudo_hidden(load_task())
    score = score_single_task(args.out)

    verify_payload = {
        "model": str(args.out),
        "checker_full_check": checker_ok,
        "strict_shape_inference": shape_inference_ok,
        "equivalence": equiv_summary,
        "pseudo_hidden": pseudo_summary,
        "pseudo_hidden_failures": failures,
        "score": score,
    }
    args.verify_json.parent.mkdir(parents=True, exist_ok=True)
    args.verify_json.write_text(json.dumps(verify_payload, indent=2))

    write_report(args.report, args.out, score, equiv_summary, pseudo_summary, checker_ok, shape_inference_ok)

    print(
        f"wrote {args.out} ({args.out.stat().st_size} bytes, {len(model.graph.node)} nodes); "
        f"cost={score['cost']} pass={score['local_pass']}/{score['local_total']}"
    )
    print(f"wrote {args.report}")
    print(f"wrote {args.verify_json}")
    print(f"wrote {args.equiv_csv}")


if __name__ == "__main__":
    main()
