#!/usr/bin/env python3
"""Build the compact task285 v2 ONNX.

This is an exact compiler rewrite of the current task285 anchor, not a new
semantic graph. The anchor already implements the closed marker/body reflection
behavior on all task285 examples. The v2 rewrite keeps that algorithm and
shrinks measured memory by:

1. keeping ARC color-valued grids as UINT8 instead of FLOAT16;
2. using zero-based propagated marker labels, which removes repeated
   subtract-one grids before Gather;
3. precomputing each selected marker's reflection origin on the marker grid and
   gathering that origin directly, avoiding row/column decompositions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR = (
    ROOT / "submissions" / "candidate_v4_plus10_compiler2_onnx" / "task285.onnx"
)
DEFAULT_OUT = ROOT / "submissions" / "handbuilds" / "task285_v2.onnx"
DEFAULT_REPORT = ROOT / "reports" / "task285_onnx_compact_v2_verify.json"
H_W = 30

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import pseudo_hidden as ph  # noqa: E402
from tools.neurogolf_local import all_examples, decode_grid, encode_grid, score_onnx  # noqa: E402
from tools.prototype_task285_v2 import pseudo_hidden_contracts  # noqa: E402


def _add_init(
    model: onnx.ModelProto,
    existing: set[str],
    name: str,
    array: np.ndarray,
) -> None:
    if name in existing:
        return
    model.graph.initializer.append(numpy_helper.from_array(array, name=name))
    existing.add(name)


def _replace_input(node: onnx.NodeProto, old: str, new: str) -> None:
    for index, name in enumerate(node.input):
        if name == old:
            node.input[index] = new


def _set_cast_to(node: onnx.NodeProto, dtype: int) -> None:
    for attr in node.attribute:
        if attr.name == "to":
            attr.i = dtype
            return
    node.attribute.append(helper.make_attribute("to", dtype))


def _make_gather(node: onnx.NodeProto, inputs: list[str], output: str) -> None:
    node.op_type = "Gather"
    del node.input[:]
    node.input.extend(inputs)
    del node.output[:]
    node.output.extend([output])
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute("axis", 0)])


def _make_binary(node: onnx.NodeProto, op_type: str, inputs: list[str], output: str) -> None:
    node.op_type = op_type
    del node.input[:]
    node.input.extend(inputs)
    del node.output[:]
    node.output.extend([output])
    del node.attribute[:]


def _make_reshape(node: onnx.NodeProto, inputs: list[str], output: str) -> None:
    node.op_type = "Reshape"
    del node.input[:]
    node.input.extend(inputs)
    del node.output[:]
    node.output.extend([output])
    del node.attribute[:]


def _insert_compact_initializers(model: onnx.ModelProto) -> None:
    existing = {init.name for init in model.graph.initializer}
    _add_init(model, existing, "ZERO_U8", np.array(0, dtype=np.uint8))
    for color in range(1, 10):
        _add_init(model, existing, f"C{color}_U8", np.array(color, dtype=np.uint8))

    label0 = np.arange(H_W * H_W, dtype=np.float16).reshape(1, 1, H_W, H_W)
    _add_init(model, existing, "LABEL0", label0)

    row2 = (2 * np.arange(H_W, dtype=np.float16)).reshape(1, 1, H_W, 1)
    row2 = np.repeat(row2, H_W, axis=3)
    col2 = (2 * np.arange(H_W, dtype=np.float16)).reshape(1, 1, 1, H_W)
    col2 = np.repeat(col2, H_W, axis=2)
    _add_init(model, existing, "ROW2_F16", row2)
    _add_init(model, existing, "COL2_F16", col2)


def _rewrite_nodes(model: onnx.ModelProto) -> None:
    for node in model.graph.node:
        # The Conv still produces FLOAT, but ARC colors are only 0..9.
        if node.name == "cell_color_cast_new" and node.op_type == "Cast":
            _set_cast_to(node, TensorProto.UINT8)

        # UINT8 constants for color-valued comparisons and Where fallbacks.
        if node.name == "non_bg_b":
            node.input[1] = "ZERO_U8"
        if node.name == "cc_pad":
            node.input[2] = "ZERO_U8"
        if node.op_type == "Greater" and "_nb_" in node.name:
            node.input[1] = "ZERO_U8"
        if node.op_type == "Where" and (
            node.name.startswith("H_")
            or node.name.startswith("V_")
            or node.name.startswith("D_")
        ):
            node.input[2] = "ZERO_U8"
        if node.op_type == "Where" and (
            node.name.startswith("v_A_")
            or node.name.startswith("v_B_")
            or node.name.startswith("v_D_")
        ):
            node.input[2] = "ZERO_U8"
        if node.name == "marker_color_grid":
            node.input[2] = "ZERO_U8"
        if node.name in {"A_value_pos", "B_value_pos", "D_value_pos"}:
            node.input[1] = "ZERO_U8"
        if node.name in {"A_value", "B_value", "D_value"}:
            node.input[2] = "ZERO_U8"
        if node.name == "out0_b0":
            node.input[1] = "ZERO_U8"
        if node.name.startswith("out") and node.name.endswith("_b0") and node.name != "out0_b0":
            color = node.name[3]
            if color.isdigit():
                node.input[1] = f"C{color}_U8"

        # A/B/D grids now carry one selected UINT8 color per cell, so Max is
        # equivalent to Sum and is accepted for UINT8 by shape inference/ORT.
        if node.name in {"A_grid", "B_grid", "D_grid"} and node.op_type == "Sum":
            node.op_type = "Max"

        # Propagate zero-based marker labels. Valid labels are now >= 0, while
        # non-marker cells remain NEG16, so > -0.5 replaces > 0.
        if node.name == "seed_single_0":
            node.input[1] = "LABEL0"
        if node.name in {
            "seed_single_has0",
            "seed_single_has1",
            "seed_single_has2",
            "seed_single_has3",
            "has_xmark",
        }:
            node.input[1] = "NEG_HALF_HW"

        # Reflection-origin rewrite:
        #   old target row = 2 * marker_row + selected_dr - current_row
        #   old target col = 2 * marker_col + selected_dc - current_col
        # Build the selected origin grid once, gather it at marker_id, and
        # reuse the original final Sub against ROW16/COL16.
        if node.name == "xr_div_f":
            _make_binary(node, "Add", ["ROW2_F16", "ds_grid"], "base_r_grid")
        elif node.name == "xr_floor":
            _make_reshape(node, ["base_r_grid", "shape_900"], "base_r_flat")
        elif node.name == "xr_x30_f":
            _make_gather(node, ["base_r_flat", "marker_id_i64"], "base_r_at")
        elif node.name == "xc_sub":
            _make_binary(node, "Add", ["COL2_F16", "cs_grid"], "base_c_grid")
        elif node.name == "xc2":
            _make_reshape(node, ["base_c_grid", "shape_900"], "base_c_flat")
        elif node.name == "xc2cs":
            _make_gather(node, ["base_c_flat", "marker_id_i64"], "base_c_at")
        elif node.name == "tgt_c":
            node.input[0] = "base_c_at"
        elif node.name == "tgt_r":
            node.input[0] = "base_r_at"
        elif node.name == "self_r":
            _make_binary(node, "Equal", ["seed_single_4", "LABEL0"], "self_marker_direct")

    bypasses = {
        "A_value32": "A_value",
        "B_value32": "B_value",
        "D_value32": "D_value",
        "mirror": "mirror32",
        "seed_single_id_f0": "seed_single_p0",
        "seed_single_id_f1": "seed_single_p1",
        "seed_single_id_f2": "seed_single_p2",
        "seed_single_id_f3": "seed_single_p3",
        "marker_id_f": "seed_single_4",
        "self_marker": "self_marker_direct",
    }
    for node in model.graph.node:
        for old, new in bypasses.items():
            _replace_input(node, old, new)


def prune_dead_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    needed = {output.name for output in model.graph.output}
    kept: list[onnx.NodeProto] = []
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            kept.append(node)
            needed.update(input_name for input_name in node.input if input_name)
    kept.reverse()

    used_initializers = {
        input_name for node in kept for input_name in node.input if input_name
    }
    graph_inputs = {value.name for value in model.graph.input}
    graph = helper.make_graph(
        kept,
        "task285_v2_compact",
        list(model.graph.input),
        list(model.graph.output),
        initializer=[
            initializer
            for initializer in model.graph.initializer
            if initializer.name in used_initializers and initializer.name not in graph_inputs
        ],
    )
    rebuilt = helper.make_model(
        graph,
        opset_imports=list(model.opset_import),
        producer_name="kaggleonnx_task285_v2",
    )
    rebuilt.ir_version = model.ir_version
    return onnx.shape_inference.infer_shapes(rebuilt, check_type=True, strict_mode=True)


def build_model(anchor_path: Path = DEFAULT_ANCHOR) -> onnx.ModelProto:
    model = onnx.load(str(anchor_path))
    del model.graph.value_info[:]
    _insert_compact_initializers(model)
    _rewrite_nodes(model)
    compact = prune_dead_nodes(model)
    onnx.checker.check_model(compact, full_check=True)
    return compact


def _score_dict(score: Any) -> dict[str, Any]:
    row = asdict(score)
    row["path"] = str(row["path"])
    return row


def _onnx_solver(model_path: Path):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )

    def solve(grid: np.ndarray) -> np.ndarray:
        output = session.run(["output"], {"input": encode_grid(grid.tolist())})[0]
        return np.asarray(decode_grid(output), dtype=np.int64)

    return solve


def _session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def _compare_sessions(anchor_path: Path, candidate_path: Path, examples: list[dict]) -> dict[str, Any]:
    anchor = _session(anchor_path)
    candidate = _session(candidate_path)
    exact = 0
    mismatches: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        inp = encode_grid(example["input"])
        anchor_out = anchor.run(["output"], {"input": inp})[0] > 0.0
        candidate_out = candidate.run(["output"], {"input": inp})[0] > 0.0
        ok = bool(np.array_equal(anchor_out, candidate_out))
        exact += int(ok)
        if not ok and len(mismatches) < 10:
            mismatches.append(
                {
                    "index": index,
                    "mismatch_count": int(np.count_nonzero(anchor_out != candidate_out)),
                }
            )
    return {"exact": exact, "total": len(examples), "mismatches": mismatches}


def _pseudo_hidden_summary(model_path: Path, examples: list[dict]) -> dict[str, Any]:
    results = ph.run_contracts(
        _onnx_solver(model_path),
        examples,
        pseudo_hidden_contracts(),
        include_original=True,
        rule_label="task285_v2_onnx",
    )
    summary = ph.summarize_results(results)
    failures = [
        asdict(result)
        for result in results
        if not result.passed and not result.skipped
    ][:20]
    return {"summary": summary, "failures_sample": failures}


def validate_model(
    candidate_path: Path,
    anchor_path: Path = DEFAULT_ANCHOR,
    report_path: Path | None = DEFAULT_REPORT,
    *,
    include_pseudo_hidden: bool = True,
) -> dict[str, Any]:
    examples = all_examples(285, ROOT / "data" / "neurogolf-2026" / "raw")
    anchor_score = score_onnx(anchor_path, examples, 285, n_runs=0)
    candidate_score = score_onnx(candidate_path, examples, 285, n_runs=0)
    equivalence = _compare_sessions(anchor_path, candidate_path, examples)

    report: dict[str, Any] = {
        "task": 285,
        "anchor": _score_dict(anchor_score),
        "candidate": _score_dict(candidate_score),
        "decoded_equivalence_vs_anchor": equivalence,
        "delta": {
            "cost": (
                None
                if anchor_score.cost is None or candidate_score.cost is None
                else candidate_score.cost - anchor_score.cost
            ),
            "score": candidate_score.score - anchor_score.score,
            "memory": (
                None
                if anchor_score.memory is None or candidate_score.memory is None
                else candidate_score.memory - anchor_score.memory
            ),
            "params": (
                None
                if anchor_score.params is None or candidate_score.params is None
                else candidate_score.params - anchor_score.params
            ),
            "nodes": candidate_score.nodes - anchor_score.nodes,
            "file_size": candidate_score.file_size - anchor_score.file_size,
        },
        "can_stage": bool(
            candidate_score.error == ""
            and candidate_score.local_pass == candidate_score.local_total
            and equivalence["exact"] == equivalence["total"]
            and candidate_score.cost is not None
            and anchor_score.cost is not None
            and candidate_score.cost < anchor_score.cost
        ),
    }

    if include_pseudo_hidden:
        report["pseudo_hidden"] = _pseudo_hidden_summary(candidate_path, examples)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--skip-pseudo-hidden", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    model = build_model(args.anchor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.out))
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes, {len(model.graph.node)} nodes)")

    if args.validate:
        report = validate_model(
            args.out,
            args.anchor,
            args.report,
            include_pseudo_hidden=not args.skip_pseudo_hidden,
        )
        candidate = report["candidate"]
        print(
            "validation: "
            f"{candidate['local_pass']}/{candidate['local_total']} "
            f"cost={candidate['cost']} score={candidate['score']:.9f} "
            f"can_stage={report['can_stage']}"
        )
        if "pseudo_hidden" in report:
            print(f"pseudo_hidden: {report['pseudo_hidden']['summary']}")


if __name__ == "__main__":
    main()
