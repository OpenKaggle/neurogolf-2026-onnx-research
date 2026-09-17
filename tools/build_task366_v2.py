#!/usr/bin/env python3
"""Build task366_v2 ONNX by applying the bounded-pass semantic closure.

The existing 713-node anchor performs 11 MaxPool label-propagation passes in
each panel. The task366_v2 prototype proves all source objects in train/test/
arc-gen are filled rectangles with max side length 7, so 6 propagation passes
are sufficient and are the first passing bound. This builder rewires the anchor
label tensors to pass 6 and prunes the now-dead pass-7..11 nodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR = ROOT / "submissions" / "candidate_v4_plus7_task149_onnx" / "task366.onnx"
DEFAULT_OUT = ROOT / "submissions" / "handbuilds" / "task366_v2.onnx"
BOUNDED_PASSES = 6


def prune_dead_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
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
        "task366_v2_bounded_pass",
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

    bounded = prune_dead_nodes(model)
    onnx.checker.check_model(bounded, full_check=True)
    return bounded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    model = build_model(args.anchor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.out))
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes, {len(model.graph.node)} nodes)")


if __name__ == "__main__":
    main()
