#!/usr/bin/env python3
"""Build task191_v2 by narrowing the terminal output lifetime.

The task191 exact anchor already implements the closed boundary-aware semantic
rule. Its final node casts the complete one-hot tensor from FLOAT16 to FLOAT.
The local scorer decodes by sign/positivity, so exposing the pre-cast FLOAT16
tensor as the graph output preserves exact decoded behavior while removing one
full output-shaped intermediate from measured memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR = ROOT / "submissions" / "candidate_v4_plus9_compiler_onnx" / "task191.onnx"
DEFAULT_OUT = ROOT / "submissions" / "handbuilds" / "task191_v2.onnx"


def strip_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    rebuilt = onnx.ModelProto.FromString(model.SerializeToString())
    del rebuilt.graph.value_info[:]
    return rebuilt


def prune_dead_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    needed = {output.name for output in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            kept.append(node)
            needed.update(input_name for input_name in node.input if input_name)
    kept.reverse()

    used_initializers = {input_name for node in kept for input_name in node.input if input_name}
    graph_inputs = {value.name for value in model.graph.input}
    graph = helper.make_graph(
        kept,
        "task191_v2_terminal_lifetime",
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
        producer_name="kaggleonnx_task191_v2",
    )
    rebuilt.ir_version = model.ir_version
    return rebuilt


def build_model(anchor_path: Path = DEFAULT_ANCHOR) -> onnx.ModelProto:
    model = strip_value_info(onnx.load(str(anchor_path)))
    if not model.graph.output or model.graph.output[0].name != "output":
        raise ValueError("expected single graph output named output")
    if not model.graph.node:
        raise ValueError("anchor graph has no nodes")

    terminal = model.graph.node[-1]
    if terminal.op_type != "Cast" or list(terminal.output) != ["output"] or len(terminal.input) != 1:
        raise ValueError("expected terminal Cast producing output")

    pre_cast_output = terminal.input[0]
    producer = None
    for node in model.graph.node:
        if pre_cast_output in node.output:
            producer = node
            break
    if producer is None:
        raise ValueError(f"could not find producer for {pre_cast_output}")

    for i, name in enumerate(producer.output):
        if name == pre_cast_output:
            producer.output[i] = "output"
            break
    else:
        raise ValueError(f"producer does not output {pre_cast_output}")

    del model.graph.node[-1]
    model.graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    narrowed = prune_dead_nodes(model)
    checked = onnx.shape_inference.infer_shapes(narrowed, check_type=True, strict_mode=True)
    onnx.checker.check_model(checked, full_check=True)
    return narrowed


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
