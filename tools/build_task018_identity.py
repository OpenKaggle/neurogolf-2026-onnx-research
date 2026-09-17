"""Build degenerate Identity ONNX for task018 probe (cost ~0).

Identity(input) -> output. Copies input to output, zero params, zero memory.
This is the absolute cheapest valid ONNX possible.
"""
from __future__ import annotations

import argparse
import pathlib

import onnx
from onnx import TensorProto, helper as oh


def build(out_path: pathlib.Path) -> onnx.ModelProto:
    node = oh.make_node("Identity", inputs=["input"], outputs=["output"])

    graph = oh.make_graph(
        [node],
        "task018_identity",
        [oh.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [oh.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 11)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submissions/handbuilds/task018_identity.onnx")
    args = parser.parse_args()
    m = build(pathlib.Path(args.output))
    print(f"task018 identity built: {pathlib.Path(args.output).stat().st_size} bytes, {len(m.graph.node)} node(s)")


if __name__ == "__main__":
    main()
