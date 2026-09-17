"""Build degenerate constant-zero ONNX for task018 probe.

Produces a Constant node that always outputs [1,10,30,30] all-zeros,
ignoring the input. This is the cheapest possible valid ONNX for H1 probing.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper as oh, numpy_helper as onh


def build(out_path: pathlib.Path) -> onnx.ModelProto:
    zero_tensor = onh.from_array(
        np.zeros((1, 10, 30, 30), dtype=np.float32),
        name="constant_zero",
    )
    node = oh.make_node("Constant", inputs=[], outputs=["output"], value=zero_tensor)

    graph = oh.make_graph(
        [node],
        "task018_degenerate",
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
    parser.add_argument("--output", default="submissions/handbuilds/task018_degenerate.onnx")
    args = parser.parse_args()
    m = build(pathlib.Path(args.output))
    print(f"task018 degenerate built: {pathlib.Path(args.output).stat().st_size} bytes, {len(m.graph.node)} node(s)")


if __name__ == "__main__":
    main()
