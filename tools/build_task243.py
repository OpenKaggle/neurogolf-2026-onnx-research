from __future__ import annotations

import argparse
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def node(op_type: str, inputs: list[str], outputs: list[str], **kwargs) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, outputs, name=outputs[0], **kwargs)


def build_model(iterations: int = 28) -> onnx.ModelProto:
    h = w = 30
    cross = np.zeros((1, 1, 3, 3), dtype=np.float32)
    cross[0, 0, 1, 1] = 1.0
    cross[0, 0, 0, 1] = 1.0
    cross[0, 0, 2, 1] = 1.0
    cross[0, 0, 1, 0] = 1.0
    cross[0, 0, 1, 2] = 1.0

    initializers = [
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero_f32"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half_f32"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_ch"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_ch"),
        numpy_helper.from_array(np.zeros((1, 1, h, w), dtype=np.int64), "zero_ch_idx"),
        numpy_helper.from_array(np.ones((1, 1, h, w), dtype=np.int64), "one_ch_idx"),
        numpy_helper.from_array(cross, "cross_w"),
    ]
    nodes: list[onnx.NodeProto] = [
        node("Greater", ["input", "zero_f32"], ["input_bool"]),
        node("Gather", ["input_bool", "zero_ch"], ["ch0"], axis=1),
        node("Gather", ["input_bool", "one_ch"], ["ch1"], axis=1),
    ]

    reach = "ch1"
    for idx in range(iterations):
        nodes.extend(
            [
                node("Cast", [reach], [f"reach_{idx}_f"], to=TensorProto.FLOAT),
                node(
                    "Conv",
                    [f"reach_{idx}_f", "cross_w"],
                    [f"neighbor_{idx}"],
                    pads=[1, 1, 1, 1],
                ),
                node("Greater", [f"neighbor_{idx}", "half_f32"], [f"dilate_{idx}"]),
                node("And", [f"dilate_{idx}", "ch0"], [f"dilate0_{idx}"]),
                node("Or", ["ch1", f"dilate0_{idx}"], [f"reach_{idx + 1}"]),
            ]
        )
        reach = f"reach_{idx + 1}"

    nodes.extend(
        [
            node("And", [reach, "ch0"], ["filled0"]),
            node("Not", ["filled0"], ["not_filled0"]),
            node("And", ["ch0", "not_filled0"], ["ch0_out"]),
            node("ScatterElements", ["input_bool", "zero_ch_idx", "ch0_out"], ["out_ch0"], axis=1),
            node("ScatterElements", ["out_ch0", "one_ch_idx", reach], ["output"], axis=1),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task243_flood_fill_from_one",
        [value_info("input", TensorProto.FLOAT, [1, 10, h, w])],
        [value_info("output", TensorProto.BOOL, [1, 10, h, w])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submissions/handbuilds/task243.onnx")
    parser.add_argument("--iterations", type=int, default=28)
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(args.iterations), out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
