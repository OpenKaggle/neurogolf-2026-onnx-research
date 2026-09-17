from __future__ import annotations

import argparse
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def build_model() -> onnx.ModelProto:
    h = w = 30

    row_plus_col_plus_one = np.fromfunction(
        lambda _b, _c, r, col: r + col + 1,
        (1, 1, h, w),
        dtype=np.int64,
    ).astype(np.int64)
    row_plus_one = np.arange(1, h + 1, dtype=np.int64).reshape(1, 1, h, 1)
    col_plus_one = np.arange(1, w + 1, dtype=np.int64).reshape(1, 1, 1, w)
    channel_updates = np.zeros((1, 3, h, w), dtype=np.int64)
    channel_updates[:, 0, :, :] = 0
    channel_updates[:, 1, :, :] = 2
    channel_updates[:, 2, :, :] = 4
    zero = np.zeros((1, 1, 1, 1), dtype=np.float32)
    one_i64 = np.ones((1, 1, 1), dtype=np.int64)
    zero_i64 = np.zeros((), dtype=np.int64)
    channels_024 = np.array([0, 2, 4], dtype=np.int64)

    initializers = [
        numpy_helper.from_array(row_plus_col_plus_one, "row_plus_col_plus_one"),
        numpy_helper.from_array(row_plus_one, "row_plus_one"),
        numpy_helper.from_array(col_plus_one, "col_plus_one"),
        numpy_helper.from_array(channel_updates, "channel_updates"),
        numpy_helper.from_array(zero, "zero"),
        numpy_helper.from_array(one_i64, "one_i64"),
        numpy_helper.from_array(zero_i64, "zero_i64"),
        numpy_helper.from_array(channels_024, "channels_024"),
    ]

    selected_channels = ["ch0", "ch2", "ch4"]

    nodes = [
        helper.make_node("Gather", ["input", "zero_i64"], ["input_col0"], axis=3, name="input_col0"),
        helper.make_node("ReduceSum", ["input_col0"], ["n"], axes=[1, 2], keepdims=1, name="n"),
        helper.make_node("Cast", ["n"], ["n_i64"], to=TensorProto.INT64, name="n_i64"),
        helper.make_node("Add", ["n_i64", "one_i64"], ["n_plus_one"], name="n_plus_one"),
        helper.make_node("Greater", ["input", "zero"], ["input_bool"], name="input_bool"),
        helper.make_node("Gather", ["input_bool", "channels_024"], ["ch024"], axis=1, name="ch024"),
        helper.make_node("Split", ["ch024"], selected_channels, axis=1, split=[1, 1, 1], name="split_024"),
        helper.make_node("Greater", ["col_plus_one", "one_i64"], ["not_col0"], name="not_col0"),
        helper.make_node(
            "Equal",
            ["row_plus_col_plus_one", "n_i64"],
            ["anti_bool0"],
            name="anti_bool0",
        ),
        helper.make_node("And", ["anti_bool0", "not_col0"], ["anti_bool"], name="anti_bool"),
        helper.make_node("Equal", ["row_plus_one", "n_i64"], ["last_row_bool"], name="last_row_bool"),
        helper.make_node("Less", ["col_plus_one", "n_plus_one"], ["in_width"], name="in_width"),
        helper.make_node("And", ["last_row_bool", "in_width"], ["last_valid_bool"], name="last_valid_bool"),
        helper.make_node("And", ["last_valid_bool", "not_col0"], ["last_bool"], name="last_bool"),
        helper.make_node("Or", ["anti_bool", "last_bool"], ["colored"], name="colored"),
        helper.make_node("Not", ["colored"], ["not_colored"], name="not_colored"),
        helper.make_node("And", ["ch0", "not_colored"], ["ch0_out"], name="ch0_out"),
        helper.make_node("Or", ["ch2", "anti_bool"], ["ch2_out"], name="ch2_out"),
        helper.make_node("Or", ["ch4", "last_bool"], ["ch4_out"], name="ch4_out"),
        helper.make_node("Concat", ["ch0_out", "ch2_out", "ch4_out"], ["updates"], axis=1, name="updates"),
        helper.make_node(
            "ScatterElements",
            ["input_bool", "channel_updates", "updates"],
            ["output"],
            axis=1,
            name="output",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task084_semantic",
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
    parser.add_argument("--out", default="submissions/handbuilds/task084.onnx")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
