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
    col = np.arange(w, dtype=np.int64).reshape(1, 1, 1, w)
    row = np.arange(h, dtype=np.int64).reshape(1, 1, h, 1)
    row9_index = np.array(9, dtype=np.int64)
    zero_f32 = np.array(0.0, dtype=np.float32)
    zero_i64 = np.array(0, dtype=np.int64)
    one_i64 = np.array(1, dtype=np.int64)
    two_i64 = np.array(2, dtype=np.int64)
    three_i64 = np.array(3, dtype=np.int64)
    four_i64 = np.array(4, dtype=np.int64)
    ten_i64 = np.array(10, dtype=np.int64)
    ch0_selector = np.zeros((1, 10, 1, 1), dtype=np.bool_)
    ch0_selector[:, 0] = True
    not_ch0_selector = np.ones((1, 10, 1, 1), dtype=np.bool_)
    not_ch0_selector[:, 0] = False
    ch5_selector = np.zeros((1, 10, 1, 1), dtype=np.bool_)
    ch5_selector[:, 5] = True
    channels_1_9 = np.arange(1, 10, dtype=np.int64)

    initializers = [
        numpy_helper.from_array(col, "col"),
        numpy_helper.from_array(row, "row"),
        numpy_helper.from_array(row9_index, "row9_index"),
        numpy_helper.from_array(zero_f32, "zero_f32"),
        numpy_helper.from_array(zero_i64, "zero_i64"),
        numpy_helper.from_array(one_i64, "one_i64"),
        numpy_helper.from_array(two_i64, "two_i64"),
        numpy_helper.from_array(three_i64, "three_i64"),
        numpy_helper.from_array(four_i64, "four_i64"),
        numpy_helper.from_array(ten_i64, "ten_i64"),
        numpy_helper.from_array(ch0_selector, "ch0_selector"),
        numpy_helper.from_array(not_ch0_selector, "not_ch0_selector"),
        numpy_helper.from_array(ch5_selector, "ch5_selector"),
        numpy_helper.from_array(channels_1_9, "channels_1_9"),
    ]

    nodes = [
        helper.make_node("Gather", ["input", "channels_1_9"], ["color_input"], axis=1, name="color_input"),
        helper.make_node("ReduceSum", ["color_input"], ["color_valid"], axes=[1], keepdims=1, name="color_valid"),
        helper.make_node("Gather", ["color_valid", "row9_index"], ["valid_row9"], axis=2, name="valid_row9"),
        helper.make_node("Cast", ["col"], ["col_f32"], to=TensorProto.FLOAT, name="col_f32"),
        helper.make_node("Mul", ["valid_row9", "col_f32"], ["dot_col_weighted"], name="dot_col_weighted"),
        helper.make_node("ReduceSum", ["dot_col_weighted"], ["dot_col_f32"], axes=[3], keepdims=1, name="dot_col_f32"),
        helper.make_node("Cast", ["dot_col_f32"], ["dot_col"], to=TensorProto.INT64, name="dot_col"),
        helper.make_node(
            "ReduceSum",
            ["input"],
            ["channel_sum"],
            axes=[2, 3],
            keepdims=1,
            name="channel_sum",
        ),
        helper.make_node("Greater", ["channel_sum", "zero_f32"], ["dot_channel"], name="dot_channel"),
        helper.make_node("And", ["dot_channel", "not_ch0_selector"], ["dot_color"], name="dot_color"),
        helper.make_node("Less", ["row", "ten_i64"], ["row_in_10"], name="row_in_10"),
        helper.make_node("Less", ["col", "ten_i64"], ["col_in_10"], name="col_in_10"),
        helper.make_node("And", ["row_in_10", "col_in_10"], ["area"], name="area"),
        helper.make_node("Less", ["col", "dot_col"], ["before_dot"], name="before_dot"),
        helper.make_node("Not", ["before_dot"], ["from_dot"], name="from_dot"),
        helper.make_node("Sub", ["col", "dot_col"], ["offset"], name="offset"),
        helper.make_node("Mod", ["offset", "two_i64"], ["offset_mod2"], fmod=0, name="offset_mod2"),
        helper.make_node("Equal", ["offset_mod2", "zero_i64"], ["even_offset"], name="even_offset"),
        helper.make_node("And", ["area", "from_dot"], ["area_from_dot"], name="area_from_dot"),
        helper.make_node("And", ["area_from_dot", "even_offset"], ["color_cells"], name="color_cells"),
        helper.make_node("Equal", ["row", "zero_i64"], ["top_row"], name="top_row"),
        helper.make_node("Equal", ["row", "ten_i64"], ["never_row_10"], name="never_row_10"),
        helper.make_node("Sub", ["ten_i64", "one_i64"], ["nine_i64"], name="nine_i64"),
        helper.make_node("Equal", ["row", "nine_i64"], ["bottom_row"], name="bottom_row"),
        helper.make_node("Mod", ["offset", "four_i64"], ["offset_mod4"], fmod=0, name="offset_mod4"),
        helper.make_node("Equal", ["offset_mod4", "one_i64"], ["mod4_is_1"], name="mod4_is_1"),
        helper.make_node("Equal", ["offset_mod4", "three_i64"], ["mod4_is_3"], name="mod4_is_3"),
        helper.make_node("And", ["top_row", "mod4_is_1"], ["top_mark0"], name="top_mark0"),
        helper.make_node("And", ["bottom_row", "mod4_is_3"], ["bottom_mark0"], name="bottom_mark0"),
        helper.make_node("Or", ["top_mark0", "bottom_mark0"], ["mark0"], name="mark0"),
        helper.make_node("And", ["area_from_dot", "mark0"], ["mark_cells"], name="mark_cells"),
        helper.make_node("And", ["dot_color", "color_cells"], ["color_output"], name="color_output"),
        helper.make_node("And", ["ch5_selector", "mark_cells"], ["mark_output"], name="mark_output"),
        helper.make_node("Or", ["color_cells", "mark_cells"], ["nonzero_cells"], name="nonzero_cells"),
        helper.make_node("Not", ["nonzero_cells"], ["not_nonzero"], name="not_nonzero"),
        helper.make_node("And", ["area", "not_nonzero"], ["zero_cells"], name="zero_cells"),
        helper.make_node("And", ["ch0_selector", "zero_cells"], ["zero_output"], name="zero_output"),
        helper.make_node("Or", ["color_output", "mark_output"], ["nonzero_output"], name="nonzero_output"),
        helper.make_node("Or", ["nonzero_output", "zero_output"], ["output"], name="output"),
    ]

    graph = helper.make_graph(
        nodes,
        "task200_semantic",
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
    parser.add_argument("--out", default="submissions/handbuilds/task200.onnx")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
