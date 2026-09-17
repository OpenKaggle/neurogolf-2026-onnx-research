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


def build_model() -> onnx.ModelProto:
    h = w = 30
    ch0 = np.zeros((1, 10, 1, 1), dtype=np.bool_)
    ch0[:, 0] = True
    not_ch0 = np.ones((1, 10, 1, 1), dtype=np.bool_)
    not_ch0[:, 0] = False

    row = np.arange(h, dtype=np.float32).reshape(1, 1, h, 1)
    col = np.arange(w, dtype=np.float32).reshape(1, 1, 1, w)

    initializers = [
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half"),
        numpy_helper.from_array(np.array(11.5, dtype=np.float32), "line_threshold"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
        numpy_helper.from_array(row, "row"),
        numpy_helper.from_array(col, "col"),
        numpy_helper.from_array(ch0, "ch0"),
        numpy_helper.from_array(not_ch0, "not_ch0"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), "channel_axis"),
    ]

    nodes = [
        node("Greater", ["input", "zero"], ["input_bool"]),
        node("ReduceSum", ["input"], ["area_sum"], axes=[1], keepdims=1),
        node("Greater", ["area_sum", "half"], ["valid_area"]),
        node("And", ["input_bool", "not_ch0"], ["color_input"]),
        node("Cast", ["color_input"], ["color_input_f"], to=TensorProto.FLOAT),
        node("ReduceSum", ["color_input_f"], ["row_counts"], axes=[3], keepdims=1),
        node("ReduceSum", ["color_input_f"], ["col_counts"], axes=[2], keepdims=1),
        node("Greater", ["row_counts", "line_threshold"], ["row_lines"]),
        node("Greater", ["col_counts", "line_threshold"], ["col_lines"]),
        node("ReduceSum", ["row_counts"], ["color_total"], axes=[2], keepdims=1),
        node("Greater", ["color_total", "line_threshold"], ["line_color"]),
        node("And", ["row_lines", "color_input"], ["on_row_line"]),
        node("And", ["col_lines", "color_input"], ["on_col_line"]),
        node("Or", ["on_row_line", "on_col_line"], ["on_line"]),
        node("Not", ["on_line"], ["not_on_line"]),
        node("And", ["color_input", "not_on_line"], ["stray"]),
        node("Cast", ["row_lines"], ["row_lines_f2"], to=TensorProto.FLOAT),
        node("Cast", ["col_lines"], ["col_lines_f2"], to=TensorProto.FLOAT),
        node("ReduceSum", ["row_lines_f2"], ["row_line_count"], axes=[2], keepdims=1),
        node("ReduceSum", ["col_lines_f2"], ["col_line_count"], axes=[3], keepdims=1),
        node("Greater", ["row_line_count", "half"], ["has_row_line"]),
        node("Greater", ["col_line_count", "half"], ["has_col_line"]),
        node("Cast", ["row_lines"], ["row_lines_f"], to=TensorProto.FLOAT),
        node("Cast", ["col_lines"], ["col_lines_f"], to=TensorProto.FLOAT),
        node("Mul", ["row_lines_f", "row"], ["row_line_weighted"]),
        node("Mul", ["col_lines_f", "col"], ["col_line_weighted"]),
        node("ReduceSum", ["row_line_weighted"], ["row_line_pos"], axes=[2], keepdims=1),
        node("ReduceSum", ["col_line_weighted"], ["col_line_pos"], axes=[3], keepdims=1),
        node("Sub", ["row_line_pos", "one"], ["row_above"]),
        node("Add", ["row_line_pos", "one"], ["row_below"]),
        node("Sub", ["col_line_pos", "one"], ["col_left"]),
        node("Add", ["col_line_pos", "one"], ["col_right"]),
        node("Less", ["row", "row_line_pos"], ["above_line"]),
        node("Less", ["row_line_pos", "row"], ["below_line"]),
        node("Less", ["col", "col_line_pos"], ["left_of_line"]),
        node("Less", ["col_line_pos", "col"], ["right_of_line"]),
        node("Equal", ["row", "row_above"], ["row_target_above"]),
        node("Equal", ["row", "row_below"], ["row_target_below"]),
        node("Equal", ["col", "col_left"], ["col_target_left"]),
        node("Equal", ["col", "col_right"], ["col_target_right"]),
        node("And", ["stray", "has_row_line"], ["row_stray0"]),
        node("And", ["row_stray0", "above_line"], ["row_stray_above"]),
        node("And", ["row_stray0", "below_line"], ["row_stray_below"]),
        node("And", ["stray", "has_col_line"], ["col_stray0"]),
        node("And", ["col_stray0", "left_of_line"], ["col_stray_left"]),
        node("And", ["col_stray0", "right_of_line"], ["col_stray_right"]),
        node("Cast", ["row_stray_above"], ["row_stray_above_f"], to=TensorProto.FLOAT),
        node("Cast", ["row_stray_below"], ["row_stray_below_f"], to=TensorProto.FLOAT),
        node("Cast", ["col_stray_left"], ["col_stray_left_f"], to=TensorProto.FLOAT),
        node("Cast", ["col_stray_right"], ["col_stray_right_f"], to=TensorProto.FLOAT),
        node("ReduceMax", ["row_stray_above_f"], ["row_above_cols"], axes=[2], keepdims=1),
        node("ReduceMax", ["row_stray_below_f"], ["row_below_cols"], axes=[2], keepdims=1),
        node("ReduceMax", ["col_stray_left_f"], ["col_left_rows"], axes=[3], keepdims=1),
        node("ReduceMax", ["col_stray_right_f"], ["col_right_rows"], axes=[3], keepdims=1),
        node("Greater", ["row_above_cols", "half"], ["row_above_cols_b"]),
        node("Greater", ["row_below_cols", "half"], ["row_below_cols_b"]),
        node("Greater", ["col_left_rows", "half"], ["col_left_rows_b"]),
        node("Greater", ["col_right_rows", "half"], ["col_right_rows_b"]),
        node("And", ["row_target_above", "row_above_cols_b"], ["row_move_above"]),
        node("And", ["row_target_below", "row_below_cols_b"], ["row_move_below"]),
        node("And", ["col_target_left", "col_left_rows_b"], ["col_move_left"]),
        node("And", ["col_target_right", "col_right_rows_b"], ["col_move_right"]),
        node("Or", ["row_move_above", "row_move_below"], ["row_moves"]),
        node("Or", ["col_move_left", "col_move_right"], ["col_moves"]),
        node("Or", ["row_moves", "col_moves"], ["moves"]),
        node("Or", ["on_line", "moves"], ["nonzero_output"]),
        node("Cast", ["nonzero_output"], ["nonzero_output_f"], to=TensorProto.FLOAT),
        node("CumSum", ["nonzero_output_f", "channel_axis"], ["higher_color_count"], exclusive=1, reverse=1),
        node("Greater", ["higher_color_count", "half"], ["higher_color_exists"]),
        node("Not", ["higher_color_exists"], ["no_higher_color"]),
        node("And", ["nonzero_output", "no_higher_color"], ["priority_nonzero"]),
        node("Cast", ["priority_nonzero"], ["priority_nonzero_f"], to=TensorProto.FLOAT),
        node("ReduceSum", ["priority_nonzero_f"], ["nonzero_count"], axes=[1], keepdims=1),
        node("Less", ["nonzero_count", "half"], ["empty_cells"]),
        node("And", ["valid_area", "empty_cells"], ["zero_cells"]),
        node("And", ["ch0", "zero_cells"], ["zero_output"]),
        node("Or", ["priority_nonzero", "zero_output"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "task025_semantic",
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
    parser.add_argument("--out", default="submissions/handbuilds/task025.onnx")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
