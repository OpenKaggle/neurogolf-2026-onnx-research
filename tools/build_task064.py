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
    zero_f32 = np.array(0.0, dtype=np.float32)
    half_f32 = np.array(0.5, dtype=np.float32)
    three_half_f32 = np.array(3.5, dtype=np.float32)
    conv_weight = np.ones((10, 1, 2, 2), dtype=np.float32)

    # MatMul uses B[k, j], so triu builds prefix sums over k <= j and tril suffix sums.
    prefix_mat = np.triu(np.ones((w, w), dtype=np.float32))
    suffix_mat = np.tril(np.ones((w, w), dtype=np.float32))

    initializers = [
        numpy_helper.from_array(zero_f32, "zero_f32"),
        numpy_helper.from_array(half_f32, "half_f32"),
        numpy_helper.from_array(three_half_f32, "three_half_f32"),
        numpy_helper.from_array(conv_weight, "conv_weight"),
        numpy_helper.from_array(prefix_mat, "prefix_mat"),
        numpy_helper.from_array(suffix_mat, "suffix_mat"),
    ]

    nodes: list[onnx.NodeProto] = [
        node("Greater", ["input", "zero_f32"], ["input_bool"]),
        node("ReduceSum", ["input"], ["count"], axes=[2, 3], keepdims=1),
        node("ReduceMax", ["count"], ["max_count"], axes=[1], keepdims=1),
        node("Greater", ["count", "zero_f32"], ["positive"]),
        node("Equal", ["count", "max_count"], ["bg_sel"]),
        node("Conv", ["input", "conv_weight"], ["solid2"], group=10),
        node("ReduceMax", ["solid2"], ["solid2_max"], axes=[2, 3], keepdims=1),
        node("Greater", ["solid2_max", "three_half_f32"], ["has_solid2"]),
        node("Not", ["has_solid2"], ["no_solid2"]),
        node("And", ["positive", "no_solid2"], ["marker_sel"]),
        node("Not", ["bg_sel"], ["not_bg"]),
        node("And", ["positive", "has_solid2"], ["solid_positive"]),
        node("And", ["solid_positive", "not_bg"], ["obj_sel"]),
        node("And", ["input_bool", "marker_sel"], ["marker_ch"]),
        node("And", ["input_bool", "obj_sel"], ["obj_ch"]),
        node("Cast", ["marker_ch"], ["marker_ch_f"], to=TensorProto.FLOAT),
        node("Cast", ["obj_ch"], ["obj_ch_f"], to=TensorProto.FLOAT),
        node("ReduceSum", ["marker_ch_f"], ["marker_sum"], axes=[1], keepdims=1),
        node("ReduceSum", ["obj_ch_f"], ["obj_sum"], axes=[1], keepdims=1),
        node("Greater", ["marker_sum", "half_f32"], ["marker_cells"]),
        node("Greater", ["obj_sum", "half_f32"], ["obj_cells"]),
        node("Cast", ["obj_cells"], ["obj_cells_f"], to=TensorProto.FLOAT),
        node("ReduceSum", ["obj_cells_f"], ["row_obj_sum"], axes=[3], keepdims=1),
        node("ReduceSum", ["obj_cells_f"], ["col_obj_sum"], axes=[2], keepdims=1),
        node("Greater", ["row_obj_sum", "half_f32"], ["row_has_obj"]),
        node("Greater", ["col_obj_sum", "half_f32"], ["col_has_obj"]),
        node("Cast", ["row_has_obj"], ["row_has_obj_f"], to=TensorProto.FLOAT),
        node("Cast", ["col_has_obj"], ["col_has_obj_f"], to=TensorProto.FLOAT),
        node("MatMul", ["col_has_obj_f", "prefix_mat"], ["col_prefix"]),
        node("MatMul", ["col_has_obj_f", "suffix_mat"], ["col_suffix"]),
        node("Greater", ["col_prefix", "half_f32"], ["col_prefix_any"]),
        node("Greater", ["col_suffix", "half_f32"], ["col_suffix_any"]),
        node("Not", ["col_prefix_any"], ["before_obj_cols"]),
        node("Not", ["col_suffix_any"], ["after_obj_cols"]),
        node("Transpose", ["row_has_obj_f"], ["row_has_obj_t"], perm=[0, 1, 3, 2]),
        node("MatMul", ["row_has_obj_t", "prefix_mat"], ["row_prefix_t"]),
        node("MatMul", ["row_has_obj_t", "suffix_mat"], ["row_suffix_t"]),
        node("Transpose", ["row_prefix_t"], ["row_prefix"], perm=[0, 1, 3, 2]),
        node("Transpose", ["row_suffix_t"], ["row_suffix"], perm=[0, 1, 3, 2]),
        node("Greater", ["row_prefix", "half_f32"], ["row_prefix_any"]),
        node("Greater", ["row_suffix", "half_f32"], ["row_suffix_any"]),
        node("Not", ["row_prefix_any"], ["before_obj_rows"]),
        node("Not", ["row_suffix_any"], ["after_obj_rows"]),
        node("And", ["marker_cells", "row_has_obj"], ["marker_obj_rows"]),
        node("And", ["marker_obj_rows", "before_obj_cols"], ["marker_left"]),
        node("And", ["marker_obj_rows", "after_obj_cols"], ["marker_right"]),
        node("Cast", ["marker_left"], ["marker_left_f"], to=TensorProto.FLOAT),
        node("Cast", ["marker_right"], ["marker_right_f"], to=TensorProto.FLOAT),
        node("MatMul", ["marker_left_f", "prefix_mat"], ["left_prefix"]),
        node("MatMul", ["marker_right_f", "suffix_mat"], ["right_suffix"]),
        node("Greater", ["left_prefix", "half_f32"], ["left_reach"]),
        node("Greater", ["right_suffix", "half_f32"], ["right_reach"]),
        node("And", ["left_reach", "row_has_obj"], ["left_row_reach"]),
        node("And", ["right_reach", "row_has_obj"], ["right_row_reach"]),
        node("And", ["left_row_reach", "before_obj_cols"], ["fill_left"]),
        node("And", ["right_row_reach", "after_obj_cols"], ["fill_right"]),
        node("And", ["marker_cells", "col_has_obj"], ["marker_obj_cols"]),
        node("And", ["marker_obj_cols", "before_obj_rows"], ["marker_above"]),
        node("And", ["marker_obj_cols", "after_obj_rows"], ["marker_below"]),
        node("Cast", ["marker_above"], ["marker_above_f"], to=TensorProto.FLOAT),
        node("Cast", ["marker_below"], ["marker_below_f"], to=TensorProto.FLOAT),
        node("Transpose", ["marker_above_f"], ["marker_above_t"], perm=[0, 1, 3, 2]),
        node("Transpose", ["marker_below_f"], ["marker_below_t"], perm=[0, 1, 3, 2]),
        node("MatMul", ["marker_above_t", "prefix_mat"], ["above_prefix_t"]),
        node("MatMul", ["marker_below_t", "suffix_mat"], ["below_suffix_t"]),
        node("Transpose", ["above_prefix_t"], ["above_prefix"], perm=[0, 1, 3, 2]),
        node("Transpose", ["below_suffix_t"], ["below_suffix"], perm=[0, 1, 3, 2]),
        node("Greater", ["above_prefix", "half_f32"], ["above_reach"]),
        node("Greater", ["below_suffix", "half_f32"], ["below_reach"]),
        node("And", ["above_reach", "col_has_obj"], ["above_col_reach"]),
        node("And", ["below_reach", "col_has_obj"], ["below_col_reach"]),
        node("And", ["above_col_reach", "before_obj_rows"], ["fill_above"]),
        node("And", ["below_col_reach", "after_obj_rows"], ["fill_below"]),
        node("Or", ["fill_left", "fill_right"], ["fill_h"]),
        node("Or", ["fill_above", "fill_below"], ["fill_v"]),
        node("Or", ["fill_h", "fill_v"], ["extension"]),
        node("And", ["marker_sel", "extension"], ["marker_extension"]),
        node("Or", ["input_bool", "marker_extension"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "task064_semantic",
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
    parser.add_argument("--out", default="submissions/handbuilds/task064.onnx")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
