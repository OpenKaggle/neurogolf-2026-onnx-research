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
    row = np.arange(h, dtype=np.float32).reshape(1, 1, h, 1)
    col = np.arange(w, dtype=np.float32).reshape(1, 1, 1, w)
    row_i = np.arange(h, dtype=np.int64).reshape(1, 1, h, 1)
    col_i = np.arange(w, dtype=np.int64).reshape(1, 1, 1, w)
    color_ids = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    other_map = np.zeros((1, 10, 10, 1, 1), dtype=np.bool_)
    for source in range(10):
        for target in range(10):
            other_map[0, source, target, 0, 0] = source != 0 and target != 0 and source != target
    ch0 = np.zeros((1, 10, 1, 1), dtype=np.bool_)
    ch0[:, 0] = True
    not_ch0 = np.ones((1, 10, 1, 1), dtype=np.bool_)
    not_ch0[:, 0] = False

    # Input corner gaps are r1-r0/c1-c0 in [5, 11]. Connector offsets are relative
    # to the first cell between the two 3x3 corner stamps.
    patterns = {
        5: [0, 1],
        6: [0, 2],
        7: [0, 3],
        8: [0, 2, 4],
        9: [0, 2, 3, 5],
        10: [0, 2, 4, 6],
        11: [0, 2, 5, 7],
    }
    initializers = [
        numpy_helper.from_array(row, "row"),
        numpy_helper.from_array(col, "col"),
        numpy_helper.from_array(row_i, "row_i"),
        numpy_helper.from_array(col_i, "col_i"),
        numpy_helper.from_array(color_ids, "color_ids"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half"),
        numpy_helper.from_array(np.array(-1000.0, dtype=np.float32), "neg_big"),
        numpy_helper.from_array(np.array(5, dtype=np.int64), "five_i"),
        numpy_helper.from_array(np.array(2, dtype=np.int64), "two_i"),
        numpy_helper.from_array(np.array(30, dtype=np.int64), "thirty_i"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), "zero_i"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), "one_i"),
        numpy_helper.from_array(other_map, "other_map"),
        numpy_helper.from_array(ch0, "ch0"),
        numpy_helper.from_array(not_ch0, "not_ch0"),
    ]

    nodes: list[onnx.NodeProto] = [
        node("Greater", ["input", "zero"], ["input_bool"]),
        node("ReduceSum", ["input"], ["area_sum"], axes=[1], keepdims=1),
        node("Greater", ["area_sum", "half"], ["valid_area"]),
        node("And", ["input_bool", "not_ch0"], ["nonzero_ch"]),
        node("Cast", ["nonzero_ch"], ["nonzero_ch_f"], to=TensorProto.FLOAT),
        node("ReduceSum", ["nonzero_ch_f"], ["point_sum"], axes=[1], keepdims=1),
        node("Greater", ["point_sum", "half"], ["point_cells"]),
        node("Cast", ["point_cells"], ["point_f"], to=TensorProto.FLOAT),
        node("Mul", ["point_f", "row"], ["nz_row"]),
        node("Mul", ["point_f", "col"], ["nz_col"]),
        node("ReduceMax", ["nz_row"], ["r1_f"], axes=[2, 3], keepdims=1),
        node("ReduceMax", ["nz_col"], ["c1_f"], axes=[2, 3], keepdims=1),
        node("Not", ["point_cells"], ["not_points"]),
        node("Cast", ["not_points"], ["not_points_f"], to=TensorProto.FLOAT),
        node("Mul", ["not_points_f", "neg_big"], ["not_points_big_neg"]),
        node("Sub", ["zero", "row"], ["neg_row"]),
        node("Sub", ["zero", "col"], ["neg_col"]),
        node("Add", ["neg_row", "not_points_big_neg"], ["point_neg_row"]),
        node("Add", ["neg_col", "not_points_big_neg"], ["point_neg_col"]),
        node("ReduceMax", ["point_neg_row"], ["r0_neg"], axes=[2, 3], keepdims=1),
        node("ReduceMax", ["point_neg_col"], ["c0_neg"], axes=[2, 3], keepdims=1),
        node("Sub", ["zero", "r0_neg"], ["r0_f"]),
        node("Sub", ["zero", "c0_neg"], ["c0_f"]),
        node("Cast", ["r0_f"], ["r0"], to=TensorProto.INT64),
        node("Cast", ["r1_f"], ["r1"], to=TensorProto.INT64),
        node("Cast", ["c0_f"], ["c0"], to=TensorProto.INT64),
        node("Cast", ["c1_f"], ["c1"], to=TensorProto.INT64),
        node("Sub", ["r1", "r0"], ["row_gap"]),
        node("Sub", ["c1", "c0"], ["col_gap"]),
        node("Add", ["r0", "two_i"], ["r0_p2"]),
        node("Add", ["c0", "two_i"], ["c0_p2"]),
        node("Sub", ["row_i", "r0_p2"], ["row_off"]),
        node("Sub", ["col_i", "c0_p2"], ["col_off"]),
        node("Equal", ["row_i", "r0"], ["top_row"]),
        node("Equal", ["row_i", "r1"], ["bottom_row"]),
        node("Equal", ["col_i", "c0"], ["left_col"]),
        node("Equal", ["col_i", "c1"], ["right_col"]),
        node("Or", ["top_row", "bottom_row"], ["h_rows"]),
        node("Or", ["left_col", "right_col"], ["v_cols"]),
    ]

    def pattern_expr(gap_name: str, off_name: str, axis_name: str) -> str:
        terms: list[str] = []
        for gap, offsets in patterns.items():
            gap_const = f"{axis_name}_gap_{gap}"
            initializers.append(numpy_helper.from_array(np.array(gap, dtype=np.int64), gap_const))
            nodes.append(node("Equal", [gap_name, gap_const], [f"{axis_name}_is_gap_{gap}"]))
            offset_terms: list[str] = []
            for off in offsets:
                off_const = f"{axis_name}_off_{gap}_{off}"
                initializers.append(numpy_helper.from_array(np.array(off, dtype=np.int64), off_const))
                nodes.append(node("Equal", [off_name, off_const], [f"{axis_name}_offeq_{gap}_{off}"]))
                offset_terms.append(f"{axis_name}_offeq_{gap}_{off}")
            merged = offset_terms[0]
            for idx, term in enumerate(offset_terms[1:], start=1):
                out = f"{axis_name}_offany_{gap}_{idx}"
                nodes.append(node("Or", [merged, term], [out]))
                merged = out
            out = f"{axis_name}_pat_gap_{gap}"
            nodes.append(node("And", [f"{axis_name}_is_gap_{gap}", merged], [out]))
            terms.append(out)
        merged = terms[0]
        for idx, term in enumerate(terms[1:], start=1):
            out = f"{axis_name}_pat_any_{idx}"
            nodes.append(node("Or", [merged, term], [out]))
            merged = out
        return merged

    row_connector_pos = pattern_expr("row_gap", "row_off", "row")
    col_connector_pos = pattern_expr("col_gap", "col_off", "col")
    nodes.extend(
        [
            node("And", ["h_rows", col_connector_pos], ["h_conn"]),
            node("And", ["v_cols", row_connector_pos], ["v_conn"]),
            node("Or", ["h_conn", "v_conn"], ["conn_cells"]),
        ]
    )

    # Corner stamps. Each input color is dilated to the 3x3 stamp area. The
    # stamp center keeps its original color; the ring uses the other present
    # color because the inputs contain exactly two non-background colors.
    nodes.extend(
        [
            node("ReduceSum", ["nonzero_ch_f"], ["channel_count"], axes=[2, 3], keepdims=1),
            node("Greater", ["channel_count", "half"], ["present_ch"]),
            node("Conv", ["nonzero_ch_f", "stamp_w"], ["stamp_by_color_f"], pads=[1, 1, 1, 1], group=10),
            node("Greater", ["stamp_by_color_f", "half"], ["stamp_by_color"]),
            node("ReduceSum", ["stamp_by_color_f"], ["stamp_any_sum"], axes=[1], keepdims=1),
            node("Greater", ["stamp_any_sum", "half"], ["stamp_cells"]),
            node("Not", ["nonzero_ch"], ["not_source_center"]),
            node("And", ["stamp_by_color", "not_source_center"], ["ring_source"]),
            node("Unsqueeze", ["ring_source"], ["ring_source_u"], axes=[2]),
            node("Unsqueeze", ["present_ch"], ["present_target_u"], axes=[1]),
            node("And", ["ring_source_u", "other_map"], ["ring_target_all_u"]),
            node("And", ["ring_target_all_u", "present_target_u"], ["ring_target_u"]),
            node("Cast", ["ring_target_u"], ["ring_target_u_f"], to=TensorProto.FLOAT),
            node("ReduceSum", ["ring_target_u_f"], ["ring_target_count"], axes=[1], keepdims=0),
            node("Greater", ["ring_target_count", "half"], ["stamp_ring"]),
            node("Or", ["nonzero_ch", "stamp_ring"], ["stamp_output"]),
            node("Equal", ["color_ids_i", "five_i"], ["five_ch"]),
            node("And", ["five_ch", "conn_cells"], ["conn_output"]),
            node("Or", ["stamp_output", "conn_output"], ["nonzero_output"]),
            node("Cast", ["nonzero_output"], ["nonzero_output_f"], to=TensorProto.FLOAT),
            node("ReduceSum", ["nonzero_output_f"], ["nonzero_count"], axes=[1], keepdims=1),
            node("Less", ["nonzero_count", "half"], ["empty_cells"]),
            node("And", ["valid_area", "empty_cells"], ["zero_cells"]),
            node("And", ["ch0", "zero_cells"], ["zero_output"]),
            node("Or", ["nonzero_output", "zero_output"], ["output"]),
        ]
    )
    initializers.extend(
        [
            numpy_helper.from_array(np.ones((10, 1, 3, 3), dtype=np.float32), "stamp_w"),
            numpy_helper.from_array(np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1), "color_ids_i"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task387_semantic",
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
    parser.add_argument("--out", default="submissions/handbuilds/task387.onnx")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
