from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples, encode_grid
from prototype_task319 import analyze_objects


H = W = 30
CHANNELS = 10
TASK = 319


def value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def node(op_type: str, inputs: list[str], outputs: list[str], **kwargs) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, outputs, name=outputs[0], **kwargs)


def build_model() -> onnx.ModelProto:
    row_idx = np.arange(H, dtype=np.int64).reshape(1, 1, H)
    col_idx = np.arange(W, dtype=np.int64).reshape(1, 1, W)

    initializers = [
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero_f"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_i"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_i"),
        numpy_helper.from_array(np.array([31], dtype=np.int64), "big_i"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one_f"),
        numpy_helper.from_array(np.arange(CHANNELS, dtype=np.int64).reshape(1, CHANNELS), "color_idx"),
        numpy_helper.from_array(np.array([0, 0, 1, 0], dtype=np.int64), "row_tail_starts"),
        numpy_helper.from_array(np.array([1, CHANNELS, H, W], dtype=np.int64), "row_tail_ends"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), "row_head_starts"),
        numpy_helper.from_array(np.array([1, CHANNELS, H - 1, W], dtype=np.int64), "row_head_ends"),
        numpy_helper.from_array(np.array([0, 0, 0, 1], dtype=np.int64), "col_tail_starts"),
        numpy_helper.from_array(np.array([1, CHANNELS, H, W], dtype=np.int64), "col_tail_ends"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), "col_head_starts"),
        numpy_helper.from_array(np.array([1, CHANNELS, H, W - 1], dtype=np.int64), "col_head_ends"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), "slice_axes4"),
        numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), "slice_axes3"),
        numpy_helper.from_array(np.zeros((1, CHANNELS, 1), dtype=np.bool_), "false_row_pad"),
        numpy_helper.from_array(np.zeros((1, CHANNELS, 1), dtype=np.bool_), "false_col_pad"),
        numpy_helper.from_array(np.arange(H, dtype=np.int64).reshape(1, 1, H), "row_idx"),
        numpy_helper.from_array(np.arange(W, dtype=np.int64).reshape(1, 1, W), "col_idx"),
    ]

    nodes = [
        node("Greater", ["input", "zero_f"], ["mask_b"]),
        node("Cast", ["mask_b"], ["mask_i"], to=TensorProto.INT64),
        node("ReduceSum", ["mask_i"], ["color_counts"], axes=[2, 3], keepdims=0),
        node("Greater", ["color_counts", "zero_i"], ["has_cells_b"]),
        node("ArgMax", ["color_counts"], ["bg_idx"], axis=1, keepdims=0),
        node("Equal", ["color_idx", "bg_idx"], ["is_bg_b"]),
        node("Not", ["is_bg_b"], ["not_bg_b"]),
        node("And", ["has_cells_b", "not_bg_b"], ["valid_nonbg_b"]),
        node("Cast", ["valid_nonbg_b"], ["valid_nonbg_i"], to=TensorProto.INT64),
        node("ReduceMax", ["mask_i"], ["row_occ_i"], axes=[3], keepdims=0),
        node("ReduceMax", ["mask_i"], ["col_occ_i"], axes=[2], keepdims=0),
        node("Cast", ["row_occ_i"], ["row_occ_b"], to=TensorProto.BOOL),
        node("Cast", ["col_occ_i"], ["col_occ_b"], to=TensorProto.BOOL),
        node("Unsqueeze", ["color_counts"], ["color_counts_u"], axes=[2]),
        node("ReduceMin", ["row_idx"], ["row_idx_min_seed"], axes=[0, 1], keepdims=0),
        node("ReduceMin", ["col_idx"], ["col_idx_min_seed"], axes=[0, 1], keepdims=0),
        node("Unsqueeze", ["row_idx"], ["row_idx_u4"], axes=[3]),
        node("Unsqueeze", ["col_idx"], ["col_idx_u4"], axes=[2]),
        node("Where", ["row_occ_b", "row_idx", "big_i"], ["row_idx_or_big"]),
        node("Where", ["col_occ_b", "col_idx", "big_i"], ["col_idx_or_big"]),
        node("ReduceMin", ["row_idx_or_big"], ["row_min"], axes=[2], keepdims=0),
        node("ReduceMin", ["col_idx_or_big"], ["col_min"], axes=[2], keepdims=0),
        node("Mul", ["row_occ_i", "row_idx"], ["row_occ_weighted"]),
        node("Mul", ["col_occ_i", "col_idx"], ["col_occ_weighted"]),
        node("ReduceMax", ["row_occ_weighted"], ["row_max"], axes=[2], keepdims=0),
        node("ReduceMax", ["col_occ_weighted"], ["col_max"], axes=[2], keepdims=0),
        node("Unsqueeze", ["row_min"], ["row_min_u"], axes=[2]),
        node("Unsqueeze", ["row_max"], ["row_max_u"], axes=[2]),
        node("Unsqueeze", ["col_min"], ["col_min_u"], axes=[2]),
        node("Unsqueeze", ["col_max"], ["col_max_u"], axes=[2]),
        node("Less", ["row_idx", "row_min_u"], ["row_lt_min"]),
        node("Not", ["row_lt_min"], ["row_ge_min"]),
        node("Greater", ["row_idx", "row_max_u"], ["row_gt_max"]),
        node("Not", ["row_gt_max"], ["row_le_max"]),
        node("And", ["row_ge_min", "row_le_max"], ["row_in_bbox"]),
        node("Less", ["col_idx", "col_min_u"], ["col_lt_min"]),
        node("Not", ["col_lt_min"], ["col_ge_min"]),
        node("Greater", ["col_idx", "col_max_u"], ["col_gt_max"]),
        node("Not", ["col_gt_max"], ["col_le_max"]),
        node("And", ["col_ge_min", "col_le_max"], ["col_in_bbox"]),
        node("Unsqueeze", ["row_in_bbox"], ["row_in_bbox_u"], axes=[3]),
        node("Unsqueeze", ["col_in_bbox"], ["col_in_bbox_u"], axes=[2]),
        node("And", ["row_in_bbox_u", "col_in_bbox_u"], ["bbox_mask_b"]),
        node("And", ["mask_b", "bbox_mask_b"], ["patch_b"]),
        node("Cast", ["patch_b"], ["patch_i"], to=TensorProto.INT64),
        node("Slice", ["patch_i", "row_tail_starts", "row_tail_ends", "slice_axes4"], ["row_tail"]),
        node("Slice", ["patch_i", "row_head_starts", "row_head_ends", "slice_axes4"], ["row_head"]),
        node("Equal", ["row_tail", "row_head"], ["row_equal_cell"]),
        node("Cast", ["row_equal_cell"], ["row_equal_cell_i"], to=TensorProto.INT64),
        node("ReduceMin", ["row_equal_cell_i"], ["row_equal_prev_i"], axes=[3], keepdims=0),
        node("Equal", ["row_equal_prev_i", "zero_i"], ["row_diff_prev"]),
        node("Concat", ["false_row_pad", "row_diff_prev"], ["row_diff_pad"], axis=2),
        node("Equal", ["row_idx", "row_min_u"], ["row_first"]),
        node("Or", ["row_first", "row_diff_pad"], ["row_keep_seed"]),
        node("And", ["row_keep_seed", "row_in_bbox"], ["row_keep"]),
        node("Unsqueeze", ["row_keep"], ["row_keep_u"], axes=[3]),
        node("Cast", ["row_keep_u"], ["row_keep_i4"], to=TensorProto.INT64),
        node("Mul", ["patch_i", "row_keep_i4"], ["row_compressed_i"]),
        node("Slice", ["row_compressed_i", "col_tail_starts", "col_tail_ends", "slice_axes4"], ["col_tail"]),
        node("Slice", ["row_compressed_i", "col_head_starts", "col_head_ends", "slice_axes4"], ["col_head"]),
        node("Equal", ["col_tail", "col_head"], ["col_equal_cell"]),
        node("Cast", ["col_equal_cell"], ["col_equal_cell_i"], to=TensorProto.INT64),
        node("ReduceMin", ["col_equal_cell_i"], ["col_equal_prev_i"], axes=[2], keepdims=0),
        node("Equal", ["col_equal_prev_i", "zero_i"], ["col_diff_prev"]),
        node("Concat", ["false_col_pad", "col_diff_prev"], ["col_diff_pad"], axis=2),
        node("Equal", ["col_idx", "col_min_u"], ["col_first"]),
        node("Or", ["col_first", "col_diff_pad"], ["col_keep_seed"]),
        node("And", ["col_keep_seed", "col_in_bbox"], ["col_keep"]),
        node("Unsqueeze", ["col_keep"], ["col_keep_u"], axes=[2]),
        node("Cast", ["col_keep_u"], ["col_keep_i4"], to=TensorProto.INT64),
        node("Mul", ["row_compressed_i", "col_keep_i4"], ["compressed_i"]),
        node("ReduceSum", ["compressed_i"], ["compressed_cells0"], axes=[2, 3], keepdims=0),
        node("Cast", ["row_keep"], ["row_keep_i"], to=TensorProto.INT64),
        node("Cast", ["col_keep"], ["col_keep_i"], to=TensorProto.INT64),
        node("ReduceSum", ["row_keep_i"], ["compressed_rows0"], axes=[2], keepdims=0),
        node("ReduceSum", ["col_keep_i"], ["compressed_cols0"], axes=[2], keepdims=0),
        node("Mul", ["compressed_cells0", "valid_nonbg_i"], ["compressed_cells"]),
        node("Mul", ["compressed_rows0", "valid_nonbg_i"], ["compressed_rows"]),
        node("Mul", ["compressed_cols0", "valid_nonbg_i"], ["compressed_cols"]),
        node("Mul", ["compressed_rows", "compressed_cols"], ["compressed_area"]),
        node("Cast", ["compressed_cells"], ["compressed_cells_f"], to=TensorProto.FLOAT),
        node("Cast", ["compressed_area"], ["compressed_area_f"], to=TensorProto.FLOAT),
        node("Where", ["valid_nonbg_b", "compressed_area_f", "one_f"], ["safe_area_f"]),
        node("Div", ["compressed_cells_f", "safe_area_f"], ["compressed_density_raw"]),
        node("Where", ["valid_nonbg_b", "compressed_density_raw", "zero_f"], ["compressed_density"]),
        node("ReduceSum", ["compressed_i"], ["row_sums"], axes=[3], keepdims=0),
        node("Cast", ["row_sums"], ["row_sums_f"], to=TensorProto.FLOAT),
        node("Where", ["row_keep", "row_sums", "big_i"], ["row_sums_or_big"]),
        node("ReduceMax", ["row_sums"], ["row_sums_max"], axes=[2], keepdims=0),
        node("ReduceMin", ["row_sums_or_big"], ["row_sums_min0"], axes=[2], keepdims=0),
        node("Where", ["valid_nonbg_b", "row_sums_min0", "zero_i"], ["row_sums_min"]),
        node("Sub", ["row_sums_max", "row_sums_min"], ["compressed_row_range0"]),
        node("Mul", ["compressed_row_range0", "valid_nonbg_i"], ["compressed_row_range"]),
        node("Cast", ["compressed_rows"], ["compressed_rows_f"], to=TensorProto.FLOAT),
        node("Where", ["valid_nonbg_b", "compressed_rows_f", "one_f"], ["safe_rows_f"]),
        node("ReduceSum", ["row_sums_f"], ["row_sum_total"], axes=[2], keepdims=0),
        node("Div", ["row_sum_total", "safe_rows_f"], ["row_mean"]),
        node("Mul", ["row_sums_f", "row_sums_f"], ["row_sums_sq"]),
        node("ReduceSum", ["row_sums_sq"], ["row_sq_total"], axes=[2], keepdims=0),
        node("Div", ["row_sq_total", "safe_rows_f"], ["row_sq_mean"]),
        node("Mul", ["row_mean", "row_mean"], ["row_mean_sq"]),
        node("Sub", ["row_sq_mean", "row_mean_sq"], ["compressed_row_var_raw"]),
        node("Where", ["valid_nonbg_b", "compressed_row_var_raw", "zero_f"], ["compressed_row_var"]),
    ]

    graph = helper.make_graph(
        nodes,
        "task319_compress_feature_probe",
        [value_info("input", TensorProto.FLOAT, [1, CHANNELS, H, W])],
        [
            value_info("color_counts", TensorProto.INT64, [1, CHANNELS]),
            value_info("compressed_cells", TensorProto.INT64, [1, CHANNELS]),
            value_info("compressed_rows", TensorProto.INT64, [1, CHANNELS]),
            value_info("compressed_cols", TensorProto.INT64, [1, CHANNELS]),
            value_info("compressed_density", TensorProto.FLOAT, [1, CHANNELS]),
            value_info("compressed_row_range", TensorProto.INT64, [1, CHANNELS]),
            value_info("compressed_row_var", TensorProto.FLOAT, [1, CHANNELS]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def python_features(inp: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.zeros((1, CHANNELS), dtype=np.int64)
    compressed_cells = np.zeros((1, CHANNELS), dtype=np.int64)
    compressed_rows = np.zeros((1, CHANNELS), dtype=np.int64)
    compressed_cols = np.zeros((1, CHANNELS), dtype=np.int64)
    compressed_density = np.zeros((1, CHANNELS), dtype=np.float32)
    compressed_row_range = np.zeros((1, CHANNELS), dtype=np.int64)
    compressed_row_var = np.zeros((1, CHANNELS), dtype=np.float32)

    values, raw_counts = np.unique(inp, return_counts=True)
    for value, count in zip(values.tolist(), raw_counts.tolist()):
        counts[0, value] = count

    for obj in analyze_objects(inp):
        color = obj["color"]
        comp = obj["compressed"]
        row_sums = comp.sum(axis=1)
        compressed_cells[0, color] = int(comp.sum())
        compressed_rows[0, color] = int(comp.shape[0])
        compressed_cols[0, color] = int(comp.shape[1])
        compressed_density[0, color] = float(comp.sum() / comp.size)
        compressed_row_range[0, color] = int(row_sums.max() - row_sums.min())
        compressed_row_var[0, color] = float(np.var(row_sums))

    return {
        "color_counts": counts,
        "compressed_cells": compressed_cells,
        "compressed_rows": compressed_rows,
        "compressed_cols": compressed_cols,
        "compressed_density": compressed_density,
        "compressed_row_range": compressed_row_range,
        "compressed_row_var": compressed_row_var,
    }


def verify_model(model_path: pathlib.Path, comp_dir: pathlib.Path) -> None:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    examples = all_examples(TASK, comp_dir)
    names = [o.name for o in sess.get_outputs()]
    for idx, example in enumerate(examples):
        inp = np.array(example["input"], dtype=np.int8)
        expected = python_features(inp)
        outputs = dict(zip(names, sess.run(None, {"input": encode_grid(example["input"])})))
        for name in expected:
            if expected[name].dtype.kind == "f":
                if not np.allclose(outputs[name], expected[name], atol=1e-6):
                    raise AssertionError(f"{name} mismatch at example {idx}")
            else:
                if not np.array_equal(outputs[name], expected[name]):
                    raise AssertionError(f"{name} mismatch at example {idx}")
    print(f"task{TASK:03d} compress probe verified on {len(examples)}/{len(examples)} examples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--output", default="submissions/handbuilds/task319_compress_features.onnx")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    onnx.save(model, out_path)
    print(f"saved {out_path}")
    print(f"nodes {len(model.graph.node)}")
    print(f"file_size {out_path.stat().st_size}")
    if args.verify:
        verify_model(out_path, pathlib.Path(args.comp_dir))


if __name__ == "__main__":
    main()
