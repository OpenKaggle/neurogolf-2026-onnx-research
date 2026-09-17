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
PACK = 5
TASK = 319


def value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def node(op_type: str, inputs: list[str], outputs: list[str], **kwargs) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, outputs, name=outputs[0], **kwargs)


def build_model() -> onnx.ModelProto:
    initializers = [
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero_f"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_i"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_i"),
        numpy_helper.from_array(np.array([31], dtype=np.int64), "big_i"),
        numpy_helper.from_array(np.array([100], dtype=np.int64), "rank_base_i"),
        numpy_helper.from_array(np.arange(CHANNELS, dtype=np.int64).reshape(1, CHANNELS), "color_idx"),
        numpy_helper.from_array(np.array([PACK], dtype=np.int64), "pack_i"),
        numpy_helper.from_array(np.arange(H, dtype=np.int64).reshape(1, 1, H), "row_idx"),
        numpy_helper.from_array(np.arange(W, dtype=np.int64).reshape(1, 1, W), "col_idx"),
        numpy_helper.from_array(np.array([0, 0, 1, 0], dtype=np.int64), "row_tail_starts"),
        numpy_helper.from_array(np.array([1, CHANNELS, H, W], dtype=np.int64), "row_tail_ends"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), "row_head_starts"),
        numpy_helper.from_array(np.array([1, CHANNELS, H - 1, W], dtype=np.int64), "row_head_ends"),
        numpy_helper.from_array(np.array([0, 0, 0, 1], dtype=np.int64), "col_tail_starts"),
        numpy_helper.from_array(np.array([1, CHANNELS, H, W], dtype=np.int64), "col_tail_ends"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), "col_head_starts"),
        numpy_helper.from_array(np.array([1, CHANNELS, H, W - 1], dtype=np.int64), "col_head_ends"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), "slice_axes4"),
        numpy_helper.from_array(np.zeros((1, CHANNELS, 1), dtype=np.bool_), "false_row_pad"),
        numpy_helper.from_array(np.zeros((1, CHANNELS, 1), dtype=np.bool_), "false_col_pad"),
        numpy_helper.from_array(np.array([1, CHANNELS, PACK, W], dtype=np.int64), "shape_row_expand"),
        numpy_helper.from_array(np.array([1, CHANNELS, PACK, PACK], dtype=np.int64), "shape_col_expand"),
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
        node("Unsqueeze", ["valid_nonbg_i"], ["valid_nonbg_row_u"], axes=[2]),
        node("Unsqueeze", ["valid_nonbg_i"], ["valid_nonbg_col_u"], axes=[2]),
        node("Unsqueeze", ["valid_nonbg_i"], ["valid_nonbg_pack_u"], axes=[2, 3]),
        node("ReduceMax", ["mask_i"], ["row_occ_i"], axes=[3], keepdims=0),
        node("ReduceMax", ["mask_i"], ["col_occ_i"], axes=[2], keepdims=0),
        node("Cast", ["row_occ_i"], ["row_occ_b"], to=TensorProto.BOOL),
        node("Cast", ["col_occ_i"], ["col_occ_b"], to=TensorProto.BOOL),
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
        # rank rows/cols so the first PACK kept ones can be gathered compactly
        node("Sub", ["rank_base_i", "row_idx"], ["row_rank_raw"]),
        node("Where", ["row_keep", "row_rank_raw", "zero_i"], ["row_rank_masked"]),
        node("TopK", ["row_rank_masked", "pack_i"], ["row_rank_values", "row_topk_idx_raw"], axis=2),
        node("Greater", ["row_rank_values", "zero_i"], ["row_topk_valid_b"]),
        node("Cast", ["row_topk_valid_b"], ["row_topk_valid_i"], to=TensorProto.INT64),
        node("Unsqueeze", ["row_topk_valid_b"], ["row_topk_valid_u"], axes=[3]),
        node("Cast", ["row_topk_valid_u"], ["row_topk_valid_i4"], to=TensorProto.INT64),
        node("Mul", ["row_topk_idx_raw", "row_topk_valid_i"], ["row_topk_idx0"]),
        node("Mul", ["row_topk_idx0", "valid_nonbg_row_u"], ["row_topk_idx"]),
        node("Sub", ["rank_base_i", "col_idx"], ["col_rank_raw"]),
        node("Where", ["col_keep", "col_rank_raw", "zero_i"], ["col_rank_masked"]),
        node("TopK", ["col_rank_masked", "pack_i"], ["col_rank_values", "col_topk_idx_raw"], axis=2),
        node("Greater", ["col_rank_values", "zero_i"], ["col_topk_valid_b"]),
        node("Cast", ["col_topk_valid_b"], ["col_topk_valid_i"], to=TensorProto.INT64),
        node("Unsqueeze", ["col_topk_valid_b"], ["col_topk_valid_u"], axes=[2]),
        node("Cast", ["col_topk_valid_u"], ["col_topk_valid_i4"], to=TensorProto.INT64),
        node("Mul", ["col_topk_idx_raw", "col_topk_valid_i"], ["col_topk_idx0"]),
        node("Mul", ["col_topk_idx0", "valid_nonbg_col_u"], ["col_topk_idx"]),
        node("Unsqueeze", ["row_topk_idx"], ["row_topk_idx_u"], axes=[3]),
        node("Expand", ["row_topk_idx_u", "shape_row_expand"], ["row_gather_idx"]),
        node("GatherElements", ["row_compressed_i", "row_gather_idx"], ["packed_rows"], axis=2),
        node("Mul", ["packed_rows", "row_topk_valid_i4"], ["packed_rows_valid"]),
        node("Unsqueeze", ["col_topk_idx"], ["col_topk_idx_u"], axes=[2]),
        node("Expand", ["col_topk_idx_u", "shape_col_expand"], ["col_gather_idx"]),
        node("GatherElements", ["packed_rows_valid", "col_gather_idx"], ["packed_compressed0"], axis=3),
        node("Mul", ["packed_compressed0", "col_topk_valid_i4"], ["packed_compressed1"]),
        node("Mul", ["packed_compressed1", "valid_nonbg_pack_u"], ["packed_compressed"]),
    ]

    graph = helper.make_graph(
        nodes,
        "task319_packed_compressed_probe",
        [value_info("input", TensorProto.FLOAT, [1, CHANNELS, H, W])],
        [
            value_info("packed_compressed", TensorProto.INT64, [1, CHANNELS, PACK, PACK]),
            value_info("row_topk_idx", TensorProto.INT64, [1, CHANNELS, PACK]),
            value_info("col_topk_idx", TensorProto.INT64, [1, CHANNELS, PACK]),
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
    packed = np.zeros((1, CHANNELS, PACK, PACK), dtype=np.int64)
    row_idx = np.zeros((1, CHANNELS, PACK), dtype=np.int64)
    col_idx = np.zeros((1, CHANNELS, PACK), dtype=np.int64)
    for obj in analyze_objects(inp):
        color = obj["color"]
        comp = obj["compressed"]
        packed[0, color, : comp.shape[0], : comp.shape[1]] = comp
        r0, c0, r1, c1 = obj["bbox"]
        patch = (inp[r0 : r1 + 1, c0 : c1 + 1] == color).astype(np.int8)
        keep_rows = [r0]
        for local_row in range(1, patch.shape[0]):
            if not np.array_equal(patch[local_row], patch[local_row - 1]):
                keep_rows.append(r0 + local_row)
        keep_cols = [c0]
        for local_col in range(1, patch.shape[1]):
            if not np.array_equal(patch[:, local_col], patch[:, local_col - 1]):
                keep_cols.append(c0 + local_col)
        row_idx[0, color, : len(keep_rows)] = np.array(keep_rows[:PACK], dtype=np.int64)
        col_idx[0, color, : len(keep_cols)] = np.array(keep_cols[:PACK], dtype=np.int64)
    return {
        "packed_compressed": packed,
        "row_topk_idx": row_idx,
        "col_topk_idx": col_idx,
    }


def verify_model(model_path: pathlib.Path, comp_dir: pathlib.Path) -> None:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    examples = all_examples(TASK, comp_dir)
    for idx, example in enumerate(examples):
        expected = python_features(np.array(example["input"], dtype=np.int8))
        actual = dict(zip(names, sess.run(None, {"input": encode_grid(example["input"])})))
        for name in expected:
            if not np.array_equal(actual[name], expected[name]):
                raise AssertionError(f"{name} mismatch at example {idx}")
    print(f"task{TASK:03d} packed compressed probe verified on {len(examples)}/{len(examples)} examples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--output", default="submissions/handbuilds/task319_packed_compressed.onnx")
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
