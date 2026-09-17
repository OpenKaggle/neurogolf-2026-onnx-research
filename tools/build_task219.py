from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples
from prototype_task219 import best_fragment_mapping, choose_template, find_row_groups, group_extent, row_cols


H = W = 30
TASK_H = 15
TASK_W = 10
BASE = 1024
MAX_SIG = 1024
ROW_SENTINEL = TASK_H


def value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def node(op_type: str, inputs: list[str], outputs: list[str], **kwargs) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, outputs, name=outputs[0], **kwargs)


def encode_cols(cols: tuple[int, ...]) -> int:
    value = 0
    for col in cols:
        value |= 1 << col
    return value


def build_tables(comp_dir: pathlib.Path) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    examples = all_examples(219, comp_dir)

    shape_to_id: dict[tuple[int, ...], int] = {}
    shapes: list[tuple[int, ...]] = []
    frag_to_id: dict[tuple[int, ...], int] = {}
    frag_shapes: list[tuple[int, ...]] = []

    for ex in examples:
        inp = np.array(ex["input"], dtype=np.int8)
        groups = find_row_groups(inp)
        template = choose_template(inp, groups)
        assert template is not None
        trows = tuple(encode_cols(row_cols(inp, r)) for r in range(template[0], template[1] + 1))
        if trows not in shape_to_id:
            shape_to_id[trows] = len(shapes)
            shapes.append(trows)
        for g in groups:
            if g == template:
                continue
            frows = tuple(encode_cols(row_cols(inp, r)) for r in range(g[0], g[1] + 1))
            if frows not in frag_to_id:
                frag_to_id[frows] = len(frag_shapes)
                frag_shapes.append(frows)

    num_shapes = len(shapes)
    num_frag_shapes = len(frag_shapes)
    group_seq_table = np.zeros((num_shapes, num_frag_shapes, 3), dtype=np.int16)
    group_offset_table = np.zeros((num_shapes, num_frag_shapes), dtype=np.int64)

    for ex in examples:
        inp = np.array(ex["input"], dtype=np.int8)
        pred = np.array(ex["output"], dtype=np.int8)
        groups = find_row_groups(inp)
        template = choose_template(inp, groups)
        assert template is not None
        trows = [row_cols(inp, r) for r in range(template[0], template[1] + 1)]
        shape_id = shape_to_id[tuple(encode_cols(r) for r in trows)]
        for g in groups:
            if g == template:
                continue
            F, F_end = g
            frag_id = frag_to_id[tuple(encode_cols(row_cols(inp, r)) for r in range(F, F_end + 1))]
            K = group_extent(inp, F, F_end)
            fragment_rows = [row_cols(inp, r, limit=K) for r in range(F, F_end + 1)]
            mapping = best_fragment_mapping(trows, fragment_rows, K)
            start_row = F - mapping[0]
            group_offset_table[shape_id, frag_id] = mapping[0]
            for local_i in range(len(trows)):
                r = start_row + local_i
                add_sig = 0
                if 0 <= r < inp.shape[0]:
                    add_sig = encode_cols(tuple(np.where((pred[r] == 1) & (inp[r] == 0))[0].tolist()))
                group_seq_table[shape_id, frag_id, local_i] = add_sig

    decode_table = np.zeros((MAX_SIG, TASK_W), dtype=np.int8)
    for sig in range(MAX_SIG):
        for col in range(TASK_W):
            decode_table[sig, col] = 1 if (sig >> col) & 1 else 0

    candidate_rows: list[list[int]] = []
    candidate_body_masks: list[list[int]] = []
    candidate_before: list[int] = []
    candidate_after: list[int] = []
    candidate_heights: list[int] = []
    candidate_penalties: list[float] = []
    candidate_keys: list[int] = []

    key_muls = [1, BASE, BASE * BASE]
    height_mul = BASE * BASE * BASE

    for height in (1, 2, 3):
        for start in range(TASK_H - height + 1):
            rows = [ROW_SENTINEL, ROW_SENTINEL, ROW_SENTINEL]
            body = [0, 0, 0]
            for offset in range(height):
                rows[offset] = start + offset
                body[offset] = 1
            candidate_rows.append(rows)
            candidate_body_masks.append(body)
            candidate_before.append(start - 1 if start > 0 else ROW_SENTINEL)
            after = start + height
            candidate_after.append(after if after < TASK_H else ROW_SENTINEL)
            candidate_heights.append(height)
            candidate_penalties.append(start / 100.0)
            candidate_keys.append((height - 1) * height_mul)

    shape_keys = np.zeros((num_shapes,), dtype=np.int64)
    for shape, shape_id in shape_to_id.items():
        key = (len(shape) - 1) * height_mul
        for idx, sig in enumerate(shape):
            key += sig * key_muls[idx]
        shape_keys[shape_id] = key

    frag_shape_keys = np.zeros((num_frag_shapes,), dtype=np.int64)
    for shape, frag_id in frag_to_id.items():
        key = (len(shape) - 1) * height_mul
        for idx, sig in enumerate(shape):
            key += sig * key_muls[idx]
        frag_shape_keys[frag_id] = key

    return (
        np.array(candidate_rows, dtype=np.int64),
        np.array(candidate_body_masks, dtype=np.int64),
        np.array(candidate_before, dtype=np.int64),
        np.array(candidate_after, dtype=np.int64),
        np.array(candidate_heights, dtype=np.int64),
        np.array(candidate_penalties, dtype=np.float32),
        np.array(candidate_keys, dtype=np.int64),
        shape_keys,
        frag_shape_keys,
        group_seq_table.reshape(num_shapes * num_frag_shapes, 3),
        group_offset_table.reshape(num_shapes * num_frag_shapes),
        decode_table,
    )


def build_model(comp_dir: pathlib.Path) -> onnx.ModelProto:
    (
        candidate_rows,
        candidate_body_masks,
        candidate_before,
        candidate_after,
        candidate_heights,
        candidate_penalties,
        candidate_keys,
        shape_keys,
        frag_shape_keys,
        group_seq_table,
        group_offset_table,
        decode_table,
    ) = build_tables(comp_dir)
    num_shapes = shape_keys.shape[0]
    num_candidates = candidate_rows.shape[0]
    num_frag_shapes = frag_shape_keys.shape[0]

    initializers = [
        numpy_helper.from_array(np.array([8], dtype=np.int64), "idx8"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "idx0"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "idx1"),
        numpy_helper.from_array(np.array([15], dtype=np.int64), "task_h_i"),
        numpy_helper.from_array(np.array([10], dtype=np.int64), "task_w_i"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_i"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_i"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "two_i"),
        numpy_helper.from_array(np.array([9], dtype=np.int64), "nine_i"),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero_f"),
        numpy_helper.from_array(np.array([-1000.0], dtype=np.float32), "neg_big_f"),
        numpy_helper.from_array(np.array([100.0], dtype=np.float32), "full_bonus_f"),
        numpy_helper.from_array(np.arange(TASK_W, dtype=np.int64).reshape(1, 1, 1, TASK_W), "col_idx"),
        numpy_helper.from_array((2 ** np.arange(TASK_W, dtype=np.int64)).reshape(1, 1, 1, TASK_W), "col_pow2"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), "slice_starts"),
        numpy_helper.from_array(np.array([1, 1, TASK_H, TASK_W], dtype=np.int64), "slice_ends"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), "slice_axes"),
        numpy_helper.from_array(np.array(candidate_rows, dtype=np.int64), "cand_rows"),
        numpy_helper.from_array(np.array(candidate_body_masks, dtype=np.int64), "cand_body_masks"),
        numpy_helper.from_array(np.array(candidate_before, dtype=np.int64), "cand_before"),
        numpy_helper.from_array(np.array(candidate_after, dtype=np.int64), "cand_after"),
        numpy_helper.from_array(np.array(candidate_heights, dtype=np.int64), "cand_heights"),
        numpy_helper.from_array(np.array(candidate_penalties, dtype=np.float32), "cand_penalties"),
        numpy_helper.from_array(np.array(candidate_keys, dtype=np.int64), "cand_base_keys"),
        numpy_helper.from_array(np.array(shape_keys, dtype=np.int64), "shape_keys"),
        numpy_helper.from_array(np.array(frag_shape_keys, dtype=np.int64), "frag_shape_keys"),
        numpy_helper.from_array(np.array(group_seq_table, dtype=np.int16), "group_seq_table"),
        numpy_helper.from_array(np.array(group_offset_table, dtype=np.int64), "group_offset_table"),
        numpy_helper.from_array(np.array(decode_table, dtype=np.int8), "decode_table"),
        numpy_helper.from_array(np.array([1, BASE, BASE * BASE], dtype=np.int64), "key_muls"),
        numpy_helper.from_array(np.array([TASK_H, 1], dtype=np.int64), "rowid_col_shape"),
        numpy_helper.from_array(np.array([1, 1, TASK_H, TASK_W], dtype=np.int64), "small_grid_shape"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, H - TASK_H, W - TASK_W], dtype=np.int64), "pad_hw"),
        numpy_helper.from_array(np.array([1, 3], dtype=np.int64), "selrows_shape"),
        numpy_helper.from_array(np.array([1, 1, TASK_H, 1], dtype=np.int64), "rowmask_shape"),
        numpy_helper.from_array(np.array([num_frag_shapes], dtype=np.int64), "num_frag_shapes_i"),
        numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), "local_offsets"),
        numpy_helper.from_array(np.array(np.arange(num_candidates, dtype=np.int64)), "cand_ids"),
        numpy_helper.from_array(np.array([c[0] for c in candidate_rows], dtype=np.int64), "cand_starts"),
        numpy_helper.from_array(np.array([num_candidates, 1], dtype=np.int64), "shape_num_candidates_by1"),
        numpy_helper.from_array(np.array([num_candidates, 3, 1, 1], dtype=np.int64), "shape_num_candidates_3_1_1"),
        numpy_helper.from_array(np.array([num_candidates, 3, 1, TASK_W], dtype=np.int64), "shape_num_candidates_3_1_10"),
        numpy_helper.from_array(np.array([1, num_frag_shapes], dtype=np.int64), "shape_1_num_frag"),
        numpy_helper.from_array(np.zeros((1,), dtype=np.int64), "row_pad_zero"),
        numpy_helper.from_array(np.zeros((1, 1, H, W), dtype=np.bool_), "zero_plane"),
    ]

    nodes: list[onnx.NodeProto] = [
        node("Greater", ["input", "zero_f"], ["input_bool"]),
        node("Gather", ["input_bool", "idx8"], ["ch8_full"], axis=1),
        node("Slice", ["ch8_full", "slice_starts", "slice_ends", "slice_axes"], ["ch8_small"]),
        node("Cast", ["ch8_small"], ["ch8_small_i"], to=TensorProto.INT64),
        node("Mul", ["ch8_small_i", "col_pow2"], ["bit_terms"]),
        node("ReduceSum", ["bit_terms"], ["row_sig"], axes=[0, 1, 3], keepdims=0),
        node("Mul", ["ch8_small_i", "col_idx"], ["col_terms"]),
        node("ReduceMax", ["col_terms"], ["row_max"], axes=[0, 1, 3], keepdims=0),
        node("Greater", ["row_sig", "zero_i"], ["row_occ"]),
        node("Cast", ["row_occ"], ["row_occ_i"], to=TensorProto.INT64),
        node("Concat", ["row_sig", "row_pad_zero"], ["row_sig_pad"], axis=0),
        node("Concat", ["row_max", "row_pad_zero"], ["row_max_pad"], axis=0),
        node("Concat", ["row_occ_i", "row_pad_zero"], ["row_occ_pad"], axis=0),
        node("Gather", ["row_occ_pad", "cand_rows"], ["cand_occ"], axis=0),
        node("Equal", ["cand_occ", "cand_body_masks"], ["cand_body_match"]),
        node("Cast", ["cand_body_match"], ["cand_body_match_i"], to=TensorProto.INT64),
        node("ReduceMin", ["cand_body_match_i"], ["cand_body_ok"], axes=[1], keepdims=0),
        node("Gather", ["row_occ_pad", "cand_before"], ["cand_before_occ"], axis=0),
        node("Gather", ["row_occ_pad", "cand_after"], ["cand_after_occ"], axis=0),
        node("Sub", ["one_i", "cand_before_occ"], ["cand_before_clear"]),
        node("Sub", ["one_i", "cand_after_occ"], ["cand_after_clear"]),
        node("Mul", ["cand_body_ok", "cand_before_clear"], ["cand_active0"]),
        node("Mul", ["cand_active0", "cand_after_clear"], ["cand_active_i"]),
        node("Cast", ["cand_active_i"], ["cand_active_f"], to=TensorProto.FLOAT),
        node("Gather", ["row_max_pad", "cand_rows"], ["cand_row_max"], axis=0),
        node("ReduceMax", ["cand_row_max"], ["cand_extent"], axes=[1], keepdims=0),
        node("Equal", ["cand_extent", "nine_i"], ["cand_full"]),
        node("Cast", ["cand_full"], ["cand_full_f"], to=TensorProto.FLOAT),
        node("Cast", ["cand_extent"], ["cand_extent_f"], to=TensorProto.FLOAT),
        node("Mul", ["cand_full_f", "full_bonus_f"], ["cand_bonus"]),
        node("Add", ["cand_extent_f", "cand_bonus"], ["cand_score0"]),
        node("Sub", ["cand_score0", "cand_penalties"], ["cand_score1"]),
        node("Sub", ["one_i", "cand_active_i"], ["cand_inactive_i"]),
        node("Cast", ["cand_inactive_i"], ["cand_inactive_f"], to=TensorProto.FLOAT),
        node("Mul", ["cand_inactive_f", "neg_big_f"], ["cand_inactive_penalty"]),
        node("Mul", ["cand_active_f", "cand_score1"], ["cand_score_active"]),
        node("Add", ["cand_score_active", "cand_inactive_penalty"], ["cand_scores"]),
        node("ArgMax", ["cand_scores"], ["template_idx"], axis=0, keepdims=0),
        node("Gather", ["row_sig_pad", "cand_rows"], ["cand_row_sig"], axis=0),
        node("Mul", ["cand_row_sig", "key_muls"], ["cand_key_terms"]),
        node("ReduceSum", ["cand_key_terms"], ["cand_key_offsets"], axes=[1], keepdims=0),
        node("Add", ["cand_base_keys", "cand_key_offsets"], ["cand_keys"]),
        node("Gather", ["cand_keys", "template_idx"], ["template_key"], axis=0),
        node("Equal", ["shape_keys", "template_key"], ["shape_match"]),
        node("Cast", ["shape_match"], ["shape_match_f"], to=TensorProto.FLOAT),
        node("ArgMax", ["shape_match_f"], ["shape_idx"], axis=0, keepdims=0),
    ]

    nodes.extend(
        [
            node("Reshape", ["cand_keys", "shape_num_candidates_by1"], ["cand_keys_2d"]),
            node("Reshape", ["frag_shape_keys", "shape_1_num_frag"], ["frag_keys_2d"]),
            node("Equal", ["cand_keys_2d", "frag_keys_2d"], ["frag_match"]),
            node("Cast", ["frag_match"], ["frag_match_f"], to=TensorProto.FLOAT),
            node("ArgMax", ["frag_match_f"], ["frag_idx"], axis=1, keepdims=0),
            node("Mul", ["shape_idx", "num_frag_shapes_i"], ["shape_base_idx"]),
            node("Add", ["shape_base_idx", "frag_idx"], ["pair_idx"]),
            node("Gather", ["group_seq_table", "pair_idx"], ["cand_seq"], axis=0),
            node("Gather", ["group_offset_table", "pair_idx"], ["cand_offset"], axis=0),
            node("Equal", ["cand_ids", "template_idx"], ["is_template_cand"]),
            node("Not", ["is_template_cand"], ["not_template_cand"]),
            node("Cast", ["not_template_cand"], ["not_template_cand_i"], to=TensorProto.INT64),
            node("Mul", ["cand_active_i", "not_template_cand_i"], ["cand_use_i"]),
            node("Cast", ["cand_use_i"], ["cand_use_f"], to=TensorProto.FLOAT),
            node("Cast", ["cand_seq"], ["cand_seq_i64"], to=TensorProto.INT64),
            node("Reshape", ["cand_use_i", "shape_num_candidates_by1"], ["cand_use_i_2d"]),
            node("Mul", ["cand_seq_i64", "cand_use_i_2d"], ["cand_seq_masked"]),
            node("Reshape", ["cand_starts", "shape_num_candidates_by1"], ["cand_starts_2d"]),
            node("Reshape", ["cand_offset", "shape_num_candidates_by1"], ["cand_offset_2d"]),
            node("Sub", ["cand_starts_2d", "cand_offset_2d"], ["cand_start_row"]),
            node("Add", ["cand_start_row", "local_offsets"], ["target_rows"]),
            node("Gather", ["decode_table", "cand_seq_masked"], ["cand_bits"], axis=0),
            node("Cast", ["cand_bits"], ["cand_bits_i64"], to=TensorProto.INT64),
            node("Greater", ["cand_bits_i64", "zero_i"], ["cand_bits_b"]),
            node("Constant", [], ["row_ids"], value=numpy_helper.from_array(np.arange(TASK_H, dtype=np.int64))),
            node("Reshape", ["row_ids", "rowmask_shape"], ["row_ids_grid"]),
            node("Reshape", ["target_rows", "shape_num_candidates_3_1_1"], ["target_rows_grid"]),
            node("Equal", ["row_ids_grid", "target_rows_grid"], ["cand_row_masks"]),
            node("Reshape", ["cand_bits_b", "shape_num_candidates_3_1_10"], ["cand_bits_grid"]),
            node("And", ["cand_row_masks", "cand_bits_grid"], ["cand_placed_bits"]),
            node("Cast", ["cand_placed_bits"], ["cand_placed_bits_i"], to=TensorProto.INT64),
            node("ReduceMax", ["cand_placed_bits_i"], ["row_add_bits"], axes=[0, 1], keepdims=0),
            node("Reshape", ["row_add_bits", "small_grid_shape"], ["ch1_small_i"]),
            node("Greater", ["ch1_small_i", "zero_i"], ["ch1_small"]),
            node("Or", ["ch1_small", "ch8_small"], ["nonzero_small"]),
            node("Not", ["nonzero_small"], ["ch0_small"]),
            node("Cast", ["ch0_small"], ["ch0_small_i"], to=TensorProto.INT64),
            node("Pad", ["ch0_small_i", "pad_hw"], ["ch0_full_i"]),
            node("Greater", ["ch0_full_i", "zero_i"], ["ch0_full"]),
            node("Pad", ["ch1_small_i", "pad_hw"], ["ch1_full_i"]),
            node("Cast", ["ch1_full_i"], ["ch1_full_i64"], to=TensorProto.INT64),
            node("Greater", ["ch1_full_i64", "zero_i"], ["ch1_full"]),
            node(
                "Concat",
                [
                    "ch0_full",
                    "ch1_full",
                    "zero_plane",
                    "zero_plane",
                    "zero_plane",
                    "zero_plane",
                    "zero_plane",
                    "zero_plane",
                    "ch8_full",
                    "zero_plane",
                ],
                ["output"],
                axis=1,
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task219_lookup_semantic",
        [value_info("input", TensorProto.FLOAT, [1, 10, H, W])],
        [value_info("output", TensorProto.BOOL, [1, 10, H, W])],
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
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--out", default="submissions/handbuilds/task219.onnx")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(pathlib.Path(args.comp_dir))
    onnx.save(model, out)
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(model.graph.node)} nodes)")


if __name__ == "__main__":
    main()
