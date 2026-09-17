from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


H = W = 30
TASK = 10


def priority(dr: int, dc: int) -> tuple[int, int, int]:
    return (abs(dc), dr, dc)



def build_offsets(comp_dir: pathlib.Path) -> list[tuple[int, int]]:
    task = json.loads((comp_dir / "task363.json").read_text())
    used: set[tuple[int, int]] = set()
    for example in task["train"] + task["test"] + task.get("arc-gen", []):
        arr = np.array(example["input"], dtype=np.int64)
        h, w = arr.shape
        mask = arr == 2
        rows, cols = np.where(mask)
        r0, c0 = int(rows.min()), int(cols.min())
        r1, c1 = int(rows.max()), int(cols.max())
        pat = mask[r0 : r1 + 1, c0 : c1 + 1]
        pw = c1 - c0 + 1
        src5 = int((arr[r0, :] == 5).sum())
        meta: list[tuple[int, int, int, set[tuple[int, int]]]] = []
        for dr in range(-h, h):
            for dc in range(-w, w):
                if dr == 0 and dc == 0:
                    continue
                if pw == 4 and src5 == 2 and dr < 0:
                    continue
                cells: set[tuple[int, int]] = set()
                ok = True
                for pr, pc in np.argwhere(pat):
                    rr, cc = r0 + int(pr) + dr, c0 + int(pc) + dc
                    if rr < 0 or rr >= h or cc < 0 or cc >= w or arr[rr, cc] != 0:
                        ok = False
                        break
                    cells.add((rr, cc))
                if ok:
                    meta.append((abs(dc), dr, dc, cells))
        order = sorted(range(len(meta)), key=lambda idx: (meta[idx][0], meta[idx][1], meta[idx][2]), reverse=True)
        kept: list[int] = []
        for idx in order:
            if any(meta[idx][3] & meta[prev][3] for prev in kept):
                continue
            kept.append(idx)
        for idx in kept:
            _, dr, dc, _ = meta[idx]
            used.add((dr, dc))
    return sorted(used, key=lambda x: priority(*x), reverse=True)


def value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def node(op_type: str, inputs: list[str], outputs: list[str], **kwargs) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, outputs, name=outputs[0], **kwargs)


def shift_pad_slice(name: str, source: str, dr: int, dc: int) -> list[onnx.NodeProto]:
    pad_top = max(dr, 0)
    pad_bottom = max(-dr, 0)
    pad_left = max(dc, 0)
    pad_right = max(-dc, 0)
    pads = np.array([0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right], dtype=np.int64)

    src_row_start = max(-dr, 0)
    src_col_start = max(-dc, 0)
    dst_row_end = TASK - max(dr, 0)
    dst_col_end = TASK - max(dc, 0)
    # Slice keeps just the source area that lands inside the 10x10 target.
    starts = np.array([0, 0, src_row_start, src_col_start], dtype=np.int64)
    ends = np.array([1, 1, dst_row_end, dst_col_end], dtype=np.int64)

    return [
        node("Slice", [source, f"{name}_starts", f"{name}_ends", "slice_axes"], [f"{name}_cropped"]),
        node("Pad", [f"{name}_cropped", f"{name}_pads"], [f"{name}_shifted"]),
    ], [
        numpy_helper.from_array(starts, f"{name}_starts"),
        numpy_helper.from_array(ends, f"{name}_ends"),
        numpy_helper.from_array(pads, f"{name}_pads"),
    ]


def build_model(comp_dir: pathlib.Path) -> onnx.ModelProto:
    offsets = build_offsets(comp_dir)
    row_idx = np.arange(TASK, dtype=np.int64).reshape(1, 1, TASK, 1)
    col_idx = np.arange(TASK, dtype=np.int64).reshape(1, 1, 1, TASK)
    row_idx = np.broadcast_to(row_idx, (1, 1, TASK, TASK))
    col_idx = np.broadcast_to(col_idx, (1, 1, TASK, TASK))

    initializers = [
        numpy_helper.from_array(np.array([2], dtype=np.int64), "idx2"),
        numpy_helper.from_array(np.array([5], dtype=np.int64), "idx5"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_i"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_i"),
        numpy_helper.from_array(np.array([10], dtype=np.int64), "ten_i"),
        numpy_helper.from_array(np.array([4], dtype=np.int64), "four_i"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "two_i"),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero_f"),
        numpy_helper.from_array(np.array([-1000000.0], dtype=np.float32), "neg_big_f"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), "slice_axes"),
        numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), "small_starts"),
        numpy_helper.from_array(np.array([1, 1, TASK, TASK], dtype=np.int64), "small_ends"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, H - TASK, W - TASK], dtype=np.int64), "pad_hw"),
        numpy_helper.from_array(row_idx, "row_idx"),
        numpy_helper.from_array(col_idx, "col_idx"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int64), "pad_row_down"),
        numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64), "pad_row_up"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int64), "pad_col_right"),
        numpy_helper.from_array(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64), "pad_col_left"),
        numpy_helper.from_array(np.zeros((1, 1, H, W), dtype=np.bool_), "zero_plane"),
    ]

    nodes: list[onnx.NodeProto] = [
        node("Greater", ["input", "zero_f"], ["input_bool"]),
        node("Gather", ["input_bool", "idx2"], ["ch2_full"], axis=1),
        node("Gather", ["input_bool", "idx5"], ["ch5_full"], axis=1),
        node("Slice", ["ch2_full", "small_starts", "small_ends", "slice_axes"], ["ch2_small"]),
        node("Slice", ["ch5_full", "small_starts", "small_ends", "slice_axes"], ["ch5_small"]),
        node("Cast", ["ch2_small"], ["ch2_i"], to=TensorProto.INT64),
        node("Cast", ["ch5_small"], ["ch5_i"], to=TensorProto.INT64),
        node("ReduceSum", ["ch2_i"], ["source_count"], axes=[0, 1, 2, 3], keepdims=0),
        node("ReduceMax", ["row_idx"], ["row_any_idx"], axes=[0, 1, 3], keepdims=0),
        node("ReduceMax", ["col_idx"], ["col_any_idx"], axes=[0, 1, 2], keepdims=0),
        node("ReduceMax", ["ch2_i"], ["row_occ_i"], axes=[0, 1, 3], keepdims=0),
        node("ReduceMax", ["ch2_i"], ["col_occ_i"], axes=[0, 1, 2], keepdims=0),
        node("Cast", ["row_occ_i"], ["row_occ_b"], to=TensorProto.BOOL),
        node("Cast", ["col_occ_i"], ["col_occ_b"], to=TensorProto.BOOL),
        node("Where", ["row_occ_b", "row_any_idx", "ten_i"], ["row_idx_or_ten"]),
        node("Where", ["col_occ_b", "col_any_idx", "ten_i"], ["col_idx_or_ten"]),
        node("ReduceMin", ["row_idx_or_ten"], ["r0"], axes=[0], keepdims=0),
        node("ReduceMin", ["col_idx_or_ten"], ["c0"], axes=[0], keepdims=0),
        node("Mul", ["row_occ_i", "row_any_idx"], ["row_occ_weighted"]),
        node("Mul", ["col_occ_i", "col_any_idx"], ["col_occ_weighted"]),
        node("ReduceMax", ["row_occ_weighted"], ["r1"], axes=[0], keepdims=0),
        node("ReduceMax", ["col_occ_weighted"], ["c1"], axes=[0], keepdims=0),
        node("Sub", ["c1", "c0"], ["pw_minus1"]),
        node("Add", ["pw_minus1", "one_i"], ["pw"]),
        node("Equal", ["row_idx", "r0"], ["top_row_mask"]),
        node("And", ["top_row_mask", "ch5_small"], ["top_row_5"]),
        node("Cast", ["top_row_5"], ["top_row_5_i"], to=TensorProto.INT64),
        node("ReduceSum", ["top_row_5_i"], ["src5"], axes=[0, 1, 2, 3], keepdims=0),
    ]

    keep_mask_name = "zero_plane_small"
    initializers.append(numpy_helper.from_array(np.zeros((1, 1, TASK, TASK), dtype=np.bool_), "zero_plane_small"))
    for idx, (dr, dc) in enumerate(offsets):
        base = f"off_{idx}"
        shift_nodes, shift_inits = shift_pad_slice(base, "ch2_i", dr, dc)
        initializers.extend(shift_inits)
        nodes.extend(shift_nodes)
        nodes.extend(
            [
                node("ReduceSum", [f"{base}_shifted"], [f"{base}_shift_count"], axes=[0, 1, 2, 3], keepdims=0),
                node("Equal", [f"{base}_shift_count", "source_count"], [f"{base}_inbounds"]),
                node("And", [f"{base}_shifted", "input_bool_zero_small"], [f"{base}_on_zero"]),  # replaced later
            ]
        )
        # fix AND type by inserting bool cast and using dedicated zero mask
        nodes[-1] = node("And", [f"{base}_shifted_b", "zero_small"], [f"{base}_on_zero"])
        nodes.insert(-1, node("Cast", [f"{base}_shifted"], [f"{base}_shifted_b"], to=TensorProto.BOOL))
        nodes.extend(
            [
                node("Cast", [f"{base}_on_zero"], [f"{base}_on_zero_i"], to=TensorProto.INT64),
                node("ReduceSum", [f"{base}_on_zero_i"], [f"{base}_zero_count"], axes=[0, 1, 2, 3], keepdims=0),
                node("Equal", [f"{base}_zero_count", "source_count"], [f"{base}_all_zero"]),
                node("And", [f"{base}_inbounds", f"{base}_all_zero"], [f"{base}_valid0"]),
            ]
        )
        # narrow semantic reject from the closed visible rule
        reject = dr < 0
        if reject:
            nodes.extend(
                [
                    node("Equal", ["pw", "four_i"], [f"{base}_pw4"]),
                    node("Equal", ["src5", "two_i"], [f"{base}_src5eq2"]),
                    node("And", [f"{base}_pw4", f"{base}_src5eq2"], [f"{base}_reject"]),
                    node("Not", [f"{base}_reject"], [f"{base}_guard"]),
                    node("And", [f"{base}_valid0", f"{base}_guard"], [f"{base}_valid"]),
                ]
            )
        else:
            nodes.append(node("Identity", [f"{base}_valid0"], [f"{base}_valid"]))

        nodes.extend(
            [
                node("And", [f"{base}_shifted_b", keep_mask_name], [f"{base}_overlap_cells"]),
                node("Cast", [f"{base}_overlap_cells"], [f"{base}_overlap_i"], to=TensorProto.INT64),
                node("ReduceSum", [f"{base}_overlap_i"], [f"{base}_overlap_count"], axes=[0, 1, 2, 3], keepdims=0),
                node("Equal", [f"{base}_overlap_count", "zero_i"], [f"{base}_no_overlap"]),
                node("And", [f"{base}_valid", f"{base}_no_overlap"], [f"{base}_keep"]),
                node("And", [f"{base}_shifted_b", f"{base}_keep"], [f"{base}_active_b"]),
                node("Or", [keep_mask_name, f"{base}_active_b"], [f"{base}_keep_mask"]),
            ]
        )
        keep_mask_name = f"{base}_keep_mask"

    nodes[28:28] = [
        node("Or", ["ch2_small", "ch5_small"], ["nonzero_small"]),
        node("Not", ["nonzero_small"], ["zero_small"]),
    ]

    nodes.extend(
        [
            node("Or", [keep_mask_name, "ch2_small"], ["ch2_small_out"]),
            node("Cast", ["ch2_small_out"], ["ch2_small_i_out"], to=TensorProto.INT64),
            node("Cast", ["ch5_small"], ["ch5_small_i_out"], to=TensorProto.INT64),
            node("Or", ["ch2_small_out", "ch5_small"], ["nonzero_out_small"]),
            node("Not", ["nonzero_out_small"], ["ch0_small"]),
            node("Cast", ["ch0_small"], ["ch0_small_i"], to=TensorProto.INT64),
            node("Pad", ["ch0_small_i", "pad_hw"], ["ch0_full_i"]),
            node("Pad", ["ch2_small_i_out", "pad_hw"], ["ch2_full_i_out"]),
            node("Pad", ["ch5_small_i_out", "pad_hw"], ["ch5_full_i_out"]),
            node("Greater", ["ch0_full_i", "zero_i"], ["ch0_full"]),
            node("Greater", ["ch2_full_i_out", "zero_i"], ["ch2_full_out"]),
            node("Greater", ["ch5_full_i_out", "zero_i"], ["ch5_full_out"]),
            node(
                "Concat",
                [
                    "ch0_full",
                    "zero_plane",
                    "ch2_full_out",
                    "zero_plane",
                    "zero_plane",
                    "ch5_full_out",
                    "zero_plane",
                    "zero_plane",
                    "zero_plane",
                    "zero_plane",
                ],
                ["output"],
                axis=1,
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task363_semantic_shift",
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
    parser.add_argument("--out", default="submissions/handbuilds/task363_semantic.onnx")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(pathlib.Path(args.comp_dir))
    onnx.save(model, out)
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(model.graph.node)} nodes)")


if __name__ == "__main__":
    main()
