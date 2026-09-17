"""task233 ONNX builder — color propagation from patches outside the main wall.

Algorithm:
  1. Extract ch2 (color-2 "wall" cells)
  2. Component-label ch2 via 30× MaxPool propagation
  3. Largest component → main wall
  4. Bbox of main component, inside-bbox mask
  5. Hole cells = all non-main cells inside bbox
  6. Outside cells = non-ch2, non-bbox cells with colors 1-9
  7. Propagate outside colors inward via MaxPool (15×)
  8. Cells inside bbox get the nearest outside color
  9. Assign colors to hole cells
  10. Output = outside_bbox + inside_bbox_wall + hole_fill
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper as oh, numpy_helper as onh


H_W = 30
BIG = 999.0
POOL_ITERS = 30
PROP_ITERS = 90


def _n(op: str, ins: list[str], outs: list[str], **attrs) -> onnx.NodeProto:
    return oh.make_node(op, ins, outs, name=outs[0], **attrs)


def _arr(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return onh.from_array(arr, name)


def build(out_path: pathlib.Path) -> onnx.ModelProto:
    inits = []
    nodes = []

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    for name, val in [("BIG", BIG), ("NEG_BIG", -BIG),
                      ("ZERO", 0.0), ("ONE", 1.0), ("TWO", 2.0)]:
        inits.append(_arr(name, np.array(val, dtype=np.float32)))

    inits.append(_arr("axes_all", np.array([0, 1, 2, 3], dtype=np.int64)))
    inits.append(_arr("steps_all", np.array([1, 1, 1, 1], dtype=np.int64)))
    inits.append(_arr("starts_ch2", np.array([0, 2, 0, 0], dtype=np.int64)))
    inits.append(_arr("ends_ch2", np.array([1, 3, H_W, H_W], dtype=np.int64)))

    # neg_idx — negative unique index per cell for component labeling
    neg_idx = -np.arange(H_W * H_W, dtype=np.float32).reshape(1, 1, H_W, H_W)
    inits.append(_arr("neg_idx", neg_idx))

    # Row / col coordinate maps
    row_coord = np.arange(H_W, dtype=np.float32).reshape(1, 1, H_W, 1).repeat(H_W, axis=3)
    col_coord = np.arange(H_W, dtype=np.float32).reshape(1, 1, 1, H_W).repeat(H_W, axis=2)
    inits.append(_arr("row_coord", row_coord))
    inits.append(_arr("col_coord", col_coord))

    inits.append(_arr("hist_shape", np.array([1000], dtype=np.int64)))
    inits.append(_arr("axes_unsq", np.array([4], dtype=np.int64)))
    inits.append(_arr("axis0_i64", np.array([0], dtype=np.int64)))
    inits.append(_arr("axis1_i64", np.array([1], dtype=np.int64)))
    inits.append(_arr("axes023_i64", np.array([0, 2, 3], dtype=np.int64)))

    # big / neg_big histograms for min/max scatter
    big_hist = np.full((1000,), BIG, dtype=np.float32)
    inits.append(_arr("big_hist", big_hist))
    neg_big_hist = np.full((1000,), -BIG, dtype=np.float32)
    inits.append(_arr("neg_big_hist", neg_big_hist))

    # Expand shapes
    inits.append(_arr("expand_1ch_shape", np.array([1, 1, H_W, H_W], dtype=np.int64)))
    inits.append(_arr("expand_10ch_shape", np.array([1, 10, H_W, H_W], dtype=np.int64)))
    inits.append(_arr("scalar_shape", np.array([1, 1, 1, 1], dtype=np.int64)))

    # Slice/Pad constants
    inits.append(_arr("shape1", np.array([1], dtype=np.int64)))
    inits.append(_arr("zero_i64_1d", np.array([0], dtype=np.int64).reshape(1)))
    inits.append(_arr("one_i64_1d", np.array([1], dtype=np.int64).reshape(1)))
    inits.append(_arr("ten_i64_1d", np.array([10], dtype=np.int64).reshape(1)))
    inits.append(_arr("thirty_i64_1d", np.array([H_W], dtype=np.int64).reshape(1)))
    inits.append(_arr("axes_23", np.array([2, 3], dtype=np.int64)))
    inits.append(_arr("steps_11", np.array([1, 1], dtype=np.int64)))

    # Conv weight to project a [1,1,H,W] mask into channel-2 of the 10-channel output
    proj_W = np.zeros((10, 1, 1, 1), dtype=np.float32)
    proj_W[2, 0, 0, 0] = 1.0
    inits.append(_arr("proj_W", proj_W))

    # One-hot lookup table
    inits.append(_arr("onehot_lookup", np.eye(10, dtype=np.float32)))
    inits.append(_arr("flat_shape", np.array([H_W * H_W], dtype=np.int64)))
    inits.append(_arr("flat_ch10_shape", np.array([1, 10, H_W * H_W], dtype=np.int64)))

    # Gather/concat constants
    inits.append(_arr("TWENTYNINE", np.float32(H_W - 1.0)))
    inits.append(_arr("THIRTY_F", np.float32(H_W * 1.0)))
    inits.append(_arr("hw_shape", np.array([H_W, H_W], dtype=np.int64)))

    # ------------------------------------------------------------------
    # Step 1: Extract ch2 and create mask
    # ------------------------------------------------------------------
    nodes.append(_n("Slice", ["input", "starts_ch2", "ends_ch2", "axes_all", "steps_all"], ["ch2"]))
    nodes.append(_n("Greater", ["ch2", "ZERO"], ["mask2_bool"]))

    # ------------------------------------------------------------------
    # Step 2: Component labeling via 30× MaxPool (3×3, 4-conn)
    # ------------------------------------------------------------------
    nodes.append(_n("Where", ["mask2_bool", "neg_idx", "NEG_BIG"], ["nlabels_0"]))
    prev = "nlabels_0"
    for it in range(POOL_ITERS):
        pooled = f"pooled_{it}"
        nodes.append(_n("MaxPool", [prev], [pooled], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]))
        nxt = f"nlabels_{it + 1}"
        nodes.append(_n("Where", ["mask2_bool", pooled, "NEG_BIG"], [nxt]))
        prev = nxt
    nodes.append(_n("Neg", [prev], ["labels_neg"]))
    nodes.append(_n("Cast", ["labels_neg"], ["labels_i64"], to=TensorProto.INT64))
    nodes.append(_n("Unsqueeze", ["labels_i64", "axes_unsq"], ["labels_idx"]))

    # ------------------------------------------------------------------
    # Step 3: Component sizes via ScatterND sum
    # ------------------------------------------------------------------
    nodes.append(_n("ConstantOfShape", ["hist_shape"], ["size_hist_init"],
                     value=onh.from_array(np.array([0.0], dtype=np.float32))))
    nodes.append(_n("Expand", ["ONE", "expand_1ch_shape"], ["ones_all"]))
    nodes.append(_n("Where", ["mask2_bool", "ones_all", "ZERO"], ["ones_grid"]))
    nodes.append(_n("ScatterND", ["size_hist_init", "labels_idx", "ones_grid"],
                    ["size_hist"], reduction="sum"))
    nodes.append(_n("Gather", ["size_hist", "labels_i64"], ["per_cell_size"], axis=0))

    # ------------------------------------------------------------------
    # Step 4: Largest component → main wall
    # ------------------------------------------------------------------
    nodes.append(_n("ArgMax", ["size_hist"], ["main_label"], axis=0, keepdims=0))

    # ------------------------------------------------------------------
    # Step 5: Main mask
    # ------------------------------------------------------------------
    nodes.append(_n("Reshape", ["main_label", "scalar_shape"], ["main_label_4d"]))
    nodes.append(_n("Expand", ["main_label_4d", "expand_1ch_shape"], ["main_label_grid"]))
    nodes.append(_n("Equal", ["labels_i64", "main_label_grid"], ["main_mask_bool"]))
    nodes.append(_n("Cast", ["main_mask_bool"], ["main_mask"], to=TensorProto.FLOAT))

    # ------------------------------------------------------------------
    # Step 6: Bbox of main component via ScatterND min/max
    # ------------------------------------------------------------------
    nodes.append(_n("Where", ["main_mask_bool", "row_coord", "BIG"], ["row_for_min"]))
    nodes.append(_n("Where", ["main_mask_bool", "row_coord", "NEG_BIG"], ["row_for_max"]))
    nodes.append(_n("Where", ["main_mask_bool", "col_coord", "BIG"], ["col_for_min"]))
    nodes.append(_n("Where", ["main_mask_bool", "col_coord", "NEG_BIG"], ["col_for_max"]))

    nodes.append(_n("ScatterND", ["big_hist", "labels_idx", "row_for_min"], ["hist_min_r"], reduction="min"))
    nodes.append(_n("ScatterND", ["big_hist", "labels_idx", "col_for_min"], ["hist_min_c"], reduction="min"))
    nodes.append(_n("ScatterND", ["neg_big_hist", "labels_idx", "row_for_max"], ["hist_max_r"], reduction="max"))
    nodes.append(_n("ScatterND", ["neg_big_hist", "labels_idx", "col_for_max"], ["hist_max_c"], reduction="max"))

    nodes.append(_n("Gather", ["hist_min_r", "main_label"], ["main_min_r"], axis=0))
    nodes.append(_n("Gather", ["hist_max_r", "main_label"], ["main_max_r"], axis=0))
    nodes.append(_n("Gather", ["hist_min_c", "main_label"], ["main_min_c"], axis=0))
    nodes.append(_n("Gather", ["hist_max_c", "main_label"], ["main_max_c"], axis=0))

    # Broadcast to [1,1,30,30]
    for src, dst in [("main_min_r", "main_min_r_grid"), ("main_max_r", "main_max_r_grid"),
                     ("main_min_c", "main_min_c_grid"), ("main_max_c", "main_max_c_grid")]:
        rsh = f"reshape_{src}"
        nodes.append(_n("Reshape", [src, "scalar_shape"], [rsh]))
        nodes.append(_n("Expand", [rsh, "expand_1ch_shape"], [dst]))

    # ------------------------------------------------------------------
    # Step 7: Inside-bbox mask
    # ------------------------------------------------------------------
    nodes.append(_n("Less", ["row_coord", "main_min_r_grid"], ["row_lt_min"]))
    nodes.append(_n("Not", ["row_lt_min"], ["row_ge_min"]))
    nodes.append(_n("Greater", ["row_coord", "main_max_r_grid"], ["row_gt_max"]))
    nodes.append(_n("Not", ["row_gt_max"], ["row_le_max"]))
    nodes.append(_n("Less", ["col_coord", "main_min_c_grid"], ["col_lt_min"]))
    nodes.append(_n("Not", ["col_lt_min"], ["col_ge_min"]))
    nodes.append(_n("Greater", ["col_coord", "main_max_c_grid"], ["col_gt_max"]))
    nodes.append(_n("Not", ["col_gt_max"], ["col_le_max"]))

    nodes.append(_n("And", ["row_ge_min", "row_le_max"], ["row_inside"]))
    nodes.append(_n("And", ["col_ge_min", "col_le_max"], ["col_inside"]))
    nodes.append(_n("And", ["row_inside", "col_inside"], ["inside_bbox"]))

    # ------------------------------------------------------------------
    # Step 8: Non-main inside bbox mask
    # ------------------------------------------------------------------
    nodes.append(_n("Not", ["main_mask_bool"], ["not_main"]))
    nodes.append(_n("And", ["inside_bbox", "not_main"], ["inside_not_main_bool"]))
    nodes.append(_n("Cast", ["inside_not_main_bool"], ["inside_not_main"], to=TensorProto.FLOAT))

    # Hole mask = all non-main cells inside bbox (zero cells AND non-zero content)
    nodes.append(_n("Cast", ["inside_not_main_bool"], ["hole_mask"], to=TensorProto.FLOAT))

    # ------------------------------------------------------------------
    # Step 9: Color propagation from outside patches into the bbox
    #
    # Non-2 cells outside the bbox → seed colors (1-9)
    # MaxPool propagates these colors inward
    # Each bbox cell gets the max (nearest) color from outside
    # ------------------------------------------------------------------
    # Get per-cell dominant color (argmax over channels)
    nodes.append(_n("ArgMax", ["input"], ["cell_color"], axis=1, keepdims=1))  # [1,1,30,30] INT64

    # Cast cell_color to FLOAT for comparison
    nodes.append(_n("Cast", ["cell_color"], ["cell_color_f"], to=TensorProto.FLOAT))
    # Zero out wall cells (color 2) so they don't propagate as fill
    nodes.append(_n("Equal", ["cell_color_f", "TWO"], ["is_c2"]))
    nodes.append(_n("Where", ["is_c2", "ZERO", "cell_color_f"], ["color_no_wall"]))  # FLOAT

    # Zero out cells inside the bbox (they get their color from outside)
    nodes.append(_n("Cast", ["inside_bbox"], ["inside_bbox_f"], to=TensorProto.FLOAT))
    nodes.append(_n("Where", ["inside_bbox", "ZERO", "color_no_wall"], ["seed_colors"]))  # FLOAT

    # Propagate colors inward via MaxPool
    prop_prev = "seed_colors"
    for it in range(PROP_ITERS):
        p_pooled = f"cp_pooled_{it}"
        p_better = f"cp_better_{it}"
        p_next = f"cp_next_{it}"
        nodes.append(_n("MaxPool", [prop_prev], [p_pooled], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]))
        # Where propagated value > current, take propagated (handles propagation)
        nodes.append(_n("Greater", [p_pooled, prop_prev], [p_better]))
        nodes.append(_n("Where", [p_better, p_pooled, prop_prev], [p_next]))
        prop_prev = p_next

    # For cells that are holes (inside bbox, not main wall), use propagated color
    # For non-hole cells, keep the original propagated color (which is 0 for outside bbox)
    nodes.append(_n("Mul", [prop_prev, "hole_mask"], ["hole_fill_color_f"]))  # [1,1,30,30] FLOAT

    # Convert to INT64 for one-hot lookup
    nodes.append(_n("Cast", ["hole_fill_color_f"], ["hole_fill_color"], to=TensorProto.INT64))

    # Convert to one-hot via Gather from identity matrix
    nodes.append(_n("Reshape", ["hole_fill_color", "flat_shape"], ["hfc_flat"]))
    nodes.append(_n("Gather", ["onehot_lookup", "hfc_flat"], ["hfc_onehot"], axis=0))
    nodes.append(_n("Reshape", ["hfc_onehot", "expand_10ch_shape"], ["hole_onehot"]))

    # Mask by hole positions
    nodes.append(_n("Expand", ["hole_mask", "expand_10ch_shape"], ["hole_mask_10ch"]))
    nodes.append(_n("Mul", ["hole_onehot", "hole_mask_10ch"], ["hole_fill"]))

    # ------------------------------------------------------------------
    # Step 10: Build output — remap bbox region to (0,0) via Gather
    #
    # The combined (wall + hole_fill) is at the bbox position in the grid.
    # For output cell (r,c), the corresponding input fill is at (r+r0, c+c0).
    # Use Gather with a flat index to remap, then apply canvas mask.
    # ------------------------------------------------------------------
    # Fill entire inside_bbox with ch2=1 (wall color)
    nodes.append(_n("Conv", ["inside_bbox_f", "proj_W"], ["bbox_wall"]))

    # Remove wall at hole positions (to be filled)
    nodes.append(_n("Sub", ["ONE", "hole_mask"], ["hole_mask_inv"]))
    nodes.append(_n("Expand", ["hole_mask_inv", "expand_10ch_shape"], ["hole_mask_inv_10ch"]))
    nodes.append(_n("Mul", ["bbox_wall", "hole_mask_inv_10ch"], ["bbox_wall_no_holes"]))

    # Combine wall + hole fill at bbox position
    nodes.append(_n("Add", ["bbox_wall_no_holes", "hole_fill"], ["mid_out"]))  # [1,10,30,30]

    # Flatten spatial dims for Gather
    nodes.append(_n("Reshape", ["mid_out", "flat_ch10_shape"], ["mid_flat"]))  # [1,10,900]

    # Compute shifted coordinates (FLOAT path)
    nodes.append(_n("Reshape", ["main_min_r", "scalar_shape"], ["r0_4d"]))
    nodes.append(_n("Reshape", ["main_min_c", "scalar_shape"], ["c0_4d"]))
    nodes.append(_n("Expand", ["r0_4d", "expand_1ch_shape"], ["r0_grid"]))
    nodes.append(_n("Expand", ["c0_4d", "expand_1ch_shape"], ["c0_grid"]))

    nodes.append(_n("Add", ["row_coord", "r0_grid"], ["shifted_r"]))  # [1,1,30,30]
    nodes.append(_n("Add", ["col_coord", "c0_grid"], ["shifted_c"]))
    nodes.append(_n("Clip", ["shifted_r", "ZERO", "TWENTYNINE"], ["sr_clipped"]))
    nodes.append(_n("Clip", ["shifted_c", "ZERO", "TWENTYNINE"], ["sc_clipped"]))

    # Flat index = sr * 30 + sc
    nodes.append(_n("Mul", ["sr_clipped", "THIRTY_F"], ["sr_times_30"]))
    nodes.append(_n("Add", ["sr_times_30", "sc_clipped"], ["flat_idx_f"]))
    nodes.append(_n("Cast", ["flat_idx_f"], ["flat_idx"], to=TensorProto.INT64))

    # Reshape to [30,30] for Gather (remove batch/channel dims)
    nodes.append(_n("Reshape", ["flat_idx", "hw_shape"], ["flat_idx_2d"]))  # [30,30]

    # Gather from flattened spatial at computed indices
    nodes.append(_n("Gather", ["mid_flat", "flat_idx_2d"], ["reordered"], axis=2))  # [1,10,30,30]

    # Canvas mask: output cells (r < C_H and c < C_W)
    nodes.append(_n("Sub", ["main_max_r", "main_min_r"], ["ch_m1"]))
    nodes.append(_n("Add", ["ch_m1", "ONE"], ["canvas_h_f"]))  # C_H = r1-r0+1
    nodes.append(_n("Sub", ["main_max_c", "main_min_c"], ["cw_m1"]))
    nodes.append(_n("Add", ["cw_m1", "ONE"], ["canvas_w_f"]))  # C_W = c1-c0+1

    nodes.append(_n("Reshape", ["canvas_h_f", "scalar_shape"], ["ch_4d"]))
    nodes.append(_n("Reshape", ["canvas_w_f", "scalar_shape"], ["cw_4d"]))
    nodes.append(_n("Expand", ["ch_4d", "expand_1ch_shape"], ["ch_grid"]))
    nodes.append(_n("Expand", ["cw_4d", "expand_1ch_shape"], ["cw_grid"]))

    nodes.append(_n("Less", ["row_coord", "ch_grid"], ["row_in_canvas"]))
    nodes.append(_n("Less", ["col_coord", "cw_grid"], ["col_in_canvas"]))
    nodes.append(_n("And", ["row_in_canvas", "col_in_canvas"], ["inside_canvas"]))
    nodes.append(_n("Expand", ["inside_canvas", "expand_10ch_shape"], ["inside_canvas_10ch"]))

    # Zero out cells outside canvas
    nodes.append(_n("Where", ["inside_canvas_10ch", "reordered", "ZERO"], ["output"]))

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    input_vi = oh.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H_W, H_W])
    output_vi = oh.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, H_W, H_W])
    graph = oh.make_graph(nodes=nodes, name="task233",
                          inputs=[input_vi], outputs=[output_vi],
                          initializer=inits)
    opset = oh.make_opsetid("", 17)
    model = oh.make_model(graph, opset_imports=[opset], ir_version=8)
    model.producer_name = ""
    onnx.checker.check_model(model, full_check=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submissions/handbuilds/task233.onnx")
    args = parser.parse_args()
    m = build(pathlib.Path(args.out))
    print(f"task233 built: {pathlib.Path(args.out).stat().st_size} bytes, {len(m.graph.node)} nodes")


if __name__ == "__main__":
    main()
