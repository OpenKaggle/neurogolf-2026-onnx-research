"""task204 ONNX builder.

Rule (verified 268/268 in prototype_task204.py with 4-conn flood-fill;
also 268/268 with 8-conn flood + 4-conn leak detection):
  For each connected component of color-0 cells:
    - "leak" if any cell is adjacent (4-conn) to a non-{0,1} cell or out of grid.
    - if leak: keep cells as color 0 (open region).
    - else (surrounded by color 1 / grid edge):
        bbox H = max_r - min_r + 1; W = max_c - min_c + 1
        fill = 7 if (H odd or W odd) else 2.

ONNX strategy (no Loop/Scan/NonZero):
  1. Extract ch0 (mask), ch1 (walls). Shape [1,1,30,30].
  2. leak_source = 1 - ch0 - ch1. Cells of color 2..9 and out-of-grid cells.
  3. leak_neighbor_4conn = (Conv leak_source by cross-kernel [[0,1,0],[1,0,1],[0,1,0]]) > 0.
     cell_has_leak = ch0 * leak_neighbor_4conn.
  4. Component labels for ch0 via 5 iters MaxPool 3x3 on -neg_idx (octavi pattern).
  5. Histogram via ScatterND-with-max: has_leak[label] |= cell_has_leak[label_at_cell]
     (reduction='max', init 0).
  6. Bbox via ScatterND-with-min/max on row_coord and col_coord initializers.
  7. Decision: H_is_odd OR W_is_odd -> color 7 else color 2 (only for surrounded color-0 cells).
  8. Build 10 output channels: ch0_out = ch0 * has_leak; ch1_out = ch1; ch2_out = ch0 * !has_leak * !any_odd; ch7_out = ch0 * !has_leak * any_odd; others zero.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper as oh, numpy_helper as onh


BIG = 999.0
H_W = 30


def fp(name, value):
    return onh.from_array(np.array(value, dtype=np.float32), name=name)


def i64(name, value):
    return onh.from_array(np.array(value, dtype=np.int64), name=name)


def build(out_path: pathlib.Path) -> onnx.ModelProto:
    inits = []

    neg_idx = -np.arange(H_W * H_W, dtype=np.float32).reshape(1, 1, H_W, H_W)
    inits.append(onh.from_array(neg_idx, name="neg_idx"))

    inits.append(fp("NEG_BIG", -BIG))
    inits.append(fp("BIG", BIG))
    inits.append(fp("ZERO", 0.0))
    inits.append(fp("ONE", 1.0))
    inits.append(fp("TWO", 2.0))

    inits.append(i64("starts_ch0", [0, 0, 0, 0]))
    inits.append(i64("ends_ch0", [1, 1, H_W, H_W]))
    inits.append(i64("starts_ch1", [0, 1, 0, 0]))
    inits.append(i64("ends_ch1", [1, 2, H_W, H_W]))
    inits.append(i64("axes_all", [0, 1, 2, 3]))

    # Cross kernel for 4-conn leak detection: [[[[0,1,0],[1,0,1],[0,1,0]]]] shape [1,1,3,3]
    cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32).reshape(1, 1, 3, 3)
    inits.append(onh.from_array(cross, name="cross_kernel"))

    # Row/col coord initializers (float, for ScatterND on coords)
    row_coord = np.arange(H_W, dtype=np.float32).reshape(1, 1, H_W, 1).repeat(H_W, axis=3)
    col_coord = np.arange(H_W, dtype=np.float32).reshape(1, 1, 1, H_W).repeat(H_W, axis=2)
    inits.append(onh.from_array(row_coord, name="row_coord"))
    inits.append(onh.from_array(col_coord, name="col_coord"))

    # Histogram shapes (size 1000 to absorb BIG=999 bucket for non-mask cells)
    inits.append(i64("hist_shape", [1000]))
    inits.append(i64("axes_unsqueeze_last", [4]))

    # Static zero channel for output padding
    inits.append(onh.from_array(np.zeros((1, 1, H_W, H_W), dtype=np.float32), name="zero_ch"))

    # Parity lookup table: idx -> 1 if odd else 0. Need for H,W in 1..30. Size 32 safe.
    odd_table = np.array([i % 2 for i in range(32)], dtype=np.float32)
    inits.append(onh.from_array(odd_table, name="odd_table"))

    # BIG init for min-reduction histograms (so min behaves like inf for unset slots)
    big_hist = np.full((1000,), BIG, dtype=np.float32)
    inits.append(onh.from_array(big_hist, name="big_hist"))

    nodes = []

    # --- Step 1: extract masks ---
    nodes.append(oh.make_node("Slice", ["input", "starts_ch0", "ends_ch0", "axes_all"], ["ch0"], name="slice_ch0"))
    nodes.append(oh.make_node("Slice", ["input", "starts_ch1", "ends_ch1", "axes_all"], ["ch1"], name="slice_ch1"))
    nodes.append(oh.make_node("Greater", ["ch0", "ZERO"], ["mask_bool"], name="mask_bool"))

    # --- Step 2: leak source = 1 - ch0 - ch1 ---
    nodes.append(oh.make_node("Sub", ["ONE", "ch0"], ["one_minus_ch0"], name="one_minus_ch0"))
    nodes.append(oh.make_node("Sub", ["one_minus_ch0", "ch1"], ["leak_source"], name="leak_source"))

    # --- Step 3: 4-conn leak neighbor via Conv with cross kernel ---
    nodes.append(oh.make_node("Conv", ["leak_source", "cross_kernel"], ["leak_4n_count"],
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
                              name="leak_conv"))
    nodes.append(oh.make_node("Greater", ["leak_4n_count", "ZERO"], ["leak_4n_bool"], name="leak_4n_bool"))
    nodes.append(oh.make_node("Cast", ["leak_4n_bool"], ["leak_4n"], to=TensorProto.FLOAT, name="cast_leak_4n"))
    # cell_has_leak = ch0 * leak_4n (only color-0 cells)
    nodes.append(oh.make_node("Mul", ["ch0", "leak_4n"], ["cell_has_leak"], name="cell_has_leak"))

    # --- Step 4: component label via MaxPool 3x3 propagation.
    # 3x3 only sees direct 8-neighbors, preserving wall separation between components.
    # Need enough iters to cover max component span; 30 iters covers full 30x30 diagonal.
    # (5x5 was tested but spans across walls and incorrectly merges separate components.)
    nodes.append(oh.make_node("Where", ["mask_bool", "neg_idx", "NEG_BIG"], ["nlabels_0"], name="nlabels_init"))
    prev = "nlabels_0"
    POOL_ITERS = 30
    for it in range(POOL_ITERS):
        pooled = f"pooled_{it}"
        nodes.append(oh.make_node("MaxPool", [prev], [pooled],
                                  kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1],
                                  name=f"pool_{it}"))
        nxt = f"nlabels_{it+1}"
        nodes.append(oh.make_node("Where", ["mask_bool", pooled, "NEG_BIG"], [nxt], name=f"remask_{it}"))
        prev = nxt
    # Un-negate
    nodes.append(oh.make_node("Neg", [prev], ["labels"], name="labels_final"))
    # Cast to int64 for ScatterND indices
    nodes.append(oh.make_node("Cast", ["labels"], ["labels_i64"], to=TensorProto.INT64, name="cast_labels"))
    # Unsqueeze to [1,1,30,30,1] for ScatterND indices
    nodes.append(oh.make_node("Unsqueeze", ["labels_i64", "axes_unsqueeze_last"], ["labels_idx"], name="unsq_labels"))

    # --- Step 5: histogram has_leak[L] = max over (cells with label L) of cell_has_leak[cell] ---
    nodes.append(oh.make_node("ConstantOfShape", ["hist_shape"], ["hist_leak_init"],
                              value=onh.from_array(np.array([0.0], dtype=np.float32)),
                              name="hist_leak_init"))
    nodes.append(oh.make_node("ScatterND", ["hist_leak_init", "labels_idx", "cell_has_leak"],
                              ["hist_leak"], reduction="max", name="scatter_leak"))
    nodes.append(oh.make_node("Gather", ["hist_leak", "labels_i64"], ["per_cell_leak"], axis=0, name="gather_leak"))

    # --- Step 6: bbox via min/max scatter on row/col coords ---
    # For min: init to BIG, scatter with min reduction on row_coord and col_coord
    # For cells where mask=0, set their coord contribution to BIG (so min unaffected) and 0 (so max unaffected, then we need separate paths).
    # Simpler: only scatter cells where mask=1. We accomplish this via mask multiplication:
    # For min: where mask=1 use row_coord else BIG; for max: where mask=1 use row_coord else NEG_BIG (or -1).

    # row contributions
    nodes.append(oh.make_node("Where", ["mask_bool", "row_coord", "BIG"], ["row_for_min"], name="row_for_min"))
    nodes.append(oh.make_node("Where", ["mask_bool", "row_coord", "NEG_BIG"], ["row_for_max"], name="row_for_max"))
    nodes.append(oh.make_node("Where", ["mask_bool", "col_coord", "BIG"], ["col_for_min"], name="col_for_min"))
    nodes.append(oh.make_node("Where", ["mask_bool", "col_coord", "NEG_BIG"], ["col_for_max"], name="col_for_max"))

    # Scatter row min
    nodes.append(oh.make_node("ScatterND", ["big_hist", "labels_idx", "row_for_min"],
                              ["hist_min_r"], reduction="min", name="scatter_min_r"))
    nodes.append(oh.make_node("Gather", ["hist_min_r", "labels_i64"], ["per_cell_min_r"], axis=0, name="gather_min_r"))

    # Scatter row max (init to NEG_BIG to allow Max to pick correct values)
    nodes.append(oh.make_node("ConstantOfShape", ["hist_shape"], ["neg_big_hist"],
                              value=onh.from_array(np.array([-BIG], dtype=np.float32)),
                              name="neg_big_hist_init"))
    nodes.append(oh.make_node("ScatterND", ["neg_big_hist", "labels_idx", "row_for_max"],
                              ["hist_max_r"], reduction="max", name="scatter_max_r"))
    nodes.append(oh.make_node("Gather", ["hist_max_r", "labels_i64"], ["per_cell_max_r"], axis=0, name="gather_max_r"))

    # Scatter col min/max (reuse big_hist / neg_big_hist)
    nodes.append(oh.make_node("ScatterND", ["big_hist", "labels_idx", "col_for_min"],
                              ["hist_min_c"], reduction="min", name="scatter_min_c"))
    nodes.append(oh.make_node("Gather", ["hist_min_c", "labels_i64"], ["per_cell_min_c"], axis=0, name="gather_min_c"))
    nodes.append(oh.make_node("ScatterND", ["neg_big_hist", "labels_idx", "col_for_max"],
                              ["hist_max_c"], reduction="max", name="scatter_max_c"))
    nodes.append(oh.make_node("Gather", ["hist_max_c", "labels_i64"], ["per_cell_max_c"], axis=0, name="gather_max_c"))

    # H = max_r - min_r + 1, W = max_c - min_c + 1
    nodes.append(oh.make_node("Sub", ["per_cell_max_r", "per_cell_min_r"], ["H_minus_1"], name="H_minus_1"))
    nodes.append(oh.make_node("Add", ["H_minus_1", "ONE"], ["H_float"], name="H_float"))
    nodes.append(oh.make_node("Sub", ["per_cell_max_c", "per_cell_min_c"], ["W_minus_1"], name="W_minus_1"))
    nodes.append(oh.make_node("Add", ["W_minus_1", "ONE"], ["W_float"], name="W_float"))

    # --- Step 7: parity: Gather odd_table[H], odd_table[W] ---
    # Non-mask cells have garbage H/W (negative). Clamp to [0,31] before Gather; final output
    # multiplies by surrounded mask anyway so values at non-mask cells don't matter.
    inits.append(fp("CLIP_MIN", 0.0))
    inits.append(fp("CLIP_MAX", 31.0))
    nodes.append(oh.make_node("Clip", ["H_float", "CLIP_MIN", "CLIP_MAX"], ["H_clipped"], name="clip_H"))
    nodes.append(oh.make_node("Clip", ["W_float", "CLIP_MIN", "CLIP_MAX"], ["W_clipped"], name="clip_W"))
    nodes.append(oh.make_node("Cast", ["H_clipped"], ["H_i64"], to=TensorProto.INT64, name="cast_H"))
    nodes.append(oh.make_node("Cast", ["W_clipped"], ["W_i64"], to=TensorProto.INT64, name="cast_W"))
    nodes.append(oh.make_node("Gather", ["odd_table", "H_i64"], ["H_odd"], axis=0, name="gather_H_odd"))
    nodes.append(oh.make_node("Gather", ["odd_table", "W_i64"], ["W_odd"], axis=0, name="gather_W_odd"))
    # any_odd = max(H_odd, W_odd) (both are 0/1 floats)
    nodes.append(oh.make_node("Max", ["H_odd", "W_odd"], ["any_odd"], name="any_odd"))

    # --- Step 8: build output channels ---
    # Surrounded mask: ch0 * (1 - has_leak) — only color-0 cells in closed components
    nodes.append(oh.make_node("Sub", ["ONE", "per_cell_leak"], ["one_minus_leak"], name="one_minus_leak"))
    nodes.append(oh.make_node("Mul", ["ch0", "one_minus_leak"], ["surrounded"], name="surrounded"))
    # Open mask: ch0 * has_leak — color-0 cells that stay color 0
    nodes.append(oh.make_node("Mul", ["ch0", "per_cell_leak"], ["open_mask"], name="open_mask"))
    # ch_7_out = surrounded * any_odd
    nodes.append(oh.make_node("Mul", ["surrounded", "any_odd"], ["ch7_out"], name="ch7_out"))
    # ch_2_out = surrounded * (1 - any_odd)
    nodes.append(oh.make_node("Sub", ["ONE", "any_odd"], ["one_minus_any_odd"], name="one_minus_any_odd"))
    nodes.append(oh.make_node("Mul", ["surrounded", "one_minus_any_odd"], ["ch2_out"], name="ch2_out"))

    # ch_0_out = open_mask (color-0 cells in open components keep being 0)
    # ch_1_out = ch1 unchanged
    # ch_2_out, ch_7_out as above; others zero.
    concat_inputs = [
        "open_mask",   # ch0
        "ch1",         # ch1
        "ch2_out",     # ch2
        "zero_ch",     # ch3
        "zero_ch",     # ch4
        "zero_ch",     # ch5
        "zero_ch",     # ch6
        "ch7_out",     # ch7
        "zero_ch",     # ch8
        "zero_ch",     # ch9
    ]
    nodes.append(oh.make_node("Concat", concat_inputs, ["output"], axis=1, name="concat_out"))

    input_vi = oh.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H_W, H_W])
    output_vi = oh.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, H_W, H_W])
    graph = oh.make_graph(nodes=nodes, name="task204",
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
    parser.add_argument("--out", default="submissions/handbuilds/task204.onnx")
    args = parser.parse_args()
    m = build(pathlib.Path(args.out))
    print(f"task204 built: {pathlib.Path(args.out).stat().st_size} bytes, {len(m.graph.node)} nodes")


if __name__ == "__main__":
    main()
