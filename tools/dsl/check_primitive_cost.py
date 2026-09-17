"""Compile each new primitive in isolation, run the scorer's cost machinery,
and confirm the result is *measurable* (not `None` -> "unmeasurable").

This is the smoke test that catches the v0 Slice+Pad failure mode: producing a
correct ONNX whose `calculate_memory` returns None because of a dynamic-shape
intermediate.

Run:  .venv/bin/python -m tools.dsl.check_primitive_cost
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

THIS = pathlib.Path(__file__).resolve()
ROOT = THIS.parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from neurogolf_local import (
    _sanitize_model,
    calculate_memory,
    calculate_params,
    encode_grid,
)
from tools.dsl.compiler import DslOp, DslProgram, compile_to_onnx


def measure(program: DslProgram, h: int, w: int) -> dict:
    model = compile_to_onnx(program)
    sanitized = _sanitize_model(model)
    if sanitized is None:
        return {"status": "sanitize-fail"}

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.profile_file_prefix = f"costchk_pid{os.getpid()}"
    sess = ort.InferenceSession(sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"])

    grid = [[(r * w + c) % 10 for c in range(w)] for r in range(h)]
    inp = encode_grid(grid)
    _ = sess.run(["output"], {"input": inp})[0]
    trace = sess.end_profiling()
    try:
        memory = calculate_memory(sanitized, trace)
    finally:
        if trace and os.path.exists(trace):
            os.remove(trace)
    params = calculate_params(sanitized)
    if memory is None or params is None:
        return {"status": "unmeasurable", "memory": memory, "params": params}
    return {"status": "ok", "memory": int(memory), "params": int(params), "cost": int(memory + params)}


def main() -> int:
    cases = [
        ("identity",         [DslOp("identity", {})], 5, 5),
        ("transpose",        [DslOp("transpose", {})], 5, 5),
        ("swap_colors",      [DslOp("swap_colors", {"c1": 1, "c2": 2})], 5, 5),
        ("replace_color",    [DslOp("replace_color", {"src": 1, "dst": 2})], 5, 5),
        ("flip_h",           [DslOp("flip_h", {})], 5, 5),
        ("flip_v",           [DslOp("flip_v", {})], 5, 5),
        ("rotate180_v0",     [DslOp("rotate180", {})], 5, 5),
        ("rotate90_v0",      [DslOp("rotate90", {})], 5, 5),
        ("rotate180_static_3x3", [DslOp("rotate180_static", {"h": 3, "w": 3})], 3, 3),
        ("rotate180_static_5x5", [DslOp("rotate180_static", {"h": 5, "w": 5})], 5, 5),
        ("rotate180_static_9x9", [DslOp("rotate180_static", {"h": 9, "w": 9})], 9, 9),
        ("rotate90_static_3x3",  [DslOp("rotate90_static",  {"h": 3, "w": 3})], 3, 3),
        ("rotate90_static_5x5",  [DslOp("rotate90_static",  {"h": 5, "w": 5})], 5, 5),
        ("rotate90_static_9x9",  [DslOp("rotate90_static",  {"h": 9, "w": 9})], 9, 9),
        ("count_color_c4",   [DslOp("count_color", {"c": 4})], 5, 5),
        ("dominant_color",   [DslOp("dominant_color", {})], 5, 5),
        ("fill_bg_c3",       [DslOp("fill_bg", {"bg_color": 3})], 5, 5),
        ("bbox_crop_c4",     [DslOp("bbox_crop", {"target_color": 4})], 5, 5),
        ("tile_h_n2",        [DslOp("tile_h", {"n": 2})], 5, 5),
        ("tile_v_n2",        [DslOp("tile_v", {"n": 2})], 5, 5),
        ("tile_h_n3",        [DslOp("tile_h", {"n": 3})], 5, 5),
        ("tile_v_n3",        [DslOp("tile_v", {"n": 3})], 5, 5),
        ("largest_blob_c4",  [DslOp("largest_blob", {"target_color": 4})], 5, 5),
        ("select_channel_4_1_7", [DslOp("select_channel", {"cond_ch": 4, "if_ch": 1, "else_ch": 7})], 5, 5),
        ("fill_c4",          [DslOp("fill", {"c": 4})], 5, 5),
        ("shift_down_k2",    [DslOp("shift_down", {"k": 2})], 5, 5),
        ("shift_right_k2",   [DslOp("shift_right", {"k": 2})], 5, 5),
        ("crop_0_0_3_3",     [DslOp("crop", {"r0": 0, "c0": 0, "h": 3, "w": 3})], 5, 5),
        ("crop_1_2_4_5",     [DslOp("crop", {"r0": 1, "c0": 2, "h": 4, "w": 5})], 5, 5),
        # ---- Batch 2 ----
        ("mask_foreground",  [DslOp("mask_foreground", {})], 5, 5),
        ("remove_color_c3",  [DslOp("remove_color", {"target_color": 3})], 5, 5),
        ("thicken_c2",       [DslOp("thicken", {"target_color": 2})], 5, 5),
        ("hollow_out_c2",    [DslOp("hollow_out", {"target_color": 2})], 5, 5),
        ("flood_fill_c2_f5", [DslOp("flood_fill", {"target_color": 2, "fill_color": 5})], 5, 5),
        ("trim_border_k1",   [DslOp("trim_border", {"k": 1})], 5, 5),
    ]

    print(f"{'primitive':30}  {'status':>14}  {'memory':>8}  {'params':>8}  {'cost':>8}")
    print("-" * 78)
    failed = 0
    for label, ops, h, w in cases:
        result = measure(DslProgram.of(ops), h, w)
        mem = result.get("memory", "")
        par = result.get("params", "")
        cost = result.get("cost", "")
        print(f"{label:30}  {result['status']:>14}  {mem:>8}  {par:>8}  {cost:>8}")
        if result["status"] != "ok":
            failed += 1
    print()
    print(f"{len(cases) - failed}/{len(cases)} primitives produce measurable cost")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
