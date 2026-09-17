"""Isolated full-example pass/fail evaluation for one ONNX file.

The bundle scorer reuses one Python process per task, which has been observed
to mis-flag certain tasks (e.g. task230 reads as 0/3 if task220 was scored
first in the same process). This script intentionally evaluates a SINGLE
task ONNX file in its own process so the verdict is not contaminated.

Output: a single JSON line on stdout with task_id, path, train/test/arc-gen
pass counts, total per split, cost, score, nodes, file_size, status.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import (
    EXCLUDED_OP_TYPES,
    FILESIZE_LIMIT_IN_BYTES,
    _sanitize_model,
    calculate_memory,
    calculate_params,
    encode_grid,
    load_examples,
)


def evaluate(onnx_path: pathlib.Path, task_id: int, comp_dir: pathlib.Path) -> dict:
    out: dict = {
        "task_id": task_id,
        "path": str(onnx_path),
        "splits": {"train": [0, 0], "test": [0, 0], "arc-gen": [0, 0]},
        "cost": None,
        "score": 0.0,
        "nodes": 0,
        "file_size": 0,
        "status": "ok",
    }
    if not onnx_path.is_file():
        out["status"] = "missing"
        return out
    out["file_size"] = onnx_path.stat().st_size
    if out["file_size"] > FILESIZE_LIMIT_IN_BYTES:
        out["status"] = "too_large"
        return out

    model = onnx.load(str(onnx_path))
    out["nodes"] = len(model.graph.node)
    if any(n.op_type.upper() in EXCLUDED_OP_TYPES or "Sequence" in n.op_type for n in model.graph.node):
        out["status"] = "excluded_op"
        return out
    sanitized = _sanitize_model(model)
    if sanitized is None:
        out["status"] = "sanitize"
        return out

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.profile_file_prefix = f"iso_task{task_id:03d}_pid{os.getpid()}"
    sess = ort.InferenceSession(sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"])

    examples = load_examples(task_id, comp_dir)
    for split in ("train", "test", "arc-gen"):
        for ex in examples.get(split, []):
            grid_in = ex["input"]
            grid_out = ex["output"]
            if max(len(grid_in), len(grid_in[0]), len(grid_out), len(grid_out[0])) > 30:
                continue
            out["splits"][split][1] += 1
            try:
                pred = sess.run(["output"], {"input": encode_grid(grid_in)})[0]
                target = encode_grid(grid_out) > 0.0
                if np.array_equal(pred > 0.0, target):
                    out["splits"][split][0] += 1
            except Exception:
                pass

    trace = sess.end_profiling()
    try:
        memory = calculate_memory(sanitized, trace)
    finally:
        if trace and os.path.exists(trace):
            os.remove(trace)
    params = calculate_params(sanitized)
    if memory is None or params is None or memory < 0 or params < 0:
        out["status"] = "unmeasurable"
        return out
    cost = int(memory + params)
    import math
    out["cost"] = cost
    out["score"] = max(1.0, 25.0 - math.log(max(1.0, cost)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    args = parser.parse_args()
    res = evaluate(pathlib.Path(args.onnx), args.task_id, pathlib.Path(args.comp_dir))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
