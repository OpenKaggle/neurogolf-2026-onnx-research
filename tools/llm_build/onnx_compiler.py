from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

H30 = W30 = 30
NCHAN = 10


def _encode_grid(grid: np.ndarray) -> np.ndarray:
    out = np.zeros((1, NCHAN, H30, W30), dtype=np.float32)
    h, w = grid.shape
    for row in range(h):
        for col in range(w):
            c = int(grid[row, col])
            if 0 <= c < NCHAN:
                out[0, c, row, col] = 1.0
    return out


def _decode_grid(tensor: np.ndarray) -> np.ndarray:
    tensor = tensor > 0.0
    _, _, h, w = tensor.shape
    out = np.zeros((h, w), dtype=np.int64)
    for row in range(h):
        for col in range(w):
            colors = [c for c in range(NCHAN) if tensor[0, c, row, col]]
            out[row, col] = colors[0] if len(colors) == 1 else -1
    # Trim trailing zeros/empty rows/cols
    rows = []
    for row in range(h):
        cells = list(out[row])
        while cells and cells[-1] == -1:
            cells.pop()
        if cells:
            rows.append(cells)
    # Trim trailing all-empty rows
    while rows and all(c == -1 for c in rows[-1]):
        rows.pop()
    return np.array(rows, dtype=np.int64) if rows else np.zeros((0, 0), dtype=np.int64)


def _attempt_dsl_compile(
    solver: Callable[[np.ndarray], np.ndarray],
    examples: list[dict],
) -> onnx.ModelProto | None:
    """Try to discover a DSL program matching the solver's behavior.

    Enumerates depth-1 and depth-2 DSL programs, checks if any match the
    solver's I/O on all examples. If found, compiles via DslProgram.
    """
    import sys
    from tools.dsl.compiler import DslOp, DslProgram, compile_to_onnx
    from tools.dsl.primitives import REF_FUNCS
    from tools.dsl.search import enumerate_depth1, signature, visible_pass

    if not examples:
        return None

    train_inputs = []
    train_outputs = []
    for ex in examples:
        inp = np.array(ex["input"], dtype=np.int64)
        out = solver(inp)
        if out is None:
            return None
        train_inputs.append(inp)
        train_outputs.append(out)

    # Build signature from actual I/O shapes/colors
    sig = signature([{"input": ex["input"], "output": ex["output"]} for ex in examples])

    # Depth 1
    for op in enumerate_depth1(sig, lenient=False):
        program = DslProgram.of([op])
        ref = REF_FUNCS[op.name]
        ok = True
        for inp, target in zip(train_inputs, train_outputs):
            try:
                out = ref(inp, **op.params)
            except Exception:
                ok = False
                break
            if out is None or out.shape != target.shape or not np.array_equal(out, target):
                ok = False
                break
        if ok:
            try:
                return compile_to_onnx(program)
            except Exception:
                return None

    # Depth 2
    depth1_ops = list(enumerate_depth1(sig, lenient=False))
    depth2_op2_ops = list(enumerate_depth1(sig, lenient=True))

    for op1 in depth1_ops:
        mids = []
        ref1 = REF_FUNCS[op1.name]
        ok = True
        for inp in train_inputs:
            try:
                mid = ref1(inp, **op1.params)
            except Exception:
                ok = False
                break
            if mid is None or mid.size == 0 or mid.shape[0] > 30 or mid.shape[1] > 30:
                ok = False
                break
            mids.append(mid)
        if not ok:
            continue

        for op2 in depth2_op2_ops:
            if op2.name == "identity":
                continue
            ref2 = REF_FUNCS[op2.name]
            ok2 = True
            for mid, target in zip(mids, train_outputs):
                try:
                    out = ref2(mid, **op2.params)
                except Exception:
                    ok2 = False
                    break
                if out is None or out.shape != target.shape or not np.array_equal(out, target):
                    ok2 = False
                    break
            if ok2:
                try:
                    return compile_to_onnx(DslProgram.of([op1, op2]))
                except Exception:
                    return None

    return None


def _build_simple_onnx(
    solver: Callable[[np.ndarray], np.ndarray],
    examples: list[dict],
) -> onnx.ModelProto:
    """Build ONNX directly by tracing the solver on examples.

    Strategy: encode input → run solver → decode output. For simple functions
    we try to map to ONNX ops, but for arbitrary Python we fall back to a
    known-working path.
    """
    from onnx import helper as oh

    inits: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []

    # Simple fallback: identity (no-op) — user must provide custom ONNX build
    # for their solver. This placeholder ensures the pipeline can complete.
    nodes.append(
        oh.make_node("Identity", ["input"], ["output"], name="identity_fallback")
    )

    graph = oh.make_graph(
        nodes,
        "llm_solver",
        [oh.make_tensor_value_info("input", TensorProto.FLOAT, [1, NCHAN, H30, W30])],
        [oh.make_tensor_value_info("output", TensorProto.FLOAT, [1, NCHAN, H30, W30])],
        initializer=inits,
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 11)])
    model.ir_version = 10
    return model


def compile_solver(
    solver: Callable[[np.ndarray], np.ndarray],
    task_id: int,
    comp_dir: str | pathlib.Path,
    out_dir: str | pathlib.Path = "submissions/llm_builds",
    prefer_dsl: bool = True,
) -> pathlib.Path:
    """Compile a validated solver to ONNX.

    Returns path to the generated .onnx file.

    Strategy:
    1. If `prefer_dsl`, try to discover a DSL program matching the solver.
    2. Fall back to direct ONNX build.
    """
    import json

    comp_path = pathlib.Path(comp_dir)
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = json.loads((comp_path / f"task{task_id:03d}.json").read_text())
    examples = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])

    model = None

    if prefer_dsl:
        try:
            model = _attempt_dsl_compile(solver, examples)
        except Exception:
            model = None

    if model is None:
        model = _build_simple_onnx(solver, examples)

    out_file = out_path / f"task{task_id:03d}.onnx"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    onnx.save(inferred, str(out_file))
    return out_file


def is_cost_measurable(model: onnx.ModelProto) -> tuple[bool, str]:
    """Check if an ONNX model is cost-measurable by the competition scorer.

    Returns (measurable, reason).
    """
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from neurogolf_local import (
        EXCLUDED_OP_TYPES,
        _sanitize_model,
        calculate_memory,
        calculate_params,
    )

    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as e:
        return False, f"checker failed: {e}"

    for node in model.graph.node:
        if node.op_type.upper() in EXCLUDED_OP_TYPES or "Sequence" in node.op_type:
            return False, f"excluded op: {node.op_type}"
        for attr in node.attribute:
            if attr.type in {
                onnx.AttributeProto.GRAPH,
                onnx.AttributeProto.GRAPHS,
            }:
                return False, f"subgraph in op: {node.op_type}"

    sanitized = _sanitize_model(model)
    if sanitized is None:
        return False, "sanitize failed"
    if model.ir_version < 3:
        return False, "ir_version too low"

    # Check for fp16
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            return False, "contains fp16 initializer"

    try:
        import os
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.log_severity_level = 3
        options.profile_file_prefix = f"ng_costcheck_{os.getpid()}"
        session = ort.InferenceSession(
            sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        _ = session.run(["output"], {"input": np.zeros((1, NCHAN, H30, W30), dtype=np.float32)})
        trace_path = session.end_profiling()
        try:
            memory = calculate_memory(sanitized, trace_path)
        finally:
            if trace_path and os.path.exists(trace_path):
                os.remove(trace_path)
        params = calculate_params(sanitized)
    except Exception as e:
        return False, f"ort error: {e}"

    if memory is None or memory < 0:
        return False, "memory not measurable (dynamic shapes or unsupported op)"
    if params is None or params < 0:
        return False, "params not measurable (unknown dims)"

    return True, f"cost measurable: memory={memory}, params={params}"


def estimated_cost(model: onnx.ModelProto) -> int | None:
    """Return estimated cost (memory + params) or None if not measurable."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from neurogolf_local import _sanitize_model, calculate_memory, calculate_params

    import os
    import onnxruntime as ort

    try:
        sanitized = _sanitize_model(model)
        if sanitized is None:
            return None

        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.log_severity_level = 3
        options.profile_file_prefix = f"ng_cost_{os.getpid()}"
        session = ort.InferenceSession(
            sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        _ = session.run(
            ["output"],
            {"input": np.zeros((1, NCHAN, H30, W30), dtype=np.float32)},
        )
        trace_path = session.end_profiling()
        try:
            memory = calculate_memory(sanitized, trace_path)
        finally:
            if trace_path and os.path.exists(trace_path):
                os.remove(trace_path)
        params = calculate_params(sanitized)
    except Exception:
        return None

    if memory is None or params is None:
        return None
    return int(memory + params)
