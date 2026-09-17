from __future__ import annotations

import json
import math
import os
import pathlib
import traceback
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import onnx
import onnxruntime


BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = 1, 10, 30, 30
GRID_SHAPE = (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
FILESIZE_LIMIT_IN_BYTES = int(1.44 * 1024 * 1024)
EXCLUDED_OP_TYPES = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


@dataclass(frozen=True)
class TaskScore:
    task_id: int
    cost: int | None
    score: float
    memory: int | None
    params: int | None
    local_pass: int
    local_total: int
    nodes: int
    file_size: int
    path: str
    error: str = ""


def task_path(root: pathlib.Path, task_id: int, suffix: str) -> pathlib.Path:
    return root / f"task{task_id:03d}.{suffix}"


def load_examples(task_id: int, comp_dir: pathlib.Path) -> dict:
    return json.loads(task_path(comp_dir, task_id, "json").read_text())


def all_examples(task_id: int, comp_dir: pathlib.Path) -> list[dict]:
    examples = load_examples(task_id, comp_dir)
    return examples["train"] + examples["test"] + examples.get("arc-gen", [])


def encode_grid(grid: list[list[int]]) -> np.ndarray:
    arr = np.array(grid, dtype=np.int32)
    out = np.zeros(GRID_SHAPE, dtype=np.float32)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            color = int(arr[row, col])
            if 0 <= color < CHANNELS:
                out[0, color, row, col] = 1.0
    return out


def decode_grid(tensor: np.ndarray) -> list[list[int]]:
    tensor = tensor > 0.0
    rows: list[list[int]] = []
    _, channels, height, width = tensor.shape
    for row in range(height):
        cells: list[int] = []
        for col in range(width):
            colors = [c for c in range(channels) if tensor[0, c, row, col]]
            cells.append(colors[0] if len(colors) == 1 else (11 if colors else 10))
        while cells and cells[-1] == 10:
            cells.pop()
        rows.append(cells)
    while rows and not rows[-1]:
        rows.pop()
    return rows


def _sanitize_model(model: onnx.ModelProto) -> onnx.ModelProto | None:
    for node in model.graph.node:
        if node.output:
            node.name = node.output[0]
            if "kernel_time" in node.output[0]:
                return None

    name_map: dict[str, str] = {}
    counter = 0

    def safe_name(old_name: str) -> str:
        nonlocal counter
        if not old_name or old_name in {"input", "output"}:
            return old_name
        if old_name not in name_map:
            name_map[old_name] = f"safe_name_{counter}"
            counter += 1
        return name_map[old_name]

    for inp in model.graph.input:
        inp.name = safe_name(inp.name)
    for init in model.graph.initializer:
        init.name = safe_name(init.name)
    for init in model.graph.sparse_initializer:
        init.values.name = safe_name(init.values.name)
        init.indices.name = safe_name(init.indices.name)
    for node in model.graph.node:
        for i, value in enumerate(node.input):
            node.input[i] = safe_name(value)
        for i, value in enumerate(node.output):
            node.output[i] = safe_name(value)
        if node.output and node.output[0]:
            node.name = node.output[0]
    for out in model.graph.output:
        out.name = safe_name(out.name)
    for value_info in model.graph.value_info:
        value_info.name = safe_name(value_info.name)
    for node in model.graph.node:
        if node.output:
            node.name = node.output[0]
    return model


def calculate_params(model: onnx.ModelProto) -> int | None:
    params = 0
    for init in model.graph.initializer:
        if any(dim <= 0 for dim in init.dims):
            return None
        params += math.prod(init.dims) if init.dims else 1
    for sparse_init in model.graph.sparse_initializer:
        if any(dim <= 0 for dim in sparse_init.values.dims):
            return None
        params += math.prod(sparse_init.values.dims) if sparse_init.values.dims else 1
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                if any(dim <= 0 for dim in attr.t.dims):
                    return None
                params += math.prod(attr.t.dims) if attr.t.dims else 1
            elif attr.name == "sparse_value":
                if any(dim <= 0 for dim in attr.sparse_tensor.values.dims):
                    return None
                params += math.prod(attr.sparse_tensor.values.dims) if attr.sparse_tensor.values.dims else 1
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return int(params)


def calculate_memory(model: onnx.ModelProto, trace_path: str) -> int | None:
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    if len(graph.input) > 1 or len(graph.output) > 1:
        return None

    init_names = {init.name for init in graph.initializer}
    init_names.update(init.name for init in graph.sparse_initializer)
    io_names = {t.name for t in list(graph.input) + list(graph.output)}
    if io_names.intersection(init_names) or model.functions:
        return None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return None

    node_outputs: dict[str, list[str]] = {}
    tensor_names: set[str] = set()
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}:
                return None
        node_outputs[node.name] = list(node.output)
        for output_name in node.output:
            if output_name:
                tensor_names.add(output_name)

    tensor_memory: dict[str, int] = {}
    tensor_dtypes: dict[str, np.dtype] = {}
    tensor_map = {t.name: t for t in list(graph.input) + list(graph.value_info) + list(graph.output)}
    tensor_names.update(tensor_map.keys())
    for tensor_name in tensor_names:
        item = tensor_map.get(tensor_name)
        if not item:
            return None
        if item.type.HasField("sequence_type"):
            return None
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        num_elements = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value"):
                return None
            if dim.dim_value <= 0:
                return None
            num_elements *= dim.dim_value
        if tensor_name in {"input", "output"}:
            continue
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        tensor_memory[tensor_name] = int(num_elements * np.dtype(np_dtype).itemsize)
        tensor_dtypes[tensor_name] = np.dtype(np_dtype)

    seen = set()
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if item.name in seen:
            return None
        seen.add(item.name)
    for node in graph.node:
        for output_name in node.output:
            if output_name and output_name != "output":
                item = tensor_map.get(output_name)
                if item is None or not item.type.HasField("tensor_type"):
                    return None

    trace_data = json.loads(pathlib.Path(trace_path).read_text())
    for event in trace_data:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        if "output_type_shape" not in event["args"]:
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        if node_name not in node_outputs:
            continue
        for i, shape_dict in enumerate(event["args"]["output_type_shape"]):
            if i >= len(node_outputs[node_name]):
                continue
            output_name = node_outputs[node_name][i]
            if output_name not in tensor_dtypes:
                continue
            itemsize = np.dtype(tensor_dtypes[output_name]).itemsize
            mem = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            tensor_memory[output_name] = max(tensor_memory[output_name], int(mem))
    return int(sum(tensor_memory.values()))


def verify_onnx(model_path: pathlib.Path, examples: Iterable[dict]) -> tuple[int, int]:
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    session = onnxruntime.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    passed = 0
    total = 0
    for example in examples:
        total += 1
        try:
            output = session.run(["output"], {"input": encode_grid(example["input"])})[0]
            target = encode_grid(example["output"]) > 0.0
            if np.array_equal(output > 0.0, target):
                passed += 1
        except onnxruntime.ONNXRuntimeError:
            pass
    return passed, total


def score_onnx(model_path: pathlib.Path, examples: list[dict], task_id: int, n_runs: int = 3) -> TaskScore:
    try:
        if not model_path.is_file():
            return TaskScore(task_id, None, 0.0, None, None, 0, len(examples), 0, 0, str(model_path), "missing")
        file_size = model_path.stat().st_size
        if file_size > FILESIZE_LIMIT_IN_BYTES:
            return TaskScore(task_id, None, 0.0, None, None, 0, len(examples), 0, file_size, str(model_path), "too_large")

        model = onnx.load(str(model_path))
        nodes = len(model.graph.node)
        if any(node.op_type.upper() in EXCLUDED_OP_TYPES or "Sequence" in node.op_type for node in model.graph.node):
            return TaskScore(task_id, None, 0.0, None, None, 0, len(examples), nodes, file_size, str(model_path), "excluded_op")
        sanitized = _sanitize_model(model)
        if sanitized is None:
            return TaskScore(task_id, None, 0.0, None, None, 0, len(examples), nodes, file_size, str(model_path), "sanitize")

        options = onnxruntime.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.log_severity_level = 3
        # Include the PID so concurrent probe scripts do not fight over the same
        # ORT profile prefix when scoring the same task in parallel.
        options.profile_file_prefix = f"ng_task{task_id:03d}_pid{os.getpid()}"
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"])

        local_pass = 0
        local_total = 0
        for example in examples:
            local_total += 1
            output = session.run(["output"], {"input": encode_grid(example["input"])})[0]
            target = encode_grid(example["output"]) > 0.0
            if np.array_equal(output > 0.0, target):
                local_pass += 1
            if local_total >= n_runs and n_runs > 0:
                break
        trace_path = session.end_profiling()
        try:
            memory = calculate_memory(sanitized, trace_path)
        finally:
            if trace_path and os.path.exists(trace_path):
                os.remove(trace_path)
        params = calculate_params(sanitized)
        if memory is None or params is None or memory < 0 or params < 0:
            return TaskScore(task_id, None, 0.0, memory, params, local_pass, local_total, nodes, file_size, str(model_path), "unmeasurable")
        cost = int(memory + params)
        score = max(1.0, 25.0 - math.log(max(1.0, cost)))
        return TaskScore(task_id, cost, score, int(memory), int(params), local_pass, local_total, nodes, file_size, str(model_path))
    except Exception:
        return TaskScore(
            task_id=task_id,
            cost=None,
            score=0.0,
            memory=None,
            params=None,
            local_pass=0,
            local_total=len(examples),
            nodes=0,
            file_size=model_path.stat().st_size if model_path.exists() else 0,
            path=str(model_path),
            error=traceback.format_exc(limit=2).strip().replace("\n", " | "),
        )
