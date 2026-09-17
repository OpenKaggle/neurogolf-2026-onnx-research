from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from neurogolf_local import all_examples, encode_grid, score_onnx


F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
FP16_MAX = 65504.0
VALUE_PRESERVING_OPS = {"Slice", "Gather", "Transpose", "Reshape", "Squeeze", "Unsqueeze", "Identity"}


def _fits_fp16(array: np.ndarray) -> bool:
    return array.size == 0 or float(np.nanmax(np.abs(array))) <= FP16_MAX


def _guard_constants(graph: onnx.GraphProto) -> None:
    for init in graph.initializer:
        if init.data_type == F32 and not _fits_fp16(numpy_helper.to_array(init)):
            raise ValueError("float initializer exceeds fp16 range")
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == F32:
                if not _fits_fp16(numpy_helper.to_array(attr.t)):
                    raise ValueError("float tensor attribute exceeds fp16 range")
            if attr.name == "value_floats" and attr.floats:
                if max(abs(value) for value in attr.floats) > FP16_MAX:
                    raise ValueError("value_floats exceeds fp16 range")


def fp16_surgery_v2(model: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(model.SerializeToString())
    graph = model.graph
    _guard_constants(graph)

    initializer_names = {init.name for init in graph.initializer}
    region = {"input"}
    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if node.op_type not in VALUE_PRESERVING_OPS:
                continue
            if any(output in region for output in node.output):
                continue
            data_inputs = [value for value in node.input if value and value not in initializer_names]
            if data_inputs and all(value in region for value in data_inputs):
                for output in node.output:
                    if output:
                        region.add(output)
                changed = True
    region.discard("output")

    def is_region_value_preserving(node: onnx.NodeProto) -> bool:
        return node.op_type in VALUE_PRESERVING_OPS and any(output in region for output in node.output)

    boundary: set[str] = set()
    for node in graph.node:
        if is_region_value_preserving(node):
            continue
        for value in node.input:
            if value in region and value != "input":
                boundary.add(value)
    if any(value == "input" for node in graph.node if not is_region_value_preserving(node) for value in node.input):
        boundary.add("input")

    for init in graph.initializer:
        if init.data_type == F32:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name))

    cast_map = {value: f"{value}__h16" for value in boundary}
    new_nodes = []
    if "input" in boundary:
        new_nodes.append(helper.make_node("Cast", ["input"], ["input__h16"], to=F16, name="input__h16"))
    for node in graph.node:
        new_nodes.append(node)
        for output in node.output:
            if output in cast_map:
                new_nodes.append(helper.make_node("Cast", [output], [cast_map[output]], to=F16, name=cast_map[output]))
    del graph.node[:]
    graph.node.extend(new_nodes)

    for node in graph.node:
        if node.op_type == "Cast" and node.output and node.output[0].endswith("__h16"):
            continue
        if is_region_value_preserving(node):
            continue
        for i, value in enumerate(node.input):
            if value in cast_map:
                node.input[i] = cast_map[value]

    for node in graph.node:
        if node.op_type == "Cast" and not (node.output and node.output[0].endswith("__h16")):
            for attr in node.attribute:
                if attr.name == "to" and attr.i == F32:
                    attr.i = F16
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == F32:
                attr.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attr.t).astype(np.float16), attr.t.name))

    del graph.value_info[:]
    for output in graph.output:
        if output.type.tensor_type.elem_type == F32:
            output.type.tensor_type.elem_type = F16

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    return model


def same_decoded_outputs(original_path: pathlib.Path, candidate_path: pathlib.Path, examples: list[dict]) -> bool:
    original = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(str(candidate_path), providers=["CPUExecutionProvider"])
    original_input = original.get_inputs()[0].name
    candidate_input = candidate.get_inputs()[0].name
    for example in examples:
        target = np.array(example["output"], dtype=np.int64)
        height, width = target.shape
        input_grid = example["input"]
        if max(len(input_grid), len(input_grid[0]), height, width) > 30:
            continue
        x = encode_grid(example["input"])
        original_output = original.run(None, {original_input: x})[0]
        candidate_output = candidate.run(None, {candidate_input: x})[0]
        if original_output.shape[2] < height or original_output.shape[3] < width:
            return False
        if candidate_output.shape[2] < height or candidate_output.shape[3] < width:
            return False
        original_grid = np.argmax(original_output[0, :, :height, :width], axis=0)
        candidate_grid = np.argmax(candidate_output[0, :, :height, :width], axis=0)
        if not np.array_equal(original_grid, candidate_grid):
            return False
    return True


def score_path(path: pathlib.Path, task_id: int, comp_dir: pathlib.Path) -> tuple[int | None, float]:
    result = score_onnx(path, all_examples(task_id, comp_dir), task_id, n_runs=3)
    return result.cost, result.score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--report", default="reports/fp16_surgery_report.json")
    parser.add_argument("--min-save", type=int, default=1800)
    parser.add_argument("--tasks", nargs="*", type=int)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    comp_dir = pathlib.Path(args.comp_dir)
    report_path = pathlib.Path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    task_ids = args.tasks or list(range(1, 401))
    records = []
    kept = 0
    total_delta = 0.0

    for task_id in task_ids:
        name = f"task{task_id:03d}.onnx"
        source = input_dir / name
        destination = output_dir / name
        shutil.copy2(source, destination)
        examples = all_examples(task_id, comp_dir)
        record = {"task_id": task_id, "kept": False, "error": ""}
        try:
            original_cost, original_score = score_path(source, task_id, comp_dir)
            if not original_cost:
                raise ValueError("original has no measurable cost")
            candidate = output_dir / f"_{name}.fp16.onnx"
            onnx.save(fp16_surgery_v2(onnx.load(str(source))), str(candidate))
            candidate_cost, candidate_score = score_path(candidate, task_id, comp_dir)
            if not candidate_cost:
                raise ValueError("candidate has no measurable cost")
            record.update(
                original_cost=original_cost,
                candidate_cost=candidate_cost,
                original_score=original_score,
                candidate_score=candidate_score,
                save=original_cost - candidate_cost,
                delta=candidate_score - original_score,
            )
            if (original_cost - candidate_cost) >= args.min_save and candidate_score > original_score:
                if same_decoded_outputs(source, candidate, examples):
                    shutil.copy2(candidate, destination)
                    record["kept"] = True
                    kept += 1
                    total_delta += candidate_score - original_score
            candidate.unlink(missing_ok=True)
        except Exception as exc:
            record["error"] = repr(exc)
        records.append(record)
        if task_id % 10 == 0 or record["kept"] or record["error"]:
            print(
                f"{task_id:03d} kept={record['kept']} save={record.get('save')} "
                f"delta={record.get('delta')} err={record['error']}",
                flush=True,
            )

    report = {"kept": kept, "predicted_delta": total_delta, "records": records}
    report_path.write_text(json.dumps(report, indent=2))
    print(f"kept {kept} tasks; predicted delta {total_delta:.4f}")


if __name__ == "__main__":
    main()
