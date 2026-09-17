from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from fp16_surgery import same_decoded_outputs
from neurogolf_local import all_examples, score_onnx


INT_TYPES = {TensorProto.INT64, TensorProto.INT32, TensorProto.INT16}
SHAPE_OPS = {"Reshape", "Slice", "Squeeze", "Unsqueeze", "Expand", "Tile", "Pad", "ConstantOfShape"}
INT64 = TensorProto.INT64
INT32 = TensorProto.INT32


def _tensor_names_used_as_shape_inputs(graph: onnx.GraphProto) -> set[str]:
    names: set[str] = set()
    for node in graph.node:
        if node.op_type == "Reshape" and len(node.input) > 1:
            names.add(node.input[1])
        elif node.op_type == "Slice":
            names.update(value for value in node.input[1:5] if value)
        elif node.op_type in {"Squeeze", "Unsqueeze"}:
            if len(node.input) > 1:
                names.add(node.input[1])
        elif node.op_type == "Expand" and len(node.input) > 1:
            names.add(node.input[1])
        elif node.op_type == "Tile" and len(node.input) > 1:
            names.add(node.input[1])
        elif node.op_type == "Pad" and len(node.input) > 1:
            names.add(node.input[1])
        elif node.op_type == "ConstantOfShape" and node.input:
            names.add(node.input[0])
    return names


def _producer_map(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in graph.node for output in node.output if output}


def _initializer_map(graph: onnx.GraphProto) -> dict[str, onnx.TensorProto]:
    return {init.name: init for init in graph.initializer}


def _can_cast_initializer(init: onnx.TensorProto) -> bool:
    if init.data_type != INT64:
        return False
    arr = numpy_helper.to_array(init)
    return arr.size == 0 or (arr.min() >= np.iinfo(np.int32).min and arr.max() <= np.iinfo(np.int32).max)


def int64_data_to_int32(model: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(model.SerializeToString())
    graph = model.graph
    shape_inputs = _tensor_names_used_as_shape_inputs(graph)
    producers = _producer_map(graph)
    initializers = _initializer_map(graph)

    # Keep all static shape/index constants as int64. Convert only data constants
    # and the int64 tensors whose consumers can accept int32.
    for init in graph.initializer:
        if init.name not in shape_inputs and _can_cast_initializer(init):
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.int32), init.name))

    for node in graph.node:
        # Constant nodes used as shape/index inputs must stay int64.
        if any(output in shape_inputs for output in node.output):
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.t.data_type == INT64:
                arr = numpy_helper.to_array(attr.t)
                if arr.size == 0 or (arr.min() >= np.iinfo(np.int32).min and arr.max() <= np.iinfo(np.int32).max):
                    attr.t.CopyFrom(numpy_helper.from_array(arr.astype(np.int32), attr.t.name))
            elif attr.name == "value_ints" and attr.ints:
                if min(attr.ints) >= np.iinfo(np.int32).min and max(attr.ints) <= np.iinfo(np.int32).max:
                    # value_ints has no dtype annotation, so leave it alone.
                    pass

    for node in graph.node:
        if node.op_type == "Cast" and node.output and node.output[0] not in shape_inputs:
            for attr in node.attribute:
                if attr.name == "to" and attr.i == INT64:
                    attr.i = INT32

    for value_info in list(graph.value_info) + list(graph.output):
        tt = value_info.type.tensor_type
        if tt.elem_type == INT64 and value_info.name not in shape_inputs:
            tt.elem_type = INT32

    onnx.checker.check_model(model, full_check=True)
    return onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--report", default="reports/int_surgery_probe_report.json")
    parser.add_argument("--tasks", nargs="+", type=int, required=True)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    comp_dir = pathlib.Path(args.comp_dir)
    report_path = pathlib.Path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for task_id in args.tasks:
        source = input_dir / f"task{task_id:03d}.onnx"
        candidate = output_dir / f"task{task_id:03d}.onnx"
        examples = all_examples(task_id, comp_dir)
        record = {"task_id": task_id, "kept": False, "error": ""}
        try:
            original = score_onnx(source, examples, task_id, n_runs=3)
            if original.cost is None:
                record["error"] = f"original_unmeasurable:{original.error or 'cost_none'}"
                records.append(record)
                print(record, flush=True)
                continue
            onnx.save(int64_data_to_int32(onnx.load(str(source))), str(candidate))
            converted = score_onnx(candidate, examples, task_id, n_runs=3)
            same = False
            if converted.cost is not None and converted.score > original.score:
                same = same_decoded_outputs(source, candidate, examples)
            record.update(
                original_cost=original.cost,
                original_score=original.score,
                candidate_cost=converted.cost,
                candidate_score=converted.score,
                save=(original.cost or 0) - (converted.cost or 0),
                delta=converted.score - original.score,
                same=same,
                kept=bool(same and converted.cost is not None and converted.score > original.score),
            )
        except Exception as exc:
            record["error"] = repr(exc)
        records.append(record)
        print(record, flush=True)

    report_path.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
