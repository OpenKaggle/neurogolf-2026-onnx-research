from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from fp16_surgery import fp16_surgery_v2, same_decoded_outputs
from neurogolf_local import all_examples, score_onnx


F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
FP16_MAX = 65504.0


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


def fp16_surgery_v1(model: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(model.SerializeToString())
    graph = model.graph
    _guard_constants(graph)

    for init in graph.initializer:
        if init.data_type == F32:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name))

    new_nodes = [helper.make_node("Cast", ["input"], ["input_h16"], to=F16, name="input_h16")]
    for node in graph.node:
        for i, value in enumerate(node.input):
            if value == "input":
                node.input[i] = "input_h16"
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)

    for node in graph.node:
        if node.op_type == "Cast" and node.name != "input_h16":
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


def candidate_score(path: pathlib.Path, task_id: int, comp_dir: pathlib.Path) -> tuple[int | None, float]:
    result = score_onnx(path, all_examples(task_id, comp_dir), task_id, n_runs=3)
    return result.cost, result.score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--report", default="reports/fp16_dual_surgery_report.json")
    parser.add_argument("--min-save", type=int, default=1800)
    parser.add_argument("--tasks", nargs="*", type=int)
    parser.add_argument("--exclude", nargs="*", type=int, default=[])
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    comp_dir = pathlib.Path(args.comp_dir)
    report_path = pathlib.Path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    task_ids = args.tasks or list(range(1, 401))
    exclude = set(args.exclude)
    records = []
    kept = 0
    total_delta = 0.0
    methods = [("v2", fp16_surgery_v2), ("v1", fp16_surgery_v1)]

    for task_id in task_ids:
        name = f"task{task_id:03d}.onnx"
        source = input_dir / name
        destination = output_dir / name
        shutil.copy2(source, destination)
        examples = all_examples(task_id, comp_dir)
        record = {"task_id": task_id, "kept": False, "method": "", "error": "", "excluded": task_id in exclude}
        try:
            original = score_onnx(source, examples, task_id, n_runs=3)
            record.update(original_cost=original.cost, original_score=original.score)
            if task_id in exclude:
                records.append(record)
                continue
            if not original.cost:
                raise ValueError("original has no measurable cost")

            best = None
            source_model = onnx.load(str(source))
            for method_name, method in methods:
                candidate = output_dir / f"_{name}.{method_name}.onnx"
                try:
                    onnx.save(method(source_model), str(candidate))
                    candidate_result = score_onnx(candidate, examples, task_id, n_runs=3)
                    if not candidate_result.cost:
                        raise ValueError(candidate_result.error or "candidate has no measurable cost")
                    save = original.cost - candidate_result.cost
                    delta = candidate_result.score - original.score
                    ok = save >= args.min_save and delta > 0 and same_decoded_outputs(source, candidate, examples)
                    if ok and (best is None or candidate_result.cost < best[1].cost):
                        best = (method_name, candidate_result, candidate)
                except Exception as exc:
                    record.setdefault("candidate_errors", []).append({"method": method_name, "error": repr(exc)})
                    candidate.unlink(missing_ok=True)

            if best is not None:
                method_name, candidate_result, candidate = best
                shutil.copy2(candidate, destination)
                record.update(
                    kept=True,
                    method=method_name,
                    candidate_cost=candidate_result.cost,
                    candidate_score=candidate_result.score,
                    save=original.cost - candidate_result.cost,
                    delta=candidate_result.score - original.score,
                )
                kept += 1
                total_delta += candidate_result.score - original.score

            for candidate in output_dir.glob(f"_{name}.*.onnx"):
                candidate.unlink(missing_ok=True)
        except Exception as exc:
            record["error"] = repr(exc)
        records.append(record)
        if task_id % 10 == 0 or record.get("kept") or record.get("error"):
            print(
                f"{task_id:03d} kept={record.get('kept')} method={record.get('method')} "
                f"save={record.get('save')} delta={record.get('delta')} err={record.get('error')}",
                flush=True,
            )

    report = {"kept": kept, "predicted_delta": total_delta, "records": records}
    report_path.write_text(json.dumps(report, indent=2))
    print(f"kept {kept} tasks; predicted delta {total_delta:.4f}")


if __name__ == "__main__":
    main()
