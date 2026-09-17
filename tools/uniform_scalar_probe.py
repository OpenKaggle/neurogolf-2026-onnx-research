from __future__ import annotations

import argparse
import csv
import math
import pathlib
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import defs, numpy_helper

from neurogolf_local import decode_grid, encode_grid, load_examples, score_onnx


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR = ROOT / "submissions/candidate_v4_plus8_task149_bool_onnx"
DEFAULT_OUTPUT = ROOT / "submissions/uniform_scalar_v1_onnx"
DEFAULT_REPORT_PREFIX = ROOT / "reports/uniform_scalar_v1"
DEFAULT_COMP_DIR = ROOT / "data/neurogolf-2026/raw"


# A deliberately narrow set of ONNX ops whose schemas describe NumPy-style
# multidirectional broadcasting over all data inputs.
SCALAR_BROADCAST_OPS = {
    "Add",
    "And",
    "Div",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Max",
    "Mean",
    "Min",
    "Mod",
    "Mul",
    "Or",
    "Pow",
    "Sub",
    "Sum",
    "Where",
    "Xor",
}

NUMERIC_KINDS = {"b", "i", "u", "f"}
CSV_FIELDS = [
    "task_id",
    "status",
    "kept",
    "safe_initializer_count",
    "candidate_param_save",
    "original_cost",
    "candidate_cost",
    "cost_delta",
    "original_score",
    "candidate_score",
    "score_delta",
    "exact_train",
    "exact_test",
    "exact_arc_gen",
    "exact_total",
    "changed_initializers",
    "consumer_positions",
    "source_path",
    "kept_path",
    "error",
]


@dataclass(frozen=True)
class ScalarCandidate:
    name: str
    original_dims: tuple[int, ...]
    dtype: str
    param_save: int
    value_repr: str
    consumers: tuple[str, ...]


def _relative_to_root(path: pathlib.Path) -> pathlib.Path:
    return path.resolve().relative_to(ROOT)


def _require_report_path(path: pathlib.Path) -> None:
    resolved = path.resolve()
    if resolved.parent != (ROOT / "reports").resolve():
        raise ValueError(f"report path must be directly under reports/: {path}")
    if not resolved.name.startswith("uniform_scalar_"):
        raise ValueError(f"report path must start with uniform_scalar_: {path}")
    if resolved.suffix not in {".csv", ".md"}:
        raise ValueError(f"report path must end in .csv or .md: {path}")


def _require_submission_path(path: pathlib.Path) -> None:
    resolved = path.resolve()
    rel = resolved.relative_to(ROOT)
    if len(rel.parts) < 2 or rel.parts[0] != "submissions":
        raise ValueError(f"submission artifact must live under submissions/: {path}")
    if not rel.parts[1].startswith("uniform_scalar_"):
        raise ValueError(f"submission artifact must start with submissions/uniform_scalar_: {path}")


def _repo_path(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _opset_version(model: onnx.ModelProto, domain: str = "") -> int:
    versions = [opset.version for opset in model.opset_import if opset.domain == domain]
    if not versions:
        versions = [opset.version for opset in model.opset_import if opset.domain == "ai.onnx"]
    return max(versions) if versions else defs.onnx_opset_version()


def _schema_allows_scalar_broadcast(op_type: str, input_index: int, opset_version: int) -> bool:
    if op_type not in SCALAR_BROADCAST_OPS:
        return False
    try:
        schema = defs.get_schema(op_type, opset_version, "")
    except Exception:
        return False
    doc = (schema.doc or "").lower()
    if "broadcast" not in doc or "multidirectional" not in doc:
        return False
    if not schema.inputs:
        return False
    if schema.inputs[0].option == defs.OpSchema.FormalParameterOption.Variadic:
        return True
    return input_index < len(schema.inputs)


def _initializer_consumers(graph: onnx.GraphProto) -> dict[str, list[tuple[str, int]]]:
    init_names = {init.name for init in graph.initializer}
    consumers: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for node in graph.node:
        for index, value_name in enumerate(node.input):
            if value_name in init_names:
                consumers[value_name].append((node.op_type, index))
    return consumers


def _is_uniform_non_scalar(init: onnx.TensorProto) -> tuple[bool, np.ndarray | None]:
    try:
        array = numpy_helper.to_array(init)
    except Exception:
        return False, None
    if array.size <= 1 or array.dtype.kind not in NUMERIC_KINDS:
        return False, None
    first = array.reshape(-1)[0]
    if not bool(np.all(array == first)):
        return False, None
    return True, array


def _value_repr(array: np.ndarray) -> str:
    value = array.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    return repr(value)


def find_scalar_candidates(model: onnx.ModelProto) -> list[ScalarCandidate]:
    graph = model.graph
    opset_version = _opset_version(model)
    consumers = _initializer_consumers(graph)
    graph_io_names = {value.name for value in list(graph.input) + list(graph.output)}
    candidates: list[ScalarCandidate] = []
    for init in graph.initializer:
        init_consumers = consumers.get(init.name, [])
        if not init_consumers or init.name in graph_io_names:
            continue
        is_uniform, array = _is_uniform_non_scalar(init)
        if not is_uniform or array is None:
            continue
        if not all(_schema_allows_scalar_broadcast(op_type, index, opset_version) for op_type, index in init_consumers):
            continue
        consumer_labels = tuple(f"{op_type}:{index}" for op_type, index in sorted(init_consumers))
        candidates.append(
            ScalarCandidate(
                name=init.name,
                original_dims=tuple(int(dim) for dim in init.dims),
                dtype=str(array.dtype),
                param_save=int(array.size - 1),
                value_repr=_value_repr(array),
                consumers=consumer_labels,
            )
        )
    return candidates


def scalarize_initializers(model: onnx.ModelProto, candidates: Iterable[ScalarCandidate]) -> onnx.ModelProto:
    candidate_names = {candidate.name for candidate in candidates}
    edited = onnx.ModelProto.FromString(model.SerializeToString())
    for init in edited.graph.initializer:
        if init.name not in candidate_names:
            continue
        array = numpy_helper.to_array(init)
        scalar = np.asarray(array.reshape(-1)[0], dtype=array.dtype)
        init.CopyFrom(numpy_helper.from_array(scalar, init.name))
    return edited


def check_and_infer_strict(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)


def split_examples(task_id: int, comp_dir: pathlib.Path) -> dict[str, list[dict]]:
    data = load_examples(task_id, comp_dir)
    return {
        "train": list(data.get("train", [])),
        "test": list(data.get("test", [])),
        "arc-gen": list(data.get("arc-gen", [])),
    }


def flattened_examples(examples_by_split: dict[str, list[dict]]) -> list[dict]:
    return examples_by_split["train"] + examples_by_split["test"] + examples_by_split["arc-gen"]


def _session(model_path: pathlib.Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def exact_decoded_equivalence(
    original_path: pathlib.Path,
    candidate_path: pathlib.Path,
    examples_by_split: dict[str, list[dict]],
) -> tuple[bool, dict[str, int]]:
    original_session = _session(original_path)
    candidate_session = _session(candidate_path)
    original_input = original_session.get_inputs()[0].name
    candidate_input = candidate_session.get_inputs()[0].name
    original_output = original_session.get_outputs()[0].name
    candidate_output = candidate_session.get_outputs()[0].name
    counts: dict[str, int] = {}
    for split_name, examples in examples_by_split.items():
        passed = 0
        for example in examples:
            encoded = encode_grid(example["input"])
            lhs = original_session.run([original_output], {original_input: encoded})[0]
            rhs = candidate_session.run([candidate_output], {candidate_input: encoded})[0]
            if decode_grid(lhs) != decode_grid(rhs):
                counts[split_name] = passed
                return False, counts
            passed += 1
        counts[split_name] = passed
    return True, counts


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        return repr(value)
    return f"{value:.12g}"


def _candidate_summary(candidates: list[ScalarCandidate]) -> tuple[str, str]:
    names = ";".join(candidate.name for candidate in candidates)
    positions = ";".join(f"{candidate.name}=[{','.join(candidate.consumers)}]" for candidate in candidates)
    return names, positions


def _empty_record(task_id: int, source: pathlib.Path, output: pathlib.Path) -> dict[str, object]:
    return {
        "task_id": task_id,
        "status": "",
        "kept": False,
        "safe_initializer_count": 0,
        "candidate_param_save": 0,
        "original_cost": "",
        "candidate_cost": "",
        "cost_delta": "",
        "original_score": "",
        "candidate_score": "",
        "score_delta": "",
        "exact_train": "",
        "exact_test": "",
        "exact_arc_gen": "",
        "exact_total": "",
        "changed_initializers": "",
        "consumer_positions": "",
        "source_path": str(_relative_to_root(source)),
        "kept_path": str(_relative_to_root(output)),
        "error": "",
    }


def _write_csv(path: pathlib.Path, records: list[dict[str, object]]) -> None:
    _require_report_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in CSV_FIELDS})


def _write_summary(path: pathlib.Path, records: list[dict[str, object]], args: argparse.Namespace) -> None:
    _require_report_path(path)
    kept = [record for record in records if record.get("kept")]
    total_gain = sum(float(record.get("score_delta") or 0.0) for record in kept)
    total_cost_save = sum(int(record.get("cost_delta") or 0) for record in kept)
    status_counts = Counter(str(record.get("status", "")) for record in records)
    top = sorted(kept, key=lambda record: float(record.get("score_delta") or 0.0), reverse=True)[:20]

    lines = [
        "# Uniform Scalar Probe",
        "",
        f"- Anchor: `{_relative_to_root(args.input_dir)}`",
        f"- Output: `{_relative_to_root(args.output_dir)}`",
        f"- Tasks scanned: {len(records)}",
        f"- Tasks kept: {len(kept)}",
        f"- Total score gain: {total_gain:.6f}",
        f"- Total cost saved: {total_cost_save}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## Top Kept Tasks", ""])
    if top:
        lines.append("| task | score gain | cost saved | safe initializers | changed initializers |")
        lines.append("| ---: | ---: | ---: | ---: | --- |")
        for record in top:
            changed = str(record.get("changed_initializers", ""))
            if len(changed) > 120:
                changed = changed[:117] + "..."
            lines.append(
                f"| {int(record['task_id']):03d} | {float(record['score_delta']):.6f} | "
                f"{int(record['cost_delta'])} | {int(record['safe_initializer_count'])} | `{changed}` |"
            )
    else:
        lines.append("No tasks were kept.")
    lines.extend(
        [
            "",
            "## Safety Gates",
            "",
            "- Converted only non-scalar uniform graph initializers.",
            "- Required every consumer input position to be on the schema-checked scalar-broadcast whitelist.",
            "- Required ONNX checker and strict shape inference on every edited task model.",
            "- Required exact decoded equivalence against the anchor on all train, test, and arc-gen examples.",
            "- Required measured local scorer cost and score improvement before replacing the copied anchor model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _zip_dir(source_dir: pathlib.Path, zip_path: pathlib.Path) -> None:
    _require_submission_path(zip_path)
    if zip_path.suffix != ".zip":
        raise ValueError(f"zip output must end in .zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.glob("task*.onnx")):
            archive.write(path, arcname=path.name)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    input_dir = args.input_dir
    output_dir = args.output_dir
    comp_dir = args.comp_dir
    report_prefix = args.report_prefix
    _require_submission_path(output_dir)
    _require_report_path(report_prefix.with_suffix(".csv"))
    _require_report_path(report_prefix.with_name(report_prefix.name + "_kept").with_suffix(".csv"))
    _require_report_path(report_prefix.with_suffix(".md"))
    if args.zip_out is not None:
        _require_submission_path(args.zip_out)

    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite or choose a new output dir")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    task_ids = args.tasks or list(range(1, 401))
    for task_id in task_ids:
        source = input_dir / f"task{task_id:03d}.onnx"
        if source.exists():
            shutil.copy2(source, output_dir / source.name)

    tmp_dir = output_dir / "_uniform_scalar_tmp"
    tmp_dir.mkdir()
    records: list[dict[str, object]] = []
    try:
        for task_id in task_ids:
            source = input_dir / f"task{task_id:03d}.onnx"
            output = output_dir / f"task{task_id:03d}.onnx"
            record = _empty_record(task_id, source, output)
            try:
                if not source.exists():
                    record["status"] = "missing_source"
                    records.append(record)
                    continue
                model = onnx.load(str(source))
                candidates = find_scalar_candidates(model)
                changed_names, consumer_positions = _candidate_summary(candidates)
                record.update(
                    safe_initializer_count=len(candidates),
                    candidate_param_save=sum(candidate.param_save for candidate in candidates),
                    changed_initializers=changed_names,
                    consumer_positions=consumer_positions,
                )
                if not candidates:
                    record["status"] = "no_safe_uniform_initializer"
                    records.append(record)
                    continue

                edited = scalarize_initializers(model, candidates)
                check_and_infer_strict(edited)
                candidate_path = tmp_dir / f"task{task_id:03d}.onnx"
                onnx.save(edited, str(candidate_path))

                examples_by_split = split_examples(task_id, comp_dir)
                examples = flattened_examples(examples_by_split)
                exact, exact_counts = exact_decoded_equivalence(source, candidate_path, examples_by_split)
                total_exact = sum(exact_counts.values())
                record.update(
                    exact_train=exact_counts.get("train", 0),
                    exact_test=exact_counts.get("test", 0),
                    exact_arc_gen=exact_counts.get("arc-gen", 0),
                    exact_total=total_exact,
                )
                if not exact:
                    record["status"] = "decoded_mismatch"
                    records.append(record)
                    continue

                original = score_onnx(source, examples, task_id, n_runs=args.n_runs)
                converted = score_onnx(candidate_path, examples, task_id, n_runs=args.n_runs)
                record.update(
                    original_cost=original.cost if original.cost is not None else "",
                    candidate_cost=converted.cost if converted.cost is not None else "",
                    cost_delta=(original.cost - converted.cost)
                    if original.cost is not None and converted.cost is not None
                    else "",
                    original_score=_format_float(original.score),
                    candidate_score=_format_float(converted.score),
                    score_delta=_format_float(converted.score - original.score),
                )
                if original.cost is None:
                    record["status"] = "original_unmeasurable"
                    record["error"] = original.error
                elif converted.cost is None:
                    record["status"] = "candidate_unmeasurable"
                    record["error"] = converted.error
                elif converted.cost < original.cost and converted.score > original.score:
                    shutil.copy2(candidate_path, output)
                    record["status"] = "kept"
                    record["kept"] = True
                else:
                    record["status"] = "not_improved"
                records.append(record)
            except Exception as exc:
                record["status"] = "error"
                record["error"] = repr(exc)
                records.append(record)
            if args.verbose or record.get("kept") or record.get("status") not in {
                "no_safe_uniform_initializer",
                "not_improved",
            }:
                print(
                    f"{task_id:03d} status={record['status']} kept={record['kept']} "
                    f"safe={record['safe_initializer_count']} save={record['candidate_param_save']} "
                    f"delta={record.get('score_delta', '')} err={record.get('error', '')}",
                    flush=True,
                )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    all_csv = report_prefix.with_suffix(".csv")
    kept_csv = report_prefix.with_name(report_prefix.name + "_kept").with_suffix(".csv")
    summary_md = report_prefix.with_suffix(".md")
    _write_csv(all_csv, records)
    _write_csv(kept_csv, [record for record in records if record.get("kept")])
    _write_summary(summary_md, records, args)
    if args.zip_out is not None:
        _zip_dir(output_dir, args.zip_out)
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe uniform initializer to scalar conversions.")
    parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--comp-dir", type=pathlib.Path, default=DEFAULT_COMP_DIR)
    parser.add_argument("--report-prefix", type=pathlib.Path, default=DEFAULT_REPORT_PREFIX)
    parser.add_argument("--zip-out", type=pathlib.Path)
    parser.add_argument("--tasks", nargs="*", type=int)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.input_dir = _repo_path(args.input_dir)
    args.output_dir = _repo_path(args.output_dir)
    args.comp_dir = _repo_path(args.comp_dir)
    args.report_prefix = _repo_path(args.report_prefix)
    if args.zip_out is not None:
        args.zip_out = _repo_path(args.zip_out)
    return args


def main(argv: list[str] | None = None) -> None:
    records = run(parse_args(argv))
    kept = [record for record in records if record.get("kept")]
    total_gain = sum(float(record.get("score_delta") or 0.0) for record in kept)
    total_cost_save = sum(int(record.get("cost_delta") or 0) for record in kept)
    print(f"kept={len(kept)} total_gain={total_gain:.6f} total_cost_save={total_cost_save}")
    for record in sorted(kept, key=lambda item: float(item.get("score_delta") or 0.0), reverse=True)[:10]:
        print(
            f"top task{int(record['task_id']):03d} gain={float(record['score_delta']):.6f} "
            f"cost_save={int(record['cost_delta'])} inits={record['safe_initializer_count']}"
        )


if __name__ == "__main__":
    sys.exit(main())
