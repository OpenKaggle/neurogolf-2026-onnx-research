from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


TOOL_DIR = pathlib.Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
sys.path.insert(0, str(TOOL_DIR))

from neurogolf_local import TaskScore, decode_grid, encode_grid, load_examples, score_onnx


ANCHOR_DIR = ROOT / "submissions" / "candidate_v4_plus10_compiler2_onnx"
OUT_DIR = ROOT / "submissions" / "compiler_backend_sweep_v3_onnx"
COMP_DIR = ROOT / "data" / "neurogolf-2026" / "raw"
REPORT_PREFIX = ROOT / "reports" / "compiler_backend_sweep_v3"
ANCHOR_SCORE_CSV = ROOT / "reports" / "candidate_v4_plus10_compiler2_score_n3.csv"
SPLITS = ("train", "test", "arc-gen")


@dataclass(frozen=True)
class TypeInfo:
    elem_type: int
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        if any(dim <= 0 for dim in self.shape):
            return 0
        try:
            dtype = helper.tensor_dtype_to_np_dtype(self.elem_type)
        except Exception:
            return 0
        elements = int(math.prod(self.shape)) if self.shape else 1
        return int(elements * np.dtype(dtype).itemsize)


@dataclass(frozen=True)
class Rewrite:
    model: onnx.ModelProto
    rule: str
    note: str


def clone_model(model: onnx.ModelProto) -> onnx.ModelProto:
    return onnx.ModelProto.FromString(model.SerializeToString())


def rel(path: pathlib.Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def require_allowed_paths(output_dir: pathlib.Path, report_prefix: pathlib.Path) -> None:
    output_rel = output_dir.resolve().relative_to(ROOT)
    report_rel = report_prefix.resolve().relative_to(ROOT)
    if output_rel.parts != ("submissions", "compiler_backend_sweep_v3_onnx"):
        raise ValueError(f"refusing output path outside branch workspace: {output_dir}")
    if report_rel.parts != ("reports", "compiler_backend_sweep_v3"):
        raise ValueError(f"refusing report path outside branch workspace: {report_prefix}")


def strip_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    edited = clone_model(model)
    del edited.graph.value_info[:]
    return edited


def strict_check(model: onnx.ModelProto) -> None:
    stripped = strip_value_info(model)
    onnx.checker.check_model(stripped, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(stripped, check_type=True, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)


def inferred_types(model: onnx.ModelProto) -> dict[str, TypeInfo]:
    inferred = onnx.shape_inference.infer_shapes(strip_value_info(model), check_type=True, strict_mode=True)
    out: dict[str, TypeInfo] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        static = True
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                static = False
                break
            dims.append(int(dim.dim_value))
        if static:
            out[value.name] = TypeInfo(int(tensor_type.elem_type), tuple(dims))
    return out


def producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output if output}


def use_counts(model: onnx.ModelProto) -> Counter[str]:
    counts: Counter[str] = Counter()
    for node in model.graph.node:
        for value in node.input:
            if value:
                counts[value] += 1
    return counts


def remove_node(model: onnx.ModelProto, target: onnx.NodeProto) -> None:
    kept = [node for node in model.graph.node if node is not target]
    del model.graph.node[:]
    model.graph.node.extend(kept)


def remove_nodes(model: onnx.ModelProto, targets: set[onnx.NodeProto]) -> None:
    kept = [node for node in model.graph.node if node not in targets]
    del model.graph.node[:]
    model.graph.node.extend(kept)


def replace_value_uses(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new


def set_cast_to(node: onnx.NodeProto, elem_type: int) -> None:
    for attr in node.attribute:
        if attr.name == "to":
            attr.i = int(elem_type)
            return
    node.attribute.append(helper.make_attribute("to", int(elem_type)))


def get_int_attr(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def set_int_attr(node: onnx.NodeProto, name: str, value: int) -> None:
    for attr in node.attribute:
        if attr.name == name:
            attr.i = int(value)
            return
    node.attribute.append(helper.make_attribute(name, int(value)))


def dtype_name(elem_type: int | None) -> str:
    if elem_type is None:
        return "UNKNOWN"
    try:
        return TensorProto.DataType.Name(int(elem_type))
    except Exception:
        return str(elem_type)


def tensor_constants(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    values = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                values[node.output[0]] = attr.t
    return values


def tensor_array(model: onnx.ModelProto, name: str) -> np.ndarray | None:
    tensor = tensor_constants(model).get(name)
    if tensor is None:
        return None
    try:
        return numpy_helper.to_array(tensor)
    except Exception:
        return None


def axes_from_reduce(model: onnx.ModelProto, node: onnx.NodeProto) -> tuple[int, ...] | None:
    for attr in node.attribute:
        if attr.name == "axes":
            return tuple(int(axis) for axis in attr.ints)
    if len(node.input) >= 2 and node.input[1]:
        array = tensor_array(model, node.input[1])
        if array is None:
            return None
        return tuple(int(axis) for axis in np.asarray(array).reshape(-1))
    return None


def normalize_axes(axes: tuple[int, ...], rank: int) -> tuple[int, ...] | None:
    out: list[int] = []
    for axis in axes:
        normalized = axis + rank if axis < 0 else axis
        if normalized < 0 or normalized >= rank:
            return None
        out.append(normalized)
    if len(set(out)) != len(out):
        return None
    return tuple(out)


def set_reduce_axes(model: onnx.ModelProto, node: onnx.NodeProto, axes: tuple[int, ...]) -> None:
    for attr in node.attribute:
        if attr.name == "axes":
            del attr.ints[:]
            attr.ints.extend(int(axis) for axis in axes)
            return
    if len(node.input) >= 2 and node.input[1]:
        name = f"{node.output[0]}__backend_v3_axes"
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(axes, dtype=np.int64), name))
        node.input[1] = name
        return
    node.attribute.append(helper.make_attribute("axes", list(int(axis) for axis in axes)))


def bool_cast_chain_rewrites(model: onnx.ModelProto) -> list[Rewrite]:
    """Collapse Cast(Cast(bool, Tmid), Tout) using an all-values proof.

    ONNX Cast maps bool to exact 0/1 values for numeric outputs, and maps those
    values back to the original bool exactly. We do not collapse float/int
    chains, because intermediate rounding, clipping, or overflow can be
    semantically visible outside the public examples.
    """
    types = inferred_types(model)
    producers = producer_map(model)
    uses = use_counts(model)
    rewrites: list[Rewrite] = []
    numeric_or_bool = {
        TensorProto.BOOL,
        TensorProto.UINT8,
        TensorProto.INT8,
        TensorProto.UINT16,
        TensorProto.INT16,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.FLOAT16,
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
        TensorProto.UINT32,
        TensorProto.UINT64,
    }

    for outer in model.graph.node:
        if outer.op_type != "Cast" or not outer.input or not outer.output:
            continue
        inner = producers.get(outer.input[0])
        if inner is None or inner.op_type != "Cast" or not inner.input or not inner.output:
            continue
        source = inner.input[0]
        middle = inner.output[0]
        final = outer.output[0]
        source_info = types.get(source)
        final_info = types.get(final)
        if source_info is None or final_info is None:
            continue
        if source_info.elem_type != TensorProto.BOOL or final_info.elem_type not in numeric_or_bool:
            continue
        if final == "output":
            # Keep graph-output naming simple and avoid output-contract drift.
            continue

        edited = strip_value_info(model)
        edited_producers = producer_map(edited)
        edited_inner = edited_producers[middle]
        edited_outer = edited_producers[final]
        if final_info.elem_type == TensorProto.BOOL:
            replace_value_uses(edited, final, source)
            remove_node(edited, edited_outer)
            note = f"remove bool->{dtype_name(types.get(middle).elem_type if types.get(middle) else None)}->bool outer Cast"
        else:
            edited_outer.input[0] = source
            set_cast_to(edited_outer, final_info.elem_type)
            note = f"compose bool->{dtype_name(final_info.elem_type)} Cast"
        if uses[middle] == 1:
            remove_node(edited, edited_inner)
            note += "; remove now-unused inner Cast"
        rewrites.append(Rewrite(edited, "exact Cast-chain collapse", note))
    return rewrites


def reduce_sum_chain_rewrites(model: onnx.ModelProto) -> list[Rewrite]:
    types = inferred_types(model)
    producers = producer_map(model)
    uses = use_counts(model)
    rewrites: list[Rewrite] = []
    for outer in model.graph.node:
        if outer.op_type != "ReduceSum" or not outer.input or not outer.output:
            continue
        inner = producers.get(outer.input[0])
        if inner is None or inner.op_type != "ReduceSum" or not inner.input or not inner.output:
            continue
        if uses[inner.output[0]] != 1:
            continue
        if get_int_attr(inner, "keepdims", 1) != 1 or get_int_attr(outer, "keepdims", 1) != 1:
            continue
        if get_int_attr(inner, "noop_with_empty_axes", 0) or get_int_attr(outer, "noop_with_empty_axes", 0):
            continue
        input_info = types.get(inner.input[0])
        if input_info is None:
            continue
        inner_axes = axes_from_reduce(model, inner)
        outer_axes = axes_from_reduce(model, outer)
        if inner_axes is None or outer_axes is None:
            continue
        rank = len(input_info.shape)
        inner_axes = normalize_axes(inner_axes, rank)
        outer_axes = normalize_axes(outer_axes, rank)
        if inner_axes is None or outer_axes is None or not inner_axes or not outer_axes:
            continue
        fused_axes = tuple(sorted(set(inner_axes) | set(outer_axes)))

        edited = strip_value_info(model)
        edited_producers = producer_map(edited)
        edited_inner = edited_producers[inner.output[0]]
        edited_outer = edited_producers[outer.output[0]]
        edited_outer.input[0] = edited_inner.input[0]
        set_reduce_axes(edited, edited_outer, fused_axes)
        set_int_attr(edited_outer, "keepdims", 1)
        remove_node(edited, edited_inner)
        rewrites.append(
            Rewrite(
                edited,
                "ReduceSum-axis fusion/collapse",
                f"fuse keepdims=1 axes {inner_axes}+{outer_axes} -> {fused_axes}",
            )
        )
    return rewrites


def identity_transpose_rewrites(model: onnx.ModelProto) -> list[Rewrite]:
    rewrites: list[Rewrite] = []
    types = inferred_types(model)
    for node in model.graph.node:
        if node.op_type != "Transpose" or not node.input or not node.output:
            continue
        rank = len(types.get(node.input[0], TypeInfo(0, ())).shape)
        perm = next((tuple(int(axis) for axis in attr.ints) for attr in node.attribute if attr.name == "perm"), None)
        if perm is None:
            perm = tuple(reversed(range(rank)))
        if perm != tuple(range(rank)) or node.output[0] == "output":
            continue
        edited = strip_value_info(model)
        edited_node = producer_map(edited).get(node.output[0])
        if edited_node is None:
            continue
        replace_value_uses(edited, node.output[0], node.input[0])
        remove_node(edited, edited_node)
        rewrites.append(Rewrite(edited, "identity Transpose cleanup", "remove perm identity Transpose"))
    return rewrites


ACTIVE_REWRITERS = (bool_cast_chain_rewrites, reduce_sum_chain_rewrites, identity_transpose_rewrites)


def split_examples(task_id: int) -> dict[str, list[dict[str, Any]]]:
    raw = load_examples(task_id, COMP_DIR)
    return {split: list(raw.get(split, [])) for split in SPLITS}


def flat_examples(examples_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in SPLITS:
        out.extend(examples_by_split[split])
    return out


def session(path: pathlib.Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def decoded_equivalence(
    anchor_path: pathlib.Path,
    candidate_path: pathlib.Path,
    examples_by_split: dict[str, list[dict[str, Any]]],
) -> tuple[bool, dict[str, str], str]:
    anchor = session(anchor_path)
    candidate = session(candidate_path)
    anchor_input = anchor.get_inputs()[0].name
    candidate_input = candidate.get_inputs()[0].name
    anchor_output = anchor.get_outputs()[0].name
    candidate_output = candidate.get_outputs()[0].name
    counts: dict[str, str] = {}
    for split in SPLITS:
        examples = examples_by_split[split]
        passed = 0
        for index, example in enumerate(examples):
            try:
                encoded = encode_grid(example["input"])
                lhs = decode_grid(anchor.run([anchor_output], {anchor_input: encoded})[0])
                rhs = decode_grid(candidate.run([candidate_output], {candidate_input: encoded})[0])
            except Exception as exc:
                counts[split] = f"{passed}/{len(examples)}"
                return False, counts, f"{split}[{index}] runtime/decode error: {exc!r}"
            if lhs != rhs:
                counts[split] = f"{passed}/{len(examples)}"
                return False, counts, f"{split}[{index}] decoded mismatch"
            passed += 1
        counts[split] = f"{passed}/{len(examples)}"
    return True, counts, ""


def load_anchor_scores() -> dict[int, TaskScore]:
    rows: dict[int, TaskScore] = {}
    with ANCHOR_SCORE_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            task_id = int(row["task_id"])
            rows[task_id] = TaskScore(
                task_id=task_id,
                cost=int(row["cost"]) if row["cost"] else None,
                score=float(row["score"]),
                memory=int(row["memory"]) if row["memory"] else None,
                params=int(row["params"]) if row["params"] else None,
                local_pass=int(row["local_pass"]),
                local_total=int(row["local_total"]),
                nodes=int(row["nodes"]),
                file_size=int(row["file_size"]),
                path=row["path"],
                error=row["error"],
            )
    return rows


def empty_record(task_id: int, rule: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "rule": rule,
        "kept": False,
        "applied": False,
        "note": "",
        "current_cost": "",
        "candidate_cost": "",
        "cost_save": "",
        "current_score": "",
        "candidate_score": "",
        "score_gain": "",
        "current_local": "",
        "candidate_local": "",
        "candidate_memory": "",
        "candidate_params": "",
        "candidate_nodes": "",
        "candidate_file_size": "",
        "equiv_train": "",
        "equiv_test": "",
        "equiv_arc_gen": "",
        "error": "",
    }


def evaluate_rewrite(
    task_id: int,
    rewrite: Rewrite,
    anchor_path: pathlib.Path,
    current_score: TaskScore,
    candidate_path: pathlib.Path,
    examples_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    record = empty_record(task_id, rewrite.rule)
    record["note"] = rewrite.note
    record["current_cost"] = current_score.cost if current_score.cost is not None else ""
    record["current_score"] = current_score.score
    record["current_local"] = f"{current_score.local_pass}/{current_score.local_total}"
    try:
        strict_check(rewrite.model)
        onnx.save(strip_value_info(rewrite.model), str(candidate_path))
        candidate = score_onnx(candidate_path, flat_examples(examples_by_split), task_id, n_runs=3)
        record.update(
            candidate_cost=candidate.cost if candidate.cost is not None else "",
            cost_save=(current_score.cost - candidate.cost)
            if current_score.cost is not None and candidate.cost is not None
            else "",
            candidate_score=candidate.score,
            score_gain=candidate.score - current_score.score,
            candidate_local=f"{candidate.local_pass}/{candidate.local_total}",
            candidate_memory=candidate.memory if candidate.memory is not None else "",
            candidate_params=candidate.params if candidate.params is not None else "",
            candidate_nodes=candidate.nodes,
            candidate_file_size=candidate.file_size,
        )
        if candidate.error:
            record["error"] = f"candidate scorer failed: {candidate.error}"
            return record
        if candidate.cost is None or current_score.cost is None:
            record["error"] = "missing measurable n=3 cost"
            return record
        if candidate.local_pass != current_score.local_pass or candidate.local_total != current_score.local_total:
            record["error"] = "n=3 local-pass count changed"
            return record
        if candidate.cost >= current_score.cost or candidate.score <= current_score.score:
            record["error"] = "no n=3 score improvement"
            return record
        same, counts, error = decoded_equivalence(anchor_path, candidate_path, examples_by_split)
        record.update(
            equiv_train=counts.get("train", ""),
            equiv_test=counts.get("test", ""),
            equiv_arc_gen=counts.get("arc-gen", ""),
        )
        if not same:
            record["error"] = error
            return record
        record["kept"] = True
        return record
    except Exception as exc:
        record["error"] = repr(exc)
        return record


def scan_theory(anchor_dir: pathlib.Path) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}

    def add(rule: str, task_id: int, estimate: int, note: str) -> None:
        item = totals.setdefault(
            rule,
            {"rule": rule, "candidate_count": 0, "task_ids": set(), "estimated_intermediate_bytes": 0, "note": note},
        )
        item["candidate_count"] += 1
        item["task_ids"].add(task_id)
        item["estimated_intermediate_bytes"] += max(0, int(estimate))

    for path in sorted(anchor_dir.glob("task*.onnx")):
        task_id = int(path.stem[4:])
        model = onnx.load(str(path))
        types = inferred_types(model)
        producers = producer_map(model)
        uses = use_counts(model)
        for node in model.graph.node:
            if node.op_type == "Cast" and node.input and node.output:
                inner = producers.get(node.input[0])
                if inner and inner.op_type == "Cast" and inner.input and inner.output:
                    info = types.get(inner.output[0])
                    source_info = types.get(inner.input[0])
                    final_info = types.get(node.output[0])
                    if source_info and source_info.elem_type == TensorProto.BOOL:
                        add("exact Cast-chain collapse", task_id, info.nbytes if info else 0, "active: bool source has all-values cast proof")
                    else:
                        add("unsafe Cast-chain collapse", task_id, info.nbytes if info else 0, "rejected: rounding/overflow/domain effects possible")
            if node.op_type == "ReduceSum" and node.input:
                inner = producers.get(node.input[0])
                if inner and inner.op_type == "ReduceSum":
                    info = types.get(inner.output[0])
                    if uses[inner.output[0]] == 1 and get_int_attr(inner, "keepdims", 1) == 1 and get_int_attr(node, "keepdims", 1) == 1:
                        add("ReduceSum-axis fusion/collapse", task_id, info.nbytes if info else 0, "active: static axes and keepdims=1 only")
                    else:
                        add("unsafe/shared ReduceSum chain", task_id, info.nbytes if info else 0, "rejected: shared inner or rank-changing keepdims")
            if node.op_type == "Transpose" and node.input and node.output:
                input_info = types.get(node.input[0])
                output_info = types.get(node.output[0])
                if input_info and output_info and input_info.shape == output_info.shape:
                    perm = next((tuple(int(axis) for axis in attr.ints) for attr in node.attribute if attr.name == "perm"), None)
                    rank = len(input_info.shape)
                    if perm == tuple(range(rank)):
                        add("identity Transpose cleanup", task_id, input_info.nbytes, "active only when perm is identity")
                    else:
                        add("same-shape Transpose", task_id, input_info.nbytes, "rejected: same shape is not a semantic proof")
    rows = []
    for item in totals.values():
        row = dict(item)
        row["task_count"] = len(row.pop("task_ids"))
        rows.append(row)
    rows.sort(key=lambda row: (row["rule"].startswith("unsafe"), -row["estimated_intermediate_bytes"], row["rule"]))
    return rows


def write_candidate_csv(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    fields = list(empty_record(0, "").keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def write_score_csv(path: pathlib.Path, scores: dict[int, TaskScore]) -> None:
    fields = ["task_id", "score", "cost", "memory", "params", "local_pass", "local_total", "nodes", "file_size", "path", "error"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for task_id in range(1, 401):
            row = scores[task_id]
            writer.writerow({field: getattr(row, field) for field in fields})


def write_reports(
    prefix: pathlib.Path,
    records: list[dict[str, Any]],
    theory: list[dict[str, Any]],
    scores: dict[int, TaskScore],
    anchor_scores: dict[int, TaskScore],
) -> None:
    gate_passed = [record for record in records if record.get("kept")]
    applied = [record for record in records if record.get("applied")]
    total = sum(score.score for score in scores.values())
    anchor_total = sum(score.score for score in anchor_scores.values())
    summary = {
        "anchor_dir": rel(ANCHOR_DIR),
        "output_dir": rel(OUT_DIR),
        "candidate_count": len(records),
        "gate_passed_count": len(gate_passed),
        "applied_count": len(applied),
        "applied_tasks": sorted({int(record["task_id"]) for record in applied}),
        "applied_by_rule": dict(Counter(str(record["rule"]) for record in applied)),
        "anchor_total_n3": anchor_total,
        "final_total_n3": total,
        "total_gain_n3": total - anchor_total,
        "theory": theory,
    }
    prefix.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Compiler Backend Sweep v3",
        "",
        f"- Anchor: `{rel(ANCHOR_DIR)}`",
        f"- Output: `{rel(OUT_DIR)}`",
        f"- Active rules: exact bool-source Cast-chain collapse; static keepdims=1 ReduceSum-axis fusion/collapse; identity-only Transpose cleanup.",
        f"- Candidates evaluated: {len(records)}",
        f"- Gate-passed candidates: {len(gate_passed)}",
        f"- Applied rewrites: {len(applied)}",
        f"- Final n=3 total: {total:.6f}",
        f"- Gain vs v4_plus10_compiler2 n=3: {total - anchor_total:.6f}",
        "",
        "## Structural Scan",
        "",
        "| rule | candidates | tasks | estimated intermediate bytes | disposition |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in theory:
        lines.append(
            "| {rule} | {count} | {tasks} | {bytes} | {note} |".format(
                rule=str(row["rule"]).replace("|", "/"),
                count=row["candidate_count"],
                tasks=row["task_count"],
                bytes=row["estimated_intermediate_bytes"],
                note=str(row["note"]).replace("|", "/"),
            )
        )
    lines.extend(["", "## Applied Rewrites", ""])
    if applied:
        lines.extend(["| task | rule | cost save | score gain | decoded train/test/arc-gen | note |", "| ---: | --- | ---: | ---: | --- | --- |"])
        for record in applied:
            lines.append(
                "| {task:03d} | {rule} | {save} | {gain:.6f} | {eq} | {note} |".format(
                    task=int(record["task_id"]),
                    rule=str(record["rule"]).replace("|", "/"),
                    save=record.get("cost_save", ""),
                    gain=float(record.get("score_gain") or 0.0),
                    eq=f"{record.get('equiv_train')}; {record.get('equiv_test')}; {record.get('equiv_arc_gen')}",
                    note=str(record.get("note", "")).replace("|", "/"),
                )
            )
    else:
        lines.append("No candidate was applied.")
    rejected = [record for record in records if not record.get("kept")]
    if rejected:
        lines.extend(["", "## Rejection Summary", ""])
        for reason, count in Counter(str(record.get("error", "")) for record in rejected).most_common(12):
            lines.append(f"- `{reason}`: {count}")
    lines.extend(
        [
            "",
            "## Safety Gates",
            "",
            "- Every applied rewrite passed ONNX checker with `full_check=True`.",
            "- Every applied rewrite passed strict shape inference with `check_type=True`.",
            "- Every applied rewrite matched the v4_plus10 anchor's decoded outputs on all train, test, and arc-gen examples.",
            "- Every applied rewrite improved measured `n_runs=3` cost/score and preserved the n=3 local-pass count.",
            "- Same-shape Transpose candidates were rejected unless `perm` was identity.",
            "- Standalone dynamic-dim/value-info cleanup and visible-only output compression are not active in this branch.",
            "- Edited models are saved after strict inference with stale inferred `value_info` stripped, matching the checker path.",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    require_allowed_paths(args.output_dir, args.report_prefix)
    args.report_prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.anchor_dir, args.output_dir)
    for temp in args.output_dir.glob("_*.onnx*"):
        temp.unlink()

    anchor_scores = load_anchor_scores()
    current_scores = dict(anchor_scores)
    records: list[dict[str, Any]] = []
    theory = scan_theory(args.anchor_dir)
    tasks = args.tasks or list(range(1, 401))

    for task_id in tasks:
        name = f"task{task_id:03d}.onnx"
        anchor_path = args.anchor_dir / name
        current_path = args.output_dir / name
        examples_by_split: dict[str, list[dict[str, Any]]] | None = None
        task_kept = 0
        for rewriter in ACTIVE_REWRITERS:
            while True:
                model = onnx.load(str(current_path))
                rewrites = rewriter(model)
                if not rewrites:
                    break
                if examples_by_split is None:
                    examples_by_split = split_examples(task_id)
                best: tuple[dict[str, Any], pathlib.Path] | None = None
                for index, rewrite in enumerate(rewrites):
                    candidate_path = args.output_dir / f"_task{task_id:03d}_{task_kept}_{index}.onnx"
                    record = evaluate_rewrite(
                        task_id,
                        rewrite,
                        anchor_path,
                        current_scores[task_id],
                        candidate_path,
                        examples_by_split,
                    )
                    records.append(record)
                    if record.get("kept"):
                        kept_path = args.output_dir / f"_task{task_id:03d}_kept_{task_kept}_{index}.onnx"
                        shutil.copy2(candidate_path, kept_path)
                        if best is None or float(record["score_gain"]) > float(best[0]["score_gain"]):
                            if best is not None:
                                best[1].unlink(missing_ok=True)
                            best = (record, kept_path)
                        else:
                            kept_path.unlink(missing_ok=True)
                    candidate_path.unlink(missing_ok=True)
                if best is None:
                    break
                best[0]["applied"] = True
                shutil.copy2(best[1], current_path)
                best[1].unlink(missing_ok=True)
                current_scores[task_id] = score_onnx(current_path, flat_examples(examples_by_split), task_id, n_runs=3)
                task_kept += 1
                print(
                    f"task{task_id:03d} applied {best[0]['rule']} save={best[0]['cost_save']} gain={float(best[0]['score_gain']):.6f}",
                    flush=True,
                )
        if task_kept or task_id % 50 == 0:
            print(f"task{task_id:03d} done kept={task_kept}", flush=True)

    for temp in args.output_dir.glob("_*.onnx*"):
        temp.unlink()
    write_candidate_csv(args.report_prefix.with_suffix(".candidates.csv"), records)
    write_score_csv(args.report_prefix.with_suffix(".score_n3.csv"), current_scores)
    write_reports(args.report_prefix, records, theory, current_scores, anchor_scores)
    applied = [record for record in records if record.get("applied")]
    print(f"applied {len(applied)} rewrites across {len({int(record['task_id']) for record in applied})} tasks")
    print(f"final_total_n3={sum(score.score for score in current_scores.values()):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", type=pathlib.Path, default=ANCHOR_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--report-prefix", type=pathlib.Path, default=REPORT_PREFIX)
    parser.add_argument("--tasks", nargs="*", type=int)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
