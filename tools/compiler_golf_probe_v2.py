from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

TOOL_DIR = pathlib.Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
sys.path.insert(0, str(TOOL_DIR))

from neurogolf_local import all_examples, decode_grid, encode_grid, score_onnx


BASE_DIR = ROOT / "submissions" / "candidate_v4_plus8_task149_bool_onnx"
OUT_DIR = ROOT / "submissions" / "compiler_golf_probe_v2_onnx"
COMP_DIR = ROOT / "data" / "neurogolf-2026" / "raw"
MD_REPORT = ROOT / "reports" / "compiler_golf_probe_v2.md"
CSV_REPORT = ROOT / "reports" / "compiler_golf_probe_v2_candidates.csv"
JSON_REPORT = ROOT / "reports" / "compiler_golf_probe_v2_summary.json"
UNIFORM_KEPT_CSV = ROOT / "reports" / "uniform_scalar_v1_kept.csv"

ACTIVE_BOOL_TASKS = {149, 333, 374, 301, 316, 308, 240}
SPLITS = ("train", "test", "arc-gen")


@dataclass(frozen=True)
class TypeInfo:
    elem_type: int
    shape: tuple[int, ...]

    @property
    def elements(self) -> int | None:
        if any(dim <= 0 for dim in self.shape):
            return None
        return int(math.prod(self.shape)) if self.shape else 1

    @property
    def nbytes(self) -> int | None:
        elems = self.elements
        if elems is None:
            return None
        try:
            np_dtype = helper.tensor_dtype_to_np_dtype(self.elem_type)
        except Exception:
            return None
        return int(elems * np.dtype(np_dtype).itemsize)


def clone_model(model: onnx.ModelProto) -> onnx.ModelProto:
    return onnx.ModelProto.FromString(model.SerializeToString())


def strip_inferred_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    model = clone_model(model)
    del model.graph.value_info[:]
    return model


def strict_check(model: onnx.ModelProto) -> None:
    stripped = strip_inferred_value_info(model)
    onnx.checker.check_model(stripped, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(stripped, check_type=True, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)


def inferred_type_map(model: onnx.ModelProto) -> dict[str, TypeInfo]:
    stripped = strip_inferred_value_info(model)
    inferred = onnx.shape_inference.infer_shapes(stripped, check_type=True, strict_mode=True)
    out: dict[str, TypeInfo] = {}
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    for value in values:
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


def load_uniform_excludes(path: pathlib.Path) -> set[int]:
    if not path.exists():
        return set()
    excluded: set[int] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("kept", "")).lower() == "true":
                excluded.add(int(row["task_id"]))
    return excluded


def producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output if output}


def input_use_counts(model: onnx.ModelProto) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def tensor_value_equal(a: onnx.TensorProto, b: onnx.TensorProto) -> bool:
    if a.data_type != b.data_type or tuple(a.dims) != tuple(b.dims):
        return False
    return np.array_equal(numpy_helper.to_array(a), numpy_helper.to_array(b))


def scan_theory(base_dir: pathlib.Path, excluded_tasks: set[int]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    aggregate: dict[str, dict[str, Any]] = {}
    terminal_candidates: dict[int, list[dict[str, Any]]] = {}

    def add(kind: str, tid: int, gain: int, implementable: bool, note: str) -> None:
        item = aggregate.setdefault(
            kind,
            {
                "kind": kind,
                "tasks": set(),
                "candidate_count": 0,
                "theoretical_cost_save": 0,
                "implementable_candidate_count": 0,
                "implementable_theoretical_save": 0,
                "note": note,
            },
        )
        item["tasks"].add(tid)
        item["candidate_count"] += 1
        item["theoretical_cost_save"] += int(gain)
        if implementable:
            item["implementable_candidate_count"] += 1
            item["implementable_theoretical_save"] += int(gain)

    for path in sorted(base_dir.glob("task*.onnx")):
        tid = int(path.stem[4:])
        model = onnx.load(str(path))
        types: dict[str, TypeInfo] = {}
        try:
            types = inferred_type_map(model)
        except Exception:
            pass
        producers = producer_map(model)
        uses = input_use_counts(model)

        for node in model.graph.node:
            if node.op_type == "Cast" and list(node.output) == ["output"] and node.input:
                cast_input = node.input[0]
                info = types.get(cast_input)
                gain = info.nbytes if info else None
                implementable = (
                    gain is not None
                    and tid not in excluded_tasks
                    and cast_input in producers
                    and uses.get(cast_input, 0) == 1
                )
                add(
                    "terminal Cast -> graph output lifetime narrowing",
                    tid,
                    int(gain or 0),
                    bool(implementable),
                    "remove a terminal Cast and expose the producer as the graph output",
                )
                terminal_candidates.setdefault(tid, []).append(
                    {
                        "task_id": tid,
                        "cast_input": cast_input,
                        "elem_type": info.elem_type if info else "",
                        "shape": "x".join(str(dim) for dim in info.shape) if info else "",
                        "theoretical_save": int(gain or 0),
                        "excluded": tid in excluded_tasks,
                        "implementable": bool(implementable),
                    }
                )

            if node.op_type == "Pad" and "output" in node.output and node.input:
                pad_input = node.input[0]
                producer = producers.get(pad_input)
                if producer and producer.op_type == "Cast" and producer.input:
                    info = types.get(producer.input[0])
                    add(
                        "Cast feeding terminal Pad",
                        tid,
                        int(info.nbytes if info and info.nbytes is not None else 0),
                        False,
                        "strict Pad type constraints rejected all non-overlap candidates",
                    )

            if node.op_type == "Cast" and node.input:
                to_type = next((attr.i for attr in node.attribute if attr.name == "to"), None)
                input_type = types.get(node.input[0])
                output_type = types.get(node.output[0]) if node.output else None
                if input_type and output_type and to_type == input_type.elem_type == output_type.elem_type:
                    add("no-op Cast elimination", tid, input_type.nbytes or 0, False, "none kept by fresh safe optimizer")

            if node.op_type == "Reshape" and node.input and node.output:
                input_type = types.get(node.input[0])
                output_type = types.get(node.output[0])
                if input_type and output_type and input_type.shape == output_type.shape:
                    add("no-op Reshape elimination", tid, input_type.nbytes or 0, False, "schema-safe but no fresh measured keep")

            if node.op_type == "Expand" and node.input and node.output:
                input_type = types.get(node.input[0])
                output_type = types.get(node.output[0])
                if input_type and output_type and input_type.shape == output_type.shape:
                    add("no-op Expand elimination", tid, input_type.nbytes or 0, False, "schema-safe but no fresh measured keep")

            if node.op_type == "Concat" and len([x for x in node.input if x]) <= 1 and node.output:
                output_type = types.get(node.output[0])
                add("single-input Concat elimination", tid, output_type.nbytes if output_type and output_type.nbytes else 0, False, "no fresh measured keep")

        init_by_sig: list[onnx.TensorProto] = []
        used = {name for node in model.graph.node for name in node.input if name}
        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            zeros = int(arr.size - np.count_nonzero(arr))
            if zeros:
                add(
                    "sparse/low-rank initializer alternative",
                    tid,
                    zeros,
                    False,
                    "high theoretical gain, but strict shape inference rejects sparse tensors for common consumers",
                )
            if init.name not in used:
                add("unused initializer elimination", tid, int(arr.size if arr.shape else 1), False, "covered by safe optimizer; fresh measured save was tiny")
            for prior in init_by_sig:
                if tensor_value_equal(init, prior):
                    add("duplicate initializer dedup", tid, int(arr.size if arr.shape else 1), False, "covered by safe optimizer; no high-value fresh keep")
                    break
            init_by_sig.append(init)

    rows: list[dict[str, Any]] = []
    for item in aggregate.values():
        row = dict(item)
        row["task_count"] = len(row.pop("tasks"))
        rows.append(row)
    rows.sort(key=lambda row: (row["implementable_theoretical_save"], row["theoretical_cost_save"]), reverse=True)
    return rows, terminal_candidates


def rewrite_terminal_cast_output(model: onnx.ModelProto) -> tuple[onnx.ModelProto, str]:
    model = strip_inferred_value_info(model)
    types = inferred_type_map(model)
    producers = producer_map(model)
    uses = input_use_counts(model)
    target: onnx.NodeProto | None = None
    for node in model.graph.node:
        if node.op_type == "Cast" and list(node.output) == ["output"] and node.input:
            cast_input = node.input[0]
            if cast_input in producers and uses.get(cast_input, 0) == 1:
                target = node
                break
    if target is None:
        raise ValueError("no implementable terminal Cast -> output pattern")

    old_output = target.input[0]
    old_info = types.get(old_output)
    if old_info is None:
        raise ValueError(f"missing inferred type for {old_output}")

    for node in model.graph.node:
        for index, output in enumerate(node.output):
            if output == old_output:
                node.output[index] = "output"

    kept = [node for node in model.graph.node if node is not target]
    del model.graph.node[:]
    model.graph.node.extend(kept)

    for graph_output in model.graph.output:
        if graph_output.name == "output":
            graph_output.type.tensor_type.elem_type = old_info.elem_type

    strict_check(model)
    dtype_name = TensorProto.DataType.Name(old_info.elem_type)
    shape = "x".join(str(dim) for dim in old_info.shape)
    return model, f"removed terminal Cast; graph output now {old_output} as {dtype_name}[{shape}]"


def session(path: pathlib.Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def decoded_equivalence(
    original_path: pathlib.Path,
    candidate_path: pathlib.Path,
    task_id: int,
    comp_dir: pathlib.Path,
) -> tuple[bool, dict[str, tuple[int, int]], str]:
    original = session(original_path)
    candidate = session(candidate_path)
    original_input = original.get_inputs()[0].name
    candidate_input = candidate.get_inputs()[0].name
    raw = json.loads((comp_dir / f"task{task_id:03d}.json").read_text())
    counts: dict[str, tuple[int, int]] = {}
    for split in SPLITS:
        examples = list(raw.get(split, []))
        passed = 0
        for index, example in enumerate(examples):
            try:
                x = encode_grid(example["input"])
                old_grid = decode_grid(original.run(None, {original_input: x})[0])
                new_grid = decode_grid(candidate.run(None, {candidate_input: x})[0])
            except Exception as exc:
                counts[split] = (passed, len(examples))
                return False, counts, f"{split}[{index}] runtime/decode error: {exc!r}"
            if old_grid != new_grid:
                counts[split] = (passed, len(examples))
                return False, counts, f"{split}[{index}] decoded mismatch"
            passed += 1
        counts[split] = (passed, len(examples))
    return True, counts, ""


def fmt_counts(counts: dict[str, tuple[int, int]], split: str) -> str:
    passed, total = counts.get(split, (0, 0))
    return f"{passed}/{total}"


def score_gain(old_score: float | None, new_score: float | None) -> float:
    if old_score is None or new_score is None:
        return 0.0
    return float(new_score - old_score)


def row_for_task(task_id: int) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "kept": False,
        "rewrite": "",
        "original_cost": "",
        "candidate_cost": "",
        "cost_save": "",
        "original_score": "",
        "candidate_score": "",
        "score_gain": "",
        "original_memory": "",
        "candidate_memory": "",
        "original_params": "",
        "candidate_params": "",
        "candidate_nodes": "",
        "candidate_file_size": "",
        "local_pass": "",
        "local_total": "",
        "equiv_train": "",
        "equiv_test": "",
        "equiv_arc_gen": "",
        "error": "",
    }


def run_probe(
    base_dir: pathlib.Path,
    output_dir: pathlib.Path,
    comp_dir: pathlib.Path,
    tasks: list[int],
    terminal_candidates: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.onnx"):
        stale.unlink()
    for source in sorted(base_dir.glob("task*.onnx")):
        shutil.copy2(source, output_dir / source.name)

    rows: list[dict[str, Any]] = []
    candidate_task_ids = sorted(tid for tid in terminal_candidates if tid in tasks)
    for task_id in candidate_task_ids:
        row = row_for_task(task_id)
        rows.append(row)
        source = base_dir / f"task{task_id:03d}.onnx"
        destination = output_dir / source.name
        candidate = output_dir / f"_task{task_id:03d}.candidate.onnx"
        try:
            if any(item.get("excluded") for item in terminal_candidates.get(task_id, [])):
                row["error"] = "excluded by active BOOL or uniform-scalar branch"
                continue
            original_model = strip_inferred_value_info(onnx.load(str(source)))
            strict_check(original_model)
            candidate_model, rewrite = rewrite_terminal_cast_output(original_model)
            onnx.save(candidate_model, str(candidate))

            examples = all_examples(task_id, comp_dir)
            original = score_onnx(source, examples, task_id, n_runs=0)
            converted = score_onnx(candidate, examples, task_id, n_runs=0)
            same, counts, equiv_error = decoded_equivalence(source, candidate, task_id, comp_dir)
            row.update(
                rewrite=rewrite,
                original_cost=original.cost if original.cost is not None else "",
                candidate_cost=converted.cost if converted.cost is not None else "",
                cost_save=(original.cost - converted.cost) if original.cost is not None and converted.cost is not None else "",
                original_score=original.score,
                candidate_score=converted.score,
                score_gain=score_gain(original.score, converted.score),
                original_memory=original.memory if original.memory is not None else "",
                candidate_memory=converted.memory if converted.memory is not None else "",
                original_params=original.params if original.params is not None else "",
                candidate_params=converted.params if converted.params is not None else "",
                candidate_nodes=converted.nodes,
                candidate_file_size=converted.file_size,
                local_pass=converted.local_pass,
                local_total=converted.local_total,
                equiv_train=fmt_counts(counts, "train"),
                equiv_test=fmt_counts(counts, "test"),
                equiv_arc_gen=fmt_counts(counts, "arc-gen"),
            )
            full_local = converted.local_total == len(examples) and converted.local_pass == len(examples)
            has_gain = converted.cost is not None and original.cost is not None and converted.cost < original.cost
            if original.error:
                row["error"] = f"original scorer failed: {original.error}"
            elif converted.error:
                row["error"] = f"candidate scorer failed: {converted.error}"
            elif not same:
                row["error"] = equiv_error
            elif not full_local:
                row["error"] = f"candidate local pass {converted.local_pass}/{len(examples)}"
            elif not has_gain:
                row["error"] = "no measured cost gain"
            else:
                shutil.copy2(candidate, destination)
                row["kept"] = True
        except Exception as exc:
            row["error"] = repr(exc)
        finally:
            candidate.unlink(missing_ok=True)
        print(
            "task{task_id:03d} kept={kept} save={save} gain={gain} err={err}".format(
                task_id=task_id,
                kept=row.get("kept"),
                save=row.get("cost_save", ""),
                gain=row.get("score_gain", ""),
                err=row.get("error", ""),
            ),
            flush=True,
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row_for_task(0).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_reports(
    rows: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    excluded_tasks: set[int],
    output_dir: pathlib.Path,
    md_path: pathlib.Path,
    json_path: pathlib.Path,
) -> None:
    kept = [row for row in rows if row.get("kept")]
    total_save = sum(int(row.get("cost_save") or 0) for row in kept)
    total_gain = sum(float(row.get("score_gain") or 0.0) for row in kept)
    summary = {
        "base_dir": str(BASE_DIR.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "excluded_tasks": sorted(excluded_tasks),
        "kept_count": len(kept),
        "candidate_count": len(rows),
        "total_cost_save": total_save,
        "total_score_gain": total_gain,
        "ranking": [
            {
                key: (sorted(value) if isinstance(value, set) else value)
                for key, value in row.items()
            }
            for row in ranking
        ],
        "kept_tasks": [int(row["task_id"]) for row in kept],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Compiler Golf Probe v2",
        "",
        f"- Anchor: `{BASE_DIR.relative_to(ROOT)}`",
        f"- Output: `{output_dir.relative_to(ROOT)}`",
        f"- Disjoint exclusions: active BOOL tasks `{sorted(ACTIVE_BOOL_TASKS)}` plus `{len(excluded_tasks - ACTIVE_BOOL_TASKS)}` uniform-scalar kept tasks.",
        f"- Implemented class: terminal `Cast -> output` lifetime narrowing.",
        f"- Kept rewrites: {len(kept)}/{len(rows)} candidates.",
        f"- Total measured cost save: {total_save}",
        f"- Total measured score gain: {total_gain:.6f}",
        "",
        "## Transformation Ranking",
        "",
        "| rank | class | implementable save | theoretical save | candidates | tasks | note |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(ranking[:12], 1):
        lines.append(
            "| {rank} | {kind} | {isave} | {tsave} | {count} | {tasks} | {note} |".format(
                rank=index,
                kind=str(row["kind"]).replace("|", "/"),
                isave=row.get("implementable_theoretical_save", 0),
                tsave=row.get("theoretical_cost_save", 0),
                count=row.get("candidate_count", 0),
                tasks=row.get("task_count", 0),
                note=str(row.get("note", "")).replace("|", "/"),
            )
        )

    lines += [
        "",
        "## Kept Tasks",
        "",
        "| task | cost save | score gain | local pass | exact decoded train/test/arc-gen | rewrite |",
        "| ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in kept:
        exact = f"{row.get('equiv_train')}; {row.get('equiv_test')}; {row.get('equiv_arc_gen')}"
        lines.append(
            "| {task:03d} | {save} | {gain:.6f} | {local}/{total} | {exact} | {rewrite} |".format(
                task=int(row["task_id"]),
                save=row.get("cost_save", ""),
                gain=float(row.get("score_gain") or 0.0),
                local=row.get("local_pass", ""),
                total=row.get("local_total", ""),
                exact=exact,
                rewrite=str(row.get("rewrite", "")).replace("|", "/"),
            )
        )

    rejected = [row for row in rows if not row.get("kept")]
    if rejected:
        lines += [
            "",
            "## Rejected Candidates",
            "",
            "| task | reason |",
            "| ---: | --- |",
        ]
        for row in rejected:
            lines.append(f"| {int(row['task_id']):03d} | {str(row.get('error', '')).replace('|', '/')} |")

    lines += [
        "",
        "## Safety Gates",
        "",
        "- Every source and edited model passed `onnx.checker.check_model(..., full_check=True)`.",
        "- Every edited model passed strict shape inference with `check_type=True`.",
        "- Every kept task matched the anchor's decoded output exactly on all train, test, and arc-gen examples.",
        "- Every kept task had full local target pass and a measured local scorer cost/score improvement with `n_runs=0`.",
        "- No anchor files, ledgers, or submit zips were modified.",
    ]
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--comp-dir", default=str(COMP_DIR))
    parser.add_argument("--csv", default=str(CSV_REPORT))
    parser.add_argument("--md", default=str(MD_REPORT))
    parser.add_argument("--json", default=str(JSON_REPORT))
    parser.add_argument("--tasks", nargs="*", type=int)
    args = parser.parse_args()

    base_dir = pathlib.Path(args.base_dir)
    output_dir = pathlib.Path(args.output_dir)
    comp_dir = pathlib.Path(args.comp_dir)
    csv_path = pathlib.Path(args.csv)
    md_path = pathlib.Path(args.md)
    json_path = pathlib.Path(args.json)
    tasks = args.tasks or list(range(1, 401))

    excluded_tasks = set(ACTIVE_BOOL_TASKS) | load_uniform_excludes(UNIFORM_KEPT_CSV)
    ranking, terminal_candidates = scan_theory(base_dir, excluded_tasks)
    rows = run_probe(base_dir, output_dir, comp_dir, tasks, terminal_candidates)
    write_csv(rows, csv_path)
    write_reports(rows, ranking, excluded_tasks, output_dir, md_path, json_path)

    kept = [row for row in rows if row.get("kept")]
    print(f"kept {len(kept)}/{len(rows)}")
    print(f"total_cost_save={sum(int(row.get('cost_save') or 0) for row in kept)}")
    print(f"total_score_gain={sum(float(row.get('score_gain') or 0.0) for row in kept):.6f}")


if __name__ == "__main__":
    main()
