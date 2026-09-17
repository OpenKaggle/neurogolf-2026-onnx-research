from __future__ import annotations

import argparse
import csv
import math
import pathlib
import shutil
import sys
from collections.abc import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

TOOL_DIR = pathlib.Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
sys.path.insert(0, str(TOOL_DIR))

from neurogolf_local import all_examples, decode_grid, encode_grid, score_onnx


BASE_DIR = ROOT / "submissions" / "candidate_v4_plus8_task149_bool_onnx"
OUT_DIR = ROOT / "submissions" / "bool_memory_probe_onnx"
CANDIDATE_DIR = ROOT / "submissions" / "bool_memory_probe_candidates"
CSV_REPORT = ROOT / "reports" / "bool_memory_probe.csv"
MD_REPORT = ROOT / "reports" / "bool_memory_probe.md"
DEFAULT_TASKS = [333, 374, 301, 316, 308, 240]
COMP_DIR = ROOT / "data" / "neurogolf-2026" / "raw"

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL


RewriteFn = Callable[[onnx.ModelProto], tuple[onnx.ModelProto, str]]


def _clone(model: onnx.ModelProto) -> onnx.ModelProto:
    return onnx.ModelProto.FromString(model.SerializeToString())


def _set_output_type(model: onnx.ModelProto, elem_type: int) -> None:
    for output in model.graph.output:
        output.type.tensor_type.elem_type = elem_type


def _clear_inferred_types(model: onnx.ModelProto) -> None:
    del model.graph.value_info[:]


def _node_outputs(node: onnx.NodeProto) -> set[str]:
    return {name for name in node.output if name}


def _drop_nodes_by_output(model: onnx.ModelProto, outputs: set[str]) -> None:
    kept = [node for node in model.graph.node if not (_node_outputs(node) & outputs)]
    del model.graph.node[:]
    model.graph.node.extend(kept)


def _replace_initializer(model: onnx.ModelProto, tensor: onnx.TensorProto) -> None:
    graph = model.graph
    for index, init in enumerate(graph.initializer):
        if init.name == tensor.name:
            graph.initializer[index].CopyFrom(tensor)
            return
    graph.initializer.append(tensor)


def _add_bool_scalar(model: onnx.ModelProto, name: str, value: bool) -> None:
    _replace_initializer(model, numpy_helper.from_array(np.asarray(value, dtype=np.bool_), name))


def _add_bool_mask(model: onnx.ModelProto, name: str, values: list[bool]) -> None:
    arr = np.asarray(values, dtype=np.bool_).reshape(1, 10, 1, 1)
    _replace_initializer(model, numpy_helper.from_array(arr, name))


def _used_initializer_names(model: onnx.ModelProto) -> set[str]:
    used: set[str] = set()
    for node in model.graph.node:
        used.update(value for value in node.input if value)
    return used


def _prune_unused_initializers(model: onnx.ModelProto) -> None:
    used = _used_initializer_names(model)
    kept = [init for init in model.graph.initializer if init.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def _strict_checked(model: onnx.ModelProto) -> onnx.ModelProto:
    _clear_inferred_types(model)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def _set_node_inputs(node: onnx.NodeProto, inputs: list[str]) -> None:
    del node.input[:]
    node.input.extend(inputs)


def rewrite_task308(model: onnx.ModelProto) -> tuple[onnx.ModelProto, str]:
    model = _clone(model)
    _drop_nodes_by_output(model, {"out7"})
    for node in model.graph.node:
        if node.op_type == "Pad" and list(node.output) == ["output"]:
            _set_node_inputs(node, ["out7_b", "pads_to_30"])
            break
    else:
        raise ValueError("task308 Pad(output) not found")
    _set_output_type(model, BOOL)
    return _strict_checked(model), "drop Cast(out7_b->float); Pad bool out7_b directly"


def rewrite_task316(model: onnx.ModelProto) -> tuple[onnx.ModelProto, str]:
    model = _clone(model)
    _drop_nodes_by_output(model, {"onehot_3x3"})
    for node in model.graph.node:
        if node.op_type == "Pad" and list(node.output) == ["output"]:
            _set_node_inputs(node, ["onehot_b", "pads_to_30"])
            break
    else:
        raise ValueError("task316 Pad(output) not found")
    _set_output_type(model, BOOL)
    return _strict_checked(model), "drop Cast(onehot_b->float); Pad bool onehot_b directly"


def rewrite_task301(model: onnx.ModelProto) -> tuple[onnx.ModelProto, str]:
    model = _clone(model)
    color_ids = np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1)
    _replace_initializer(model, numpy_helper.from_array(color_ids, "bm_color_ids_i64"))
    _drop_nodes_by_output(model, {"valid_mask", "tail_hot", "tail_cond", "output"})
    model.graph.node.extend(
        [
            helper.make_node("Unsqueeze", ["tail_idx_squeezed"], ["bm_tail_idx_u"], axes=[1], name="bm_tail_idx_u"),
            helper.make_node("Equal", ["bm_tail_idx_u", "bm_color_ids_i64"], ["bm_tail_onehot_b"], name="bm_tail_onehot_b"),
            helper.make_node("Unsqueeze", ["valid_bool"], ["bm_valid_u"], axes=[0, 1], name="bm_valid_u"),
            helper.make_node("And", ["bm_tail_onehot_b", "bm_valid_u"], ["output"], name="output"),
        ]
    )
    _set_output_type(model, BOOL)
    _prune_unused_initializers(model)
    return _strict_checked(model), "replace final OneHot/Where with bool Equal tail index against color ids"


def rewrite_task374(model: onnx.ModelProto) -> tuple[onnx.ModelProto, str]:
    model = _clone(model)
    _add_bool_scalar(model, "bm_false", False)
    _drop_nodes_by_output(model, {"ch1", "ch2", "ch4", "zero_ch", "out10", "output"})
    model.graph.node.extend(
        [
            helper.make_node("Greater", ["ch0", "ZERO_F"], ["ch0_b"], name="ch0_b"),
            helper.make_node("And", ["ch0_b", "bm_false"], ["zero_b"], name="zero_b"),
            helper.make_node(
                "Concat",
                [
                    "ch0_b",
                    "ch1_b",
                    "ch2_b",
                    "zero_b",
                    "ch4_b",
                    "zero_b",
                    "zero_b",
                    "zero_b",
                    "zero_b",
                    "zero_b",
                ],
                ["out10_b"],
                axis=1,
                name="out10_b",
            ),
            helper.make_node("Pad", ["out10_b", "pads_to_30"], ["output"], mode="constant", name="output"),
        ]
    )
    _set_output_type(model, BOOL)
    _prune_unused_initializers(model)
    return _strict_checked(model), "bool ch0/ch1/ch2/ch4 concat; Pad bool out10_b"


def rewrite_task333(model: onnx.ModelProto) -> tuple[onnx.ModelProto, str]:
    model = _clone(model)
    _add_bool_mask(
        model,
        "bm_add_color_mask",
        [False, True, True, False, True, True, True, True, True, True],
    )
    _add_bool_mask(
        model,
        "bm_bg_channel_mask",
        [True, False, False, False, False, False, False, False, False, False],
    )
    _drop_nodes_by_output(
        model,
        {
            "v60",
            "v61",
            "v62",
            "v63",
            "v64",
            "v65",
            "v66",
            "v67",
            "v68",
            "v69",
            "v70",
            "v71",
            "v72",
            "v73",
            "v74",
            "v75",
            "v76",
            "v77",
            "v78",
            "v79",
            "output",
        },
    )
    model.graph.node.extend(
        [
            helper.make_node("Greater", ["v20", "v1"], ["bm_input_b"], name="bm_input_b"),
            helper.make_node("Equal", ["v59", "v2"], ["bm_eq_color_b"], name="bm_eq_color_b"),
            helper.make_node("And", ["bm_eq_color_b", "bm_add_color_mask"], ["bm_add_color_b"], name="bm_add_color_b"),
            helper.make_node("Or", ["bm_input_b", "bm_add_color_b"], ["bm_out_pre_b"], name="bm_out_pre_b"),
            helper.make_node("Greater", ["v59", "v11"], ["bm_v59_nonzero_b"], name="bm_v59_nonzero_b"),
            helper.make_node(
                "And",
                ["bm_v59_nonzero_b", "bm_bg_channel_mask"],
                ["bm_clear_bg_b"],
                name="bm_clear_bg_b",
            ),
            helper.make_node("Not", ["bm_clear_bg_b"], ["bm_keep_b"], name="bm_keep_b"),
            helper.make_node("And", ["bm_out_pre_b", "bm_keep_b"], ["bm_out_b"], name="bm_out_b"),
            helper.make_node("Cast", ["bm_out_b"], ["bm_out_f"], to=F32, name="bm_out_f"),
            helper.make_node("Pad", ["bm_out_f", "v10", "v11"], ["output"], mode="constant", name="output"),
        ]
    )
    return _strict_checked(model), "replace final float one-hot Sum/Concat with bool logic, then Cast before opset11 Pad"


def rewrite_task240(model: onnx.ModelProto) -> tuple[onnx.ModelProto, str]:
    model = _clone(model)
    base = next((init for init in model.graph.initializer if init.name == "base_bg"), None)
    if base is None:
        raise ValueError("task240 base_bg initializer not found")
    base_bool = numpy_helper.to_array(base) > 0.0
    _replace_initializer(model, numpy_helper.from_array(base_bool.astype(np.bool_), "base_bg"))
    _drop_nodes_by_output(model, {"comp_onehot_f"})
    for node in model.graph.node:
        if node.op_type == "Reshape" and list(node.output) == ["updates"]:
            _set_node_inputs(node, ["comp_onehot_b", "shape_updates"])
            break
    else:
        raise ValueError("task240 Reshape(updates) not found")
    _set_output_type(model, BOOL)
    _prune_unused_initializers(model)
    return _strict_checked(model), "convert base_bg/updates to bool and ScatterND bool output"


REWRITES: dict[int, RewriteFn] = {
    333: rewrite_task333,
    374: rewrite_task374,
    301: rewrite_task301,
    316: rewrite_task316,
    308: rewrite_task308,
    240: rewrite_task240,
}


def _session(path: pathlib.Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def decoded_equivalence(
    original_path: pathlib.Path,
    candidate_path: pathlib.Path,
    splits: dict[str, list[dict]],
) -> tuple[bool, dict[str, tuple[int, int]], str]:
    original = _session(original_path)
    candidate = _session(candidate_path)
    original_input = original.get_inputs()[0].name
    candidate_input = candidate.get_inputs()[0].name
    counts: dict[str, tuple[int, int]] = {}
    for split_name in ("train", "test", "arc-gen"):
        examples = splits.get(split_name, [])
        passed = 0
        for index, example in enumerate(examples):
            x = encode_grid(example["input"])
            old_out = original.run(None, {original_input: x})[0]
            new_out = candidate.run(None, {candidate_input: x})[0]
            if decode_grid(old_out) != decode_grid(new_out):
                counts[split_name] = (passed, len(examples))
                return False, counts, f"{split_name}[{index}] decoded mismatch"
            passed += 1
        counts[split_name] = (passed, len(examples))
    return True, counts, ""


def _score_dict(result) -> dict[str, int | float | str | None]:
    return {
        "score": result.score,
        "cost": result.cost,
        "memory": result.memory,
        "params": result.params,
        "local_pass": result.local_pass,
        "local_total": result.local_total,
        "error": result.error,
    }


def _score_gain(old_score: float | None, new_score: float | None) -> float:
    if old_score is None or new_score is None:
        return 0.0
    return float(new_score - old_score)


def _format_counts(counts: dict[str, tuple[int, int]], name: str) -> str:
    passed, total = counts.get(name, (0, 0))
    return f"{passed}/{total}"


def probe_task(
    task_id: int,
    base_dir: pathlib.Path,
    output_dir: pathlib.Path,
    candidate_dir: pathlib.Path,
    comp_dir: pathlib.Path,
) -> dict[str, object]:
    source = base_dir / f"task{task_id:03d}.onnx"
    destination = output_dir / source.name
    candidate = candidate_dir / source.name
    row: dict[str, object] = {
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
        "local_pass": "",
        "local_total": "",
        "equiv_train": "",
        "equiv_test": "",
        "equiv_arc_gen": "",
        "error": "",
    }
    if not source.exists():
        row["error"] = "missing source"
        return row

    shutil.copy2(source, destination)
    if task_id not in REWRITES:
        row["error"] = "no rewrite registered"
        return row

    splits_raw = __import__("json").loads((comp_dir / f"task{task_id:03d}.json").read_text())
    splits = {
        "train": list(splits_raw.get("train", [])),
        "test": list(splits_raw.get("test", [])),
        "arc-gen": list(splits_raw.get("arc-gen", [])),
    }
    examples = splits["train"] + splits["test"] + splits["arc-gen"]

    try:
        original_model = _strict_checked(onnx.load(str(source)))
        original = score_onnx(source, examples, task_id, n_runs=0)
        if original.cost is None or original.error:
            row["error"] = f"original scorer failed: {original.error or 'cost_none'}"
            return row

        candidate_model, rewrite = REWRITES[task_id](original_model)
        onnx.save(candidate_model, str(candidate))

        converted = score_onnx(candidate, examples, task_id, n_runs=0)
        same, counts, equiv_error = decoded_equivalence(source, candidate, splits)
        row.update(
            rewrite=rewrite,
            original_cost=original.cost,
            candidate_cost=converted.cost if converted.cost is not None else "",
            cost_save=(original.cost - converted.cost) if converted.cost is not None else "",
            original_score=original.score,
            candidate_score=converted.score,
            score_gain=_score_gain(original.score, converted.score),
            original_memory=original.memory,
            candidate_memory=converted.memory if converted.memory is not None else "",
            original_params=original.params,
            candidate_params=converted.params if converted.params is not None else "",
            local_pass=converted.local_pass,
            local_total=converted.local_total,
            equiv_train=_format_counts(counts, "train"),
            equiv_test=_format_counts(counts, "test"),
            equiv_arc_gen=_format_counts(counts, "arc-gen"),
        )

        full_local_pass = converted.local_total == len(examples) and converted.local_pass == len(examples)
        has_gain = converted.cost is not None and converted.cost < original.cost and converted.score > original.score
        if converted.error:
            row["error"] = f"candidate scorer failed: {converted.error}"
        elif not same:
            row["error"] = equiv_error
        elif not full_local_pass:
            row["error"] = f"candidate local pass {converted.local_pass}/{len(examples)}"
        elif not has_gain:
            row["error"] = "no measured gain"
        else:
            shutil.copy2(candidate, destination)
            row["kept"] = True
    except Exception as exc:
        row["error"] = repr(exc)
    finally:
        if not row.get("kept"):
            candidate.unlink(missing_ok=True)
    return row


def write_reports(rows: list[dict[str, object]], csv_path: pathlib.Path, md_path: pathlib.Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "kept",
        "rewrite",
        "original_cost",
        "candidate_cost",
        "cost_save",
        "original_score",
        "candidate_score",
        "score_gain",
        "original_memory",
        "candidate_memory",
        "original_params",
        "candidate_params",
        "local_pass",
        "local_total",
        "equiv_train",
        "equiv_test",
        "equiv_arc_gen",
        "error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    kept = [row for row in rows if row.get("kept")]
    total_gain = sum(float(row.get("score_gain") or 0.0) for row in kept)
    total_save = sum(int(row.get("cost_save") or 0) for row in kept)
    lines = [
        "# Bool Memory Probe",
        "",
        f"Base: `{BASE_DIR.relative_to(ROOT)}`",
        f"Output: `{OUT_DIR.relative_to(ROOT)}`",
        f"Kept rewrites: {len(kept)}/{len(rows)}",
        f"Total score gain: {total_gain:.6f}",
        f"Total measured cost save: {total_save}",
        "",
        "| task | kept | cost save | score gain | local pass | exact decoded train/test/arc-gen | rewrite | error |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        exact = (
            f"train {row.get('equiv_train','')}; "
            f"test {row.get('equiv_test','')}; "
            f"arc-gen {row.get('equiv_arc_gen','')}"
        )
        score_gain = row.get("score_gain", "")
        if isinstance(score_gain, float) and math.isfinite(score_gain):
            score_gain_s = f"{score_gain:.6f}"
        else:
            score_gain_s = str(score_gain)
        lines.append(
            "| {task_id:03d} | {kept} | {cost_save} | {score_gain} | {local_pass}/{local_total} | {exact} | {rewrite} | {error} |".format(
                task_id=int(row["task_id"]),
                kept="yes" if row.get("kept") else "no",
                cost_save=row.get("cost_save", ""),
                score_gain=score_gain_s,
                local_pass=row.get("local_pass", ""),
                local_total=row.get("local_total", ""),
                exact=exact,
                rewrite=str(row.get("rewrite", "")).replace("|", "/"),
                error=str(row.get("error", "")).replace("|", "/"),
            )
        )
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--candidate-dir", default=str(CANDIDATE_DIR))
    parser.add_argument("--comp-dir", default=str(COMP_DIR))
    parser.add_argument("--csv", default=str(CSV_REPORT))
    parser.add_argument("--md", default=str(MD_REPORT))
    parser.add_argument("--tasks", nargs="+", type=int, default=DEFAULT_TASKS)
    args = parser.parse_args()

    base_dir = pathlib.Path(args.base_dir)
    output_dir = pathlib.Path(args.output_dir)
    candidate_dir = pathlib.Path(args.candidate_dir)
    comp_dir = pathlib.Path(args.comp_dir)
    csv_path = pathlib.Path(args.csv)
    md_path = pathlib.Path(args.md)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(base_dir.glob("task*.onnx")):
        shutil.copy2(source, output_dir / source.name)

    rows = []
    for task_id in args.tasks:
        row = probe_task(task_id, base_dir, output_dir, candidate_dir, comp_dir)
        rows.append(row)
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

    write_reports(rows, csv_path, md_path)
    kept = [row for row in rows if row.get("kept")]
    print(f"kept {len(kept)}/{len(rows)}")
    print(f"total_score_gain={sum(float(row.get('score_gain') or 0.0) for row in kept):.6f}")
    print(f"total_cost_save={sum(int(row.get('cost_save') or 0) for row in kept)}")
    print(f"report_csv={csv_path}")
    print(f"report_md={md_path}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
