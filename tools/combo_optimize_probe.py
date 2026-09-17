"""Try multiple compression pipelines per task; keep the best safe result.

For each task, attempt all permutations/subsets of {onnxsim, onnxoptimizer, int_surgery}.
Keep the candidate with highest score that passes same_decoded_outputs against original.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import onnx
import onnxoptimizer
from onnxsim import simplify

from fp16_surgery import same_decoded_outputs
from int_surgery_probe import int64_data_to_int32
from neurogolf_local import all_examples, score_onnx
from onnx_optimize_probe import SAFE_PASSES


TOOLS = ["opt", "sim", "int"]


def apply_tool(model: onnx.ModelProto, tool: str) -> onnx.ModelProto:
    if tool == "opt":
        out = model
        for _ in range(4):
            new = onnxoptimizer.optimize(out, SAFE_PASSES)
            if new.SerializeToString() == out.SerializeToString():
                break
            out = new
        return out
    if tool == "sim":
        simplified, _ = simplify(model, perform_optimization=True)
        return simplified
    if tool == "int":
        return int64_data_to_int32(model)
    raise ValueError(tool)


def all_pipelines():
    # subsets size 1..3, all orderings
    seen = set()
    for r in (1, 2, 3):
        for combo in itertools.permutations(TOOLS, r):
            key = combo
            if key in seen:
                continue
            seen.add(key)
            yield combo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--report", default="reports/combo_optimize_probe.json")
    parser.add_argument("--tasks", nargs="+", type=int)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    comp_dir = pathlib.Path(args.comp_dir)
    report_path = pathlib.Path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or list(range(1, 401))
    pipelines = list(all_pipelines())
    records = []
    for task_id in tasks:
        source = input_dir / f"task{task_id:03d}.onnx"
        candidate = output_dir / f"task{task_id:03d}.onnx"
        if not source.exists():
            records.append({"task_id": task_id, "kept": False, "error": "missing source"})
            continue
        examples = all_examples(task_id, comp_dir)
        try:
            original = score_onnx(source, examples, task_id, n_runs=3)
        except Exception as exc:
            records.append({"task_id": task_id, "kept": False, "error": repr(exc)[:200]})
            continue

        if original.cost is None:
            records.append({
                "task_id": task_id,
                "original_cost": original.cost,
                "original_score": original.score,
                "best_cost": None,
                "best_score": original.score,
                "delta": 0.0,
                "pipeline": [],
                "kept": False,
                "error": f"original_unmeasurable:{original.error or 'cost_none'}",
            })
            continue

        best_score = original.score
        best_cost = original.cost
        best_pipeline = ()
        best_path = None

        for pipeline in pipelines:
            try:
                model = onnx.load(str(source))
                for tool in pipeline:
                    model = apply_tool(model, tool)
                tmp_path = output_dir / f"task{task_id:03d}__{'_'.join(pipeline)}.onnx"
                onnx.save(model, str(tmp_path))
                cscore = score_onnx(tmp_path, examples, task_id, n_runs=3)
                if cscore.cost is not None and cscore.score > best_score:
                    if same_decoded_outputs(source, tmp_path, examples):
                        best_score = cscore.score
                        best_cost = cscore.cost
                        best_pipeline = pipeline
                        best_path = tmp_path
                # cleanup tmp candidates other than best path on the fly
            except Exception:
                continue

        record = {
            "task_id": task_id,
            "original_cost": original.cost,
            "original_score": original.score,
            "best_cost": best_cost,
            "best_score": best_score,
            "delta": best_score - original.score,
            "pipeline": list(best_pipeline),
            "kept": bool(best_path is not None),
        }
        if best_path is not None:
            # Move best to canonical file
            onnx.save(onnx.load(str(best_path)), str(candidate))
        # Clean tmp files
        for tmp in output_dir.glob(f"task{task_id:03d}__*.onnx"):
            tmp.unlink(missing_ok=True)
        records.append(record)
        if record["kept"]:
            print(f"task{task_id:03d}: {'+'.join(best_pipeline)} save={(original.cost or 0)-best_cost} delta={record['delta']:+.4f}", flush=True)

    report_path.write_text(json.dumps(records, indent=2))
    kept = [r for r in records if r.get("kept")]
    print(f"kept {len(kept)} tasks, total delta {sum(r['delta'] for r in kept):+.4f}")


if __name__ == "__main__":
    main()
