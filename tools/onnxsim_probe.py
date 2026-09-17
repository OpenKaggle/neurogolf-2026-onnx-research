"""Run onnxsim per task and keep candidates that improve and pass same_decoded_outputs."""
from __future__ import annotations

import argparse
import json
import pathlib

import onnx
from onnxsim import simplify

from fp16_surgery import same_decoded_outputs
from neurogolf_local import all_examples, score_onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--report", default="reports/onnxsim_probe.json")
    parser.add_argument("--tasks", nargs="+", type=int)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    comp_dir = pathlib.Path(args.comp_dir)
    report_path = pathlib.Path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or list(range(1, 401))
    records = []
    for task_id in tasks:
        source = input_dir / f"task{task_id:03d}.onnx"
        candidate = output_dir / f"task{task_id:03d}.onnx"
        if not source.exists():
            records.append({"task_id": task_id, "kept": False, "error": "missing source"})
            continue
        examples = all_examples(task_id, comp_dir)
        record = {"task_id": task_id, "kept": False, "error": ""}
        try:
            model = onnx.load(str(source))
            original = score_onnx(source, examples, task_id, n_runs=3)
            if original.cost is None:
                record["error"] = f"original_unmeasurable:{original.error or 'cost_none'}"
                records.append(record)
                continue
            simplified, check = simplify(model, perform_optimization=True)
            onnx.save(simplified, str(candidate))
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
                check=bool(check),
                kept=bool(same and converted.cost is not None and converted.score > original.score),
            )
        except Exception as exc:
            record["error"] = repr(exc)[:200]
        records.append(record)
        if record.get("kept"):
            print(f"task{task_id:03d}: kept save={record['save']} delta={record['delta']:+.4f}", flush=True)

    report_path.write_text(json.dumps(records, indent=2))
    kept = [r for r in records if r.get("kept")]
    print(f"kept {len(kept)} tasks, total delta {sum(r['delta'] for r in kept):+.4f}")


if __name__ == "__main__":
    main()
