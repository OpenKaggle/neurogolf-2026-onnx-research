"""Full-example verification for cherry-pick swap candidates.

For each candidate (task_id, source_dir), runs ALL examples (train + test + arc-gen)
through the candidate ONNX and reports the pass count. Lets us filter out tasks where
the candidate network does NOT pass every local example.

Usage:
  .venv/bin/python tools/verify_swap_candidates.py \
    --source data/konbu17_v36 \
    --tasks 319 285 219 255 044 153 233 133 157 076 118 023 209 066 101 \
    --out reports/konbu17_full_verify.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import onnxruntime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples, encode_grid


def verify_task(task_id: int, source_dir: pathlib.Path, comp_dir: pathlib.Path) -> dict:
    onnx_path = source_dir / f"task{task_id:03d}.onnx"
    if not onnx_path.is_file():
        return {"task": task_id, "ok": False, "error": "missing_file", "pass": 0, "total": 0}
    examples = all_examples(task_id, comp_dir)
    total = len(examples)
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    try:
        session = onnxruntime.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        return {"task": task_id, "ok": False, "error": f"load:{exc}", "pass": 0, "total": total}
    passed = 0
    first_fail = None
    for idx, example in enumerate(examples):
        try:
            out = session.run(["output"], {"input": encode_grid(example["input"])})[0]
            target = encode_grid(example["output"]) > 0.0
            if np.array_equal(out > 0.0, target):
                passed += 1
            else:
                if first_fail is None:
                    first_fail = idx
        except Exception as exc:
            if first_fail is None:
                first_fail = idx
    return {
        "task": task_id,
        "ok": passed == total,
        "pass": passed,
        "total": total,
        "first_fail": first_fail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source onnx directory")
    parser.add_argument("--tasks", nargs="+", required=True, type=int)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--out", default="reports/swap_verify.json")
    args = parser.parse_args()

    source_dir = pathlib.Path(args.source)
    comp_dir = pathlib.Path(args.comp_dir)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for task_id in args.tasks:
        result = verify_task(task_id, source_dir, comp_dir)
        results.append(result)
        flag = "PASS" if result["ok"] else "FAIL"
        extra = ""
        if not result["ok"] and result.get("first_fail") is not None:
            extra = f" first_fail={result['first_fail']}"
        if result.get("error"):
            extra += f" err={result['error']}"
        print(f"  task{task_id:03d}: {flag} {result['pass']}/{result['total']}{extra}", flush=True)

    summary = {
        "source": args.source,
        "n_tested": len(results),
        "n_pass_all": sum(1 for r in results if r["ok"]),
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}: {summary['n_pass_all']}/{summary['n_tested']} fully passing")


if __name__ == "__main__":
    main()
