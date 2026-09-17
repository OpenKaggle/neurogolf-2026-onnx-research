"""Score individual ONNX files with full example profiling.

For each (task_id, source_dir) pair, computes the cost & score using ALL examples
for profiling (n_runs=0). More accurate than the bundle n=3 scan for monster
candidates whose memory varies across examples.

Usage:
  .venv/bin/python tools/single_task_score.py \
    --source data/konbu17_v36 \
    --tasks 319 285 219 255 \
    --out reports/konbu17_full_cost.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples, score_onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--tasks", nargs="+", required=True, type=int)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--out", default="reports/full_cost.json")
    parser.add_argument("--n-runs", type=int, default=0)
    args = parser.parse_args()

    source_dir = pathlib.Path(args.source)
    comp_dir = pathlib.Path(args.comp_dir)

    results = []
    for task_id in args.tasks:
        examples = all_examples(task_id, comp_dir)
        result = score_onnx(source_dir / f"task{task_id:03d}.onnx", examples, task_id, args.n_runs)
        results.append({
            "task": task_id,
            "score": result.score,
            "cost": result.cost,
            "memory": result.memory,
            "params": result.params,
            "local_pass": result.local_pass,
            "local_total": result.local_total,
            "nodes": result.nodes,
            "error": result.error,
        })
        err = f" err={result.error}" if result.error else ""
        print(f"  task{task_id:03d}: cost={result.cost} score={result.score:.3f} pass={result.local_pass}/{result.local_total}{err}", flush=True)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"source": args.source, "n_runs": args.n_runs, "results": results}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
