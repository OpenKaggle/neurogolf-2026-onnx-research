from __future__ import annotations

import argparse
import csv
import glob
import os
import pathlib
import sys

from neurogolf_local import all_examples, score_onnx


def _clean_stale_profile_traces() -> None:
    # ORT profiling writes traces with prefix configured in score_onnx as
    # `ng_task{NNN}_{timestamp}.json` into CWD. If a previous run was killed
    # or wrote a malformed trace, the same task id may pick up a stale file
    # when reading session.end_profiling(). Wipe them up-front to avoid
    # silent score errors like task356/358 going to 0 because of a bad JSON.
    for path in glob.glob("ng_task*.json"):
        try:
            os.remove(path)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--out", default="reports/task_scores.csv")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=400)
    args = parser.parse_args()
    _clean_stale_profile_traces()

    onnx_dir = pathlib.Path(args.onnx_dir)
    comp_dir = pathlib.Path(args.comp_dir)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "task_id",
        "score",
        "cost",
        "memory",
        "params",
        "local_pass",
        "local_total",
        "nodes",
        "file_size",
        "path",
        "error",
    ]
    rows = []
    total = 0.0
    for task_id in range(args.start, args.end + 1):
        examples = all_examples(task_id, comp_dir)
        result = score_onnx(onnx_dir / f"task{task_id:03d}.onnx", examples, task_id, args.n_runs)
        rows.append(result)
        total += result.score
        if task_id % 10 == 0 or result.error:
            print(f"{task_id:03d} total={total:.2f} last={result.score:.3f} cost={result.cost} err={result.error}", flush=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})

    top = sorted(rows, key=lambda row: row.cost or -1, reverse=True)[:25]
    print(f"predicted_total,{total:.4f}")
    print("top_cost_tasks")
    for row in top:
        print(
            f"{row.task_id:03d},score={row.score:.3f},cost={row.cost},mem={row.memory},"
            f"params={row.params},nodes={row.nodes},size={row.file_size},err={row.error}"
        )


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()

