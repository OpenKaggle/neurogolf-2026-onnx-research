"""Phase 4: build a DSL ONNX for each sampled task, isolated-eval it, and
record cost/pass against the v4_plus3 baseline.

Output:
  reports/dsl_phase4_results.csv

CSV columns:
  task_id, dsl_program, dsl_cost, dsl_train_pass, dsl_test_pass, dsl_arcgen_pass,
  baseline_cost, baseline_pass, cost_ratio (baseline/dsl, >1 means DSL is cheaper),
  status
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile

THIS = pathlib.Path(__file__).resolve()
ROOT = THIS.parent.parent.parent  # workspace root
sys.path.insert(0, str(ROOT / "tools"))

from tools.dsl.compiler import DslOp, DslProgram, compile_to_onnx
import onnx

# Sampled tasks with their DSL programs.
TASKS = [
    (87, [DslOp("rotate180", {})]),
    (140, [DslOp("rotate180", {})]),
    (150, [DslOp("flip_h", {})]),
    (155, [DslOp("flip_v", {})]),
    (179, [DslOp("transpose", {})]),
    (241, [DslOp("transpose", {})]),
    (380, [DslOp("rotate90", {})]),
    # task276 dataset never has color 2 in input, so swap and replace
    # collapse to the same semantics; we use the cheaper swap_colors.
    (276, [DslOp("swap_colors", {"c1": 6, "c2": 2})]),
    # task309 dataset never has color 5 in input.
    (309, [DslOp("swap_colors", {"c1": 7, "c2": 5})]),
    (337, [DslOp("swap_colors", {"c1": 5, "c2": 8})]),
]


def baseline_lookup() -> dict[int, dict]:
    path = ROOT / "reports" / "candidate_v4_plus3_score_n3.csv"
    out: dict[int, dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["task_id"])] = row
    return out


def isolated_eval(onnx_path: pathlib.Path, task_id: int) -> dict:
    # Use the existing helper so the score is comparable to v4_plus3 numbers.
    cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "tools/isolated_task_eval.py"),
        "--onnx", str(onnx_path),
        "--task-id", str(task_id),
        "--comp-dir", str(ROOT / "data/neurogolf-2026/raw"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip())


def main() -> None:
    baseline = baseline_lookup()
    out_rows: list[dict] = []
    out_dir = ROOT / "submissions/handbuilds/dsl_phase4"
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_id, ops in TASKS:
        program = DslProgram.of(ops)
        try:
            model = compile_to_onnx(program)
        except Exception as e:
            out_rows.append({
                "task_id": task_id,
                "dsl_program": repr(program),
                "dsl_cost": "",
                "dsl_train_pass": "",
                "dsl_test_pass": "",
                "dsl_arcgen_pass": "",
                "baseline_cost": baseline.get(task_id, {}).get("cost", ""),
                "baseline_pass": baseline.get(task_id, {}).get("local_pass", ""),
                "cost_ratio": "",
                "status": f"compile-fail: {e}",
            })
            continue
        onnx_path = out_dir / f"dsl_task{task_id:03d}.onnx"
        onnx.save(model, onnx_path)
        res = isolated_eval(onnx_path, task_id)
        b = baseline.get(task_id, {})
        b_cost = b.get("cost", "")
        dsl_cost = res.get("cost")
        cost_ratio = ""
        if dsl_cost is not None and b_cost not in ("", None):
            try:
                bc = int(b_cost)
                if dsl_cost > 0:
                    cost_ratio = round(bc / dsl_cost, 3)
                elif dsl_cost == 0:
                    cost_ratio = "∞" if bc > 0 else 1.0
            except Exception:
                pass
        status = res.get("status", "ok")
        sp = res.get("splits", {})
        out_rows.append({
            "task_id": task_id,
            "dsl_program": repr(program),
            "dsl_cost": dsl_cost,
            "dsl_train_pass": f'{sp.get("train",[0,0])[0]}/{sp.get("train",[0,0])[1]}',
            "dsl_test_pass": f'{sp.get("test",[0,0])[0]}/{sp.get("test",[0,0])[1]}',
            "dsl_arcgen_pass": f'{sp.get("arc-gen",[0,0])[0]}/{sp.get("arc-gen",[0,0])[1]}',
            "baseline_cost": b_cost,
            "baseline_pass": b.get("local_pass", ""),
            "cost_ratio": cost_ratio,
            "status": status,
        })

    out_csv = ROOT / "reports" / "dsl_phase4_results.csv"
    fieldnames = list(out_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    # Print summary
    print(f"wrote {out_csv}")
    print()
    print(f'{"task":>5} {"baseline":>10} {"dsl":>10} {"ratio":>8} {"train":>6} {"test":>5} {"arc-gen":>10} {"program"}')
    for r in out_rows:
        print(f'{r["task_id"]:>5} {str(r["baseline_cost"]):>10} {str(r["dsl_cost"]):>10} {str(r["cost_ratio"]):>8} '
              f'{r["dsl_train_pass"]:>6} {r["dsl_test_pass"]:>5} {r["dsl_arcgen_pass"]:>10} {r["dsl_program"]}')


if __name__ == "__main__":
    main()
