"""Full-example isolated audit for every Nadeem-source task in v3.

For each task whose v3 file came from Nadeem (per task_provenance.csv), run
isolated_task_eval against:
  - the v3 staged file (= the Nadeem version unless lossless rewrote it)
  - the corresponding current_best_6122 version (= the LB-verified fallback)

We're hunting for "task264 lookalikes": tasks where Nadeem passes our quick
n=3 scorer but actually fails when scored over all visible examples in an
isolated process.

Output: reports/v3_nadeem_audit.csv with per-task visible pass counts for
both candidates and a risk_label.
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import pathlib
import subprocess
import sys


def run_isolated(args):
    onnx_path, task_id = args
    cmd = [
        sys.executable,
        "tools/isolated_task_eval.py",
        "--onnx", str(onnx_path),
        "--task-id", str(task_id),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"task_id": task_id, "path": str(onnx_path), "splits": {"train": [0,0], "test": [0,0], "arc-gen": [0,0]}, "status": "subprocess_error", "stderr": proc.stderr[:200]}


def main() -> None:
    nadeem_tasks = []
    with open("reports/task_provenance.csv") as f:
        for r in csv.DictReader(f):
            if r["v2_source_label"] == "nadeem_6252":
                nadeem_tasks.append(int(r["task_id"].replace("task", "")))

    print(f"auditing {len(nadeem_tasks)} Nadeem-source tasks", flush=True)

    v3_dir = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v3_onnx")
    ours_dir = pathlib.Path("submissions/current_best_6122_onnx")

    args_v3 = [(v3_dir / f"task{t:03d}.onnx", t) for t in nadeem_tasks]
    args_ours = [(ours_dir / f"task{t:03d}.onnx", t) for t in nadeem_tasks]

    pool = multiprocessing.Pool(processes=4)
    res_v3 = pool.map(run_isolated, args_v3)
    res_ours = pool.map(run_isolated, args_ours)
    pool.close()
    pool.join()

    by_v3 = {r["task_id"]: r for r in res_v3}
    by_ours = {r["task_id"]: r for r in res_ours}

    rows = []
    for tid in nadeem_tasks:
        rv = by_v3[tid]
        ro = by_ours[tid]
        v3_split = rv["splits"]
        ours_split = ro["splits"]
        v3_total_pass = v3_split["train"][0] + v3_split["test"][0] + v3_split["arc-gen"][0]
        v3_total = v3_split["train"][1] + v3_split["test"][1] + v3_split["arc-gen"][1]
        ours_total_pass = ours_split["train"][0] + ours_split["test"][0] + ours_split["arc-gen"][0]
        ours_total = ours_split["train"][1] + ours_split["test"][1] + ours_split["arc-gen"][1]

        if v3_total > 0 and v3_total_pass < v3_total and ours_total_pass == ours_total:
            risk = "REGRESSION_VS_OURS"
        elif v3_total > 0 and v3_total_pass < v3_total and ours_total_pass < ours_total:
            risk = "BOTH_LOCAL_FAIL"
        elif v3_total_pass == v3_total:
            risk = "ok"
        else:
            risk = "unknown"

        rows.append({
            "task_id": f"task{tid:03d}",
            "v3_train_pass": f"{v3_split['train'][0]}/{v3_split['train'][1]}",
            "v3_test_pass":  f"{v3_split['test'][0]}/{v3_split['test'][1]}",
            "v3_argen_pass": f"{v3_split['arc-gen'][0]}/{v3_split['arc-gen'][1]}",
            "v3_total_pass_ratio": f"{v3_total_pass}/{v3_total}",
            "v3_status": rv.get("status", "?"),
            "v3_cost": rv.get("cost"),
            "v3_score": f"{rv.get('score',0):.4f}",
            "ours_total_pass_ratio": f"{ours_total_pass}/{ours_total}",
            "ours_status": ro.get("status", "?"),
            "ours_cost": ro.get("cost"),
            "ours_score": f"{ro.get('score',0):.4f}",
            "risk_label": risk,
        })

    out = pathlib.Path("reports/v3_nadeem_audit.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_risk = {}
    for r in rows:
        by_risk.setdefault(r["risk_label"], []).append(r["task_id"])
    print(f"\nwrote {out}\n")
    print("risk summary:")
    for k, v in sorted(by_risk.items()):
        print(f"  {k:<25s} {len(v):3d}")
        if k == "REGRESSION_VS_OURS":
            for t in v:
                print(f"    {t}")


if __name__ == "__main__":
    main()
