"""Full 400-task isolated audit on v4.

Runs `tools/isolated_task_eval.py` against every task in v4 (and the same
task in current_best_6122 for comparison). Goal: find ANY visible failure
in v4 (not just kept-Nadeem); flag any silent regression class similar to
task191.

Output: reports/v4_full_audit.csv
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import pathlib
import subprocess
import sys


V4_DIR = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v4_onnx")
OURS_DIR = pathlib.Path("submissions/current_best_6122_onnx")


def run_isolated(args):
    onnx_path, task_id = args
    cmd = [
        sys.executable,
        "tools/isolated_task_eval.py",
        "--onnx", str(onnx_path),
        "--task-id", str(task_id),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "task_id": task_id,
            "path": str(onnx_path),
            "splits": {"train": [0, 0], "test": [0, 0], "arc-gen": [0, 0]},
            "status": "subprocess_error",
            "stderr": proc.stderr[:200],
        }


def main() -> None:
    task_ids = []
    for p in sorted(V4_DIR.glob("task*.onnx")):
        task_ids.append(int(p.stem.replace("task", "")))
    print(f"auditing {len(task_ids)} v4 tasks (full isolated)", flush=True)

    args_v4 = [(V4_DIR / f"task{t:03d}.onnx", t) for t in task_ids]
    args_ours = [(OURS_DIR / f"task{t:03d}.onnx", t) for t in task_ids]

    pool = multiprocessing.Pool(processes=4)
    res_v4 = pool.map(run_isolated, args_v4)
    res_ours = pool.map(run_isolated, args_ours)
    pool.close()
    pool.join()

    by_v4 = {r["task_id"]: r for r in res_v4}
    by_ours = {r["task_id"]: r for r in res_ours}

    rows = []
    for tid in task_ids:
        rv = by_v4[tid]
        ro = by_ours[tid]
        v4_split = rv["splits"]
        ours_split = ro["splits"]
        v4_total_pass = v4_split["train"][0] + v4_split["test"][0] + v4_split["arc-gen"][0]
        v4_total = v4_split["train"][1] + v4_split["test"][1] + v4_split["arc-gen"][1]
        ours_total_pass = ours_split["train"][0] + ours_split["test"][0] + ours_split["arc-gen"][0]
        ours_total = ours_split["train"][1] + ours_split["test"][1] + ours_split["arc-gen"][1]

        if v4_total > 0 and v4_total_pass < v4_total and ours_total_pass == ours_total:
            risk = "REGRESSION_VS_OURS"
        elif v4_total > 0 and v4_total_pass < v4_total and ours_total_pass < ours_total:
            risk = "BOTH_LOCAL_FAIL"
        elif v4_total > 0 and v4_total_pass == v4_total:
            risk = "ok"
        else:
            risk = "unknown"

        rows.append({
            "task_id": f"task{tid:03d}",
            "v4_train_pass": f"{v4_split['train'][0]}/{v4_split['train'][1]}",
            "v4_test_pass":  f"{v4_split['test'][0]}/{v4_split['test'][1]}",
            "v4_argen_pass": f"{v4_split['arc-gen'][0]}/{v4_split['arc-gen'][1]}",
            "v4_total_pass_ratio": f"{v4_total_pass}/{v4_total}",
            "v4_status": rv.get("status", "?"),
            "v4_cost": rv.get("cost"),
            "v4_score": f"{rv.get('score',0):.4f}",
            "ours_total_pass_ratio": f"{ours_total_pass}/{ours_total}",
            "ours_status": ro.get("status", "?"),
            "ours_cost": ro.get("cost"),
            "ours_score": f"{ro.get('score',0):.4f}",
            "risk_label": risk,
            "perfect": int(v4_total > 0 and v4_total_pass == v4_total),
        })

    out = pathlib.Path("reports/v4_full_audit.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_risk = {}
    for r in rows:
        by_risk.setdefault(r["risk_label"], []).append(r["task_id"])
    print(f"\nwrote {out}\n")
    print(f"perfect: {sum(r['perfect'] for r in rows)}/{len(rows)}")
    print("risk summary:")
    for k, v in sorted(by_risk.items()):
        print(f"  {k:<25s} {len(v):3d}")
        if k == "REGRESSION_VS_OURS":
            for t in v:
                print(f"    {t}")
        elif k == "BOTH_LOCAL_FAIL":
            for t in v[:20]:
                print(f"    {t}")


if __name__ == "__main__":
    main()
