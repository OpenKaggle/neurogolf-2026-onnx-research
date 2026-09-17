"""Full 400-task isolated audit on v4_plus5 vs v4_plus4.

Baseline is the LB-verified v4_plus4 anchor (6253.25).
Any task where v4_plus5 visible pass rate is strictly lower than v4_plus4's
is flagged REGRESSION_VS_V4PLUS4 — those swaps must be reverted.

Output: reports/v4_plus5_full_audit.csv
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import pathlib
import subprocess
import sys


V4P5_DIR = pathlib.Path("submissions/candidate_v4_plus5_onnx")
V4P4_DIR = pathlib.Path("submissions/candidate_v4_plus4_onnx")


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
    for p in sorted(V4P5_DIR.glob("task*.onnx")):
        task_ids.append(int(p.stem.replace("task", "")))
    print(f"auditing {len(task_ids)} v4_plus5 tasks (full isolated)", flush=True)

    args_v4p5 = [(V4P5_DIR / f"task{t:03d}.onnx", t) for t in task_ids]
    args_v4p4 = [(V4P4_DIR / f"task{t:03d}.onnx", t) for t in task_ids]

    pool = multiprocessing.Pool(processes=4)
    res_v4p5 = pool.map(run_isolated, args_v4p5)
    res_v4p4 = pool.map(run_isolated, args_v4p4)
    pool.close()
    pool.join()

    by_v4p5 = {r["task_id"]: r for r in res_v4p5}
    by_v4p4 = {r["task_id"]: r for r in res_v4p4}

    rows = []
    regressions = []
    for tid in task_ids:
        rv = by_v4p5[tid]
        rp = by_v4p4[tid]
        v4p5_split = rv["splits"]
        v4p4_split = rp["splits"]
        v4p5_total_pass = v4p5_split["train"][0] + v4p5_split["test"][0] + v4p5_split["arc-gen"][0]
        v4p5_total = v4p5_split["train"][1] + v4p5_split["test"][1] + v4p5_split["arc-gen"][1]
        v4p4_total_pass = v4p4_split["train"][0] + v4p4_split["test"][0] + v4p4_split["arc-gen"][0]
        v4p4_total = v4p4_split["train"][1] + v4p4_split["test"][1] + v4p4_split["arc-gen"][1]

        if v4p5_total > 0 and v4p4_total > 0 and v4p5_total_pass < v4p4_total_pass:
            risk = "REGRESSION_VS_V4PLUS4"
            regressions.append((tid, v4p5_total_pass, v4p4_total_pass, v4p5_total))
        elif v4p5_total > 0 and v4p5_total_pass == v4p5_total:
            risk = "ok"
        elif v4p5_total > 0 and v4p5_total_pass < v4p5_total and v4p4_total_pass == v4p5_total_pass:
            risk = "BOTH_VISIBLE_FAIL_TIED"
        else:
            risk = "other"

        rows.append({
            "task_id": f"task{tid:03d}",
            "v4p5_train_pass": f"{v4p5_split['train'][0]}/{v4p5_split['train'][1]}",
            "v4p5_test_pass":  f"{v4p5_split['test'][0]}/{v4p5_split['test'][1]}",
            "v4p5_argen_pass": f"{v4p5_split['arc-gen'][0]}/{v4p5_split['arc-gen'][1]}",
            "v4p5_total_pass_ratio": f"{v4p5_total_pass}/{v4p5_total}",
            "v4p5_status": rv.get("status", "?"),
            "v4p5_cost": rv.get("cost"),
            "v4p5_score": f"{rv.get('score',0):.4f}",
            "v4p4_total_pass_ratio": f"{v4p4_total_pass}/{v4p4_total}",
            "v4p4_status": rp.get("status", "?"),
            "v4p4_cost": rp.get("cost"),
            "v4p4_score": f"{rp.get('score',0):.4f}",
            "risk_label": risk,
            "perfect": int(v4p5_total > 0 and v4p5_total_pass == v4p5_total),
        })

    out = pathlib.Path("reports/v4_plus5_full_audit.csv")
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
    if regressions:
        print(f"\n!!! {len(regressions)} v4_plus5 tasks REGRESSED vs v4_plus4:")
        for tid, p5, p4, tot in regressions:
            print(f"  task{tid:03d}: v4_plus5 {p5}/{tot}  v4_plus4 {p4}/{tot}")
    else:
        print("\nALL v4_plus5 tasks >= v4_plus4 visible pass rate. No revert required.")


if __name__ == "__main__":
    main()
