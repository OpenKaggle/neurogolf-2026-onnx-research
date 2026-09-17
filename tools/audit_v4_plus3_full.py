"""Full 400-task isolated audit on v4_plus3 vs v4_plus.

Mirrors `tools/audit_v4_full.py` but the baseline is the LB-verified v4_plus
anchor (not ours/6122). Any task where v4_plus3 visible pass rate is strictly
lower than v4_plus's pass rate is flagged REGRESSION_VS_V4PLUS — those swaps
must be reverted (copy v4_plus task ONNX back over v4_plus3) before submission.

Output: reports/v4_plus3_full_audit.csv
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import pathlib
import subprocess
import sys


V4P3_DIR = pathlib.Path("submissions/candidate_v4_plus3_onnx")
V4PLUS_DIR = pathlib.Path("submissions/candidate_v4_plus_onnx")


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
    for p in sorted(V4P3_DIR.glob("task*.onnx")):
        task_ids.append(int(p.stem.replace("task", "")))
    print(f"auditing {len(task_ids)} v4_plus3 tasks (full isolated)", flush=True)

    args_v4p3 = [(V4P3_DIR / f"task{t:03d}.onnx", t) for t in task_ids]
    args_v4plus = [(V4PLUS_DIR / f"task{t:03d}.onnx", t) for t in task_ids]

    pool = multiprocessing.Pool(processes=4)
    res_v4p3 = pool.map(run_isolated, args_v4p3)
    res_v4plus = pool.map(run_isolated, args_v4plus)
    pool.close()
    pool.join()

    by_v4p3 = {r["task_id"]: r for r in res_v4p3}
    by_v4plus = {r["task_id"]: r for r in res_v4plus}

    rows = []
    regressions = []
    for tid in task_ids:
        rv = by_v4p3[tid]
        rp = by_v4plus[tid]
        v4p3_split = rv["splits"]
        v4plus_split = rp["splits"]
        v4p3_total_pass = v4p3_split["train"][0] + v4p3_split["test"][0] + v4p3_split["arc-gen"][0]
        v4p3_total = v4p3_split["train"][1] + v4p3_split["test"][1] + v4p3_split["arc-gen"][1]
        v4plus_total_pass = v4plus_split["train"][0] + v4plus_split["test"][0] + v4plus_split["arc-gen"][0]
        v4plus_total = v4plus_split["train"][1] + v4plus_split["test"][1] + v4plus_split["arc-gen"][1]

        if v4p3_total > 0 and v4plus_total > 0 and v4p3_total_pass < v4plus_total_pass:
            risk = "REGRESSION_VS_V4PLUS"
            regressions.append((tid, v4p3_total_pass, v4plus_total_pass, v4p3_total))
        elif v4p3_total > 0 and v4p3_total_pass == v4p3_total:
            risk = "ok"
        elif v4p3_total > 0 and v4p3_total_pass < v4p3_total and v4plus_total_pass == v4p3_total_pass:
            risk = "BOTH_VISIBLE_FAIL_TIED"
        else:
            risk = "other"

        rows.append({
            "task_id": f"task{tid:03d}",
            "v4p3_train_pass": f"{v4p3_split['train'][0]}/{v4p3_split['train'][1]}",
            "v4p3_test_pass":  f"{v4p3_split['test'][0]}/{v4p3_split['test'][1]}",
            "v4p3_argen_pass": f"{v4p3_split['arc-gen'][0]}/{v4p3_split['arc-gen'][1]}",
            "v4p3_total_pass_ratio": f"{v4p3_total_pass}/{v4p3_total}",
            "v4p3_status": rv.get("status", "?"),
            "v4p3_cost": rv.get("cost"),
            "v4p3_score": f"{rv.get('score',0):.4f}",
            "v4plus_total_pass_ratio": f"{v4plus_total_pass}/{v4plus_total}",
            "v4plus_status": rp.get("status", "?"),
            "v4plus_cost": rp.get("cost"),
            "v4plus_score": f"{rp.get('score',0):.4f}",
            "risk_label": risk,
            "perfect": int(v4p3_total > 0 and v4p3_total_pass == v4p3_total),
        })

    out = pathlib.Path("reports/v4_plus3_full_audit.csv")
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
        print(f"\n!!! {len(regressions)} v4_plus3 tasks REGRESSED vs v4_plus:")
        for tid, p3, pp, tot in regressions:
            print(f"  task{tid:03d}: v4_plus3 {p3}/{tot}  v4_plus {pp}/{tot}")
    else:
        print("\nALL v4_plus3 tasks >= v4_plus visible pass rate. No revert required.")


if __name__ == "__main__":
    main()
