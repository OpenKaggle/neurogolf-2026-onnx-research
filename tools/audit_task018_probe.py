"""Full 400-task isolated audit on task018_probe vs v4_plus4.

Baseline is the LB-verified v4_plus4 anchor (6253.25). Any task where the
probe visible pass rate is strictly lower than v4_plus4's is flagged
REGRESSION — expected only for task018 (which intentionally switches to
identity, cost=0, visible-fail=0/3).

If any OTHER task regresses, the probe is unsafe.

Output: reports/task018_probe_full_audit.csv
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import pathlib
import subprocess
import sys


PROBE_DIR = pathlib.Path("submissions/candidate_task018_probe_onnx")
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
    for p in sorted(PROBE_DIR.glob("task*.onnx")):
        task_ids.append(int(p.stem.replace("task", "")))
    print(f"auditing {len(task_ids)} probe tasks vs v4_plus4 (full isolated)", flush=True)

    args_probe = [(PROBE_DIR / f"task{t:03d}.onnx", t) for t in task_ids]
    args_v4p4 = [(V4P4_DIR / f"task{t:03d}.onnx", t) for t in task_ids]

    pool = multiprocessing.Pool(processes=4)
    res_probe = pool.map(run_isolated, args_probe)
    res_v4p4 = pool.map(run_isolated, args_v4p4)
    pool.close()
    pool.join()

    by_probe = {r["task_id"]: r for r in res_probe}
    by_v4p4 = {r["task_id"]: r for r in res_v4p4}

    rows = []
    regressions = []
    for tid in task_ids:
        rv = by_probe[tid]
        rp = by_v4p4[tid]
        probe_split = rv["splits"]
        v4p4_split = rp["splits"]
        probe_total_pass = probe_split["train"][0] + probe_split["test"][0] + probe_split["arc-gen"][0]
        probe_total = probe_split["train"][1] + probe_split["test"][1] + probe_split["arc-gen"][1]
        v4p4_total_pass = v4p4_split["train"][0] + v4p4_split["test"][0] + v4p4_split["arc-gen"][0]
        v4p4_total = v4p4_split["train"][1] + v4p4_split["test"][1] + v4p4_split["arc-gen"][1]

        if probe_total > 0 and v4p4_total > 0 and probe_total_pass < v4p4_total_pass:
            risk = "REGRESSION"
            regressions.append((tid, probe_total_pass, v4p4_total_pass, probe_total))
        elif probe_total > 0 and probe_total_pass == probe_total:
            risk = "ok"
        elif probe_total > 0 and probe_total_pass < probe_total and v4p4_total_pass == probe_total_pass:
            risk = "BOTH_VISIBLE_FAIL_TIED"
        else:
            risk = "other"

        rows.append({
            "task_id": f"task{tid:03d}",
            "probe_train_pass": f"{probe_split['train'][0]}/{probe_split['train'][1]}",
            "probe_test_pass":  f"{probe_split['test'][0]}/{probe_split['test'][1]}",
            "probe_argen_pass": f"{probe_split['arc-gen'][0]}/{probe_split['arc-gen'][1]}",
            "probe_total_pass_ratio": f"{probe_total_pass}/{probe_total}",
            "probe_status": rv.get("status", "?"),
            "probe_cost": rv.get("cost"),
            "probe_score": f"{rv.get('score',0):.4f}",
            "v4p4_total_pass_ratio": f"{v4p4_total_pass}/{v4p4_total}",
            "v4p4_status": rp.get("status", "?"),
            "v4p4_cost": rp.get("cost"),
            "v4p4_score": f"{rp.get('score',0):.4f}",
            "risk_label": risk,
            "perfect": int(probe_total > 0 and probe_total_pass == probe_total),
        })

    out = pathlib.Path("reports/task018_probe_full_audit.csv")
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
        print(f"\n!!! {len(regressions)} probe tasks REGRESSED vs v4_plus4:")
        for tid, probe_p, v4p4_p, tot in regressions:
            print(f"  task{tid:03d}: probe {probe_p}/{tot}  v4_plus4 {v4p4_p}/{tot}")
    else:
        print("\nALL probe tasks >= v4_plus4 visible pass rate. No regression.")

    # Check specifically that task018 is the only regression
    probe_t018 = by_probe.get(18, {})
    v4p4_t018 = by_v4p4.get(18, {})
    print(f"\ntask018 probe:    score={probe_t018.get('score','?'):>8}  cost={probe_t018.get('cost','?'):>8}  "
          f"pass={probe_t018.get('splits',{})}")
    print(f"task018 v4_plus4: score={v4p4_t018.get('score','?'):>8}  cost={v4p4_t018.get('cost','?'):>8}  "
          f"pass={v4p4_t018.get('splits',{})}")

    unexpected = [(tid, pp, vp) for tid, pp, vp, tot in regressions if tid != 18]
    if unexpected:
        print(f"\n!!! {len(unexpected)} UNEXPECTED regression(s) (not task018):")
        for tid, pp, vp in unexpected:
            print(f"  task{tid:03d}: probe {pp}  v4_plus4 {vp}")
    else:
        print("\nNo unexpected regressions. Only task018 changed (intentional).")


if __name__ == "__main__":
    main()
