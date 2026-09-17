"""Full 400-task isolated audit on v4_plus, with regression detection
against v4 and the LB-verified 6122 baseline.

If ANY task in v4_plus drops below v4's isolated pass count, the
caller is expected to revert that task and re-run the audit. This
script only reports — it does NOT modify the staged bundle.

Output: reports/v4_plus_full_audit.csv with columns mirroring
reports/v4_full_audit.csv plus a `regression_from_v4` flag.
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import pathlib
import subprocess
import sys


V4_PLUS_DIR = pathlib.Path("submissions/candidate_v4_plus_onnx")
V4_DIR = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v4_onnx")
OURS_DIR = pathlib.Path("submissions/current_best_6122_onnx")


def run_isolated(args):
    onnx_path, task_id = args
    cmd = [sys.executable, "tools/isolated_task_eval.py", "--onnx", str(onnx_path), "--task-id", str(task_id)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "task_id": task_id,
            "path": str(onnx_path),
            "splits": {"train": [0, 0], "test": [0, 0], "arc-gen": [0, 0]},
            "status": "subprocess_error",
            "stderr": proc.stderr[:200] if hasattr(proc, "stderr") else "",
        }


def main() -> None:
    task_ids = []
    for p in sorted(V4_PLUS_DIR.glob("task*.onnx")):
        task_ids.append(int(p.stem.replace("task", "")))
    print(f"auditing {len(task_ids)} v4_plus tasks (full isolated, 3-way)", flush=True)

    args_p = [(V4_PLUS_DIR / f"task{t:03d}.onnx", t) for t in task_ids]
    args_v = [(V4_DIR / f"task{t:03d}.onnx", t) for t in task_ids]
    args_o = [(OURS_DIR / f"task{t:03d}.onnx", t) for t in task_ids]

    pool = multiprocessing.Pool(processes=4)
    res_p = pool.map(run_isolated, args_p)
    res_v = pool.map(run_isolated, args_v)
    res_o = pool.map(run_isolated, args_o)
    pool.close()
    pool.join()

    by_p = {r["task_id"]: r for r in res_p}
    by_v = {r["task_id"]: r for r in res_v}
    by_o = {r["task_id"]: r for r in res_o}

    rows = []
    regressions = []
    perfect = 0
    for tid in task_ids:
        rp = by_p[tid]
        rv = by_v[tid]
        ro = by_o[tid]
        sp_p = sum(rp["splits"].get(k, [0, 0])[0] for k in ("train", "test", "arc-gen"))
        st_p = sum(rp["splits"].get(k, [0, 0])[1] for k in ("train", "test", "arc-gen"))
        sp_v = sum(rv["splits"].get(k, [0, 0])[0] for k in ("train", "test", "arc-gen"))
        st_v = sum(rv["splits"].get(k, [0, 0])[1] for k in ("train", "test", "arc-gen"))
        sp_o = sum(ro["splits"].get(k, [0, 0])[0] for k in ("train", "test", "arc-gen"))
        st_o = sum(ro["splits"].get(k, [0, 0])[1] for k in ("train", "test", "arc-gen"))

        regression = sp_p < sp_v
        if st_p > 0 and sp_p == st_p:
            perfect += 1
        if regression:
            regressions.append((f"task{tid:03d}", sp_p, sp_v, st_p))

        rows.append({
            "task_id": f"task{tid:03d}",
            "v4plus_pass": f"{sp_p}/{st_p}",
            "v4_pass": f"{sp_v}/{st_v}",
            "ours_pass": f"{sp_o}/{st_o}",
            "v4plus_status": rp.get("status", "?"),
            "v4plus_cost": rp.get("cost"),
            "v4plus_score": f"{rp.get('score',0):.4f}",
            "v4_cost": rv.get("cost"),
            "ours_cost": ro.get("cost"),
            "regression_from_v4": int(regression),
            "perfect": int(st_p > 0 and sp_p == st_p),
        })

    out = pathlib.Path("reports/v4_plus_full_audit.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {out}")
    print(f"perfect (v4_plus): {perfect}/{len(rows)}")
    if regressions:
        print(f"\n!!! {len(regressions)} task(s) regressed from v4:")
        for t, sp_p, sp_v, st_p in regressions:
            print(f"  {t}: v4_plus {sp_p}/{st_p}  v4 {sp_v}/{st_p}")
        sys.exit(2)
    else:
        print("\nALL v4_plus tasks match-or-beat v4 isolated pass count. Audit clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
