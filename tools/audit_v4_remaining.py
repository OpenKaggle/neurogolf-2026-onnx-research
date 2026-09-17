"""Final gatekeeper audit for v4: full-example isolated check on every
Nadeem-source task that remained in v4 after revert.
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
    cmd = [sys.executable, "tools/isolated_task_eval.py", "--onnx", str(onnx_path), "--task-id", str(task_id)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"task_id": task_id, "splits": {"train": [0,0], "test": [0,0], "arc-gen": [0,0]}, "status": "subprocess_error"}


def main() -> None:
    kept_nadeem = []
    with open("reports/v4_revert_decisions.csv") as f:
        for r in csv.DictReader(f):
            if r["decision"] == "keep_nadeem":
                kept_nadeem.append(int(r["task_id"].replace("task", "")))

    print(f"auditing {len(kept_nadeem)} kept-Nadeem tasks in v4", flush=True)

    v4_dir = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v4_onnx")
    args_v4 = [(v4_dir / f"task{t:03d}.onnx", t) for t in kept_nadeem]

    pool = multiprocessing.Pool(processes=4)
    res = pool.map(run_isolated, args_v4)
    pool.close()
    pool.join()

    rows = []
    fails = []
    for r in res:
        sp = r["splits"]
        total_pass = sp["train"][0] + sp["test"][0] + sp["arc-gen"][0]
        total = sp["train"][1] + sp["test"][1] + sp["arc-gen"][1]
        rows.append({
            "task_id": f"task{r['task_id']:03d}",
            "train": f"{sp['train'][0]}/{sp['train'][1]}",
            "test": f"{sp['test'][0]}/{sp['test'][1]}",
            "argen": f"{sp['arc-gen'][0]}/{sp['arc-gen'][1]}",
            "total_pass": f"{total_pass}/{total}",
            "perfect": int(total_pass == total),
        })
        if total_pass < total:
            fails.append((r["task_id"], total_pass, total, sp))

    out = pathlib.Path("reports/v4_kept_nadeem_audit.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {out}")
    print(f"kept Nadeem perfect: {sum(r['perfect'] for r in rows)}/{len(rows)}")
    if fails:
        print(f"\n!!! {len(fails)} kept-Nadeem tasks NOT perfect:")
        for t, p, n, sp in fails:
            print(f"  task{t:03d}: {p}/{n} (train {sp['train']}, test {sp['test']}, argen {sp['arc-gen']})")
    else:
        print("\nALL kept Nadeem tasks pass full-example audit. v4 is clean.")


if __name__ == "__main__":
    main()
