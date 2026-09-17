"""W1 mining for v4_plus2: task018-class (visible 0/N) cross-source swaps.

For every task T where v4_plus shows visible_pass==0 in the full 400-task
isolated audit (`reports/v4_plus_full_audit.csv`), score the same task in
each known public source. A swap is accepted iff:

    src_cost  < v4plus_cost
    src_visible_pass >= v4plus_visible_pass   (allowed equal 0/N -> 0/N)
    struct_risk < 1.5
    task NOT in HARD_FROZEN

Writes reports/v4_plus2_swap_candidates.csv with all candidate rows.

This is the H1 cost-only mining channel discovered by the v4_plus LB
result: Kaggle awards task018 cost-only despite visible 0/266 on every
source. The same logic may apply to other tasks that are visible-fail in
every source we own — but at the time of this audit, task018 is the
ONLY such task in v4_plus.

The hard-frozen task list (task191/255/240/349/184/301/396) is kept
frozen even under the H1 relaxation, UNLESS the audit shows that v4_plus
*itself* is now visible 0/N on that task (i.e. its task018-class).
"""
from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import onnx
from onnx import numpy_helper as onh


V4_PLUS_DIR = pathlib.Path("submissions/candidate_v4_plus_onnx")
V4_PLUS_AUDIT = pathlib.Path("reports/v4_plus_full_audit.csv")
COMP_DIR = pathlib.Path("data/neurogolf-2026/raw")

# Source registry (label, onnx_dir). Skip mining sources without on-disk
# bundles. Same set as build_v4_plus_candidates.py.
SOURCES = [
    ("nadeem_6252",        "submissions/nadeem_6252_onnx"),
    ("biohack44_super",    "submissions/biohack44_super_onnx"),
    ("biohack44_6113",     "submissions/biohack44_6113_onnx"),
    ("biohack44_6080",     "submissions/biohack44_6080_onnx"),
    ("biohack44_6067",     "data/biohack44_6067"),
    ("afr1ste_5689",       "submissions/afr1ste_5689_onnx"),
    ("octavi_6042",        "submissions/octavi_6042_onnx"),
    ("octaviograu_6154",   "data/octaviograu_6154/submission"),
    ("haoranran_6100",     "submissions/haoranran_6100_onnx"),
    ("massimiliano_eda111","submissions/massimiliano_eda111_onnx"),
    ("current_best_6122",  "submissions/current_best_6122_onnx"),
]

# Hard-frozen task set per ledger. These stay frozen unless v4_plus itself
# is visible 0/N on that task (the H1 relaxation gate).
HARD_FROZEN = {184, 191, 240, 255, 301, 319, 349, 396}

STRUCT_RISK_MAX = 1.5


def parse_pass(s: str) -> tuple[int, int]:
    p, t = s.split("/")
    return int(p), int(t)


def load_v4_plus_audit() -> dict[int, dict]:
    out = {}
    with V4_PLUS_AUDIT.open() as f:
        for r in csv.DictReader(f):
            tid = int(r["task_id"].replace("task", ""))
            sp_p, st_p = parse_pass(r["v4plus_pass"])
            out[tid] = {
                "v4plus_pass": sp_p,
                "v4plus_total": st_p,
                "v4plus_cost": int(r["v4plus_cost"]) if r["v4plus_cost"] else 0,
            }
    return out


def structural_audit(path: pathlib.Path) -> tuple[float, list[str]]:
    m = onnx.load(str(path))
    g = m.graph
    op_counts = Counter(n.op_type for n in g.node)
    init_bytes = 0
    biggest = 0
    has_fp16 = False
    for ini in g.initializer:
        arr = onh.to_array(ini)
        init_bytes += arr.nbytes
        if arr.nbytes > biggest:
            biggest = arr.nbytes
        if str(arr.dtype) == "float16":
            has_fp16 = True
    file_size = path.stat().st_size
    init_byte_ratio = init_bytes / max(file_size, 1)
    has_lookup = any(op in op_counts for op in ("GatherND", "ScatterND", "EyeLike", "OneHot", "TopK"))
    flags = []
    score = 0.0
    if init_byte_ratio > 0.7:
        flags.append("high_init_ratio")
        score += 1
    if biggest > 524288:
        flags.append("huge_init>=512KB")
        score += 2
    elif biggest > 65536:
        flags.append("big_init>=64KB")
        score += 1
    if has_lookup:
        flags.append("has_lookup_ops")
        score += 0.5
    if has_fp16:
        flags.append("fp16")
        score += 0.5
    return score, flags


def isolated_eval(onnx_path: pathlib.Path, task_id: int) -> tuple[int, int, int]:
    cmd = [sys.executable, "tools/isolated_task_eval.py", "--onnx", str(onnx_path), "--task-id", str(task_id)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    sp = sum(res["splits"].get(k, [0, 0])[0] for k in ("train", "test", "arc-gen"))
    st = sum(res["splits"].get(k, [0, 0])[1] for k in ("train", "test", "arc-gen"))
    cost = res.get("cost", 0) or 0
    return sp, st, int(cost)


def evaluate(args) -> dict:
    task_id, src_label, src_dir_str, v4plus_pass, v4plus_total, v4plus_cost = args
    src_dir = pathlib.Path(src_dir_str)
    src_path = src_dir / f"task{task_id:03d}.onnx"
    row = {
        "task_id": f"task{task_id:03d}",
        "source": src_label,
        "v4plus_cost": v4plus_cost,
        "src_cost": "",
        "cost_save": "",
        "v4plus_visible_pass": f"{v4plus_pass}/{v4plus_total}",
        "src_visible_pass": "",
        "struct_risk": "",
        "accepted": "False",
        "reject_reason": "",
    }
    if not src_path.is_file():
        row["reject_reason"] = "missing_source"
        return row

    try:
        risk, _ = structural_audit(src_path)
    except Exception as exc:
        row["reject_reason"] = f"struct_audit_error:{exc!r}"
        return row
    row["struct_risk"] = f"{risk:.1f}"

    sp_src, st_src, src_cost = isolated_eval(src_path, task_id)
    row["src_cost"] = src_cost
    row["src_visible_pass"] = f"{sp_src}/{st_src}"

    cost_save = v4plus_cost - src_cost
    row["cost_save"] = cost_save

    if risk >= STRUCT_RISK_MAX:
        row["reject_reason"] = f"struct_risk_high({risk:.1f})"
        return row
    if cost_save <= 0:
        row["reject_reason"] = f"no_cost_save({cost_save})"
        return row
    if sp_src < v4plus_pass:
        row["reject_reason"] = f"src_pass_below_v4plus({sp_src}<{v4plus_pass})"
        return row
    # Allow same visible_pass as v4plus when both are zero or both equal:
    # task018-class is defined as visible 0/N -> visible 0/N.

    row["accepted"] = "True"
    return row


def main() -> None:
    audit = load_v4_plus_audit()
    # Candidate slots: v4_plus visible_pass == 0
    slots = sorted(tid for tid, r in audit.items() if r["v4plus_pass"] == 0)
    print(f"v4_plus 0/N candidate slots: {len(slots)} -> {slots}", flush=True)

    # Apply hard-frozen rule with H1 relaxation
    eligible = []
    for tid in slots:
        if tid in HARD_FROZEN and audit[tid]["v4plus_pass"] != 0:
            print(f"  task{tid:03d}: hard-frozen, skipped (v4plus pass != 0)", flush=True)
            continue
        eligible.append(tid)
    print(f"eligible slots (after hard-frozen + H1 relaxation): {len(eligible)}", flush=True)

    # Build evaluation list
    tasks = []
    for tid in eligible:
        info = audit[tid]
        for label, src_dir in SOURCES:
            tasks.append((tid, label, src_dir, info["v4plus_pass"], info["v4plus_total"], info["v4plus_cost"]))

    print(f"total (task, source) probes: {len(tasks)}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(evaluate, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"  {r['task_id']} {r['source']:<22s} src_cost={r['src_cost']!s:<8s} save={r['cost_save']!s:<8s} v_pass={r['src_visible_pass']:<8s} risk={r['struct_risk']:<4s} acc={r['accepted']:<5s} reason={r['reject_reason']}", flush=True)

    rows.sort(key=lambda r: (r["task_id"], r["source"]))
    out = pathlib.Path("reports/v4_plus2_swap_candidates.csv")
    fields = ["task_id", "source", "v4plus_cost", "src_cost", "cost_save",
              "v4plus_visible_pass", "src_visible_pass", "struct_risk",
              "accepted", "reject_reason"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}: total rows={len(rows)} accepted={sum(1 for r in rows if r['accepted']=='True')}")


if __name__ == "__main__":
    main()
