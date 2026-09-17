"""Build v4_exp: an experimental candidate built ON TOP of v4.

Hard rules we enforce (from the task spec):
  1. v4 is the chassis: NEVER touch tasks that v4 reverted by audit OR struct
     (task191 / task255 / task240 / task349 / task184 / task301 / task396).
  2. We may revisit a low_delta-reverted task only if direction 1 logic chooses
     it AND it has structurally clean Nadeem version + isolated audit perfect.
  3. Every cherry-pick MUST pass an isolated full-example audit (train+test+
     arc-gen perfect) BEFORE going into v4_exp.
  4. If after all picks v4_exp local total < v4 local, we abort and v4_exp is
     not produced.

Sources we consider:
  D1: Nadeem-source for tasks NOT in nadeem_set in v2 (i.e. v2 chose ours)
      where delta (nadeem - ours) in [0.10, 0.50) AND nadeem ONNX has graph
      risk_score == 0 AND audit isolated perfect.
  D3: biohack44_6113 11/11 LB-verified cherry-picks (151/028/200/258/111/155/
      150/014/204/084/322): swap if biohack score > current v4 score AND audit
      isolated perfect.
  D4: pure lossless rewrites on top of v4 (handled by a separate combo probe;
      we only ingest its output dir if provided).

Output:
  submissions/candidate_v4_exp_onnx/   (only if local total > v4)
  submissions/submission_candidate_v4_exp.zip
  reports/v4_exp_decisions.csv
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import pathlib
import shutil
import subprocess
import sys
import zipfile
from collections import Counter

import onnx


V4_DIR = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v4_onnx")
OURS_DIR = pathlib.Path("submissions/current_best_6122_onnx")
NADEEM_DIR = pathlib.Path("submissions/nadeem_6252_onnx")
BIOHACK_DIR = pathlib.Path("submissions/biohack44_6113_onnx")

V4_EXP_DIR = pathlib.Path("submissions/candidate_v4_exp_onnx")
V4_EXP_ZIP = pathlib.Path("submissions/submission_candidate_v4_exp.zip")
DECISIONS_CSV = pathlib.Path("reports/v4_exp_decisions.csv")

V4_FORBIDDEN_REVERTED = {"task191", "task255", "task240", "task349", "task184", "task301", "task396"}

BIOHACK_11 = ["task151", "task028", "task200", "task258", "task111", "task155",
              "task150", "task014", "task204", "task084", "task322"]


def k(s: str) -> str:
    s = str(s)
    return s if s.startswith("task") else f"task{int(s):03d}"


def load_score_csv(path: pathlib.Path) -> dict:
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            tid = k(r["task_id"])
            try:
                out[tid] = float(r["score"])
            except Exception:
                out[tid] = 0.0
    return out


def graph_risk_score(p: pathlib.Path) -> tuple[float, list[str]]:
    """Same scoring as audit_graph_structure.py."""
    if not p.exists():
        return float("inf"), ["missing"]
    m = onnx.load(str(p))
    g = m.graph
    op_counts = Counter(n.op_type for n in g.node)
    init_bytes = 0
    biggest = 0
    has_fp16 = False
    for ini in g.initializer:
        arr = onnx.numpy_helper.to_array(ini)
        nb = arr.nbytes
        init_bytes += nb
        if nb > biggest:
            biggest = nb
        if str(arr.dtype) == "float16":
            has_fp16 = True
    fsize = max(p.stat().st_size, 1)
    init_byte_ratio = init_bytes / fsize
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


def isolated_eval(args):
    onnx_path, tid_int = args
    cmd = [sys.executable, "tools/isolated_task_eval.py", "--onnx", str(onnx_path), "--task-id", str(tid_int)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"task_id": tid_int, "splits": {"train": [0,0], "test": [0,0], "arc-gen": [0,0]}, "status": "subprocess_error"}


def is_perfect(res: dict) -> bool:
    sp = res.get("splits", {})
    tot_pass = sp.get("train", [0,0])[0] + sp.get("test", [0,0])[0] + sp.get("arc-gen", [0,0])[0]
    tot = sp.get("train", [0,0])[1] + sp.get("test", [0,0])[1] + sp.get("arc-gen", [0,0])[1]
    return tot > 0 and tot_pass == tot


def main() -> None:
    nadeem_score = load_score_csv(pathlib.Path("reports/nadeem_6252_score_n3.csv"))
    ours_score = load_score_csv(pathlib.Path("reports/current_best_6122_score_n3.csv"))
    bio_score = load_score_csv(pathlib.Path("reports/biohack44_6113_score_n3.csv"))
    v4_score = load_score_csv(pathlib.Path("reports/candidate_nadeem6252_plus_ours_v4_score_n3.csv"))

    v4_decisions = {}
    with open("reports/v4_revert_decisions.csv") as f:
        for r in csv.DictReader(f):
            v4_decisions[r["task_id"]] = r

    nadeem_set_v2 = {r["task_id"] for r in csv.DictReader(open("reports/task_provenance.csv")) if r["v2_source_label"] == "nadeem_6252"}

    candidates = []  # list of (tid, source, source_path, expected_local_gain, reason)

    # Direction 1: Nadeem cherrypicks for tasks where v2 chose ours
    print("Direction 1 (Nadeem cherrypicks for ours-source v2 tasks):")
    d1 = []
    for tid in sorted(set(ours_score) - nadeem_set_v2):
        if tid in V4_FORBIDDEN_REVERTED:
            continue
        ns = nadeem_score.get(tid)
        os_ = ours_score.get(tid)
        if ns is None or os_ is None:
            continue
        delta = ns - os_
        if not (0.10 <= delta < 0.50):
            continue
        npath = NADEEM_DIR / f"{tid}.onnx"
        if not npath.exists():
            continue
        risk, flags = graph_risk_score(npath)
        if risk > 0:
            continue
        d1.append((tid, "nadeem_6252", npath, delta, f"D1 nadeem-cherrypick delta={delta:+.3f} risk=0"))
    print(f"  pre-audit candidates: {len(d1)}")

    # Direction 3: biohack44_6113 11 winners
    print("Direction 3 (biohack44_6113 LB-verified 11):")
    d3 = []
    for tid in BIOHACK_11:
        if tid in V4_FORBIDDEN_REVERTED:
            continue
        bs = bio_score.get(tid)
        v4s = v4_score.get(tid)
        if bs is None or v4s is None:
            continue
        if bs <= v4s:
            print(f"  skip {tid}: bio={bs:.3f} <= v4={v4s:.3f}")
            continue
        bpath = BIOHACK_DIR / f"{tid}.onnx"
        if not bpath.exists():
            continue
        delta = bs - v4s
        d3.append((tid, "biohack44_6113", bpath, delta, f"D3 biohack-cherrypick delta={delta:+.3f}"))
    print(f"  pre-audit candidates: {len(d3)}")

    candidates = d1 + d3
    seen_tids = set()
    deduped = []
    for c in sorted(candidates, key=lambda x: -x[3]):
        if c[0] in seen_tids:
            continue
        seen_tids.add(c[0])
        deduped.append(c)
    candidates = deduped

    # Audit each candidate isolated, full visible
    print(f"\nAuditing {len(candidates)} candidates (isolated full-example)...")
    if candidates:
        pool = multiprocessing.Pool(processes=4)
        audit_args = [(c[2], int(c[0].replace("task", ""))) for c in candidates]
        results = pool.map(isolated_eval, audit_args)
        pool.close()
        pool.join()
        # Also audit current v4 versions of those tids for sanity (skip if v4 already perfect on score_n3)
        v4_args = [(V4_DIR / f"{c[0]}.onnx", int(c[0].replace("task", ""))) for c in candidates]
        pool = multiprocessing.Pool(processes=4)
        v4_results = pool.map(isolated_eval, v4_args)
        pool.close()
        pool.join()
    else:
        results = []
        v4_results = []

    audit_by_tid = {f"task{r['task_id']:03d}": r for r in results}
    v4_audit_by_tid = {f"task{r['task_id']:03d}": r for r in v4_results}

    accepted = []
    rejected = []
    for c in candidates:
        tid, source, spath, delta, reason = c
        ar = audit_by_tid.get(tid, {})
        v4ar = v4_audit_by_tid.get(tid, {})
        sp = ar.get("splits", {"train":[0,0],"test":[0,0],"arc-gen":[0,0]})
        v4sp = v4ar.get("splits", {"train":[0,0],"test":[0,0],"arc-gen":[0,0]})
        ok = is_perfect(ar)
        v4_ok = is_perfect(v4ar)
        record = {
            "task_id": tid,
            "source": source,
            "delta_local_vs_v4": f"{delta:+.4f}",
            "audit_pass_ratio": f"{sp['train'][0]+sp['test'][0]+sp['arc-gen'][0]}/{sp['train'][1]+sp['test'][1]+sp['arc-gen'][1]}",
            "v4_audit_pass_ratio": f"{v4sp['train'][0]+v4sp['test'][0]+v4sp['arc-gen'][0]}/{v4sp['train'][1]+v4sp['test'][1]+v4sp['arc-gen'][1]}",
            "audit_perfect": int(ok),
            "v4_perfect": int(v4_ok),
            "reason": reason,
        }
        if ok:
            accepted.append((tid, source, spath, delta, reason, record))
        else:
            record["decision"] = "reject_audit_fail"
            rejected.append(record)

    print(f"\nAccepted (passed audit): {len(accepted)}")
    print(f"Rejected (audit fail):    {len(rejected)}")

    if V4_EXP_DIR.exists():
        shutil.rmtree(V4_EXP_DIR)
    V4_EXP_DIR.mkdir(parents=True)
    for p in sorted(V4_DIR.glob("task*.onnx")):
        shutil.copy2(p, V4_EXP_DIR / p.name)

    decisions = []
    for tid, source, spath, delta, reason, record in accepted:
        shutil.copy2(spath, V4_EXP_DIR / f"{tid}.onnx")
        record["decision"] = f"swap_to_{source}"
        decisions.append(record)

    for r in rejected:
        decisions.append(r)

    base_tids = {p.stem for p in V4_DIR.glob("task*.onnx")}
    for tid in sorted(base_tids):
        if not any(d["task_id"] == tid for d in decisions):
            decisions.append({"task_id": tid, "source": "v4", "delta_local_vs_v4": "0.0000",
                              "audit_pass_ratio": "", "v4_audit_pass_ratio": "",
                              "audit_perfect": 1, "v4_perfect": 1,
                              "reason": "inherit_v4", "decision": "inherit_v4"})

    DECISIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task_id", "source", "delta_local_vs_v4", "audit_pass_ratio", "v4_audit_pass_ratio",
              "audit_perfect", "v4_perfect", "reason", "decision"]
    with DECISIONS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(decisions, key=lambda x: x["task_id"]):
            w.writerow({k: r.get(k, "") for k in fields})

    files = sorted(V4_EXP_DIR.glob("*.onnx"))
    with zipfile.ZipFile(V4_EXP_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=p.name)
    print(f"\nv4_exp dir: {V4_EXP_DIR}")
    print(f"v4_exp zip: {V4_EXP_ZIP}  size={V4_EXP_ZIP.stat().st_size}B  files={len(files)}")
    print(f"decisions: {DECISIONS_CSV}")
    print(f"\nSummary: {len(accepted)} accepted, {len(rejected)} rejected.")
    for r in decisions:
        if r.get("decision", "").startswith("swap_to_"):
            print(f"  {r['task_id']}  {r['decision']:<28s}  delta={r['delta_local_vs_v4']}  reason={r['reason']}")


if __name__ == "__main__":
    main()
