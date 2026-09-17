"""Phase B for v4_plus probe.

For each public source S and each task T, compare to v4. A swap is a
candidate iff:

    delta = score(S, T) - score(v4, T) >= MIN_DELTA   (default 0.10)

Then it must pass three gates:

    1. structural risk (audit_graph_structure-style): risk_score < 1.5
    2. isolated full-example audit: pass_count >= v4 pass_count
       (and never drop below ours-anchor pass count)
    3. decoded-output match vs v4 on all visible examples
       (informational only — failure does not auto-reject if isolated
       audit is still perfect, per task spec)

Forbidden tasks (per hard rule, see anchor_integration_ledger.md):
    task191/255/240/349/184/301/396 (v4 force-reverts)
    task319                          (frozen self-research solver)

Output:
    reports/v4_swap_candidates.csv with one row per (task_id, source)
    that crossed the delta gate, plus gate results and final decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Source registry. Keep in sync with tools/build_provenance.py and the per-
# source score CSVs under reports/.
# ---------------------------------------------------------------------------
SOURCES = [
    # 6 newly-mined / re-ingested sources for the v4_plus probe.
    ("octavi_6042",        "submissions/octavi_6042_onnx",        "reports/octavi_6042_score_n3.csv"),
    ("haoranran_6100",     "submissions/haoranran_6100_onnx",     "reports/haoranran_6100_score_n3.csv"),
    ("afr1ste_5689",       "submissions/afr1ste_5689_onnx",       "reports/afr1ste_5689_score_n3.csv"),
    ("biohack44_super",    "submissions/biohack44_super_onnx",    "reports/biohack44_super_score_n3.csv"),
    ("biohack44_6080",     "submissions/biohack44_6080_onnx",     "reports/biohack44_6080_score_n3.csv"),
    ("biohack44_superior", "submissions/biohack44_superior_onnx", "reports/biohack44_superior_score_n3.csv"),
    # Already-on-disk sources we also screen for completeness; many of these
    # have been previously inspected so the delta gate will usually find zero
    # new candidates, but it costs us almost nothing to confirm.
    ("octaviograu_6154",   "data/octaviograu_6154/submission",    "reports/octaviograu_6154_score_n3.csv"),
    ("biohack44_6067",     "data/biohack44_6067",                 "reports/biohack44_6067_score_n3.csv"),
    ("biohack44_6113",     "submissions/biohack44_6113_onnx",     "reports/biohack44_6113_score_n3.csv"),
    ("massimiliano_eda111","submissions/massimiliano_eda111_onnx","reports/massimiliano_eda111_score_n3.csv"),
]


V4_DIR = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v4_onnx")
OURS_DIR = pathlib.Path("submissions/current_best_6122_onnx")
V4_SCORE_CSV = pathlib.Path("reports/candidate_nadeem6252_plus_ours_v4_score_n3.csv")
OURS_SCORE_CSV = pathlib.Path("reports/current_best_6122_score_n3.csv")
COMP_DIR = pathlib.Path("data/neurogolf-2026/raw")

FORBIDDEN = {191, 255, 240, 349, 184, 301, 396, 319}

MIN_DELTA = 0.10
STRUCT_RISK_MAX = 1.5


# ---------------------------------------------------------------------------


@dataclass
class TaskScore:
    cost: int = 0
    score: float = 0.0
    local_pass: int = 0
    local_total: int = 0
    error: str = ""


def load_score_csv(path: pathlib.Path) -> dict[int, TaskScore]:
    out: dict[int, TaskScore] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                tid = int(r["task_id"])
            except (KeyError, ValueError):
                continue
            try:
                cost = int(float(r.get("cost") or 0))
            except (TypeError, ValueError):
                cost = 0
            try:
                score = float(r.get("score") or 0)
            except (TypeError, ValueError):
                score = 0.0
            try:
                lp = int(r.get("local_pass") or 0)
                lt = int(r.get("local_total") or 0)
            except (TypeError, ValueError):
                lp = lt = 0
            out[tid] = TaskScore(cost=cost, score=score, local_pass=lp, local_total=lt, error=(r.get("error") or "").strip())
    return out


# ---------------------------------------------------------------------------
# Structural risk audit (mirrors tools/audit_graph_structure.py).
# ---------------------------------------------------------------------------


def numpy_size(t: onnx.TensorProto) -> tuple[int, int, str]:
    arr = onnx.numpy_helper.to_array(t)
    return arr.size, arr.nbytes, str(arr.dtype)


def structural_audit(path: pathlib.Path) -> tuple[float, list[str], dict]:
    m = onnx.load(str(path))
    g = m.graph
    op_counts = Counter(n.op_type for n in g.node)
    init_bytes = 0
    biggest = (0, "", 0, "")
    has_fp16 = False
    for ini in g.initializer:
        size, nbytes, dt = numpy_size(ini)
        init_bytes += nbytes
        if nbytes > biggest[0]:
            biggest = (nbytes, ini.name, size, dt)
        if dt == "float16":
            has_fp16 = True
    file_size = path.stat().st_size
    init_byte_ratio = init_bytes / max(file_size, 1)
    has_lookup = any(op in op_counts for op in ("GatherND", "ScatterND", "EyeLike", "OneHot", "TopK"))
    flags = []
    score = 0.0
    if init_byte_ratio > 0.7:
        flags.append("high_init_ratio")
        score += 1
    if biggest[0] > 524288:
        flags.append("huge_init>=512KB")
        score += 2
    elif biggest[0] > 65536:
        flags.append("big_init>=64KB")
        score += 1
    if has_lookup:
        flags.append("has_lookup_ops")
        score += 0.5
    if has_fp16:
        flags.append("fp16")
        score += 0.5
    info = {
        "init_byte_ratio": round(init_byte_ratio, 4),
        "biggest_init_bytes": biggest[0],
        "biggest_init_dtype": biggest[3],
    }
    return score, flags, info


# ---------------------------------------------------------------------------
# Isolated full-example audit (calls tools/isolated_task_eval.py per file).
# ---------------------------------------------------------------------------


def isolated_eval(onnx_path: pathlib.Path, task_id: int) -> dict:
    cmd = [sys.executable, "tools/isolated_task_eval.py", "--onnx", str(onnx_path), "--task-id", str(task_id)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"task_id": task_id, "splits": {"train": [0, 0], "test": [0, 0], "arc-gen": [0, 0]}, "status": "subprocess_error"}


def pass_total(res: dict) -> tuple[int, int]:
    sp = res.get("splits", {})
    p = sum(sp.get(k, [0, 0])[0] for k in ("train", "test", "arc-gen"))
    t = sum(sp.get(k, [0, 0])[1] for k in ("train", "test", "arc-gen"))
    return p, t


# ---------------------------------------------------------------------------
# Decoded-output equivalence with v4 (per task) on visible examples.
# ---------------------------------------------------------------------------


def encode_grid(grid):
    arr = np.array(grid, dtype=np.int32)
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            color = int(arr[row, col])
            if 0 <= color < 10:
                out[0, color, row, col] = 1.0
    return out


def decoded_match(v4_path: pathlib.Path, cand_path: pathlib.Path, examples) -> bool:
    sa = ort.InferenceSession(str(v4_path), providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(str(cand_path), providers=["CPUExecutionProvider"])
    na = sa.get_inputs()[0].name
    nb = sb.get_inputs()[0].name
    for ex in examples:
        ig = ex["input"]
        og = ex["output"]
        if max(len(ig), len(ig[0]), len(og), len(og[0])) > 30:
            continue
        x = encode_grid(ig)
        try:
            a = sa.run(None, {na: x})[0]
            b = sb.run(None, {nb: x})[0]
        except Exception:
            return False
        th, tw = len(og), len(og[0])
        if a.shape[2] < th or a.shape[3] < tw:
            return False
        if b.shape[2] < th or b.shape[3] < tw:
            return False
        if not np.array_equal(np.argmax(a[0, :, :th, :tw], axis=0), np.argmax(b[0, :, :th, :tw], axis=0)):
            return False
    return True


def all_examples_for(task_id: int) -> list:
    p = COMP_DIR / f"task{task_id:03d}.json"
    if not p.exists():
        return []
    j = json.loads(p.read_text())
    return j.get("train", []) + j.get("test", []) + j.get("arc-gen", [])


# ---------------------------------------------------------------------------
# Per-(task, source) worker.
# ---------------------------------------------------------------------------


def process_candidate(args: tuple) -> dict:
    task_id, src_label, src_dir_str, v4_score, ours_score = args
    src_dir = pathlib.Path(src_dir_str)
    src_path = src_dir / f"task{task_id:03d}.onnx"
    v4_path = V4_DIR / f"task{task_id:03d}.onnx"
    ours_path = OURS_DIR / f"task{task_id:03d}.onnx"
    out = {
        "task_id": f"task{task_id:03d}",
        "source": src_label,
        "v4_score": f"{v4_score:.4f}",
        "src_score": "",
        "delta": "",
        "struct_risk": "",
        "struct_flags": "",
        "isolated_v4_pass": "",
        "isolated_src_pass": "",
        "isolated_ours_pass": "",
        "decoded_match": "",
        "accepted": "False",
        "reject_reason": "",
    }

    if not src_path.is_file():
        out["reject_reason"] = "missing_source"
        return out

    # Struct audit
    try:
        risk, flags, info = structural_audit(src_path)
    except Exception as exc:
        risk, flags = 99.0, [f"struct_audit_error:{exc!r}"]
    out["struct_risk"] = f"{risk:.1f}"
    out["struct_flags"] = ";".join(flags)
    if risk >= STRUCT_RISK_MAX:
        out["reject_reason"] = "struct_risk"
        return out

    # Isolated audits (v4 + src + ours). Run ours only when needed for
    # context but always include for the CSV.
    iso_src = isolated_eval(src_path, task_id)
    sp_src, st_src = pass_total(iso_src)
    iso_v4 = isolated_eval(v4_path, task_id)
    sp_v4, st_v4 = pass_total(iso_v4)
    iso_ours = isolated_eval(ours_path, task_id)
    sp_ours, st_ours = pass_total(iso_ours)
    out["isolated_v4_pass"] = f"{sp_v4}/{st_v4}"
    out["isolated_src_pass"] = f"{sp_src}/{st_src}"
    out["isolated_ours_pass"] = f"{sp_ours}/{st_ours}"
    src_score_iso = iso_src.get("score", 0.0)
    out["src_score"] = f"{src_score_iso:.4f}"
    delta_iso = src_score_iso - v4_score
    out["delta"] = f"{delta_iso:+.4f}"

    # Reconfirm delta on isolated path (the per-source CSV may be slightly
    # different from the isolated full-example cost).
    if delta_iso < MIN_DELTA:
        out["reject_reason"] = f"delta_below_threshold_iso({delta_iso:+.4f})"
        return out

    if st_src == 0:
        out["reject_reason"] = "no_visible_examples"
        return out

    if sp_src < st_src:
        # Source itself fails some visible example. Require strict
        # equality between source-pass and v4-pass to allow the swap.
        if sp_src < sp_v4:
            out["reject_reason"] = f"isolated_src_pass_below_v4({sp_src}<{sp_v4})"
            return out
        if sp_src < sp_ours:
            out["reject_reason"] = f"isolated_src_pass_below_ours({sp_src}<{sp_ours})"
            return out
        # Same n_fail as v4 means the same hidden-set behaviour is plausibly
        # preserved (think: task018-class tasks where both visible-fail).
        out["reject_reason"] = ""  # allowed; n_fail equals v4

    # Decoded equality with v4 (informational).
    if sp_src == st_src and st_src > 0:
        try:
            exs = all_examples_for(task_id)
            ok_decode = decoded_match(v4_path, src_path, exs)
        except Exception as exc:
            ok_decode = False
            out["struct_flags"] = (out["struct_flags"] + ";" if out["struct_flags"] else "") + f"decode_err:{exc!r}"
        out["decoded_match"] = "True" if ok_decode else "False"
    else:
        out["decoded_match"] = "NA"

    out["accepted"] = "True"
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=MIN_DELTA)
    parser.add_argument("--out", default="reports/v4_swap_candidates.csv")
    args = parser.parse_args()

    v4_scores = load_score_csv(V4_SCORE_CSV)
    ours_scores = load_score_csv(OURS_SCORE_CSV)

    raw_candidates: list[tuple] = []
    delta_dist: dict[str, dict[str, int]] = {}
    for label, src_dir, csv_path in SOURCES:
        src_scores = load_score_csv(pathlib.Path(csv_path))
        bucket = {"missing": 0, "neg": 0, "lt_min": 0, "ge_min": 0, "forbidden": 0, "src_unscorable": 0}
        for tid in range(1, 401):
            v4 = v4_scores.get(tid)
            src = src_scores.get(tid)
            if v4 is None or src is None:
                bucket["missing"] += 1
                continue
            if tid in FORBIDDEN:
                bucket["forbidden"] += 1
                continue
            if src.error or src.score <= 0:
                bucket["src_unscorable"] += 1
                continue
            delta = src.score - v4.score
            if delta < 0:
                bucket["neg"] += 1
            elif delta < args.min_delta:
                bucket["lt_min"] += 1
            else:
                bucket["ge_min"] += 1
                raw_candidates.append((tid, label, src_dir, v4.score, ours_scores.get(tid).score if ours_scores.get(tid) else 0.0))
        delta_dist[label] = bucket

    print(f"per-source delta distribution (forbidden={sorted(FORBIDDEN)} excluded):")
    for label, b in delta_dist.items():
        print(f"  {label:<22s} ge_min={b['ge_min']:3d} lt_min={b['lt_min']:3d} neg={b['neg']:3d} src_unscorable={b['src_unscorable']:3d} forbidden={b['forbidden']:2d} missing={b['missing']:3d}")
    print(f"\ntotal raw delta>={args.min_delta} candidates: {len(raw_candidates)}")

    # Run gates in parallel processes (isolated_task_eval already forks
    # per-task subprocesses, so we keep modest parallelism here).
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_candidate, c): c for c in raw_candidates}
        done = 0
        for f in as_completed(futs):
            row = f.result()
            rows.append(row)
            done += 1
            if done % 25 == 0 or row["accepted"] == "True":
                print(f"  [{done}/{len(raw_candidates)}] {row['task_id']:<8s} src={row['source']:<22s} delta={row['delta']:>8s} struct={row['struct_risk']:>4s} iso_src={row['isolated_src_pass']:<8s} iso_v4={row['isolated_v4_pass']:<8s} acc={row['accepted']} reason={row['reject_reason']}", flush=True)

    rows.sort(key=lambda r: (r["task_id"], r["source"]))
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["task_id", "source", "v4_score", "src_score", "delta", "struct_risk", "struct_flags", "isolated_v4_pass", "isolated_src_pass", "isolated_ours_pass", "decoded_match", "accepted", "reject_reason"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    accepted = [r for r in rows if r["accepted"] == "True"]
    reasons = Counter(r["reject_reason"] for r in rows if r["accepted"] != "True")
    print(f"\nwrote {out_path}: total rows={len(rows)} accepted={len(accepted)}")
    print("rejection reasons:")
    for rsn, n in reasons.most_common():
        print(f"  {rsn or '(none)':<60s} {n}")

    # For each accepted task, if multiple sources qualify, prefer the one
    # with the largest delta (printed as a convenience; the actual v4_plus
    # builder will read this CSV and apply the same rule).
    best_per_task: dict[str, dict] = {}
    for r in accepted:
        tid = r["task_id"]
        d = float(r["delta"])
        if tid not in best_per_task or d > float(best_per_task[tid]["delta"]):
            best_per_task[tid] = r
    print(f"\nunique accepted task swaps: {len(best_per_task)}")
    for tid in sorted(best_per_task):
        r = best_per_task[tid]
        print(f"  {tid}  +{float(r['delta']):.4f}  <-  {r['source']}  (iso_src {r['isolated_src_pass']} vs v4 {r['isolated_v4_pass']})")


if __name__ == "__main__":
    main()
