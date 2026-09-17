"""W1 mining for v4_plus4: revive low-delta Nadeem reverts (post-H2_full).

After H2_full was verified on v4_plus3, the v4 "struct_risk_score >= 1.5"
heuristic was demoted from "safety proxy" to "deprecated". The only remaining
v4 reverts are:
  - 1 AUDIT_REGRESSION (task191) — true visible regression, stays frozen
  - 64 LOW_DELTA(< 0.10) tasks — reverted on delta-too-small rule, NOT on
    audit grounds. Under H2_full + visible-pass-parity gate these should
    be safe to revive when v3-version visible pass count >= v4_plus3-version
    visible pass count.

This script audits the 64 LOW_DELTA candidates by running
`tools/isolated_task_eval.py` on both:
  * v4_plus3 version (== current_best_6122 / "ours" bytes for these 64)
  * v3 version at submissions/candidate_nadeem6252_plus_ours_v3_onnx/
    (already carries any opt/int/sim lossless rewrite per
    reports/v3_lossless_decisions.csv; otherwise raw Nadeem)

Accept iff v3_visible_pass >= v4_plus3_visible_pass AND no split goes from
some/N -> 0/N (silent split regression).

Outputs:
  reports/v4_plus4_swap_candidates.csv  one row per candidate (64 rows)
  reports/v4_plus4_w1_swaps.csv         accepted swaps (subset)
  submissions/candidate_v4_plus4_onnx/  staged bundle (v4_plus3 + accepted v3 overlay)

Audit, score, and submission gating are run separately (see audit_v4_plus4_full.py
and the score_bundle invocation in the ledger).
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


V4P3_DIR = pathlib.Path("submissions/candidate_v4_plus3_onnx")
V3_DIR = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v3_onnx")
V4P4_DIR = pathlib.Path("submissions/candidate_v4_plus4_onnx")

REVERT_CSV = pathlib.Path("reports/v4_revert_decisions.csv")
LOSSLESS_CSV = pathlib.Path("reports/v3_lossless_decisions.csv")
CAND_CSV = pathlib.Path("reports/v4_plus4_swap_candidates.csv")
SWAPS_CSV = pathlib.Path("reports/v4_plus4_w1_swaps.csv")


def isolated_eval(onnx_path: pathlib.Path, task_id: int) -> dict:
    """Returns dict with splits (dict of [pass,total]), cost, score, status."""
    cmd = [
        sys.executable, "tools/isolated_task_eval.py",
        "--onnx", str(onnx_path),
        "--task-id", str(task_id),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "splits": {"train": [0, 0], "test": [0, 0], "arc-gen": [0, 0]},
            "cost": None,
            "score": 0.0,
            "status": "subprocess_error",
            "stderr": proc.stderr[:200],
        }


def total_pass(splits: dict) -> tuple[int, int]:
    p = sum(splits[k][0] for k in ("train", "test", "arc-gen"))
    t = sum(splits[k][1] for k in ("train", "test", "arc-gen"))
    return p, t


def silent_split_regression(v3_splits: dict, v4p3_splits: dict) -> str | None:
    """Reject if any single split goes some/N -> 0/N. Returns reason or None."""
    for k in ("train", "test", "arc-gen"):
        v3p, v3t = v3_splits[k]
        op, ot = v4p3_splits[k]
        if ot > 0 and op > 0 and v3t > 0 and v3p == 0:
            return f"split_{k}_silent({op}/{ot}->0/{v3t})"
    return None


def evaluate(task_id: int, lossless_choice: str) -> dict:
    tid_str = f"task{task_id:03d}"
    v3_path = V3_DIR / f"{tid_str}.onnx"
    v4p3_path = V4P3_DIR / f"{tid_str}.onnx"

    v3_res = isolated_eval(v3_path, task_id)
    v4p3_res = isolated_eval(v4p3_path, task_id)

    v3_pass, v3_tot = total_pass(v3_res["splits"])
    v4p3_pass, v4p3_tot = total_pass(v4p3_res["splits"])

    if lossless_choice in ("opt", "int", "sim"):
        source = f"v3-lossless({lossless_choice})"
    else:
        source = "v3-raw-nadeem"

    accepted = False
    reject_reason = ""

    if v3_res.get("status") != "ok":
        reject_reason = f"v3_status:{v3_res.get('status')}"
    elif v4p3_res.get("status") != "ok":
        reject_reason = f"v4p3_status:{v4p3_res.get('status')}"
    elif v3_pass < v4p3_pass:
        reject_reason = f"v3_pass_below({v3_pass}<{v4p3_pass})"
    else:
        split_reg = silent_split_regression(v3_res["splits"], v4p3_res["splits"])
        if split_reg:
            reject_reason = split_reg
        else:
            accepted = True

    delta_score = float(v3_res.get("score", 0.0) or 0.0) - float(v4p3_res.get("score", 0.0) or 0.0)

    return {
        "task_id": tid_str,
        "source": source,
        "v4plus3_visible_pass": f"{v4p3_pass}/{v4p3_tot}",
        "v3_visible_pass": f"{v3_pass}/{v3_tot}",
        "v4plus3_cost": v4p3_res.get("cost"),
        "v3_cost": v3_res.get("cost"),
        "v4plus3_score": f"{float(v4p3_res.get('score',0.0) or 0.0):.4f}",
        "v3_score": f"{float(v3_res.get('score',0.0) or 0.0):.4f}",
        "delta_score": f"{delta_score:+.4f}",
        "accepted": "True" if accepted else "False",
        "reject_reason": reject_reason,
    }


def main() -> None:
    lossless: dict[str, str] = {}
    with LOSSLESS_CSV.open() as f:
        for r in csv.DictReader(f):
            lossless[r["task_id"]] = r["chosen"]

    candidates: list[tuple[int, str]] = []
    with REVERT_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["decision"] != "revert_to_ours":
                continue
            if not r["reason"].startswith("LOW_DELTA"):
                continue
            tid = int(r["task_id"].replace("task", ""))
            candidates.append((tid, lossless.get(r["task_id"], "v2")))

    print(f"LOW_DELTA candidates: {len(candidates)}", flush=True)
    if len(candidates) != 64:
        print(f"  WARNING: expected 64, got {len(candidates)}", flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(evaluate, tid, choice): tid for tid, choice in candidates}
        for f in as_completed(futs):
            row = f.result()
            rows.append(row)
            print(
                f"  {row['task_id']}  {row['source']:<22s} "
                f"v4p3={row['v4plus3_visible_pass']:<8s} v3={row['v3_visible_pass']:<8s} "
                f"cost {row['v4plus3_cost']!s}->{row['v3_cost']!s} "
                f"delta={row['delta_score']} acc={row['accepted']:<5s} {row['reject_reason']}",
                flush=True,
            )

    rows.sort(key=lambda r: r["task_id"])
    CAND_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id", "source",
        "v4plus3_visible_pass", "v3_visible_pass",
        "v4plus3_cost", "v3_cost",
        "v4plus3_score", "v3_score",
        "delta_score", "accepted", "reject_reason",
    ]
    with CAND_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    accepted_rows = [r for r in rows if r["accepted"] == "True"]
    print(f"\nwrote {CAND_CSV}: {len(rows)} candidates, {len(accepted_rows)} accepted")

    # Stage v4_plus4 dir
    if V4P4_DIR.exists():
        shutil.rmtree(V4P4_DIR)
    V4P4_DIR.mkdir(parents=True)
    for p in sorted(V4P3_DIR.glob("task*.onnx")):
        shutil.copy2(p, V4P4_DIR / p.name)

    swap_records = []
    for r in accepted_rows:
        tid_str = r["task_id"]
        v3_path = V3_DIR / f"{tid_str}.onnx"
        dst_path = V4P4_DIR / f"{tid_str}.onnx"
        shutil.copy2(v3_path, dst_path)
        swap_records.append({
            "task_id": tid_str,
            "source": r["source"],
            "src_path": str(v3_path),
            "src_sha256": hashlib.sha256(v3_path.read_bytes()).hexdigest(),
            "v4plus3_visible_pass": r["v4plus3_visible_pass"],
            "v3_visible_pass": r["v3_visible_pass"],
            "v4plus3_cost": r["v4plus3_cost"],
            "v3_cost": r["v3_cost"],
            "delta_score": r["delta_score"],
        })

    with SWAPS_CSV.open("w", newline="") as f:
        if swap_records:
            w = csv.DictWriter(f, fieldnames=list(swap_records[0].keys()))
            w.writeheader()
            w.writerows(swap_records)
        else:
            f.write("task_id,source,src_path,src_sha256,v4plus3_visible_pass,v3_visible_pass,v4plus3_cost,v3_cost,delta_score\n")

    print(f"wrote {SWAPS_CSV}: {len(swap_records)} swaps applied to {V4P4_DIR}")
    print(f"staged {V4P4_DIR} with {len(swap_records)} overlays")


if __name__ == "__main__":
    main()
