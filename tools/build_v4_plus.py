"""Phase C for v4_plus probe.

Reads `reports/v4_swap_candidates.csv`, copies v4 into
`submissions/candidate_v4_plus_onnx/`, then applies one swap per task
(largest accepted delta wins).

After the bundle is staged, run `tools/score_bundle.py` separately to
confirm local total > v4, then run the 400-task isolated audit
(`tools/audit_v4_plus_full.py`, mirroring `audit_v4_full.py`) and revert
any swap that regresses pass count vs v4.

Outputs:
    submissions/candidate_v4_plus_onnx/                # staged bundle
    reports/v4_plus_swap_plan.csv                      # per-task swap source
    submissions/submission_candidate_v4_plus.zip       # final zip (after audit)
    reports/v4_plus_zip_sha256.txt                     # for the ledger
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import shutil
import zipfile


V4_DIR = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v4_onnx")
V4_PLUS_DIR = pathlib.Path("submissions/candidate_v4_plus_onnx")
CAND_CSV = pathlib.Path("reports/v4_swap_candidates.csv")
PLAN_CSV = pathlib.Path("reports/v4_plus_swap_plan.csv")
ZIP_PATH = pathlib.Path("submissions/submission_candidate_v4_plus.zip")
SHA_PATH = pathlib.Path("reports/v4_plus_zip_sha256.txt")


SOURCE_DIRS = {
    "octavi_6042":         "submissions/octavi_6042_onnx",
    "haoranran_6100":      "submissions/haoranran_6100_onnx",
    "afr1ste_5689":        "submissions/afr1ste_5689_onnx",
    "biohack44_super":     "submissions/biohack44_super_onnx",
    "biohack44_6080":      "submissions/biohack44_6080_onnx",
    "biohack44_superior":  "submissions/biohack44_superior_onnx",
    "octaviograu_6154":    "data/octaviograu_6154/submission",
    "biohack44_6067":      "data/biohack44_6067",
    "biohack44_6113":      "submissions/biohack44_6113_onnx",
    "massimiliano_eda111": "submissions/massimiliano_eda111_onnx",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-zip", action="store_true", help="stage dir only; do not write zip yet (zip after audit)")
    parser.add_argument("--prefer", action="append", default=[],
                        help="repeatable: task_id=source override for tie-breaks")
    args = parser.parse_args()

    overrides: dict[str, str] = {}
    for raw in args.prefer:
        tid, src = raw.split("=", 1)
        overrides[tid.strip()] = src.strip()

    # Pick best swap per task: largest delta among accepted=True.
    if not CAND_CSV.exists():
        raise SystemExit(f"missing {CAND_CSV}; run tools/build_v4_plus_candidates.py first")

    accepted_by_task: dict[str, list[dict]] = {}
    with CAND_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["accepted"] != "True":
                continue
            accepted_by_task.setdefault(r["task_id"], []).append(r)

    plan: dict[str, dict] = {}
    for tid, rows in accepted_by_task.items():
        if tid in overrides:
            chosen = next((r for r in rows if r["source"] == overrides[tid]), None)
            if chosen is None:
                raise SystemExit(f"override {tid}={overrides[tid]} not in accepted candidates for {tid}: {[r['source'] for r in rows]}")
        else:
            chosen = max(rows, key=lambda r: float(r["delta"]))
        plan[tid] = chosen

    if V4_PLUS_DIR.exists():
        shutil.rmtree(V4_PLUS_DIR)
    V4_PLUS_DIR.mkdir(parents=True)
    for p in sorted(V4_DIR.glob("task*.onnx")):
        shutil.copy2(p, V4_PLUS_DIR / p.name)

    PLAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    plan_rows = []
    for tid in sorted(plan):
        chosen = plan[tid]
        src_dir = pathlib.Path(SOURCE_DIRS[chosen["source"]])
        src_path = src_dir / f"{tid}.onnx"
        if not src_path.is_file():
            raise SystemExit(f"missing swap source: {src_path}")
        dst_path = V4_PLUS_DIR / f"{tid}.onnx"
        shutil.copy2(src_path, dst_path)
        plan_rows.append({
            "task_id": tid,
            "source": chosen["source"],
            "delta": chosen["delta"],
            "v4_score": chosen["v4_score"],
            "src_score": chosen["src_score"],
            "isolated_src_pass": chosen["isolated_src_pass"],
            "isolated_v4_pass": chosen["isolated_v4_pass"],
            "decoded_match": chosen["decoded_match"],
            "struct_risk": chosen["struct_risk"],
            "struct_flags": chosen["struct_flags"],
            "src_path": str(src_path),
            "sha256": hashlib.sha256(src_path.read_bytes()).hexdigest(),
        })

    with PLAN_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(plan_rows[0].keys()) if plan_rows else ["task_id", "source"])
        w.writeheader()
        w.writerows(plan_rows)

    print(f"staged v4_plus at {V4_PLUS_DIR} with {len(plan_rows)} swap(s):")
    for r in plan_rows:
        print(f"  {r['task_id']}  +{float(r['delta']):.4f}  <-  {r['source']}")

    if args.skip_zip:
        print(f"--skip-zip: dir staged but no zip written; run audit then re-invoke without --skip-zip")
        return

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(V4_PLUS_DIR.glob("task*.onnx")):
            zf.write(p, arcname=p.name)
    sha = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
    SHA_PATH.write_text(sha + "\n")
    print(f"\nwrote {ZIP_PATH}  ({ZIP_PATH.stat().st_size} bytes)  sha256={sha}")


if __name__ == "__main__":
    main()
