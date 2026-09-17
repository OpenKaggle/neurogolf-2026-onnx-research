"""Build v5 = v4 plus an extra revert pass.

Rule: of the kept_nadeem tasks in v4 (currently 101), additionally revert any
whose delta vs ours is in [0.1, 0.25). This is the "extra-conservative" sister
to v4 used as a control candidate.

Output:
  submissions/candidate_nadeem6252_plus_ours_v5_onnx/
  submissions/submission_candidate_nadeem6252_plus_ours_v5.zip
  reports/v5_revert_decisions.csv

Notes:
  - Builds on v4 (NOT v3): all v4 reverts are preserved.
  - Only delta cutoff differs from v4. We do not loosen audit/struct rules.
"""
from __future__ import annotations

import csv
import pathlib
import shutil
import zipfile


V4_DIR = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v4_onnx")
OURS_DIR = pathlib.Path("submissions/current_best_6122_onnx")
V5_DIR = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v5_onnx")
V5_ZIP = pathlib.Path("submissions/submission_candidate_nadeem6252_plus_ours_v5.zip")
DECISIONS_CSV = pathlib.Path("reports/v5_revert_decisions.csv")

DELTA_LO = 0.10
DELTA_HI = 0.25


def k(s: str) -> str:
    s = str(s)
    return s if s.startswith("task") else f"task{int(s):03d}"


def main() -> None:
    nadeem_score = {k(r["task_id"]): float(r["score"]) for r in csv.DictReader(open("reports/nadeem_6252_score_n3.csv"))}
    ours_score = {k(r["task_id"]): float(r["score"]) for r in csv.DictReader(open("reports/current_best_6122_score_n3.csv"))}

    v4_decisions = {}
    with open("reports/v4_revert_decisions.csv") as f:
        for r in csv.DictReader(f):
            v4_decisions[r["task_id"]] = r

    if V5_DIR.exists():
        shutil.rmtree(V5_DIR)
    V5_DIR.mkdir(parents=True)

    rows = []
    extra_reverted = 0
    inherited_revert = 0
    inherited_keep = 0
    for v4_path in sorted(V4_DIR.glob("task*.onnx")):
        tid = v4_path.stem
        v4_dec = v4_decisions.get(tid, {"decision": "keep_v3", "reason": "non_nadeem_source"})

        if v4_dec["decision"] == "keep_nadeem":
            delta = nadeem_score.get(tid, 0.0) - ours_score.get(tid, 0.0)
            if DELTA_LO <= delta < DELTA_HI:
                shutil.copy2(OURS_DIR / v4_path.name, V5_DIR / v4_path.name)
                extra_reverted += 1
                rows.append({
                    "task_id": tid,
                    "v4_decision": v4_dec["decision"],
                    "v5_decision": "extra_revert_to_ours",
                    "reason": f"DELTA_IN_[{DELTA_LO},{DELTA_HI})",
                    "delta_vs_ours": f"{delta:+.4f}",
                })
            else:
                shutil.copy2(v4_path, V5_DIR / v4_path.name)
                inherited_keep += 1
                rows.append({
                    "task_id": tid,
                    "v4_decision": v4_dec["decision"],
                    "v5_decision": "keep_nadeem",
                    "reason": f"delta>={DELTA_HI}",
                    "delta_vs_ours": f"{delta:+.4f}",
                })
        else:
            shutil.copy2(v4_path, V5_DIR / v4_path.name)
            if v4_dec["decision"] == "revert_to_ours":
                inherited_revert += 1
            rows.append({
                "task_id": tid,
                "v4_decision": v4_dec["decision"],
                "v5_decision": "inherit_v4",
                "reason": v4_dec.get("reason", ""),
                "delta_vs_ours": v4_dec.get("delta_vs_ours", ""),
            })

    with DECISIONS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "v4_decision", "v5_decision", "reason", "delta_vs_ours"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"v5 build: extra_reverted={extra_reverted}  inherited_keep_nadeem={inherited_keep}  inherited_revert={inherited_revert}")
    print(f"v5 dir: {V5_DIR}")

    files = sorted(V5_DIR.glob("*.onnx"))
    with zipfile.ZipFile(V5_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=p.name)
    print(f"v5 zip: {V5_ZIP}  size={V5_ZIP.stat().st_size}B  files={len(files)}")


if __name__ == "__main__":
    main()
