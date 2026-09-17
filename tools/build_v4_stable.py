"""Build v4 stable candidate by reverting risky Nadeem-source tasks in v3.

Revert rule (any one triggers fallback to current_best_6122):
  1. audit detected REGRESSION_VS_OURS (e.g. task191)
  2. structure risk_score >= 1.5 (large fp16 initializers, suspicious lookup)
  3. delta vs ours < 0.1 (low local gain not worth hidden risk)

Output:
  submissions/candidate_nadeem6252_plus_ours_v4_onnx/
  submissions/submission_candidate_nadeem6252_plus_ours_v4.zip
  reports/v4_revert_decisions.csv
"""
from __future__ import annotations

import csv
import pathlib
import shutil
import zipfile


V3_DIR = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v3_onnx")
OURS_DIR = pathlib.Path("submissions/current_best_6122_onnx")
V4_DIR = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v4_onnx")
V4_ZIP = pathlib.Path("submissions/submission_candidate_nadeem6252_plus_ours_v4.zip")
DECISIONS_CSV = pathlib.Path("reports/v4_revert_decisions.csv")


def k(s: str) -> str:
    s = str(s)
    return s if s.startswith("task") else f"task{int(s):03d}"


def main() -> None:
    nadeem_score = {k(r["task_id"]): float(r["score"]) for r in csv.DictReader(open("reports/nadeem_6252_score_n3.csv"))}
    ours_score = {k(r["task_id"]): float(r["score"]) for r in csv.DictReader(open("reports/current_best_6122_score_n3.csv"))}
    v3_score = {k(r["task_id"]): float(r["score"]) for r in csv.DictReader(open("reports/candidate_nadeem6252_plus_ours_v3_score_n3.csv"))}

    nadeem_set = {r["task_id"] for r in csv.DictReader(open("reports/task_provenance.csv")) if r["v2_source_label"] == "nadeem_6252"}

    audit_regress = set()
    with open("reports/v3_nadeem_audit.csv") as f:
        for r in csv.DictReader(f):
            if r["risk_label"] == "REGRESSION_VS_OURS":
                audit_regress.add(r["task_id"])

    high_struct_risk = set()
    with open("reports/v3_graph_structure_audit.csv") as f:
        for r in csv.DictReader(f):
            if float(r["risk_score"]) >= 1.5:
                high_struct_risk.add(r["task_id"])

    if V4_DIR.exists():
        shutil.rmtree(V4_DIR)
    V4_DIR.mkdir(parents=True)

    decisions = []
    reverted = 0
    for v3_path in sorted(V3_DIR.glob("task*.onnx")):
        tid = v3_path.stem
        if tid not in nadeem_set:
            shutil.copy2(v3_path, V4_DIR / v3_path.name)
            decisions.append({"task_id": tid, "decision": "keep_v3", "reason": "non_nadeem_source"})
            continue

        delta = nadeem_score.get(tid, 0) - ours_score.get(tid, 0)
        reasons = []
        if tid in audit_regress:
            reasons.append("AUDIT_REGRESSION")
        if tid in high_struct_risk:
            reasons.append("HIGH_STRUCT_RISK")
        if delta < 0.1:
            reasons.append(f"LOW_DELTA({delta:+.3f})")

        if reasons:
            shutil.copy2(OURS_DIR / v3_path.name, V4_DIR / v3_path.name)
            reverted += 1
            decisions.append({"task_id": tid, "decision": "revert_to_ours", "reason": ";".join(reasons), "delta_vs_ours": f"{delta:+.4f}"})
        else:
            shutil.copy2(v3_path, V4_DIR / v3_path.name)
            decisions.append({"task_id": tid, "decision": "keep_nadeem", "reason": f"high_delta({delta:+.3f})", "delta_vs_ours": f"{delta:+.4f}"})

    with DECISIONS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "decision", "reason", "delta_vs_ours"])
        w.writeheader()
        for r in decisions:
            r.setdefault("delta_vs_ours", "")
            w.writerow(r)

    print(f"reverted {reverted} Nadeem-source tasks (out of {len(nadeem_set)})")
    print(f"v4 dir: {V4_DIR}")

    files = sorted(V4_DIR.glob("*.onnx"))
    with zipfile.ZipFile(V4_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=p.name)
    print(f"v4 zip: {V4_ZIP}  size={V4_ZIP.stat().st_size}B  files={len(files)}")


if __name__ == "__main__":
    main()
