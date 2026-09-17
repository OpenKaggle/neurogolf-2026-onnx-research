"""Build v3 candidate: v2 (Nadeem + ours reverse stack) plus best lossless rewrite per task.

For every task that has been probed by int_surgery_probe / onnx_optimize_probe
/ onnxsim_probe on top of v2, pick the highest-score variant whose
`same_decoded_outputs` already returned True (the probes only set `kept=True`
when the score strictly improves AND decoded outputs match). If no probe
keeps a variant, fall back to the v2 staged file.

Output: submissions/candidate_nadeem6252_plus_ours_v3_onnx
        submissions/submission_candidate_nadeem6252_plus_ours_v3.zip
        reports/v3_lossless_decisions.csv
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import shutil
import zipfile


V2_DIR = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v2_onnx")
V3_DIR = pathlib.Path("submissions/candidate_nadeem6252_plus_ours_v3_onnx")
V3_ZIP = pathlib.Path("submissions/submission_candidate_nadeem6252_plus_ours_v3.zip")
DECISIONS_CSV = pathlib.Path("reports/v3_lossless_decisions.csv")


PROBE_DIRS = {
    "int":  pathlib.Path("submissions/int_on_v2"),
    "opt":  pathlib.Path("submissions/opt_on_v2"),
    "sim":  pathlib.Path("submissions/sim_on_v2"),
}
PROBE_REPORTS = {
    "int":  pathlib.Path("reports/int_on_v2_report.json"),
    "opt":  pathlib.Path("reports/opt_on_v2_report.json"),
    "sim":  pathlib.Path("reports/sim_on_v2_report.json"),
}


def _load_kept(report_path: pathlib.Path) -> dict[int, dict]:
    if not report_path.is_file():
        return {}
    rows = json.loads(report_path.read_text())
    return {r["task_id"]: r for r in rows if r.get("kept")}


def main() -> None:
    V3_DIR.mkdir(parents=True, exist_ok=True)
    DECISIONS_CSV.parent.mkdir(parents=True, exist_ok=True)

    kept = {label: _load_kept(p) for label, p in PROBE_REPORTS.items()}

    decisions = []
    total_delta = 0.0

    for v2_file in sorted(V2_DIR.glob("task*.onnx")):
        tid = int(v2_file.stem.replace("task", ""))
        candidates = []
        for label in ("int", "opt", "sim"):
            r = kept[label].get(tid)
            if not r:
                continue
            cand_path = PROBE_DIRS[label] / f"task{tid:03d}.onnx"
            if not cand_path.is_file():
                continue
            candidates.append((label, r, cand_path))

        if not candidates:
            shutil.copy2(v2_file, V3_DIR / v2_file.name)
            decisions.append({
                "task_id": f"task{tid:03d}",
                "chosen": "v2",
                "delta": 0.0,
                "v2_score": "",
                "chosen_score": "",
                "v2_cost": "",
                "chosen_cost": "",
            })
            continue

        chosen_label, chosen_record, chosen_path = max(
            candidates, key=lambda c: c[1].get("candidate_score", 0.0)
        )
        shutil.copy2(chosen_path, V3_DIR / f"task{tid:03d}.onnx")
        v2_cost = chosen_record.get("original_cost")
        v2_score = chosen_record.get("original_score")
        ch_cost = chosen_record.get("candidate_cost")
        ch_score = chosen_record.get("candidate_score")
        delta = chosen_record.get("delta", 0.0)
        total_delta += delta
        decisions.append({
            "task_id": f"task{tid:03d}",
            "chosen": chosen_label,
            "delta": f"{delta:+.6f}",
            "v2_score": f"{v2_score:.6f}" if v2_score is not None else "",
            "chosen_score": f"{ch_score:.6f}" if ch_score is not None else "",
            "v2_cost": v2_cost if v2_cost is not None else "",
            "chosen_cost": ch_cost if ch_cost is not None else "",
        })

    with DECISIONS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "chosen", "delta", "v2_score", "chosen_score", "v2_cost", "chosen_cost"])
        w.writeheader()
        w.writerows(decisions)

    with zipfile.ZipFile(V3_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(V3_DIR.glob("task*.onnx")):
            zf.write(f, arcname=f.name)
    sha = hashlib.sha256(V3_ZIP.read_bytes()).hexdigest()

    n_changed = sum(1 for d in decisions if d["chosen"] != "v2")
    print(f"v3 directory: {V3_DIR}")
    print(f"v3 zip: {V3_ZIP} bytes={V3_ZIP.stat().st_size} sha256={sha}")
    print(f"tasks changed vs v2: {n_changed}/400, expected lossless delta: {total_delta:+.4f}")
    print(f"per-task decisions: {DECISIONS_CSV}")


if __name__ == "__main__":
    main()
