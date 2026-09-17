"""Build the maximum-gain zip from {current, konbu, biohack} per-task best.

For each task: pick the source with highest score amongst {current_best,
konbu17_v36, biohack44_6113_onnx}, restricted to candidates that fully pass
local examples. Outputs the merged zip and a summary csv showing the picks.

Usage:
  .venv/bin/python tools/build_max_swap.py
"""
from __future__ import annotations

import hashlib
import pathlib
import shutil
import sys
import zipfile

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


CURRENT_DIR = pathlib.Path("submissions/current_best_6120plus_onnx")
KONBU_DIR = pathlib.Path("data/konbu17_v36")
BIOHACK_DIR = pathlib.Path("submissions/biohack44_6113_onnx")
CUR_CSV = "reports/current_best_6120plus_score_n3.csv"
KON_CSV = "reports/konbu17_v36_score_n3.csv"
BIO_CSV = "reports/biohack44_6113_score_n3.csv"
# Tasks known unsafe (fail full local even though n=3 csv shows pass)
UNSAFE = {191, 366}
OUT_DIR = pathlib.Path("submissions/staging_max_safe")
OUT_ZIP = pathlib.Path("submissions/submission_candidate_max_safe.zip")


def main() -> None:
    cur = pd.read_csv(CUR_CSV).set_index("task_id")
    kon = pd.read_csv(KON_CSV).set_index("task_id")
    bio = pd.read_csv(BIO_CSV).set_index("task_id")

    picks = []
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_id in cur.index:
        candidates = [("current", cur.loc[task_id], CURRENT_DIR)]
        if task_id in kon.index and task_id not in UNSAFE:
            row = kon.loc[task_id]
            if row["local_pass"] == row["local_total"] and row["score"] > 0:
                candidates.append(("konbu", row, KONBU_DIR))
        if task_id in bio.index and task_id not in UNSAFE:
            row = bio.loc[task_id]
            if row["local_pass"] == row["local_total"] and row["score"] > 0:
                candidates.append(("biohack", row, BIOHACK_DIR))
        candidates.sort(key=lambda x: x[1]["score"], reverse=True)
        chosen_name, chosen_row, chosen_dir = candidates[0]
        src = chosen_dir / f"task{task_id:03d}.onnx"
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, OUT_DIR / src.name)
        picks.append({
            "task": task_id,
            "source": chosen_name,
            "score": chosen_row["score"],
            "cost": chosen_row["cost"],
            "delta_vs_cur": chosen_row["score"] - cur.loc[task_id, "score"],
        })

    swaps = [p for p in picks if p["source"] != "current"]
    summary = pd.DataFrame(picks)
    summary.to_csv("reports/staging_max_safe_picks.csv", index=False)

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT_DIR.glob("task*.onnx")):
            zf.write(p, p.name)

    sha = hashlib.sha256(OUT_ZIP.read_bytes()).hexdigest()
    n_konbu = sum(1 for p in swaps if p["source"] == "konbu")
    n_bio = sum(1 for p in swaps if p["source"] == "biohack")
    total_gain = sum(p["delta_vs_cur"] for p in swaps)
    print(f"built {OUT_ZIP} size={OUT_ZIP.stat().st_size} sha256={sha[:32]}")
    print(f"swapped {len(swaps)} tasks: konbu={n_konbu} biohack={n_bio}")
    print(f"local cumulative gain: +{total_gain:.3f} (excluding stale-trace measurement noise)")


if __name__ == "__main__":
    main()
