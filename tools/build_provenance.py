"""Build the task provenance and source manifest layer.

Generates two CSVs:

  reports/source_manifests/all_sources.csv
    one row per (source, task_id) with sha256, file_size, nodes.
    Sources covered: every directory of task*.onnx we have on disk and
    that we want to track.

  reports/task_provenance.csv
    one row per task_id. Columns:
      - task_id
      - v2_source         : which source this task currently comes from in
                            submissions/candidate_nadeem6252_plus_ours_v2_onnx
      - v2_sha256         : sha of the file actually staged in v2
      - v2_risk_label     : risk classification of the v2 source for this task
      - matches_*         : 1/0 columns showing whether the v2 file matches
                            each tracked source by sha. Lets you spot-check
                            "is this Nadeem task actually a konbu repackage?"
      - source_score_*    : local score from existing scoring CSVs where
                            available (current_best, nadeem_6252,
                            candidate_v2 itself)

The risk labels follow the ledger conventions in reports/research.md:

  lb-verified-current-best  current_best_6122_onnx (already on the public LB)
  safe-source               biohack44_6113_onnx (11/11 official-perfect cherry-picks)
  public-anchor-candidate   nadeem_6252_onnx (high public LB score, not
                            individually proven hidden-safe)
  cherrypick-unsafe         konbu17_v36, afr1ste_6335, octaviograu_6154
                            (confirmed regression on cherry-pick despite
                            high local pass rate)
  untested-public           biohack44_6067, needless090_v31, cdeotte, yash9439,
                            jsrdcht_6029 (not yet probed, treat as unknown)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib

import onnx


SOURCES = [
    ("current_best_6122",       "submissions/current_best_6122_onnx",       "lb-verified-current-best"),
    ("nadeem_6252",             "submissions/nadeem_6252_onnx",             "public-anchor-candidate"),
    ("candidate_v2",            "submissions/candidate_nadeem6252_plus_ours_v2_onnx", "v2-staging"),
    ("candidate_v4",            "submissions/candidate_nadeem6252_plus_ours_v4_onnx", "lb-verified-v4-anchor"),
    ("biohack44_6113",          "submissions/biohack44_6113_onnx",          "safe-source"),
    ("biohack44_6067",          "data/biohack44_6067",                      "untested-public"),
    ("konbu17_v36",             "data/konbu17_v36",                         "cherrypick-unsafe"),
    ("afr1ste_6335",            "data/afr1ste_6335",                        "cherrypick-unsafe"),
    ("octaviograu_6154",        "data/octaviograu_6154/submission",         "cherrypick-unsafe"),
    ("needless090_v31",         "data/needless090_v31",                     "untested-public"),
    ("cdeotte",                 "data/cdeotte",                             "untested-public"),
    ("yash9439",                "data/yash9439",                            "untested-public"),
    ("jsrdcht_6029",            "data/jsrdcht_6029",                        "untested-public"),
    ("massimiliano_eda111",     "submissions/massimiliano_eda111_onnx",     "untested-public"),
    # New public-source ingest 2026-06-03 (v4-plus probe).
    ("octavi_6042",             "submissions/octavi_6042_onnx",             "untested-public"),
    ("haoranran_6100",          "submissions/haoranran_6100_onnx",          "untested-public"),
    ("afr1ste_5689",            "submissions/afr1ste_5689_onnx",            "untested-public"),
    ("biohack44_super",         "submissions/biohack44_super_onnx",         "untested-public"),
    ("biohack44_6080",          "submissions/biohack44_6080_onnx",          "untested-public"),
    ("biohack44_superior",      "submissions/biohack44_superior_onnx",      "untested-public"),
]


SCORE_CSVS = {
    "current_best_6122":   "reports/current_best_6122_score_n3.csv",
    "nadeem_6252":         "reports/nadeem_6252_score_n3.csv",
    "candidate_v2":        "reports/candidate_nadeem6252_plus_ours_v2_score_n3.csv",
    "candidate_v4":        "reports/candidate_nadeem6252_plus_ours_v4_score_n3.csv",
    "massimiliano_eda111": "reports/massimiliano_eda111_score_n3.csv",
    "biohack44_6113":      "reports/biohack44_6113_score_n3.csv",
    "biohack44_6067":      "reports/biohack44_6067_score_n3.csv",
    "octaviograu_6154":    "reports/octaviograu_6154_score_n3.csv",
    "octavi_6042":         "reports/octavi_6042_score_n3.csv",
    "haoranran_6100":      "reports/haoranran_6100_score_n3.csv",
    "afr1ste_5689":        "reports/afr1ste_5689_score_n3.csv",
    "biohack44_super":     "reports/biohack44_super_score_n3.csv",
    "biohack44_6080":      "reports/biohack44_6080_score_n3.csv",
    "biohack44_superior":  "reports/biohack44_superior_score_n3.csv",
    "afr1ste_6335":        "reports/afr1ste_6335_score_n3.csv",
}


def hsh(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def node_count(p: pathlib.Path) -> int:
    try:
        m = onnx.load(str(p))
        return len(m.graph.node)
    except Exception:
        return -1


def collect_source_files(label: str, root: pathlib.Path):
    if not root.exists():
        return {}
    files = {}
    for f in sorted(root.glob("task*.onnx")):
        try:
            tid = int(f.stem.replace("task", ""))
        except ValueError:
            continue
        files[tid] = f
    return files


def load_score_csv(path: pathlib.Path) -> dict[int, float]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                tid = int(row.get("task_id") or row.get("task"))
            except (TypeError, ValueError):
                continue
            try:
                out[tid] = float(row.get("score") or 0.0)
            except (TypeError, ValueError):
                pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests-dir", default="reports/source_manifests")
    parser.add_argument("--provenance", default="reports/task_provenance.csv")
    args = parser.parse_args()

    manifests_dir = pathlib.Path(args.manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    source_files: dict[str, dict[int, pathlib.Path]] = {}
    source_risk: dict[str, str] = {}
    for label, root, risk in SOURCES:
        files = collect_source_files(label, pathlib.Path(root))
        source_files[label] = files
        source_risk[label] = risk

    scores = {label: load_score_csv(pathlib.Path(p)) for label, p in SCORE_CSVS.items()}

    all_rows = []
    for label, files in source_files.items():
        rows = []
        for tid in sorted(files):
            f = files[tid]
            row = {
                "source": label,
                "risk_label": source_risk[label],
                "task_id": f"task{tid:03d}",
                "sha256": hsh(f),
                "file_size": f.stat().st_size,
                "nodes": node_count(f),
                "local_score": "" if label not in scores else f"{scores[label].get(tid, ''):.6f}" if scores[label].get(tid) is not None else "",
            }
            rows.append(row)
            all_rows.append(row)
        manifest_path = manifests_dir / f"{label}.csv"
        with manifest_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source", "risk_label", "task_id", "sha256", "file_size", "nodes", "local_score"])
            w.writeheader()
            w.writerows(rows)

    combined_path = manifests_dir / "all_sources.csv"
    with combined_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "risk_label", "task_id", "sha256", "file_size", "nodes", "local_score"])
        w.writeheader()
        w.writerows(all_rows)

    by_source_by_task: dict[str, dict[int, dict]] = {label: {} for label in source_files}
    for row in all_rows:
        tid = int(row["task_id"].replace("task", ""))
        by_source_by_task[row["source"]][tid] = row

    v2 = source_files.get("candidate_v2", {})
    fieldnames = [
        "task_id",
        "v2_sha256",
        "v2_score",
        "v2_source_label",
        "v2_risk_label",
    ]
    other_labels = [lab for lab, _, _ in SOURCES if lab != "candidate_v2"]
    for lab in other_labels:
        fieldnames.append(f"matches_{lab}")
    fieldnames.append("known_match_summary")

    rows_out = []
    for tid in sorted(v2):
        v2_row = by_source_by_task["candidate_v2"].get(tid)
        if v2_row is None:
            continue
        cand_sha = v2_row["sha256"]
        matches = []
        for lab in other_labels:
            other_row = by_source_by_task[lab].get(tid)
            matches.append(int(bool(other_row and other_row["sha256"] == cand_sha)))

        # Resolve attribution: prefer current_best_6122, then nadeem_6252,
        # then any safe source, then any other match. This reflects how we
        # built v2 (start from Nadeem, override with ours where safer).
        prio = ["current_best_6122", "nadeem_6252", "biohack44_6113", "biohack44_6067",
                "konbu17_v36", "afr1ste_6335", "octaviograu_6154", "needless090_v31",
                "cdeotte", "yash9439", "jsrdcht_6029", "massimiliano_eda111"]
        attribution = None
        for lab in prio:
            i = other_labels.index(lab)
            if matches[i]:
                attribution = lab
                break
        if attribution is None:
            attribution = "unique"
        risk = source_risk.get(attribution, "unique")
        v2_score = scores.get("candidate_v2", {}).get(tid)
        out = {
            "task_id": f"task{tid:03d}",
            "v2_sha256": cand_sha,
            "v2_score": "" if v2_score is None else f"{v2_score:.6f}",
            "v2_source_label": attribution,
            "v2_risk_label": risk,
        }
        for lab, m in zip(other_labels, matches):
            out[f"matches_{lab}"] = m
        out["known_match_summary"] = ",".join(lab for lab, m in zip(other_labels, matches) if m) or "unique"
        rows_out.append(out)

    prov_path = pathlib.Path(args.provenance)
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with prov_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    risk_count: dict[str, int] = {}
    for r in rows_out:
        risk_count[r["v2_risk_label"]] = risk_count.get(r["v2_risk_label"], 0) + 1
    print(f"wrote {len(rows_out)} rows to {prov_path}")
    print("v2 attribution breakdown:")
    for k in sorted(risk_count):
        print(f"  {k:<30s} {risk_count[k]:>3d}")


if __name__ == "__main__":
    main()
