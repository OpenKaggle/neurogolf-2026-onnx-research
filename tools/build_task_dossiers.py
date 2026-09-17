"""Build lightweight task dossiers and an attempt registry.

The output is meant for agents, not for a UI.  It joins the current anchor
score/cost rows with source provenance, family features, public-tail tags,
pseudo-hidden audit summaries, task-specific builders/prototypes, and a small
next-action hint.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
from collections import Counter, defaultdict
from datetime import datetime


PUBLIC_BOTTOM15 = {
    158,
    233,
    173,
    54,
    25,
    285,
    366,
    133,
    286,
    255,
    349,
    18,
    187,
    145,
    243,
}

PUBLIC_BIG_GAP = {255, 233, 366, 18, 285, 133, 243}

PUBLIC_OVERFIT_RISK = {192, 319, 118, 359, 18, 285, 96, 48, 355, 219}

PUBLIC_SLOW_TASKS = {358, 350, 212, 335, 246, 22, 375, 9, 74, 70}

FROZEN_TASKS = {
    319: "hidden-safety audit failed; current ONNX milestones are engineering artifacts only",
}

SPECIAL_ACTIONS = {
    18: "Do not cost-only replace; run scorer-decoder or semantic audit before any probe.",
    219: "Avoid visible-family lookup staging; generalize the closed Python rule against pseudo-hidden shape families.",
    255: "Retire brittle variants; rebuild semantic rectangle/gutter rule and demand clean pseudo-hidden before ONNX.",
    319: "FROZEN: no submission path; revisit only with a new hidden-safety theory.",
}

SOURCE_PRIORITY = {
    "v4_plus10_compiler2_unique": 0,
    "candidate_v4": 1,
    "current_best_6122": 2,
    "biohack44_6113": 3,
    "nadeem_6252": 4,
    "biohack44_super": 5,
}

CSV_FIELDS = [
    "task_id",
    "anchor_score",
    "anchor_cost",
    "anchor_memory",
    "anchor_params",
    "anchor_nodes",
    "anchor_file_size",
    "local_pass",
    "local_total",
    "source_label",
    "source_risk",
    "source_matches",
    "known_match_summary",
    "best_seen_score",
    "best_seen_cost",
    "best_seen_attempt",
    "attempt_count",
    "family_primary",
    "family_tags",
    "family_features",
    "shape_profile",
    "public_tail_tags",
    "pseudo_hidden_status",
    "pseudo_hidden_summary",
    "known_prototypes",
    "known_builders",
    "known_reports",
    "next_allowed_action",
    "priority_score",
]


def norm_task_id(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def task_label(task_id: int) -> str:
    return f"task{task_id:03d}"


def to_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def to_int(value: object, default: int | None = None) -> int | None:
    number = to_float(value)
    if number is None:
        return default
    return int(number)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fmt_float(value: object, digits: int = 6) -> str:
    number = to_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def fmt_int(value: object) -> str:
    number = to_int(value)
    if number is None:
        return ""
    return str(number)


def read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_anchor_scores(path: pathlib.Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for row in read_csv_rows(path):
        tid = norm_task_id(row.get("task_id") or row.get("task"))
        if tid is not None:
            rows[tid] = row
    return rows


def load_attempts(reports_dir: pathlib.Path) -> dict[int, list[dict[str, object]]]:
    attempts: dict[int, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(reports_dir.glob("*score*.csv")):
        if path.name.startswith("task_dossiers_"):
            continue
        label = path.stem
        for row in read_csv_rows(path):
            tid = norm_task_id(row.get("task_id") or row.get("task"))
            score = to_float(row.get("score"))
            cost = to_int(row.get("cost"))
            if tid is None or (score is None and cost is None):
                continue
            attempts[tid].append(
                {
                    "attempt": label,
                    "score": score,
                    "cost": cost,
                    "memory": to_int(row.get("memory")),
                    "params": to_int(row.get("params")),
                    "local_pass": to_int(row.get("local_pass")),
                    "local_total": to_int(row.get("local_total")),
                    "nodes": to_int(row.get("nodes")),
                    "file_size": to_int(row.get("file_size")),
                    "path": row.get("path", ""),
                }
            )
    return attempts


def summarize_attempts(attempts: list[dict[str, object]]) -> dict[str, object]:
    if not attempts:
        return {
            "attempt_count": 0,
            "best_seen_score": None,
            "best_seen_cost": None,
            "best_seen_attempt": "",
            "top_attempts": [],
        }
    score_ranked = sorted(
        attempts,
        key=lambda item: (
            to_float(item.get("score"), -9999.0) or -9999.0,
            -(to_int(item.get("cost"), 10**18) or 10**18),
        ),
        reverse=True,
    )
    cost_values = [to_int(item.get("cost")) for item in attempts if to_int(item.get("cost")) is not None]
    best = score_ranked[0]
    return {
        "attempt_count": len(attempts),
        "best_seen_score": best.get("score"),
        "best_seen_cost": min(cost_values) if cost_values else None,
        "best_seen_attempt": best.get("attempt", ""),
        "top_attempts": score_ranked[:6],
    }


def load_source_provenance(
    reports_dir: pathlib.Path,
    current_manifest_path: pathlib.Path,
) -> tuple[dict[int, dict[str, object]], dict[int, dict[str, str]]]:
    all_sources = read_csv_rows(reports_dir / "source_manifests" / "all_sources.csv")
    by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_sources:
        sha = row.get("sha256", "")
        if sha:
            by_sha[sha].append(row)

    current_manifest: dict[int, dict[str, str]] = {}
    for row in read_csv_rows(current_manifest_path):
        tid = norm_task_id(row.get("task_id"))
        if tid is not None:
            current_manifest[tid] = row

    v2_provenance: dict[int, dict[str, str]] = {}
    for row in read_csv_rows(reports_dir / "task_provenance.csv"):
        tid = norm_task_id(row.get("task_id"))
        if tid is not None:
            v2_provenance[tid] = row

    result: dict[int, dict[str, object]] = {}
    for tid in range(1, 401):
        manifest = current_manifest.get(tid, {})
        sha = manifest.get("candidate_sha256", "")
        matches = by_sha.get(sha, [])
        match_labels = sorted({row.get("source", "") for row in matches if row.get("source")})
        selected = select_source(match_labels, matches)
        fallback = v2_provenance.get(tid, {})
        if not selected:
            if sha:
                selected = {
                    "source": "v4_plus10_compiler2_unique",
                    "risk_label": "lb-verified-current-anchor",
                }
            else:
                selected = {
                    "source": fallback.get("v2_source_label", ""),
                    "risk_label": fallback.get("v2_risk_label", ""),
                }
        result[tid] = {
            "source_label": selected.get("source", ""),
            "source_risk": selected.get("risk_label", ""),
            "source_matches": match_labels,
            "known_match_summary": fallback.get("known_match_summary", ""),
            "current_sha256": sha,
            "manifest_changed": as_bool(manifest.get("changed", "False")),
        }
    return result, v2_provenance


def select_source(labels: list[str], rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not labels:
        return None
    label = sorted(labels, key=lambda item: (SOURCE_PRIORITY.get(item, 100), item))[0]
    for row in rows:
        if row.get("source") == label:
            return row
    return {"source": label, "risk_label": ""}


def load_pattern_table(path: pathlib.Path) -> dict[int, dict[str, str]]:
    patterns: dict[int, dict[str, str]] = {}
    if not path.exists():
        return patterns
    row_re = re.compile(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|")
    for line in path.read_text().splitlines():
        match = row_re.match(line)
        if not match:
            continue
        tid = int(match.group(1))
        patterns[tid] = {
            "primary": match.group(2).strip(),
            "tags": match.group(3).strip(),
        }
    return patterns


def load_raw_task_summary(raw_dir: pathlib.Path) -> dict[int, dict[str, object]]:
    summaries: dict[int, dict[str, object]] = {}
    for path in sorted(raw_dir.glob("task*.json")):
        tid = norm_task_id(path.stem)
        if tid is None:
            continue
        try:
            task = json.loads(path.read_text())
        except Exception:
            continue
        train = task.get("train", [])
        test = task.get("test", [])
        examples = train + test
        if not examples:
            continue
        in_shapes: list[tuple[int, int]] = []
        out_shapes: list[tuple[int, int]] = []
        in_palettes: list[int] = []
        out_palettes: list[int] = []
        for example in examples:
            inp = example.get("input", [])
            out = example.get("output", [])
            if inp and out:
                in_shapes.append((len(inp), len(inp[0])))
                out_shapes.append((len(out), len(out[0])))
                in_palettes.append(len({cell for row in inp for cell in row}))
                out_palettes.append(len({cell for row in out for cell in row}))
        if not in_shapes:
            continue
        summaries[tid] = {
            "n_train": len(train),
            "n_test": len(test),
            "min_h_in": min(h for h, _ in in_shapes),
            "max_h_in": max(h for h, _ in in_shapes),
            "min_w_in": min(w for _, w in in_shapes),
            "max_w_in": max(w for _, w in in_shapes),
            "min_h_out": min(h for h, _ in out_shapes),
            "max_h_out": max(h for h, _ in out_shapes),
            "min_w_out": min(w for _, w in out_shapes),
            "max_w_out": max(w for _, w in out_shapes),
            "in_palette_size": max(in_palettes),
            "out_palette_size": max(out_palettes),
        }
    return summaries


def load_family_features(
    reports_dir: pathlib.Path,
    raw_dir: pathlib.Path,
) -> dict[int, dict[str, object]]:
    pattern_table = load_pattern_table(reports_dir / "task_pattern_analysis.md")
    raw_summary = load_raw_task_summary(raw_dir)
    families: dict[int, dict[str, object]] = {}

    for row in read_csv_rows(reports_dir / "uncovered_task_features.csv"):
        tid = norm_task_id(row.get("task_id"))
        if tid is None:
            continue
        pattern = pattern_table.get(tid, {})
        tags = split_tags(pattern.get("tags") or derive_feature_tags(row))
        primary = pattern.get("primary") or derive_primary(row, tags)
        families[tid] = {
            "primary": primary,
            "tags": tags,
            "features": row,
            "shape_profile": shape_profile(row),
            "family_features": feature_summary(row),
        }

    for tid in range(1, 401):
        if tid in families:
            continue
        raw = raw_summary.get(tid, {})
        families[tid] = {
            "primary": "covered_or_unclassified",
            "tags": ["covered_or_unclassified"],
            "features": raw,
            "shape_profile": shape_profile(raw),
            "family_features": "covered_by_existing_dsl_or_missing_feature_report",
        }
    return families


def split_tags(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def derive_feature_tags(row: dict[str, object]) -> str:
    tags = []
    checks = [
        ("tile", "tiling"),
        ("crop", "crop"),
        ("trim_border", "trim_border"),
        ("scalar_output", "scalar_output"),
        ("expand", "expand"),
        ("reduce", "reduce"),
        ("color_remap", "color_remap"),
        ("color_fill", "color_fill"),
        ("symmetry_candidate", "symmetry/flip/rotate"),
    ]
    for key, label in checks:
        if as_bool(row.get(key)):
            tags.append(label)
    if not tags and as_bool(row.get("out_smaller")):
        tags.append("reduce")
    if not tags and as_bool(row.get("out_larger")):
        tags.append("expand")
    return ", ".join(tags or ["other"])


def derive_primary(row: dict[str, object], tags: list[str]) -> str:
    priority = [
        "tiling",
        "crop",
        "trim_border",
        "scalar_output",
        "expand",
        "reduce",
        "color_remap",
        "color_fill",
        "symmetry/flip/rotate",
        "other",
    ]
    tag_set = set(tags)
    for item in priority:
        if item in tag_set:
            return item
    return "other"


def shape_profile(row: dict[str, object]) -> str:
    values = {key: row.get(key, "") for key in row}
    if not values:
        return ""
    return (
        f"in {values.get('min_h_in', '?')}-{values.get('max_h_in', '?')}x"
        f"{values.get('min_w_in', '?')}-{values.get('max_w_in', '?')} -> "
        f"out {values.get('min_h_out', '?')}-{values.get('max_h_out', '?')}x"
        f"{values.get('min_w_out', '?')}-{values.get('max_w_out', '?')}; "
        f"pal {values.get('in_palette_size', '?')}->{values.get('out_palette_size', '?')}"
    )


def feature_summary(row: dict[str, object]) -> str:
    flags = []
    for key in [
        "shape_preserving",
        "tile",
        "crop",
        "trim_border",
        "out_larger",
        "out_smaller",
        "color_subset",
        "color_superset",
        "color_same",
        "color_disjoint",
        "scalar_output",
        "symmetry_candidate",
        "color_remap",
        "color_fill",
        "expand",
        "reduce",
    ]:
        if as_bool(row.get(key)):
            flags.append(key)
    n_train = row.get("n_train", "")
    n_test = row.get("n_test", "")
    prefix = f"train={n_train};test={n_test}" if n_train != "" or n_test != "" else ""
    return ";".join([part for part in [prefix, ",".join(flags)] if part])


def collect_task_files(root: pathlib.Path, pattern: str) -> dict[int, list[str]]:
    found: dict[int, list[str]] = defaultdict(list)
    for path in sorted(root.glob(pattern)):
        tid = norm_task_id(path.name)
        if tid is None:
            continue
        found[tid].append(path.name)
    return found


def parse_json_pseudo(path: pathlib.Path) -> dict[str, int]:
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return {"passed": 0, "failed": 0, "skipped": 0, "total": 0}
    counts = {"passed": 0, "failed": 0, "skipped": 0, "total": 0}
    absorb_summary(obj, counts)
    return counts


def absorb_summary(obj: object, counts: dict[str, int]) -> None:
    if isinstance(obj, dict):
        if {"passed", "failed", "skipped", "total"} <= set(obj):
            counts["passed"] += to_int(obj.get("passed"), 0) or 0
            counts["failed"] += to_int(obj.get("failed"), 0) or 0
            counts["skipped"] += to_int(obj.get("skipped"), 0) or 0
            counts["total"] += to_int(obj.get("total"), 0) or 0
            return
        if "rows" in obj and isinstance(obj["rows"], list):
            for row in obj["rows"]:
                if not isinstance(row, dict):
                    continue
                skipped = as_bool(row.get("skipped")) or not as_bool(row.get("eligible", "1"))
                exact = as_bool(row.get("exact"))
                counts["total"] += 1
                if skipped:
                    counts["skipped"] += 1
                elif exact:
                    counts["passed"] += 1
                else:
                    counts["failed"] += 1
            return
        for value in obj.values():
            absorb_summary(value, counts)
    elif isinstance(obj, list):
        for value in obj:
            absorb_summary(value, counts)


def parse_csv_pseudo(path: pathlib.Path) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "total": 0}
    for row in read_csv_rows(path):
        if "exact_rate" in row and "eligible" in row:
            eligible = to_int(row.get("eligible"), 0) or 0
            exact = to_int(row.get("exact"), 0) or 0
            total_rows = to_int(row.get("total_rows"), eligible) or eligible
            counts["total"] += total_rows
            counts["passed"] += min(exact, eligible)
            counts["failed"] += max(eligible - exact, 0)
            counts["skipped"] += max(total_rows - eligible, 0)
            continue
        if "passed" in row and "skipped" in row:
            counts["total"] += 1
            if as_bool(row.get("skipped")):
                counts["skipped"] += 1
            elif as_bool(row.get("passed")):
                counts["passed"] += 1
            else:
                counts["failed"] += 1
    return counts


def load_pseudo_hidden(reports_dir: pathlib.Path) -> dict[int, dict[str, object]]:
    summaries: dict[int, dict[str, object]] = {}
    by_task: dict[int, list[pathlib.Path]] = defaultdict(list)
    for path in sorted(reports_dir.glob("task*")):
        if "pseudo_hidden" not in path.name:
            continue
        tid = norm_task_id(path.name)
        if tid is not None and path.suffix in {".csv", ".json"}:
            by_task[tid].append(path)

    for tid in range(1, 401):
        counts = {"passed": 0, "failed": 0, "skipped": 0, "total": 0}
        paths = by_task.get(tid, [])
        count_paths = select_pseudo_count_paths(paths)
        for path in count_paths:
            parsed = parse_json_pseudo(path) if path.suffix == ".json" else parse_csv_pseudo(path)
            for key in counts:
                counts[key] += parsed.get(key, 0)

        status = "not_run"
        note = ""
        if tid in FROZEN_TASKS:
            status = "hidden_safety_failed"
            note = FROZEN_TASKS[tid]
        elif paths:
            if counts["failed"] == 0 and counts["passed"] > 0:
                status = "stress_clean"
            elif counts["failed"] > 0 and counts["passed"] > 0:
                status = "stress_mixed"
            elif counts["failed"] > 0:
                status = "stress_failed"
            else:
                status = "insufficient"

        summary = (
            f"passed={counts['passed']} failed={counts['failed']} "
            f"skipped={counts['skipped']} total={counts['total']} "
            f"reports={len(paths)} counted={len(count_paths)}"
        )
        if note:
            summary = f"{summary}; {note}"
        summaries[tid] = {
            "status": status,
            "summary": summary,
            "reports": [path.name for path in paths],
            **counts,
        }
    return summaries


def select_pseudo_count_paths(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """Avoid double-counting detail files and their summary sidecars."""
    detail = [
        path
        for path in paths
        if "summary" not in path.stem and not path.name.endswith("_failures.txt")
    ]
    if detail:
        return detail
    return [path for path in paths if path.suffix in {".csv", ".json"}]


def build_public_tail_tags(
    task_id: int,
    score_row: dict[str, str],
    source: dict[str, object],
    cost_monsters: set[int],
    low_score_tasks: set[int],
) -> list[str]:
    tags = []
    if task_id in PUBLIC_BOTTOM15:
        tags.append("public_bottom15")
    if task_id in PUBLIC_BIG_GAP:
        tags.append("public_big_gap")
    if task_id in PUBLIC_OVERFIT_RISK:
        tags.append("public_overfit_risk")
    if task_id in PUBLIC_SLOW_TASKS:
        tags.append("public_slowest")
    if task_id in cost_monsters:
        tags.append("cost_monster_top30")
    if task_id in low_score_tasks:
        tags.append("low_score_bottom30")
    if (to_int(score_row.get("local_pass"), 0) or 0) < (to_int(score_row.get("local_total"), 0) or 0):
        tags.append("local_visible_not_full_pass")
    matches = set(source.get("source_matches", []))
    if matches & {"konbu17_v36", "afr1ste_6335", "octaviograu_6154"}:
        tags.append("matches_cherrypick_unsafe_public")
    if source.get("manifest_changed"):
        tags.append("v4_plus10_changed")
    return tags


def priority_score(
    task_id: int,
    score_row: dict[str, str],
    public_tags: list[str],
    pseudo_status: str,
    has_task_artifacts: bool,
) -> int:
    score = 0
    tag_set = set(public_tags)
    score += 40 if "public_big_gap" in tag_set else 0
    score += 30 if "public_bottom15" in tag_set else 0
    score += 25 if "cost_monster_top30" in tag_set else 0
    score += 15 if "low_score_bottom30" in tag_set else 0
    score += 15 if "public_overfit_risk" in tag_set else 0
    score += 10 if "matches_cherrypick_unsafe_public" in tag_set else 0
    score += 10 if pseudo_status in {"stress_mixed", "stress_failed", "hidden_safety_failed"} else 0
    score += 5 if pseudo_status == "stress_clean" else 0
    score += 5 if has_task_artifacts else 0
    cost = to_int(score_row.get("cost"), 0) or 0
    if cost > 0:
        score += min(20, max(0, int((math.log10(cost) - 4.0) * 8)))
    if task_id in SPECIAL_ACTIONS:
        score += 5
    return score


def choose_next_action(
    task_id: int,
    family_primary: str,
    public_tags: list[str],
    pseudo_status: str,
    prototypes: list[str],
    builders: list[str],
) -> str:
    if task_id in SPECIAL_ACTIONS:
        return SPECIAL_ACTIONS[task_id]
    tag_set = set(public_tags)
    risky_public = "matches_cherrypick_unsafe_public" in tag_set
    high_priority = bool(tag_set & {"public_big_gap", "public_bottom15", "cost_monster_top30"})

    if risky_public and pseudo_status != "stress_clean":
        return "Require semantic proof and pseudo-hidden audit; no raw public-source swap."
    if pseudo_status in {"stress_mixed", "stress_failed"}:
        return "Revise semantic prototype; do not stage until pseudo-hidden is clean."
    if pseudo_status == "hidden_safety_failed":
        return "Research-only; do not stage or submit from current line."
    if prototypes and not builders:
        return "Compile a compact ONNX builder, then cost-audit against the anchor."
    if builders and pseudo_status == "stress_clean":
        return "Cost-audit builder and verify full visible plus stress suite before staging."
    if builders:
        return "Run or extend pseudo-hidden audit before using the builder."
    if high_priority:
        return "Open semantic-factory pass: inspect examples, write prototype, then stress-test."
    if family_primary in {"tiling", "crop", "color_remap", "color_fill", "reduce", "symmetry/flip/rotate"}:
        return "Route to family DSL/compiler search; record any exact rule."
    return "Defer; revisit through family-level compiler rewrites or new public evidence."


def build_dossiers(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, object]]:
    root = pathlib.Path(args.root)
    reports_dir = root / "reports"
    tools_dir = root / "tools"
    raw_dir = root / "data" / "neurogolf-2026" / "raw"

    anchor_scores = load_anchor_scores(root / args.anchor_score_csv)
    attempts = load_attempts(reports_dir)
    families = load_family_features(reports_dir, raw_dir)
    source_by_task, _ = load_source_provenance(reports_dir, root / args.current_manifest)
    pseudo = load_pseudo_hidden(reports_dir)

    prototypes = collect_task_files(tools_dir, "prototype_task*.py")
    builders = collect_task_files(tools_dir, "build_task*.py")
    task_reports = collect_task_files(reports_dir, "task*")

    costs = sorted(
        (
            (tid, to_int(row.get("cost"), 0) or 0)
            for tid, row in anchor_scores.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    cost_monsters = {tid for tid, _ in costs[: args.cost_monster_count]}
    low_scores = sorted(
        (
            (tid, to_float(row.get("score"), 0.0) or 0.0)
            for tid, row in anchor_scores.items()
        ),
        key=lambda item: item[1],
    )
    low_score_tasks = {tid for tid, _ in low_scores[: args.low_score_count]}

    csv_rows: list[dict[str, str]] = []
    json_tasks: list[dict[str, object]] = []
    for tid in range(1, 401):
        score_row = anchor_scores.get(tid, {})
        source = source_by_task.get(tid, {})
        family = families.get(tid, {})
        pseudo_row = pseudo.get(tid, {"status": "not_run", "summary": ""})
        public_tags = build_public_tail_tags(tid, score_row, source, cost_monsters, low_score_tasks)
        attempt_summary = summarize_attempts(attempts.get(tid, []))
        task_prototypes = prototypes.get(tid, [])
        task_builders = builders.get(tid, [])
        reports = task_reports.get(tid, [])
        task_priority = priority_score(
            tid,
            score_row,
            public_tags,
            str(pseudo_row.get("status", "")),
            bool(task_prototypes or task_builders or reports),
        )
        next_action = choose_next_action(
            tid,
            str(family.get("primary", "")),
            public_tags,
            str(pseudo_row.get("status", "")),
            task_prototypes,
            task_builders,
        )

        csv_row = {
            "task_id": task_label(tid),
            "anchor_score": fmt_float(score_row.get("score")),
            "anchor_cost": fmt_int(score_row.get("cost")),
            "anchor_memory": fmt_int(score_row.get("memory")),
            "anchor_params": fmt_int(score_row.get("params")),
            "anchor_nodes": fmt_int(score_row.get("nodes")),
            "anchor_file_size": fmt_int(score_row.get("file_size")),
            "local_pass": fmt_int(score_row.get("local_pass")),
            "local_total": fmt_int(score_row.get("local_total")),
            "source_label": str(source.get("source_label", "")),
            "source_risk": str(source.get("source_risk", "")),
            "source_matches": ";".join(source.get("source_matches", [])),
            "known_match_summary": str(source.get("known_match_summary", "")),
            "best_seen_score": fmt_float(attempt_summary.get("best_seen_score")),
            "best_seen_cost": fmt_int(attempt_summary.get("best_seen_cost")),
            "best_seen_attempt": str(attempt_summary.get("best_seen_attempt", "")),
            "attempt_count": str(attempt_summary.get("attempt_count", 0)),
            "family_primary": str(family.get("primary", "")),
            "family_tags": ";".join(family.get("tags", [])),
            "family_features": str(family.get("family_features", "")),
            "shape_profile": str(family.get("shape_profile", "")),
            "public_tail_tags": ";".join(public_tags),
            "pseudo_hidden_status": str(pseudo_row.get("status", "not_run")),
            "pseudo_hidden_summary": str(pseudo_row.get("summary", "")),
            "known_prototypes": ";".join(task_prototypes),
            "known_builders": ";".join(task_builders),
            "known_reports": ";".join(reports),
            "next_allowed_action": next_action,
            "priority_score": str(task_priority),
        }
        csv_rows.append(csv_row)
        json_tasks.append(
            {
                **csv_row,
                "task_num": tid,
                "source": source,
                "family": family,
                "pseudo_hidden": pseudo_row,
                "attempt_summary": attempt_summary,
                "top_attempts": attempt_summary.get("top_attempts", []),
                "public_tail_tags": public_tags,
                "known_prototypes": task_prototypes,
                "known_builders": task_builders,
                "known_reports": reports,
                "priority_score": task_priority,
            }
        )

    metadata = {
        "generated_at_local": datetime.now().replace(microsecond=0).isoformat(),
        "anchor_name": args.anchor_name,
        "anchor_lb": args.anchor_lb,
        "anchor_score_csv": args.anchor_score_csv,
        "current_manifest": args.current_manifest,
        "scope": "task dossier / attempt registry only; no anchor, ledger, submission, or ONNX writes",
        "inputs": [
            args.anchor_score_csv,
            args.current_manifest,
            "reports/task_provenance.csv",
            "reports/source_manifests/all_sources.csv",
            "reports/uncovered_task_features.csv",
            "reports/task_pattern_analysis.md",
            "reports/task*_pseudo_hidden*",
            "reports/*score*.csv",
            "tools/prototype_task*.py",
            "tools/build_task*.py",
        ],
        "schema": CSV_FIELDS,
        "public_tail_sources": {
            "public_bottom15": sorted(task_label(tid) for tid in PUBLIC_BOTTOM15),
            "public_big_gap": sorted(task_label(tid) for tid in PUBLIC_BIG_GAP),
            "public_overfit_risk": sorted(task_label(tid) for tid in PUBLIC_OVERFIT_RISK),
            "public_slowest": sorted(task_label(tid) for tid in PUBLIC_SLOW_TASKS),
        },
    }
    registry = {
        "metadata": metadata,
        "summary": summarize_registry(csv_rows, json_tasks),
        "tasks": json_tasks,
    }
    return csv_rows, registry


def summarize_registry(csv_rows: list[dict[str, str]], json_tasks: list[dict[str, object]]) -> dict[str, object]:
    status_counts = Counter(row["pseudo_hidden_status"] for row in csv_rows)
    action_counts = Counter(row["next_allowed_action"] for row in csv_rows)
    family_counts = Counter(row["family_primary"] for row in csv_rows)
    tag_counts: Counter[str] = Counter()
    for row in csv_rows:
        for tag in row["public_tail_tags"].split(";"):
            if tag:
                tag_counts[tag] += 1
    priority = sorted(
        json_tasks,
        key=lambda row: (int(row["priority_score"]), to_float(row["anchor_score"], 0.0) or 0.0),
        reverse=True,
    )
    return {
        "task_count": len(csv_rows),
        "pseudo_hidden_status_counts": dict(sorted(status_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "public_tail_tag_counts": dict(sorted(tag_counts.items())),
        "next_action_counts": dict(action_counts.most_common()),
        "top_priority_tasks": [
            {
                "task_id": row["task_id"],
                "priority_score": row["priority_score"],
                "anchor_score": row["anchor_score"],
                "anchor_cost": row["anchor_cost"],
                "family_primary": row["family_primary"],
                "public_tail_tags": row["public_tail_tags"],
                "pseudo_hidden_status": row["pseudo_hidden_status"],
                "next_allowed_action": row["next_allowed_action"],
            }
            for row in priority[:40]
        ],
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: pathlib.Path, registry: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")


def write_markdown(path: pathlib.Path, registry: dict[str, object]) -> None:
    metadata = registry["metadata"]
    summary = registry["summary"]
    top = summary["top_priority_tasks"][:25]
    lines = [
        f"# Task Dossiers - {metadata['anchor_name']} - {metadata['anchor_lb']}",
        "",
        "Generated task dossier / attempt registry artifacts for agent consumption.",
        "No anchor, ledger, submission, or ONNX files are modified by this tool.",
        "",
        "## Artifacts",
        "",
        "- CSV: `reports/task_dossiers_20260606.csv`",
        "- JSON registry: `reports/task_dossiers_20260606.json`",
        "- Markdown summary: `reports/task_dossiers_20260606.md`",
        "",
        "## Summary",
        "",
        f"- Tasks: {summary['task_count']}",
        f"- Pseudo-hidden statuses: {json.dumps(summary['pseudo_hidden_status_counts'], sort_keys=True)}",
        f"- Public-tail tag counts: {json.dumps(summary['public_tail_tag_counts'], sort_keys=True)}",
        f"- Family counts: {json.dumps(summary['family_counts'], sort_keys=True)}",
        "",
        "## Priority Queue",
        "",
        "| task | prio | score | cost | family | tags | pseudo | next allowed action |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in top:
        row = {**row, "public_tail_tags": ";".join(row.get("public_tail_tags", []))}
        lines.append(
            "| {task_id} | {priority_score} | {anchor_score} | {anchor_cost} | {family_primary} | "
            "{public_tail_tags} | {pseudo_hidden_status} | {next_allowed_action} |".format(
                **{key: md_escape(str(value)) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Schema",
            "",
            "The CSV has one row per task. The JSON keeps the same fields plus nested",
            "`source`, `family`, `pseudo_hidden`, `attempt_summary`, and `top_attempts`",
            "objects for agents that need fuller context.",
            "",
            "Key fields:",
            "",
            "- `public_tail_tags`: public hard-tail, overfit-risk, slow-task, cost-monster, and source-risk labels.",
            "- `pseudo_hidden_status`: `not_run`, `stress_clean`, `stress_mixed`, `stress_failed`, or `hidden_safety_failed`.",
            "- `next_allowed_action`: conservative next engineering move, not an instruction to submit.",
            "- `best_seen_*` and `top_attempts` in JSON: compact attempt registry mined from score CSVs.",
            "",
            "## Inputs",
            "",
        ]
    )
    for item in metadata["inputs"]:
        lines.append(f"- `{item}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--date", default="20260606")
    parser.add_argument("--anchor-name", default="v4_plus10_compiler2")
    parser.add_argument("--anchor-lb", default="6260.16")
    parser.add_argument(
        "--anchor-score-csv",
        default="reports/candidate_v4_plus10_compiler2_score_n3.csv",
    )
    parser.add_argument(
        "--current-manifest",
        default="reports/candidate_v4_plus10_compiler2_manifest.csv",
    )
    parser.add_argument("--cost-monster-count", type=int, default=30)
    parser.add_argument("--low-score-count", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = pathlib.Path(args.root)
    prefix = root / "reports" / f"task_dossiers_{args.date}"
    csv_rows, registry = build_dossiers(args)
    write_csv(prefix.with_suffix(".csv"), csv_rows)
    write_json(prefix.with_suffix(".json"), registry)
    write_markdown(prefix.with_suffix(".md"), registry)
    print(f"wrote {prefix.with_suffix('.csv')}")
    print(f"wrote {prefix.with_suffix('.json')}")
    print(f"wrote {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
