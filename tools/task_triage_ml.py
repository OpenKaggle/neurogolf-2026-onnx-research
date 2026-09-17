#!/usr/bin/env python3
"""Rank NeuroGolf task work from the local dossier registry.

Inputs are intentionally local and auditable.  The canonical task registry is
Socrates' `reports/task_dossiers_20260606.{csv,json,md}`; this script adds a
transparent value/risk model on top of the dossier fields and emits ranked
queues plus family/direction summaries.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

RUN_DATE = "20260606"
ANCHOR_NAME = "v4_plus10_compiler2"
ANCHOR_LB = 6260.16
DOSSIER_CSV = REPORTS / f"task_dossiers_{RUN_DATE}.csv"
DOSSIER_JSON = REPORTS / f"task_dossiers_{RUN_DATE}.json"
PREV_SCORE_FILE = REPORTS / "candidate_v4_plus9_compiler_score_n3.csv"
BASE_6122_SCORE_FILE = REPORTS / "current_best_6122_score_n3.csv"
GRAPH_AUDIT_FILE = REPORTS / "ours_graph_structure_audit.csv"
COMPILER_GOLF_FILE = REPORTS / "compiler_golf_probe_v2_candidates.csv"
BOOL_MEMORY_FILE = REPORTS / "bool_memory_priority.csv"

PARENT_PRIORITY = [
    "task018",
    "task285",
    "task255",
    "task133",
    "task366",
    "task233",
    "task025",
    "task173",
    "task243",
    "task158",
    "task054",
    "task286",
    "task096",
]

PUBLIC_GAP = {
    "task255": 4.57,
    "task233": 3.98,
    "task366": 3.77,
    "task018": 3.52,
    "task285": 3.02,
}


FAMILY_DESCRIPTIONS = {
    "symmetry/flip/rotate": "Shape-preserving symmetry and geometric completion tasks; high reuse if a robust transform matcher lands.",
    "reduce": "Output-smaller crop/extract/object selection tasks; strong value for reusable component and bbox primitives.",
    "other": "Mixed hard tasks needing semantic factory passes before compiler work.",
    "color_remap": "Palette and recoloring tasks; best attacked by color-invariant pseudo-hidden audits plus compact maps.",
    "crop": "Crop/localization tasks; graphizable when bbox detection is reliable.",
    "color_fill": "Fill/paint tasks; useful for mask/flood-fill primitive reuse.",
    "expand": "Output-larger expansion/tiling variants.",
    "tiling": "Tile/repeat patterns; usually DSL/compiler-search friendly.",
    "trim_border": "Border removal/trim operations.",
    "scalar_output": "Scalar/metadata outputs; prone to overfit unless semantics are explicit.",
}

DIRECTION_DESCRIPTIONS = {
    "semantic_rebuild": "Inspect examples, build or repair the Python rule, then stress-test before ONNX.",
    "compact_compiler": "Semantic story is plausible; next value is a cheaper exact ONNX implementation.",
    "pseudo_hidden_first": "Build or extend pseudo-hidden/metamorphic tests before trusting any replacement.",
    "audit_before_swap": "Raw source or builder exists, but hidden-safety proof is missing.",
    "compiler_golf": "Exact-equivalence backend rewrite lane.",
    "family_dsl": "Route through reusable DSL/compiler family search.",
    "frozen": "No submission path without a new hidden-safety theory.",
    "defer": "Low current EV; revisit after stronger evidence or reusable primitive work.",
}


def task_key(value: Any) -> str:
    match = re.search(r"(\d{1,3})", str(value))
    if not match:
        raise ValueError(f"Cannot parse task id from {value!r}")
    return f"task{int(match.group(1)):03d}"


def task_num(value: Any) -> int:
    return int(task_key(value)[4:])


def fnum(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    if isinstance(value, (list, tuple, set)):
        return ";".join(map(str, value))
    return value


def score_from_cost(cost: float) -> float:
    return max(0.0, 25.0 - math.log(max(0.0, cost) + 1.0))


def gain_to_cost(current_score: float, target_cost: float) -> float:
    return max(0.0, score_from_cost(target_cost) - current_score)


def parse_tags(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(";") if part.strip()]


def parse_pseudo_summary(text: str) -> dict[str, float]:
    out = {"passed": 0.0, "failed": 0.0, "skipped": 0.0, "total": 0.0, "reports": 0.0, "counted": 0.0}
    for key, value in re.findall(r"([a-z_]+)=([0-9.]+)", str(text or "")):
        out[key] = fnum(value)
    denom = out["passed"] + out["failed"]
    out["pass_rate"] = out["passed"] / denom if denom else 0.0
    return out


def load_score_map(path: Path) -> dict[str, float]:
    out = {}
    for row in read_csv(path):
        out[task_key(row.get("task_id", ""))] = fnum(row.get("score"))
    return out


def load_graph_audit() -> dict[str, dict[str, Any]]:
    out = {}
    for row in read_csv(GRAPH_AUDIT_FILE):
        out[task_key(row.get("task_id", ""))] = {
            "graph_risk_score": fnum(row.get("risk_score")),
            "graph_risk_flags": row.get("risk_flags", ""),
            "graph_init_ratio": fnum(row.get("init_byte_ratio")),
            "graph_biggest_init_bytes": fnum(row.get("biggest_init_bytes")),
            "graph_has_lookup_ops": inum(row.get("has_lookup_ops")),
            "graph_has_fp16": inum(row.get("has_fp16")),
        }
    return out


def load_probe_signals() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"compiler_golf_gain": 0.0, "compiler_golf_signal": ""}
    )
    for path in (COMPILER_GOLF_FILE, BOOL_MEMORY_FILE):
        for row in read_csv(path):
            if str(row.get("kept", "")).lower() not in {"true", "1"}:
                continue
            key = task_key(row.get("task_id", ""))
            gain = fnum(row.get("score_gain"))
            if gain > out[key]["compiler_golf_gain"]:
                out[key]["compiler_golf_gain"] = gain
                out[key]["compiler_golf_signal"] = f"{path.name}: {row.get('rewrite', '')}"
    return dict(out)


def load_dossiers() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(DOSSIER_CSV)
    nested: dict[str, Any] = {}
    if DOSSIER_JSON.exists():
        nested = json.loads(DOSSIER_JSON.read_text())
    nested_tasks = {}
    for item in nested.get("tasks", []):
        nested_tasks[item["task_id"]] = item

    graph = load_graph_audit()
    probes = load_probe_signals()
    prev_scores = load_score_map(PREV_SCORE_FILE)
    base_scores = load_score_map(BASE_6122_SCORE_FILE)

    out = []
    costs = np.array([fnum(row.get("anchor_cost")) for row in rows], dtype=float)
    cost_rank = {
        task_key(row["task_id"]): rank + 1
        for rank, row in enumerate(sorted(rows, key=lambda item: fnum(item.get("anchor_cost")), reverse=True))
    }
    low_score_rank = {
        task_key(row["task_id"]): rank + 1
        for rank, row in enumerate(sorted(rows, key=lambda item: fnum(item.get("anchor_score"))))
    }

    for raw in rows:
        key = task_key(raw["task_id"])
        nested_row = nested_tasks.get(key, {})
        pseudo = parse_pseudo_summary(raw.get("pseudo_hidden_summary", ""))
        tags = parse_tags(raw.get("public_tail_tags", ""))
        family_tags = parse_tags(raw.get("family_tags", ""))
        source_matches = parse_tags(raw.get("source_matches", ""))
        known_prototypes = parse_tags(raw.get("known_prototypes", ""))
        known_builders = parse_tags(raw.get("known_builders", ""))
        known_reports = parse_tags(raw.get("known_reports", ""))

        row = {
            "task_id": key,
            "task_num": task_num(key),
            "anchor_score": fnum(raw.get("anchor_score")),
            "anchor_cost": fnum(raw.get("anchor_cost")),
            "anchor_memory": fnum(raw.get("anchor_memory")),
            "anchor_params": fnum(raw.get("anchor_params")),
            "anchor_nodes": fnum(raw.get("anchor_nodes")),
            "anchor_file_size": fnum(raw.get("anchor_file_size")),
            "local_pass": inum(raw.get("local_pass")),
            "local_total": inum(raw.get("local_total")),
            "source_label": raw.get("source_label", ""),
            "source_risk": raw.get("source_risk", ""),
            "source_matches": source_matches,
            "source_match_count": len(source_matches),
            "known_match_summary": raw.get("known_match_summary", ""),
            "best_seen_score": fnum(raw.get("best_seen_score")),
            "best_seen_cost": fnum(raw.get("best_seen_cost")),
            "best_seen_attempt": raw.get("best_seen_attempt", ""),
            "attempt_count": inum(raw.get("attempt_count")),
            "family_primary": raw.get("family_primary", "other"),
            "family_tags": family_tags,
            "family_features": raw.get("family_features", ""),
            "shape_profile": raw.get("shape_profile", ""),
            "public_tail_tags": tags,
            "pseudo_hidden_status": raw.get("pseudo_hidden_status", "not_run"),
            "pseudo_passed": pseudo["passed"],
            "pseudo_failed": pseudo["failed"],
            "pseudo_skipped": pseudo["skipped"],
            "pseudo_total": pseudo["total"],
            "pseudo_pass_rate": pseudo["pass_rate"],
            "known_prototypes": known_prototypes,
            "known_builders": known_builders,
            "known_reports": known_reports,
            "next_allowed_action": raw.get("next_allowed_action", ""),
            "dossier_priority_score": fnum(raw.get("priority_score")),
            "cost_rank": cost_rank[key],
            "low_score_rank": low_score_rank[key],
            "cost_percentile": float(np.mean(costs <= fnum(raw.get("anchor_cost")))) if len(costs) else 0.0,
            "prev_anchor_score": prev_scores.get(key, 0.0),
            "delta_vs_prev_anchor": fnum(raw.get("anchor_score")) - prev_scores.get(key, 0.0),
            "delta_vs_6122_anchor": fnum(raw.get("anchor_score")) - base_scores.get(key, 0.0),
            "top_attempts": nested_row.get("top_attempts", []),
        }
        row.update(graph.get(key, {}))
        row.update(probes.get(key, {"compiler_golf_gain": 0.0, "compiler_golf_signal": ""}))
        out.append(row)

    metadata = {
        "dossier_csv": str(DOSSIER_CSV.relative_to(ROOT)),
        "dossier_json": str(DOSSIER_JSON.relative_to(ROOT)),
        "dossier_metadata": nested.get("metadata", {}),
        "dossier_summary": nested.get("summary", {}),
    }
    return out, metadata


def infer_direction(row: dict[str, Any]) -> str:
    action = str(row["next_allowed_action"]).lower()
    key = row["task_id"]
    if "frozen" in action or row["pseudo_hidden_status"] == "hidden_safety_failed":
        return "frozen"
    if key == "task018" or "do not cost-only" in action:
        return "pseudo_hidden_first"
    if "compile a compact" in action or "cost-audit builder" in action:
        return "compact_compiler"
    if "retire brittle" in action or "revise semantic" in action or "open semantic-factory" in action:
        return "semantic_rebuild"
    if "run or extend pseudo-hidden" in action:
        return "pseudo_hidden_first"
    if "require semantic proof" in action:
        return "audit_before_swap"
    if row.get("compiler_golf_gain", 0.0) > 0.02 or "v4_plus10_changed" in row["public_tail_tags"]:
        return "compiler_golf"
    if "route to family dsl" in action:
        return "family_dsl"
    return "defer"


def hidden_risk(row: dict[str, Any]) -> float:
    status = row["pseudo_hidden_status"]
    risk = {
        "stress_clean": 0.22,
        "stress_mixed": 0.68,
        "hidden_safety_failed": 0.97,
        "stress_failed": 0.90,
        "not_run": 0.52,
    }.get(status, 0.55)

    tags = set(row["public_tail_tags"])
    source_risk = row["source_risk"]
    if "public_overfit_risk" in tags:
        risk += 0.10
    if "matches_cherrypick_unsafe_public" in tags:
        risk += 0.07
    if "local_visible_not_full_pass" in tags:
        risk += 0.15
    if row["local_total"] and row["local_pass"] < row["local_total"]:
        risk += 0.20
    if source_risk == "public-anchor-candidate":
        risk += 0.04
    if source_risk == "untested-public":
        risk += 0.08
    if row.get("graph_risk_score", 0.0) >= 2:
        risk += 0.05 * row["graph_risk_score"]
    if row.get("graph_has_lookup_ops", 0):
        risk += 0.06
    if row["pseudo_failed"] > 0 and row["pseudo_passed"] > 0:
        fail_rate = row["pseudo_failed"] / (row["pseudo_passed"] + row["pseudo_failed"])
        risk += min(0.18, 0.35 * fail_rate)
    if row["pseudo_hidden_status"] == "stress_clean" and row["pseudo_passed"] >= 500:
        risk -= 0.08
    if row["source_match_count"] >= 6 and "public_overfit_risk" not in tags:
        risk -= 0.04
    return clamp(risk, 0.03, 0.98)


def graphizability(row: dict[str, Any]) -> float:
    family = row["family_primary"]
    base = {
        "color_remap": 0.72,
        "symmetry/flip/rotate": 0.66,
        "tiling": 0.68,
        "crop": 0.58,
        "trim_border": 0.62,
        "reduce": 0.48,
        "color_fill": 0.44,
        "expand": 0.50,
        "scalar_output": 0.40,
        "other": 0.36,
    }.get(family, 0.38)
    if row["known_builders"]:
        base += 0.12
    if row["known_prototypes"]:
        base += 0.08
    if row["known_reports"]:
        base += 0.04
    if row.get("anchor_nodes", 0.0) > 500:
        base -= 0.08
    if row.get("graph_risk_score", 0.0) >= 2:
        base -= 0.06
    if row["pseudo_hidden_status"] == "stress_clean":
        base += 0.08
    if row["pseudo_hidden_status"] in {"stress_mixed", "hidden_safety_failed"}:
        base -= 0.08
    return clamp(base, 0.05, 0.92)


def family_reuse(row: dict[str, Any], family_counts: Counter[str]) -> float:
    family = row["family_primary"]
    count = family_counts[family]
    base = clamp(math.log1p(count) / math.log1p(105), 0.0, 1.0)
    if family in {"symmetry/flip/rotate", "reduce", "color_remap", "crop", "color_fill"}:
        base += 0.08
    if row["known_prototypes"] or row["known_builders"]:
        base += 0.05
    return clamp(base)


def gross_upside(row: dict[str, Any]) -> float:
    key = row["task_id"]
    current = row["anchor_score"]
    cost = row["anchor_cost"]
    public_gap = PUBLIC_GAP.get(key, 0.0)
    best_seen_delta = max(0.0, row["best_seen_score"] - current)
    compiler_gain = fnum(row.get("compiler_golf_gain"))

    gain_10k = gain_to_cost(current, 10_000)
    gain_50k = gain_to_cost(current, 50_000)
    gain_100k = gain_to_cost(current, 100_000)

    if key == "task018":
        # The cost-0 probe is known bad; cap to public gap until a real solver exists.
        return public_gap
    if public_gap:
        return max(public_gap, min(gain_10k, public_gap + 0.60), best_seen_delta)
    if best_seen_delta >= 0.25 and "probe" not in row["best_seen_attempt"]:
        return max(best_seen_delta, min(gain_10k, best_seen_delta + 0.60))
    if cost >= 500_000:
        return gain_50k
    if cost >= 100_000:
        return max(gain_100k, 0.60 * gain_10k)
    if compiler_gain > 0:
        return compiler_gain
    if cost >= 30_000:
        return 0.35 * gain_10k
    return 0.0


def success_probability(row: dict[str, Any], risk: float, graph: float, reuse: float) -> float:
    status = row["pseudo_hidden_status"]
    direction = row["direction"]
    p = {
        "stress_clean": 0.50,
        "stress_mixed": 0.22,
        "hidden_safety_failed": 0.03,
        "not_run": 0.24,
    }.get(status, 0.22)
    if direction == "compact_compiler":
        p += 0.08
    if direction == "semantic_rebuild":
        p += 0.04
    if direction == "audit_before_swap":
        p -= 0.02
    if direction == "pseudo_hidden_first":
        p -= 0.03
    if row["known_prototypes"]:
        p += 0.05
    if row["known_builders"]:
        p += 0.04
    if "public_big_gap" in row["public_tail_tags"]:
        p += 0.03
    if row["best_seen_score"] > row["anchor_score"] + 0.25 and "probe" not in row["best_seen_attempt"]:
        p += 0.04
    p += 0.08 * (graph - 0.5)
    p += 0.05 * (reuse - 0.5)
    p -= 0.16 * max(0.0, risk - 0.45)
    if row["local_total"] and row["local_pass"] < row["local_total"]:
        p -= 0.08
    return clamp(p, 0.02, 0.78)


def effort_units(row: dict[str, Any], direction: str, graph: float) -> float:
    cost = row["anchor_cost"]
    effort = 0.9 + 0.13 * math.log10(max(1.0, cost + 1.0))
    if direction == "compact_compiler":
        effort += 0.10
    if direction == "semantic_rebuild":
        effort += 0.20
    if direction == "pseudo_hidden_first":
        effort += 0.12
    if direction == "compiler_golf":
        effort -= 0.20
    if row.get("anchor_nodes", 0.0) > 500:
        effort += 0.18
    if row["known_prototypes"] or row["known_builders"]:
        effort -= 0.08
    effort += 0.10 * (1.0 - graph)
    return max(0.65, effort)


def rationale(row: dict[str, Any]) -> str:
    bits = []
    if row["task_id"] in PARENT_PRIORITY:
        bits.append("parent-priority")
    if "public_big_gap" in row["public_tail_tags"]:
        gap = PUBLIC_GAP.get(row["task_id"])
        bits.append(f"public gap +{gap:.2f}" if gap else "public big-gap set")
    elif "public_bottom15" in row["public_tail_tags"]:
        bits.append("public bottom15")
    if row["cost_rank"] <= 30:
        bits.append(f"cost rank {row['cost_rank']}")
    if row["pseudo_hidden_status"] != "not_run":
        bits.append(row["pseudo_hidden_status"].replace("_", " "))
    if row["best_seen_score"] > row["anchor_score"] + 0.05:
        bits.append(f"best-seen +{row['best_seen_score'] - row['anchor_score']:.2f}")
    if row["known_prototypes"]:
        bits.append("prototype")
    if row["known_builders"]:
        bits.append("builder")
    if row.get("compiler_golf_gain", 0.0) > 0:
        bits.append(f"compiler +{row['compiler_golf_gain']:.2f}")
    if row.get("graph_risk_score", 0.0) >= 2:
        bits.append(f"graph risk {row['graph_risk_score']:.0f}")
    bits.append(row["next_allowed_action"])
    return " | ".join(bits)


def add_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family_counts = Counter(row["family_primary"] for row in rows)
    parent_index = {task: idx + 1 for idx, task in enumerate(PARENT_PRIORITY)}
    priority_values = np.array([row["dossier_priority_score"] for row in rows], dtype=float)
    max_priority = max(float(priority_values.max()), 1.0)

    for row in rows:
        row["direction"] = infer_direction(row)
        row["hidden_risk"] = hidden_risk(row)
        row["graphizability"] = graphizability(row)
        row["family_reuse"] = family_reuse(row, family_counts)
        row["gross_upside_lb"] = gross_upside(row)
        row["success_probability"] = success_probability(
            row, row["hidden_risk"], row["graphizability"], row["family_reuse"]
        )
        row["effort_units"] = effort_units(row, row["direction"], row["graphizability"])
        row["expected_lb"] = row["gross_upside_lb"] * row["success_probability"]
        dossier_norm = row["dossier_priority_score"] / max_priority
        row["model_score"] = (
            0.55 * row["expected_lb"] / max(0.75, row["effort_units"])
            + 0.18 * row["family_reuse"]
            + 0.17 * row["graphizability"]
            + 0.18 * dossier_norm
            + 0.05 * row.get("compiler_golf_gain", 0.0)
            - 0.28 * row["hidden_risk"]
        )
        if row["task_id"] in parent_index:
            row["parent_priority_rank"] = parent_index[row["task_id"]]
            row["queue_score"] = 10_000.0 - parent_index[row["task_id"]]
        else:
            row["parent_priority_rank"] = 0
            row["queue_score"] = row["model_score"]
        row["rationale"] = rationale(row)
        row["next_gate"] = row["next_allowed_action"]
    return rows


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            row["queue_score"],
            row["model_score"],
            row["dossier_priority_score"],
            row["gross_upside_lb"],
        ),
        reverse=True,
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def aggregate_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        members[row["family_primary"]].append(row)

    out = []
    for family, items in members.items():
        ranked = sorted(items, key=lambda row: row["model_score"], reverse=True)
        top = ranked[:10]
        clean = sum(1 for row in items if row["pseudo_hidden_status"] == "stress_clean")
        mixed = sum(1 for row in items if row["pseudo_hidden_status"] == "stress_mixed")
        out.append(
            {
                "family": family,
                "description": FAMILY_DESCRIPTIONS.get(family, ""),
                "task_count": len(items),
                "top_tasks": ",".join(row["task_id"] for row in top[:10]),
                "top10_expected_lb": sum(row["expected_lb"] for row in top),
                "avg_family_reuse_top10": statistics.mean(row["family_reuse"] for row in top),
                "avg_graphizability_top10": statistics.mean(row["graphizability"] for row in top),
                "avg_hidden_risk_top10": statistics.mean(row["hidden_risk"] for row in top),
                "pseudo_hidden_status": f"clean={clean};mixed={mixed}",
                "recommended_gate": family_gate(family),
            }
        )
    return sorted(out, key=lambda row: (row["top10_expected_lb"], row["avg_family_reuse_top10"]), reverse=True)


def family_gate(family: str) -> str:
    if family == "symmetry/flip/rotate":
        return "Prioritize invariant stress suites and reusable D4/placement compiler pieces."
    if family == "reduce":
        return "Invest in bbox/component selection primitives; require pseudo-hidden before swaps."
    if family == "color_remap":
        return "Use color-bijection stress tests before trusting public or local-only gains."
    if family == "crop":
        return "Validate bbox detection under translations and distractors."
    if family == "color_fill":
        return "Develop mask/flood-fill contracts before graph staging."
    return "Use dossier next_allowed_action per task."


def aggregate_directions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        members[row["direction"]].append(row)

    out = []
    for direction, items in members.items():
        ranked = sorted(items, key=lambda row: row["model_score"], reverse=True)
        top = ranked[:12]
        out.append(
            {
                "direction": direction,
                "description": DIRECTION_DESCRIPTIONS.get(direction, ""),
                "task_count": len(items),
                "top_tasks": ",".join(row["task_id"] for row in top[:12]),
                "top12_expected_lb": sum(row["expected_lb"] for row in top),
                "avg_hidden_risk_top12": statistics.mean(row["hidden_risk"] for row in top),
                "avg_graphizability_top12": statistics.mean(row["graphizability"] for row in top),
                "next_gate": direction_gate(direction),
            }
        )
    return sorted(out, key=lambda row: row["top12_expected_lb"], reverse=True)


def direction_gate(direction: str) -> str:
    return {
        "semantic_rebuild": "Prototype and stress-test first; no ONNX until pseudo-hidden story improves.",
        "compact_compiler": "Set target cost, build exact graph, verify visible/stress equivalence.",
        "pseudo_hidden_first": "Write falsifiable contracts before considering any replacement.",
        "audit_before_swap": "No raw public-source swap; prove semantics and hidden safety.",
        "compiler_golf": "Exact-equivalence proof and structural audit before batching.",
        "family_dsl": "Mine family primitive, then isolated eval against anchor cost.",
        "frozen": "Leave untouched unless a new hidden-safety theory appears.",
        "defer": "Revisit through family-level compiler work or new evidence.",
    }.get(direction, "Use task-level gate.")


def missing_data(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> list[str]:
    pseudo_count = sum(1 for row in rows if row["pseudo_hidden_status"] != "not_run")
    clean_count = sum(1 for row in rows if row["pseudo_hidden_status"] == "stress_clean")
    graph_count = sum(1 for row in rows if row.get("graph_risk_score", 0.0) > 0 or row.get("graph_risk_flags"))
    proto_count = sum(1 for row in rows if row["known_prototypes"])
    return [
        f"Pseudo-hidden/metamorphic coverage is sparse: {pseudo_count}/400 tasks have any dossier status beyond not_run, and only {clean_count}/400 are stress_clean.",
        f"Graph audit rows are incomplete or anchor-mismatched for model training: {graph_count}/400 tasks have risk stats.",
        f"Only {proto_count}/400 tasks list known prototypes; semantic state should be structured for every high-priority task.",
        "Need branch-level training labels: submitted/not submitted, accepted/rejected, LB delta, local delta, and rollback reason.",
        "Need single-task or small-batch LB observations to map bundle outcomes back to task-level hidden risk.",
        "Need effort/cycle-time labels: human hours, generated graph size, failed attempts, and compile/test time.",
        "Need public competitor per-task scores beyond the bottom-15 list and five explicit gap estimates.",
        "Need machine-readable pseudo-hidden contract names and failure modes, not just aggregate pass/fail counts.",
        "Need graphizability labels from completed compilers: which family primitives compiled compactly and which blew up.",
        "Need source-trust labels for best_seen attempts; some best_seen deltas are useful evidence but not safe swap instructions.",
    ]


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    shown = rows[:limit] if limit is not None else rows
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in shown:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                if key in {"anchor_cost", "best_seen_cost"}:
                    cells.append(f"{value:,.0f}")
                elif key in {"hidden_risk", "graphizability", "family_reuse", "success_probability"}:
                    cells.append(f"{value:.2f}")
                else:
                    cells.append(f"{value:.3f}")
            else:
                text = str(value).replace("|", "/").replace("\n", " ")
                if len(text) > 110:
                    text = text[:107] + "..."
                cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(
    ranked: list[dict[str, Any]],
    next25_50: list[dict[str, Any]],
    families: list[dict[str, Any]],
    directions: list[dict[str, Any]],
    metadata: dict[str, Any],
    missing: list[str],
) -> str:
    top50 = ranked[:50]
    anchor_total = sum(row["anchor_score"] for row in ranked)
    high_risk_top50 = sum(1 for row in top50 if row["hidden_risk"] >= 0.70)
    clean_top50 = sum(1 for row in top50 if row["pseudo_hidden_status"] == "stress_clean")
    mixed_top50 = sum(1 for row in top50 if row["pseudo_hidden_status"] == "stress_mixed")

    lines = [
        "# Task Triage ML - 2026-06-06",
        "",
        "## Snapshot",
        "",
        f"- Anchor: `{ANCHOR_NAME}`, LB `{ANCHOR_LB:.2f}`, local dossier sum `{anchor_total:.4f}`.",
        f"- Registry: `{metadata['dossier_csv']}` and `{metadata['dossier_json']}`.",
        "- Parent priority queue is locked as ranks 1-13; ranks 14+ are the transparent model extension.",
        "- Model balances score upside, pseudo-hidden status, family reuse, graphizability, dossier priority, and hidden risk.",
        "- `sklearn`/`torch` were not used; available labels are too sparse for a model that would be more trustworthy than this auditable baseline.",
        f"- Top 50 has `{clean_top50}` stress-clean tasks, `{mixed_top50}` stress-mixed tasks, and `{high_risk_top50}` tasks with hidden risk >= 0.70.",
        "",
        "## Method",
        "",
        "Gross upside uses public gap estimates when available, otherwise cost-model savings from `score=max(0,25-ln(cost+1))` and safe best-seen deltas. Success probability is a calibrated heuristic from pseudo-hidden status, known prototypes/builders, graphizability, and risk. The queue score locks the parent 13-task order and then ranks the rest by expected value per effort with reuse and graphizability bonuses.",
        "",
        "## Top Families",
        "",
        md_table(
            families[:10],
            [
                ("family", "family"),
                ("tasks", "task_count"),
                ("top tasks", "top_tasks"),
                ("top10 EV", "top10_expected_lb"),
                ("reuse", "avg_family_reuse_top10"),
                ("graph", "avg_graphizability_top10"),
                ("risk", "avg_hidden_risk_top10"),
                ("gate", "recommended_gate"),
            ],
        ),
        "",
        "## Directions",
        "",
        md_table(
            directions,
            [
                ("direction", "direction"),
                ("tasks", "task_count"),
                ("top tasks", "top_tasks"),
                ("top12 EV", "top12_expected_lb"),
                ("risk", "avg_hidden_risk_top12"),
                ("graph", "avg_graphizability_top12"),
                ("gate", "next_gate"),
            ],
        ),
        "",
        "## Ranked Queue 1-50",
        "",
        md_table(
            top50,
            [
                ("rank", "rank"),
                ("task", "task_id"),
                ("dir", "direction"),
                ("family", "family_primary"),
                ("cost", "anchor_cost"),
                ("gross", "gross_upside_lb"),
                ("EV", "expected_lb"),
                ("risk", "hidden_risk"),
                ("graph", "graphizability"),
                ("reuse", "family_reuse"),
                ("pseudo", "pseudo_hidden_status"),
                ("rationale", "rationale"),
            ],
        ),
        "",
        "## Next 25-50 Focus",
        "",
        "These are the extension slots after the most urgent public-tail and parent-priority work. They skew toward graphizable, reusable families unless hidden-safety risk says audit first.",
        "",
        md_table(
            next25_50,
            [
                ("rank", "rank"),
                ("task", "task_id"),
                ("dir", "direction"),
                ("family", "family_primary"),
                ("cost", "anchor_cost"),
                ("best delta", "best_seen_delta"),
                ("EV", "expected_lb"),
                ("risk", "hidden_risk"),
                ("graph", "graphizability"),
                ("rationale", "rationale"),
            ],
        ),
        "",
        "## Missing Data For A Stronger Model",
        "",
    ]
    lines.extend(f"- {item}" for item in missing)
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `reports/task_triage_ml_features_{RUN_DATE}.csv`",
            f"- `reports/task_triage_ml_top50_{RUN_DATE}.csv`",
            f"- `reports/task_triage_ml_next25_50_{RUN_DATE}.csv`",
            f"- `reports/task_triage_ml_families_{RUN_DATE}.csv`",
            f"- `reports/task_triage_ml_directions_{RUN_DATE}.csv`",
            f"- `reports/task_triage_ml_summary_{RUN_DATE}.json`",
        ]
    )
    return "\n".join(lines) + "\n"


FEATURE_FIELDS = [
    "task_id",
    "rank",
    "parent_priority_rank",
    "direction",
    "family_primary",
    "family_tags_text",
    "public_tail_tags_text",
    "anchor_score",
    "anchor_cost",
    "anchor_memory",
    "anchor_params",
    "anchor_nodes",
    "anchor_file_size",
    "local_pass",
    "local_total",
    "cost_rank",
    "low_score_rank",
    "dossier_priority_score",
    "best_seen_score",
    "best_seen_cost",
    "best_seen_attempt",
    "best_seen_delta",
    "attempt_count",
    "gross_upside_lb",
    "success_probability",
    "hidden_risk",
    "graphizability",
    "family_reuse",
    "effort_units",
    "expected_lb",
    "model_score",
    "queue_score",
    "pseudo_hidden_status",
    "pseudo_passed",
    "pseudo_failed",
    "pseudo_skipped",
    "pseudo_pass_rate",
    "source_label",
    "source_risk",
    "source_match_count",
    "known_prototypes_text",
    "known_builders_text",
    "known_reports_text",
    "graph_risk_score",
    "graph_risk_flags",
    "graph_has_lookup_ops",
    "compiler_golf_gain",
    "compiler_golf_signal",
    "delta_vs_prev_anchor",
    "delta_vs_6122_anchor",
    "shape_profile",
    "family_features",
    "rationale",
    "next_gate",
]

TOP_FIELDS = [
    "rank",
    "task_id",
    "parent_priority_rank",
    "direction",
    "family_primary",
    "anchor_score",
    "anchor_cost",
    "cost_rank",
    "dossier_priority_score",
    "best_seen_delta",
    "gross_upside_lb",
    "expected_lb",
    "success_probability",
    "hidden_risk",
    "graphizability",
    "family_reuse",
    "pseudo_hidden_status",
    "source_risk",
    "rationale",
    "next_gate",
]

FAMILY_FIELDS = [
    "family",
    "description",
    "task_count",
    "top_tasks",
    "top10_expected_lb",
    "avg_family_reuse_top10",
    "avg_graphizability_top10",
    "avg_hidden_risk_top10",
    "pseudo_hidden_status",
    "recommended_gate",
]

DIRECTION_FIELDS = [
    "direction",
    "description",
    "task_count",
    "top_tasks",
    "top12_expected_lb",
    "avg_hidden_risk_top12",
    "avg_graphizability_top12",
    "next_gate",
]


def serialize_for_csv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["family_tags_text"] = ";".join(row["family_tags"])
        item["public_tail_tags_text"] = ";".join(row["public_tail_tags"])
        item["known_prototypes_text"] = ";".join(row["known_prototypes"])
        item["known_builders_text"] = ";".join(row["known_builders"])
        item["known_reports_text"] = ";".join(row["known_reports"])
        item["best_seen_delta"] = max(0.0, row["best_seen_score"] - row["anchor_score"])
        out.append(item)
    return out


def main() -> None:
    rows, metadata = load_dossiers()
    rows = add_scores(rows)
    ranked = rank_rows(rows)
    serial = serialize_for_csv(ranked)
    top50 = serial[:50]
    next25_50 = [row for row in serial if 25 <= row["rank"] <= 50]
    families = aggregate_families(ranked)
    directions = aggregate_directions(ranked)
    missing = missing_data(ranked, metadata)

    features_path = REPORTS / f"task_triage_ml_features_{RUN_DATE}.csv"
    top50_path = REPORTS / f"task_triage_ml_top50_{RUN_DATE}.csv"
    next25_path = REPORTS / f"task_triage_ml_next25_50_{RUN_DATE}.csv"
    families_path = REPORTS / f"task_triage_ml_families_{RUN_DATE}.csv"
    directions_path = REPORTS / f"task_triage_ml_directions_{RUN_DATE}.csv"
    summary_path = REPORTS / f"task_triage_ml_summary_{RUN_DATE}.json"
    report_path = REPORTS / f"task_triage_ml_{RUN_DATE}.md"

    write_csv(features_path, serial, FEATURE_FIELDS)
    write_csv(top50_path, top50, TOP_FIELDS)
    write_csv(next25_path, next25_50, TOP_FIELDS)
    write_csv(families_path, families, FAMILY_FIELDS)
    write_csv(directions_path, directions, DIRECTION_FIELDS)

    summary = {
        "run_date": RUN_DATE,
        "anchor": ANCHOR_NAME,
        "anchor_lb": ANCHOR_LB,
        "registry": {
            "dossier_csv": str(DOSSIER_CSV.relative_to(ROOT)),
            "dossier_json": str(DOSSIER_JSON.relative_to(ROOT)),
        },
        "parent_priority_locked": PARENT_PRIORITY,
        "top50": [row["task_id"] for row in top50],
        "next25_50": [row["task_id"] for row in next25_50],
        "top_families": families[:10],
        "directions": directions,
        "missing_data": missing,
        "model": {
            "type": "transparent heuristic baseline",
            "uses_numpy": True,
            "uses_sklearn": False,
            "uses_torch": False,
            "reason": "Task-level labels are sparse; dossier-first heuristic is more auditable than a fitted model.",
        },
        "outputs": [
            str(features_path.relative_to(ROOT)),
            str(top50_path.relative_to(ROOT)),
            str(next25_path.relative_to(ROOT)),
            str(families_path.relative_to(ROOT)),
            str(directions_path.relative_to(ROOT)),
            str(summary_path.relative_to(ROOT)),
            str(report_path.relative_to(ROOT)),
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_path.write_text(render_report(ranked, next25_50, families, directions, metadata, missing))

    # Parse after writing so syntax errors are caught without creating __pycache__.
    ast.parse(Path(__file__).read_text())
    for path in summary["outputs"]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
