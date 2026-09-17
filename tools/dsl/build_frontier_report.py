from __future__ import annotations

import csv
import json
import pathlib
import re
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

SEARCH_CSV = REPORTS / "dsl_v4_search_results.csv"
SWAP_CSV = REPORTS / "dsl_v4_swap_candidates.csv"
CURRENT_SCORE_CSV = REPORTS / "candidate_v4_plus10_compiler2_score_n3.csv"
TRIAGE_TOP50_CSV = REPORTS / "task_triage_ml_top50_20260606.csv"
TRIAGE_NEXT_CSV = REPORTS / "task_triage_ml_next25_50_20260606.csv"
OUT_CSV = REPORTS / "dsl_frontier_20260606.csv"
OUT_JSON = REPORTS / "dsl_frontier_20260606.json"
OUT_MD = REPORTS / "dsl_frontier_20260606.md"


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def task_num(value: str | int) -> int:
    text = str(value)
    if text.startswith("task"):
        return int(text[4:])
    return int(text)


def split_program(program: str) -> list[str]:
    return [part.strip() for part in program.split("|") if part.strip()]


def primitive_name(op: str) -> str:
    return op.split("(", 1)[0].strip()


def pass_fraction(text: str) -> tuple[int, int]:
    if not text or "/" not in text:
        return (0, 0)
    left, right = text.split("/", 1)
    return int(left), int(right)


def full_pass(row: dict[str, str]) -> bool:
    for key in ("dsl_train_pass", "dsl_test_pass", "dsl_arcgen_pass"):
        ok, total = pass_fraction(row.get(key, ""))
        if total == 0 or ok != total:
            return False
    return True


def load_current_scores() -> dict[int, dict[str, float | str]]:
    out: dict[int, dict[str, float | str]] = {}
    for row in read_rows(CURRENT_SCORE_CSV):
        tid = task_num(row["task_id"])
        out[tid] = {
            "score": float(row["score"]),
            "cost": float(row["cost"]),
            "path": row.get("path", ""),
        }
    return out


def load_triage() -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for path in (TRIAGE_TOP50_CSV, TRIAGE_NEXT_CSV):
        for row in read_rows(path):
            try:
                tid = task_num(row["task_id"])
            except Exception:
                continue
            out.setdefault(tid, row)
    return out


def classify_primitive(name: str) -> tuple[str, str]:
    cheap_exact = {"identity", "transpose", "swap_colors", "replace_color", "fill_bg"}
    shallow_geometry = {"flip_h", "flip_v", "rotate180_static", "rotate90_static", "crop"}
    high_cost_grid = {
        "rotate180",
        "rotate90",
        "tile_h",
        "tile_v",
        "shift_down",
        "shift_right",
        "dominant_color",
        "fill",
        "mask_foreground",
        "select_channel",
    }
    object_mask = {
        "bbox_crop",
        "largest_blob",
        "flood_fill",
        "thicken",
        "hollow_out",
        "trim_border",
        "remove_color",
    }
    if name in cheap_exact:
        return "mature_cheap", "Keep as trusted DSL leaf; mostly already saturated in v10."
    if name in shallow_geometry:
        return "partly_mature", "Task-specialized versions work, but current anchor often has tighter hand graphs."
    if name in high_cost_grid:
        return "needs_typed_rewrite", "Rewrite as BOOL/mask/small-shape typed primitive before broad search."
    if name in object_mask:
        return "needs_object_ir", "Do not brute-force as full FLOAT grid; build bounded object/mask IR first."
    return "unknown", "Review manually."


def recommendation_for(name: str, stats: dict[str, object]) -> str:
    if name in {"flip_h", "flip_v"}:
        return "Already integrated through v4_plus5; use as sanity template, not a new score lane."
    if name in {"transpose", "swap_colors", "replace_color", "fill_bg"}:
        return "Keep in depth-1/depth-2 search; no engineering needed unless a new exact hit beats v10."
    if name in {"rotate180_static", "rotate90_static"}:
        return "Add lower-level static Slice/Pad templates per task; current generic version still loses to v10 rotate handbuilds."
    if name in {"crop"}:
        return "Keep, but compare against v10 before staging; only tiny crops are near competitive."
    if name in {"tile_h", "tile_v", "shift_down", "shift_right"}:
        return "Rebuild as typed bbox/index primitive with BOOL masks and no full-grid mask multiply."
    if name in {"dominant_color", "fill", "mask_foreground", "select_channel"}:
        return "Split into BOOL mask plus paint/cover primitives; avoid materializing full FLOAT one-hot until final output."
    if name in {"bbox_crop", "largest_blob", "flood_fill", "thicken", "hollow_out", "trim_border"}:
        return "Promote to object/mask IR; depth search over current FLOAT builders is a local maximum."
    return "Review manually."


def main() -> None:
    current = load_current_scores()
    triage = load_triage()
    search_rows = read_rows(SEARCH_CSV)
    swap_rows = read_rows(SWAP_CSV)
    swap_tasks = {task_num(row["task_id"]) for row in swap_rows}

    primitive_stats: dict[str, dict[str, object]] = {}

    def stat(name: str) -> dict[str, object]:
        if name not in primitive_stats:
            category, note = classify_primitive(name)
            primitive_stats[name] = {
                "primitive": name,
                "category": category,
                "note": note,
                "search_hits": 0,
                "full_onnx_pass_hits": 0,
                "train_only_or_runtime_mismatch_hits": 0,
                "v4_swap_candidate_hits": 0,
                "v10_cheaper_hits": 0,
                "v10_equal_hits": 0,
                "v10_loser_hits": 0,
                "tasks": set(),
                "full_pass_tasks": set(),
                "swap_tasks": set(),
                "top50_tasks": set(),
                "next25_50_tasks": set(),
                "cost_ratios": [],
                "example_programs": [],
            }
        return primitive_stats[name]

    for row in search_rows:
        tid = task_num(row["task_id"])
        program = row.get("program_str", "")
        ops = split_program(program)
        names = [primitive_name(op) for op in ops]
        row_full = full_pass(row)
        dsl_cost = float(row["dsl_cost"]) if row.get("dsl_cost") not in {"", None} else None
        current_cost = current.get(tid, {}).get("cost")
        v10_cost = float(current_cost) if current_cost is not None else None
        for name in names:
            s = stat(name)
            s["search_hits"] = int(s["search_hits"]) + 1
            s["tasks"].add(tid)
            if row_full:
                s["full_onnx_pass_hits"] = int(s["full_onnx_pass_hits"]) + 1
                s["full_pass_tasks"].add(tid)
            else:
                s["train_only_or_runtime_mismatch_hits"] = int(s["train_only_or_runtime_mismatch_hits"]) + 1
            if tid in swap_tasks:
                s["v4_swap_candidate_hits"] = int(s["v4_swap_candidate_hits"]) + 1
                s["swap_tasks"].add(tid)
            if tid in triage:
                rank = triage[tid].get("rank", "")
                if rank and int(rank) <= 50:
                    s["top50_tasks"].add(tid)
                else:
                    s["next25_50_tasks"].add(tid)
            if row_full and dsl_cost is not None and v10_cost is not None:
                ratio = v10_cost / dsl_cost if dsl_cost > 0 else (999999.0 if v10_cost > 0 else 1.0)
                s["cost_ratios"].append(ratio)
                if dsl_cost < v10_cost:
                    s["v10_cheaper_hits"] = int(s["v10_cheaper_hits"]) + 1
                elif dsl_cost == v10_cost:
                    s["v10_equal_hits"] = int(s["v10_equal_hits"]) + 1
                else:
                    s["v10_loser_hits"] = int(s["v10_loser_hits"]) + 1
            examples = s["example_programs"]
            if len(examples) < 3:
                examples.append(f"task{tid:03d}: {program}")

    rows: list[dict[str, object]] = []
    for name, s in primitive_stats.items():
        ratios = list(s["cost_ratios"])
        row = {
            "primitive": name,
            "category": s["category"],
            "search_hits": s["search_hits"],
            "task_count": len(s["tasks"]),
            "full_onnx_pass_hits": s["full_onnx_pass_hits"],
            "train_only_or_runtime_mismatch_hits": s["train_only_or_runtime_mismatch_hits"],
            "v4_swap_candidate_hits": s["v4_swap_candidate_hits"],
            "v10_cheaper_hits": s["v10_cheaper_hits"],
            "v10_equal_hits": s["v10_equal_hits"],
            "v10_loser_hits": s["v10_loser_hits"],
            "median_v10_over_dsl_cost": round(sorted(ratios)[len(ratios) // 2], 3) if ratios else "",
            "best_v10_over_dsl_cost": round(max(ratios), 3) if ratios else "",
            "tasks": " ".join(f"task{tid:03d}" for tid in sorted(s["tasks"])),
            "full_pass_tasks": " ".join(f"task{tid:03d}" for tid in sorted(s["full_pass_tasks"])),
            "top50_overlap": " ".join(f"task{tid:03d}" for tid in sorted(s["top50_tasks"])),
            "next25_50_overlap": " ".join(f"task{tid:03d}" for tid in sorted(s["next25_50_tasks"])),
            "recommendation": recommendation_for(name, s),
            "note": s["note"],
            "examples": " || ".join(s["example_programs"]),
        }
        rows.append(row)

    category_order = {
        "needs_object_ir": 0,
        "needs_typed_rewrite": 1,
        "partly_mature": 2,
        "mature_cheap": 3,
        "unknown": 4,
    }
    rows.sort(
        key=lambda r: (
            category_order.get(str(r["category"]), 9),
            -int(r["search_hits"]),
            str(r["primitive"]),
        )
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "inputs": {
            "search_csv": str(SEARCH_CSV.relative_to(ROOT)),
            "swap_csv": str(SWAP_CSV.relative_to(ROOT)),
            "current_score_csv": str(CURRENT_SCORE_CSV.relative_to(ROOT)),
            "triage_top50_csv": str(TRIAGE_TOP50_CSV.relative_to(ROOT)),
            "triage_next25_50_csv": str(TRIAGE_NEXT_CSV.relative_to(ROOT)),
        },
        "search_hit_rows": len(search_rows),
        "swap_candidate_rows": len(swap_rows),
        "primitive_rows": len(rows),
        "category_counts": Counter(str(row["category"]) for row in rows),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=list) + "\n")

    lines = [
        "# DSL Frontier - 2026-06-06",
        "",
        "Current comparison anchor: `v4_plus10_compiler2`.",
        "",
        "## Readout",
        "",
        "- The existing DSL is correct and useful, but current search hits do not expose a fresh v10 submission branch.",
        "- Mature leaves (`transpose`, `swap_colors`, `flip_h`, `flip_v`) are already saturated or integrated.",
        "- The next non-local-maximum step is typed object/mask IR, not deeper search over full-FLOAT grid primitives.",
        "- Treat train-only hits as diagnostics. They are not candidates unless isolated ONNX full eval passes.",
        "",
        "## Primitive Frontier",
        "",
        "| primitive | category | hits | full | v10 cheaper/equal/loser | tasks | recommendation |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {primitive} | {category} | {search_hits} | {full_onnx_pass_hits} | {c}/{e}/{l} | {tasks} | {rec} |".format(
                primitive=row["primitive"],
                category=row["category"],
                search_hits=row["search_hits"],
                full_onnx_pass_hits=row["full_onnx_pass_hits"],
                c=row["v10_cheaper_hits"],
                e=row["v10_equal_hits"],
                l=row["v10_loser_hits"],
                tasks=str(row["tasks"]).replace("|", "/"),
                rec=str(row["recommendation"]).replace("|", "/"),
            )
        )

    lines += [
        "",
        "## Next Typed Primitive Bets",
        "",
        "1. `mask_color(c) -> BOOL[1,1,H,W]`: cheap channel extraction retained as BOOL.",
        "2. `paint_mask(grid, mask, color) -> grid`: final-stage paint with no FLOAT one-hot until output.",
        "3. `cover_mask(grid, mask) -> grid`: remove/zero selected cells as BOOL arithmetic.",
        "4. `where_grid(mask, a, b) -> grid`: typed selection that avoids `[1,10,30,30]` FLOAT mask casts when possible.",
        "5. `shift_mask_{up,down,left,right}` and `shift_grid_static`: Slice/Pad or Gather on BOOL/small spatial envelopes.",
        "6. `bbox(mask) -> scalar coords`: reuse small INT64 vectors/scalars across crop, shift, and object rules.",
        "7. `crop_static(r,c,h,w)`: preserve the cheap tiny-crop behavior and route dynamic bbox crop through object IR.",
        "8. `holes_not_bordering(mask)`: needed by task233/task255-style hole/corridor tasks.",
        "9. `majority_body_and_anchor_masks`: task285-like connected-object decomposition as a bounded template.",
        "10. `bounded_rect_fill`: task366-like rectangle/fill primitive with static envelopes.",
        "",
        "## Immediate Use",
        "",
        "- Keep Hegel on `task285` compact compile and Hume on `task191` compact compile.",
        "- Do not spend more time increasing depth over the current v4 primitive list; that is the local maximum.",
        "- Open the next semantic-factory pass only after either `task285/task191` returns or the typed primitives above have stubs.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"primitive_rows={len(rows)} search_hit_rows={len(search_rows)}")


if __name__ == "__main__":
    main()
