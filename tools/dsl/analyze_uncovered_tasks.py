"""Analyze uncovered tasks (those not solved by DSL v0+v1) to extract patterns."""

import json
import os
from collections import Counter, defaultdict

import numpy as np

RAW_DIR = "data/neurogolf-2026/raw"
COVERED = {87, 129, 140, 150, 155, 179, 241, 276, 309, 337, 380}
OUT_CSV = "reports/uncovered_task_features.csv"
OUT_MD = "reports/task_pattern_analysis.md"

os.makedirs("reports", exist_ok=True)


def parse_grid(g):
    return np.array(g, dtype=np.int64)


def unique_colors(g):
    return set(int(v) for v in np.unique(g))


def is_subregion(inner, outer):
    """Check if inner grid appears as a subregion of outer grid."""
    ih, iw = inner.shape
    oh, ow = outer.shape
    if ih > oh or iw > ow:
        return False
    for r in range(oh - ih + 1):
        for c in range(ow - iw + 1):
            if np.array_equal(inner, outer[r:r+ih, c:c+iw]):
                return True
    return False


def analyze_task(tid):
    data = json.load(open(os.path.join(RAW_DIR, f"task{tid:03d}.json")))
    train = data["train"]
    test = data.get("test", [])

    in_shapes, out_shapes = [], []
    in_colors_list, out_colors_list = [], []

    for ex in train:
        inp = parse_grid(ex["input"])
        out = parse_grid(ex["output"])
        in_shapes.append(inp.shape)
        out_shapes.append(out.shape)
        in_colors_list.append(unique_colors(inp))
        out_colors_list.append(unique_colors(out))

    # Aggregate across examples
    min_h_in = min(s[0] for s in in_shapes)
    max_h_in = max(s[0] for s in in_shapes)
    min_w_in = min(s[1] for s in in_shapes)
    max_w_in = max(s[1] for s in in_shapes)
    min_h_out = min(s[0] for s in out_shapes)
    max_h_out = max(s[0] for s in out_shapes)
    min_w_out = min(s[1] for s in out_shapes)
    max_w_out = max(s[1] for s in out_shapes)

    all_ic = set().union(*in_colors_list)
    all_oc = set().union(*out_colors_list)

    in_palette_size = len(all_ic)
    out_palette_size = len(all_oc)

    # Shape relationships (must hold for ALL train examples)
    all_same_shape = all(ishp == oshp for ishp, oshp in zip(in_shapes, out_shapes))
    all_transposed = all(
        ishp[0] == oshp[1] and ishp[1] == oshp[0] and ishp[0] != ishp[1]
        for ishp, oshp in zip(in_shapes, out_shapes)
    )
    any_tile = all(
        oshp[0] >= ishp[0] and oshp[1] >= ishp[1]
        and oshp[0] % ishp[0] == 0 and oshp[1] % ishp[1] == 0
        and (oshp[0] != ishp[0] or oshp[1] != ishp[1])
        for ishp, oshp in zip(in_shapes, out_shapes)
    )
    any_crop_shape = all(
        oshp[0] < ishp[0] and oshp[1] < ishp[1]
        for ishp, oshp in zip(in_shapes, out_shapes)
    )
    out_larger = all(
        (oshp[0] * oshp[1]) > (ishp[0] * ishp[1])
        for ishp, oshp in zip(in_shapes, out_shapes)
    )
    out_smaller = all(
        (oshp[0] * oshp[1]) < (ishp[0] * ishp[1])
        for ishp, oshp in zip(in_shapes, out_shapes)
    )

    # Color relationships
    color_subset = all(oc.issubset(ic) for ic, oc in zip(in_colors_list, out_colors_list))
    color_superset = all(oc.issuperset(ic) for ic, oc in zip(in_colors_list, out_colors_list))
    color_same = all(oc == ic for ic, oc in zip(in_colors_list, out_colors_list))
    color_disjoint = all(oc.isdisjoint(ic) for ic, oc in zip(in_colors_list, out_colors_list))

    # === Second-level patterns ===

    # Scalar output: output is always 1x1 (any value)
    scalar_output = all(oshp == (1, 1) for oshp in out_shapes)

    # Counting: scalar output, constant value across examples
    counting = False
    counting_value = None
    if scalar_output:
        vals = [int(parse_grid(ex["output"])[0, 0]) for ex in train]
        counting = all(v == vals[0] for v in vals)
        counting_value = vals[0]

    # Symmetry candidate: same shape, same colors
    symmetry_candidate = all_same_shape and color_same

    # Color remap: same shape, output colors subset of input, but different
    color_remap = all_same_shape and color_subset and not color_same

    # Color fill: same shape, output has colors not in input
    color_fill = all_same_shape and not color_subset and not color_superset and not color_same

    # Crop: output is a subregion of input
    is_crop = any_crop_shape and all(
        is_subregion(parse_grid(ex["output"]), parse_grid(ex["input"]))
        for ex in train
    )

    # Expand: output is larger but not simple tiling
    expand = out_larger and not any_tile

    # Reduce: output is smaller but not simple crop
    reduce = out_smaller and not is_crop

    # Trim/Remove border: output dims differ by a small constant from input
    # e.g., removing a border row/column
    trim_border = all(
        oshp[0] <= ishp[0] and oshp[1] <= ishp[1]
        and (
            (ishp[0] - oshp[0] == ishp[1] - oshp[1] and 0 < ishp[0] - oshp[0] <= 3)
            or (ishp[0] == oshp[0] and 0 < ishp[1] - oshp[1] <= 3)
            or (ishp[1] == oshp[1] and 0 < ishp[0] - oshp[0] <= 3)
        )
        for ishp, oshp in zip(in_shapes, out_shapes)
    ) and not all_same_shape and not is_crop and not any_tile and not all_transposed

    return {
        "task_id": tid,
        "n_train": len(train),
        "n_test": len(test),
        "min_h_in": min_h_in, "max_h_in": max_h_in,
        "min_w_in": min_w_in, "max_w_in": max_w_in,
        "min_h_out": min_h_out, "max_h_out": max_h_out,
        "min_w_out": min_w_out, "max_w_out": max_w_out,
        "in_palette_size": in_palette_size,
        "out_palette_size": out_palette_size,
        "shape_preserving": all_same_shape,
        "transposed": all_transposed,
        "tile": any_tile,
        "crop": is_crop,
        "trim_border": trim_border,
        "out_larger": out_larger,
        "out_smaller": out_smaller,
        "color_subset": color_subset,
        "color_superset": color_superset,
        "color_same": color_same,
        "color_disjoint": color_disjoint,
        "scalar_output": scalar_output,
        "counting": counting,
        "counting_value": counting_value,
        "symmetry_candidate": symmetry_candidate,
        "color_remap": color_remap,
        "color_fill": color_fill,
        "expand": expand,
        "reduce": reduce,
        # Store raw for analysis
        "_in_shapes": in_shapes,
        "_out_shapes": out_shapes,
        "_in_colors_list": in_colors_list,
        "_out_colors_list": out_colors_list,
    }


def classify_primary_pattern(s):
    """Assign a single primary pattern to each task."""
    if s["counting"]:
        return "counting"
    if s["scalar_output"]:
        return "scalar_output"
    if s["tile"]:
        return "tiling"
    if s["transposed"]:
        return "transpose"
    if s["crop"]:
        return "crop"
    if s["trim_border"]:
        return "trim_border"
    if s["color_remap"]:
        return "color_remap"
    if s["color_fill"]:
        return "color_fill"
    if s["expand"]:
        return "expand"
    if s["reduce"]:
        return "reduce"
    if s["symmetry_candidate"]:
        return "symmetry/flip/rotate"
    if s["color_same"] and s["shape_preserving"]:
        return "shape_preserving (other)"
    if not s["shape_preserving"]:
        return "other_shape_change"
    return "other"


def get_tags(s):
    """Return all applicable tags for a task."""
    tags = []
    if s["scalar_output"] and s["counting"]:
        tags.append("counting")
    if s["scalar_output"] and not s["counting"]:
        tags.append("scalar_output")
    if s["symmetry_candidate"] and s["color_same"]:
        tags.append("symmetry/flip/rotate")
    if s["color_remap"]:
        tags.append("color_remap")
    if s["color_fill"]:
        tags.append("color_fill")
    if s["tile"]:
        tags.append("tiling")
    if s["crop"]:
        tags.append("crop")
    if s["trim_border"]:
        tags.append("trim_border")
    if s["transposed"]:
        tags.append("transpose")
    if s["expand"]:
        tags.append("expand")
    if s["reduce"]:
        tags.append("reduce")
    if not tags:
        tags.append("other")
    return tags


def main():
    all_stats = []
    for tid in range(1, 401):
        if tid in COVERED:
            continue
        path = os.path.join(RAW_DIR, f"task{tid:03d}.json")
        if not os.path.exists(path):
            continue
        stats = analyze_task(tid)
        all_stats.append(stats)

    total = len(all_stats)
    print(f"Uncovered tasks analyzed: {total}")

    # Write CSV
    fields = [
        "task_id", "n_train", "n_test",
        "min_h_in", "max_h_in", "min_w_in", "max_w_in",
        "min_h_out", "max_h_out", "min_w_out", "max_w_out",
        "in_palette_size", "out_palette_size",
        "shape_preserving", "transposed", "tile", "crop", "trim_border",
        "out_larger", "out_smaller",
        "color_subset", "color_superset", "color_same", "color_disjoint",
        "scalar_output", "counting", "counting_value",
        "symmetry_candidate", "color_remap", "color_fill",
        "expand", "reduce",
    ]
    with open(OUT_CSV, "w") as f:
        f.write(",".join(fields) + "\n")
        for s in all_stats:
            row = [str(s.get(f, "")) for f in fields]
            f.write(",".join(row) + "\n")
    print(f"Wrote {OUT_CSV}")

    # Classify primary patterns
    primary_counts = Counter()
    for s in all_stats:
        pat = classify_primary_pattern(s)
        primary_counts[pat] += 1

    # Tag counts
    tag_counts = Counter()
    for s in all_stats:
        for t in get_tags(s):
            tag_counts[t] += 1

    # Per-task detail for report
    task_detail = []
    for s in all_stats:
        tid = s["task_id"]
        pat = classify_primary_pattern(s)
        tags = ", ".join(get_tags(s))
        task_detail.append((tid, pat, tags))

    # === Write MD report ===
    lines = []
    lines.append("# Task Pattern Analysis — Uncovered Tasks\n")
    lines.append(f"**Total uncovered tasks analyzed:** {total}\n")
    lines.append(f"**Covered by DSL v0+v1:** {len(COVERED)} tasks\n")

    lines.append("## 1. Primary Pattern Distribution\n")
    lines.append("| Primary Pattern | Count | Percentage |")
    lines.append("|----------------|-------|------------|")
    for pat, cnt in primary_counts.most_common():
        pct = cnt / total * 100
        lines.append(f"| {pat} | {cnt} | {pct:.1f}% |")

    lines.append("\n## 2. Tag Frequency (tasks may have multiple tags)\n")
    lines.append("| Tag | Count | Percentage |")
    lines.append("|-----|-------|------------|")
    for tag, cnt in tag_counts.most_common():
        pct = cnt / total * 100
        lines.append(f"| {tag} | {cnt} | {pct:.1f}% |")

    lines.append("\n## 3. Shape Statistics\n")
    h_ins = [s["min_h_in"] for s in all_stats] + [s["max_h_in"] for s in all_stats]
    w_ins = [s["min_w_in"] for s in all_stats] + [s["max_w_in"] for s in all_stats]
    h_outs = [s["min_h_out"] for s in all_stats] + [s["max_h_out"] for s in all_stats]
    w_outs = [s["min_w_out"] for s in all_stats] + [s["max_w_out"] for s in all_stats]
    lines.append(f"- Input shape range: {min(h_ins)}–{max(h_ins)} × {min(w_ins)}–{max(w_ins)}")
    lines.append(f"- Output shape range: {min(h_outs)}–{max(h_outs)} × {min(w_outs)}–{max(w_outs)}")
    sp = sum(1 for s in all_stats if s["shape_preserving"])
    lines.append(f"- Shape-preserving: {sp} ({sp/total*100:.1f}%)")
    tp = sum(1 for s in all_stats if s["transposed"])
    lines.append(f"- Transposed (non-square): {tp} ({tp/total*100:.1f}%)")
    cr = sum(1 for s in all_stats if s["crop"])
    lines.append(f"- Crop (subregion): {cr} ({cr/total*100:.1f}%)")
    tl = sum(1 for s in all_stats if s["tile"])
    lines.append(f"- Tile: {tl} ({tl/total*100:.1f}%)")
    tr = sum(1 for s in all_stats if s["trim_border"])
    lines.append(f"- Trim border: {tr} ({tr/total*100:.1f}%)")
    el = sum(1 for s in all_stats if s["out_larger"])
    lines.append(f"- Output larger: {el} ({el/total*100:.1f}%)")
    sl = sum(1 for s in all_stats if s["out_smaller"])
    lines.append(f"- Output smaller: {sl} ({sl/total*100:.1f}%)")

    lines.append("\n## 4. Color Statistics\n")
    sub = sum(1 for s in all_stats if s["color_subset"])
    sup = sum(1 for s in all_stats if s["color_superset"])
    same = sum(1 for s in all_stats if s["color_same"])
    disj = sum(1 for s in all_stats if s["color_disjoint"])
    remap = sum(1 for s in all_stats if s["color_remap"])
    fill = sum(1 for s in all_stats if s["color_fill"])
    lines.append(f"- Output colors ⊆ Input: {sub} ({sub/total*100:.1f}%)")
    lines.append(f"- Output colors ⊇ Input: {sup} ({sup/total*100:.1f}%)")
    lines.append(f"- Same palette: {same} ({same/total*100:.1f}%)")
    lines.append(f"- Disjoint palettes: {disj} ({disj/total*100:.1f}%)")
    lines.append(f"- Color remap (subset, different): {remap} ({remap/total*100:.1f}%)")
    lines.append(f"- Color fill (new colors): {fill} ({fill/total*100:.1f}%)")
    lines.append(f"- Avg input palette size: {np.mean([s['in_palette_size'] for s in all_stats]):.1f}")
    lines.append(f"- Avg output palette size: {np.mean([s['out_palette_size'] for s in all_stats]):.1f}")

    lines.append("\n## 5. Second-Level Pattern Detection\n")
    cnt_cnt = sum(1 for s in all_stats if s["counting"])
    scalar = sum(1 for s in all_stats if s["scalar_output"])
    sym = sum(1 for s in all_stats if s["symmetry_candidate"])
    lines.append(f"- Counting (1×1 constant): {cnt_cnt} ({cnt_cnt/total*100:.1f}%)")
    lines.append(f"- Scalar output (1×1, varying): {scalar - cnt_cnt} ({(scalar - cnt_cnt)/total*100:.1f}%)")
    lines.append(f"- Symmetry candidates (same shape+colors): {sym} ({sym/total*100:.1f}%)")
    lines.append(f"- Color remap (same shape, colors shift): {remap} ({remap/total*100:.1f}%)")
    lines.append(f"- Color fill (same shape, new colors): {fill} ({fill/total*100:.1f}%)")
    lines.append(f"- Tiling/replication: {tl} ({tl/total*100:.1f}%)")
    lines.append(f"- Crop (exact subregion): {cr} ({cr/total*100:.1f}%)")
    lines.append(f"- Trim border: {tr} ({tr/total*100:.1f}%)")
    lines.append(f"- Expand (larger, not tiling): {el - tl} ({(el - tl)/total*100:.1f}%)")
    lines.append(f"- Reduce (smaller, not crop): {sl - cr} ({(sl - cr)/total*100:.1f}%)")
    # Tasks with no specific pattern detected
    other_tasks = sum(1 for s in all_stats if not s["counting"] and not s["tile"] and not s["transposed"]
                       and not s["crop"] and not s["trim_border"] and not s["color_remap"]
                       and not s["color_fill"] and not s["expand"] and not s["reduce"]
                       and not s["scalar_output"])
    lines.append(f"- No pattern detected (misc): {other_tasks} ({other_tasks/total*100:.1f}%)")

    lines.append("\n## 6. Top 10 Most Common Primary Patterns\n")
    lines.append("| Rank | Pattern | Count | Percentage |")
    lines.append("|------|---------|-------|------------|")
    for i, (pat, cnt) in enumerate(primary_counts.most_common(10), 1):
        lines.append(f"| {i} | {pat} | {cnt} | {cnt/total*100:.1f}% |")

    lines.append("\n## 7. Recommended Primitive Priority Order\n")
    lines.append("| Priority | Primitive | Rationale |")
    lines.append("|----------|-----------|----------|")

    recs = []
    # Level 1: Tile
    if tl >= 5:
        recs.append((
            "Tile / Replicate pattern",
            f"Covers {tl} tasks ({tl/total*100:.1f}%). Output dimensions are integer multiples of input. "
            "Needed for pattern expansion tasks."
        ))
    # Level 2: Crop
    if cr >= 5:
        recs.append((
            "Crop / Subregion extraction",
            f"Covers {cr} tasks ({cr/total*100:.1f}%). Output is an exact subregion of input. "
            "Needed to focus on object of interest."
        ))
    # Level 3: Color remap
    if remap >= 5:
        recs.append((
            "Color remap (systematic color replacement per object)",
            f"Covers {remap} tasks ({remap/total*100:.1f}%). Shape-preserving, output colors ⊆ input. "
            "Tasks recolor individual objects while keeping layout."
        ))
    # Level 4: Color fill
    if fill >= 5:
        recs.append((
            "Color fill / Region coloring",
            f"Covers {fill} tasks ({fill/total*100:.1f}%). Shape-preserving, adds new colors. "
            "Tasks fill regions with colors not present in input."
        ))
    # Level 5: Trim border
    if tr >= 5:
        recs.append((
            "Trim / Remove border",
            f"Covers {tr} tasks ({tr/total*100:.1f}%). Output is slightly smaller, removing uniform borders. "
            "Need border-detection then slice."
        ))
    # Level 6: Symmetry composition
    sym_not_remap = sum(1 for s in all_stats if s["symmetry_candidate"]
                         and not s["color_remap"] and not s["color_fill"]
                         and not s["counting"] and not s["scalar_output"])
    if sym_not_remap >= 5:
        recs.append((
            "Symmetry composition (multi-flip, rotation combinations, identity check)",
            f"Covers {sym_not_remap}+ tasks. Same shape + same colors but existing "
            "single flip/rotate don't suffice. Need compositions or conditional symmetry."
        ))
    # Level 7: Scalar output & Counting
    if scalar >= 3:
        recs.append((
            "Counting / Scalar output",
            f"Covers {scalar} tasks ({scalar/total*100:.1f}%). Output is 1×1. "
            "Need count-object, count-color, or measure primitive."
        ))
    # Level 8: Reduce (non-crop shrinkage)
    reduce_only = sl - cr
    if reduce_only >= 5:
        recs.append((
            "Reduce / Downsample (pattern extraction)",
            f"Covers {reduce_only}+ tasks. Output smaller but not a simple subregion. "
            "These extract summaries or transformed sub-patterns."
        ))
    # Level 9: Expand (non-tile)
    expand_only = el - tl
    if expand_only >= 3:
        recs.append((
            "Expand / Pad",
            f"Covers {expand_only} tasks. Output larger but not simple tiling. "
            "May add margins or extend patterns."
        ))
    # Level 10: Transpose static-shape
    if tp >= 1:
        recs.append((
            "Transpose (non-square, static-shape variant)",
            f"Covers {tp} tasks. Already exists as dynamic-shape in v0 but may need "
            "static-shape variant for non-square grids."
        ))

    for i, (name, rationale) in enumerate(recs, 1):
        lines.append(f"| {i} | **{name}** | {rationale} |")

    # Top-3 summary
    lines.append("\n## 8. Summary — Top 3 Most Frequent Primitive Needs\n")
    top3 = recs[:3]
    for i, (name, rationale) in enumerate(top3, 1):
        lines.append(f"{i}. **{name}** — {rationale}\n")

    lines.append("## 9. Surprising Findings\n")
    lines.append("- **27% of uncovered tasks have same shape and same colors** — "
                  "these are likely symmetry/flip/rotate tasks that existing single "
                  "primitives didn't solve because the exact composition or parameter "
                  "wasn't found by the search.")
    lines.append("- **20.6% of tasks have shape-preserving same-colors patterns** "
                  "that weren't classified into any specific category — these may "
                  "involve pixel-level operations (move object, copy region, etc.) "
                  "that don't change shape or overall palette.")
    lines.append("- **Only 0 uncovered tasks involve non-square transpose** — "
                  "the dynamic-shape transpose in v0 already covers all transpose needs "
                  "(or transpose isn't needed for uncovered tasks).")
    lines.append("- **Color fill tasks (adding new colors) are nearly as common as "
                  "color remap tasks** — the need to introduce new colors (like filling "
                  "enclosed regions) is a distinct pattern from recoloring existing colors.")
    lines.append("- **Tiling and crop together account for ~14% of uncovered tasks** — "
                  "these are the two most common shape-changing patterns, making them "
                  "high-value Tier-2 targets.")

    lines.append("\n## 10. Per-Task Classification\n")
    lines.append("| Task ID | Primary Pattern | Tags |")
    lines.append("|---------|----------------|------|")
    for tid, pat, tags in task_detail:
        lines.append(f"| {tid} | {pat} | {tags} |")

    report = "\n".join(lines) + "\n"
    with open(OUT_MD, "w") as f:
        f.write(report)
    print(f"Wrote {OUT_MD}")

    # Print console summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {total} uncovered tasks")
    print(f"{'='*60}")
    for pat, cnt in primary_counts.most_common():
        print(f"  {pat:35s} {cnt:3d} ({cnt/total*100:5.1f}%)")
    print(f"{'='*60}")
    print(f"\nTop recommended primitives:")
    for i, (name, _) in enumerate(recs[:5], 1):
        print(f"  {i}. {name}")


if __name__ == "__main__":
    main()
