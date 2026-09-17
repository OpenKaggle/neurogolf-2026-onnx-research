"""task319 hidden-safety audit.

Splits the visible data into train (the official ARC-AGI demo pairs), test
(the official held-out pairs), and arc-gen (the bulk procedurally-generated
pairs). Then for each fallback combination of the prototype's `choose_target`
selector, reports per-split pass rates so we can tell which fallbacks are
robust on the unseen-leaning splits and which only buy visible coverage.

We do NOT build a per-task lookup table here, because the prototype is a
heuristic — there's nothing to "fit" on train. Instead we ablate fallbacks
and read the per-split delta as a proxy for hidden generalization:

  base                : `max(...)` ordering only
  +aspect             : adds the aspect-gap swap
  +aspect+mirrored    : also adds mirrored-largest short-circuit
  +aspect+mirrored+motif : also adds the 2x2 / 3x2 motif overrides
  full                : also adds the compressed-superset override
                        (this is the same as `choose_target` in the prototype)

Hidden-safe smell test: a robust fallback should fire (and help) at a similar
rate on `arc-gen` as on `train+test`. If a fallback's only visible
contribution is in `train+test` and `arc-gen` is unchanged, treat it as a
visible-only patch.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import load_examples
from prototype_task319 import (
    analyze_objects,
    dominant_color,
    motif_counts,
    motif_overlap,
    ASPECT_GAP_SWITCH,
    MOTIF_IOU_SWITCH,
)


TASK = 319


def _base_chosen(candidates):
    return max(
        candidates,
        key=lambda obj: (
            obj["score"],
            obj["match_largest"],
            -obj["compressed_row_range"],
            -obj["aspect_gap"],
            -obj["compressed_row_var"],
            -obj["compressed_density"],
        ),
    )


def selector(objects, *, use_aspect: bool, use_mirrored: bool, use_motif: bool, use_superset: bool):
    largest = objects[0]
    candidates = objects[1:]

    # mirrored_largest short-circuit (the prototype runs this BEFORE the
    # base ordering, returning largest itself when triggered).
    mirrored = [
        obj
        for obj in candidates
        if obj["match_largest"]
        and obj["area"] == largest["area"]
        and obj["cells"] == largest["cells"]
    ]
    if use_mirrored and len(mirrored) == 1:
        return largest

    chosen = _base_chosen(candidates)
    other = candidates[0] if chosen is candidates[1] else candidates[1]

    if use_aspect and chosen["aspect_gap"] - other["aspect_gap"] >= ASPECT_GAP_SWITCH:
        chosen, other = other, chosen

    if use_motif:
        large_2x2 = motif_counts(largest["compressed"], 2, 2)
        large_3x2 = motif_counts(largest["compressed"], 3, 2)

        def patch_lcomp_2x2_iou(obj):
            inter, union = motif_overlap(motif_counts(obj["patch"], 2, 2), large_2x2)
            return inter / max(1, union)

        def comp_lcomp_3x2_inter(obj):
            inter, _ = motif_overlap(motif_counts(obj["compressed"], 3, 2), large_3x2)
            return inter

        if (
            patch_lcomp_2x2_iou(other) - patch_lcomp_2x2_iou(chosen) >= MOTIF_IOU_SWITCH
            or comp_lcomp_3x2_inter(other) > comp_lcomp_3x2_inter(chosen)
        ):
            chosen, other = other, chosen

    if use_superset and chosen["compressed"].shape == other["compressed"].shape:
        chosen_mask = chosen["compressed"] == 1
        other_mask = other["compressed"] == 1
        if np.all(chosen_mask <= other_mask) and other["compressed_cells"] == chosen["compressed_cells"] + 1:
            return other

    return chosen


def render(inp, target_obj):
    bg = dominant_color(inp)
    patch = target_obj["patch"]
    return np.where(patch == 1, target_obj["color"], bg).astype(np.int8)


def evaluate_split(examples, **selector_kwargs):
    passed = 0
    failed_idx = []
    for idx, ex in enumerate(examples):
        inp = np.array(ex["input"], dtype=np.int8)
        gt = np.array(ex["output"], dtype=np.int8)
        objects = analyze_objects(inp)
        target = selector(objects, **selector_kwargs)
        pred = render(inp, target)
        if np.array_equal(pred, gt):
            passed += 1
        else:
            failed_idx.append(idx)
    return passed, failed_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    args = parser.parse_args()

    ex = load_examples(TASK, pathlib.Path(args.comp_dir))
    splits = {
        "train": ex["train"],
        "test": ex["test"],
        "arc-gen": ex.get("arc-gen", []),
    }

    configs = [
        ("base               ", dict(use_aspect=False, use_mirrored=False, use_motif=False, use_superset=False)),
        ("+aspect            ", dict(use_aspect=True,  use_mirrored=False, use_motif=False, use_superset=False)),
        ("+aspect+mirrored   ", dict(use_aspect=True,  use_mirrored=True,  use_motif=False, use_superset=False)),
        ("+aspect+mir+motif  ", dict(use_aspect=True,  use_mirrored=True,  use_motif=True,  use_superset=False)),
        ("+aspect+mir+motif+sup (full)", dict(use_aspect=True, use_mirrored=True, use_motif=True, use_superset=True)),
    ]

    print(f"task{TASK:03d} per-split fallback ablation (pass / total):\n")
    header = f"{'config':<32} | " + " | ".join(f"{name:>10s}" for name in splits)
    print(header)
    print("-" * len(header))
    for label, kw in configs:
        cells = []
        for name, split_examples in splits.items():
            passed, _failed = evaluate_split(split_examples, **kw)
            cells.append(f"{passed}/{len(split_examples)}")
        print(f"{label:<32} | " + " | ".join(f"{c:>10s}" for c in cells))

    print("\nper-fallback delta (pass-count change vs the previous row):")
    prev = None
    for label, kw in configs:
        passes = []
        for split_examples in splits.values():
            passes.append(evaluate_split(split_examples, **kw)[0])
        if prev is not None:
            deltas = [p - q for p, q in zip(passes, prev)]
            print(f"  {label:<32} | " + " | ".join(f"{d:+10d}" for d in deltas))
        prev = passes


if __name__ == "__main__":
    main()
