from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples


ASPECT_GAP_SWITCH = 1.0 / 3.0
MOTIF_IOU_SWITCH = 1.0 / 6.0


def dominant_color(inp: np.ndarray) -> int:
    values, counts = np.unique(inp, return_counts=True)
    return int(values[np.argmax(counts)])


def color_patch(inp: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(inp == color)
    r0, c0 = coords.min(0)
    r1, c1 = coords.max(0)
    patch = (inp[r0 : r1 + 1, c0 : c1 + 1] == color).astype(np.int8)
    return patch, (int(r0), int(c0), int(r1), int(c1))


def compress_runs(arr: np.ndarray) -> np.ndarray:
    rows = [arr[0]]
    for idx in range(1, arr.shape[0]):
        if not np.array_equal(arr[idx], rows[-1]):
            rows.append(arr[idx])
    arr = np.stack(rows)

    keep = [0]
    for idx in range(1, arr.shape[1]):
        if not np.array_equal(arr[:, idx], arr[:, keep[-1]]):
            keep.append(idx)
    return arr[:, keep]


def patch_variants(arr: np.ndarray) -> list[np.ndarray]:
    base = compress_runs(arr)
    variants: list[np.ndarray] = []
    for rot in range(4):
        turned = np.rot90(base, rot)
        for flip in [False, True]:
            oriented = np.fliplr(turned) if flip else turned
            for trim_top in [0, 1]:
                for trim_bottom in [0, 1]:
                    if trim_top and trim_bottom:
                        continue
                    for trim_left in [0, 1]:
                        for trim_right in [0, 1]:
                            if trim_left and trim_right:
                                continue
                            cropped = oriented
                            if trim_top:
                                cropped = cropped[1:, :]
                            if trim_bottom:
                                cropped = cropped[:-1, :]
                            if trim_left:
                                cropped = cropped[:, 1:]
                            if trim_right:
                                cropped = cropped[:, :-1]
                            if cropped.size == 0:
                                continue
                            if not any(
                                cropped.shape == prior.shape and np.array_equal(cropped, prior)
                                for prior in variants
                            ):
                                variants.append(cropped)
    return variants


def match_variant(variants: list[np.ndarray], target: np.ndarray) -> bool:
    return any(variant.shape == target.shape and np.array_equal(variant, target) for variant in variants)


def motif_counts(arr: np.ndarray, kh: int, kw: int) -> dict[tuple[int, ...], int]:
    counts: dict[tuple[int, ...], int] = {}
    if arr.shape[0] < kh or arr.shape[1] < kw:
        return counts
    for r in range(arr.shape[0] - kh + 1):
        for c in range(arr.shape[1] - kw + 1):
            key = tuple(int(v) for v in arr[r : r + kh, c : c + kw].reshape(-1))
            counts[key] = counts.get(key, 0) + 1
    return counts


def motif_overlap(a: dict[tuple[int, ...], int], b: dict[tuple[int, ...], int]) -> tuple[int, int]:
    keys = set(a) | set(b)
    inter = sum(min(a.get(key, 0), b.get(key, 0)) for key in keys)
    union = sum(max(a.get(key, 0), b.get(key, 0)) for key in keys)
    return inter, union


def analyze_objects(inp: np.ndarray) -> list[dict]:
    bg = dominant_color(inp)
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(inp) if int(v) != bg):
        patch, bbox = color_patch(inp, color)
        compressed = compress_runs(patch)
        row_sums = compressed.sum(axis=1)
        objects.append(
            {
                "color": color,
                "patch": patch,
                "compressed": compressed,
                "bbox": bbox,
                "cells": int(patch.sum()),
                "area": int(patch.shape[0] * patch.shape[1]),
                "height": int(patch.shape[0]),
                "width": int(patch.shape[1]),
                "compressed_cells": int(compressed.sum()),
                "compressed_density": float(compressed.sum() / compressed.size),
                "compressed_row_var": float(np.var(compressed.sum(axis=1))),
                "compressed_row_range": int(row_sums.max() - row_sums.min()),
            }
        )
    objects.sort(key=lambda obj: (-obj["area"], -obj["cells"], obj["color"]))
    largest_aspect = objects[0]["height"] / objects[0]["width"]
    largest_color = objects[0]["color"]
    for obj in objects:
        variants = patch_variants(obj["patch"])
        score = 0
        match_largest = 0
        for other in objects:
            if other["color"] == obj["color"]:
                continue
            compressed_other = other["compressed"]
            if match_variant(variants, compressed_other):
                score += 1
                if other["color"] == largest_color:
                    match_largest = 1
        obj["score"] = score
        obj["match_largest"] = match_largest
        obj["aspect_gap"] = abs((obj["height"] / obj["width"]) - largest_aspect)
    return objects


def choose_target(objects: list[dict]) -> dict:
    largest = objects[0]
    candidates = objects[1:]
    mirrored_largest = [
        obj
        for obj in candidates
        if obj["match_largest"] and obj["area"] == largest["area"] and obj["cells"] == largest["cells"]
    ]
    if len(mirrored_largest) == 1:
        return largest
    chosen = max(
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
    other = candidates[0] if chosen is candidates[1] else candidates[1]
    if chosen["aspect_gap"] - other["aspect_gap"] >= ASPECT_GAP_SWITCH:
        chosen, other = other, chosen

    large_2x2 = motif_counts(largest["compressed"], 2, 2)
    large_3x2 = motif_counts(largest["compressed"], 3, 2)

    def patch_lcomp_2x2_iou(obj: dict) -> float:
        inter, union = motif_overlap(motif_counts(obj["patch"], 2, 2), large_2x2)
        return inter / max(1, union)

    def comp_lcomp_3x2_inter(obj: dict) -> int:
        inter, _ = motif_overlap(motif_counts(obj["compressed"], 3, 2), large_3x2)
        return inter

    if (
        patch_lcomp_2x2_iou(other) - patch_lcomp_2x2_iou(chosen) >= MOTIF_IOU_SWITCH
        or comp_lcomp_3x2_inter(other) > comp_lcomp_3x2_inter(chosen)
    ):
        chosen, other = other, chosen

    if chosen["compressed"].shape == other["compressed"].shape:
        chosen_mask = chosen["compressed"] == 1
        other_mask = other["compressed"] == 1
        if np.all(chosen_mask <= other_mask) and other["compressed_cells"] == chosen["compressed_cells"] + 1:
            return other
    return chosen


def solve_task319(inp: np.ndarray) -> np.ndarray:
    bg = dominant_color(inp)
    objects = analyze_objects(inp)
    target = choose_target(objects)
    patch = target["patch"]
    return np.where(patch == 1, target["color"], bg).astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=319)
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--show-fails", type=int, default=8)
    args = parser.parse_args()

    examples = all_examples(args.task, pathlib.Path(args.comp_dir))
    fails: list[int] = []
    for idx, example in enumerate(examples):
        inp = np.array(example["input"], dtype=np.int8)
        gt = np.array(example["output"], dtype=np.int8)
        pred = solve_task319(inp)
        if not np.array_equal(pred, gt):
            fails.append(idx)

    print(f"task{args.task:03d}: {len(examples) - len(fails)}/{len(examples)}")
    if fails:
        print("fail_indices", fails[: args.show_fails])


if __name__ == "__main__":
    main()
