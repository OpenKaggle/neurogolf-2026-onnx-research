from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples
from prototype_task319 import analyze_objects, patch_variants


TASK = 319
CANVAS = 5


def pad5(arr: np.ndarray) -> np.ndarray:
    out = np.zeros((CANVAS, CANVAS), dtype=np.int8)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def canonical_variants(arr: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    return [((variant.shape[0], variant.shape[1]), pad5(variant)) for variant in patch_variants(arr)]


def canonical_match(variants: list[tuple[tuple[int, int], np.ndarray]], target: np.ndarray) -> bool:
    target_shape = (target.shape[0], target.shape[1])
    target5 = pad5(target)
    return any(shape == target_shape and np.array_equal(variant, target5) for shape, variant in variants)


def main() -> None:
    examples = all_examples(TASK, pathlib.Path("data/neurogolf-2026/raw"))
    mismatches: list[tuple[int, int, int, int, int]] = []
    total_objects = 0
    max_variants = 0
    variant_hist: dict[int, int] = {}
    max_comp_shape = (0, 0)

    for idx, example in enumerate(examples):
        inp = np.array(example["input"], dtype=np.int8)
        objects = analyze_objects(inp)
        largest_color = objects[0]["color"]

        for obj in objects:
            total_objects += 1
            comp = obj["compressed"]
            max_comp_shape = max(max_comp_shape, comp.shape)
            variants5 = canonical_variants(obj["patch"])
            max_variants = max(max_variants, len(variants5))
            variant_hist[len(variants5)] = variant_hist.get(len(variants5), 0) + 1

            score = 0
            match_largest = 0
            for other in objects:
                if other["color"] == obj["color"]:
                    continue
                if canonical_match(variants5, other["compressed"]):
                    score += 1
                    if other["color"] == largest_color:
                        match_largest = 1

            if score != obj["score"] or match_largest != obj["match_largest"]:
                mismatches.append((idx, obj["color"], obj["score"], score, match_largest))

    print(f"task{TASK:03d} canonical variant audit on {total_objects} objects")
    print(f"max compressed shape: {max_comp_shape}")
    print(f"max variant count: {max_variants}")
    print(f"variant histogram: {dict(sorted(variant_hist.items()))}")
    print(f"mismatches: {len(mismatches)}")
    if mismatches:
        print("first mismatches", mismatches[:10])


if __name__ == "__main__":
    main()
