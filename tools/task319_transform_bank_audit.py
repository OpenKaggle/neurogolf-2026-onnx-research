from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples
from prototype_task319 import analyze_objects, patch_variants


TASK = 319
CANVAS = 5


def pad5(arr: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    out = np.zeros((CANVAS, CANVAS), dtype=np.int8)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return (int(arr.shape[0]), int(arr.shape[1])), out


def fixed_transform_bank(arr: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    base = arr.copy()
    variants: list[tuple[tuple[int, int], np.ndarray]] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
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
                            shape, padded = pad5(cropped)
                            key = (shape, tuple(int(v) for v in padded.reshape(-1)))
                            if key not in seen:
                                seen.add(key)
                                variants.append((shape, padded))
    return variants


def canonical_variants(arr: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    return [pad5(variant) for variant in patch_variants(arr)]


def canonical_match(
    variants: list[tuple[tuple[int, int], np.ndarray]], target: np.ndarray
) -> bool:
    target_shape, target_pad = pad5(target)
    return any(shape == target_shape and np.array_equal(padded, target_pad) for shape, padded in variants)


def main() -> None:
    examples = all_examples(TASK, pathlib.Path("data/neurogolf-2026/raw"))
    object_mismatches: list[tuple[int, int]] = []
    selector_mismatches: list[tuple[int, int, int, int]] = []
    max_bank = 0

    for idx, example in enumerate(examples):
        inp = np.array(example["input"], dtype=np.int8)
        objects = analyze_objects(inp)
        largest_color = objects[0]["color"]

        for obj in objects:
            exact = canonical_variants(obj["compressed"])
            bank = fixed_transform_bank(obj["compressed"])
            max_bank = max(max_bank, len(bank))
            exact_set = {(shape, tuple(int(v) for v in padded.reshape(-1))) for shape, padded in exact}
            bank_set = {(shape, tuple(int(v) for v in padded.reshape(-1))) for shape, padded in bank}
            if exact_set != bank_set:
                object_mismatches.append((idx, obj["color"]))

            score = 0
            match_largest = 0
            for other in objects:
                if other["color"] == obj["color"]:
                    continue
                if canonical_match(bank, other["compressed"]):
                    score += 1
                    if other["color"] == largest_color:
                        match_largest = 1
            if score != obj["score"] or match_largest != obj["match_largest"]:
                selector_mismatches.append((idx, obj["color"], score, match_largest))

    print(f"task{TASK:03d} transform-bank audit")
    print(f"max universal bank size seen: {max_bank}")
    print(f"object-level variant mismatches: {len(object_mismatches)}")
    print(f"selector-level mismatches: {len(selector_mismatches)}")
    if object_mismatches:
        print("first object mismatches", object_mismatches[:10])
    if selector_mismatches:
        print("first selector mismatches", selector_mismatches[:10])


if __name__ == "__main__":
    main()
