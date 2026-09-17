from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples
from prototype_task319 import analyze_objects
from task319_graph_feature_audit import surrogate_compress
from task319_transform_bank_audit import canonical_match, fixed_transform_bank


TASK = 319


def base_choice_python(objects: list[dict]) -> int:
    candidates = objects[1:]
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
    return int(chosen["color"])


def base_choice_graph_friendly(objects: list[dict]) -> int:
    largest = objects[0]
    largest_color = largest["color"]
    largest_aspect = largest["height"] / largest["width"]

    derived: list[dict] = []
    for obj in objects[1:]:
        compressed = surrogate_compress(obj["patch"])
        row_sums = compressed.sum(axis=1)
        variants = fixed_transform_bank(compressed)
        score = 0
        match_largest = 0
        for other in objects:
            if other["color"] == obj["color"]:
                continue
            if canonical_match(variants, surrogate_compress(other["patch"])):
                score += 1
                if other["color"] == largest_color:
                    match_largest = 1
        derived.append(
            {
                "color": obj["color"],
                "score": score,
                "match_largest": match_largest,
                "compressed_row_range": int(row_sums.max() - row_sums.min()),
                "aspect_gap": abs((obj["height"] / obj["width"]) - largest_aspect),
                "compressed_row_var": float(np.var(row_sums)),
                "compressed_density": float(compressed.sum() / compressed.size),
            }
        )

    chosen = max(
        derived,
        key=lambda obj: (
            obj["score"],
            obj["match_largest"],
            -obj["compressed_row_range"],
            -obj["aspect_gap"],
            -obj["compressed_row_var"],
            -obj["compressed_density"],
        ),
    )
    return int(chosen["color"])


def main() -> None:
    examples = all_examples(TASK, pathlib.Path("data/neurogolf-2026/raw"))
    mismatches: list[tuple[int, int, int]] = []

    for idx, example in enumerate(examples):
        inp = np.array(example["input"], dtype=np.int8)
        objects = analyze_objects(inp)
        expected = base_choice_python(objects)
        actual = base_choice_graph_friendly(objects)
        if expected != actual:
            mismatches.append((idx, expected, actual))

    print(f"task{TASK:03d} base-selector audit")
    print(f"mismatches: {len(mismatches)}")
    if mismatches:
        print("first mismatches", mismatches[:10])


if __name__ == "__main__":
    main()
