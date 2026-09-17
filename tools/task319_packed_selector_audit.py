from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_task319_compress_features import python_features as compress_python_features
from build_task319_packed_compressed import python_features as packed_python_features
from neurogolf_local import all_examples
from prototype_task319 import analyze_objects
from task319_transform_bank_audit import canonical_match, fixed_transform_bank


TASK = 319


def unpack_compressed(
    packed: np.ndarray,
    compressed_rows: int,
    compressed_cols: int,
) -> np.ndarray:
    return packed[:compressed_rows, :compressed_cols].astype(np.int8)


def base_choice_from_packed(inp: np.ndarray) -> int:
    objects = analyze_objects(inp)
    packed = packed_python_features(inp)
    comp = compress_python_features(inp)

    largest = objects[0]
    largest_color = int(largest["color"])
    largest_aspect = largest["height"] / largest["width"]

    derived: list[dict] = []
    for obj in objects[1:]:
        color = int(obj["color"])
        compressed = unpack_compressed(
            packed["packed_compressed"][0, color],
            int(comp["compressed_rows"][0, color]),
            int(comp["compressed_cols"][0, color]),
        )
        row_sums = compressed.sum(axis=1)
        variants = fixed_transform_bank(compressed)
        score = 0
        match_largest = 0
        for other in objects:
            other_color = int(other["color"])
            if other_color == color:
                continue
            other_compressed = unpack_compressed(
                packed["packed_compressed"][0, other_color],
                int(comp["compressed_rows"][0, other_color]),
                int(comp["compressed_cols"][0, other_color]),
            )
            if canonical_match(variants, other_compressed):
                score += 1
                if other_color == largest_color:
                    match_largest = 1
        derived.append(
            {
                "color": color,
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
        expected = max(
            objects[1:],
            key=lambda obj: (
                obj["score"],
                obj["match_largest"],
                -obj["compressed_row_range"],
                -obj["aspect_gap"],
                -obj["compressed_row_var"],
                -obj["compressed_density"],
            ),
        )["color"]
        actual = base_choice_from_packed(inp)
        if int(expected) != int(actual):
            mismatches.append((idx, int(expected), int(actual)))

    print(f"task{TASK:03d} packed-selector audit")
    print(f"mismatches: {len(mismatches)}")
    if mismatches:
        print("first mismatches", mismatches[:10])


if __name__ == "__main__":
    main()
