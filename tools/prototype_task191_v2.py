from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pseudo_hidden import (  # noqa: E402
    ArcExample,
    InvarianceContract,
    format_failures,
    run_contracts,
    summarize_results,
)


TASK_PATH = ROOT / "data" / "neurogolf-2026" / "raw" / "task191.json"


Grid = np.ndarray


@dataclass(frozen=True)
class StampVariant:
    name: str
    grid: Grid

    @property
    def fill_mask(self) -> Grid:
        return self.grid == 1

    @property
    def hole_mask(self) -> Grid:
        return self.grid == 4


def as_grid(rows: Iterable[Iterable[int]]) -> Grid:
    return np.asarray(list(rows), dtype=np.int64)


def grid_to_lines(grid: Grid) -> list[str]:
    return ["".join(str(int(v)) for v in row) for row in grid]


def load_labeled_examples() -> list[tuple[str, int, Grid, Grid]]:
    data = json.loads(TASK_PATH.read_text())
    examples: list[tuple[str, int, Grid, Grid]] = []
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data[split]):
            examples.append((split, index, as_grid(example["input"]), as_grid(example["output"])))
    return examples


def source_stamp(grid: Grid) -> Grid:
    rows, cols = np.where(grid == 1)
    if len(rows) == 0:
        raise ValueError("task191 grids must contain a source stamp")
    return grid[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1].copy()


def stamp_variants(stamp: Grid) -> list[StampVariant]:
    candidates: list[tuple[str, Grid]] = []
    for k in range(4):
        rot = np.rot90(stamp, k)
        suffix = "" if k == 0 else f"_rot{k * 90}"
        candidates.append((f"id{suffix}", rot))
        candidates.append((f"mirror{suffix}", np.fliplr(rot)))

    variants: list[StampVariant] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for name, candidate in candidates:
        key = (tuple(int(x) for x in candidate.shape), candidate.tobytes())
        if key not in seen and np.any(candidate == 1) and np.any(candidate == 4):
            seen.add(key)
            variants.append(StampVariant(name, candidate.copy()))
    return variants


def _placement_slices(top: int, left: int, height: int, width: int, grid_shape: tuple[int, int]) -> tuple[slice, slice, slice, slice] | None:
    rows, cols = grid_shape
    grid_r0 = max(0, top)
    grid_c0 = max(0, left)
    grid_r1 = min(rows, top + height)
    grid_c1 = min(cols, left + width)
    if grid_r0 >= grid_r1 or grid_c0 >= grid_c1:
        return None
    stamp_r0 = grid_r0 - top
    stamp_c0 = grid_c0 - left
    stamp_r1 = stamp_r0 + (grid_r1 - grid_r0)
    stamp_c1 = stamp_c0 + (grid_c1 - grid_c0)
    return (
        slice(grid_r0, grid_r1),
        slice(grid_c0, grid_c1),
        slice(stamp_r0, stamp_r1),
        slice(stamp_c0, stamp_c1),
    )


def _matches_variant_at(grid: Grid, variant: StampVariant, top: int, left: int) -> bool:
    h, w = variant.grid.shape
    slices = _placement_slices(top, left, h, w, grid.shape)
    if slices is None:
        return False
    hole_rows, hole_cols = np.where(variant.hole_mask)
    for stamp_r, stamp_c in zip(hole_rows, hole_cols, strict=True):
        grid_r = top + int(stamp_r)
        grid_c = left + int(stamp_c)
        if grid_r < 0 or grid_r >= grid.shape[0] or grid_c < 0 or grid_c >= grid.shape[1]:
            return False
        if grid[grid_r, grid_c] != 4:
            return False

    gr, gc, sr, sc = slices
    patch = grid[gr, gc]
    visible = variant.grid[sr, sc]
    fill = visible == 1
    if int(np.sum(fill)) == 0:
        return False
    if not np.all((patch[fill] == 0) | (patch[fill] == 1)):
        return False
    if np.all(patch[fill] == 1):
        return False
    return True


def _apply_variant_at(base: Grid, out: Grid, variant: StampVariant, top: int, left: int) -> None:
    h, w = variant.grid.shape
    slices = _placement_slices(top, left, h, w, base.shape)
    if slices is None:
        return
    gr, gc, sr, sc = slices
    patch = base[gr, gc]
    visible = variant.grid[sr, sc]
    fill = visible == 1
    view = out[gr, gc]
    view[fill & (patch == 0)] = 1


def _candidate_placements(base: Grid, variant: StampVariant) -> set[tuple[int, int]]:
    placements: set[tuple[int, int]] = set()
    grid_h, grid_w = base.shape
    hole_rows, hole_cols = np.where(variant.hole_mask)
    grid_holes = list(zip(*np.where(base == 4), strict=True))
    for grid_r, grid_c in grid_holes:
        for stamp_r, stamp_c in zip(hole_rows, hole_cols, strict=True):
            top = int(grid_r) - int(stamp_r)
            left = int(grid_c) - int(stamp_c)
            if top < -variant.grid.shape[0] + 1 or top >= grid_h:
                continue
            if left < -variant.grid.shape[1] + 1 or left >= grid_w:
                continue
            placements.add((top, left))
    return placements


def solve_task191(grid: Grid) -> Grid:
    """Complete every 4-marked occurrence of the source stamp.

    A task191 input contains one source stamp made from color 1 with color-4
    holes. Other groups of 4s are incomplete copies of that stamp, possibly
    rotated/reflected and possibly clipped by an edge. The output keeps all
    existing cells and fills the missing color-1 cells around each matching
    group of holes.
    """

    base = np.asarray(grid, dtype=np.int64)
    out = base.copy()
    variants = stamp_variants(source_stamp(base))

    for variant in variants:
        for top, left in _candidate_placements(base, variant):
            if _matches_variant_at(base, variant, top, left):
                _apply_variant_at(base, out, variant, top, left)
    return out


def visible_summary() -> tuple[int, int, list[tuple[str, int, int, int, int]]]:
    failures: list[tuple[str, int, int, int, int]] = []
    examples = load_labeled_examples()
    for split, index, input_grid, expected in examples:
        actual = solve_task191(input_grid)
        if not np.array_equal(actual, expected):
            extra = int(np.sum((actual == 1) & (expected == 0)))
            missed = int(np.sum((actual == 0) & (expected == 1)))
            failures.append((split, index, int(np.sum(actual != expected)), extra, missed))
    return len(examples) - len(failures), len(examples), failures


def rotate_contract(k: int) -> InvarianceContract:
    def transform(grid: Grid) -> Grid:
        return np.rot90(np.asarray(grid, dtype=np.int64), k).copy()

    return InvarianceContract(
        name=f"rot{k * 90}",
        input_transform=transform,
        expected_output_transform=transform,
        description="Rotating the whole canvas should rotate the completed stamps.",
    )


def mirror_contract() -> InvarianceContract:
    def transform(grid: Grid) -> Grid:
        return np.fliplr(np.asarray(grid, dtype=np.int64)).copy()

    return InvarianceContract(
        name="mirror_lr",
        input_transform=transform,
        expected_output_transform=transform,
        description="Mirroring the whole canvas should mirror the completed stamps.",
    )


def color_swap_contract() -> InvarianceContract:
    table = np.arange(10, dtype=np.int64)
    table[2] = 3
    table[3] = 2
    table[5] = 6
    table[6] = 5

    def transform(grid: Grid) -> Grid:
        arr = np.asarray(grid, dtype=np.int64)
        return table[arr].copy()

    return InvarianceContract(
        name="irrelevant_color_swap_2356",
        input_transform=transform,
        expected_output_transform=transform,
        description="Colors not used by the rule may be permuted without changing the geometry.",
    )


def run_pseudo_hidden() -> list:
    raw = load_labeled_examples()
    # Use the hand-authored examples plus a deterministic spread of generated
    # cases, including examples that exercise clipped placements.
    selected = raw[:5] + [raw[5 + i] for i in (0, 1, 6, 8, 35, 60, 124, 193, 206, 261)]
    examples = [
        ArcExample(input_grid=inp, output_grid=out, label=f"{split}{index}")
        for split, index, inp, out in selected
    ]
    contracts = [rotate_contract(1), rotate_contract(2), rotate_contract(3), mirror_contract(), color_swap_contract()]
    return run_contracts(solve_task191, examples, contracts, include_original=True)


def print_debug_example(split: str, index: int) -> None:
    for ex_split, ex_index, input_grid, expected in load_labeled_examples():
        if ex_split == split and ex_index == index:
            actual = solve_task191(input_grid)
            print(f"{split} {index}")
            print("input:")
            print("\n".join(grid_to_lines(input_grid)))
            print("expected:")
            print("\n".join(grid_to_lines(expected)))
            print("actual:")
            print("\n".join(grid_to_lines(actual)))
            print("diff expected/actual:")
            diff = np.where(expected == actual, expected, 9)
            print("\n".join(grid_to_lines(diff)))
            return
    raise SystemExit(f"example not found: {split} {index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", nargs=2, metavar=("SPLIT", "INDEX"))
    args = parser.parse_args()

    if args.debug:
        print_debug_example(args.debug[0], int(args.debug[1]))
        return

    passed, total, failures = visible_summary()
    print(f"visible_pass={passed}/{total}")
    if failures:
        print("visible_failures:")
        for failure in failures[:40]:
            print(f"  {failure}")
        if len(failures) > 40:
            print(f"  ... {len(failures) - 40} more")

    results = run_pseudo_hidden()
    print(f"pseudo_hidden={summarize_results(results)}")
    failure_text = format_failures(results, limit=40)
    if failure_text:
        print("pseudo_hidden_failures:")
        print(failure_text)


if __name__ == "__main__":
    main()
