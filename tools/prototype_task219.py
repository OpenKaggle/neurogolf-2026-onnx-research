"""task219 prototype.

Rule (derived from train examples):
  Input has color 0 (background) and 8 (markers).
  Markers organize into row-groups (consecutive rows containing 8s, separated by all-0 rows).
  Template = row-group whose 8s reach col w-1 if any; else widest-extent group.
  For each other row-group (fragment) starting at row F:
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples


def find_row_groups(inp: np.ndarray) -> list[tuple[int, int]]:
    h, w = inp.shape
    has_8 = (inp == 8).any(axis=1)
    groups = []
    start = None
    for r in range(h):
        if has_8[r]:
            if start is None:
                start = r
        else:
            if start is not None:
                groups.append((start, r - 1))
                start = None
    if start is not None:
        groups.append((start, h - 1))
    return groups


def group_extent(inp: np.ndarray, start: int, end: int) -> int:
    """Max col index with value 8 in inp rows [start..end] (inclusive)."""
    sub = inp[start:end + 1]
    cols_with_8 = (sub == 8).any(axis=0)
    if not cols_with_8.any():
        return -1
    return int(np.argwhere(cols_with_8).max())


def row_max8(row: np.ndarray) -> int:
    cols = np.where(row == 8)[0]
    return int(cols.max()) if cols.size else -1


def row_count8(row: np.ndarray) -> int:
    return int((row == 8).sum())


def row_island_count(cols: tuple[int, ...]) -> int:
    if not cols:
        return 0
    islands = 1
    for a, b in zip(cols, cols[1:]):
        if b != a + 1:
            islands += 1
    return islands


def choose_template(inp: np.ndarray, groups: list[tuple[int, int]]) -> tuple[int, int] | None:
    w = inp.shape[1]
    template = None
    best_ext = -1
    for g in groups:
        ext = group_extent(inp, *g)
        if ext == w - 1:
            return g
        if ext > best_ext:
            best_ext = ext
            template = g
    return template


def row_cols(inp: np.ndarray, row: int, limit: int | None = None) -> tuple[int, ...]:
    cols = np.where(inp[row] == 8)[0].tolist()
    if limit is not None:
        cols = [col for col in cols if col <= limit]
    return tuple(cols)


def row_tail_kind(cols: tuple[int, ...], width_limit: int) -> str:
    tail = [col for col in cols if col > width_limit]
    if len(tail) >= 2 and all(b - a == 1 for a, b in zip(tail, tail[1:])):
        return "solid"
    if len(tail) >= 2 and all(b - a == 2 for a, b in zip(tail, tail[1:])):
        return "alt"
    return "direct"


def best_fragment_mapping(
    template_rows: list[tuple[int, ...]],
    fragment_rows: list[tuple[int, ...]],
    width_limit: int,
) -> tuple[int, ...]:
    best = None
    g = len(template_rows)
    h = len(fragment_rows)
    for mapping in itertools.combinations(range(g), h):
        penalty = 0
        for frag_row, template_index in zip(fragment_rows, mapping):
            tmpl_row = tuple(col for col in template_rows[template_index] if col <= width_limit)
            # Visible fragments are often a compressed left-prefix of a template row
            # rather than a literal clipped subset. Row length and island count are
            # more reliable than exact column equality for choosing vertical alignment.
            penalty += 20 * abs(len(frag_row) - len(tmpl_row))
            penalty += 20 * abs(row_island_count(frag_row) - row_island_count(tmpl_row))
        # When two alignments look equally plausible under the coarse row-shape
        # score, prefer the later template slice. Short fragments more often
        # correspond to the lower part of the exemplar than to its top rows.
        key = (penalty, tuple(-idx for idx in mapping))
        if best is None or key < best[0]:
            best = (key, mapping)
    assert best is not None
    return best[1]


def solve_task219(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    out = inp.copy()
    groups = find_row_groups(inp)
    if not groups:
        return out
    # Template: prefer full-width group, else widest group.
    template = choose_template(inp, groups)
    if template is None:
        return out
    R, R_end = template
    G = R_end - R + 1
    template_rows = [row_cols(inp, row) for row in range(R, R_end + 1)]
    for g in groups:
        if g == template:
            continue
        F, F_end = g
        K_f = group_extent(inp, F, F_end)
        if K_f < 0:
            continue
        fragment_rows = [row_cols(inp, row, limit=K_f) for row in range(F, F_end + 1)]
        mapping = best_fragment_mapping(template_rows, fragment_rows, K_f)
        start_row = F - mapping[0]
        for i in range(G):
            r = start_row + i
            tr = R + i
            if r < 0 or r >= h or tr > R_end:
                continue
            for c in range(K_f + 1, w):
                if inp[tr, c] == 8:
                    out[r, c] = 1
        # Small-width alternating rows sometimes appear as compressed or shifted
        # versions of the exemplar prefix; in those cases the right-hand pattern
        # should be translated with the fragment rather than copied literally.
        if K_f <= 3:
            for local_i, r in enumerate(range(F, F_end + 1)):
                frag = fragment_rows[local_i]
                mapped = template_rows[mapping[local_i]]
                prefix = tuple(col for col in mapped if col <= K_f)
                if row_tail_kind(mapped, K_f) == "alt" and frag and prefix and frag != prefix:
                    if len(frag) < len(prefix):
                        delta = frag[0] - prefix[0]
                    else:
                        delta = frag[-1] - prefix[-1]
                    row_max = max(frag)
                    out[r, row_max + 1:] = 0
                    for col in mapped:
                        shifted = col + delta
                        if row_max < shifted < w:
                            out[r, shifted] = 1
                # A direct singleton just beyond K can already be absorbed into a
                # compressed contiguous fragment, in which case nothing should be
                # added to the right.
                tail = tuple(col for col in mapped if col > K_f)
                if tail == (K_f + 1,) and frag == tuple(range(K_f + 1)):
                    out[r, K_f + 1:] = 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--verbose-fails", type=int, default=3)
    args = parser.parse_args()
    examples = all_examples(219, pathlib.Path(args.comp_dir))
    passed = 0
    fails = []
    for i, ex in enumerate(examples):
        inp = np.array(ex["input"], dtype=np.int8)
        gt = np.array(ex["output"], dtype=np.int8)
        pred = solve_task219(inp)
        if np.array_equal(pred, gt):
            passed += 1
        else:
            fails.append(i)
    print(f"task219: {passed}/{len(examples)}")
    if fails:
        print(f"  first fail indices: {fails[:10]}")
        for idx in fails[:args.verbose_fails]:
            inp = np.array(examples[idx]["input"], dtype=np.int8)
            gt = np.array(examples[idx]["output"], dtype=np.int8)
            pred = solve_task219(inp)
            print(f"\n  --- fail [{idx}] shape {inp.shape} ---")
            print("  INPUT:")
            for r in inp:
                print("    " + " ".join(str(int(x)) for x in r))
            print("  GT:")
            for r in gt:
                print("    " + " ".join(str(int(x)) for x in r))
            print("  PRED:")
            for r in pred:
                print("    " + " ".join(str(int(x)) for x in r))
            diff = (pred != gt).astype(int)
            print("  DIFF (1=mismatch):")
            for r in diff:
                print("    " + " ".join(str(x) for x in r))


if __name__ == "__main__":
    main()
