from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from neurogolf_local import all_examples


def connected_components(arr: np.ndarray, predicate) -> list[dict]:
    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not predicate(arr[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and predicate(arr[nx, ny]):
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            rows = [x for x, _ in cells]
            cols = [y for _, y in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            comps.append({'cells': cells, 'bbox': (r0, c0, r1, c1), 'size': len(cells)})
    return comps


def component_mask(cells: list, bbox: tuple) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.int8)
    for r, c in cells:
        mask[r - r0, c - c0] = 1
    return mask


def color2_components(inp: np.ndarray) -> list[dict]:
    comps = connected_components(inp, lambda v: v == 2)
    for comp in comps:
        comp['mask'] = component_mask(comp['cells'], comp['bbox'])
    return comps


def nonzero_components(inp: np.ndarray) -> list[dict]:
    comps = connected_components(inp, lambda v: v != 0)
    for comp in comps:
        r0, c0, r1, c1 = comp['bbox']
        comp['patch'] = inp[r0:r1 + 1, c0:c1 + 1].copy()
    return comps


def hole_components(main_mask: np.ndarray) -> list[dict]:
    holes = connected_components(main_mask.astype(np.int8), lambda v: v == 0)
    for hole in holes:
        hole['mask'] = component_mask(hole['cells'], hole['bbox'])
    return holes


def patch_orientations(patch: np.ndarray) -> list[np.ndarray]:
    variants = []
    for rot in range(4):
        turned = np.rot90(patch, rot)
        for flip in (False, True):
            oriented = np.fliplr(turned) if flip else turned
            if not any(oriented.shape == v.shape and np.array_equal(oriented, v) for v in variants):
                variants.append(oriented.copy())
    return variants


def patch_two_components(patch: np.ndarray) -> list[tuple[np.ndarray, tuple]]:
    out = []
    for comp in connected_components(patch, lambda v: v == 2):
        mask = component_mask(comp['cells'], comp['bbox'])
        out.append((mask, (comp['bbox'][0], comp['bbox'][1])))
    return out


def non2_stats(ori: np.ndarray, sr: int, sc: int, mm: np.ndarray) -> tuple[int, int]:
    on_main = 0
    in_holes = 0
    for dr in range(ori.shape[0]):
        for dc in range(ori.shape[1]):
            if ori[dr, dc] != 2:
                if mm[sr + dr, sc + dc] == 0:
                    in_holes += 1
                else:
                    on_main += 1
    return on_main, in_holes


def find_holes_covered(ori: np.ndarray, sr: int, sc: int, holes: list, main_mask: np.ndarray) -> set:
    covered = set()
    for hi, h in enumerate(holes):
        hr0, hc0, hr1, hc1 = h['bbox']
        hm = h['mask']
        for dr in range(ori.shape[0]):
            for dc in range(ori.shape[1]):
                if ori[dr, dc] == 2:
                    cr, cc = sr + dr, sc + dc
                    if hr0 <= cr <= hr1 and hc0 <= cc <= hc1 and hm[cr - hr0, cc - hc0] == 1:
                        covered.add(hi)
    return covered


def center_dist(mr0: int, mc0: int, ph: int, pw: int) -> float:
    return abs(mr0 - (ph - 1) / 2) + abs(mc0 - (pw - 1) / 2)


def solve_task233(inp: np.ndarray) -> np.ndarray:
    two_comps = color2_components(inp)
    if not two_comps:
        return np.zeros((1, 1), dtype=np.int8)

    main = max(two_comps, key=lambda c: c['size'])
    r0, c0, r1, c1 = main['bbox']

    hc = r1 - r0 + 1
    wc = c1 - c0 + 1
    canvas = np.full((hc, wc), 2, dtype=np.int8)
    main_mask = np.zeros_like(canvas, dtype=np.int8)
    for r, c in main['cells']:
        main_mask[r - r0, c - c0] = 1

    holes = [h for h in hole_components(main_mask) if h['size'] < hc * wc // 2]
    n_h = len(holes)

    main_cells_set = set(main['cells'])
    patches = []
    for comp in nonzero_components(inp):
        if set(comp['cells']) == main_cells_set:
            continue
        vals = set(np.unique(comp['patch']).tolist())
        if 2 not in vals or vals <= {2}:
            continue
        patches.append(comp)

    patches_sorted = sorted(patches, key=lambda x: x['size'])
    n_p = len(patches_sorted)

    def hole_cover(ori: np.ndarray, sr: int, sc: int) -> int:
        c2 = (ori == 2).astype(np.int8)
        er = sr + ori.shape[0]
        ec = sc + ori.shape[1]
        return int(np.sum(c2 & (main_mask[sr:er, sc:ec] == 0).astype(np.int8)))

    used_patches = set()
    filled_holes = set()

    # --- Phase 1: type-1 matching (global greedy) ---
    candidates_t1 = []
    for hi, hole in enumerate(holes):
        h_r0, h_c0, _, _ = hole['bbox']
        hole_cr = r0 + (hole['bbox'][0] + hole['bbox'][2]) / 2
        hole_cc = c0 + (hole['bbox'][1] + hole['bbox'][3]) / 2
        for ci, comp in enumerate(patches_sorted):
            cr0, cc0, cr1, cc1 = comp['bbox']
            patch_cr = (cr0 + cr1) / 2
            patch_cc = (cc0 + cc1) / 2
            sd = abs(patch_cc - hole_cc) * 4 + abs(patch_cr - hole_cr)
            for ori in patch_orientations(comp['patch']):
                for mask2, (mr0, mc0) in patch_two_components(ori):
                    if mask2.shape != hole['mask'].shape or not np.array_equal(mask2, hole['mask']):
                        continue
                    sr = h_r0 - mr0
                    sc = h_c0 - mc0
                    er = sr + ori.shape[0]
                    ec = sc + ori.shape[1]
                    if sr < 0 or sc < 0 or er > hc or ec > wc:
                        continue
                    hc_val = hole_cover(ori, sr, sc)
                    nm, nh = non2_stats(ori, sr, sc, main_mask)
                    margin = min(sr, sc, hc - er, wc - ec)
                    cov = find_holes_covered(ori, sr, sc, holes, main_mask)
                    cd = center_dist(mr0, mc0, ori.shape[0], ori.shape[1])
                    candidates_t1.append((
                        (hc_val, nm, -nh, -sd, margin, -ci, -cd),
                        hi, ori, sr, sc, ci, cov
                    ))

    candidates_t1.sort(key=lambda x: x[0], reverse=True)

    for key, hi, ori, sr, sc, ci, cov in candidates_t1:
        if hi in filled_holes or ci in used_patches:
            continue
        if cov & filled_holes:
            continue
        filled_holes.add(hi)
        filled_holes.update(cov)
        used_patches.add(ci)
        region = canvas[sr:sr + ori.shape[0], sc:sc + ori.shape[1]]
        region[ori != 2] = ori[ori != 2]

    # --- Phase 2: type-0 matching (remaining holes) ---
    candidates_t0 = []
    for hi, hole in enumerate(holes):
        if hi in filled_holes:
            continue
        h_r0, h_c0, _, _ = hole['bbox']
        hole_cr = r0 + (hole['bbox'][0] + hole['bbox'][2]) / 2
        hole_cc = c0 + (hole['bbox'][1] + hole['bbox'][3]) / 2
        for ci, comp in enumerate(patches_sorted):
            if ci in used_patches:
                continue
            cr0, cc0, cr1, cc1 = comp['bbox']
            patch_cr = (cr0 + cr1) / 2
            patch_cc = (cc0 + cc1) / 2
            sd = abs(patch_cc - hole_cc) * 4 + abs(patch_cr - hole_cr)
            for ori in patch_orientations(comp['patch']):
                if ori.shape != hole['mask'].shape:
                    continue
                sr, sc = h_r0, h_c0
                if sr + ori.shape[0] > hc or sc + ori.shape[1] > wc:
                    continue
                hc_val = hole_cover(ori, sr, sc)
                if hc_val == 0:
                    continue
                nm, nh = non2_stats(ori, sr, sc, main_mask)
                cov = find_holes_covered(ori, sr, sc, holes, main_mask)
                if cov & filled_holes:
                    continue
                margin = min(sr, sc, hc - (sr + ori.shape[0]), wc - (sc + ori.shape[1]))
                cd = center_dist(0, 0, ori.shape[0], ori.shape[1])
                candidates_t0.append((
                    (hc_val, nm, -nh, -sd, margin, -ci, -cd),
                    hi, ori, sr, sc, ci, cov
                ))

    candidates_t0.sort(key=lambda x: x[0], reverse=True)

    for key, hi, ori, sr, sc, ci, cov in candidates_t0:
        if hi in filled_holes or ci in used_patches:
            continue
        if cov & filled_holes:
            continue
        filled_holes.add(hi)
        filled_holes.update(cov)
        used_patches.add(ci)
        region = canvas[sr:sr + ori.shape[0], sc:sc + ori.shape[1]]
        region[ori != 2] = ori[ori != 2]

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--task-id", type=int, default=233)
    args = parser.parse_args()

    comp_dir = pathlib.Path(__file__).resolve().parent.parent / "data" / "neurogolf-2026" / "raw"
    examples = all_examples(args.task_id, comp_dir)

    correct = 0
    wrong = 0
    for ex in examples:
        inp = np.array(ex["input"], dtype=np.int8)
        out = np.array(ex["output"], dtype=np.int8)
        pred = solve_task233(inp)
        if np.array_equal(pred, out):
            correct += 1
        else:
            wrong += 1

    print(f"Pass: {correct}/{correct + wrong}")
    if args.output is not None:
        import json
        args.output.write_text(json.dumps({"pass": correct, "total": correct + wrong}, indent=2))


if __name__ == "__main__":
    main()
