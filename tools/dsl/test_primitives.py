"""Equivalence tests: for each primitive, python_ref(x) == onnx_run(x) on
a fixed set of random grids spanning the H/W shape space (1..15).

Run:  .venv/bin/python -m tools.dsl.test_primitives
"""
from __future__ import annotations

import sys

import numpy as np
import onnxruntime as ort

from .compiler import DslOp, DslProgram, compile_to_onnx
from .primitives import REF_FUNCS


GRID_SHAPE = (1, 10, 30, 30)


def encode(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(GRID_SHAPE, dtype=np.float32)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            if 0 <= v < 10:
                out[0, v, r, c] = 1.0
    return out


def decode(tensor: np.ndarray, h: int, w: int) -> np.ndarray:
    """Decode the top-left h x w region into [h, w] int grid. Cells with
    no channel set are decoded as 0 (caller is responsible for using a target
    shape that matches the actual content)."""
    t = tensor > 0.0
    out = np.zeros((h, w), dtype=np.int64)
    for r in range(h):
        for c in range(w):
            colors = [k for k in range(10) if t[0, k, r, c]]
            if not colors:
                out[r, c] = 0
            elif len(colors) == 1:
                out[r, c] = colors[0]
            else:
                out[r, c] = -1  # invalid (will trip the equality check)
    return out


def run_onnx(model, grid: np.ndarray) -> np.ndarray:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    inp = encode(grid)
    return sess.run(["output"], {"input": inp})[0]


def _random_grids(rng, n=8, h_range=(2, 16), w_range=(2, 16)) -> list[np.ndarray]:
    grids = []
    for _ in range(n):
        h = int(rng.integers(h_range[0], h_range[1]))
        w = int(rng.integers(w_range[0], w_range[1]))
        grids.append(rng.integers(0, 10, size=(h, w), dtype=np.int64))
    return grids


def _check_general(name, params, ref_fn, grids):
    """Compile a single-op program, run on each grid, compare decoded output
    to `ref_fn(grid, **params)` cell-by-cell within the ref's bbox.

    - Allows the ref to have a *different* shape than the input (transpose,
      bbox_crop, rotate_static, count_color).
    - Skips grids where the input doesn't fit the [30,30] envelope.
    - For ref shape (0, 0) (e.g. bbox_crop with target absent) only checks
      that the ONNX output has no positive cells.
    """
    program = DslProgram.of([DslOp(name=name, params=params)])
    model = compile_to_onnx(program)
    for grid in grids:
        gh, gw = grid.shape
        if gh > 30 or gw > 30:
            continue
        ref = ref_fn(grid, **params)
        out_t = run_onnx(model, grid)
        rh, rw = ref.shape
        if rh > 30 or rw > 30:
            continue
        if rh == 0 or rw == 0:
            if (out_t > 0.0).any():
                return False, f"  expected empty output for grid {grid.shape}, got positive cells"
            continue
        decoded = decode(out_t, rh, rw)
        outside_any = (out_t > 0.0)[0, :, rh:, :].any() or (out_t > 0.0)[0, :, :, rw:].any()
        if outside_any:
            return False, f"  ONNX produced cells outside ref bbox h={rh},w={rw}"
        if not np.array_equal(decoded, ref):
            return False, (
                f"  mismatch on input shape {grid.shape} -> ref shape {ref.shape}\n"
                f"  INPUT:\n{grid}\n  REF:\n{ref}\n  GOT:\n{decoded}"
            )
    return True, ""


def _check_shape_preserving(name, params, ref_fn):
    rng = np.random.default_rng(42)
    return _check_general(name, params, ref_fn, _random_grids(rng))


def _bbox_crop_grids(rng, n=8) -> list[np.ndarray]:
    """Grids guaranteed to contain a contiguous target-color (=4) bbox."""
    grids = []
    for _ in range(n):
        h = int(rng.integers(4, 16))
        w = int(rng.integers(4, 16))
        grid = rng.integers(0, 10, size=(h, w), dtype=np.int64)
        # Stamp a contiguous block of color 4 somewhere
        bh = int(rng.integers(1, max(2, h // 2 + 1)))
        bw = int(rng.integers(1, max(2, w // 2 + 1)))
        r0 = int(rng.integers(0, h - bh + 1))
        c0 = int(rng.integers(0, w - bw + 1))
        # zero out the rest of color 4 first so the bbox is exactly the stamped block
        grid[grid == 4] = 5
        grid[r0:r0 + bh, c0:c0 + bw] = 4
        grids.append(grid)
    return grids


def main() -> int:
    failures = 0
    # ---------- v0 primitives (shape-preserving / well-behaved) ----------
    v0_cases = [
        ("identity", {}),
        ("transpose", {}),
        ("flip_h", {}),
        ("flip_v", {}),
        ("rotate180", {}),
        ("rotate90", {}),
        ("swap_colors", {"c1": 1, "c2": 8}),
        ("swap_colors", {"c1": 5, "c2": 8}),
        ("swap_colors", {"c1": 0, "c2": 7}),
        ("replace_color", {"src": 6, "dst": 2}),
        ("replace_color", {"src": 7, "dst": 5}),
        ("replace_color", {"src": 1, "dst": 2}),
    ]
    for name, params in v0_cases:
        ok, msg = _check_shape_preserving(name, params, REF_FUNCS[name])
        status = "OK" if ok else "FAIL"
        param_str = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
        print(f"  {status}  {name}({param_str})")
        if not ok:
            print(msg)
            failures += 1

    total_count = len(v0_cases)

    # ---------- v1: static rotates ----------
    rng = np.random.default_rng(91)
    for (h, w) in [(3, 3), (5, 5), (4, 7), (7, 4), (1, 1)]:
        # grids of exactly (h, w)
        grids = [rng.integers(0, 10, size=(h, w), dtype=np.int64) for _ in range(4)]
        for name in ("rotate180_static", "rotate90_static"):
            ok, msg = _check_general(name, {"h": h, "w": w}, REF_FUNCS[name], grids)
            status = "OK" if ok else "FAIL"
            print(f"  {status}  {name}(h={h},w={w})")
            if not ok:
                print(msg)
                failures += 1
            total_count += 1

    # ---------- v1: count_color, dominant_color, fill_bg ----------
    rng = np.random.default_rng(11)
    grids = _random_grids(rng)
    for c in (0, 4, 9):
        ok, msg = _check_general("count_color", {"c": c}, REF_FUNCS["count_color"], grids)
        status = "OK" if ok else "FAIL"
        print(f"  {status}  count_color(c={c})")
        if not ok:
            print(msg)
            failures += 1
        total_count += 1

    ok, msg = _check_general("dominant_color", {}, REF_FUNCS["dominant_color"], grids)
    print(f"  {'OK' if ok else 'FAIL'}  dominant_color()")
    if not ok:
        print(msg)
        failures += 1
    total_count += 1

    for bg in (3, 7):
        ok, msg = _check_general("fill_bg", {"bg_color": bg}, REF_FUNCS["fill_bg"], grids)
        status = "OK" if ok else "FAIL"
        print(f"  {status}  fill_bg(bg_color={bg})")
        if not ok:
            print(msg)
            failures += 1
        total_count += 1

    # ---------- v1: bbox_crop with contiguous target ----------
    rng = np.random.default_rng(23)
    bc_grids = _bbox_crop_grids(rng)
    ok, msg = _check_general("bbox_crop", {"target_color": 4}, REF_FUNCS["bbox_crop"], bc_grids)
    print(f"  {'OK' if ok else 'FAIL'}  bbox_crop(target_color=4)")
    if not ok:
        print(msg)
        failures += 1
    total_count += 1

    # ---------- P1: tile_h / tile_v ----------
    rng = np.random.default_rng(202)
    tile_grids = _random_grids(rng, n=6)
    for n in (2, 3):
        ok, msg = _check_general("tile_h", {"n": n}, REF_FUNCS["tile_h"], tile_grids)
        status = "OK" if ok else "FAIL"
        print(f"  {status}  tile_h(n={n})")
        if not ok:
            print(msg)
            failures += 1
        total_count += 1
    for n in (2, 3):
        ok, msg = _check_general("tile_v", {"n": n}, REF_FUNCS["tile_v"], tile_grids)
        status = "OK" if ok else "FAIL"
        print(f"  {status}  tile_v(n={n})")
        if not ok:
            print(msg)
            failures += 1
        total_count += 1

    # ---------- P2: dominant_color BOOL pipeline ----------
    dc_grids = _random_grids(np.random.default_rng(303), n=6)
    ok, msg = _check_general("dominant_color", {}, REF_FUNCS["dominant_color"], dc_grids)
    print(f"  {'OK' if ok else 'FAIL'}  dominant_color (BOOL pipeline)")
    if not ok:
        print(msg)
        failures += 1
    total_count += 1

    # ---------- P3: largest_blob ----------
    def _largest_blob_grids(rng, n=6) -> list[np.ndarray]:
        grids = []
        for _ in range(n):
            h = int(rng.integers(4, 12))
            w = int(rng.integers(4, 12))
            grid = rng.integers(0, 6, size=(h, w), dtype=np.int64)
            grids.append(grid)
        return grids
    lb_rng = np.random.default_rng(404)
    lb_grids = _largest_blob_grids(lb_rng)
    for tc in (0, 3):
        ok, msg = _check_general("largest_blob", {"target_color": tc}, REF_FUNCS["largest_blob"], lb_grids)
        status = "OK" if ok else "FAIL"
        print(f"  {status}  largest_blob(target_color={tc})")
        if not ok:
            print(msg)
            failures += 1
        total_count += 1

    # ---------- Batch 1: select_channel ----------
    rng = np.random.default_rng(42)
    sc_grids = _random_grids(rng, n=6)
    for cond_ch in (0, 4, 9):
        ok, msg = _check_general("select_channel", {"cond_ch": cond_ch, "if_ch": 1, "else_ch": 7},
                                 REF_FUNCS["select_channel"], sc_grids)
        print(f"  {'OK' if ok else 'FAIL'}  select_channel(cond_ch={cond_ch},if_ch=1,else_ch=7)")
        if not ok: print(msg); failures += 1
        total_count += 1

    # ---------- Batch 1: fill ----------
    rng = np.random.default_rng(42)
    fi_grids = _random_grids(rng, n=6)
    for c in (0, 4, 9):
        ok, msg = _check_general("fill", {"c": c}, REF_FUNCS["fill"], fi_grids)
        print(f"  {'OK' if ok else 'FAIL'}  fill(c={c})")
        if not ok: print(msg); failures += 1
        total_count += 1

    # ---------- Batch 1: shift_down ----------
    rng = np.random.default_rng(42)
    sd_grids = _random_grids(rng, n=6)
    for k in (1, 2, 3):
        ok, msg = _check_general("shift_down", {"k": k}, REF_FUNCS["shift_down"], sd_grids)
        print(f"  {'OK' if ok else 'FAIL'}  shift_down(k={k})")
        if not ok: print(msg); failures += 1
        total_count += 1

    # ---------- Batch 1: shift_right ----------
    rng = np.random.default_rng(42)
    sr_grids = _random_grids(rng, n=6)
    for k in (1, 2, 3):
        ok, msg = _check_general("shift_right", {"k": k}, REF_FUNCS["shift_right"], sr_grids)
        print(f"  {'OK' if ok else 'FAIL'}  shift_right(k={k})")
        if not ok: print(msg); failures += 1
        total_count += 1

    # ---------- Batch 1: crop ----------
    rng = np.random.default_rng(42)
    cr_grids = _random_grids(rng, n=6)
    for (r0, c0, h, w) in [(0, 0, 3, 3), (1, 2, 4, 5), (2, 1, 3, 4)]:
        ok, msg = _check_general("crop", {"r0": r0, "c0": c0, "h": h, "w": w}, REF_FUNCS["crop"], cr_grids)
        print(f"  {'OK' if ok else 'FAIL'}  crop(r0={r0},c0={c0},h={h},w={w})")
        if not ok: print(msg); failures += 1
        total_count += 1

    # ---------- Batch 2: mask_foreground ----------
    rng = np.random.default_rng(505)
    mf_grids = _random_grids(rng, n=6)
    ok, msg = _check_general("mask_foreground", {}, REF_FUNCS["mask_foreground"], mf_grids)
    print(f"  {'OK' if ok else 'FAIL'}  mask_foreground()")
    if not ok: print(msg); failures += 1
    total_count += 1

    # ---------- Batch 2: remove_color ----------
    rng = np.random.default_rng(606)
    rc_grids = _random_grids(rng, n=6)
    for tc in (0, 4, 7):
        ok, msg = _check_general("remove_color", {"target_color": tc}, REF_FUNCS["remove_color"], rc_grids)
        status = "OK" if ok else "FAIL"
        print(f"  {status}  remove_color(target_color={tc})")
        if not ok: print(msg); failures += 1
        total_count += 1

    # ---------- Batch 2: thicken ----------
    rng = np.random.default_rng(707)
    th_grids = _random_grids(rng, n=6)
    for tc in (1, 5, 8):
        ok, msg = _check_general("thicken", {"target_color": tc}, REF_FUNCS["thicken"], th_grids)
        status = "OK" if ok else "FAIL"
        print(f"  {status}  thicken(target_color={tc})")
        if not ok: print(msg); failures += 1
        total_count += 1

    # ---------- Batch 2: hollow_out ----------
    rng = np.random.default_rng(808)
    ho_grids = _random_grids(rng, n=6)
    for tc in (2, 6):
        ok, msg = _check_general("hollow_out", {"target_color": tc}, REF_FUNCS["hollow_out"], ho_grids)
        status = "OK" if ok else "FAIL"
        print(f"  {status}  hollow_out(target_color={tc})")
        if not ok: print(msg); failures += 1
        total_count += 1

    # ---------- Batch 2: flood_fill ----------
    rng = np.random.default_rng(909)
    ff_grids = _random_grids(rng, n=6)
    for (tc, fc) in [(1, 5), (3, 7), (8, 2)]:
        ok, msg = _check_general("flood_fill", {"target_color": tc, "fill_color": fc}, REF_FUNCS["flood_fill"], ff_grids)
        status = "OK" if ok else "FAIL"
        print(f"  {status}  flood_fill(target_color={tc},fill_color={fc})")
        if not ok: print(msg); failures += 1
        total_count += 1

    # ---------- Batch 2: trim_border ----------
    rng = np.random.default_rng(1001)
    tb_grids = _random_grids(rng, n=6)
    for k in (1, 2):
        ok, msg = _check_general("trim_border", {"k": k}, REF_FUNCS["trim_border"], tb_grids)
        status = "OK" if ok else "FAIL"
        print(f"  {status}  trim_border(k={k})")
        if not ok: print(msg); failures += 1
        total_count += 1

    # ---------- composition tests ----------
    rng = np.random.default_rng(7)
    program = DslProgram.of([
        DslOp("swap_colors", {"c1": 0, "c2": 1}),
        DslOp("swap_colors", {"c1": 0, "c2": 1}),
    ])
    model = compile_to_onnx(program)
    composition_ok = True
    for grid in _random_grids(rng, n=4):
        out_t = run_onnx(model, grid)
        h, w = grid.shape
        decoded = decode(out_t, h, w)
        if not np.array_equal(decoded, grid):
            composition_ok = False
            print(f"  FAIL  swap | swap composition not identity on shape {grid.shape}")
            break
    if composition_ok:
        print("  OK  swap | swap == identity (composition test)")
    else:
        failures += 1
    total_count += 1

    print(f"\n{total_count - failures}/{total_count} primitive equivalence tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
