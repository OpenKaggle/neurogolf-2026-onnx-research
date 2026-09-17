"""DSL primitives v0.

Every primitive contributes two artifacts:

1.  A pure-Python reference: `ref_<name>(grid: np.ndarray, **params) -> np.ndarray`.
    Operates on a *small* `[H, W]` integer grid (the natural grid shape, not the
    `[1,10,30,30]` competition tensor). Used by the unit tests.

2.  An ONNX subgraph builder: `build_<name>(prefix, in_name, out_name, **params)`.
    The subgraph reads a `[1,10,30,30]` FLOAT input named `in_name` and writes a
    `[1,10,30,30]` FLOAT output named `out_name`. The compiler chains these
    by setting `in_name` of op `i+1` to `out_name` of op `i`.

Design notes (matching v4_plus3 elite baselines):
- We keep the pipeline in FLOAT throughout; the scorer thresholds at `> 0.0`,
  so FLOAT in {0.0, 1.0} works identically to BOOL.
- Whenever possible, primitives are a SINGLE op that gathers from `input`
  directly into `output` (no [1,10,30,30] intermediates).
- Bounding-box-aware primitives compute `H` / `W` as INT64 scalars via
  ReduceSum on a small line-summary, then build a dynamic `[30]` index
  vector and Gather along the relevant spatial axis. All intermediates are
  small (scalars or `[30]` vectors); the only [1,10,30,30] tensor is `input`
  (excluded from memory) and `output` (excluded).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


H30 = 30
W30 = 30
NCHAN = 10


# ---------------------------------------------------------------------------
# Python reference implementations
# ---------------------------------------------------------------------------

def ref_identity(grid: np.ndarray) -> np.ndarray:
    return grid.copy()


def ref_transpose(grid: np.ndarray) -> np.ndarray:
    return grid.T.copy()


def ref_flip_h(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=1).copy()


def ref_flip_v(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=0).copy()


def ref_rotate180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 2).copy()


def ref_rotate90(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 1).copy()


def ref_replace_color(grid: np.ndarray, src: int, dst: int) -> np.ndarray:
    out = grid.copy()
    out[grid == src] = dst
    return out


def ref_swap_colors(grid: np.ndarray, c1: int, c2: int) -> np.ndarray:
    out = grid.copy()
    out[grid == c1] = c2
    out[grid == c2] = c1
    return out


def ref_rotate180_static(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    """Static-shape rotate180 — only well-defined when grid.shape == (h, w)."""
    return np.rot90(grid, 2).copy()


def ref_rotate90_static(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    """Static-shape rotate90 ccw — only well-defined when grid.shape == (h, w)."""
    return np.rot90(grid, 1).copy()


def ref_count_color(grid: np.ndarray, c: int) -> np.ndarray:
    """Count occurrences of c, clipped to [0, 9]. Output shape (1, 1)."""
    cnt = int((grid == c).sum())
    cnt = min(cnt, 9)
    out = np.zeros((1, 1), dtype=np.int64)
    out[0, 0] = cnt
    return out


def ref_dominant_color(grid: np.ndarray) -> np.ndarray:
    """Fill every cell of the original h×w grid with the most-frequent color.

    Tie-breaking matches numpy/ONNX ArgMax: lowest channel index wins.
    """
    counts = np.bincount(grid.ravel(), minlength=10)[:10]
    dominant = int(np.argmax(counts))
    return np.full_like(grid, dominant)


def ref_fill_bg(grid: np.ndarray, bg_color: int) -> np.ndarray:
    """Replace every cell currently of color 0 (ARC background) with bg_color."""
    return ref_replace_color(grid, src=0, dst=bg_color)


def ref_bbox_crop(grid: np.ndarray, target_color: int) -> np.ndarray:
    """Crop the bounding box of `target_color` and return it top-left-aligned.

    If `target_color` is absent from the grid, returns a (0, 0) array.
    Bbox is the smallest enclosing axis-aligned rectangle; non-target cells
    *inside* the bbox are preserved.
    """
    rows = np.any(grid == target_color, axis=1)
    cols = np.any(grid == target_color, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((0, 0), dtype=grid.dtype)
    r0, r1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return grid[r0:r1 + 1, c0:c1 + 1].copy()


def ref_tile_h(grid: np.ndarray, n: int) -> np.ndarray:
    """Replicate grid horizontally by n times, clip to 30."""
    new_w = min(grid.shape[1] * n, W30)
    return np.tile(grid, (1, n))[:, :new_w]


def ref_tile_v(grid: np.ndarray, n: int) -> np.ndarray:
    """Replicate grid vertically by n times, clip to 30."""
    new_h = min(grid.shape[0] * n, H30)
    return np.tile(grid, (n, 1))[:new_h, :]


def ref_largest_blob(grid: np.ndarray, target_color: int) -> np.ndarray:
    """Find the largest 4-connected blob of `target_color` using
    scipy.ndimage.label. Returns grid with only that blob visible,
    all other cells zero."""
    target = (grid == target_color)
    if not target.any():
        return np.zeros_like(grid)
    try:
        from scipy.ndimage import label
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool)
        labeled, n_feat = label(target, structure=structure)
        if n_feat == 0:
            return np.zeros_like(grid)
        sizes = np.bincount(labeled.ravel())[1:]
        largest = np.argmax(sizes) + 1
        out = np.where(labeled == largest, target_color, 0)
        return out
    except ImportError:
        rows, cols = np.where(target)
        seed = (rows[0], cols[0])
        mask = np.zeros_like(grid, dtype=bool)
        stack = [seed]
        while stack:
            r, c = stack.pop()
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and not mask[r, c] and target[r, c]:
                mask[r, c] = True
                stack.append((r - 1, c))
                stack.append((r + 1, c))
                stack.append((r, c - 1))
                stack.append((r, c + 1))
        out = np.where(mask, target_color, 0)
        return out


def ref_select_channel(grid: np.ndarray, cond_ch: int, if_ch: int, else_ch: int) -> np.ndarray:
    """For each cell, if the cell's color == cond_ch, output if_ch, else output else_ch."""
    return np.where(grid == cond_ch, if_ch, else_ch)


def ref_fill(grid: np.ndarray, c: int) -> np.ndarray:
    """Ignore input, output a uniform grid filled with color c (same shape as input)."""
    return np.full_like(grid, c)


def ref_shift_down(grid: np.ndarray, k: int) -> np.ndarray:
    """Shift grid content down by k rows; zeros fill in from top."""
    out = np.zeros_like(grid)
    h = grid.shape[0]
    if k < h:
        out[k:, :] = grid[:h - k, :]
    return out


def ref_shift_right(grid: np.ndarray, k: int) -> np.ndarray:
    """Shift grid content right by k columns; zeros fill in from left."""
    out = np.zeros_like(grid)
    w = grid.shape[1]
    if k < w:
        out[:, k:] = grid[:, :w - k]
    return out


def ref_crop(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    """Extract h×w subgrid starting at (r0, c0)."""
    return grid[r0:r0 + h, c0:c0 + w].copy()


# ---------------------------------------------------------------------------
# Batch 2: mask_foreground, flood_fill, thicken, hollow_out, remove_color, trim_border
# ---------------------------------------------------------------------------


def ref_mask_foreground(grid: np.ndarray) -> np.ndarray:
    """Output binary mask where non-zero cells become color 1, zeros stay 0."""
    return np.where(grid > 0, 1, 0).astype(grid.dtype)


def ref_flood_fill(grid: np.ndarray, target_color: int, fill_color: int) -> np.ndarray:
    """Grow all target_color blobs outward via 8 iterations of 4-conn dilation,
    then fill the dilated region with fill_color."""
    mask = (grid == target_color)
    h, w = grid.shape
    for _ in range(8):
        dilated = mask.copy()
        if h > 1:
            dilated[1:, :] |= mask[:-1, :]
            dilated[:-1, :] |= mask[1:, :]
        if w > 1:
            dilated[:, 1:] |= mask[:, :-1]
            dilated[:, :-1] |= mask[:, 1:]
        mask = dilated
    out = np.where(mask, fill_color, grid).astype(grid.dtype)
    return out


def ref_thicken(grid: np.ndarray, target_color: int) -> np.ndarray:
    """Add a 1-cell 4-conn border to all blobs of target_color."""
    mask = (grid == target_color)
    h, w = grid.shape
    dilated = mask.copy()
    if h > 1:
        dilated[1:, :] |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
    if w > 1:
        dilated[:, 1:] |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
    out = np.where(dilated, target_color, grid).astype(grid.dtype)
    return out


def ref_hollow_out(grid: np.ndarray, target_color: int) -> np.ndarray:
    """Remove interior of target_color blobs, keep only border cells
    (cells with at least one non-target-color 4-neighbor).

    Edge cells are treated as border (padded with non-target).
    """
    mask = (grid == target_color)
    h, w = grid.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:h + 1, 1:w + 1] = mask
    interior = np.ones_like(mask, dtype=bool)
    for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        interior &= padded[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc]
    border = mask & ~interior
    out = np.where(border, target_color, 0).astype(grid.dtype)
    return out


def ref_remove_color(grid: np.ndarray, target_color: int) -> np.ndarray:
    """Delete all cells of target_color (set to 0)."""
    out = grid.copy()
    out[grid == target_color] = 0
    return out


def ref_trim_border(grid: np.ndarray, k: int) -> np.ndarray:
    """Remove k rows/cols from each border."""
    h, w = grid.shape
    if 2 * k >= h or 2 * k >= w:
        return np.zeros((0, 0), dtype=grid.dtype)
    return grid[k:h - k, k:w - k].copy()


REF_FUNCS = {
    "identity": ref_identity,
    "transpose": ref_transpose,
    "flip_h": ref_flip_h,
    "flip_v": ref_flip_v,
    "rotate180": ref_rotate180,
    "rotate90": ref_rotate90,
    "replace_color": ref_replace_color,
    "swap_colors": ref_swap_colors,
    "rotate180_static": ref_rotate180_static,
    "rotate90_static": ref_rotate90_static,
    "count_color": ref_count_color,
    "dominant_color": ref_dominant_color,
    "fill_bg": ref_fill_bg,
    "bbox_crop": ref_bbox_crop,
    "tile_h": ref_tile_h,
    "tile_v": ref_tile_v,
    "largest_blob": ref_largest_blob,
    "select_channel": ref_select_channel,
    "fill": ref_fill,
    "shift_down": ref_shift_down,
    "shift_right": ref_shift_right,
    "crop": ref_crop,
    "mask_foreground": ref_mask_foreground,
    "flood_fill": ref_flood_fill,
    "thicken": ref_thicken,
    "hollow_out": ref_hollow_out,
    "remove_color": ref_remove_color,
    "trim_border": ref_trim_border,
}


# ---------------------------------------------------------------------------
# ONNX subgraph builders
# ---------------------------------------------------------------------------

def _n(op: str, ins: list[str], outs: list[str], **attrs) -> onnx.NodeProto:
    return helper.make_node(op, ins, outs, name=outs[0], **attrs)


def _arr(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name)


def build_identity(prefix: str, in_name: str, out_name: str) -> tuple[list, list]:
    return [_n("Identity", [in_name], [out_name])], []


def build_transpose(prefix: str, in_name: str, out_name: str) -> tuple[list, list]:
    return [_n("Transpose", [in_name], [out_name], perm=[0, 1, 3, 2])], []


def build_swap_colors(prefix: str, in_name: str, out_name: str, c1: int, c2: int) -> tuple[list, list]:
    if not (0 <= c1 < NCHAN and 0 <= c2 < NCHAN):
        raise ValueError("swap_colors: colors must be in [0,9]")
    perm = np.arange(NCHAN, dtype=np.int64)
    perm[c1] = c2
    perm[c2] = c1
    perm_name = f"{prefix}swap_perm"
    return [_n("Gather", [in_name, perm_name], [out_name], axis=1)], [_arr(perm_name, perm)]


def build_replace_color(prefix: str, in_name: str, out_name: str, src: int, dst: int) -> tuple[list, list]:
    """Single 1x1 Conv with a 10x10 channel-mixing matrix.

    M[c_out, c_in] = 1 if (c_in == c_out and c_out != src)  -- keep
                  or (c_in == src and c_out == dst)         -- redirect src to dst
                  else 0

    Conv produces the FLOAT output directly; values are in {0, 1, 2} (2 occurs
    if both src and dst had pixels at the same position, which is impossible
    in valid one-hot input). The scorer threshold is `> 0.0` so any non-zero
    value decodes to the channel.
    """
    if not (0 <= src < NCHAN and 0 <= dst < NCHAN):
        raise ValueError("replace_color: colors must be in [0,9]")
    if src == dst:
        return [_n("Identity", [in_name], [out_name])], []

    M = np.eye(NCHAN, dtype=np.float32)
    M[src, src] = 0.0
    M[dst, src] = 1.0
    # Conv weight shape: [out_channels, in_channels, 1, 1]
    W = M.reshape(NCHAN, NCHAN, 1, 1)
    W_name = f"{prefix}rc_W"
    return [_n("Conv", [in_name, W_name], [out_name])], [_arr(W_name, W)]


# ---------------------------------------------------------------------------
# Bbox-aware primitives.
#
# Pattern matches the v4_plus3 task150/155 baseline:
#   line_sum = ReduceSum(input, axes=[1, other_spatial_axis], keepdims=1)
#       -> [1,1,30,1] or [1,1,1,30]
#   line_bool = line_sum > 0.0
#   line_valid = Cast(line_bool, FLOAT)
#   size_f = ReduceSum(line_valid, axes=all)   # scalar FLOAT count
#   size_i = Cast(size_f, INT64)               # scalar INT64
#   inside = arange30 < size_i                  # BOOL [30] or [1,30]
#   idx    = Where(inside, size_i - 1 - arange30, arange30) - all INT64
#   output = Gather(input, idx, axis=primary_spatial_axis)
#
# All intermediates are scalars or 30-length vectors. The only big
# tensors are `input` (excluded) and `output` (excluded). So memory ≈ 0.
# ---------------------------------------------------------------------------

def _bbox_size_along(prefix: str, in_name: str, primary_axis: int) -> tuple[list, list, str]:
    """Emit nodes/initializers that compute the size of the bbox along
    `primary_axis` (2 for height, 3 for width) for a top-left-aligned
    `[1,10,30,30]` FLOAT input. Returns (nodes, inits, size_i_name).

    Uses BOOL → INT64 → ReduceSum to avoid FLOAT intermediates.
    """
    other_axis = 3 if primary_axis == 2 else 2
    zero_f = f"{prefix}bs_zero_f"
    nodes = [
        _n("ReduceSum", [in_name], [f"{prefix}bs_line_sum_{primary_axis}"],
           axes=[1, other_axis], keepdims=1),
        _n("Greater", [f"{prefix}bs_line_sum_{primary_axis}", zero_f],
           [f"{prefix}bs_line_bool_{primary_axis}"]),
        _n("Cast", [f"{prefix}bs_line_bool_{primary_axis}"],
           [f"{prefix}bs_line_i64_{primary_axis}"], to=TensorProto.INT64),
        _n("ReduceSum", [f"{prefix}bs_line_i64_{primary_axis}"],
           [f"{prefix}bs_size_i_{primary_axis}"], keepdims=0),
    ]
    inits = [_arr(zero_f, np.array(0.0, dtype=np.float32))]
    return nodes, inits, f"{prefix}bs_size_i_{primary_axis}"


def _flip_idx_along(prefix: str, size_i_name: str, axis: int) -> tuple[list, list, str]:
    """Emit nodes/inits to compute the dynamic flip index vector
    `idx[i] = size-1-i if i<size else i` along `axis`. Returns
    (nodes, inits, idx_name) — idx is INT64 shape [30]."""
    arange_name = f"{prefix}fi_arange_{axis}"
    one_i = f"{prefix}fi_one_{axis}"
    nodes = [
        _n("Sub", [size_i_name, one_i], [f"{prefix}fi_sm1_{axis}"]),
        _n("Sub", [f"{prefix}fi_sm1_{axis}", arange_name], [f"{prefix}fi_rev_{axis}"]),
        _n("Less", [arange_name, size_i_name], [f"{prefix}fi_inside_{axis}"]),
        _n("Where", [f"{prefix}fi_inside_{axis}", f"{prefix}fi_rev_{axis}", arange_name],
            [f"{prefix}fi_idx_{axis}"]),
    ]
    inits = [
        _arr(arange_name, np.arange(H30, dtype=np.int64)),
        _arr(one_i, np.array(1, dtype=np.int64)),
    ]
    return nodes, inits, f"{prefix}fi_idx_{axis}"


def build_flip_h(prefix: str, in_name: str, out_name: str) -> tuple[list, list]:
    bsn, bsi, size_w = _bbox_size_along(prefix, in_name, primary_axis=3)
    fin, fii, idx_w = _flip_idx_along(prefix + "h_", size_w, axis=3)
    gather = _n("Gather", [in_name, idx_w], [out_name], axis=3)
    return bsn + fin + [gather], bsi + fii


def build_flip_v(prefix: str, in_name: str, out_name: str) -> tuple[list, list]:
    bsn, bsi, size_h = _bbox_size_along(prefix, in_name, primary_axis=2)
    fin, fii, idx_h = _flip_idx_along(prefix + "v_", size_h, axis=2)
    gather = _n("Gather", [in_name, idx_h], [out_name], axis=2)
    return bsn + fin + [gather], bsi + fii


def build_rotate180(prefix: str, in_name: str, out_name: str) -> tuple[list, list]:
    """rotate180 = bbox-aware Gather(axis=2) then Gather(axis=3).

    NOTE: a Slice+Pad approach (matching v4_plus3 task087's static slice) is
    cheaper per-task BUT produces a dynamic-shape intermediate that the
    scorer's `calculate_memory` rejects (returns None). For now we use the
    dual-Gather pattern which keeps all intermediates statically shaped.
    """
    bsh_n, bsh_i, size_h = _bbox_size_along(prefix, in_name, primary_axis=2)
    bsw_n, bsw_i, size_w = _bbox_size_along(prefix + "w_", in_name, primary_axis=3)
    fih_n, fih_i, idx_h = _flip_idx_along(prefix + "rh_", size_h, axis=2)
    fiw_n, fiw_i, idx_w = _flip_idx_along(prefix + "rw_", size_w, axis=3)
    mid = f"{prefix}r180_mid"
    gathers = [
        _n("Gather", [in_name, idx_h], [mid], axis=2),
        _n("Gather", [mid, idx_w], [out_name], axis=3),
    ]
    return bsh_n + bsw_n + fih_n + fiw_n + gathers, bsh_i + bsw_i + fih_i + fiw_i


def build_rotate90(prefix: str, in_name: str, out_name: str) -> tuple[list, list]:
    """rotate90 ccw = transpose + flip_v (within the H==W bbox).

    Sequence: Transpose(axes 2,3) -> bbox-aware Gather(axis=2) using H of the
    transposed tensor (== W of the original). Like rotate180 we accept the
    [1,10,30,30] FLOAT `mid` intermediate to keep shapes static.
    """
    tr_out = f"{prefix}r90_tr"
    tr_nodes, tr_inits = build_transpose(prefix, in_name, tr_out)
    fv_nodes, fv_inits = build_flip_v(prefix + "r90_", tr_out, out_name)
    return tr_nodes + fv_nodes, tr_inits + fv_inits


def build_rotate180_static(prefix: str, in_name: str, out_name: str, h: int, w: int) -> tuple[list, list]:
    """Static-shape rotate180.

    Crop the [1,10,h,w] content out of the [1,10,30,30] envelope, reverse it
    along both spatial axes via Gather-with-static-indices, then Pad back to
    [1,10,30,30]. All shapes are statically inferable, so the scorer can
    measure cost (unlike the dynamic Slice+Pad attempt in v0 which produced
    dynamic-shape intermediates).
    """
    if not (1 <= h <= H30 and 1 <= w <= W30):
        raise ValueError(f"rotate180_static: h,w must be in [1,30], got h={h},w={w}")
    starts = f"{prefix}r180s_starts"
    ends = f"{prefix}r180s_ends"
    axes = f"{prefix}r180s_axes"
    steps = f"{prefix}r180s_steps"
    rev_h_idx = f"{prefix}r180s_revH_idx"
    rev_w_idx = f"{prefix}r180s_revW_idx"
    pads = f"{prefix}r180s_pads"
    sliced = f"{prefix}r180s_sliced"
    after_h = f"{prefix}r180s_afterH"
    after_w = f"{prefix}r180s_afterW"
    nodes = [
        _n("Slice", [in_name, starts, ends, axes, steps], [sliced]),
        _n("Gather", [sliced, rev_h_idx], [after_h], axis=2),
        _n("Gather", [after_h, rev_w_idx], [after_w], axis=3),
        _n("Pad", [after_w, pads], [out_name], mode="constant"),
    ]
    inits = [
        _arr(starts, np.array([0, 0, 0, 0], dtype=np.int64)),
        _arr(ends, np.array([1, NCHAN, h, w], dtype=np.int64)),
        _arr(axes, np.array([0, 1, 2, 3], dtype=np.int64)),
        _arr(steps, np.array([1, 1, 1, 1], dtype=np.int64)),
        _arr(rev_h_idx, np.arange(h - 1, -1, -1, dtype=np.int64)),
        _arr(rev_w_idx, np.arange(w - 1, -1, -1, dtype=np.int64)),
        _arr(pads, np.array([0, 0, 0, 0, 0, 0, H30 - h, W30 - w], dtype=np.int64)),
    ]
    return nodes, inits


def build_rotate90_static(prefix: str, in_name: str, out_name: str, h: int, w: int) -> tuple[list, list]:
    """Static-shape rotate90 (ccw): equivalent to flip_v(transpose(grid)).

    Pipeline: Slice -> Transpose -> Gather-rev along the new H axis -> Pad.
    Result has shape (w, h) inside the envelope.
    """
    if not (1 <= h <= H30 and 1 <= w <= W30):
        raise ValueError(f"rotate90_static: h,w must be in [1,30], got h={h},w={w}")
    starts = f"{prefix}r90s_starts"
    ends = f"{prefix}r90s_ends"
    axes = f"{prefix}r90s_axes"
    steps = f"{prefix}r90s_steps"
    rev_idx = f"{prefix}r90s_rev_idx"  # length w (new H after transpose)
    pads = f"{prefix}r90s_pads"
    sliced = f"{prefix}r90s_sliced"     # [1,10,h,w]
    transposed = f"{prefix}r90s_tr"     # [1,10,w,h]
    flipped = f"{prefix}r90s_flipped"   # [1,10,w,h]
    nodes = [
        _n("Slice", [in_name, starts, ends, axes, steps], [sliced]),
        _n("Transpose", [sliced], [transposed], perm=[0, 1, 3, 2]),
        _n("Gather", [transposed, rev_idx], [flipped], axis=2),
        _n("Pad", [flipped, pads], [out_name], mode="constant"),
    ]
    inits = [
        _arr(starts, np.array([0, 0, 0, 0], dtype=np.int64)),
        _arr(ends, np.array([1, NCHAN, h, w], dtype=np.int64)),
        _arr(axes, np.array([0, 1, 2, 3], dtype=np.int64)),
        _arr(steps, np.array([1, 1, 1, 1], dtype=np.int64)),
        _arr(rev_idx, np.arange(w - 1, -1, -1, dtype=np.int64)),
        _arr(pads, np.array([0, 0, 0, 0, 0, 0, H30 - w, W30 - h], dtype=np.int64)),
    ]
    return nodes, inits


def build_count_color(prefix: str, in_name: str, out_name: str, c: int) -> tuple[list, list]:
    """Encode `count(grid==c)` (clipped to 9) as the single colored cell at (0,0).

    The output grid decodes to [[count]] (a 1x1 grid). For tasks where the
    answer is "how many pixels of color c are there", this is the right shape.
    """
    if not (0 <= c < NCHAN):
        raise ValueError("count_color: c must be in [0,9]")
    # ReduceSum(input, axes=[0,2,3]) -> [10] FLOAT, then index c
    counts10 = f"{prefix}cc_counts10"          # [10]
    c_idx = f"{prefix}cc_c_idx"
    cnt_f_arr = f"{prefix}cc_cnt_f_arr"         # [1] FLOAT after Gather
    cnt_f = f"{prefix}cc_cnt_f"                 # scalar FLOAT
    cnt_i = f"{prefix}cc_cnt_i"                 # scalar INT64
    cap = f"{prefix}cc_cap"
    over = f"{prefix}cc_over"
    cnt_clipped = f"{prefix}cc_cnt_clip"        # scalar INT64 in [0,9]
    onehot_depth = f"{prefix}cc_depth"
    onehot_vals = f"{prefix}cc_vals"
    onehot10 = f"{prefix}cc_onehot10"           # [10] FLOAT
    reshape_target = f"{prefix}cc_reshape"
    onehot_4d = f"{prefix}cc_onehot4d"          # [1,10,1,1]
    pads = f"{prefix}cc_pads"
    nodes = [
        _n("ReduceSum", [in_name], [counts10], axes=[0, 2, 3], keepdims=0),
        _n("Gather", [counts10, c_idx], [cnt_f_arr], axis=0),
        _n("Squeeze", [cnt_f_arr], [cnt_f]),
        _n("Cast", [cnt_f], [cnt_i], to=TensorProto.INT64),
        # Clamp cnt_i to [0, 9]: opset 11 Min/Max don't accept int64; use Where.
        _n("Greater", [cnt_i, cap], [over]),
        _n("Where", [over, cap, cnt_i], [cnt_clipped]),
        # OneHot requires rank>=1 indices -> Unsqueeze scalar to [1].
        _n("Unsqueeze", [cnt_clipped], [f"{prefix}cc_cnt_1"], axes=[0]),
        _n("OneHot", [f"{prefix}cc_cnt_1", onehot_depth, onehot_vals], [onehot10], axis=-1),
        _n("Reshape", [onehot10, reshape_target], [onehot_4d]),
        _n("Pad", [onehot_4d, pads], [out_name], mode="constant"),
    ]
    inits = [
        _arr(c_idx, np.array(c, dtype=np.int64)),
        _arr(cap, np.array(NCHAN - 1, dtype=np.int64)),
        _arr(onehot_depth, np.array(NCHAN, dtype=np.int64)),
        _arr(onehot_vals, np.array([0.0, 1.0], dtype=np.float32)),
        _arr(reshape_target, np.array([1, NCHAN, 1, 1], dtype=np.int64)),
        _arr(pads, np.array([0, 0, 0, 0, 0, 0, H30 - 1, W30 - 1], dtype=np.int64)),
    ]
    return nodes, inits


def build_dominant_color(prefix: str, in_name: str, out_name: str) -> tuple[list, list]:
    """Fill every occupied cell of the original h×w grid with the dominant color.

    v1 FLOAT pipeline: Greater(>0) → Cast(FLOAT) instead of the BOOL
    Equal+Not pipeline. This is simpler and cheaper: the BOOL pipeline added
    overhead without real savings since OneHot still produces FLOAT.
    """
    counts = f"{prefix}dc_counts"               # [1,10,1,1] FLOAT
    dom_idx_3d = f"{prefix}dc_dom_idx_3d"       # [1,1,1] INT64
    onehot_depth = f"{prefix}dc_depth"
    onehot_vals = f"{prefix}dc_vals"
    onehot_4d_raw = f"{prefix}dc_onehot4d_raw"  # [1,1,1,10] FLOAT
    reshape_target = f"{prefix}dc_reshape"
    onehot_4d = f"{prefix}dc_onehot4d"          # [1,10,1,1] FLOAT
    cell_sum = f"{prefix}dc_cell_sum"           # [1,1,30,30] FLOAT
    zero_f = f"{prefix}dc_zero_f"
    cell_bool = f"{prefix}dc_cell_bool"         # [1,1,30,30] BOOL
    mask_f = f"{prefix}dc_mask_f"               # [1,1,30,30] FLOAT
    nodes = [
        _n("ReduceSum", [in_name], [counts], axes=[0, 2, 3], keepdims=1),
        _n("ArgMax", [counts], [dom_idx_3d], axis=1, keepdims=0),
        _n("OneHot", [dom_idx_3d, onehot_depth, onehot_vals], [onehot_4d_raw], axis=-1),
        _n("Reshape", [onehot_4d_raw, reshape_target], [onehot_4d]),
        _n("ReduceSum", [in_name], [cell_sum], axes=[1], keepdims=1),
        _n("Greater", [cell_sum, zero_f], [cell_bool]),
        _n("Cast", [cell_bool], [mask_f], to=TensorProto.FLOAT),
        _n("Mul", [onehot_4d, mask_f], [out_name]),
    ]
    inits = [
        _arr(onehot_depth, np.array(NCHAN, dtype=np.int64)),
        _arr(onehot_vals, np.array([0.0, 1.0], dtype=np.float32)),
        _arr(reshape_target, np.array([1, NCHAN, 1, 1], dtype=np.int64)),
        _arr(zero_f, np.array(0.0, dtype=np.float32)),
    ]
    return nodes, inits


def build_fill_bg(prefix: str, in_name: str, out_name: str, bg_color: int) -> tuple[list, list]:
    """Alias for `replace_color(src=0, dst=bg_color)` — set every background
    (color 0) cell to the requested color. Kept as a separate primitive name
    so search programs surface the semantic, not the encoding trick."""
    if not (0 <= bg_color < NCHAN):
        raise ValueError("fill_bg: bg_color must be in [0,9]")
    return build_replace_color(prefix, in_name, out_name, src=0, dst=bg_color)


def build_bbox_crop(prefix: str, in_name: str, out_name: str, target_color: int) -> tuple[list, list]:
    """Crop the bbox of `target_color` and shift it to top-left.

    All intermediate dimensions are statically `[1,10,30,30]` or shorter; the
    bbox extent flows through INT64 scalars / 30-length vectors and feeds two
    Gather ops (axis 2 then axis 3). A separate [1,1,30,30] mask zeros out
    cells outside the cropped extent.

    Cost is high (two [1,10,30,30] FLOAT intermediates ≈ 72 KB) but the graph
    is *measurable* — no dynamic dim sizes. If a single task hits this cost,
    that's still a viable scoring option for tasks where no cheaper primitive
    fits. The search harness will surface it only when it actually fits.

    BOOL optimization: line-sum intermediates flow through INT64 directly,
    avoiding FLOAT intermediates for the row/column presence mask.
    """
    if not (0 <= target_color < NCHAN):
        raise ValueError("bbox_crop: target_color must be in [0,9]")
    tc_idx = f"{prefix}bc_tc_idx"
    tm = f"{prefix}bc_tm"                       # [1,1,30,30] FLOAT
    rs = f"{prefix}bc_rs"                       # [1,1,30,1]
    cs = f"{prefix}bc_cs"                       # [1,1,1,30]
    zero_f = f"{prefix}bc_zero_f"
    rh_b = f"{prefix}bc_rh_b"                   # BOOL [1,1,30,1]
    ch_b = f"{prefix}bc_ch_b"                   # BOOL [1,1,1,30]
    rh_i = f"{prefix}bc_rh_i"                   # INT64 [1,1,30,1]
    ch_i = f"{prefix}bc_ch_i"                   # INT64 [1,1,1,30]
    h_i = f"{prefix}bc_h_i"                     # INT64 scalar
    w_i = f"{prefix}bc_w_i"                     # INT64 scalar
    fh_4d = f"{prefix}bc_fh_4d"
    fw_4d = f"{prefix}bc_fw_4d"
    fh = f"{prefix}bc_fh"
    fw = f"{prefix}bc_fw"
    arange_name = f"{prefix}bc_arange30"
    idx_h_pre = f"{prefix}bc_idx_h_pre"
    idx_h_in = f"{prefix}bc_idx_h_in"
    idx_h = f"{prefix}bc_idx_h"
    idx_w_pre = f"{prefix}bc_idx_w_pre"
    idx_w_in = f"{prefix}bc_idx_w_in"
    idx_w = f"{prefix}bc_idx_w"
    g1 = f"{prefix}bc_g1"
    g2 = f"{prefix}bc_g2"
    inside_h_shape = f"{prefix}bc_ih_shape"
    inside_w_shape = f"{prefix}bc_iw_shape"
    inside_h_4d = f"{prefix}bc_ih_4d"
    inside_w_4d = f"{prefix}bc_iw_4d"
    mask_and = f"{prefix}bc_mask_and"
    mask_f = f"{prefix}bc_mask_f"
    nodes = [
        _n("Gather", [in_name, tc_idx], [tm], axis=1),
        _n("ReduceSum", [tm], [rs], axes=[1, 3], keepdims=1),
        _n("ReduceSum", [tm], [cs], axes=[1, 2], keepdims=1),
        _n("Greater", [rs, zero_f], [rh_b]),
        _n("Greater", [cs, zero_f], [ch_b]),
        _n("Cast", [rh_b], [rh_i], to=TensorProto.INT64),  # reuse for size + ArgMax
        _n("Cast", [ch_b], [ch_i], to=TensorProto.INT64),
        _n("ReduceSum", [rh_i], [h_i], keepdims=0),          # INT64 scalar directly
        _n("ReduceSum", [ch_i], [w_i], keepdims=0),
        _n("ArgMax", [rh_i], [fh_4d], axis=2, keepdims=1),  # [1,1,1,1]
        _n("ArgMax", [ch_i], [fw_4d], axis=3, keepdims=1),  # [1,1,1,1]
        _n("Squeeze", [fh_4d], [fh]),                       # scalar
        _n("Squeeze", [fw_4d], [fw]),                       # scalar
        _n("Add", [arange_name, fh], [idx_h_pre]),          # [30]
        _n("Less", [arange_name, h_i], [idx_h_in]),         # BOOL [30]
        _n("Where", [idx_h_in, idx_h_pre, fh], [idx_h]),    # [30] INT64
        _n("Add", [arange_name, fw], [idx_w_pre]),
        _n("Less", [arange_name, w_i], [idx_w_in]),
        _n("Where", [idx_w_in, idx_w_pre, fw], [idx_w]),
        _n("Gather", [in_name, idx_h], [g1], axis=2),       # [1,10,30,30]
        _n("Gather", [g1, idx_w], [g2], axis=3),            # [1,10,30,30]
        _n("Reshape", [idx_h_in, inside_h_shape], [inside_h_4d]),  # BOOL [1,1,30,1]
        _n("Reshape", [idx_w_in, inside_w_shape], [inside_w_4d]),  # BOOL [1,1,1,30]
        _n("And", [inside_h_4d, inside_w_4d], [mask_and]),  # BOOL [1,1,30,30]
        _n("Cast", [mask_and], [mask_f], to=TensorProto.FLOAT),
        _n("Mul", [g2, mask_f], [out_name]),                # [1,10,30,30]
    ]
    inits = [
        _arr(tc_idx, np.array([target_color], dtype=np.int64)),
        _arr(zero_f, np.array(0.0, dtype=np.float32)),
        _arr(arange_name, np.arange(H30, dtype=np.int64)),
        _arr(inside_h_shape, np.array([1, 1, H30, 1], dtype=np.int64)),
        _arr(inside_w_shape, np.array([1, 1, 1, W30], dtype=np.int64)),
    ]
    return nodes, inits


# ---------------------------------------------------------------------------
# Tile primitives
# ---------------------------------------------------------------------------

def build_tile_h(prefix: str, in_name: str, out_name: str, n: int) -> tuple[list, list]:
    """Tile horizontally by n. Gather with repeated index along width axis.
    Mask to keep only within-tiled-width cells.
    """
    if not (1 <= n <= 30):
        raise ValueError(f"tile_h: n must be in [1,30], got {n}")
    # Detect bbox sizes
    bsh_nodes, bsh_inits, size_h = _bbox_size_along(prefix + "h_", in_name, primary_axis=2)
    bsw_nodes, bsw_inits, size_w = _bbox_size_along(prefix + "w_", in_name, primary_axis=3)
    arange_name = f"{prefix}th_arange"
    one_i = f"{prefix}th_one"
    thirty_i = f"{prefix}th_thirty"
    n_i = f"{prefix}th_n"
    zero_i = f"{prefix}th_zero"
    sz_lt_one = f"{prefix}th_sz_lt_one"
    safe_w = f"{prefix}th_safe_w"
    tiled_raw = f"{prefix}th_tw_raw"
    tr_lt_thirty = f"{prefix}th_lt_thirty"
    tiled_w = f"{prefix}th_tw"
    idx_raw = f"{prefix}th_idx_raw"
    inside_tiled = f"{prefix}th_inside_tiled"
    idx_w = f"{prefix}th_idx_w"
    inside_h = f"{prefix}th_inside_h"
    inside_h_4d = f"{prefix}th_inside_h_4d"
    inside_w_4d = f"{prefix}th_inside_w_4d"
    inside_h_shape = f"{prefix}th_ih_shape"
    inside_w_shape = f"{prefix}th_iw_shape"
    mask_and = f"{prefix}th_mask_and"
    mask_f = f"{prefix}th_mask_f"
    g = f"{prefix}th_g"
    nodes = [
        *bsh_nodes, *bsw_nodes,
        # safe_w = max(size_w, 1) via Less + Where
        _n("Less", [size_w, one_i], [sz_lt_one]),
        _n("Where", [sz_lt_one, one_i, size_w], [safe_w]),
        # tiled_w = min(size_w * n, 30)
        _n("Mul", [size_w, n_i], [tiled_raw]),
        _n("Less", [tiled_raw, thirty_i], [tr_lt_thirty]),
        _n("Where", [tr_lt_thirty, tiled_raw, thirty_i], [tiled_w]),
        # idx_raw = arange % safe_w  (tiling wrap-around)
        _n("Mod", [arange_name, safe_w], [idx_raw]),
        # inside_tiled = arange < tiled_w
        _n("Less", [arange_name, tiled_w], [inside_tiled]),
        # idx = Where(inside_tiled, idx_raw, zero_i)
        _n("Where", [inside_tiled, idx_raw, zero_i], [idx_w]),
        # Gather with tiled index
        _n("Gather", [in_name, idx_w], [g], axis=3),
        # Mask: keep cells within bbox height and tiled width
        _n("Less", [arange_name, size_h], [inside_h]),
        _n("Reshape", [inside_h, inside_h_shape], [inside_h_4d]),
        _n("Reshape", [inside_tiled, inside_w_shape], [inside_w_4d]),
        _n("And", [inside_h_4d, inside_w_4d], [mask_and]),
        _n("Cast", [mask_and], [mask_f], to=TensorProto.FLOAT),
        _n("Mul", [g, mask_f], [out_name]),
    ]
    inits = [
        *bsh_inits, *bsw_inits,
        _arr(arange_name, np.arange(H30, dtype=np.int64)),
        _arr(one_i, np.array(1, dtype=np.int64)),
        _arr(thirty_i, np.array(H30, dtype=np.int64)),
        _arr(n_i, np.array(n, dtype=np.int64)),
        _arr(zero_i, np.array(0, dtype=np.int64)),
        _arr(inside_h_shape, np.array([1, 1, H30, 1], dtype=np.int64)),
        _arr(inside_w_shape, np.array([1, 1, 1, W30], dtype=np.int64)),
    ]
    return nodes, inits


def build_tile_v(prefix: str, in_name: str, out_name: str, n: int) -> tuple[list, list]:
    """Tile vertically by n. Gather with repeated index along height axis.
    Mask to keep only within-tiled-height cells.
    """
    if not (1 <= n <= 30):
        raise ValueError(f"tile_v: n must be in [1,30], got {n}")
    bsh_nodes, bsh_inits, size_h = _bbox_size_along(prefix + "h_", in_name, primary_axis=2)
    bsw_nodes, bsw_inits, size_w = _bbox_size_along(prefix + "w_", in_name, primary_axis=3)
    arange_name = f"{prefix}tv_arange"
    one_i = f"{prefix}tv_one"
    thirty_i = f"{prefix}tv_thirty"
    n_i = f"{prefix}tv_n"
    zero_i = f"{prefix}tv_zero"
    sz_lt_one = f"{prefix}tv_sz_lt_one"
    safe_h = f"{prefix}tv_safe_h"
    tiled_raw = f"{prefix}tv_th_raw"
    tr_lt_thirty = f"{prefix}tv_lt_thirty"
    tiled_h = f"{prefix}tv_th"
    idx_raw = f"{prefix}tv_idx_raw"
    inside_tiled = f"{prefix}tv_inside_tiled"
    idx_h = f"{prefix}tv_idx_h"
    inside_w = f"{prefix}tv_inside_w"
    inside_h_4d = f"{prefix}tv_inside_h_4d"
    inside_w_4d = f"{prefix}tv_inside_w_4d"
    inside_h_shape = f"{prefix}tv_ih_shape"
    inside_w_shape = f"{prefix}tv_iw_shape"
    mask_and = f"{prefix}tv_mask_and"
    mask_f = f"{prefix}tv_mask_f"
    g = f"{prefix}tv_g"
    nodes = [
        *bsh_nodes, *bsw_nodes,
        # safe_h = max(size_h, 1)
        _n("Less", [size_h, one_i], [sz_lt_one]),
        _n("Where", [sz_lt_one, one_i, size_h], [safe_h]),
        # tiled_h = min(size_h * n, 30)
        _n("Mul", [size_h, n_i], [tiled_raw]),
        _n("Less", [tiled_raw, thirty_i], [tr_lt_thirty]),
        _n("Where", [tr_lt_thirty, tiled_raw, thirty_i], [tiled_h]),
        _n("Mod", [arange_name, safe_h], [idx_raw]),
        _n("Less", [arange_name, tiled_h], [inside_tiled]),
        _n("Where", [inside_tiled, idx_raw, zero_i], [idx_h]),
        _n("Gather", [in_name, idx_h], [g], axis=2),
        _n("Less", [arange_name, size_w], [inside_w]),
        _n("Reshape", [inside_tiled, inside_h_shape], [inside_h_4d]),
        _n("Reshape", [inside_w, inside_w_shape], [inside_w_4d]),
        _n("And", [inside_h_4d, inside_w_4d], [mask_and]),
        _n("Cast", [mask_and], [mask_f], to=TensorProto.FLOAT),
        _n("Mul", [g, mask_f], [out_name]),
    ]
    inits = [
        *bsh_inits, *bsw_inits,
        _arr(arange_name, np.arange(H30, dtype=np.int64)),
        _arr(one_i, np.array(1, dtype=np.int64)),
        _arr(thirty_i, np.array(H30, dtype=np.int64)),
        _arr(n_i, np.array(n, dtype=np.int64)),
        _arr(zero_i, np.array(0, dtype=np.int64)),
        _arr(inside_h_shape, np.array([1, 1, H30, 1], dtype=np.int64)),
        _arr(inside_w_shape, np.array([1, 1, 1, W30], dtype=np.int64)),
    ]
    return nodes, inits


# ---------------------------------------------------------------------------
# Largest blob — flood-fill via iterative morphological dilation
# ---------------------------------------------------------------------------

def build_largest_blob(prefix: str, in_name: str, out_name: str, target_color: int) -> tuple[list, list]:
    """Find the largest 4-connected blob of `target_color` by flood-fill
    from the top-leftmost target cell (8 iterations of 4-conn dilation +
    intersection with target mask).

    Pipeline:
      1. Extract target_color channel → target_mask (BOOL)
      2. Find top-leftmost target cell → seed_mask (BOOL)
      3. For 8 iterations: Conv(3x3, 4-conn) → Greater > 0 → And(target_mask)
      4. Project result back into target_color channel via 1x1 Conv
    """
    if not (0 <= target_color < NCHAN):
        raise ValueError(f"largest_blob: target_color must be in [0,9], got {target_color}")
    tc_idx = f"{prefix}lb_tc_idx"
    tc_chan = f"{prefix}lb_tc_chan"           # [1,1,30,30] FLOAT
    zero_f = f"{prefix}lb_zero_f"
    target_mask = f"{prefix}lb_target_mask"   # BOOL [1,1,30,30]

    # --- seed finding ---
    arange_name = f"{prefix}lb_arange30"
    row_reshape = f"{prefix}lb_row_reshape"
    col_reshape = f"{prefix}lb_col_reshape"
    row_idx_4d = f"{prefix}lb_row_idx_4d"     # INT64 [1,1,30,1]
    col_idx_4d = f"{prefix}lb_col_idx_4d"     # INT64 [1,1,1,30]
    row_scale = f"{prefix}lb_row_scale"       # scalar = 30 (width)
    row_scaled = f"{prefix}lb_row_scaled"     # INT64 [1,1,30,1]
    flat_idx = f"{prefix}lb_flat_idx"         # INT64 [1,1,30,30] = row*30 + col
    sentinel = f"{prefix}lb_sentinel"         # scalar = 900 (> max valid 899)
    masked_idx = f"{prefix}lb_masked_idx"
    seed_flat = f"{prefix}lb_seed_flat"       # INT64 [1,1,1,1]
    seed_cell = f"{prefix}lb_seed_cell"       # BOOL [1,1,30,30]
    seed_mask = f"{prefix}lb_seed_mask"       # BOOL [1,1,30,30]

    # --- dilation loop (8 iterations) ---
    conn_weight = f"{prefix}lb_conn_W"
    iter_nodes = []
    prev = seed_mask
    for i in range(8):
        cur_f = f"{prefix}lb_cur{i}_f"
        dilated = f"{prefix}lb_dil{i}"
        dilated_bool = f"{prefix}lb_dilb{i}"
        next_bool = f"{prefix}lb_next{i}"
        iter_nodes.extend([
            _n("Cast", [prev], [cur_f], to=TensorProto.FLOAT),
            _n("Conv", [cur_f, conn_weight], [dilated], auto_pad="SAME_UPPER"),
            _n("Greater", [dilated, zero_f], [dilated_bool]),
            _n("And", [dilated_bool, target_mask], [next_bool]),
        ])
        prev = next_bool

    # --- output projection ---
    proj_weight = f"{prefix}lb_proj_W"
    blob_f = f"{prefix}lb_blob_f"

    nodes = [
        # Extract target channel
        _n("Gather", [in_name, tc_idx], [tc_chan], axis=1),
        _n("Greater", [tc_chan, zero_f], [target_mask]),

        # Seed: top-leftmost target cell via flat index
        # Build [1,1,30,30] flat_idx where flat_idx[r,c] = r*30 + c
        _n("Reshape", [arange_name, row_reshape], [row_idx_4d]),
        _n("Reshape", [arange_name, col_reshape], [col_idx_4d]),
        _n("Mul", [row_idx_4d, row_scale], [row_scaled]),
        _n("Add", [row_scaled, col_idx_4d], [flat_idx]),
        # Non-target cells get 900 (outside valid 0..899)
        _n("Where", [target_mask, flat_idx, sentinel], [masked_idx]),
        _n("ReduceMin", [masked_idx], [seed_flat], axes=[2, 3], keepdims=1),
        _n("Equal", [flat_idx, seed_flat], [seed_cell]),
        _n("And", [seed_cell, target_mask], [seed_mask]),

        *iter_nodes,

        # Project back: blob mask → target_color channel via 1x1 Conv
        _n("Cast", [prev], [blob_f], to=TensorProto.FLOAT),
        _n("Conv", [blob_f, proj_weight], [out_name]),
    ]
    inits = [
        _arr(tc_idx, np.array([target_color], dtype=np.int64)),
        _arr(zero_f, np.array(0.0, dtype=np.float32)),
        _arr(arange_name, np.arange(H30, dtype=np.int64)),
        _arr(row_reshape, np.array([1, 1, H30, 1], dtype=np.int64)),
        _arr(col_reshape, np.array([1, 1, 1, W30], dtype=np.int64)),
        _arr(row_scale, np.array(W30, dtype=np.int64)),
        _arr(sentinel, np.array(H30 * W30, dtype=np.int64)),
        _arr(conn_weight, np.array([[[[0, 1, 0],
                                       [1, 1, 1],
                                       [0, 1, 0]]]], dtype=np.float32)),
        _arr(proj_weight, np.zeros((NCHAN, 1, 1, 1), dtype=np.float32)),
    ]
    proj_arr = np.zeros((NCHAN, 1, 1, 1), dtype=np.float32)
    proj_arr[target_color, 0, 0, 0] = 1.0
    inits[-1] = _arr(proj_weight, proj_arr)
    return nodes, inits


# ---------------------------------------------------------------------------
# Batch 1 primitives: select_channel, fill, shift_down/right, crop
# ---------------------------------------------------------------------------


def build_select_channel(prefix: str, in_name: str, out_name: str, cond_ch: int, if_ch: int, else_ch: int) -> tuple[list, list]:
    """Per-cell channel selection: for each cell, if cond_ch is active, output
    if_ch, else output else_ch.

    ONNX pipeline: Gather(cond_ch) → Greater(>0) → BOOL mask
    Where(mask, if_onehot, else_onehot) → broadcast to [1,10,30,30]
    Mask by cell occupancy to avoid spurious output outside the content area.
    """
    if not all(0 <= c < NCHAN for c in (cond_ch, if_ch, else_ch)):
        raise ValueError("select_channel: channels must be in [0,9]")
    cond_idx = f"{prefix}sc_cond_idx"
    cond_chan = f"{prefix}sc_cond_chan"
    zero_f = f"{prefix}sc_zero_f"
    cond_bool = f"{prefix}sc_cond_bool"
    if_sel = f"{prefix}sc_if_sel"
    else_sel = f"{prefix}sc_else_sel"
    cell_sum = f"{prefix}sc_cell_sum"
    cell_bool = f"{prefix}sc_cell_bool"
    cell_mask = f"{prefix}sc_cell_mask"
    where_out = f"{prefix}sc_where"

    if_arr = np.zeros((1, NCHAN, 1, 1), dtype=np.float32)
    if_arr[0, if_ch, 0, 0] = 1.0
    else_arr = np.zeros((1, NCHAN, 1, 1), dtype=np.float32)
    else_arr[0, else_ch, 0, 0] = 1.0

    nodes = [
        _n("ReduceSum", [in_name], [cell_sum], axes=[1], keepdims=1),
        _n("Greater", [cell_sum, zero_f], [cell_bool]),
        _n("Cast", [cell_bool], [cell_mask], to=TensorProto.FLOAT),
        _n("Gather", [in_name, cond_idx], [cond_chan], axis=1),
        _n("Greater", [cond_chan, zero_f], [cond_bool]),
        _n("Where", [cond_bool, if_sel, else_sel], [where_out]),
        _n("Mul", [where_out, cell_mask], [out_name]),
    ]
    inits = [
        _arr(cond_idx, np.array([cond_ch], dtype=np.int64)),
        _arr(zero_f, np.array(0.0, dtype=np.float32)),
        _arr(if_sel, if_arr),
        _arr(else_sel, else_arr),
    ]
    return nodes, inits


def build_fill(prefix: str, in_name: str, out_name: str, c: int) -> tuple[list, list]:
    """Fill each occupied cell with color c. Uses Tile to broadcast the
    one-hot [1,10,1,1] constant to [1,10,30,30], then masks by cell
    occupancy to keep only cells that were non-zero in the input."""
    if not (0 <= c < NCHAN):
        raise ValueError(f"fill: c must be in [0,9], got {c}")
    cell_sum = f"{prefix}fi_cell_sum"
    zero_f = f"{prefix}fi_zero_f"
    cell_bool = f"{prefix}fi_cell_bool"
    cell_mask = f"{prefix}fi_cell_mask"
    onehot = f"{prefix}fi_onehot"
    tile_repeats = f"{prefix}fi_tile"
    tiled = f"{prefix}fi_tiled"

    onehot_arr = np.zeros((1, NCHAN, 1, 1), dtype=np.float32)
    onehot_arr[0, c, 0, 0] = 1.0

    nodes = [
        _n("ReduceSum", [in_name], [cell_sum], axes=[1], keepdims=1),
        _n("Greater", [cell_sum, zero_f], [cell_bool]),
        _n("Cast", [cell_bool], [cell_mask], to=TensorProto.FLOAT),
        _n("Tile", [onehot, tile_repeats], [tiled]),
        _n("Mul", [tiled, cell_mask], [out_name]),
    ]
    inits = [
        _arr(onehot, onehot_arr),
        _arr(zero_f, np.array(0.0, dtype=np.float32)),
        _arr(tile_repeats, np.array([1, 1, H30, W30], dtype=np.int64)),
    ]
    return nodes, inits


def _shift_idx_and_mask(prefix: str, size_i_name: str, k: int, axis: int) -> tuple[list, list, str, str]:
    """Build index vector and row/col mask for bbox-aware shift.

    Returns (nodes, inits, idx_name, mask_bool_name) where:
      - idx is INT64 [30]: idx[r] = r-k if r in [k, size), else r
      - mask_bool is BOOL [30]: 0 for rows [0,k), 1 for rows [k, 30)
    """
    arange = f"{prefix}sidx_arange_{axis}"
    k_i = f"{prefix}sidx_k_{axis}"
    lt_k = f"{prefix}sidx_lt_k_{axis}"
    ge_k = f"{prefix}sidx_ge_k_{axis}"
    lt_size = f"{prefix}sidx_lt_size_{axis}"
    inside = f"{prefix}sidx_inside_{axis}"
    idx_shifted = f"{prefix}sidx_shifted_{axis}"
    idx = f"{prefix}sidx_idx_{axis}"

    nodes = [
        _n("Less", [arange, k_i], [lt_k]),
        _n("Not", [lt_k], [ge_k]),
        _n("Less", [arange, size_i_name], [lt_size]),
        _n("And", [ge_k, lt_size], [inside]),
        _n("Sub", [arange, k_i], [idx_shifted]),
        _n("Where", [inside, idx_shifted, arange], [idx]),
    ]
    inits = [
        _arr(arange, np.arange(H30, dtype=np.int64)),
        _arr(k_i, np.array(k, dtype=np.int64)),
    ]
    return nodes, inits, idx, ge_k


def build_shift_down(prefix: str, in_name: str, out_name: str, k: int) -> tuple[list, list]:
    """Shift grid content down by k rows within the content bbox.

    Uses bbox-aware indexing: rows [k, size_h) of output get input rows
    [0, size_h-k). Rows outside the content area stay zero.
    """
    if not (0 <= k < H30):
        raise ValueError(f"shift_down: k must be in [0,{H30-1}], got {k}")
    bsn, bsi, size_h = _bbox_size_along(prefix, in_name, primary_axis=2)
    sin, sii, idx_h, ge_k = _shift_idx_and_mask(prefix + "sd_", size_h, k, axis=2)
    mask_shape = f"{prefix}sd_mshape"
    mask_4d = f"{prefix}sd_mask_4d"
    mask_f = f"{prefix}sd_mask_f"
    g = f"{prefix}sd_g"
    nodes = [
        *bsn, *sin,
        _n("Gather", [in_name, idx_h], [g], axis=2),
        _n("Cast", [ge_k], [mask_f], to=TensorProto.FLOAT),
        _n("Reshape", [mask_f, mask_shape], [mask_4d]),
        _n("Mul", [g, mask_4d], [out_name]),
    ]
    inits = [*bsi, *sii, _arr(mask_shape, np.array([1, 1, H30, 1], dtype=np.int64))]
    return nodes, inits


def build_shift_right(prefix: str, in_name: str, out_name: str, k: int) -> tuple[list, list]:
    """Shift grid content right by k columns within the content bbox."""
    if not (0 <= k < W30):
        raise ValueError(f"shift_right: k must be in [0,{W30-1}], got {k}")
    bsn, bsi, size_w = _bbox_size_along(prefix, in_name, primary_axis=3)
    sin, sii, idx_w, ge_k = _shift_idx_and_mask(prefix + "sr_", size_w, k, axis=3)
    mask_shape = f"{prefix}sr_mshape"
    mask_4d = f"{prefix}sr_mask_4d"
    mask_f = f"{prefix}sr_mask_f"
    g = f"{prefix}sr_g"
    nodes = [
        *bsn, *sin,
        _n("Gather", [in_name, idx_w], [g], axis=3),
        _n("Cast", [ge_k], [mask_f], to=TensorProto.FLOAT),
        _n("Reshape", [mask_f, mask_shape], [mask_4d]),
        _n("Mul", [g, mask_4d], [out_name]),
    ]
    inits = [*bsi, *sii, _arr(mask_shape, np.array([1, 1, 1, W30], dtype=np.int64))]
    return nodes, inits


def build_crop(prefix: str, in_name: str, out_name: str, r0: int, c0: int, h: int, w: int) -> tuple[list, list]:
    """Extract h×w subgrid from (r0, c0) via Slice, then Pad to [1,10,30,30].

    Clamps at build time to [0,30). If the clamped region is empty, outputs
    all zeros.
    """
    r0 = max(0, min(r0, H30 - 1))
    c0 = max(0, min(c0, W30 - 1))
    r1 = min(r0 + h, H30)
    c1 = min(c0 + w, W30)
    slice_h = max(0, r1 - r0)
    slice_w = max(0, c1 - c0)
    pads_h = H30 - slice_h
    pads_w = W30 - slice_w

    if slice_h <= 0 or slice_w <= 0:
        zero_4d = f"{prefix}cr_zero"
        pads = f"{prefix}cr_pads"
        nodes = [
            _n("Pad", [zero_4d, pads], [out_name], mode="constant"),
        ]
        inits = [
            _arr(zero_4d, np.zeros((1, NCHAN, 1, 1), dtype=np.float32)),
            _arr(pads, np.array([0, 0, 0, 0, 0, 0, H30 - 1, W30 - 1], dtype=np.int64)),
        ]
        return nodes, inits

    starts = f"{prefix}cr_starts"
    ends = f"{prefix}cr_ends"
    axes = f"{prefix}cr_axes"
    steps = f"{prefix}cr_steps"
    pads = f"{prefix}cr_pads"
    sliced = f"{prefix}cr_sliced"

    nodes = [
        _n("Slice", [in_name, starts, ends, axes, steps], [sliced]),
        _n("Pad", [sliced, pads], [out_name], mode="constant"),
    ]
    inits = [
        _arr(starts, np.array([0, 0, r0, c0], dtype=np.int64)),
        _arr(ends, np.array([1, NCHAN, r1, c1], dtype=np.int64)),
        _arr(axes, np.array([0, 1, 2, 3], dtype=np.int64)),
        _arr(steps, np.array([1, 1, 1, 1], dtype=np.int64)),
        _arr(pads, np.array([0, 0, 0, 0, 0, 0, pads_h, pads_w], dtype=np.int64)),
    ]
    return nodes, inits


# ---------------------------------------------------------------------------
# Batch 2 ONNX builders
# ---------------------------------------------------------------------------


def _merge_with_mask(
    prefix: str,
    in_name: str,
    mask_bool_name: str,
    out_name: str,
    color: int,
) -> tuple[list, list]:
    """Emit nodes to merge a BOOL [1,1,30,30] mask into the input:
    cells where mask is True get `color`, other cells keep original input.
    Returns (nodes, inits)."""
    mask_f = f"{prefix}mm_mask_f"
    one_f = f"{prefix}mm_one_f"
    inv_mask = f"{prefix}mm_inv"
    kept = f"{prefix}mm_kept"
    oh = f"{prefix}mm_oh"
    filled = f"{prefix}mm_filled"
    oh_arr = np.zeros((1, NCHAN, 1, 1), dtype=np.float32)
    oh_arr[0, color, 0, 0] = 1.0
    nodes = [
        _n("Cast", [mask_bool_name], [mask_f], to=TensorProto.FLOAT),
        _n("Sub", [one_f, mask_f], [inv_mask]),
        _n("Mul", [in_name, inv_mask], [kept]),
        _n("Mul", [oh, mask_f], [filled]),
        _n("Add", [kept, filled], [out_name]),
    ]
    inits = [
        _arr(one_f, np.array(1.0, dtype=np.float32).reshape(1, 1, 1, 1)),
        _arr(oh, oh_arr),
    ]
    return nodes, inits


def _extract_chan(
    prefix: str, in_name: str, color: int
) -> tuple[list, list, str, str]:
    """Extract a single color channel as FLOAT [1,1,30,30] and BOOL.
    Returns (nodes, inits, chan_float_name, chan_bool_name)."""
    idx = f"{prefix}ec_idx"
    chan = f"{prefix}ec_chan"
    zero_f = f"{prefix}ec_zero_f"
    chan_bool = f"{prefix}ec_chan_bool"
    nodes = [
        _n("Gather", [in_name, idx], [chan], axis=1),
        _n("Greater", [chan, zero_f], [chan_bool]),
    ]
    inits = [
        _arr(idx, np.array([color], dtype=np.int64)),
        _arr(zero_f, np.array(0.0, dtype=np.float32)),
    ]
    return nodes, inits, chan, chan_bool


def _dilate_4conn(
    prefix: str, bool_name: str, n_iters: int = 1
) -> tuple[list, list, str]:
    """Dilate a BOOL [1,1,30,30] mask using 4-conn Conv for n_iters.
    Returns (nodes, inits, dilated_bool_name)."""
    conn_weight = f"{prefix}dil_conn_W"
    prev = bool_name
    nodes = []
    for i in range(n_iters):
        f = f"{prefix}dil_f{i}"
        conv = f"{prefix}dil_conv{i}"
        zero_f = f"{prefix}dil_zero_f{i}"
        dilated_bool = f"{prefix}dil_bool{i}"
        nodes.extend([
            _n("Cast", [prev], [f], to=TensorProto.FLOAT),
            _n("Conv", [f, conn_weight], [conv], auto_pad="SAME_UPPER"),
            _n("Greater", [conv, zero_f], [dilated_bool]),
        ])
        prev = dilated_bool
    inits = [
        _arr(conn_weight, np.array([[[[0, 1, 0],
                                       [1, 1, 1],
                                       [0, 1, 0]]]], dtype=np.float32)),
    ]
    # Add zero_f inits
    for i in range(n_iters):
        inits.append(
            _arr(f"{prefix}dil_zero_f{i}", np.array(0.0, dtype=np.float32))
        )
    return nodes, inits, prev


def build_mask_foreground(
    prefix: str, in_name: str, out_name: str
) -> tuple[list, list]:
    """Output a binary [1,10,30,30] where cells with non-zero color get color 1.

    Pipeline: ReduceSum(all channels) → Subtract(ch0) → Greater(>0) → Cast →
    Tile + Mul with onehot(1) to broadcast.

    We exclude channel 0 (ARC background) from the sum so that color-0 cells
    produce no output.
    """
    cell_sum = f"{prefix}mf_sum"
    ch0_idx = f"{prefix}mf_ch0_idx"
    ch0 = f"{prefix}mf_ch0"
    no_bg = f"{prefix}mf_no_bg"
    zero_f = f"{prefix}mf_zero_f"
    cell_bool = f"{prefix}mf_bool"
    cell_f = f"{prefix}mf_f"
    oh = f"{prefix}mf_oh"
    tile_repeats = f"{prefix}mf_tile"
    tiled = f"{prefix}mf_tiled"

    oh_arr = np.zeros((1, NCHAN, 1, 1), dtype=np.float32)
    oh_arr[0, 1, 0, 0] = 1.0

    nodes = [
        _n("ReduceSum", [in_name], [cell_sum], axes=[1], keepdims=1),
        _n("Gather", [in_name, ch0_idx], [ch0], axis=1),
        _n("Sub", [cell_sum, ch0], [no_bg]),
        _n("Greater", [no_bg, zero_f], [cell_bool]),
        _n("Cast", [cell_bool], [cell_f], to=TensorProto.FLOAT),
        _n("Tile", [oh, tile_repeats], [tiled]),
        _n("Mul", [tiled, cell_f], [out_name]),
    ]
    inits = [
        _arr(ch0_idx, np.array([0], dtype=np.int64)),
        _arr(zero_f, np.array(0.0, dtype=np.float32)),
        _arr(oh, oh_arr),
        _arr(tile_repeats, np.array([1, 1, H30, W30], dtype=np.int64)),
    ]
    return nodes, inits


def build_flood_fill(
    prefix: str,
    in_name: str,
    out_name: str,
    target_color: int,
    fill_color: int,
) -> tuple[list, list]:
    """Grow all target_color blobs outward via 8 dilation iterations,
    then fill the dilated region with fill_color.

    Pipeline: extract target channel → 8× dilation → merge with fill_color
    → mask to content area.
    """
    if not (0 <= target_color < NCHAN and 0 <= fill_color < NCHAN):
        raise ValueError("flood_fill: colors must be in [0,9]")

    ec_nodes, ec_inits, _, target_bool = _extract_chan(
        prefix + "ff_", in_name, target_color
    )
    dil_nodes, dil_inits, dilated_bool = _dilate_4conn(
        prefix + "ff_", target_bool, n_iters=8
    )
    pre_mask = f"{prefix}ff_pre_mask"
    merge_nodes, merge_inits = _merge_with_mask(
        prefix + "ff_", in_name, dilated_bool, pre_mask, fill_color
    )
    mask_nodes, mask_inits = _mask_to_content(
        prefix + "ff_", in_name, pre_mask, out_name
    )
    return ec_nodes + dil_nodes + merge_nodes + mask_nodes, ec_inits + dil_inits + merge_inits + mask_inits


def _mask_to_content(
    prefix: str, in_name: str, target_name: str, out_name: str
) -> tuple[list, list]:
    """Mask `target_name` [1,10,30,30] to only allow values where `in_name`
    [1,10,30,30] has any non-zero channel (i.e. within the content area).

    Returns (nodes, inits).  The result is written to `out_name`.
    """
    cell_sum = f"{prefix}mtc_sum"
    zero_f = f"{prefix}mtc_zero_f"
    cell_bool = f"{prefix}mtc_bool"
    cell_f = f"{prefix}mtc_f"
    nodes = [
        _n("ReduceSum", [in_name], [cell_sum], axes=[1], keepdims=1),
        _n("Greater", [cell_sum, zero_f], [cell_bool]),
        _n("Cast", [cell_bool], [cell_f], to=TensorProto.FLOAT),
        _n("Mul", [target_name, cell_f], [out_name]),
    ]
    inits = [
        _arr(zero_f, np.array(0.0, dtype=np.float32)),
    ]
    return nodes, inits


def build_thicken(
    prefix: str, in_name: str, out_name: str, target_color: int
) -> tuple[list, list]:
    """Add a 1-cell 4-conn border to all blobs of target_color.

    Pipeline: extract target channel → dilate 1× → OR with original →
    merge → mask to content area.
    """
    if not (0 <= target_color < NCHAN):
        raise ValueError(f"thicken: target_color must be in [0,9], got {target_color}")

    ec_nodes, ec_inits, _, target_bool = _extract_chan(
        prefix + "th_", in_name, target_color
    )
    dil_nodes, dil_inits, dilated_bool = _dilate_4conn(
        prefix + "th_", target_bool, n_iters=1
    )
    thick_bool = f"{prefix}th_thick_bool"
    pre_mask = f"{prefix}th_pre_mask"
    nodes = [
        *ec_nodes,
        *dil_nodes,
        _n("Or", [dilated_bool, target_bool], [thick_bool]),
    ]
    merge_nodes, merge_inits = _merge_with_mask(
        prefix + "th_", in_name, thick_bool, pre_mask, target_color
    )
    mask_nodes, mask_inits = _mask_to_content(
        prefix + "th_", in_name, pre_mask, out_name
    )
    return nodes + merge_nodes + mask_nodes, ec_inits + dil_inits + merge_inits + mask_inits


def build_hollow_out(
    prefix: str, in_name: str, out_name: str, target_color: int
) -> tuple[list, list]:
    """Remove interior of target_color blobs, keep only border.

    A cell is interior iff it and all 4 neighbors are target_color. We output
    ONLY the border cells (target_color), zeroing out everything else.

    Pipeline: extract target channel → Conv(4-conn) → Greater(>4.5) detects
    fully-surrounded cells → Not → And(target) gives border → project via
    1x1 Conv to target_color channel.
    """
    if not (0 <= target_color < NCHAN):
        raise ValueError(f"hollow_out: target_color must be in [0,9], got {target_color}")

    ec_nodes, ec_inits, tc_f, target_bool = _extract_chan(
        prefix + "ho_", in_name, target_color
    )
    conn_weight = f"{prefix}ho_conn_W"
    neighbor_sum = f"{prefix}ho_nsum"
    four_p5 = f"{prefix}ho_4p5"
    interior_bool = f"{prefix}ho_interior"
    interior_fixed = f"{prefix}ho_interior_fixed"
    non_interior = f"{prefix}ho_non_interior"
    border_bool = f"{prefix}ho_border_bool"
    border_f = f"{prefix}ho_border_f"
    proj_weight = f"{prefix}ho_proj_W"
    proj_arr = np.zeros((NCHAN, 1, 1, 1), dtype=np.float32)
    proj_arr[target_color, 0, 0, 0] = 1.0

    nodes = [
        *ec_nodes,
        _n("Conv", [tc_f, conn_weight], [neighbor_sum], auto_pad="SAME_UPPER"),
        _n("Greater", [neighbor_sum, four_p5], [interior_bool]),
        _n("And", [target_bool, interior_bool], [interior_fixed]),
        _n("Not", [interior_fixed], [non_interior]),
        _n("And", [non_interior, target_bool], [border_bool]),
        _n("Cast", [border_bool], [border_f], to=TensorProto.FLOAT),
        _n("Conv", [border_f, proj_weight], [out_name]),
    ]
    inits = [
        *ec_inits,
        _arr(conn_weight, np.array([[[[0, 1, 0],
                                       [1, 1, 1],
                                       [0, 1, 0]]]], dtype=np.float32)),
        _arr(four_p5, np.array(4.5, dtype=np.float32)),
        _arr(proj_weight, proj_arr),
    ]
    return nodes, inits


def build_remove_color(
    prefix: str, in_name: str, out_name: str, target_color: int
) -> tuple[list, list]:
    """Semantic alias for replace_color(src=target_color, dst=0)."""
    if not (0 <= target_color < NCHAN):
        raise ValueError(f"remove_color: target_color must be in [0,9], got {target_color}")
    return build_replace_color(prefix, in_name, out_name, src=target_color, dst=0)


def build_trim_border(
    prefix: str, in_name: str, out_name: str, k: int
) -> tuple[list, list]:
    """Remove k rows/cols from each border of the content bbox.

    Bbox-aware implementation: compute content height/width, trim k from each
    side via Gather, then mask to the trimmed region.
    If 2k >= bbox dimension, outputs all zeros.
    """
    if not (1 <= k <= 14):
        raise ValueError(f"trim_border: k must be in [1,14], got {k}")

    # Compute bbox dimensions
    bsh_nodes, bsh_inits, size_h = _bbox_size_along(prefix + "h_", in_name, primary_axis=2)
    bsw_nodes, bsw_inits, size_w = _bbox_size_along(prefix + "w_", in_name, primary_axis=3)

    two_k = f"{prefix}tb_2k_i"
    one_k = f"{prefix}tb_1k_i"
    new_h = f"{prefix}tb_new_h"
    new_w = f"{prefix}tb_new_w"
    zero_i = f"{prefix}tb_zero_i"
    h_neg = f"{prefix}tb_h_neg"
    w_neg = f"{prefix}tb_w_neg"
    safe_new_h = f"{prefix}tb_safe_new_h"
    safe_new_w = f"{prefix}tb_safe_new_w"

    arange = f"{prefix}tb_arange"
    idx_h_base = f"{prefix}tb_idx_h_base"
    idx_w_base = f"{prefix}tb_idx_w_base"
    inside_h = f"{prefix}tb_inside_h"
    inside_w = f"{prefix}tb_inside_w"
    idx_h = f"{prefix}tb_idx_h"
    idx_w = f"{prefix}tb_idx_w"

    g1 = f"{prefix}tb_g1"
    g2 = f"{prefix}tb_g2"

    inside_h_4d = f"{prefix}tb_inside_h_4d"
    inside_w_4d = f"{prefix}tb_inside_w_4d"
    inside_h_shape = f"{prefix}tb_ih_shape"
    inside_w_shape = f"{prefix}tb_iw_shape"
    mask_and = f"{prefix}tb_mask_and"
    mask_f = f"{prefix}tb_mask_f"

    nodes = [
        *bsh_nodes, *bsw_nodes,
        # new_h = max(size_h - 2*k, 0);  idx[r] = r + k
        _n("Sub", [size_h, two_k], [new_h]),
        _n("Less", [new_h, zero_i], [h_neg]),
        _n("Where", [h_neg, zero_i, new_h], [safe_new_h]),
        _n("Sub", [size_w, two_k], [new_w]),
        _n("Less", [new_w, zero_i], [w_neg]),
        _n("Where", [w_neg, zero_i, new_w], [safe_new_w]),
        _n("Add", [arange, one_k], [idx_h_base]),
        _n("Less", [arange, safe_new_h], [inside_h]),
        _n("Where", [inside_h, idx_h_base, zero_i], [idx_h]),
        _n("Add", [arange, one_k], [idx_w_base]),
        _n("Less", [arange, safe_new_w], [inside_w]),
        _n("Where", [inside_w, idx_w_base, zero_i], [idx_w]),
        _n("Gather", [in_name, idx_h], [g1], axis=2),
        _n("Gather", [g1, idx_w], [g2], axis=3),
        _n("Reshape", [inside_h, inside_h_shape], [inside_h_4d]),
        _n("Reshape", [inside_w, inside_w_shape], [inside_w_4d]),
        _n("And", [inside_h_4d, inside_w_4d], [mask_and]),
        _n("Cast", [mask_and], [mask_f], to=TensorProto.FLOAT),
        _n("Mul", [g2, mask_f], [out_name]),
    ]
    inits = [
        *bsh_inits, *bsw_inits,
        _arr(two_k, np.array(2 * k, dtype=np.int64)),
        _arr(one_k, np.array(k, dtype=np.int64)),
        _arr(zero_i, np.array(0, dtype=np.int64)),
        _arr(arange, np.arange(H30, dtype=np.int64)),
        _arr(inside_h_shape, np.array([1, 1, H30, 1], dtype=np.int64)),
        _arr(inside_w_shape, np.array([1, 1, 1, W30], dtype=np.int64)),
    ]
    return nodes, inits


BUILDERS = {
    "identity": (set(), build_identity),
    "transpose": (set(), build_transpose),
    "swap_colors": ({"c1", "c2"}, build_swap_colors),
    "replace_color": ({"src", "dst"}, build_replace_color),
    "flip_h": (set(), build_flip_h),
    "flip_v": (set(), build_flip_v),
    "rotate180": (set(), build_rotate180),
    "rotate90": (set(), build_rotate90),
    "rotate180_static": ({"h", "w"}, build_rotate180_static),
    "rotate90_static": ({"h", "w"}, build_rotate90_static),
    "count_color": ({"c"}, build_count_color),
    "dominant_color": (set(), build_dominant_color),
    "fill_bg": ({"bg_color"}, build_fill_bg),
    "bbox_crop": ({"target_color"}, build_bbox_crop),
    "tile_h": ({"n"}, build_tile_h),
    "tile_v": ({"n"}, build_tile_v),
    "largest_blob": ({"target_color"}, build_largest_blob),
    "select_channel": ({"cond_ch", "if_ch", "else_ch"}, build_select_channel),
    "fill": ({"c"}, build_fill),
    "shift_down": ({"k"}, build_shift_down),
    "shift_right": ({"k"}, build_shift_right),
    "crop": ({"r0", "c0", "h", "w"}, build_crop),
    "mask_foreground": (set(), build_mask_foreground),
    "flood_fill": ({"target_color", "fill_color"}, build_flood_fill),
    "thicken": ({"target_color"}, build_thicken),
    "hollow_out": ({"target_color"}, build_hollow_out),
    "remove_color": ({"target_color"}, build_remove_color),
    "trim_border": ({"k"}, build_trim_border),
}
