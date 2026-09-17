"""Depth-≤2 brute-force DSL program search.

For each task, we enumerate DslPrograms of length 1..max_depth, evaluate each
on the *visible train examples* using the cheap Python reference, and keep the
shortest program(s) that match every example exactly. Test split (and
arc-gen if asked) is then verified for the surviving program(s).

Programs are pruned aggressively:
- Parameter values are drawn from the task's actual colors / shapes.
- We short-circuit on the first failed example.
- depth-1 hits short-circuit depth-2 enumeration.

Design choice: the search runs entirely on Python refs (no ONNX compile per
candidate). Compile + isolated-eval happens only for candidates that pass
visible examples, in a separate pipeline.
"""
from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterator

import numpy as np

THIS = pathlib.Path(__file__).resolve()
ROOT = THIS.parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from tools.dsl.compiler import DslOp, DslProgram  # noqa: E402
from tools.dsl.primitives import REF_FUNCS         # noqa: E402


# ---------------------------------------------------------------------------
# Task signatures
# ---------------------------------------------------------------------------

@dataclass
class TaskSig:
    colors_in: list[int]
    colors_out: list[int]
    shapes_in: list[tuple[int, int]]
    shapes_out: list[tuple[int, int]]


def signature(train: list[dict]) -> TaskSig:
    colors_in: set[int] = set()
    colors_out: set[int] = set()
    shapes_in: set[tuple[int, int]] = set()
    shapes_out: set[tuple[int, int]] = set()
    for ex in train:
        gi = np.array(ex["input"], dtype=np.int64)
        go = np.array(ex["output"], dtype=np.int64)
        colors_in.update(int(c) for c in np.unique(gi))
        colors_out.update(int(c) for c in np.unique(go))
        shapes_in.add(tuple(gi.shape))
        shapes_out.add(tuple(go.shape))
    return TaskSig(
        colors_in=sorted(colors_in),
        colors_out=sorted(colors_out),
        shapes_in=sorted(shapes_in),
        shapes_out=sorted(shapes_out),
    )


# ---------------------------------------------------------------------------
# Primitive enumeration
# ---------------------------------------------------------------------------

# Static per-primitive cost estimates (bytes, measured on 5x5 sample inputs;
# see tools/dsl/check_primitive_cost.py). Lower is cheaper. For shape-aware
# primitives we pick a plausible mid-range value.
COST_HINT: dict[str, int] = {
    "identity": 0,
    "transpose": 0,
    "swap_colors": 10,
    "replace_color": 100,
    "fill_bg": 100,
    "fill": 44115,
    "count_color": 170,
    "select_channel": 48622,
    "flip_h": 948,
    "flip_v": 948,
    "rotate180_static": 1500,    # avg between 3x3 and 9x9
    "rotate90_static": 1500,
    "shift_down": 37274,
    "shift_right": 37274,
    "crop": 500,                  # depends on crop size; ~384 for 3x3, ~824 for 4x5
    "dominant_color": 8236,       # v1 FLOAT pipeline (Greater+Cast, not Equal+Not)
    "rotate180": 37896,
    "rotate90": 36948,
    "bbox_crop": 82048,
    "tile_h": 41966,             # Gather + mask on 30-wide [1,10,30,30]
    "tile_v": 41966,
    "largest_blob": 97089,       # 8× Conv iterations + seed finding
    "mask_foreground": 51316,    # ReduceSum+Sub+Greater+Cast+Tile+Mul
    "remove_color": 100,         # alias for replace_color(src=X,dst=0)
    "thicken": 136824,           # extract + Conv + Or + merge + mask_to_content
    "hollow_out": 15322,         # extract + Conv + Greater+And+Not+ 1x1 Conv
    "flood_fill": 192631,        # 8× Conv dilations + merge + mask_to_content
    "trim_border": 78453,        # bbox-aware Gather×2 + mask
}


def estimated_cost(op: DslOp) -> int:
    """Rough total-cost estimate (memory + params) for a single op. Used to
    pick the cheaper of two visible-pass-equivalent programs."""
    base = COST_HINT.get(op.name, 100)
    if op.name in {"rotate180_static", "rotate90_static"}:
        h, w = op.params.get("h", 5), op.params.get("w", 5)
        # Memory ≈ 3 * 10 * h * w * 4 (three [1,10,h,w] FLOAT intermediates).
        base = 30 + 120 * h * w + (h + w)
    if op.name == "crop":
        h, w = op.params.get("h", 5), op.params.get("w", 5)
        # One [1,10,h,w] intermediate from Slice.
        base = 24 + 40 * h * w
    return base


def program_estimated_cost(program: DslProgram) -> int:
    return sum(estimated_cost(op) for op in program.ops)


NULLARY = (
    "identity",
    "transpose",
    "flip_h",
    "flip_v",
    "rotate180",
    "rotate90",
    "dominant_color",
)

NULLARY_NO_ROT = (
    "identity",
    "transpose",
    "flip_h",
    "flip_v",
    "dominant_color",
)


def enumerate_depth1(sig: TaskSig, lenient: bool = False) -> Iterator[DslOp]:
    """Yield candidate depth-1 ops.

    `lenient=True` widens the parameter space to `cols_in ∪ cols_out` for
    every primitive — needed when this op is op2 in a depth-2 search, since
    the intermediate may carry colors that the original input did not."""
    # Nullary primitives. Prefer rotate_static where shapes are known (single
    # input shape that matches a single output shape).
    use_static_rot = (len(sig.shapes_in) == 1)
    if use_static_rot:
        for op in NULLARY_NO_ROT:
            yield DslOp(op, {})
        h, w = sig.shapes_in[0]
        if 1 <= h <= 30 and 1 <= w <= 30:
            yield DslOp("rotate180_static", {"h": h, "w": w})
            yield DslOp("rotate90_static", {"h": h, "w": w})
    else:
        for op in NULLARY:
            yield DslOp(op, {})

    cols_in = sig.colors_in
    cols_out = sig.colors_out
    union = sorted(set(cols_in) | set(cols_out))
    src_pool = union if lenient else cols_in
    dst_pool = union if lenient else cols_out

    for src in src_pool:
        for dst in dst_pool:
            if src != dst:
                yield DslOp("replace_color", {"src": src, "dst": dst})

    for i, c1 in enumerate(union):
        for c2 in union[i + 1:]:
            yield DslOp("swap_colors", {"c1": c1, "c2": c2})

    for c in (union if lenient else cols_in):
        yield DslOp("count_color", {"c": c})

    for bg in dst_pool:
        if bg != 0:
            yield DslOp("fill_bg", {"bg_color": bg})

    for tc in (union if lenient else cols_in):
        yield DslOp("bbox_crop", {"target_color": tc})

    # Tile primitives — try n=2..4 (moderate repeats)
    for n in (2, 3, 4):
        yield DslOp("tile_h", {"n": n})
        yield DslOp("tile_v", {"n": n})

    # Largest blob
    for tc in (union if lenient else cols_in):
        yield DslOp("largest_blob", {"target_color": tc})

    # Fill — iterate over union colors
    for c in union:
        yield DslOp("fill", {"c": c})

    # Shift primitives — small shifts
    for k in (1, 2, 3, 4, 5):
        yield DslOp("shift_down", {"k": k})
        yield DslOp("shift_right", {"k": k})

    # Select channel — useful per-cell conditional color replacement.
    # For each cond_ch in union, for each if_ch in dst_pool, for each else_ch in union:
    # The common pattern is cell-wise color mapping.
    for cond_ch in union:
        for if_ch in dst_pool:
            for else_ch in (union if lenient else cols_in):
                if if_ch != else_ch:
                    yield DslOp("select_channel", {"cond_ch": cond_ch, "if_ch": if_ch, "else_ch": else_ch})

    # ---- Batch 2 primitives ----
    # mask_foreground — no parameters
    yield DslOp("mask_foreground", {})

    # remove_color — remove target_color from the grid
    for tc in (union if lenient else cols_in):
        if tc != 0:
            yield DslOp("remove_color", {"target_color": tc})

    # thicken — add 1-pixel border to target_color
    for tc in (union if lenient else cols_in):
        if tc != 0:
            yield DslOp("thicken", {"target_color": tc})

    # hollow_out — remove interior of target_color blobs
    for tc in (union if lenient else cols_in):
        if tc != 0:
            yield DslOp("hollow_out", {"target_color": tc})

    # flood_fill — grow target_color and fill with fill_color
    for tc in (union if lenient else cols_in):
        for fc in union:
            if fc != tc:
                yield DslOp("flood_fill", {"target_color": tc, "fill_color": fc})

    # trim_border — remove k from each side
    for k in (1, 2, 3):
        yield DslOp("trim_border", {"k": k})

    # Crop — extract subgrids matching output shapes from input shapes.
    # Try common top-left positions; the runtime shape check filters
    # invalid combinations.
    for h_out, w_out in sig.shapes_out:
        for h_in, w_in in sig.shapes_in:
            if h_in >= h_out and w_in >= w_out:
                max_r0 = h_in - h_out
                max_c0 = w_in - w_out
                r0_limit = min(max_r0, 3)
                c0_limit = min(max_c0, 3)
                for r0 in range(r0_limit + 1):
                    for c0 in range(c0_limit + 1):
                        yield DslOp("crop", {"r0": r0, "c0": c0, "h": h_out, "w": w_out})


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_program(program: DslProgram, grid: np.ndarray) -> np.ndarray | None:
    """Apply program ops left-to-right via Python refs. Returns None on
    any error (shape mismatch, invalid param, etc)."""
    cur = grid
    for op in program.ops:
        ref = REF_FUNCS.get(op.name)
        if ref is None:
            return None
        try:
            cur = ref(cur, **op.params)
        except Exception:
            return None
        if cur is None or cur.size == 0:
            # bbox_crop legitimately returns (0,0). Subsequent ops on that
            # would be degenerate.
            return cur
        if cur.shape[0] > 30 or cur.shape[1] > 30:
            return None
    return cur


def visible_pass(program: DslProgram, examples: list[dict]) -> bool:
    for ex in examples:
        gi = np.array(ex["input"], dtype=np.int64)
        go = np.array(ex["output"], dtype=np.int64)
        out = run_program(program, gi)
        if out is None:
            return False
        if out.shape != go.shape:
            return False
        if not np.array_equal(out, go):
            return False
    return True


# ---------------------------------------------------------------------------
# Search entry point
# ---------------------------------------------------------------------------

@dataclass
class SearchHit:
    program: DslProgram
    depth: int
    train_pass: bool
    test_pass: bool
    arcgen_pass_count: tuple[int, int]


def search_program(
    task_id: int,
    comp_dir: pathlib.Path,
    max_depth: int = 2,
    verify_arcgen: bool = False,
) -> SearchHit | None:
    """Return the shortest DslProgram that fits ALL train examples for the
    task, or None if no program of depth <= max_depth works.

    Test pass-rate is reported for diagnostics; arc-gen pass is optional
    (slow to evaluate on every candidate so opt-in)."""
    path = comp_dir / f"task{task_id:03d}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    train = data.get("train", [])
    test = data.get("test", [])
    arcgen = data.get("arc-gen", []) if verify_arcgen else []
    if not train:
        return None

    sig = signature(train)
    depth1_ops = list(enumerate_depth1(sig, lenient=False))
    # For op2 in depth-2 we need to be lenient because the intermediate may
    # carry colors that the original input did not.
    depth2_op2_ops = list(enumerate_depth1(sig, lenient=True))

    def finalize(program: DslProgram, depth: int) -> SearchHit:
        test_ok = visible_pass(program, test) if test else False
        arc_pass = (0, 0)
        if verify_arcgen and arcgen:
            arc_pass = (
                sum(1 for ex in arcgen if visible_pass(program, [ex])),
                len(arcgen),
            )
        return SearchHit(program, depth, True, test_ok, arc_pass)

    # Depth 1 — collect ALL hits, return cheapest.
    depth1_hits: list[DslProgram] = []
    for op in depth1_ops:
        program = DslProgram.of([op])
        if visible_pass(program, train):
            depth1_hits.append(program)
    if depth1_hits:
        depth1_hits.sort(key=program_estimated_cost)
        return finalize(depth1_hits[0], 1)

    if max_depth < 2:
        return None

    # Depth 2 — cache op1 outputs on each train input, then iterate op2.
    train_inputs = [np.array(ex["input"], dtype=np.int64) for ex in train]
    train_outputs = [np.array(ex["output"], dtype=np.int64) for ex in train]
    intermediate_cache: dict[tuple[str, tuple], list[np.ndarray | None]] = {}

    def op_key(op: DslOp) -> tuple[str, tuple]:
        return (op.name, tuple(sorted(op.params.items())))

    def cached_apply(op: DslOp, grids: list[np.ndarray]) -> list[np.ndarray | None]:
        k = op_key(op)
        if k in intermediate_cache:
            return intermediate_cache[k]
        outs: list[np.ndarray | None] = []
        ref = REF_FUNCS[op.name]
        for g in grids:
            try:
                out = ref(g, **op.params)
                if out is None or out.shape[0] > 30 or out.shape[1] > 30:
                    outs.append(None)
                else:
                    outs.append(out)
            except Exception:
                outs.append(None)
        intermediate_cache[k] = outs
        return outs

    depth2_hits: list[DslProgram] = []
    for op1 in depth1_ops:
        mids = cached_apply(op1, train_inputs)
        if any(m is None or m.size == 0 for m in mids):
            continue
        for op2 in depth2_op2_ops:
            if op2.name == "identity":
                continue
            ref2 = REF_FUNCS[op2.name]
            ok = True
            for mid, target in zip(mids, train_outputs):
                try:
                    out = ref2(mid, **op2.params)
                except Exception:
                    ok = False
                    break
                if out is None or out.shape != target.shape or not np.array_equal(out, target):
                    ok = False
                    break
            if ok:
                depth2_hits.append(DslProgram.of([op1, op2]))

    if depth2_hits:
        depth2_hits.sort(key=program_estimated_cost)
        return finalize(depth2_hits[0], 2)

    return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--comp-dir", default=str(ROOT / "data/neurogolf-2026/raw"))
    p.add_argument("--max-depth", type=int, default=2)
    args = p.parse_args()
    hit = search_program(args.task, pathlib.Path(args.comp_dir), max_depth=args.max_depth)
    if hit is None:
        print(f"task{args.task:03d}: NO HIT")
    else:
        print(f"task{args.task:03d}: depth={hit.depth} test_pass={hit.test_pass}")
        print(f"  program: {hit.program}")
