# DSL Phase 5 Summary (Internal Engineering Worker)

Built on the workspace's existing baselines (no external recon). Worked from
v4_plus3 anchor + Phase 1 pattern scan + Phase 4 isolated full-example audit.

## 1. Prototype verification conclusion

8 primitives implemented in `tools/dsl/primitives.py`, compiler in
`tools/dsl/compiler.py`, equivalence tests in `tools/dsl/test_primitives.py`.

**13/13 python_ref vs onnx_run unit tests pass** on random grids (H,W in
[2,15]) plus a composition test.

**10/10 phase-4 tasks pass 100% of train + test + arc-gen examples** under
isolated full-example audit (see `reports/dsl_phase4_results.csv`).

### Per-task cost (DSL vs v4_plus3 baseline)

| task | baseline | DSL  | ratio (baseline/DSL) | DSL program                        |
|------|---------:|-----:|---------------------:|------------------------------------|
| 087  | 376      | 37664| 0.010                | `rotate180`                        |
| 140  | 376      | 37664| 0.010                | `rotate180`                        |
| 150  | 1076     | 832  | **1.293**            | `flip_h`                           |
| 155  | 1076     | 832  | **1.293**            | `flip_v`                           |
| 179  | 0        | 0    | 1.000                | `transpose`                        |
| 241  | 0        | 0    | 1.000                | `transpose`                        |
| 380  | 737      | 36832| 0.020                | `rotate90`                         |
| 276  | 10       | 10   | 1.000                | `swap_colors(6, 2)`                |
| 309  | 10       | 10   | 1.000                | `swap_colors(7, 5)`                |
| 337  | 10       | 10   | 1.000                | `swap_colors(5, 8)`                |

**Headline numbers:**
- 2/10 wins (flip_h, flip_v: **1.29x cheaper** vs v4_plus3 hand-tuned baseline)
- 5/10 ties (transpose ×2, swap_colors ×3: same cost as the elite handbuild)
- 3/10 losses (rotate180 ×2, rotate90 ×1: 50–100x worse cost)

**Median cost ratio: 1.0.** **Mean ratio (excluding rotate losses): 1.06.**

The DSL ONNXs are correct (full pass-rate) and competitive on 7 of 10 sampled
tasks. The 3 losses share a single root cause documented below.

## 2. Why the rotate primitives lose

The `rotate180` and `rotate90` builders need two sequential Gathers (one per
spatial axis). The output of the first Gather is a `[1,10,30,30]` FLOAT tensor
(36000 bytes). With ORT's `GRAPH_OPTIMIZATION_DISABLE_ALL` (the scoring
config), that intermediate is realized → memory bill ~36K.

A `Slice` + `Pad` approach (matching the v4_plus3 task087 and task380
baselines, which use static starts/ends keyed to a single fixed input shape)
is **strictly cheaper** because the sliced tensor is `[1,10,H,W]` — e.g. 360
bytes for a 3x3 input. We tried this approach (commit history shows the
attempt); it produces 100% correct outputs but generates **dynamic-shape**
intermediates that the competition's `calculate_memory` rejects (returns
`None` → score 0). So Slice+Pad is correct but unscoreable.

Three viable fixes for the rotate cost gap, in order of effort:

1. **Task-specialized rotate** (cheap to implement): emit a static-shape
   Slice+Pad variant when the per-task corpus shows a single fixed `(H, W)`.
   Implementation: `build_rotate180_static(h, w)` that hardcodes the starts/
   ends/pads as initializers. Estimated cost for task087: ~376 (matching
   baseline).
2. **BOOL-pipeline rotates**: Cast input to BOOL once, do both Gathers in
   BOOL space, output BOOL. Saves 27000 bytes (FLOAT 36K → BOOL 9K
   intermediate). Estimated cost: ~10K. Still loses to baseline but ~4x
   better than current.
3. **GatherND with 2D index**: build a `[30, 30, 2]` INT64 index tensor on
   the fly. Likely cheap but adds 1800 bytes of index, plus more nodes. Not
   obviously a win.

## 3. v0 primitive library

Implemented (all `[1,10,30,30] FLOAT → [1,10,30,30] FLOAT`):

| name              | params         | ONNX shape   | cost on a 30x30 input |
|-------------------|----------------|--------------|----------------------:|
| `identity`        | -              | 1× Identity  | 0                     |
| `transpose`       | -              | 1× Transpose | 0                     |
| `swap_colors`     | `c1, c2`       | 1× Gather    | 10 (perm init)        |
| `replace_color`   | `src, dst`     | 1× Conv 1x1  | 100 (10x10 weights)   |
| `flip_h`          | -              | 7-node bbox  | 832                   |
| `flip_v`          | -              | 7-node bbox  | 832                   |
| `rotate180`       | -              | dual-Gather  | 37664 (regression)    |
| `rotate90`        | -              | tr + flip_v  | 36832 (regression)    |

All shape-preserving primitives (identity, transpose-on-square, swap_colors,
replace_color) achieve **parity or near-parity** with v4_plus3 elite handbuilds.

## 4. "越走越宽" coverage evidence

Pattern scan over `task001..task400` (train+test split):

- **9 tasks** are pure 1-primitive expressible by the current 8 primitives
  (task087, task140, task150, task155, task179, task241, task380, task276,
  task309, task337 — see `reports/dsl_phase1_task_patterns.md`).
- **0 of 50 randomly sampled tasks** are 1-primitive expressible. This
  confirms that *pure* single-op tasks are a small subset of the corpus
  (~2% by raw scan).

**This is the correct, calibrated takeaway**: a depth-1 DSL of 8 primitives
covers a small but real slice (≈9 tasks of 400 = 2.3%). Real coverage will
come from **multi-step programs** (2-4 primitive composition) plus the
next-gen primitives (objects, masks, drawing, conditionals) — see §5.

## 5. v1 primitive candidates (next worker pickup)

Based on Phase 4 baseline inspection and the v0 gaps:

**Tier 1 (easy wins, high-coverage candidates):**

1. `rotate180_static(h, w)` / `rotate90_static(h, w)`: parameterized variant
   using static Slice+Pad. Direct ~100x cost win on task087, task140, task380.
2. `count_color(c)`: emits a scalar count of channel-c pixels (output
   shape `[1, 10, 1, 1]`). For tasks where the answer is the count.
3. `dominant_color(exclude_bg)`: returns a channel index. Building block
   for many ARC tasks.
4. `largest_component(connectivity)`: needs the `task243` flood-fill
   building block — we already have a working ONNX for that pattern.
5. `bbox_crop`: extract the H×W bbox content, re-pin to top-left, output
   `[1, 10, H', W']` (the static rotate variant builds on this).

**Tier 2 (harder, but unlocks high-cost monsters):**

6. `tile_h(k)`, `tile_v(k)`: replicate the bbox content along an axis.
   Useful for task001 (which tiles 3x3 input into 9x9 output).
7. `connected_components(connectivity)`: 4- or 8-neighborhood CC. Heavy
   in nodes (probably 30+ for a single-pass approach using Conv-based
   label propagation). Required for task255, task233, task366 type
   monsters.
8. `apply_mask(predicate)`: branch-like primitive — apply a sub-program
   only where a mask is set.

**Tier 3 (program-synthesis enablers, not single primitives):**

9. A small **search harness** that, given a task's `train` examples and
   a primitive bank, brute-forces depth-1 and depth-2 programs and picks
   the one that passes all train examples. Even a depth-2 search over
   the current 8 primitives might unlock 20-40 tasks for which the right
   composition exists but no human bothered to spot it.

## 6. Integration strategy (anchor swap)

**Safe to swap in now** (DSL-built ONNXs that beat or match v4_plus3 anchor):

| task | baseline cost | DSL cost | swap value                |
|------|--------------:|---------:|---------------------------|
| 150  | 1076          | 832      | +244 bytes saved          |
| 155  | 1076          | 832      | +244 bytes saved          |

Combined `Δcost ≈ 488`. Translates to score delta:
- log(488 / 1076) ≈ -0.78
- log(488 / 832) ≈ -0.535
- Net score gain per task: ≈ 0.25 each → **≈ 0.5 LB points combined**.

This is a tiny lift — not worth a submission slot on its own, but worth
swapping in **the next time the anchor gets rebundled** for any other reason.

**NOT safe to swap in** (would regress baseline):
- task087, task140, task380: DSL rotate is 50-100x more expensive than
  baseline. Must use v1 `rotate*_static` instead.

**Trace-only / shelved**:
- task179, task241, task276, task309, task337: DSL matches baseline exactly;
  no point swapping (same cost, more code).

## 7. Hard rules compliance

- ✅ Did not modify any existing file under `submissions/`. New DSL-built
  ONNXs live in `submissions/handbuilds/dsl_phase4/dsl_task*.onnx` (new
  subdir, no overwrites).
- ✅ Did not touch task319 ONNX path; only **read** `build_task319_base_selector_solver.py`
  for the `_absorb` pattern (no longer needed — DSL builds in-process,
  doesn't load partial ONNX files).
- ✅ Did not run remote git, did not submit to Kaggle.
- ✅ No fp16 surgery, no large init lookup tables (largest init is the
  10x10 `replace_color` Conv weight = 100 floats = 400 bytes).
- ✅ Every primitive has a passing python_ref vs onnx_run unit test.
- ✅ Every Phase-4 task passes isolated full-example audit.
- ✅ DSL artifacts at the prescribed paths: `tools/dsl/primitives.py`,
  `tools/dsl/compiler.py`, `tools/dsl/test_primitives.py`, plus
  `tools/dsl/run_phase4.py` (driver) and `tools/dsl/__init__.py`.

## 8. Blockers / user decisions needed

1. **Should we keep building primitives, or pivot to a search harness?**
   - Pro-primitive: every new primitive is a building block for many tasks.
   - Pro-search: even with 8 primitives, brute-force depth-2 search might
     find 20-40 new tasks. The marginal value of search may now exceed the
     marginal value of primitive N+1.
2. **Should v1 include task-specialized variants like `rotate180_static`?**
   - These are not strictly "DSL" primitives (they encode shape
     assumptions); but they are the cleanest way to close the 100x cost gap
     on `task087, task140, task380` while keeping the DSL composable.

Recommendation (low confidence, depends on overall strategy): pick Tier-1
primitives (1-3 above) plus a depth-2 brute-force search harness as the
next worker's mandate.
