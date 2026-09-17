# Task319 ONNX Feasibility Notes

## Current Python Status

- Prototype: [`tools/prototype_task319.py`]($HOME/Desktop/kaggleonnx/tools/prototype_task319.py)
- Visible coverage: **267/267**
- Current-best task319 score in our 6122 bundle: about **7.914**
- Theoretical upside if a cheap semantic ONNX closes: roughly **+7 leaderboard points**

## Strong invariants

- Every visible example has exactly **3 non-background colors**.
- Output is always exactly the bbox crop of **one input color** with the crop background reset to the dominant input color.
- Input max shape across visible examples: **19x19**
- Output max shape across visible examples: **5x5**
- Compressed patch shapes stay small; the most common are:
  - `(3, 3)`: 173 occurrences
  - `(4, 3)`: 102
  - `(3, 4)`: 88
  - `(2, 2)`: 67
  - `(4, 4)`: 61

These bounds are encouraging for a semantic graph. The problem is not spatial scale; it is selector logic.

## Why a lookup graph is a bad idea

- Distinct large compressed signatures on visible examples: **173**
- Distinct small compressed signatures: **422**
- Distinct `(largest signature, small-candidate pair)` families: **267 / 267**

Practical takeaway: the pair-family space is effectively unique per example. A direct visible-signature lookup would be another hidden-unsafe trap, just like the earlier failed cherry-picks / visible-family tricks on other tasks.

## What actually carries the Python rule

The current selector is layered:

1. rank the two smaller candidates by
   - `score`
   - `match_largest`
   - `-compressed_row_range`
   - `-aspect_gap`
   - `-compressed_row_var`
   - `-compressed_density`
2. narrow fallback: if exactly one smaller candidate is a same-area/same-cell mirrored echo of the largest object, choose the largest object itself
3. narrow fallback: if aspect-gap advantage is at least `1/3`, switch
4. narrow fallback: if local motif agreement with the largest compressed patch is better enough (`2x2` IoU advantage >= `1/6`, or stronger `3x2` overlap count), switch
5. narrow fallback: if both compressed frames match and one candidate is a strict one-cell superset, choose the denser superset

Trigger counts on visible examples:

- mirrored-largest fallback: **1** example (`189`)
- aspect-gap switch: **6** examples (`8, 85, 169, 193, 252, 266`)
- motif switch: **4** examples (`8, 25, 183, 191`)
- compressed-superset fallback: **1** example (`242`)

Without any fallback, the base selector is only **257/267**.

## Hidden-safety read

Good news:

- The late numeric thresholds are **not razor-thin**:
  - aspect switch stays perfect on visible examples from about **`1/3` to `7/12`**
  - motif IoU switch stays perfect from about **`1/6` upward**, as long as the discrete `3x2` motif-count fallback is kept
- The rule is not keyed by example ID, hash, or visible-family table.

Remaining caution:

- The selector is still a layered heuristic, not one clean invariant.
- Hidden risk now comes from whether these fallback relations stay stable off-visible, not from visible underfitting.

## ONNX path that still looks realistic

The semantic graph path should be staged, not attempted as one giant leap:

1. **Per-color bbox extraction for channels 1..9**
   - compute row/col occupancy
   - recover bbox min/max per color
   - produce each color's bbox crop mask

2. **Largest / two-smaller candidate identification**
   - sort by bbox area, then cells
   - this should be fully graphable with fixed-size reductions and comparisons

3. **Compressed-patch feature extraction**
   - we likely do **not** need full arbitrary-shape compression first
   - we may be able to derive the needed small features directly:
     - compressed row range
     - compressed row variance proxy
     - compressed density
     - local `2x2` / `3x2` motif counts

4. **Selector logic**
   - base lexicographic ranking
   - the 4 narrow overrides

5. **Output rendering**
   - paint the chosen patch color inside its bbox
   - fill zeros inside the bbox with background

## First concrete graph milestone

Added a feature-probe builder:

- [`tools/build_task319_features.py`]($HOME/Desktop/kaggleonnx/tools/build_task319_features.py)
- output artifact: [`submissions/handbuilds/task319_features.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task319_features.onnx)

What it already graphizes:

- per-color pixel counts
- per-color bbox min/max rows and cols
- per-color bbox area
- background-color detection by dominant count
- exact ordering of the 3 non-background colors by `(area, cells, -color)` to recover `largest / two-smaller`

Verification status:

- verifies against Python on **267/267** visible examples
- current artifact size: about **4 KB**
- node count: **35**

Why this matters:

- It proves the selector's *front half* is cheap and robust to graphize.
- The remaining ONNX risk is concentrated in the **compressed-shape / motif comparison** logic, not in basic object extraction or candidate ordering.

## Compression-feature breakthrough

Added a small audit:

- [`tools/task319_graph_feature_audit.py`]($HOME/Desktop/kaggleonnx/tools/task319_graph_feature_audit.py)

Result:

- on **801** visible task319 objects, the Python `compress_runs(...)` output is reproduced **exactly** by a graph-friendly definition:
  - keep the first row,
  - keep any row that differs from the previous row,
  - then do the same for columns.
- The derived features also match **exactly** on all 801 objects:
  - `compressed_density`
  - `compressed_row_var`
  - `compressed_row_range`

Why this matters:

- These three features were previously “good Python heuristics”.
- They are now promoted to **credible ONNX targets**, because we have an exact, local, adjacency-based interpretation for them instead of an opaque helper function.
- This sharply reduces uncertainty in the next graphization stage: the base selector no longer depends on any fundamentally ungraphable primitive.

## Second concrete graph milestone

Added a compression-feature probe:

- [`tools/build_task319_compress_features.py`]($HOME/Desktop/kaggleonnx/tools/build_task319_compress_features.py)
- output artifact: [`submissions/handbuilds/task319_compress_features.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task319_compress_features.onnx)

What it graphizes exactly:

- per-color `compressed_cells`
- per-color `compressed_rows`
- per-color `compressed_cols`
- per-color `compressed_density`
- per-color `compressed_row_range`
- per-color `compressed_row_var`

Verification status:

- verifies against Python on **267/267** visible examples
- node count: **103**
- file size: about **12 KB**

Practical meaning:

- The central compression-derived features of the Python selector are now not only explainable, but **implemented and validated in ONNX**.
- That materially changes `task319` from “interesting closed Python rule” to “serious semantic ONNX candidate”.
- The remaining graph work is now mostly:
  - wiring these feature tensors into the base lexicographic selector,
  - then deciding how much of the later fallback stack is worth encoding before cost blows up.

## Variant-comparison canonicalization

Added an audit:

- [`tools/task319_variant_audit.py`]($HOME/Desktop/kaggleonnx/tools/task319_variant_audit.py)

Result:

- the `score` / `match_largest` relation can be rewritten as a **fixed-size canonical comparison**
  - represent every compressed/variant patch on a `5x5` zero-padded canvas
  - carry the original `(height, width)` alongside that canvas
  - compare candidate variant families to other compressed patches by `(shape, padded-grid)` equality
- this reproduces Python `score` and `match_largest` **exactly on all 801 visible objects**

Helpful bounds:

- maximum compressed shape is only **`5x5`**
- maximum distinct variant count for any object is **72**

Why this matters:

- `score` / `match_largest` looked like the last awkward, dynamic part of the base selector.
- They now have a concrete finite representation that does not require unbounded shape logic.
- This does **not** mean the final ONNX selector is already cheap, but it moves the problem from “maybe impossible to graphize cleanly” to “finite comparison bank / engineering cost”.

## Base-selector closure audit

Added:

- [`tools/task319_base_selector_audit.py`]($HOME/Desktop/kaggleonnx/tools/task319_base_selector_audit.py)

What it proves:

- if we rebuild the base selector entirely from the graph-friendly pieces
  - adjacency-defined compression,
  - exact compression-derived features,
  - fixed-bank variant comparison on `5x5 + shape`,
- then the resulting choice matches the original Python base selector **exactly on all 267 visible examples**.

Practical meaning:

- The **entire base selector** is now closed as a graph-friendly formulation.
- Remaining semantic ONNX work is no longer about “can we express the base rule?”.
- It is now about:
  - implementing that closed base selector efficiently in ONNX,
  - then deciding which of the later narrow fallbacks are worth encoding on top.

## Main graphization risk

The risky part is **not** output size or color count. It is implementing enough of the compression/motif logic without producing another `task363`-style node explosion.

So the next practical milestone is:

- extend the existing feature probe into the next stage:
  - derive graph-friendly compressed-shape surrogates (`row_range`, density, simple motif counts)
  - test whether the **base selector** can be reproduced cheaply before attempting the later overrides

If that extension stays small, `task319` remains a serious Kaggle candidate. If motif/compression logic causes a rapid node explosion, we may need to pause `task319` at the Python-closed stage and redirect effort to another high-upside task.

## Packed-compression bridge milestone

Added:

- [`tools/build_task319_packed_compressed.py`]($HOME/Desktop/kaggleonnx/tools/build_task319_packed_compressed.py)
- output artifact: [`submissions/handbuilds/task319_packed_compressed.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task319_packed_compressed.onnx)

What it now graphizes exactly:

- per-color compressed patches packed into a fixed **`5x5`** top-left-aligned canvas
- per-color absolute kept-row indices for the first `PACK=5` compressed rows
- per-color absolute kept-column indices for the first `PACK=5` compressed columns

Verification status:

- verifies on **267/267** visible examples
- node count: **93**
- file size: about **11.3 KB**

Why this matters:

- It closes the last missing bridge between the compression-feature probe and the canonical variant-bank formulation.
- We no longer need to talk about compressed shapes abstractly; we now have a concrete ONNX tensor representation that matches the Python semantics exactly:
  - `packed_compressed[1, color, 5, 5]`
  - plus explicit kept-row / kept-column index tensors
- That makes the next engineering step much cleaner: build the actual base-selector ONNX around
  - `5x5` packed compressed canvases,
  - explicit compressed shape metadata,
  - and a fixed transform bank for the two small candidates.

Implementation note:

- The tricky bug was not the compression rule itself; it was masking invalid/background channels and aligning the Python verifier with bbox-based adjacency compression instead of occupied-row-only shortcuts.
- After fixing that, the probe now agrees exactly with visible data.

## Packed-interface selector audit

Added:

- [`tools/task319_packed_selector_audit.py`]($HOME/Desktop/kaggleonnx/tools/task319_packed_selector_audit.py)

Result:

- using only
  - `packed_compressed` from the packed-compression probe, and
  - `compressed_rows / compressed_cols` from the compression-feature probe,
  we can reconstruct the full **base selector** in Python with **0 mismatches on 267/267** visible examples.

Why this matters:

- This is a stronger interface-level proof than the earlier conceptual audits.
- It shows the future ONNX selector does not need hidden dynamic state from the original Python objects.
- The packed `5x5` canvas plus explicit compressed shape metadata is already a sufficient contract for:
  - fixed transform-bank generation,
  - exact `score` / `match_largest` recovery,
  - and the full base lexicographic ranking.

Practical next step:

- Build a true ONNX base-selector probe that consumes these fixed tensors and emits either
  - the chosen candidate color, or
  - the final base-selector output patch directly.

## Candidate-feature head milestone

Added:

- [`tools/build_task319_candidate_features.py`]($HOME/Desktop/kaggleonnx/tools/build_task319_candidate_features.py)
- output artifact: [`submissions/handbuilds/task319_candidate_features.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task319_candidate_features.onnx)

What it graphizes exactly:

- candidate colors in the `top3_colors[1:]` slots
- candidate `aspect_gap` relative to the largest object
- candidate `compressed_row_range`
- candidate `compressed_row_var`
- candidate `compressed_density`

Verification status:

- verifies on **267/267** visible examples
- node count: **29**
- file size: about **3.5 KB**

Why this matters:

- It cleanly peels the base selector into two parts:
  1. a very cheap ONNX head for all non-transform ranking features,
  2. the remaining harder `score` / `match_largest` comparison block.
- This is a useful engineering split because it keeps the next builder sharply focused on the true hard part.

Important diagnostic:

- If we rank the two smaller candidates **without** `score` / `match_largest`, using only
  - `-compressed_row_range`
  - `-aspect_gap`
  - `-compressed_row_var`
  - `-compressed_density`,
  then agreement with the full Python base selector is only **161/267**.

Practical meaning:

- The remaining transform-bank / canonical-match block is not a small polish term.
- It carries most of the discriminative power of the base selector, so that is now the right next implementation target.

## Match-feature probe milestone

Added:

- [`tools/build_task319_match_features.py`]($HOME/Desktop/kaggleonnx/tools/build_task319_match_features.py)
- output artifact: [`submissions/handbuilds/task319_match_features.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task319_match_features.onnx)

What it graphizes exactly:

- candidate `score`
- candidate `match_largest`

Input contract:

- `top3_colors`
- `packed_compressed`
- `compressed_rows`
- `compressed_cols`

Implementation idea:

- use a fixed bank of **72** transform slots (rotation / mirror / one-edge trim combinations),
- precompute shape-specific index maps for all compressed shapes up to `5x5`,
- materialize candidate variant canvases by `GatherElements` from the packed `5x5` source,
- compare those variant canvases against the three packed compressed targets by exact `(shape, padded-grid)` equality,
- then reduce over variants to recover the per-candidate `score` / `match_largest`.

Verification status:

- verifies on **267/267** visible examples
- node count: **44**
- file size: about **394 KB**

Why this matters:

- This closes the actual hard part of the base selector.
- We now have ONNX probes for **all six** base-ranking features:
  - `score`
  - `match_largest`
  - `compressed_row_range`
  - `aspect_gap`
  - `compressed_row_var`
  - `compressed_density`

Practical consequence:

- The base selector is no longer only “conceptually graph-friendly”; its feature set is now individually graphized and validated.
- The next builder can target the actual base-selector decision itself, instead of more isolated feasibility slices.
