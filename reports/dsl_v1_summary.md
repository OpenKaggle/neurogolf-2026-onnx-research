# DSL v1 Summary (Stage 2 Worker, 2026-06-03)

Built atop the v0 primitive library (8 primitives, 13/13 unit tests). Added
Tier-1 primitives plus a depth-≤2 brute-force search harness, and re-scored
the whole 400-task corpus against the v4_plus4 anchor.

## 1. Line A — new primitives

6 new primitives added to `tools/dsl/primitives.py`:

| name              | params         | shape behavior         | cost on a 5×5 input          |
|-------------------|----------------|------------------------|------------------------------:|
| `rotate180_static`| `h, w`         | static-shape Slice+Pad | **1110 (3×3) … 9762 (9×9)**  |
| `rotate90_static` | `h, w`         | static-shape, w/transpose | **1107 (3×3) … 9753 (9×9)** |
| `count_color`     | `c`            | grid → (1, 1) cell     | 170                          |
| `dominant_color`  | -              | fill h×w bbox with mode | 8236                         |
| `fill_bg`         | `bg_color`     | alias replace_color(0,_) | 100 (same as replace_color)  |
| `bbox_crop`       | `target_color` | crop bbox to top-left  | 82296                        |

Every new primitive ships with:
- a `python_ref` implementation in `primitives.py`,
- an ONNX subgraph builder in `primitives.py`,
- a passing equivalence test in `tools/dsl/test_primitives.py`
  (**30/30 v0+v1 unit tests pass**),
- a cost-measurability check in `tools/dsl/check_primitive_cost.py`
  (**18/18 measurable**, no unmeasurable failures).

### 1a. Rotate gap closed — but not eliminated

The v0 `rotate180`/`rotate90` cost 37664 / 36832 (50–100× worse than the
v4_plus3 baseline). The new `rotate*_static(h, w)` variants use a Slice+
Gather+Gather+Pad pipeline whose intermediates are `[1, 10, h, w]` FLOAT
(not `[1, 10, 30, 30]`). Concrete cost drops for the rotate test set:

| task | program                       | v0 cost | v1 static cost | v4_plus4 baseline |
|------|-------------------------------|--------:|---------------:|------------------:|
| 087  | rotate180_static(3,3)          |  37664  | **1110**       |    368            |
| 140  | rotate180_static(3,3)          |  37664  | **1110**       |    368            |
| 380  | rotate90_static(3,3)           |  36832  | **1107**       |    728            |

Static rotates are **34× cheaper than v0** but still **3× more expensive
than the baseline** for 3×3 cases. Reason: even with the smallest
intermediates, we carry three `[1, 10, h, w]` FLOAT tensors (~360 bytes
each at 3×3) + initializers. The baseline manages with a single
small-Slice that ORT happens to fuse — we cannot do that without
producing dynamic-shape intermediates that the scorer rejects.

**Net**: rotate_static **does not** displace baseline rotate ONNXs yet,
but it is now a **viable inner op** for depth-2 programs (it costs ~1k
in a composition where the v0 rotate alone would cost ~37k).

### 1b. Dynamic-shape risk: bbox_crop

`bbox_crop` was the highest-risk primitive (per task brief). It survives
the `calculate_memory` scoring (cost = 82296, *measurable*). All
intermediate dim sizes are statically inferable; the runtime-dependent
`first_h`/`first_w`/`height`/`width` are INT64 scalars or 30-length
vectors, not tensor dims. Two `[1, 10, 30, 30]` FLOAT intermediates
account for the ≈72 KB cost. Usable as a last-resort primitive but not
cheap.

### 1c. count_color & dominant_color: shape-changing primitives

`count_color(c)` produces a `(1, 1)` output grid. The ONNX builder pads
the 1×10×1×1 one-hot to `[1, 10, 30, 30]`; the decoder reads only the
top-left cell, so the result decodes to `[[count]]`. Cost 170 bytes.

`dominant_color()` projects a per-cell mask onto the dominant-color
channel. Cost 8236 bytes (dominated by the `[1, 1, 30, 30]` cell-mask
intermediate).

## 2. Line B — depth-≤2 search harness

`tools/dsl/search.py` enumerates DslPrograms of length 1..max_depth,
filters by the visible-train pass-rate via Python refs (no ONNX compile
per candidate), and returns the cheapest matching program.

Search policies:
- Parameters drawn from the union of input + output colors / shapes.
- `lenient` mode for op2 in depth-2 (intermediate may carry colors not
  in the original input).
- `rotate*_static` preferred over `rotate*` when the task has a single
  input shape.
- Cost-aware tiebreak: among programs that visible-pass, return the one
  with cheapest static `program_estimated_cost` (so the search prefers
  `swap_colors(a,b)` over `replace_color(a,b)` whenever both fit —
  cost 10 vs 100). This is what closed the task276/309 cost gap.

### Search results — random 100 sample (seed 42)

```
scanned 100 tasks
  2 hits (2.0%)  -> task309 swap_colors(5,7), task380 rotate90_static(3,3)
  1 swap candidate (task309 ties baseline 10 = 10)
  total search time 2.0s
```

### Search results — full 400-task sweep (definitive)

```
scanned 400 tasks
  11 hits (2.8%)
  7 swap candidates (DSL cost <= baseline, full pass-rate)
  total search time 8.2s
```

All 11 hits (CSV at `reports/dsl_v1_search_results_full400.csv`):

| task | depth | program                          | DSL cost | baseline | cost_ratio | in_phase4 |
|------|------:|----------------------------------|---------:|---------:|-----------:|----------|
| 087  | 1     | rotate180_static(h=3,w=3)         |    1110 |     368  | 0.332      | yes      |
| **129** | **1** | **dominant_color**                | **8236**|  **818** | **0.099**  | **NO**   |
| 140  | 1     | rotate180_static(h=3,w=3)         |    1110 |     368  | 0.332      | yes      |
| 150  | 1     | flip_h                            |     832 |    1014  | **1.219**  | yes      |
| 155  | 1     | flip_v                            |     832 |    1014  | **1.219**  | yes      |
| 179  | 1     | transpose                         |       0 |       0  | 1.000      | yes      |
| 241  | 1     | transpose                         |       0 |       0  | 1.000      | yes      |
| 276  | 1     | swap_colors(c1=2,c2=6)            |      10 |      10  | 1.000      | yes      |
| 309  | 1     | swap_colors(c1=5,c2=7)            |      10 |      10  | 1.000      | yes      |
| 337  | 1     | swap_colors(c1=5,c2=8)            |      10 |      10  | 1.000      | yes      |
| 380  | 1     | rotate90_static(h=3,w=3)          |    1107 |     728  | 0.658      | yes      |

**One new task discovered** (not in phase4): **task129** — pure
`dominant_color` (fill grid with mode color). All visible+arc-gen
examples pass, but our cost (8236) is 10× worse than baseline (818).
Not a swap candidate. Documents that the corpus does contain
`dominant_color`-shaped tasks; the next iteration should focus on
making this primitive cheaper.

**Zero depth-2 hits** across the full 400-task sweep. Depth-2 over the
current primitive set doesn't unlock any new tasks the corpus actually
contains. Practical conclusion: depth-2 search has marginal value
without richer building blocks (object/mask/conditional primitives).

## 3. Line C — accepted swap candidates

`reports/dsl_v1_swap_candidates_full400.csv`. **7 candidates** total
(2 actual wins + 5 ties; DSL cost ≤ baseline AND all examples pass):

| task | program                | Δcost (bytes saved) | score gain est. |
|------|-----------------------|--------------------:|----------------:|
| 150  | flip_h                |              **182** |  ≈ +0.198 LB    |
| 155  | flip_v                |              **182** |  ≈ +0.198 LB    |
| 179  | transpose             |                   0  |       0         |
| 241  | transpose             |                   0  |       0         |
| 276  | swap_colors(2,6)      |                   0  |       0         |
| 309  | swap_colors(5,7)      |                   0  |       0         |
| 337  | swap_colors(5,8)      |                   0  |       0         |

**Cumulative local gain estimate: ≈ +0.4 LB points.** Five ties carry
no LB value (same cost) — they only matter for code dedup / engineering
hygiene. The two real wins are task150 + task155.

DSL ONNXs at `submissions/handbuilds/dsl_v1/dsl_task*.onnx`. **No
`submissions/` files modified.** No anchor build, no zip, no submit.

## 4. Are we at the 30+ swap threshold?

**No, far from it.** 7 candidates vs the 30+ target. **Do NOT do an
anchor refresh on the basis of this round.** The two byte-saving swaps
total ≈ 0.4 LB points — well within submission noise.

## 5. Recommendation for the next worker

The bottleneck is **primitive coverage**, not search depth. Three lines
of attack ordered by ROI:

1. **Tier-2 primitives (object-aware)**:
   - `connected_components(connectivity)` — 4- or 8-neighborhood. Heavy
     in nodes (probably 30+ ops for label propagation via Conv) but
     unlocks the task255 / task233 / task366 family of monsters.
   - `tile_h(k)`, `tile_v(k)` — known to unlock task001 (3×3 → 9×9 tile).
   - `apply_mask(predicate, subprogram)` — branch-like, unlocks
     conditional-rule tasks.

2. **LLM-assisted task triage** (in parallel with #1):
   Feed task examples to an LLM and ask "what primitive sequence would
   solve this?". Triage the 393 currently-uncovered tasks into buckets:
   "needs X new primitive" / "needs object detection" / "needs
   counting" / "looks like CA". Use the bucket sizes to prioritize
   which Tier-2 primitive to build first.

3. **Optimize existing primitives**:
   - `replace_color`: cost 100. A Gather-based variant (when dst ∉
     input colors) costs 10. Add a search-time fallback that picks the
     cheap variant when the precondition holds.
   - `dominant_color`: cost 8236. The `[1, 1, 30, 30]` FLOAT mask is
     the bulk; can we use BOOL? Would shrink to ~2K, making task129
     a real swap candidate.
   - `bbox_crop`: cost 82296. Two `[1, 10, 30, 30]` FLOAT intermediates
     dominate. A BOOL pipeline would cut to ~20K; still expensive but
     potentially competitive on a 100K+ baseline.

## 6. Blockers / user decisions

- **Static rotate cost can't beat baseline** with the current op set.
  Open question: do we accept this (use static rotate for compositions,
  keep baseline rotate for standalone) or invest in a smarter
  variant (e.g., a single GatherND with a static `[h*w, 2]` index)?
- **Anchor refresh threshold**: current 7 swap candidates are ≪ 30.
  Suggest holding the anchor at v4_plus4 and adding the 0.4-LB-point
  swaps only opportunistically (e.g., bundled with Tier-2 primitive
  wins in the next round). No submission this round.

## 7. Hard rules compliance

- ✅ Did not modify any existing `submissions/` ONNX. New artifacts in
  `submissions/handbuilds/dsl_v1/`.
- ✅ Did not touch `task319`.
- ✅ No remote git, no Kaggle submission.
- ✅ No fp16 / no large lookup tables (largest init: 10-element
  arange or 10×10 Conv weight = 100 elements = 400 bytes).
- ✅ No dynamic-shape intermediate tensors. All 18 primitives
  measurable (`tools/dsl/check_primitive_cost.py`).
- ✅ Every new primitive has a unit test (30/30 pass).
- ✅ Every swap candidate is full-pass under isolated audit.
- ✅ Paths match brief: `tools/dsl/{primitives, compiler, search,
  test_primitives, check_primitive_cost, run_phase5_search}.py`,
  `reports/dsl_v1_*`, `submissions/handbuilds/dsl_v1/`.
- ✅ `reports/road_to_top10.md` §H appended with this round's notes.

## 8. Artifacts written

- `tools/dsl/primitives.py` — 6 new primitives + ref + builder pairs
- `tools/dsl/test_primitives.py` — 30/30 unit tests
- `tools/dsl/check_primitive_cost.py` — 18/18 cost-measurability tests
- `tools/dsl/search.py` — depth-≤2 search harness, cost-aware ranking
- `tools/dsl/run_phase5_search.py` — full sweep driver
- `reports/dsl_v1_search_results.csv` — 100-random-sample run
- `reports/dsl_v1_search_results_full400.csv` — full 400-task sweep
- `reports/dsl_v1_swap_candidates.csv` — 100-sample candidates
- `reports/dsl_v1_swap_candidates_full400.csv` — full-sweep candidates
- `submissions/handbuilds/dsl_v1/dsl_task{087,129,140,150,155,179,
   241,276,309,337,380}.onnx` — search-built ONNXs
