# Hodel ARC-DSL Port Plan for NeuroGolf 2026

## 1. Hodel Architecture Summary

**Source**: https://github.com/michaelhodel/arc-dsl (Michael Hodel, 2024)

**Scale**:
- ~160 primitives across 7 categories
- ~400 hand-written solver programs (one per ARC training task)
- Average solver: ~10 primitive composition, ~12 lines of Python

**Architecture**:
```
Grid (tuple of tuples)  →  Objects (frozenset of (color, (row, col))  →  Grid
            ^                                                        |
            └─────────────── compose/chain/fork/rbind ───────────────┘
```

**Key design decisions**:
- Primitives are pure Python functions, composable via higher-order combinators (`compose`, `chain`, `fork`, `rbind`, `lbind`, `power`, `mapply`, `sfilter`, etc.)
- Objects are frozensets of `(color, (i, j))` tuples — intermediate representations between grids
- Solvers are flat sequential Python functions calling primitive sequences
- No recursion (iteration is unrolled), no unbounded loops, no external dependencies

**Primitive categories** (approximate counts):

| Category | Count | Examples |
|---|---|---|
| Arithmetic/Numerical | ~20 | add, subtract, multiply, divide, invert, sign, increment, decrement, even, double, halve |
| Logic/Control | ~15 | identity, flip, equality, contained, both, either, branch, compose, chain, fork, rbind, lbind, power, matcher |
| Set/Container | ~15 | combine, intersection, difference, merge, dedupe, order, repeat, insert, remove, other, sfilter, mfilter, extract |
| Grid ops | ~30 | fill, paint, cover, underfill, underpaint, crop, subgrid, hconcat, vconcat, hsplit, vsplit, cellwise, replace, switch, canvas, trim, compress, frontiers |
| Object/Color | ~25 | objects, partition, fgpartition, recolor, color, palette, numcolors, colorcount, colorfilter, mostcolor, leastcolor, mostcommon, leastcommon, sizefilter |
| Geometry | ~30 | height, width, shape, portrait, ulcorner, urcorner, llcorner, lrcorner, uppermost, lowermost, leftmost, rightmost, center, centerofmass, position, corners, box, inbox, outbox, backdrop, delta, connect, shoot, dneighbors, ineighbors, neighbors |
| Transform | ~15 | rot90, rot180, rot270, hmirror, vmirror, dmirror, cmirror, normalize, shift, hupscale, vupscale, upscale, downscale, hfrontier, vfrontier |
| Higher-order | ~10 | apply, rapply, mapply, papply, mpapply, prapply, fork, compose, chain, power |

**Solver pattern** (canonical):
```python
def solve_XXXX(I):
    objs = objects(grid=I, univalued=T, diagonal=F, without_bg=T)
    filtered = colorfilter(objs=objs, value=X)
    transformed = mapply(some_function, container=filtered)
    O = paint(grid=I, obj=transformed)
    return O
```

Every solver follows `extract → filter → transform → paint` pipeline.

---

## 2. Primitive Comparison Table

### Our DSL (v1, 18 primitives) vs Hodel (~160)

| Category | Hodel primitives | Our coverage | Gap |
|---|---|---|---|
| **Grid identity/transpose** | identity, rot90/180/270, hmirror, vmirror, dmirror, cmirror | `identity`, `transpose`, `flip_h`, `flip_v`, `rotate180`, `rotate90`, `rotate180_static`, `rotate90_static` | Minor gap: no diagonal/counter-diagonal mirror, no rot270 (rot90+flip suffices) |
| **Color ops** | replace, switch, recolor, color, palette, numcolors, colorcount, colorfilter, mostcolor, leastcolor | `replace_color`, `swap_colors`, `fill_bg`, `dominant_color`, `count_color` | Hodel has palette analysis, per-object recolor, most/least-color; we have per-grid only |
| **Object detection** | objects, partition, fgpartition, sizefilter, colorfilter | `largest_blob` (limited — finds 1 blob of 1 color) | **Major gap**: no general connected-components, no multi-object extraction, no univalued/diagonal flags |
| **Grid shape ops** | crop, subgrid, hconcat, vconcat, hsplit, vsplit, canvas, trim, compress, frontiers | `bbox_crop` (crops 1 color's bbox) | **Major gap**: no concat, split, canvas, trim, compress |
| **Tiling/Scaling** | hupscale, vupscale, upscale, downscale, hconcat, vconcat | `tile_h`, `tile_v` | Minor gap: no upscale/downscale |
| **Geometry (bbox, corners, box)** | height, width, shape, ulcorner, urcorner, llcorner, lrcorner, uppermost, lowermost, leftmost, rightmost, center, centerofmass, corners, box, inbox, outbox, backdrop, delta, connect, shoot | None | **Major gap**: no bounding-box primitives, no corner/line/shape analysis |
| **Set ops** | combine, intersection, difference, merge, insert, remove, other, sfilter, mfilter | None except implicit in compiler | **Major gap**: no set operations on objects |
| **Higher-order combinators** | compose, chain, fork, rbind, lbind, power, apply, mapply, papply, rapply, sfilter | None | **Major gap**: no function composition, no mapping over objects |
| **Arithmetic** | add, sub, mul, div, invert, sign, increment, decrement, even, double, halve, greater, maximum, minimum, valmax, valmin | None | We don't need standalone arithmetic — ONNX handles it. Gap is in DSL expressiveness for parameters. |
| **Logic/Control** | identity, flip, equality, contained, both, either, branch | None (compiler handles linear chain) | Hodel's `branch` is critical for conditional tasks; our compose is purely sequential |

### Summary

| Metric | Count |
|---|---|
| Hodel total primitives | ~160 |
| Our primitives (overlapping functionality) | ~18 |
| Full one-to-one matches | ~8 (identity, transpose, flip_h, flip_v, rotate180/90, replace_color, swap_colors) |
| Partial matches (we have weaker version) | ~5 (largest_blob ≈ weak objects(), bbox_crop ≈ weak crop, tile_h/v ≈ weak upscale, count_color, fill_bg) |
| No coverage at all | ~130+ |

---

## 3. Top-10 Hodel Primitives for Our Cost Monsters

Based on solver_ledger.md and known cost monsters (task255, task233, task366, task191, task018):

### Task 255 (1.59M cost, flood-fill/separator bands)
**What it needs**: Connected-component labeling, color-specific mask operations, flood-fill propagation
**Hodel primitives that help**:
1. `objects(grid, univalued, diagonal, without_bg)` — extracts connected components (core primitive)
2. `colorfilter(objs, value)` — filter objects by color
3. `mfilter(container, function)` — merge+filter objects
4. `fill(grid, value, patch)` — recolor cells

### Task 233 (1.46M cost, crop largest color-2 + hole patching)
**What it needs**: Largest-component extraction, hole detection, patch reinsertion
**Hodel primitives that help**:
1. `objects()` + `argmax(objs, size)` — largest component
2. `delta(patch)` — cells inside bbox but not in patch (= holes)
3. `fill(grid, value, patch)` — fill holes
4. `subgrid(patch, grid)` — crop to component

### Task 366 (0.83M cost, crop/translation/background separation)
**What it needs**: Object extraction by color, bounding-box crop, translation
**Hodel primitives that help**:
1. `objects()` + `colorfilter()` — object extraction
2. `ulcorner/ulcorner(patch)` — bbox origin
3. `shift(patch, direction)` — translation
4. `subgrid(patch, grid)` — crop output

### Task 191 (cost 85K, symmetry/flip at baseline)
**What it needs**: Various flips and rotation detection
**Hodel primitives that help**: Already covered by our flip/rotate primitives. Cost is low enough.

### Task 018 (hidden-sparse cost monster)
**What it needs**: Possibly counting, scalar output, or conditional logic
**Hodel primitives that help**:
1. `size(container)` — cardinality
2. `colorcount(element, value)` — count color occurrences
3. `numcolors(element)` — palette size
4. `branch(condition, a, b)` — conditional output

### Top-5 Hodel Primitives with Highest ROI

| Rank | Primitive | What it does | Tasks it unlocks | ONNX difficulty |
|---|---|---|---|---|
| **1** | `objects()` | Connected-component extraction (4- or 8-neighbor) | 255, 233, 366, +many ARC patterns | **Hard** — needs iterative flood-fill or Conv-based label propagation in ONNX; 8+ iterations, dynamic component count |
| **2** | `delta(patch)` | Hole detection (bbox minus patch) | 233, 255 (hole-fill tasks), enclosures | **Easy** — single ReduceSum+Sub on masks |
| **3** | `subgrid(patch, grid)` | Crop grid to component's bbox | 233, 366, 319, component-focused tasks | **Medium** — bbox computation via ArgMax+ReduceSum (already partially built in bbox_crop) |
| **4** | `colorfilter(objs, value)` | Filter objects by color | 255, 233, 366, color-selection tasks | **Medium** — requires objects() first, but filter itself is simple |
| **5** | `size(container)` | Count components/cells | Counting tasks, scalar output tasks | **Easy** — single ReduceSum |

---

## 4. ONNX Compilation Feasibility per Primitive

| Hodel primitive | ONNX difficulty | Strategy | Envelope cost (est.) | Notes |
|---|---|---|---|---|
| **identity** | ✅ Easy | Identity op | 0 | Done |
| **add/sub/mul/div** | ✅ Easy | Elementwise ops | ~10 | Trivial in ONNX |
| **invert/sign** | ✅ Easy | Neg/Where | ~10 | |
| **both/either/flip** | ✅ Easy | And/Or/Not | ~10 | |
| **equality** | ✅ Easy | Equal | ~10 | |
| **branch** | ✅ Easy | Where op | ~10 | Critical for conditional tasks |
| **size** | ✅ Easy | ReduceSum | ~10 | |
| **colorcount** | ✅ Easy | ReduceSum+Gather | ~50 | |
| **mostcolor/leastcolor** | ✅ Easy | ReduceSum+ArgMax | ~100 | dominant_color already built (cost 8236 but can optimize) |
| **fill** | ✅ Easy | Where+channel scatter | ~100 | |
| **paint** | ✅ Easy | Add+Clip | ~100 | |
| **cover** | ✅ Easy | Where (replace with bg) | ~100 | |
| **replace** | ✅ Easy | 1x1 Conv channel mix | ~100 | Done as replace_color (cost 100) |
| **switch** | ✅ Easy | Two replace_color | ~200 | Can be done via swap_colors or Gather |
| **hmirror/vmirror** | ✅ Easy | Gather with flip index | ~1000 | Done as flip_h/flip_v |
| **rot90/180/270** | ✅ Easy | Transpose+Gather | ~1000 | Done (cost 1100 for 3x3) |
| **subgrid** | 🟡 Medium | Slice or bbox+Gather | ~1000-5000 | bbox_crop exists (cost 82K needs BOOL optimization) |
| **crop** | 🟡 Medium | Slice with static dims | ~500 | Simpler than subgrid (no bbox search) |
| **hconcat/vconcat** | 🟡 Medium | Concatenate on axis + Pad | ~5000 | Two grid sources complicate single-input flow |
| **canvas** | 🟡 Medium | Tile constant + Pad | ~1000 | |
| **shift** | 🟡 Medium | Gather with offset indices | ~1000 | Requires offset computation |
| **normalize** | 🟡 Medium | Shift to origin | ~1000 | Requires ulcorner+shift |
| **upscale/downscale** | 🟡 Medium | Repeat interleave + Gather | ~5000 | |
| **backdrop** | 🟡 Medium | Bbox fill | ~1000 | |
| **delta** | ✅ Easy | backdrop - patch | ~500 | High ROI, easy |
| **box/inbox/outbox** | 🟡 Medium | Perimeter computation via boundary conditions | ~2000 | |
| **connect** | 🟡 Medium | Line between two points | ~1000 | Diagonal lines need careful handling |
| **shoot** | 🟡 Medium | Line from point+direction | ~1000 | Special case of connect |
| **colorfilter** | 🟡 Medium | Requires objects() first | ~100 | Filter itself is easy; dependency on objects() makes it hard |
| **sfilter/mfilter** | 🟡 Medium | Requires container iteration | ~500 | |
| **objects()** | 🔴 Hard | Connected components via iterative Conv dilation | 50000+ | Requires ~10-30 Conv iterations, dynamic component count unrepresentable |
| **partition** | 🔴 Hard | Per-color grouping | ~10000 | |
| **merge** | 🟡 Medium | Frozenset union | ~500 | |
| **compose/chain/fork** | ✅ Compiler-level | No ONNX needed — compiler wires sequentially | 0 | Already done in our compiler |
| **rbind/lbind/power** | ✅ Compiler-level | No ONNX needed — compiler specializes params | 0 | Parameter binding already done via build_* args |
| **apply/mapply** | 🟡 Medium | Map function over container | ~5000 | Requires unrolling loop over max component count |
| **occurrences** | 🔴 Hard | Subgrid matching via sliding window | ~50000 | Conv-based template matching possible but expensive |
| **frontiers/compress** | 🔴 Hard | Uniform row/col detection | ~10000 | Requires row-wise unique checking |

### Difficulty distribution

| Difficulty | Count (from Top-30 Hodel primitives) | Examples |
|---|---|---|
| ✅ Easy (single op, <200 cost) | ~12 | identity, add, both, size, colorcount, delta, fill, paint, replace, branch |
| 🟡 Medium (multi-op, 500-5000) | ~14 | crop, subgrid, shift, concat, upscale, backdrop, box, connect, shoot, canvas, normalize |
| 🔴 Hard (iterative/dynamic, >5000) | ~4 | objects(), partition, occurrences, frontiers/compress |

---

## 5. Estimated Effort

| Phase | Primitives | Estimated effort | Tasks unlocked (cumulative) |
|---|---|---|---|
| **Phase 1 — Easy wins** (can do in parallel) | fill, paint, cover, replace, switch, branch, size, colorcount, delta, backdrop, box, connect, shoot, crop, canvas, shift, normalize, upscale, downscale, hconcat/vconcat | **2-3 days** | ~60-80 (all single-op tasks + simple compositions) |
| **Phase 2 — Object layer** | objects() via Conv dilation (8-iter), colorfilter, sfilter, mfilter, merge, apply, mapply, subgrid | **2-3 days** | ~150-200 (most Hodel solvers use objects() as entry point) |
| **Phase 3 — Advanced** | occurrences (template matching), partition, frontiers/compress, partition+fork recolor patterns | **1-2 days** | ~50-80 more (harder pattern tasks) |
| **Phase 4 — LLM accelerator** | Triage remaining tasks, generate programs via LLM, manually QA | **ongoing** | ~200-300 (all pass-visible tasks) |

**Total estimated effort**: **5-8 days** for full primitive coverage, **2-3 days** for a high-ROI subset (phases 1+2).

---

## 6. Recommended Port Order

### Immediate (next worker, highest ROI per day)

1. **`delta(patch)`** — hole detection. Easy (single ReduceSum+Sub). Unlocks task233 hole-fill pattern, task255 enclosure detection. **~2 hours.**

2. **`fill(grid, value, patch)` / `paint(grid, obj)`** — already conceptually doable via Where+channel scatter. **~2 hours each.**

3. **`cover(grid, patch)`** — remove object from grid (fill with background). **~1 hour.**

4. **`branch(condition, a, b)`** — conditional grid output. ONNX `Where` op. Unlocks conditional/multi-rule tasks. **~1 hour.**

5. **`size(container)`** — cardinality. Single ReduceSum. Unlocks counting tasks. **~1 hour.**

6. **`subgrid(patch, grid)`** — optimized bbox-crop (reuse existing bbox_crop code but make it cheaper via BOOL pipeline). **~4 hours.**

### Short-term (next 2 workers)

7. **`shift(patch, direction)`** — object translation. **~2 hours.**

8. **`normalize(patch)`** — shift to origin. **~1 hour.**

9. **`crop(grid, start, dims)`** — static-slice crop. **~2 hours.**

10. **`hconcat/vconcat(grid, grid)`** — concat two grids. Requires dual-input handling in our single-input compiler. **~4 hours.**

### Medium-term (after phases 1+2 above)

11. **`objects()`** — connected components. Use the Conv-based iterative dilation pattern from our existing `largest_blob` but generalize to multi-blob. The challenge is representing variable-number-of-components in ONNX. One approach: emit up to K component masks (where K is max expected, e.g., 30). **~2 days.**

12. **`colorfilter(objs, value)`** — filter objects by color. Requires objects() first. **~4 hours (after objects).**

13. **`apply/mapply(function, container)`** — map over components. Requires unrolling. **~4 hours.**

---

## 7. Key Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `objects()` (connected components) too expensive in ONNX | Medium | High — cuts off ~50% of Hodel solvers | Use fixed 8 iterations (sufficient for <=30x30 grid); cap component count at K=30; fall back to per-color partition if iterative conv too costly |
| Dynamic object count unrepresentable in ONNX | High | Medium — limits mapply/apply | Unroll to fixed max (e.g., 30 objects); mask invalid slots |
| BOOL pipeline optimization insufficient for bbox_crop/subgrid | Low | Medium — bbox_crop stays at 82K | Split into bbox_locate (cheap) + separate gather; only pay for gather when needed |
| Hodel solvers rely on Python set operations (frozenset) | High | Low — sets are intermediate representation | Our ONNX uses [1,10,30,30] channel tensor as the universal representation; convert set ops to channel-level tensor ops |
| LLM-generated programs produce wrong results on hidden test | Medium | Medium — silent correctness regression | Always run hidden-safety audit per the solver_ledger.md gate criteria |

---

## 8. Our Coverage vs Hodel

| Metric | Value |
|---|---|
| Hodel total primitives | ~160 |
| Our primitives with direct one-to-one Hodel equivalent | 8 (identity, transpose, flip_h, flip_v, rotate180, rotate90, replace_color, swap_colors) |
| Our primitives with partial/Hodel-inspired coverage | 10 (fill_bg, count_color, dominant_color, bbox_crop, tile_h, tile_v, largest_blob, rotate180_static, rotate90_static, fill_bg) |
| **Total functional overlap** | **~11%** of Hodel's primitive surface area |
| Unlocked tasks at current DSL v1 | 11/400 (2.8%) |
| Estimated tasks unlocked with Top-5 Hodel primitives | ~60-80 (15-20%) |
| Estimated tasks unlocked with full port | ~200-300 (50-75%) |

---

## 9. Recommendation

**Start porting now — do NOT wait for more LLM results.**

Rationale:
1. The Top-5 Hodel primitives (`delta`, `fill`, `cover`, `subgrid`, `branch`) are **easy or medium** ONNX difficulty and unlock ~15-20% of uncovered tasks.
2. Our current 2.8% coverage (11/400 tasks) is too low for any meaningful LB impact. Even doubling to ~6% via a 2-day port sprint doubles the candidate swap pipeline.
3. LLM-assisted task triage (Stage 4) benefit is proportional to primitive coverage — LLM can't generate programs for primitives that don't exist.
4. The cost-monster tasks (255, 233, 366) specifically need `objects()` + `delta()` + `subgrid()` — three of the Top-5.

**Immediate action**: Port `delta`, `fill`, `paint`, `cover`, `branch`, `size`, `colorcount` (Phase 1 — 1 day) then `subgrid`, `shift`, `crop` (Phase 2 — 1 day). After that, tackle `objects()` (Phase 3 — 2 days). This gives a viable path from 2.8% → ~20% coverage in ~4 days of work.

**LLM role**: Run LLM triage in parallel with Phase 1-2 engineering, so that by the time the primitives are ready, we have a prioritized task list.
