# Road to Top-10 (NeuroGolf 2026)

Strategy synthesis from external recon (LB, public kernels, ARC literature) and
internal cost-monster analysis. Started 2026-06-03.

## A. What top-10 is doing (evidence-based)

### LB landscape (2026-06-06 refresh)

- **CroDoc 7695.78** (rank 1)
- Top 3 cluster around **7621.81–7695.78**
- Top 10 cutoff is currently **7339.93**
- Highest publicly-shared kernel: `octaviograu` 6154.71 (= our `octv_6154` source)
- `nadeem_6252` (our anchor base, 6252) is a **rebundle** of public sources, not
  an original solver
- Our current LB: **6260.16** (`v4_plus10_compiler2`)
- **Gap to top-10: ≈ 1080 points**, i.e. ≈ +2.7 score per task average
- **Gap to top-3: ≈ 1362 points**, i.e. ≈ +3.4 score per task average

The refreshed leaderboard reinforces the original diagnosis: incremental
public cherry-picks and exact-equivalence compiler golf can move us a few
points, but top-10/top-3 requires a different production function — many more
per-task programs, much smaller graphs, or a still-valid scorer-boundary
insight.

The task149 BOOL rewrite adds a compiler-level pillar to that production
function. Extending BOOL mask lifetimes reduced an already-tight,
LB-verified task from cost 509 to 172 and produced `+1.08495` LB exactly.
`v4_plus9_compiler` then confirmed the general direction on hidden LB:
32 uniform-scalar rewrites, 6 BOOL-memory rewrites, and task149 v2 scored
`6258.14` on Kaggle, exactly matching local `6258.1442`. The future DSL must
therefore be typed: masks stay BOOL, broadcast-safe uniform tensors become
scalars, counts and coordinates use the narrowest legal numeric type, and
FLOAT is reserved for operators that require it.

`v4_plus10_compiler2` added a third hidden-verified backend rule: if a terminal
`Cast -> output` only widens a producer that decodes identically, expose the
BOOL/FLOAT16 producer directly as the graph output. This added `+2.0184` local
and LB on 10 tasks, reaching **6260.16**.

### 2026-06-06 public discussion calibration

Recent Kaggle Discussion posts provide a useful external calibration:

- Rank-4 Tony Li publicly listed 15 bottom tasks with scores summing **227.716**.
  On those same tasks our current anchor sums about **189.6**, a **38-point**
  gap. Another rank-19 team reported **232.16** on the same 15 tasks, a
  **42.582-point** gap over us.
- The biggest observed gaps on that public tail set are `task255 +4.57`,
  `task233 +3.98`, `task366 +3.77`, `task018 +3.52`, and `task285 +3.02`.
- Top competitors explicitly describe token/overfit/timeout management,
  LLM-assisted manual golfing, and synthetic hidden datasets. This strongly
  supports a broad semantic-production workflow rather than a single global
  optimizer.

Quantitatively, rebuilding only the top-20 cost monsters near cost 100 is
worth about `+159`; the top-50 are about `+368`. To approach top-3, the project
needs roughly 200+ tasks in the low-cost semantic-program regime, not just
hero rewrites of a few monsters.

### Cost monster reverse-engineering (top-3 in v4_plus4)

| task | cost | nodes | initializer pattern | likely algorithm |
|---|---:|---:|---|---|
| task255 | 1.59M | (matmul-heavy) | 211KB lookup table | brute-force matrix scoring |
| task233 | 1.46M | 2000+ | AND/Equal heavy | hand-written rule engine, no DSL reuse |
| task366 | 0.83M | 713 | small | label propagation via MaxPool |

Common pattern: **monolithic per-task graphs with no shared primitives**. Each
solver is essentially a bespoke compiler output. These are exactly the kind of
graphs a DSL + primitive library would shrink by 5–50×.

### ARC-AGI literature parallels

The two dominant prior-art systems for ARC-shaped problems:

1. **Hodel ARC-DSL** (Michael Hodel, 2024):
   - ~160 hand-curated primitives (geometry, color, object, grid, recursion)
   - ~400 hand-written solver programs in Python, avg ~10 primitives per solver
   - Each solver is compilable into a small executable graph
   - Public on GitHub: this is the closest existing artifact to "winning DSL for ARC"

2. **icecuber 2020 ARC** (1st-place 2020 Kaggle ARC):
   - 142 unary functions in C++
   - Brute-force search at depth 4 over a DAG of intermediate grid states
   - Per-task program synthesis with cached subexpressions
   - Demonstrated that "small DSL + search" beats large neural models

### Critical adaption to NeuroGolf

NeuroGolf has a non-obvious twist that changes the strategic landscape: **task
examples are visible at submission time**. Each submission is one ONNX per
task — the model does NOT need to generalize across tasks at inference time.
This means:

- **Test-time training / few-shot inference is not the right frame**. You can
  pre-solve each task offline and bake the solution directly into ONNX.
- **The competition is per-task program synthesis with ONNX as the executable
  format**, NOT general ARC reasoning.
- **Cost penalty makes neural models structurally disadvantaged** —
  `score = pass_rate * max(1, 25 - ln(cost))`. A 10000-node ONNX scores ≤ 25 −
  ln(10000) ≈ 15.8 even on a trivial task; a 100-node ONNX scores up to ≈ 20.4.

Top-10 are almost certainly doing one (or both) of:

- **(i) LLM-assisted per-task hand-build**: feed task examples to LLM, derive
  the rule, generate an ONNX builder. Scales linearly with engineer-hours.
- **(ii) DSL + brute-force program search**: build a primitive library, search
  for shortest program matching all visible examples, compile to ONNX.

Approach (i) is more amenable to current LLM tooling; (ii) gives a steeper
long-term curve but needs heavier engineering.

## B. Why our current path is narrowing

| Symptom | Diagnosis |
|---|---|
| 3 consecutive cherry-pick rounds (v4_exp / v4_plus / v4_plus2) yielding 0–2 swaps | Public-source space saturated. v4_plus4 captures every Nadeem/biohack swap that's audit-clean. |
| 13 hand-built solvers across 400 tasks (`tools/build_task*.py`) | 3.25% per-task hand coverage. Top-10 likely 25–50%+. |
| Each hand-built solver written from scratch | Zero cross-task amortization. task319 / task233 / task204 builders share no primitives. |
| Cost monsters all monolithic (task255 1.59M, task233 1.46M) | No primitive subgraph extraction means we never get the "shrink to a few hundred ops" effect. |
| Lossless surgery saturated on v4_plus | Bytecode-level optimization has a hard ceiling once each task is already a tight algorithm. The next gains must come from **algorithm-level change**. |

In short: we're harvesting the last drops from a closed system. To 7000+ we
must **change the substrate**, not the surface.

## C. Wider-path candidates

| # | Path | Mechanism | First-step effort | Risk | Gate (when to know it works) |
|---|---|---|---:|---|---|
| 1 | **DSL primitive library + per-task hand-built programs** | 6–12 primitives compiled to ONNX subgraphs; each task = compose primitives via Python compiler | 1–2 days for v0 (already in progress, see DSL prototype subagent) | low — primitives are unit-tested, audit-clean by construction | After 5–10 tasks done, median cost ratio < 0.5 vs current baseline → green-light expansion |
| 2 | **Hodel DSL adaption** | Port Hodel's 160 primitives + relevant solvers; add ONNX backend | 3–5 days | medium — Hodel primitives are Python, ONNX compilation is non-trivial for some (Recursion, Apply, etc.) | After 30–50 tasks ported and audit-clean, integrated into v4_plus5+ |
| 3 | **icecuber-style program search on our DSL** | Brute-force compose primitive sequences up to depth 3–4; verify on visible examples; pick shortest | 2 weeks (after path #1 mature) | high — search space combinatorial; needs strong pruning | Coverage-per-day curve breaks above hand-build path |
| 4 | **LLM-assisted per-task builder generation** | For each unsolved task, feed examples to model; generate Python rule; manually QA + compile to DSL program | 2–3 days infra; ongoing per-task | medium — LLM mistakes are silent; needs strong audit gate | Pass rate × cost-quality on first 20 LLM-built tasks |
| 5 | **Shared ONNX subgraph extraction** | Static-analyze 400 ONNX files; find recurring subgraphs (same op pattern, possibly different consts); deduplicate via subgraph reference | 1 week | medium — ONNX doesn't natively support subgraph reuse across files; would only shrink within-task | Bytes saved per task ≥ 10% on top-30 cost monsters |

Recommended order: **#1 (in progress) → #2 (port subset) → #4 (LLM accelerator)**.
#3 is end-stage. #5 is parallel quick-win, lower priority.

## D. Roadmap (6253 → 7000+)

### Stage 1 (now, ~1–2 days): DSL prototype validation

- Goal: Prove DSL path produces ≥ 5× cost reduction on ≥ 5 monster tasks
- Deliverables: `tools/dsl/primitives.py`, `tools/dsl/compiler.py`, 5–10 task
  programs, audit-clean ONNX, cost-reduction table
- Gate: Median DSL ONNX cost ratio < 0.2 vs v4_plus4 baseline on the 5 sampled
  tasks
- Failure path: If cost ratio > 0.5, DSL not competitive against existing
  Nadeem-based solvers; pivot to LLM-assisted hand-builds (#4)
- **Currently delegated to background DSL subagent**

### Stage 2 (1 week after Stage 1 green): expand primitive library

- Add 6–12 more primitives based on Stage 1 failure analysis
- Target: 30–50 tasks rewritten via DSL, integrated into v4_plus5/v4_plus6
- Estimated LB lift: +15 to +50 (each replaced monster gives ~+1 to ~+4)
- Gate: 30+ DSL-rewritten tasks, all audit-clean, total local lift +15
- Submission cadence: 1 LB probe per 10 DSL-rewritten tasks

### Stage 3 (2–4 weeks): port relevant Hodel primitives + LLM accelerator

- Pull the Hodel-style 30–50 primitives that match our task patterns
- Build LLM-assisted task triage: for each task with no DSL solver yet, feed
  to LLM, get candidate program, manually QA, compile via DSL
- Target: 150–200 tasks DSL-built or LLM-built
- Estimated LB: 6800–7300

### Stage 4 (4+ weeks): icecuber-style program search

- Brute-force search over primitive compositions up to depth 3–4, pruned by
  visible-example fitting
- Target: 250–350 tasks DSL-built
- Estimated LB: 7300–7700

### Fall-back at every stage

If a stage fails its gate, the previous LB-verified anchor is preserved (we
never roll back on LB). The DSL line and the public-source-cherry-pick line are
**parallel**, not exclusive — we keep mining new public sources opportunistically
for as long as they appear.

## E. Immediate next worker task

**The DSL prototype subagent is running**. After it completes:

1. If Stage 1 gate passes (median cost ratio < 0.2 on 5 sampled tasks):
   - Open a new subagent to expand primitive library (Stage 2)
   - Concurrent: build a v4_plus5 candidate that swaps in DSL-rewritten tasks,
     run full audit, submit if local lift > 1.0
2. If Stage 1 gate fails (cost ratio > 0.5):
   - Open a Stage-2-bridging subagent: instead of expanding DSL, switch to
     LLM-assisted hand-build (#4 path) on top 5 cost monsters
   - Document the failure mode in this file so we don't re-attempt the same
     primitive design

Either way, the **next 2–3 background subagents should be DSL/program-synthesis
focused**, not public-source mining (which is saturated).

## F. Highest-uncertainty open questions

1. **Are top-10 actually using DSL or just LLM-per-task?**
   - The cost-monster shape (monolithic, no shared primitives in our v4_plus4)
     is consistent with public sources being LLM-per-task or pre-DSL.
   - But absence of primitives in OUR sources doesn't tell us what top-10 has.
   - We'd need to download a high-LB submission and inspect. None public above
     6155 → blind here.

2. **What does the hidden test distribution look like?**
   - H1 (task018 cost-only) suggests hidden test set is sometimes empty / very
     small. If this is widespread, "visible-pass + cost-min" is enough; rule
     correctness on hidden becomes secondary.
   - But we've only seen H1 confirmed on 1 task. We do NOT know if 50% / 5%
     of tasks have empty hidden sets.

3. **Engineering bandwidth**
   - 13 hand-builds in the project to date. Building 100+ DSL solvers is the
     real ask. LLM acceleration is essential, not optional.

## G. References (web recon, partial; complete list pending re-research)

- Michael Hodel, ARC-DSL, GitHub (~160 primitives, ~400 solvers)
- icecuber, "ARC 1st place solution writeup" Kaggle 2020
- ARC Prize 2024 winning approaches summary (test-time training, program search)
- Greenblatt program-synthesis prompting work on ARC

(Strategy subagent partial recon; full literature list to be resumed.)

---

## H. DSL primitive library v0 (internal worker, 2026-06-03)

Worked from internal data only (no external recon). Delivered:

- `tools/dsl/primitives.py` — 8 primitives (identity, transpose, swap_colors,
  replace_color, flip_h, flip_v, rotate180, rotate90) with Python reference
  + ONNX subgraph builder pairs.
- `tools/dsl/compiler.py` — `DslOp`, `DslProgram`, `compile_to_onnx` that
  wires primitives left-to-right and wraps with the competition's
  `[1,10,30,30]` FLOAT I/O envelope.
- `tools/dsl/test_primitives.py` — 13/13 equivalence tests pass.

### Phase 4 validation on 10 sampled tasks

100% pass on train + test + arc-gen for all 10 tasks. Cost vs v4_plus3:

- **2 wins**: task150 (flip_h) and task155 (flip_v) at **1.29x cheaper**
  (832 vs 1076).
- **5 ties**: task179, task241 (transpose, both cost 0); task276, task309,
  task337 (color remap/swap, all cost 10).
- **3 losses**: task087, task140, task380 (rotate primitives at 50-100x
  worse cost due to FLOAT [1,10,30,30] intermediate). Root cause and three
  candidate fixes documented in `reports/dsl_phase5_summary.md`.

### Coverage probe

Brute-force scan over `task001..task400`: **9 tasks** are pure 1-primitive
expressible by v0 (the 10 we used minus task276/309 which need the
"swap≡replace iff dst absent" assumption — strictly speaking 7 are pure
geometric/transpose + 2-3 color tasks where swap_colors works as replace).

Random-sample probe (50 tasks): **0/50** are 1-primitive expressible.

**Calibrated conclusion**: v0 DSL covers ~2% of the corpus with depth-1
programs. Real coverage requires depth-2+ composition or new primitive
classes (objects, masks, counting).

### Concrete next-worker mandate (DSL v1)

Tier 1 primitive additions (high-ROI, low-effort):

1. `rotate180_static(h, w)` / `rotate90_static(h, w)` — closes the 100x gap
   on the three rotate-loss tasks (task087, task140, task380).
2. `count_color(c)` — emits scalar count, unlocks "answer is the count" tasks.
3. `dominant_color(exclude_bg)` — building block for many ARC patterns.

Tier 1+ if engineering bandwidth allows:

4. A **depth-2 brute-force search harness** over the existing 8 primitives.
   Even with no new primitives, depth-2 search may unlock 20-40 tasks that
   no human bothered to spot.

### Anchor swap candidates (safe to integrate)

Only `task150` and `task155` DSL ONNXs beat the v4_plus3 baseline (combined
~488-byte cost savings, ~0.5 LB points). Not worth a submission slot on its
own; bundle into the next anchor refresh.

All other v0 DSL ONNXs are at parity (no value swapping) or worse (must NOT
swap) — wait for v1 primitives before any anchor swap.

### Open questions for v1 worker

- Should rotate primitives accept an optional `(h, w)` hint to fall through
  to the static Slice+Pad path? If yes, the DSL becomes "shape-aware";
  if no, the static variant becomes a separate primitive name.
- Should the compiler attempt to **fuse adjacent primitives** (e.g. two
  consecutive Gathers along the same axis)? Probably not v1, but flag for v2.

---

## I. DSL v1 round (internal worker, 2026-06-03)

Full report at `reports/dsl_v1_summary.md`. Headlines:

- **+6 primitives**: `rotate180_static(h,w)`, `rotate90_static(h,w)`,
  `count_color(c)`, `dominant_color()`, `fill_bg(bg_color)`,
  `bbox_crop(target_color)`. **30/30 unit tests pass**;
  **18/18 cost-measurable** (no dynamic-shape unmeasurable failures —
  the v0 Slice+Pad trap is sidestepped).
- **Search harness** at `tools/dsl/search.py` — depth-≤2 brute force,
  cost-aware ranking, lenient depth-2 parameter pool. Drove a full
  400-task sweep in 8.2 s.
- **Static rotate closes 34× of the v0 gap** but still loses 3× to
  baseline on 3×3 cases. Useful as a depth-2 building block; not a
  standalone swap.
- **1 new task discovered**: task129 = `dominant_color`, all examples
  pass, but DSL cost 8236 vs baseline 818 → not a swap. Cheaper
  `dominant_color` builder is the obvious follow-up.
- **0 depth-2 hits** across the 400-task sweep. Depth-2 over the
  current primitive set has no marginal value; primitive coverage,
  not search depth, is the bottleneck.
- **7 swap candidates** total (2 real wins + 5 ties):
  task150/155 (≈+0.4 LB combined), task179/241/276/309/337 (ties).
  **No anchor refresh this round** — far below the 30+ threshold the
  Stage-2 plan calls for. Hold v4_plus4 anchor.

### Concrete next-worker mandate (DSL v2)

The bottleneck is **primitive coverage**, not search depth. Pick one:

1. **Tier-2 primitive sprint** (highest ROI on current evidence):
   - `connected_components(connectivity)` (Conv-based label
     propagation; targets task255/233/366 monster family).
   - `tile_h(k)` / `tile_v(k)` (targets task001).
   - `apply_mask(predicate, subprogram)` (branch-like; targets
     conditional-rule tasks).

2. **LLM-assisted task triage**: feed the 389 currently-uncovered
   tasks to an LLM, ask "what primitive sequence would solve this?",
   bucket by required-primitive class. Use bucket sizes to
   prioritize the Tier-2 build order.

3. **Optimize existing primitives**:
   - `replace_color`: 100 → 10 via Gather variant (precondition:
     `dst ∉ input colors`). Already what swap_colors does; should
     either be the default or selected by search precondition check.
   - `dominant_color`: 8236 → ~2K via BOOL pipeline. Would make
     task129 a real swap candidate.
   - `bbox_crop`: 82296 → ~20K via BOOL pipeline. Even at ~20K it
     beats some 100K+ baselines.

### Open questions for v2 worker

- Is the corpus deeper in **object-counting** / **mask-conditional** /
  **CA-style** tasks, or in **shape-changing** (tile, scale) tasks?
  Hand-triage ~20 random uncovered tasks to find out before committing
  Tier-2 build order.
- Should `bbox_crop` be **split** into `bbox_locate` (find first_h/w
  and h/w) and a separate `apply_at_offset` op so other primitives can
  reuse the locator without paying for the crop? Probably yes, when we
  also add `bbox_paint` / `bbox_replicate`.
