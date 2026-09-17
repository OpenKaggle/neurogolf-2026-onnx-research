# External Research Synthesis for NeuroGolf 2026

Date: 2026-06-06  
Workspace: `$HOME/Desktop/kaggleonnx`  
Current anchor: `v4_plus10_compiler2`, public LB 6260.16  
Scope: external research synthesis only. No Kaggle submission. No anchor edits.

## Executive Read

The external evidence points to one main production function: many small, task-specific programs found by DSL search, LLM-assisted induction, or hand-guided object reasoning, then compiled into cost-disciplined ONNX. Compiler golf remains valuable, but it is a low-ceiling, high-safety sidecar. Public source cherry-picking is saturated and has already shown a hidden-safety cliff.

Fact from this repo: `v4_plus10_compiler2` reached 6260.16 by combining source swaps, DSL micro-wins, and exact compiler rewrites. Its top cost monsters still include `task255` cost 1.59M, `task233` 1.45M, `task366` 830K, `task191` 691K, `task018` 476K, and `task285` 395K in `reports/candidate_v4_plus10_compiler2_score_n3.csv`. These are not going to be closed by another terminal-cast pass.

Inference for this repo: the next 48 hours should not chase more public bundles. It should turn the existing DSL into a typed, object-centric synthesis pipeline with hard validation gates: Python reference -> visible train/test -> arc-gen -> pseudo-hidden metamorphic contracts -> ONNX checker/shape inference -> cost -> optional single-task LB probe only after a clear hidden-safety theory. The first target is not "port all of Hodel"; it is a cheap subset that covers holes, masks, painting, covering, bounding boxes, shifts, crops, and connected-component approximations for the top monsters.

Parent-scan update incorporated: ARC Prize 2024/2025 direction confirms program synthesis, test-time adaptation, and LLM guidance are complementary. For NeuroGolf, the baked-per-task ONNX scoring changes the priority: compact typed program synthesis is primary; neural/TTT/LLM machinery should propose, rank, or stress-test rules, not become the submitted graph.

## Highest-Signal Actions

### Next 48 Hours

1. Build a DSL frontier board from current artifacts.
   - Inputs: `tools/dsl/primitives.py`, `tools/dsl/search.py`, `reports/dsl_v4_search_results.csv`, `reports/dsl_v4_swap_candidates.csv`, `reports/candidate_v4_plus10_compiler2_score_n3.csv`.
   - Output: a report or CSV ranking every primitive by measured ONNX cost, search hit count, isolated full-example pass rate, and anchor cost delta.
   - Why: current docs understate the DSL state. The code already has about 29 primitive builders, including mask/flood/thicken/hollow/trim variants, but `dsl_v4_swap_candidates.csv` still has only the old cheap/tie set. Several search hits pass train/test in Python and then fail isolated ONNX eval, so the frontier is "optimize and gate what exists", not simply "add primitives".

2. Turn compiler-golf discoveries into a pass registry.
   - Encode the three LB-verified rules as named passes with explicit preconditions: BOOL mask lifetime preservation, broadcast-safe uniform initializer scalarization, and terminal `Cast -> output` narrowing.
   - Add a scanner that emits candidates and expected cost saves against `submissions/candidate_v4_plus10_compiler2_onnx/`, but writes only to reports until a batch is reviewed.
   - Success criterion: every pass has checker, strict shape inference, decoded equivalence, and cost measurement baked in. No ad hoc script should be allowed to touch anchors.

3. Add a cheap Hodel-core object/mask layer before more broad search.
   - Implement or optimize typed primitives: `mask_color(c)` as BOOL, `bbox(mask)`, `delta_bbox(mask)`, `paint_mask(grid,c,mask)`, `cover_mask(grid,mask)`, `where_grid(cond,a,b)`, `shift_{up,down,left,right}`, static `crop`, and `holes_not_bordering(mask)`.
   - Cost target: each mask/paint/cover/where primitive should be comfortably below the existing expensive primitives such as `select_channel` (~48K), `mask_foreground` (~51K), `bbox_crop` (~82K), `largest_blob` (~97K), `thicken` (~137K), and `flood_fill` (~193K) from `tools/dsl/search.py` cost hints.
   - Why: Hodel and icecuber both get leverage from object extraction, filtering, transform, and paint. In NeuroGolf, those must be typed and memory-cheap, or they lose to the current anchor.

4. Start a top-monster semantic triage lane.
   - First queue: `task255`, `task233`, `task191`, `task018`, `task285`, then `task076`, `task101`, `task096`, `task133`, `task158`.
   - For each task, write one compact row: visible train/test rule hypothesis, object inventory, likely Hodel primitives, current anchor cost, best known unsafe source, pseudo-hidden contracts to run, and the smallest ONNX compilation path.
   - Immediate likely work: `task255` needs translation-covariant flood/attachment semantics; `task233` needs largest component plus hole/patch reinsertion without visible-key lookup; `task191` is semantically closed and needs a boundary-aware ONNX compiler, not more rule inference.

5. Build the LLM program-induction loop around Python references, not ONNX first.
   - Prompt with grid examples, object summaries, existing DSL primitive docs, and a required Python `solve(grid)` function.
   - Run train/test, arc-gen, and pseudo-hidden contracts automatically.
   - Only then ask for a DSL program or builder. Record every failure mode in `reports/solver_ledger.md`.
   - Inference: this borrows Greenblatt-style massive program sampling but makes it usable for this repo's hidden-safety and ONNX-cost constraints.

6. Promote pseudo-hidden contracts to a mandatory candidate gate.
   - Extend `tools/pseudo_hidden.py` use from selected hand prototypes to every semantic candidate.
   - Minimum contracts per candidate: color permutation with semantic colors fixed, zero-border translation/padding with explicit output behavior, distractor insertion when the rule claims object irrelevance, and leave-one-out when the rule is learned from examples.
   - Reason: `v4_plus6` showed full visible parity is not enough for arbitrary source classes.

### Next Week

1. Ship a typed DSL compiler milestone.
   - Stop treating every intermediate as `[1,10,30,30] FLOAT`.
   - Allow BOOL masks, INT64 scalar coordinates/counts, FLOAT only when required by ONNX op constraints, and terminal non-FLOAT outputs when decoder-equivalent.
   - Add primitive metadata: input kind, output kind, shape envelope, static/dynamic shape status, cost hint, and legal compiler rewrites.

2. Add object search with memoization and constraints.
   - Represent intermediate states by typed signatures: shape, palette, component counts, bbox stats, holes, symmetries, and cost.
   - Search depth 3 over cheap typed primitives first; depth 2 over expensive object primitives.
   - Prune by output shape/palette and by constraints learned from all train examples.
   - Use state hashing so equivalent object/mask states are not re-expanded.

3. Port the Hodel subset that maps directly to ONNX.
   - Good first set: `objects` only as bounded special cases, `colorfilter`, `size`, `fill`, `paint`, `cover`, `crop/subgrid`, `shift`, `normalize`, `delta`, `box/inbox/outbox`, `hconcat/vconcat`, `upscale/downscale`, and `branch`.
   - Defer full dynamic `objects()` until the special-case object layer proves cost competitive. A fixed-K component representation is likely needed, but it should be task-driven.

4. Build synthetic hidden validation per high-value task.
   - Use Hodel RE-ARC as the model: write generator contracts for the task distribution, not just perturbations.
   - Start with `task255`, `task233`, `task191`, and `task366`, where the repo already has semantic footholds.
   - Gate: a candidate should pass train/test, arc-gen, and generated cases whose invariants were specified before looking at failures.

5. Run agent lanes with shared ledgers.
   - Lane A: compiler passes and ONNX cost scanning.
   - Lane B: DSL primitive optimization and typed compiler.
   - Lane C: task semantics and pseudo-hidden contracts.
   - Lane D: LLM Python-solver sampling.
   - All lanes write candidate ledgers; only a release owner builds anchor candidates.

### Longer

1. DreamCoder-style library learning.
   - Once 30-50 DSL programs exist, mine repeated fragments into named macros and bias search toward them.
   - Do not train a neural guide first; the initial dataset is too small. Start with frequency/cost priors and macro extraction.

2. Neurally-guided or LLM-guided search.
   - Train/rank only after the symbolic pipeline produces enough solved programs and failed candidates.
   - The useful model is a proposal/ranking model over typed primitives and object summaries, not an end-to-end grid predictor for submission.

3. Equality-saturation-style ONNX superoptimization.
   - Build a tiny e-graph or rewrite saturation engine for the subset of ONNX ops used here.
   - Objective is NeuroGolf cost, not runtime.
   - Rules must be typed and shape-aware; every extracted graph must pass the same decoded-equivalence harness.

4. Source-class reputation and LB-probe economics.
   - Keep per-source trust labels: anchor-class, normal measurable, hidden-unsafe, scorer-boundary, excluded-op.
   - Do not batch-merge visible-clean sources unless their class is already LB-validated.

## Repo Execution Manifest

This section is the concrete action layer. It translates the external direction into primitives/templates, harness features, and task-family queues for this workspace.

### Primitive and Template Backlog

| priority | primitive/template | value type | ONNX shape intent | cost target | target task families | first tasks |
|---|---|---|---|---:|---|---|
| P0 | `mask_color(c)` | BOOL mask | `[1,1,30,30]` or `[1,10,30,30]` only if needed | <10K | object, color, crop | 255, 233, 366, 191 |
| P0 | `paint_mask(grid,c,mask)` | grid | preserve grid envelope; output BOOL/FLOAT only as needed | <20K | hole fill, object paint | 255, 233, 366 |
| P0 | `cover_mask(grid,mask,bg=0)` | grid | replace masked cells with background | <20K | remove/shift/repaint | 233, 366, 191 |
| P0 | `where_grid(cond,a,b)` | grid | branch between two grids without full FLOAT casts | <20K | conditional rules | 018, 285, 133 |
| P0 | `bbox(mask)` | scalar tuple | INT64 `r0,r1,c0,c1,h,w` | <15K | crop/reduce/object | 233, 366, 096, 264 |
| P0 | `crop_static(r0,c0,h,w)` | grid | static Slice/Pad, scoreable shapes | <2K for small crops | crop/reduce | 014, 029, 036, 326 |
| P0 | `shift_mask/drift(grid,dr,dc)` | mask/grid | Gather/Pad with mask, all four directions | <15K | translate/stamp | 191, 366, 285 |
| P0 | `d4_transform(grid_or_mask,id)` | mask/grid | rotate/flip specialized by known input bbox | <10K when small | symmetry/stamp | 191, 018, 076, 101 |
| P1 | `holes_not_bordering(mask)` | BOOL mask | border flood from background, invert inside bbox | <60K | enclosure/hole fill | 255, 233 |
| P1 | `largest_component(color, conn=4)` | BOOL mask | bounded/special-case first; avoid dynamic object count | <80K | object/crop | 233, 366, 204 |
| P1 | `component_summary(color)` | scalars | bbox, area, touches_border, hole_count | <80K | object filtering | 233, 255, 366 |
| P1 | `stamp_clipped(pattern, anchor, d4)` | grid | boundary-aware paint; no visible lookup table | <100K | motif completion | 191 |
| P1 | `palette_map(mapping_or_rule)` | grid | Gather/Conv 1x1 or conditional map | <5K for static map | color remap | 025, 071, 034 |
| P1 | `tile_bbox(axis,n)` | grid | repeat extracted bbox content | <20K for small bbox | tiling/expand | 001, 019, 249 |
| P2 | `objects_k(color,K)` | fixed object set | K masks plus scalar summaries | task-driven | multi-object | 133, 285, 319 |
| P2 | `line_between/end_ray` | BOOL mask | row/col/diag line mask | <30K | line/connect | 285, 118 |

Implementation notes:

- Every primitive gets a Python reference, an ONNX builder, a typed signature, a cost hint, and a one-line hidden-safety note.
- Prefer BOOL masks and INT64 scalars. Only widen to FLOAT at op boundaries that require it.
- Score all new builders with `tools/isolated_task_eval.py` before adding them to search.
- Do not add a primitive to broad search until it has at least one target task and a measured cost envelope.

### Reusable Solver Templates

| template | sketch | required primitives | candidate tasks | pass/fail gate |
|---|---|---|---|---|
| `extract_filter_paint` | find object/mask, filter by color/area/bbox, paint or cover | `mask_color`, `component_summary`, `paint_mask`, `cover_mask` | 233, 366, 204 | train/test + arc-gen + color-permutation contracts |
| `hole_fill_inside_object` | find enclosing component, compute non-border holes, fill literal or inferred color | `mask_color`, `bbox`, `holes_not_bordering`, `paint_mask` | 255, 233 | zero-border translation must match declared behavior |
| `crop_largest_or_salient` | select component by area/color/touching, crop bbox, optionally normalize | `largest_component`, `bbox`, `crop_static/dynamic`, `shift_mask` | 233, 366, 096, 264 | output shape must be predicted from train/test |
| `d4_stamp_completion` | infer source motif, apply D4 transforms, complete clipped occurrences | `d4_transform`, `stamp_clipped`, `paint_mask`, `where_grid` | 191, 018, 076, 101 | pseudo-hidden D4 and boundary cases pass |
| `object_translate_repaint` | cover old object, shift normalized object, repaint | `cover_mask`, `shift_mask`, `paint_mask`, `component_summary` | 366, 285, 133 | distractor insertion does not change unrelated output |
| `palette_rule_map` | static or conditionally inferred color mapping | `palette_map`, `count_color`, `where_grid` | 025, 071, 034, 017 | color permutation with fixed semantic colors |
| `bbox_tile_or_expand` | crop object/bbox and repeat/concat to output size | `bbox`, `crop_static`, `tile_bbox`, `hconcat/vconcat` | 001, 019, 249 | shape extrapolation contracts pass |
| `scalar_or_count_output` | count colors/components or choose scalar class | `count_color`, `component_summary`, `where_grid` | scalar-output family, 018 if applicable | leave-one-out rule induction passes |

### Harness Features To Build

| feature | file target | behavior | why it matters |
|---|---|---|---|
| DSL frontier board | new report script, e.g. `tools/dsl/frontier_report.py` | joins primitive cost hints, `dsl_v4_search_results`, isolated pass counts, anchor costs, and swap deltas | tells workers what to optimize next |
| typed primitive registry | `tools/dsl/primitives.py` or new `tools/dsl/types.py` | records input/output kind, dtype, shape envelope, cost hint, static-shape status | prevents accidental FLOAT-grid materialization |
| task object summaries | new tool or extend `tools/build_task_dossiers.py` | per-color components, bbox, area, holes, border touch, symmetry, palette deltas | feeds search pruning and LLM prompts |
| pseudo-hidden contract registry | extend `tools/pseudo_hidden.py` usage | stores per-task contracts with explicit expected output transforms | turns hidden-safety assumptions into testable artifacts |
| LLM rule sampler | new tool, report-only by default | generates Python `solve(grid)`, runs train/test/arc-gen/contracts, dedupes behavior, logs candidates | uses LLMs as rule proposers, not submitted models |
| program corpus ledger | report CSV/JSON | task, family, primitive sequence, typed trace, cost, pass status, failure reason | enables DreamCoder-style macro mining later |
| compiler pass registry | new compiler backend module | named passes with preconditions, rewrite, checker/shape/equivalence/cost gates | makes compiler golf repeatable and safe |
| source-class ledger | extend provenance reports | trust labels for public/anchor/generated source classes | prevents another visible-clean batch failure |

Minimum candidate gate:

1. Python reference passes train/test.
2. If learned from examples, leave-one-out passes.
3. Arc-gen passes or failures are explained by a train/test-grounded limitation.
4. Pseudo-hidden contracts pass or are rejected before tuning with a written reason.
5. ONNX builder passes checker and strict shape inference.
6. Isolated full-example eval passes.
7. Candidate cost is lower than anchor, or the candidate is explicitly marked research-only.

### Task-Family Queues

Use `reports/task_dossiers_20260606.md` as the queue source. The family counts there are: symmetry/flip/rotate 105, reduce 67, color_remap 37, tiling 30, color_fill 29, crop 22, expand 6, scalar_output 6, trim_border 6, plus other/unclassified.

| queue | tasks to open first | shared template | concrete next action |
|---|---|---|---|
| Hole/enclosure/flood | 255, 233, 204, 118 | `hole_fill_inside_object`, `extract_filter_paint` | write contracts first, then build cheap `mask_color`/`holes_not_bordering` refs; do not compile monoliths until translation contracts pass |
| Crop/reduce/object | 233, 366, 096, 264, 243, 029, 036, 014 | `crop_largest_or_salient` | add object-summary extractor; compare bbox/color rules across train/test; only use static crop when shape is fully predicted |
| Symmetry/stamp/D4 | 191, 018, 285, 101, 076, 133, 158, 054, 286 | `d4_stamp_completion`, `object_translate_repaint` | prioritize `task191` compiler because semantics are closed; for others generate D4 summaries and LLM Python hypotheses |
| Color remap/fill | 025, 071, 034, 017, 010, 023 | `palette_rule_map`, `extract_filter_paint` | search static maps first; if public-source matched unsafe, require pseudo-hidden color-permutation proof |
| Tiling/expand | 001, 019, 249, 003 | `bbox_tile_or_expand` | implement small-bbox tile templates and shape extrapolation checks |
| Scalar/count | scalar_output family, possibly 018 | `scalar_or_count_output` | add count/component summary refs and leave-one-out induction harness |
| Compiler-only backend | v4_plus10-changed tasks plus top-cost nonsemantic graphs | compiler pass registry | scan for terminal casts, BOOL/FLOAT lifetime widenings, uniform tensors, duplicate inits, and scoreable static-shape opportunities |

### First 10 Work Orders

1. `tools/dsl/frontier_report.py`: join DSL hit rows with v4_plus10 cost CSV and emit top primitive optimization targets.
2. `tools/task_object_features.py`: generate component/bbox/hole/palette summaries for every train/test example.
3. Typed registry: add metadata for all existing `BUILDERS` entries before adding new broad-search primitives.
4. Implement `mask_color`, `paint_mask`, `cover_mask`, and `where_grid` as cheap typed primitives with tests.
5. Add task-specific pseudo-hidden contract files or registry entries for `task255`, `task233`, `task191`, `task366`.
6. Compile `task191` through a boundary-aware `d4_stamp_completion` path; compare cost to 691342 anchor.
7. Rework `task255` Python semantics until zero-border translation/far-noise contracts stop falsifying it.
8. Rework `task233` around largest color-2 component plus holes; ban patch-key lookup.
9. Build an LLM Python-solver sampler for the top 20 dossier priority tasks, report-only.
10. Convert compiler-golf scripts into pass registry entries and run a report-only scan against v4_plus10.

## External Evidence and Repo Inference

### icecuber ARC 2020

Fact: Johan Sokrates Wind's 1st-place ARC 2020 writeup says the main component was a DSL applying up to 4 of 142 unary transformations, based on 42 functions, with duplicate reduction and greedy stacking to fit training samples. It ran multiple configurations, including depth 3 and depth 4 until time/memory ran out. Source: `ARC-solution_documentation.pdf` in the official repo, https://github.com/top-quarks/ARC-solution.

Fact: The writeup highlights hand-solving tasks to extract useful functions, and names transformations such as cut, color filter, compose, compress, rigid transforms, and pick-by-property.

Inference for this repo: the useful unit is not a neural model and not a single global optimizer. It is a per-task program search over compact transformations. NeuroGolf differs because outputs are ONNX and cost-scored, but the search shape is the same: generate many candidate programs, cache intermediate states, and pick the cheapest visible-correct one.

Concrete task: extend `tools/dsl/search.py` from depth<=2 enumeration to a cost-pruned, memoized depth-3 search over cheap typed primitives, while keeping expensive object primitives depth-limited.

### Hodel ARC-DSL and RE-ARC

Fact: `michaelhodel/arc-dsl` contains 160 primitive definitions in `dsl.py` and 400 solver functions in `solvers.py` as of this research pass. Its README describes a DSL intended to be expressive and generic, with solver programs for the 400 ARC training tasks. Source: https://github.com/michaelhodel/arc-dsl.

Fact: Hodel's README examples follow an extract/filter/transform/paint pattern: `objects`, `colorfilter`, `mfilter`, `fill`, `cover`, `shift`, `paint`, and higher-order combinators.

Fact: Hodel's RE-ARC paper says each of the 400 ARC training tasks has a procedural example generator following the transformation logic of the original examples, intended to sample broader task distributions. Source: https://arxiv.org/abs/2404.07353.

Inference for this repo: porting Hodel line-by-line is less important than importing the intermediate representation discipline: grid -> object/mask/geometry -> transformed object/mask -> painted grid. The current DSL's missing leverage is not another color replacement; it is cheap object/mask/geometry state.

Concrete task: define typed object/mask primitives as compiler concepts first, then add Python refs and ONNX builders. The compiler should know when an intermediate is a BOOL mask, scalar bbox, color index, or full grid.

### DreamCoder and Neural-Guided Program Synthesis

Fact: DreamCoder learns programs while growing a DSL/library and training a recognition model to guide search via a wake-sleep loop. Source: https://arxiv.org/abs/2006.08381.

Fact: "Neural-guided, Bidirectional Program Search for Abstraction and Reasoning" applies DreamCoder-style abstraction learning to ARC tasks and explores inverse-semantics/bidirectional search. Source: https://arxiv.org/abs/2110.11536.

Fact: Ouellette's "Towards Efficient Neurally-Guided Program Induction for ARC-AGI" compares learning grid space, program space, and transformation space, retaining program-space learning for an ARC-AGI submission while pointing to transform-space learning as promising. Source: https://arxiv.org/abs/2411.17708 and code: https://github.com/SimonOuellette35/GridCoder2024.

Inference for this repo: neural guidance is not a next-48h dependency. The repo first needs a corpus of accepted programs and failed candidates. But the DreamCoder lesson should shape the data we log now: every accepted DSL program should be saved as a reusable training example with task features, primitive sequence, cost, and failure contrasts.

Concrete task: add a `program_corpus.csv` ledger for DSL/hand/LLM solutions: task, primitive sequence, typed trace, visible pass, pseudo-hidden pass, cost, replaced anchor cost, and notes.

### Object-Centric ARC Solvers

Fact: ARGA represents grids as object-centric graphs and searches a DSL in graph space, using constraint acquisition, state hashing, and Tabu search to tame combinatorics. Source: https://arxiv.org/abs/2210.09880.

Fact: Ferre's object-centric MDL approach searches a large model space with the Minimum Description Length principle and emphasizes object-centric descriptions aligned with human-style programs. Source: https://arxiv.org/abs/2311.00545.

Fact: A study of LLMs on ARC reports that object-based representations significantly improve reasoning, while GPT-4 still fails many non-language ARC-style tasks. Source: https://arxiv.org/abs/2305.18354.

Inference for this repo: object summaries should be first-class prompt/search input. This is also a cheap way to improve LLM-generated Python rules without immediately changing ONNX builders.

Concrete task: implement a task-feature extractor that emits per-color connected components, bbox, area, holes, adjacency, symmetry, palette, and shape deltas for train/test examples. Feed that to both search pruning and LLM prompts.

### LLM Program Induction

Fact: Ryan Greenblatt's ARC-AGI writeup describes generating thousands of Python programs with GPT-4o, selecting programs that fit examples, and applying them to test inputs. Source: https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt and code: https://github.com/rgreenblatt/arc_draw_more_samples_pub.

Fact: The ARC Prize 2024 report says SOTA on ARC-AGI private evaluation rose from 33% to 55.5%, with progress driven by deep-learning-guided program synthesis and test-time training. Source: https://arxiv.org/abs/2412.04604 and official summary https://arcprize.org/blog/arc-prize-2024-winners-technical-report.

Inference for this repo: large-scale LLM sampling is actionable if it is harnessed as "generate Python reference -> validate -> compile", not as "trust model reasoning". NeuroGolf's ONNX cost objective adds a second filter: a correct Python rule is only valuable if it compiles to a smaller graph or exposes a safe source swap.

Concrete task: build a batch runner that asks for N Python solvers per task, deduplicates by behavior on generated examples, keeps only train/test/arc-gen/pseudo-hidden passers, then maps them to DSL primitives or a custom builder.

### Tensor Compiler and Golfing

Fact: ONNX Runtime provides graph optimization levels with graph simplifications, node eliminations, fusions, and layout optimizations, available online or offline. Source: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html.

Fact: ONNX Optimizer is an official optimizer library with prepackaged graph optimization passes. Source: https://github.com/onnx/optimizer.

Fact: TASO automatically generates and formally verifies DNN graph substitutions, then uses cost-based search to optimize graphs. Source: SOSP 2019 paper https://www.cs.cmu.edu/~zhihaoj2/papers/sosp19.pdf.

Fact: `egg` is a general equality saturation engine for applying many rewrite rules and extracting the lowest-cost equivalent expression. Source: https://arxiv.org/abs/2004.03082.

Inference for this repo: generic ONNX optimizers are not enough because the scoring config disables runtime graph optimization and the objective is memory/parameter cost, not latency. But TASO/egg provide the correct pattern: a typed rewrite rule set, saturation/search, and cost-based extraction.

Concrete task: start with a tiny custom pass registry, not full e-graphs. Rules already proven on LB should become reusable compiler passes. New rules should be mined from repeated ONNX motifs and validated by decoded equivalence.

### Metamorphic Testing and Synthetic Hidden Validation

Fact: The metamorphic testing survey defines metamorphic testing as test generation and verification using necessary relations between multiple inputs and outputs, especially useful when exact oracles are hard. Source: https://i.cs.hku.hk/~tse/Papers/2010s/hlmtCSUR.html.

Fact from this repo: `tools/pseudo_hidden.py` already implements explicit invariance contracts for color permutations, zero-border padding/translation, distractor insertion, and leave-one-out callbacks.

Inference for this repo: pseudo-hidden should be promoted from prototype aid to release gate. It directly addresses the observed gap between visible parity and hidden LB safety, especially after the `blend_v360` failure.

Concrete task: every semantic solver candidate must declare contracts before failure inspection. A failing contract should either block the candidate or be documented as too strong for the task with a specific reason.

### Agent Workflows

Fact: ReAct interleaves reasoning traces and actions so an LLM can update plans while using external tools/environments. Source: https://arxiv.org/abs/2210.03629.

Fact: Reflexion uses feedback signals converted into verbal memory to improve later agent attempts. Source: https://arxiv.org/abs/2303.11366.

Fact: Voyager uses an automatic curriculum, an executable skill library, and iterative prompting with environment feedback/self-verification. Source: https://arxiv.org/abs/2305.16291.

Fact: SWE-agent shows that agent-computer interface design matters for software engineering agents. Source: https://papers.neurips.cc/paper_files/paper/2024/file/5a7c947568c1b1328ccc5230172e1e7c-Paper-Conference.pdf.

Inference for this repo: "more agents" is not enough. The agentic part should be structured around artifacts: ledgers, task queues, primitive docs, pass registries, and validators. Each worker should have a narrow loop and a hard gate.

Concrete task: create repeatable worker briefs for four lanes: compiler pass mining, typed DSL primitives, task semantics/pseudo-hidden, and LLM Python-solver sampling. Each lane appends to a single canonical ledger and is forbidden from modifying anchors or submitting.

## Current Repo Fit

### What Already Works

- The local scorer has repeatedly matched LB for trusted source classes and exact graph rewrites.
- Three compiler rules are LB-verified: BOOL mask lifetime, uniform initializer scalarization, and terminal cast narrowing.
- The DSL compiler exists and has Python refs plus ONNX builders.
- `tools/dsl/run_v4_search.py` can scan 400 tasks and compile/evaluate hits.
- `tools/pseudo_hidden.py` gives the right contract-driven validation shape.
- Several semantic prototypes exist, with `task191`, `task133`, and `task366` semantically closed in the ledger.

### What Is Still Missing

- Typed intermediate representation. The current DSL still pays large costs for full FLOAT grid intermediates in many primitives.
- Object/mask geometry as first-class reusable state.
- A search frontier board that distinguishes Python train/test hits from isolated ONNX full-example passers.
- A program corpus for future library learning or neural guidance.
- A disciplined LLM generation harness.
- Synthetic hidden generation beyond simple metamorphic perturbations.

## Do Not Do

- Do not submit Kaggle from this branch.
- Do not modify `submissions/candidate_v4_plus10_compiler2_onnx/` or any LB anchor directory.
- Do not batch-merge public sources by visible parity.
- Do not build visible-only lookup tables unless explicitly labeled hidden-unsafe and kept out of candidates.
- Do not tune thresholds to arc-gen-only failures without train/test-grounded justification.
- Do not spend the next 48 hours on end-to-end neural inference; NeuroGolf's cost objective makes that structurally unattractive for submission graphs.

## Concrete Engineering Backlog

### P0: 48h Backlog

- Create DSL frontier report script: summarize primitive costs, search hits, isolated pass counts, and swap candidates against `v4_plus10`.
- Convert compiler-golf scripts into pass objects with precondition, rewrite, validation, and measured saving.
- Add object-summary extraction for every task: components, bbox, holes, adjacency, palette, symmetry, shape relation.
- Implement cheap typed `mask_color`, `paint_mask`, `cover_mask`, `where_grid`, all-direction `shift`, static `crop`, and `delta_bbox`.
- Write pseudo-hidden contract templates for `task255`, `task233`, `task191`, and `task366`.
- Add an LLM Python-solver batch runner with automatic rejection and behavior deduplication.

### P1: One-Week Backlog

- Add typed `DslValue` or equivalent metadata so builders can compose masks/scalars/grids without forced FLOAT grid materialization.
- Add state hashing and cost-pruned depth-3 search for cheap primitives.
- Implement bounded object extraction for common cases: single largest component, per-color components up to K, holes not touching border.
- Compile `task191` semantic rule with boundary-aware stamping and D4 transforms.
- Reopen `task255` and `task233` only through train-grounded semantic fixes plus pseudo-hidden contracts.
- Start a `program_corpus.csv` or markdown ledger for all accepted and rejected programs.

### P2: Longer Backlog

- Mine repeated accepted programs into macros, DreamCoder-style.
- Add learned or LLM proposal ranking once the corpus has enough accepted/rejected examples.
- Build an equality-saturation or TASO-like ONNX optimizer for the repo's actual op subset.
- Maintain source-class reputation and LB-probe policy as a formal ledger.

## Bottom Line

The repo has already proven it can execute exact ONNX rewrites safely. The missing slope is semantic production: many more small programs. The fastest credible path is a typed object/mask DSL plus LLM-assisted Python induction, both guarded by pseudo-hidden tests and compiled under a cost-aware pass registry. The next 48 hours should make that pipeline measurable and start attacking `task255`, `task233`, and `task191` with reusable primitives rather than one-off monoliths.
