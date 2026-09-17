# Handoff to Next Agent (NeuroGolf 2026)

**Last updated**: 2026-06-07 ~02:35 CST  
**Workspace**: `$HOME/Desktop/kaggleonnx`  
**Python**: `.venv/bin/python`

## 0. 2026-06-06 local/remote state check

- This workspace is **not a Git working tree** (`$HOME/Desktop/kaggleonnx`
  has no `.git`), and there is no project git remote to fetch. Treat Kaggle
  submissions + local reports as the source of truth.
- Current LB-verified anchor is **`v4_plus16_seddik_t001` at public LB
  6260.89**: `submissions/candidate_v4_plus16_seddik_t001_onnx/` and
  `submissions/submission_candidate_v4_plus16_seddik_t001.zip` (sha256
  `e7ac37d6240fca04b603db46bfa52475b15be024f626d60f71b07988cab18db7`).
- 2026-06-07 remote recheck confirms submission `53421634` is `COMPLETE` at
  public LB **6260.89**. Current checked top-10 cutoff is **7339.93**.
- 2026-06-07 minimal loop:
  - `reports/compiler_backend_sweep_v4_on_v16.md`: exact backend sweep over
    v16 evaluated 1 candidate, applied 0 rewrites, final n=3 unchanged
    **6260.896515**. The old compiler pass family is saturated on v16.
  - `submissions/handbuilds/task025_priority.onnx` and
    `reports/task025_contract_audit_20260607.md`: task025 high-color priority
    tail fixes the older semantic probe's two pseudo-hidden failures
    (`266/266` visible, `62/62` generated), but cost is **541975**, worse than
    the anchor cost **140093**. Do not stage; next task025 move is compression.
- `v4_plus16_seddik_t001` differs from `v4_plus15_bio_micro2` only on
  `task001`, using Seddik precision-source tiny exact micro surgery. Full
  bundle local n=3 **6260.896515**; Kaggle submission `53421634` completed and
  scored **6260.89**. The displayed LB is tied with v15 to two decimals, but
  the local scorer records a real `+0.006311` gain.
- `v4_plus15_bio_micro2` differs from `v4_plus14_bio_micro` on 22 additional
  full-visible-clean `biohack44_superior` micro swaps:
  `task003/034/059/089/129/175/208/212/216/224/239/247/263/265/273/275/281/289/348/360/383/397`.
  Full bundle local n=3 **6260.890204**; Kaggle submission `53421427` scored
  **6260.89**. This makes the remaining `biohack44_superior` micro lane
  LB-validated, but the gains are now tiny.
- `v4_plus14_bio_micro` differs from `v4_plus13_task366` only on
  `task145/153/248/357`, using full-visible-clean `biohack44_superior` micro
  swaps. Full bundle local n=3 **6260.747441**; Kaggle submission `53421167`
  scored **6260.74**.
- `v4_plus13_task366` differs from `v4_plus12_task285` only on `task366`.
  It overlays `submissions/handbuilds/task366_compact_v3.onnx`, combining the
  max-7 rectangle bound with density-based source-panel selection. Validation:
  semantic visible/arc-gen `266/266`, ONNX encodable target equivalence
  `255/255`, ONNX encodable semantic equivalence `255/255`, pseudo-hidden
  `1456 passed / 0 failed / 406 skipped`, checker and strict shape inference
  passed. Cost `830720 -> 734516`, local delta `+0.123081`. Full bundle local
  n=3 **6260.487936**; Kaggle submission `53420666` scored **6260.48**.
- `v4_plus12_task285` differs from `v4_plus11_compiler3` only on `task285`.
  It overlays `submissions/handbuilds/task285_v2.onnx`, an exact
  decoded-equivalent compiler rewrite of the current task285 graph. Validation:
  visible/arc-gen `265/265`, decoded equivalence vs anchor `265/265`,
  pseudo-hidden ONNX contracts `3375 passed / 0 failed / 70 skipped`, checker
  and strict shape inference passed. Cost `395468 -> 326044`, local delta
  `+0.193038`. Full bundle local n=3 **6260.364855**; Kaggle submission
  `53418978` scored **6260.36**.
- `v4_plus11_compiler3` differs from `v4_plus10_compiler2` only on
  `task157` and `task295`: 9 exact bool-source Cast-chain collapses and one
  static keepdims-axis `ReduceSum` fusion. Full bundle local n=3
  **6260.171818**; Kaggle submission `53418833` scored **6260.17**.
- `v4_plus10_compiler2` differs from `v4_plus9_compiler` on exactly 10 tasks:
  `task092/138/191/192/205/209/215/376/377/396`. It removes terminal
  `Cast -> output` nodes and exposes the BOOL/FLOAT16 producer directly as the
  graph output. All 10 changed models passed checker, strict shape inference,
  full decoded equivalence, and full bundle n=3 scoring. Local total
  **6260.162659**; Kaggle submission `53417133` scored **6260.16**.
- `v4_plus9_compiler` differs from `v4_plus8_task149_bool` on exactly 39
  tasks: 32 conservative uniform-initializer-to-scalar rewrites, 6 BOOL-memory
  rewrites (`task240/301/308/316/333/374`), plus `task149` v2 with cost
  `172 -> 171`. All changed models passed `onnx.checker.check_model(full_check=True)`,
  strict shape inference, full decoded equivalence on their train/test/arc-gen
  examples, and full n=3 bundle scoring with zero scorer errors. Local total
  **6258.144245**; Kaggle submission `53416534` scored **6258.14**.
- `v4_plus8` differs from `v4_plus7` only on `task149`. It is an exact
  decoded-equivalent rewrite of the LB-verified rule using negative-pad Conv
  plus BOOL `Greater/Not/Concat/Pad`. Full isolated audit is `267/267`; cost
  `509 -> 172`; local `6257.0466`; Kaggle submission `53415813` scored
  `6257.04`. BOOL intermediate memory reduction is now LB-proven.
- `v4_plus7_task149` differs from `v4_plus5` only on `task149`, replacing the
  anchor file with `massim_6254/task149.onnx`. Local total is **6255.9616** and
  Kaggle submission `53415130` scored **6255.96**. Isolated audit:
  `267/267`, cost `509`, score `18.7676` vs `v4_plus5` cost `5139`, score
  `16.4554`, delta **+2.3122**.
- `v4_plus5` differs from `v4_plus4` only on `task150` and `task155`, using DSL
  `flip_h`/`flip_v`. Local total is **6253.6495** and Kaggle submission
  `53355543` scored **6253.64**. Spot isolated audits on both swapped tasks:
  `266/266`, cost `832` each.
- `v4_plus6` is **not** an anchor. It overlaid 22 `blend_v360` tasks and looked
  locally strong (**6276.21**, +22.56 vs v4_plus5), but Kaggle submission
  `53384773` scored **5966.68**. A follow-up isolated audit confirmed all 22
  swapped tasks pass visible full-example eval, so this is a hidden-safety
  failure of the source, not a local preflight miss.
- Immediate implication: do not batch-absorb `blend_v360` / `konbu17
  neurogolf-2026-blend-source-v3-6-0` tasks by visible-pass parity. Any future
  use of that source needs single-task probes or a concrete hidden-safety
  theory.
- New public-source split: `massim_6254` is a normal measurable source but not
  batch-trusted. Its `task149` is now LB-verified; its apparent `task191` and
  `task264` wins are rejected because full isolated audit showed visible
  regressions (`227/267` and `0/265` respectively). `beicicc_6645` and
  `afr1ste_6284` expose scorer/environment-boundary ideas but are not directly
  mergeable under the current scorer.

This document is the canonical entry point for the next agent. Read everything
in section 1 before doing anything. Then read sections 2–4 to understand state.
Section 5 lays out the decision tree for the next phase. Section 6 has the
hard rules.

---

## 1. Two-minute orientation

### What this project is

Kaggle competition `neurogolf-2026`. Each `task_id` 1..400 takes input grids
and produces output grids (ARC-AGI style few-shot rule reasoning). For each
task we submit one ONNX model. Score per task = `visible_pass_rate * max(1,
25 - ln(cost))`. Total = sum across 400 tasks. Hidden test set decides final
LB.

### Where we are

- **LB-verified anchor**: `v4_plus16_seddik_t001` at LB **6260.89** (zip
  `submissions/submission_candidate_v4_plus16_seddik_t001.zip`)
- **Top-10 cutoff**: **7339.93**; leader **7695.78**
- **Gap**: ~1079.19 to top-10, ~1435.04 to rank 1
- **Focused threads**:
  - `019e9c90-54f6-7b53-9fee-a01270fe3e93`: `task366` compact
    bounded-rectangle compiler. Completed; v3 is LB-verified as v13.
  - `019e9c90-5605-7363-907c-e58905ed5e74`: `task191` guarded compact
    D4 matcher. Completed; guarded_v4 is non-stageable (`211/267`).
  - `019e9c95-f4f2-7b70-a2b3-6ea20181692e`: latest public
    ecosystem/discussion audit. Completed; no direct big package, source-lake
    only.
  - `019e9c96-162b-79a2-8d2a-2db5a31995d5`: semantic-factory next-pick audit
    for `task158/173/286/054/025`. Completed; chose `task025`.
  - `019e9ca0-ded0-74e3-b826-5da1f5723181`: task025 contract audit.
    Completed; anchor passes `62/62`, konbu public candidate fails `0/62`.

### Path so far (anchor evolution)

| anchor | local | LB | what changed | submitted |
|---|---:|---:|---|---|
| current_best_6122 | 6122.60 | 6122.59 | original LB-verified | yes |
| candidate_nadeem6252_plus_ours_v2 | 6253.50 | — | nadeem 6252 base, revert task018+task264 to ours | no |
| v3 (lossless on v2) | 6254.60 | — | int/opt/sim lossless surgery on Nadeem-source tasks | no |
| **v4** (stable) | 6245.41 | 6245.41 | revert 71 Nadeem tasks (task191 silent regression + 6 high-struct + 64 low-delta) | yes |
| v4_plus | 6246.17 | 6246.16 | task018←biohack44_super, task249←afr1ste_5689 | yes |
| v4_plus3 | 6251.55 | 6251.54 | revive 6 high-delta Nadeem tasks (task184/240/255/301/349/396) | yes |
| v4_plus4 | 6253.25 | 6253.25 | revive 64 low-delta audit-clean Nadeem tasks | yes |
| v4_plus5 | 6253.65 | 6253.64 | DSL flip_h/flip_v rewrites for task150/task155 | yes |
| v4_plus6 | 6276.21 | 5966.68 | 22 blend_v360 visible-clean swaps; hidden-catastrophic | yes, rejected |
| v4_plus7_task149 | 6255.96 | 6255.96 | task149←massim_6254 single-task audit-clean probe | yes |
| v4_plus8_task149_bool | 6257.05 | 6257.04 | task149 BOOL memory fusion, cost 509→172 | yes |
| v4_plus9_compiler | 6258.14 | 6258.14 | 32 uniform scalars + 6 BOOL rewrites + task149 v2 | yes |
| v4_plus10_compiler2 | 6260.16 | 6260.16 | terminal Cast output narrowing on 10 tasks | yes |
| v4_plus11_compiler3 | 6260.17 | 6260.17 | exact task157 Cast-chain + task295 ReduceSum backend rewrites | yes |
| v4_plus12_task285 | 6260.36 | 6260.36 | task285 exact compact compiler rewrite, cost 395468→326044 | yes |
| v4_plus13_task366 | 6260.49 | 6260.48 | task366 compact density selector, cost 830720→734516 | yes |
| **v4_plus14_bio_micro** | 6260.75 | **6260.74** | biohack superior micro swaps task145/153/248/357 | yes |

Local-LB delta within 0.01 across the Nadeem/DSL/Massim-task149 anchor submissions
→ our local scorer is byte-faithful for known-safe source classes. `v4_plus6`
proves that visible full-example parity is not enough for arbitrary new
public-source bundles.

### Critical proven facts

1. **task018 refined**: the `biohack44_super` / `nadeem_6252` task018 file is
   LB-safe inside our anchor, but a cost-0 Identity replacement was **not**.
   Submission `53334520` scored **6241.32** vs anchor `6253.25` (delta
   **-11.93**), despite both versions being `0/266` visible fail locally.
   Therefore "visible-fail-tied + cheaper cost" is **not** a valid generic
   hidden-safety rule.
2. **H2 (struct heuristic)**: "high init_byte_ratio + fp16" structural risk
   does NOT predict hidden failure. visible-pass-parity audit is the valid
   safety proxy. Reviving 6 + 64 = 70 Nadeem tasks blocked by structural
   heuristic gave +7.09 LB exactly as predicted locally.
3. **Compiler-golf status**: `v4_plus12` proves four general
   exact-equivalence backend rules on hidden LB: preserve mask lifetimes as
   BOOL, scalarize broadcast-safe uniform initializers, and remove terminal
   `Cast -> output` when the competition decoder accepts the producer dtype,
   plus narrow ARC color-valued grids to integer dtypes when all consumers
   accept them. Treat these as typed-IR compiler rules,
   but do not expect compiler golf alone to close the top-10 gap.
4. **Public source saturation**: 7+ public packages mined. v4_plus12 is the
   merge of every audit-clean swap from {nadeem_6252, biohack44_*, afr1ste_*,
   octv_*, haoranran_*, current_best_6122} plus the two DSL v1 flips and the
   LB-verified Massim/task149/compiler/task285 rewrites. Cherry-pick path is
   exhausted for trusted sources, but new normal sources can still yield
   single-task probes.
5. **New-source caution**: `blend_v360` / konbu17 v3.6.0 produced 22
   visible-clean improvements but lost 286.96 LB points on Kaggle. Treat it
   as hidden-unsafe until proven otherwise.

---

## 2. Strategy: how to actually reach top-10

Read **`reports/road_to_top10.md`** in full — it's the canonical strategy doc.
Key conclusions:

1. **Top-10 is doing per-task program synthesis compiled to small ONNX**, not
   neural inference. Cost penalty makes neural models structurally
   uncompetitive (a 10K-node graph caps at score ~15.8).
2. **NeuroGolf is NOT ARC-AGI**: task examples are visible at submit time, so
   you pre-solve each task offline and bake into ONNX. This is closer to
   icecuber 2020 (depth-4 brute-force search over 142 unary funcs) and Hodel
   ARC-DSL (~160 primitives + ~400 hand-written solvers) than to neural
   approaches.
3. **The right substrate**: DSL primitive library + per-task program synthesis.
   This is the only path with steep slope toward 7000+. See section 5.

### What's already done on the DSL line

- **DSL v0** (Stage 1, complete):
  - 8 primitives in `tools/dsl/primitives.py`: identity, transpose, swap_colors,
    replace_color, flip_h, flip_v, rotate180, rotate90
  - Compiler in `tools/dsl/compiler.py`
  - Unit tests 13/13 pass (`tools/dsl/test_primitives.py`)
  - Phase-4 results: 10 sample tasks all pass full-example audit. 2 win, 5 tie,
    3 lose vs v4_plus3 baseline. Median cost ratio = 1.0.
  - Single-primitive corpus coverage: ~2% (50 random tasks → 0 hit)
  - **Verdict**: shape-preserving primitives match elite handbuild. Rotate
    family loses 50–100× because of 36KB FLOAT intermediate tensor; Slice+Pad
    rewrite produces dynamic-shape output which Kaggle scorer marks
    `unmeasurable`.

- **DSL v1** (Stage 2, COMPLETE, see `reports/dsl_v1_summary.md`):
  - 6 new primitives all unit-tested and cost-measurable: `rotate180_static`,
    `rotate90_static`, `count_color`, `dominant_color`, `bbox_crop`, `fill_bg`
  - 30/30 unit tests pass; 18/18 cost-measurable (no `unmeasurable` traps)
  - `rotate*_static` is 34× cheaper than v0 rotate but still 3× more expensive
    than handbuilt baseline → only useful inside depth-2 compositions
  - `dominant_color` cost 8236 (high), `bbox_crop` cost 82K (very high) —
    optimization targets for v2
  - Depth-2 brute-force search harness `tools/dsl/search.py`, 8.2s for full 400 sweep
  - **Result**: 11 depth-1 hits across 400 tasks (2.8%), **0 depth-2 hits**.
    Bottleneck is primitive coverage, NOT search depth.
  - 7 swap candidates accumulated (2 wins task150/155 = +0.4 LB total, 5 ties)
  - **Verdict**: not enough to anchor-refresh. 7 << 30 threshold. The swaps
    are saved for bundling with the next batch (Tier-2).

---

## 3. File map (most important paths)

### Ledgers (canonical truth — read first)

| file | role |
|---|---|
| `reports/HANDOFF_NEXT_AGENT.md` | this document |
| `reports/anchor_integration_ledger.md` | every anchor decision, append-only |
| `reports/road_to_top10.md` | top-10 strategy + DSL roadmap |
| `reports/solver_ledger.md` | self-research solver status |
| `reports/self_research_assets.md` | engineering / research / cold-stored asset map |
| `reports/research.md` | running research notes |

### Anchor zips (LB-verified or candidate)

| zip | role | sha256 (head) |
|---|---|---|
| `submissions/submission_candidate_nadeem6252_plus_ours_v4.zip` | v4 LB 6245.41 | `3c940a11…` |
| `submissions/submission_candidate_v4_plus.zip` | v4_plus LB 6246.16 | `40b8f8ce…` |
| `submissions/submission_candidate_v4_plus3.zip` | v4_plus3 LB 6251.54 | `9d33bf1a…` |
| `submissions/submission_candidate_v4_plus4.zip` | v4_plus4 LB 6253.25 | `a9e6e96c…` |
| `submissions/submission_candidate_v4_plus5.zip` | v4_plus5 LB 6253.64 | `8dcf0f29…` |
| `submissions/submission_candidate_v4_plus7_task149.zip` | v4_plus7_task149 LB 6255.96 | `5cbf409f…` |
| `submissions/submission_candidate_v4_plus8_task149_bool.zip` | v4_plus8 LB 6257.04 | `246fd8f0…` |
| `submissions/submission_candidate_v4_plus9_compiler.zip` | v4_plus9 LB 6258.14 | `a635b589…` |
| `submissions/submission_candidate_v4_plus10_compiler2.zip` | v4_plus10 LB 6260.16 | `31895cd…` |
| `submissions/submission_candidate_v4_plus11_compiler3.zip` | v4_plus11 LB 6260.17 | `57f2f90b…` |
| `submissions/submission_candidate_v4_plus12_task285.zip` | v4_plus12 LB 6260.36 | `a12ba88d…` |
| `submissions/submission_candidate_v4_plus13_task366.zip` | v4_plus13 LB 6260.48 | `e986cb00…` |
| `submissions/submission_candidate_v4_plus14_bio_micro.zip` | **v4_plus14 LB 6260.74 (current)** | `fa068ef4…` |

### Anchor source dirs (DO NOT MODIFY — these are LB-verified)

- `submissions/current_best_6122_onnx/` (6122 anchor)
- `submissions/candidate_nadeem6252_plus_ours_v4_onnx/` (v4)
- `submissions/candidate_v4_plus_onnx/` (v4_plus)
- `submissions/candidate_v4_plus3_onnx/` (v4_plus3)
- `submissions/candidate_v4_plus4_onnx/` (v4_plus4)
- `submissions/candidate_v4_plus5_onnx/` (v4_plus5)
- `submissions/candidate_v4_plus7_task149_onnx/`
- `submissions/candidate_v4_plus8_task149_bool_onnx/`
- `submissions/candidate_v4_plus9_compiler_onnx/`
- `submissions/candidate_v4_plus10_compiler2_onnx/`
- `submissions/candidate_v4_plus11_compiler3_onnx/`
- `submissions/candidate_v4_plus12_task285_onnx/`
- `submissions/candidate_v4_plus13_task366_onnx/`
- `submissions/candidate_v4_plus14_bio_micro_onnx/` (**current anchor**)
- `submissions/candidate_v4_plus6_onnx/` (rejected; hidden-unsafe
  `blend_v360` overlay, do not use as anchor)

### DSL workspace

- `tools/dsl/primitives.py` — primitive library (each builds python ref + ONNX)
- `tools/dsl/compiler.py` — DslProgram → ONNX
- `tools/dsl/search.py` — brute-force program search (v1)
- `tools/dsl/test_primitives.py` — unit tests
- `tools/dsl/run_phase4.py` — v0 phase-4 batch runner
- `submissions/handbuilds/dsl_phase4/` — DSL v0 ONNX outputs (10 tasks)
- `submissions/handbuilds/dsl_v1/` — DSL v1 outputs (if completed)
- `reports/dsl_phase{1,4,5}_*.md` — v0 reports
- `reports/dsl_v1_*` — v1 reports (if completed)

### Audit / scoring tools

- `tools/score_bundle.py` — full directory scoring, n=3 sampler
  ```
  .venv/bin/python tools/score_bundle.py --onnx-dir <dir> --n-runs 3 --out <csv>
  ```
- `tools/isolated_task_eval.py` — full-example isolated audit per task
  ```
  .venv/bin/python tools/isolated_task_eval.py --onnx <path> --task-id <int>
  ```
- `tools/audit_v4_remaining.py` — full 400-task isolated audit pattern
- `tools/audit_graph_structure.py` — structural risk scoring (DEPRECATED as
  hidden-safety proxy after H2; still useful as sanity check)
- `tools/build_provenance.py` — task source attribution

### Public source pool (provenance / cherry-pick reference only — exhausted)

- `submissions/nadeem_6252_onnx/` — current dominant source
- `submissions/biohack44_6113_onnx/`, `biohack44_super_onnx/`, etc.
- `submissions/afr1ste_5689_onnx/`, `afr1ste_6335_onnx/`
- `submissions/octv_6154_onnx/`, `submissions/octavi_6042_onnx/`
- `submissions/haoranran_6100_onnx/`
- `submissions/massimiliano_eda111_onnx/`
- Manifests in `reports/source_manifests/*.csv`
- Master attribution: `reports/task_provenance.csv`

---

## 4. How to score, audit, submit

### Score a candidate locally

```bash
.venv/bin/python tools/score_bundle.py \
  --onnx-dir submissions/<candidate_dir> \
  --n-runs 3 \
  --out reports/<candidate>_score_n3.csv

.venv/bin/python -c "
import csv
print(sum(float(r['score']) for r in csv.DictReader(open('reports/<candidate>_score_n3.csv'))))
"
```

### Full-example isolated audit one task

```bash
.venv/bin/python tools/isolated_task_eval.py \
  --onnx submissions/<dir>/task255.onnx --task-id 255
# returns JSON: splits.train/test/arc-gen pass counts
```

### Full 400-task audit (slow ~10 min)

Use `tools/audit_v4_remaining.py` as a template; modify the input dir.

### Submit to Kaggle

**CRITICAL**: Kaggle CLI rejects zips not literally named `submission.zip`.
Always copy first:

```bash
cp submissions/<candidate_zip>.zip /tmp/submission.zip
.venv/bin/python -m kaggle competitions submit -c neurogolf-2026 \
  -f /tmp/submission.zip -m "<candidate description>"
sleep 30
.venv/bin/python -m kaggle competitions submissions neurogolf-2026 | head
```

### Build a candidate from scratch

Pattern (used by every v4_plus* version):

1. Copy current anchor dir to new `submissions/candidate_<name>_onnx/`
2. Overlay swapped task files (from a public source dir or DSL build)
3. Run `score_bundle.py`
4. Run full 400 isolated audit; revert any task whose visible pass count
   regresses vs current anchor; loop to stable
5. Zip: `cd submissions/candidate_<name>_onnx && zip -q ../submission_candidate_<name>.zip *.onnx`
6. Compute sha256, append to `reports/anchor_integration_ledger.md`

---

## 5. Decision tree: what to do next

Current next move after v13:

- Build a `task025` successor compiler. Preserve the anchor's line/stray
  relocation semantics, reduce the current 4-slot full-grid mask lifetime,
  target sub-90K cost, and gate on `266/266` visible plus `62/62` generated
  contracts from `tools/audit_task025_contracts.py`.
- Do not raw-swap `konbu17_v36/task025.onnx`: it is visible-clean and cheaper
  but fails the generated task025 contracts `0/62`.
- Treat latest public ecosystem results as source lake and idea leads. The
  only direct safe-positive `biohack44_superior` rows sum to about `+0.4068`,
  so they are filler after a larger clean candidate, not a standalone release.

This section below is historical. As of the 2026-06-06 reconciliation in
Section 0, DSL v1's two real wins (`task150`, `task155`) have already been
integrated into `v4_plus5`, while later DSL v2/v3/v4 expansion remained too
thin for another anchor refresh. The active path is now **Branch C / hybrid
self-research** on high-cost tasks, with `blend_v360` explicitly rejected as a
batch source after the `v4_plus6` LB failure.

Original decision tree:

### Branch A: DSL v1 has 30+ accepted swaps  [NOT ACTIVE]

If `reports/dsl_v1_swap_candidates.csv` shows ≥ 30 tasks where DSL ONNX is
strictly better than v4_plus4 in cost AND audit-clean:

1. Build `submissions/candidate_v4_plus5_onnx/` by overlaying DSL ONNX onto v4_plus4
2. Score + audit (see section 4)
3. Submit if local lift > 1.0
4. After LB result, validate H4 (DSL hidden-safety): LB delta vs local delta
5. Continue to Stage 3 (Hodel primitive port) — see road_to_top10.md §D

### Branch B (active): coverage too thin, invest in Tier-2 primitives

DSL is viable for shape-preserving tasks but covers only 2.8% of the corpus
with the current 14-primitive library. The Phase-1 task pattern analysis +
DSL v1 search miss-list points to the following Tier-2 primitives as
highest-leverage. Tackle in this order:

1. **`connected_components(connectivity=4)`** — likely highest-impact missing
   primitive; opens the path to object-oriented tasks (task233 / task255 /
   task366 monster family). Implement via Conv-based label propagation
   (avoid GatherND — likely cost or unmeasurable trap).
2. **`largest_component()` / `nth_component(n, sort_by='area')`** — natural
   composition with `connected_components`.
3. **`tile_h(n)` / `tile_v(n)` / `scale_int(k)`** — for tile-pattern tasks
   (task001-class).
4. **`apply_mask(grid, mask, primitive)`** — per-region application; opens
   compositional power.
5. **`dominant_color` BOOL-pipeline rewrite** — current cost 8236 prevents
   task129 entry; target ≤2K so task129 becomes a swap candidate.
6. **`bbox_crop` cost reduction** — current 82K prevents most uses; target
   ≤20K so it becomes a usable composition unit.

Goals for the next worker:

- Implement at least 4 primitives from the list (1, 2, 3, plus 5 OR 6)
- All must pass: unit test (python_ref ≡ onnxruntime), cost-measurable, no
  fp16 / no large lookup table / no dynamic-shape intermediate
- Re-run `tools/dsl/search.py` at depth ≤ 2 over full 400; output
  `reports/dsl_v2_search_results.csv` and `reports/dsl_v2_swap_candidates.csv`
- Target: ≥ 25 accepted swaps total (= 7 from v1 + 18+ new from v2)
- If reached, build v4_plus5 (overlay DSL ONNX onto v4_plus4), run audit,
  submit if local lift > 1.0

If even after Tier-2 the hit rate stays below 5% (i.e. no obvious primitive
can scale fast enough), shift to Branch C (LLM-assisted).

In parallel, consider Branch D quick wins (task255 LLM rebuild + biohack
ingest retry) — they don't conflict with Branch B and could unlock 5–15 LB
faster than Tier-2 primitives.

### Branch C: DSL hit-rate stays ≤ 5% after Tier-2

DSL alone cannot scale fast enough. Pivot to:

1. **LLM-assisted per-task hand-build**: write a triage tool that takes a task,
   formats train+test examples, sends to GPT/Claude/Gemini, asks for a Python
   rule, runs it on visible examples to validate, then asks for ONNX builder
   code. Engineer iteration loop with strict audit gate.
2. Target the 30 cost-monster tasks first (top of `reports/candidate_v4_plus4_score_n3.csv` by cost descending — task255/233/366/191/018/etc)
3. Each successful LLM-built task is +1 to +4 LB

### Branch D: Hybrid (most likely correct path)

Do A or B based on DSL v1 metrics, AND in parallel start Branch C (LLM-assisted
hand-build) targeting the 5 biggest cost monsters: task255 (1.59M), task233
(1.46M), task366 (0.83M), task191 (709K, currently locked at ours version due
to AUDIT_REGRESSION on Nadeem version), task018 (476K, locked due to H1).

If even ONE monster gets cracked open with a 5× cost reduction, that's
+1 to +3 LB from a single task.

### Always-true constraints

- Submission slots are precious. Only submit when local lift > 1.0 and audit
  is clean.
- The local scorer has been proven byte-faithful to LB through 4 consecutive
  submissions. Trust it within 0.01.
- Never roll back the LB-verified anchor.

---

## 6. Hard rules (violation = task failure)

1. **Do not modify any LB-verified ONNX file**. The dirs in §3 "Anchor source
   dirs" are read-only. Always make a new candidate dir.
2. **Do not re-enable any frozen task ONNX**:
   - **task319**: hand-built ONNX path is FROZEN (hidden-safety audit failed,
     fallbacks were arc-gen-specific). See `reports/solver_ledger.md`.
   - **task191**: Nadeem version has visible silent regression
     (227/267 vs ours 267/267). Locked at ours version.
   - **task264**: Nadeem version is 0/265 visible fail. Locked at ours.
3. **Do not touch git remotes**. This is a Kaggle-only workflow.
4. **No fp16 surgery without per-task allowlist + same_decoded_outputs gate**.
   `task303` is a known unsafe re-add.
5. **No large initializer lookup tables in DSL primitives**. Primitives must
   be small algorithms, not memorized answers.
6. **No dynamic-shape intermediate tensors** in DSL primitive ONNX. The Kaggle
   scorer marks them `unmeasurable` (this is a hard scorer constraint, found
   in DSL v0 rotate Slice+Pad attempt).
7. **Every new ONNX must pass `same_decoded_outputs` or full-example
   isolated audit before joining a candidate**. No exceptions.
8. **Every decision must append to `reports/anchor_integration_ledger.md`**
   with timestamp (CST). Never rewrite history.
9. **Submission zips must be copied to `/tmp/submission.zip` before
   `kaggle competitions submit`**. Direct submission of files with other names
   returns 400.
10. **Do not aggressively decompose into many sibling subagents**. Default
    to one coherent worker per request unless workstreams are obviously
    independent (e.g. external research vs internal engineering).

---

## 7. Open questions (highest uncertainty)

1. **Are top-10 actually using DSL or just LLM-per-task?** Cost-monster shape
   in our v4_plus4 is monolithic-no-shared-primitives, consistent with public
   sources being LLM-per-task or pre-DSL. But that doesn't tell us what
   top-10 has — no submissions above LB 6155 are public.
2. **What fraction of tasks have empty/tiny hidden test sets (H1 class)?**
   Confirmed only on task018 so far. If 5–20% of tasks are H1-class, our
   visible-pass-parity audit is unnecessarily conservative on them — there's
   a hidden source of upside in cost-only swaps that look bad locally.
3. **Engineering bandwidth**: 13 hand-built solvers across 400 tasks. Top-10
   likely has 100+. LLM acceleration is essential, not optional.

---

## 8. Quick win opportunities (if you want fast LB)

1. **DSL v1 task150 + task155 swap (validated)** — +0.4 LB, 2 wins. Bundle
   with whatever batch goes to v4_plus5. Already in `submissions/handbuilds/dsl_v1/`.
2. **biohack44_superior `notebook715db8358e` ingest** — Kaggle CDN failed
   previously (SSL EOF). Try `kaggle kernels output -k biohack44/notebook715db8358e -p /tmp/bioht/`
   or `kaggle datasets download -d biohack44/<dataset>`. If successful, run
   the v4_plus2-style mining over it; estimated +0.5 to +5 LB.
3. **task204 cheap-cost rebuild** — Python prototype is 268/268; v0 ONNX cost
   398K vs v4_plus4 task204 cost 119K. Rewrite using DSL v1 primitives or new
   primitives if needed; could be a single-task +2 LB win.
4. **task129 with cheaper `dominant_color`** — DSL v1 found task129 is
   correctly solved by `dominant_color` (visible all-pass), but cost 8236 vs
   baseline 818 makes it lose. If `dominant_color` is rewritten to ≤2K cost,
   task129 becomes a swap candidate (+0.5 LB single-task).
