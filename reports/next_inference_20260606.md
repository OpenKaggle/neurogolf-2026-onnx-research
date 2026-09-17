# Next Inference - 2026-06-06

Current LB-verified anchor: `v4_plus16_seddik_t001`, Kaggle ref `53421634`,
public LB `6260.89`, local n=3 `6260.896515`.

## State Check

- The workspace is not a Git repository. There is no project git remote to
  fetch; local reports plus Kaggle submissions are the source of truth.
- Kaggle remote confirms the latest submitted anchor sequence:
  `v4_plus8_task149_bool` `6257.04` ->
  `v4_plus9_compiler` `6258.14` ->
  `v4_plus10_compiler2` `6260.16` ->
  `v4_plus11_compiler3` `6260.17` ->
  `v4_plus12_task285` `6260.36` ->
  `v4_plus13_task366` `6260.48` ->
  `v4_plus14_bio_micro` `6260.74` ->
  `v4_plus15_bio_micro2` `6260.89` ->
  `v4_plus16_seddik_t001` `6260.89`.
- The current anchor zip exists and matches the recorded SHA256:
  `e7ac37d6240fca04b603db46bfa52475b15be024f626d60f71b07988cab18db7`.
- The earliest pasted handoff is stale: it still names `v4_plus4` as current.
  The active canonical docs have correctly moved to `v4_plus14`.

## What Is Now Proven

Four backend rules are hidden-LB-proven and should be first-class compiler
passes:

1. Keep BOOL masks BOOL until numeric ops require widening.
2. Scalarize broadcast-safe uniform initializers.
3. Remove terminal `Cast -> output` when the producer dtype decodes identically.
4. Keep ARC color-valued grids in narrow integer dtypes when all consumers
   accept them and decoded outputs are exact.

Public package mining is still useful for ideas, but raw bundle blending is not
the main path. `v4_plus6` proved that visible-clean public swaps can be
hidden-catastrophic.

## New Completed Assets

- `task285` compact ONNX v2 has staged and is now LB-verified inside v12:
  cost `395468 -> 326044`, score `12.112175 -> 12.305212`, visible
  `265/265`, pseudo-hidden `3375` passed and `0` failed.
- `task366` compact ONNX v3 has staged and is now LB-verified inside v13:
  cost `830720 -> 734516`, score `11.369952 -> 11.493033`,
  semantic visible/arc-gen `266/266`, ONNX encodable target equivalence
  `255/255`, ONNX encodable semantic equivalence `255/255`, pseudo-hidden
  `1456 passed / 0 failed / 406 skipped`, checker and strict shape inference
  passed. It combines the max-7 rectangle bound with density-based source
  panel selection.
- `task191` semantic rule is closed, but the only exact cost win so far is the
  terminal-cast narrowing already absorbed by v10. The compact-v3 audit found
  the missing guarded-convolution matcher; public compact graphs remain
  non-exact at `227/267`.
- `v4_plus14_bio_micro` has staged and is now LB-verified: four
  `biohack44_superior` micro swaps (`task145/153/248/357`) on top of v13.
  Changed-task full visible counts: `267/267`, `265/265`, `13/13`, `13/13`.
  Full bundle local n=3 `6260.747441`; public LB `6260.74`; local delta
  vs v13 `+0.259505`.
- `v4_plus15_bio_micro2` has staged and is now LB-verified: 22 additional
  `biohack44_superior` micro swaps on top of v14. Full bundle local n=3
  `6260.890204`; public LB `6260.89`; local delta vs v14 `+0.142763`.
- `v4_plus16_seddik_t001` has staged and is now LB-verified: one Seddik
  precision-source `task001` exact micro on top of v15. Full bundle local n=3
  `6260.896515`; public LB `6260.89`; local delta vs v15 `+0.006311`.
  The LB display ties v15 because Kaggle shows two decimals.
- 2026-06-07 follow-up:
  - Remote recheck still shows `53421634` complete at `6260.89`.
  - `compiler_backend_sweep_v4_on_v16` found no fresh exact backend rewrite.
  - `task025_priority.onnx` closes the generated-contract semantics
    (`62/62`) by resolving output-channel conflicts with high-color priority,
    but is far too expensive to stage (`541975` cost vs anchor `140093`).
- `task191_guarded_v4` is reproducible but non-stageable:
  `211/267`, cost `143028`. The public compact reference is `227/267`, cost
  `89072`; the current exact anchor remains `267/267`, cost `691342`.
- `task025` contract audit now exists:
  `tools/audit_task025_contracts.py`,
  `reports/task025_contract_audit_20260606.{md,json}`. The anchor passes
  `266/266` visible and `62/62` generated contracts at cost `140093`.
  The tempting public `konbu17_v36` file passes visible and costs `89286`, but
  fails generated contracts `0/62`, so it must not be swapped.
- Latest public ecosystem audit:
  `reports/public_ecosystem_audit_latest_20260606.md`. No new public package
  changes the anchor. `biohack44_superior` is now locally complete and scores
  `6208.07`; its safe-positive direct rows total only about `+0.4068` local
  and should be batch filler, not a main release. New idea-only/source-lake
  kernels include Seddik's precision reduction and Scott Weeden's DFA solver;
  Scott's bundle scored `5862.10` locally with multiple unmeasurable tasks.
- Task dossier infrastructure now exists:
  `reports/task_dossiers_20260606.{csv,json,md}`.
- External research synthesis now maps ARC/Hodel/icecuber/DreamCoder/LLM work
  into concrete repo actions:
  `reports/external_research_synthesis_20260606.md`.
- Transparent triage queue now exists:
  `reports/task_triage_ml_20260606.md` and
  `reports/task_triage_ml_top50_20260606.csv`.
- Compiler backend sweep v3 staged as v11: exact bool-source Cast-chain
  collapse on `task157` plus static `ReduceSum` axis fusion on `task295`
  for `+0.009159` local and LB `6260.17`.
- DSL frontier board now exists:
  `reports/dsl_frontier_20260606.{csv,json,md}` generated by
  `tools/dsl/build_frontier_report.py`. It confirms the current DSL has no
  fresh v10 submission branch; deeper search over the existing full-FLOAT
  primitive set is a local maximum.

## Current Ranking

Best near-term engineering targets from the combined dossier/triage view:

1. `task025`: build a successor semantic compiler that preserves the anchor's
   line/stray behavior but reduces the current 4-slot full-grid mask lifetime.
   Target sub-90K cost; require `266/266` visible and `62/62` generated
   contracts before release-lane consideration.
2. `task191`: needs a real boundary-aware D4 stamp compiler. Do not redo the
   already-absorbed terminal-cast branch; guarded-v4 is non-stageable unless
   it reaches exact `267/267`.
3. `task366`: v3 is LB-verified. Stop minor graph nips here; the next gain
   requires replacing the remaining full-grid shift-search tensors, not
   another selector tweak.
4. `task285`: v2 staged, but public-tail signal still implies a much tighter
   semantic graph is possible; revisit only with a new object/mask encoding,
   not more small rewrites on the current anchor graph.
5. `task158`, `task173`, `task286`, `task054`: remaining semantic-factory
   first-pass queue from public-tail plus cost ranking.
6. `task233`: huge upside, but still python-open with alignment gaps. Continue
   only with train-grounded component/hole rules and pseudo-hidden contracts.
7. `task255`: highest gross upside, but current v2 is falsified by clipped
   translation pseudo-hidden tests. Fix semantics before ONNX.

## Strategic Inference

Do not explore only one direction. The project now needs four parallel lanes:

- Release owner lane: protect `v4_plus14`; build a candidate zip only after
  local full score, exact-equivalence or semantic gates, and a clear expected
  gain.
- Compiler lane: exact-equivalence passes only. Use public kernels as rewrite
  inspiration, not as candidates.
- Semantic compiler lane: convert closed Python semantics into compact ONNX for
  `task366`, `task191`, and the post-v2 `task285` typed-object rewrite.
- Research/DSL lane: build typed object/mask primitives, a DSL frontier board,
  pseudo-hidden contract templates, and eventually an LLM Python-solver sampler.

Machine learning is useful now as triage/proposal/ranking, not as submitted
heavy inference graphs. The scoring formula makes large neural ONNX models
structurally weak unless they compile down into small symbolic programs.

## Active Delegation

Completed subagent results absorbed after this report:

- Hegel: `task285` compact ONNX v2. Staged in `v4_plus12_task285`.
- task366 compact v3 worker: staged in `v4_plus13_task366`.
- Hume: `task191` compact-v3 audit. No stageable ONNX; next target is guarded
  hole/fill convolution.
- Harvey: compiler backend sweep v3. Staged in `v4_plus11_compiler3`.
- Kierkegaard/Socrates/Aristotle: ML triage, dossiers, and external research
  synthesis reports are available.

Active delegated lanes opened after v12:

- `019e9c90-54f6-7b53-9fee-a01270fe3e93`: `task366` compact
  bounded-rectangle compiler. Completed; v3 is LB-verified as v13.
- `019e9c90-5605-7363-907c-e58905ed5e74`: `task191` guarded D4 stamp matcher.
  Completed. `task191_guarded_v4` is non-stageable (`211/267`, cost `143028`).
- `019e9c95-f4f2-7b70-a2b3-6ea20181692e`: latest public
  ecosystem/discussion audit. Completed; no direct big package, source-lake
  only.
- `019e9c96-162b-79a2-8d2a-2db5a31995d5`: semantic-factory next-pick audit
  for `task158/173/286/054/025`. Completed; chose `task025`.
- `019e9ca0-ded0-74e3-b826-5da1f5723181`: task025 contract audit. Completed;
  no stageable candidate, but produced a release gate.

Next subagents to open after capacity frees:

- Public-tail semantic-factory pass for `task158/173/286/054/025`.

Do not open another plain DSL-depth-search worker until typed mask/object
primitives exist. The immediate DSL work is a small typed primitive skeleton:
`mask_color`, `paint_mask`, `cover_mask`, `where_grid`, `bbox(mask)`,
static shifts/crops, hole masks, and task-family templates for `task285` and
`task366`.

## Submission Policy

`v4_plus16_seddik_t001` is the current LB-verified anchor. The next
submission-worthy branch is likely one of:

- `task191` true guarded D4 stamp matcher with exact `267/267` and material
  gain over cost `691342`.
- `task025` semantic compact candidate, but only after line/stray-relocation
  pseudo-hidden contracts pass and it materially beats current cost `140093`.
- A later semantic-factory batch from `task158/173/286/054`, but only if each
  task has task-specific hidden-safety contracts or decoded-equivalence gates.
- A post-v2 `task285` rewrite only if it replaces the graph architecture and
  moves materially toward the public-tail cost frontier.
