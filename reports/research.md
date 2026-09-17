# NeuroGolf 2026 Research Notes

## 2026-06-07 Remote Check and Minimal Loop

- Remote check:
  - Latest Kaggle submission remains `v4_plus16_seddik_t001`, ref `53421634`.
  - Status `COMPLETE`, public LB `6260.89`.
  - Local n=3 `6260.896515`; zip sha256
    `e7ac37d6240fca04b603db46bfa52475b15be024f626d60f71b07988cab18db7`.
  - Public top-10 cutoff in the checked leaderboard output is still `7339.93`.
- Exact backend sweep on the current v16 anchor:
  - Report: `reports/compiler_backend_sweep_v4_on_v16.md`.
  - Active rules: exact bool-source Cast-chain collapse, static keepdims=1
    ReduceSum-axis fusion, identity Transpose cleanup.
  - Result: 1 candidate evaluated, 0 gate-passed/applied; final n=3 remains
    `6260.896515`.
  - Inference: the previously proven compiler lane is locally saturated on
    v16. Do not spend the next loop rerunning the same pass family without a
    new rule.
- `task025` minimal semantic closure:
  - Builder update: `tools/build_task025.py` now has an optional priority
    tail implemented in the produced `submissions/handbuilds/task025_priority.onnx`.
  - New audit: `reports/task025_contract_audit_20260607.md`.
  - Result: `priority_semantic_v2` passes visible `266/266` and generated
    contracts `62/62`; the older semantic probe failed exactly two generated
    cases (`h_00_4c`, `v_00_4c`).
  - Cause isolated: output-channel conflicts when a moved lower-color stray
    lands on a higher-color complete line. A high-color priority tail fixes
    semantics.
  - Release decision: **do not stage**. Cost is `541975`, score `11.797025`,
    worse than the current anchor task025 cost `140093`, score `13.149938`.
  - Next `task025` move is compression, not rule hunting: preserve the
    high-color priority semantics while replacing the broad full-grid
    relocation head.

## 2026-06-06 v4_plus16 Seddik Task001 Micro Anchor

- New LB-verified anchor:
  - `v4_plus16_seddik_t001`, submission `53421634`
  - local n=3 `6260.896515`, public LB `6260.89`
  - zip sha256
    `e7ac37d6240fca04b603db46bfa52475b15be024f626d60f71b07988cab18db7`
- Composition: `v4_plus15_bio_micro2` plus one Seddik precision-source exact
  micro replacement on `task001`.
- Validation: changed task strict checker/shape-inference `1/1`, full bundle
  n=3 zero errors, local `task001` cost `2861 -> 2843`, nodes `14 -> 13`,
  score `17.041074 -> 17.047385`.
- Strategic note: this closes the tiny precision-source micro release. It is
  useful proof that Seddik-style surgery can be mined safely in isolated
  exact-equivalence form, but the next material gains must come from semantic
  compiler work, not more one-node nips.

## 2026-06-06 v4_plus15 Bio Superior Micro2 Anchor

- New LB-verified anchor:
  - `v4_plus15_bio_micro2`, submission `53421427`
  - local n=3 `6260.890204`, public LB `6260.89`
  - zip sha256
    `ef11b3aeec2552fb802fc7d107c1a91345b5c6179c684532f540c0187c933bac`
- Composition: `v4_plus14_bio_micro` plus 22 additional full-visible-clean
  `biohack44_superior` micro swaps:
  `task003/034/059/089/129/175/208/212/216/224/239/247/263/265/273/275/281/289/348/360/383/397`.
- Validation: changed tasks strict checker/shape-inference `22/22`, full
  bundle n=3 zero errors, local delta vs v14 `+0.142763`.
- Strategic note: this validates the remaining clean `biohack44_superior`
  micro lane, but also saturates it. Public-package mining should now be
  treated as idea/source-lake work unless a new source supplies a task-level
  semantic clue.

## 2026-06-06 v4_plus14 Bio Superior Micro Anchor

- New LB-verified anchor:
  - `v4_plus14_bio_micro`, submission `53421167`
  - local n=3 `6260.747441`, public LB `6260.74`
  - zip sha256
    `fa068ef49bbcdea4e26186148d175417e3a2100a635787e9cc11e990e7200d82`
- Composition: `v4_plus13_task366` plus four full-visible-clean
  `biohack44_superior` micro swaps: `task145`, `task153`, `task248`,
  `task357`.
- Validation: changed tasks strict checker/shape-inference `4/4`, changed-task
  full visible counts `267/267`, `265/265`, `13/13`, `13/13`, full bundle
  n=3 zero errors.
- Strategic note: this completes the public-micro filler lane. Remaining
  `biohack44_superior` positives are tiny; do not let them displace the next
  real semantic compiler target (`task025` successor or exact `task191`).

## 2026-06-06 v4_plus13 Task366 Anchor

- Submitted candidate:
  - `v4_plus13_task366`, submission ref `53420666`
  - public LB `6260.48`; local n=3 `6260.487936`, delta vs v12 `+0.123081`
  - zip sha256
    `e986cb000846c19e1c6d2caa5d801a6fd98e46e51cf06c9acbb5b214c64aa9b2`
- Composition:
  - base `submissions/candidate_v4_plus12_task285_onnx`
  - one overlay: `submissions/handbuilds/task366_compact_v3.onnx`
  - candidate builder: `tools/build_candidate_v4_plus13_task366.py`
- task366 compact-v3 result:
  - builder/report: `tools/build_task366_compact_v3.py`,
    `reports/task366_compact_v3.md`
  - semantic visible/arc-gen `266/266`
  - ONNX encodable target equivalence `255/255`
  - ONNX encodable semantic equivalence `255/255`
  - pseudo-hidden contracts `1456 passed / 0 failed / 406 skipped`
  - checker full_check and strict shape inference passed
  - cost `830720 -> 734516`, score `11.369952 -> 11.493033`
- Research inference:
  - The useful move was not another bounded-pass micro-nip. It came from
    replacing the anchor's label-histogram source selector with the closed
    semantic fact: choose the denser non-background panel.
  - The remaining task366 cost is still dominated by full-grid object and
    shift-search tensors. Further task366 work should require a new object-mask/shift-search
    architecture; otherwise move on.
  - Next best engineering lanes are `task191` only if it reaches exact
    `267/267`, or the semantic-factory queue `task158/173/286/054/025`.

## 2026-06-06 v4_plus12 Task285 Anchor and Compact-Lane Split

- New LB-verified anchor:
  - `v4_plus12_task285`, submission `53418978`
  - local n=3 `6260.364855`, public LB `6260.36`
  - zip sha256
    `a12ba88d37ac8a2bd6371c2b2fc0267ac9a47e67eb3a97af4da56f91f2ec2c82`
- Composition relative to `v4_plus10_compiler2`:
  - `v4_plus11_compiler3`, submission `53418833`, local `6260.171818`,
    public LB `6260.17`: exact Cast-chain collapse on `task157` and static
    `ReduceSum` axis fusion on `task295`.
  - `v4_plus12_task285`: exact compact rewrite of `task285` from
    `tools/build_task285_v2.py`.
- task285 result:
  - visible/arc-gen `265/265`
  - decoded equivalence vs anchor `265/265`
  - pseudo-hidden ONNX contracts `3375 passed / 0 failed / 70 skipped`
  - cost `395468 -> 326044`
  - memory `390848 -> 319624`
  - score `12.112175 -> 12.305212`, local delta `+0.193038`
- Compiler inference:
  - ARC color-valued tensors can be kept as `UINT8` when every consumer op is
    valid for that dtype and decoded outputs are exact.
  - Zero-based marker labels can remove repeated full-grid subtracts before
    `Gather`.
  - Precomputing selected reflection origins can be cheaper than row/column
    decomposition.
- Strategic consequence:
  - task285 v2 is a successful release, but it is still a rewrite of the
    current large marker-propagation graph. The public-tail target implies a
    much lower cost is possible only with a new object/mask graph architecture.
  - Do not spend the next cycle on small task285 graph nips unless they are
    exact and nearly free to validate. The higher-value next lanes are
    `task366` bounded-rectangle compiler, `task191` guarded D4 stamp matcher,
    and semantic-factory tasks `158/173/286/054/025`.

### task191 compact-v3 audit

- `tools/build_task191_compact_v3.py` and
  `reports/task191_compact_v3.md` document why the Massim-family compact graph
  cannot stage:
  - current exact anchor: `267/267`, cost `691342`
  - public compact: `227/267`, cost `89072`
  - holes-only probe: `200/267`, cost `89072`
  - BOOL-final probe: `227/267`, cost `90872`
  - zero-dependency probe: `227/267`, cost `107077`
- Main finding: the public compact graph uses `v286 = hole_mask - fill_mask`,
  so visible color-4 anchors under candidate fill cells cancel valid 3-hole
  matches. All 40 failures are large 3-hole stamps (`5x4` or `5x5`).
- Next architecture target: separate guarded convolutions for `hole_hits`,
  `bad_fill`, and `missing_fill`, with boundary handling that requires holes
  to be fully visible while allowing fill cells to clip. A stageable compact
  target is exact `267/267` below roughly `150k` cost.

## 2026-06-06 v4_plus10 Compiler Anchor and Public-Tail Calibration

- New LB-verified anchor:
  - `v4_plus10_compiler2`, submission `53417133`
  - local n=3 `6260.162659`, public LB `6260.16`
  - zip sha256
    `31895cd79b62e1d22298746a675d522337d06eea8a14f810936543f369e8a1b5`
- Composition relative to `v4_plus8_task149_bool` through v9:
  - 32 conservative uniform-initializer-to-scalar rewrites:
    `017, 026, 036, 051, 058, 065, 076, 077, 096, 109, 110, 114, 115,
    117, 128, 134, 144, 156, 178, 206, 219, 233, 248, 255, 264, 267,
    284, 285, 296, 357, 358, 383`
  - 6 BOOL-memory rewrites: `240, 301, 308, 316, 333, 374`
  - `task149` v2 removes the explicit false Pad constant, cost `172 -> 171`
  - 10 terminal `Cast -> output` narrowing rewrites:
    `092, 138, 191, 192, 205, 209, 215, 376, 377, 396`
- Validation:
  - v9: 39 changed task hashes; 361 tasks byte-identical to the prior anchor.
  - v10: 10 additional task hashes changed relative to v9.
  - All changed models passed full ONNX checker and strict shape inference.
  - All branch candidates had full train/test/arc-gen decoded exact
    equivalence before integration.
  - v9 full bundle score had zero scorer errors and `+1.097663` local delta.
  - v10 full bundle score had zero scorer errors and `+2.018414` local delta
    over v9.
- Inference:
  - BOOL mask lifetime and broadcast-safe uniform scalarization are now
    hidden-LB-proven backend rules.
  - Terminal output dtype narrowing is also hidden-LB-proven: if the final
    producer dtype decodes identically, the explicit cast can be removed and
    the scorer does not require FLOAT output.
  - Compiler golf remains important but is not the top-3 path; expected
    remaining exact-equivalence ceiling is small compared with the `+1364`
    top-3 gap.

Note: `reports/compiler_golf_probe_v2_score_full.csv` was produced with
`--n-runs 0` and shows six local encoding errors on unrelated historical
boundary tasks. The standard comparable n=3 score is
`reports/compiler_golf_probe_v2_score_n3.csv`, which has zero errors and
matched the v10 Kaggle submission.

### Public-tail calibration

- 2026-06-06 Discussion scan found a rank-4 competitor listing 15 current
  bottom tasks and scores. On that exact set:
  - our `v4_plus9/v10` total on those exact tasks is still about `189.6`
    because v10 did not alter that public-tail set materially.
  - rank-4 listed total: `227.716`
  - rank-19 reported total on same tasks: `232.16`
  - observed gap: `+38.14` to rank-4, `+42.58` to rank-19
- Biggest directly observed public-tail gaps:
  - `task255 +4.57`, `task233 +3.98`, `task366 +3.77`, `task018 +3.52`,
    `task285 +3.02`, `task133 +2.33`, `task243 +2.16`.
- Discussion signal:
  - top competitors are doing broad task-by-task LLM/manual golfing with
    overfit/timeout/token management, not relying on one public package.
  - synthetic hidden data is useful but imperfect; pseudo-hidden contracts
    should gate candidates, not veto all creative hypotheses.
- Strategic consequence:
  - next milestone is `+100` from 25-40 pseudo-hidden-clean semantic programs.
  - top-3 requires industrialized coverage of roughly 200+ tasks with typed
    low-cost programs.

## 2026-06-06 Boolean-Memory Anchor and Big-Vision Synthesis

- New LB-verified anchor:
  - `v4_plus8_task149_bool`, submission `53415813`
  - local n=3 `6257.0466`, public LB `6257.04`
  - zip sha256
    `246fd8f0ad982faeaba029bea5131505c133b73e6be93612b2dcf2d9dab5137c`
- task149 progression:
  - previous anchor cost `509`, score `18.767552`
  - BOOL graph cost `172`, score `19.852506`
  - exact decoded equivalence on `267/267`
  - bundle gain `+1.08495`, reproduced exactly on Kaggle
- Core compiler inference: the scorer charges intermediate bytes and decoding
  uses `output > 0`. Boolean masks should remain BOOL until an operator truly
  requires numeric input. This must become a typed-IR rule in the future DSL.
- Part 4 audit:
  - whole package local `6255.1495`, below the live anchor.
  - `task149` was the only material new win before our BOOL rewrite.
  - `task001` was about `+0.0063`; `task295` `+0.0030`;
    `task153/157` did not beat the live anchor.
- Latest leaderboard snapshot around 2026-06-06 16:20 CST:
  - rank 1 `7695.78`
  - top-3 cutoff `7621.81`
  - top-10 cutoff `7339.93`
  - gap from anchor: about `1083` to top-10 and `1365` to top-3.
- Latest public ecosystem:
  - `seddiktrk/surgical-onnx-precision-parameter-reduction`: useful lossless
    passes (dead initializer pruning, exact dedup, safe uniform-to-scalar).
  - `scottweeden/neurogolf-trace-language-dfa-solvers`: bundle is not
    competitive under our scorer, but its explicit
    analyze/build/optimize/verify/cost/package state machine is useful.
  - `beicicc_6645` and `afr1ste_6284` remain boundary artifacts, not normal
    merge sources. The first future boundary experiment should be an isolated
    `golf.Identity` tiny probe.

### Next inference order

1. Compiler golf: safe BOOL-lifetime extension, then uniform-scalar.
2. Safety substrate: explicit pseudo-hidden metamorphic harness.
3. Use it on task255 first, then task233/task366; Python semantic closure
   precedes ONNX construction.
4. Only after another stable-gain batch, spend one submission on a
   scorer-boundary tiny probe.
5. Top-3 architecture bet: typed object DSL, BOOL mask IR,
   grouped/depthwise Conv templates, and an explicit verification state
   machine.

### Pseudo-hidden MVP

- Added `tools/pseudo_hidden.py`, `tools/test_pseudo_hidden.py`, and
  `reports/pseudo_hidden_design.md`.
- The harness requires explicit invariance contracts for color permutation,
  zero-border translation/padding, and distractor insertion; it also supports
  leave-one-out rule callbacks.
- Unit tests: `9/9` passed.
- Old `tools/prototype_task255.py` is now decisively retired as a hypothesis:
  it scores `0/3 train + 0/1 test + 0/261 arc-gen`, and `0/795` under the
  original-plus-two-contract pseudo-hidden audit. The next task255 prototype
  must restart from the separator/gutter geometry, not flood-fill holes.

## 2026-06-06 Research Refresh

- Local/remote reconciliation:
  - Workspace `$HOME/Desktop/kaggleonnx` is not a Git repo; no project
    remote exists to fetch. Kaggle submissions and local reports are the source
    of truth.
  - `v4_plus5` (`53355543`) was the prior verified anchor at **6253.64**.
  - `v4_plus6` (`53384773`) is rejected at **5966.68** despite local
    **6276.21** and 22/22 swapped-task visible full audits. Treat
    `blend_v360` as hidden-unsafe unless a specific per-task theory exists.

- New public package scan:
  - `massimilianoghiotto/convolution-series-part-4` packages the normal
    measurable `massim_6254` bundle. Local total **6254.6458**; 400 tasks score
    ok locally.
  - Massim top wins:
    - `task149`: isolated **267/267**, cost **509** vs anchor cost **5139**.
      Built `v4_plus7_task149`; submission `53415130` scored **6255.96**,
      exactly matching local **6255.9616**. Promoted to anchor.
    - `task191`: quick score looked +2.07, but full isolated audit only
      **227/267** (`arc-gen 222/262`); reject.
    - `task264`: full isolated audit **0/265**; reject.
  - `beicicc/neurogolf-6645-39-public-score-open-solution` / artifact:
    current local scorer sees 78 ok, 319 unmeasurable, 3 excluded-op; many
    graphs use custom `golf.Identity`. This is not directly mergeable but is a
    strong scorer/environment-boundary research signal.
  - `agentzz/neurogolf-submit-6284-v2` / `afr1ste_6284`: local scorer sees
    59 ok, 81 unmeasurable, 259 excluded-op, plus one SSA error. Same class:
    interesting boundary artifact, not a safe source.
  - `seddiktrk/surgical-onnx-precision-parameter-reduction` contributes three
    lossless ideas: prune unused initializers, deduplicate identical
    initializers, replace uniform tensors with scalars when consumers broadcast
    safely. The first two are mostly covered by earlier optimizer passes; the
    uniform-scalar pass has about 19.6k safe parameter savings on `v4_plus5`,
    concentrated in `task285/233/366/255`, likely a small fractional-LB branch.
  - `scottweeden/neurogolf-trace-language-dfa-solvers` is mostly a workflow /
    verifier framework plus simple builders and greedy bundle blending. Its
    direct solver value is low, but the trace-language idea is useful as an
    experiment gate: analyze/build/optimize/verify/cost/package states should
    be explicit before Kaggle probes.

- Discussion/forum scan:
  - Official updates confirm early exploit classes were repeatedly patched:
    dynamic-shape/zero-shape accounting, constant parameter contribution,
    negative memory, profiler/MAC quirks, and trace filename clobbering. Current
    objective is memory + params, with MACs removed.
  - A public "new exploits" thread on 2026-05-14 says a memory-cost exploit was
    fixed the same day. This supports treating old high public artifacts with
    unmeasurable/custom-domain behavior as stale unless current single-task
    probes prove otherwise.
  - Current competitors explicitly list overfit-risk tasks including
    `task018`, `task096`, `task118`, `task192`, `task219`, `task285`,
    `task319`, `task355`, `task359`. This matches our failures on task018,
    task219, task319-like arc-gen tuning, and blend_v360. Visible full-example
    pass is not a universal hidden-safety proof.
  - Leaderboard refresh on 2026-06-06: rank 1 CroDoc **7690.45**, top-3 cutoff
    **7621.81**, top-10 cutoff **7339.93**. At that snapshot the then-current
    anchor **6255.96** was about 1084 behind top-10 and 1366 behind top-3.

- Strategic inference:
  - New normal-source cherry-picks are still possible, but only as small,
    audited probes. The `massim task149` success proves we should not discard
    all new public sources wholesale.
  - Batch bundle blending is now clearly dominated by hidden-risk unless every
    constituent source/task has its own safety story.
  - The top-5/top-3 path is not "one more public package"; it is a production
    system for many small per-task ONNX programs: DSL primitives, grouped Conv
    templates, object/component primitives, and LLM-assisted builder generation
    with strict pseudo-hidden/isolated gates.

## Competition

- Slug: `neurogolf-2026`.
- Deadline shown by Kaggle CLI on 2026-05-31: 2026-07-15 23:59:00.
- Current account has entered; there were no submissions before this work.
- Input: 400 ARC-AGI tasks, each with `train`, `test`, and `arc-gen` examples.
- Model format: one ONNX per task, named `task001.onnx` through `task400.onnx`, inside `submission.zip`.

## Current Scoring Rules

- A task receives `max(1, 25 - ln(cost))` if its ONNX is functionally correct.
- `cost = total parameters + cumulative intermediate tensor memory footprint in bytes`.
- MACs no longer contribute to score.
- File size limit per ONNX: 1.44 MB.
- Static shapes are required.
- Disallowed op types currently enforced by released `neurogolf_utils.py`: `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`, `Compress`; sequence ops and subgraphs are also rejected.
- Official utility version in the competition data is dated 2026-05-14 and includes sanitization for names, duplicate value-info rejection, profiler filename prefixes, scalar parameter cost, and stronger static-shape checks.

## Public Baseline Decision

- Highest useful public single-package baseline found: Octavi Grau's `[6042.85] Per-Task Hand-Built ONNX Solvers`.
- It uses `jsrdcht/neurogolf-6029-submission-bundle` as the 400-task anchor and overrides only tasks 277, 330, and 364 with hand-built semantic ONNX solvers.
- This fits the working constraint: use a single best public package as anchor, then avoid public-package blending.
- Important negative result from the notebook: swapping in cheaper public task versions predicted a gain locally but scored much worse online. Treat cheap public swaps as hidden-set unsafe.

## Strategy

- Build and submit the 6042.85 baseline first.
- Compute exact local cost ranking for the anchor, including the three semantic overrides.
- Target high-cost tasks where the JSON examples reveal a stable semantic rule.
- Implement new hand-built ONNX solvers only when the rule passes all local `train + test + arc-gen` examples and is likely to generalize to private checks.
- Prioritize tasks with high replacement upside: a correct low-cost solver for a high-cost task is worth roughly `new_score - old_score`; for many expensive public solvers this can be 8 to 14 leaderboard points per task.

## Current Best Submission

- As of 2026-06-02 20:49 CST, the Kaggle public best is `6122.59`.
- Latest downloaded leaderboard snapshot ranks team `Jiayi Du` at `229 / 1549` with score `6122.59` and 61 submissions.
- Best file: `submission.zip`, same SHA-256 as `submissions/submission_current_best_6122.zip`: `4aaab23d56f0b6494fad2743ed99927a2dd3cf3cfb4d5841fd5a511e8a870e8c`.
- Current ONNX directory: `submissions/current_best_6122_onnx`.
- Local score file: `reports/current_best_6122_score_n3.csv`, local predicted total `6122.5961`.
- Official progression:
  - `6042.85`: Octavi 6042 anchor, no public blend swaps.
  - `6068.69`: full fp16 pass; hidden/official score exposed unsafe fp16 tasks.
  - `6076.97`: reverted candidate fp16 tasks around the ~15 point hidden drop.
  - `6081.32`: added semantic bool hand-build `task084`.
  - `6086.01`: re-added official-safe fp16 groups A/B/C.
  - `6089.49`: added safe fp16 D tasks `308, 335, 371, 397` plus semantic `task200`.
  - `6090.27`: added official-safe fp16 tasks `030, 051, 175, 306, 345`.
  - `6090.33`: added safe v1-fp16 tasks `076, 158, 187, 285`.
  - `6090.70`: added semantic hand-build `task387`.
  - `6093.31`: added semantic hand-build `task025`.
  - `6094.17`: added semantic flood-fill hand-build `task243`.
  - `6103.94`: added 48 local-safe v2/v1 fp16 compressions on top of `6094.17`; full probe and A/B/C/D group probes all matched official expectations.
  - `6106.97`: added biohack-6113 `task151` single-task swap after independent rule verification.
  - `6109.56`: stacked biohack-6113 `task028`.
  - `6117.56`: stacked six additional biohack-6113 tasks.
  - `6118.58`: int64-to-int32 lossless surgery on 41 tasks.
  - `6119.51`: onnxoptimizer safe-pass probe on 32 tasks.
  - `6120.03`: onnxsim simplification on 17 tasks.
  - `6120.10`: biohack-6113 `task322` small safe swap.
  - `6122.59`: biohack-6113 `task200+task084` replacing our earlier hand-builds with cheaper hidden-safe equivalents.

## Local Results

- `submission_octavi_6042.zip` was submitted as `submission.zip` and returned public score `6042.85`.
- A safe fp16 compression pass over that anchor kept 105 tasks.
- Independent local rescoring of the fp16-compressed anchor predicts `6082.7189`, a `+39.8650` gain.
- The first broad fp16-compressed version scored `6068.69` online, below local prediction. Later official probes isolated and validated safe fp16 subsets; known unsafe single re-add remains `task303`.
- Already merged into current best: hand-built `task025`, `task084`, `task200`, `task243`, `task387`; safe fp16 re-adds discovered through official probes, including the latest 48-task compression probe from `6094.17`.
- Latest full fp16 probe ref `53238466` scored `6103.94`. Group probe refs `53238519`, `53238523`, `53238527`, and `53238532` each scored `6096.61`, confirming the four 12-task groups were individually safe.
- Final handoff resubmission ref `53238770` was uploaded with the same SHA as `submissions/submission_current_best_6103.zip`; confirmed `COMPLETE` at `6103.94` on 2026-06-01.
- Session 2026-06-01 follow-up: re-verified `53238770` still `COMPLETE` / `6103.94`; no new Kaggle submission (no candidate beat local `6103.94`).
- 2026-06-01 task151 probe: ref `53250965` **COMPLETE `6106.97`** (`+3.03` exact match to local). Single-task swap of `task151` from biohack 6113-bundle (1-Conv, 910 cost) into our 6103 best. Rule independently verified `266/266` (intersection of horiz+vert lines, paint 8 neighbors with color 4, keep center). Promoted to new current best: `submissions/current_best_6107_onnx`, `submissions/submission_current_best_6107.zip` (sha `43918fa9...`). Old 6103 preserved at `submissions/submission_current_best_6103.zip`.
- 2026-06-01 task028 stack probe: ref `53251105` **COMPLETE `6109.56`** (`+2.59` over 6107, matches local `6109.57` within 0.01). Stacked task028 swap to biohack 6113-bundle (Slice + ReduceMax + Mul lookup, cost 8994 vs ours 120690). Promoted to new current best: `submissions/current_best_6109_onnx`, `submissions/submission_current_best_6109.zip` (sha `8c638736...`).
- **Session result 2026-06-01 → 2026-06-02: 6103.94 -> 6117.56 (+13.62 official) via three submissions (8 single-task swaps); MEGA-SWAP probe 6115.13 rejected.**
- 2026-06-02 onnxsim simplify probe ref `53279087` **COMPLETE 6120.03 (+0.52 vs 6119, NEW BEST)** — `tools/onnxsim_probe.py` (onnxsim 0.6.4 deeper constant fold + dedup than onnxoptimizer) on 6119. 17 tasks kept after `same_decoded_outputs`. Top wins: task233 (+0.183), task367 (+0.092), task175 (+0.079), task330 (+0.050), task158 (+0.046). Local predicted +0.52, official +0.52 -- PERFECT match.
- 2026-06-02 biohack task200+084 probe ref `53283290` **COMPLETE 6122.59 (+2.49 vs 6120, NEW BEST)** — biohack-6113 replacements for our semantic hand-builds t200/t084. Local +2.49, official +2.49 PERFECT. biohack source now **11/11 hidden-safe swaps**. Rank **211/1540**.
- 2026-06-02 konbu17 task101 probe ref `53283416` **COMPLETE 6111.64 (-10.95 vs 6122)** — konbu cherry-pick safety test **FAILED**. Local predicted +2.95, actual -10.95 vs 6122 base. **konbu17 permanently closed** (same class as afr1ste/octv). All staged konbu/combo/max_safe zips rejected. Only biohack-6113 remains safe for external cherry-pick; remaining biohack wins are noise (+0.04 total).
- 2026-06-02 biohack-6113 task322 swap ref `53279581` **COMPLETE 6120.10 (+0.07 vs 6120)**
- 2026-06-02 onnxoptimizer safe-pass probe ref `53277370` **COMPLETE 6119.51 (+0.93 vs 6118, NEW BEST)** — `tools/onnx_optimize_probe.py` with 30 safe passes (eliminate dead/unused/dup initializers, fuse pad/bias/consec ops, CSE, extract_constant_to_initializer) on 6118. 32 tasks kept after `same_decoded_outputs` validation. Top wins: task209 (+0.348), task204 (+0.170), task233 (+0.102), task324 (+0.100), task363 (+0.072). Local predicted +0.93, official +0.93 -- PERFECT match. Combined with int-surgery, today: +1.95.
- 2026-06-02 int64->int32 surgery probe ref `53277088` **COMPLETE 6118.58 (+1.02 vs 6117, NEW BEST)** — ran `tools/int_surgery_probe.py` across all 400 tasks of 6117. 41 tasks converted (int64 -> int32 for non-shape-input constants, all verified `same_decoded_outputs`). Local predicted +1.02, official +1.02 -- PERFECT match. Lossless type narrowing confirmed hidden-safe (no precision change). New best `submissions/submission_current_best_6118.zip` (sha256 9351934e...).
- 2026-06-02 late tooling fix: patched `tools/neurogolf_local.py` to include PID in ORT profile prefixes, reducing trace collisions when multiple probe scripts score the same task concurrently. Also patched `tools/onnx_optimize_probe.py`, `tools/onnxsim_probe.py`, `tools/int_surgery_probe.py`, and `tools/combo_optimize_probe.py` so `original_cost is None` / `candidate_cost is None` no longer appear as false score improvements.
- 2026-06-02 mega-swap probe ref `53266001` **COMPLETE 6115.13 (-2.43 vs 6117)** — stacked 63 swaps (6 afr1ste-6335 + 57 octv-6154) onto 6117. Local predicted **+79.28 to 6196.85**, actual **-2.43**. Reverse signal of 81.72.
- 2026-06-02 bisection probe ref `53276801` **COMPLETE 6087.24 (-30.32 vs 6117)** — ONLY 4 afr1ste monsters (t319/t285/t219/t255). Local predicted **+25.07**, actual **-30.32**. Per-task reverse signal averages -13.8, regardless of swap count.
- **Cherry-pick safety verdict (final, 2026-06-02):**
  - biohack-6113: **SAFE** — 11/11 official swaps matched local within 0.01.
  - afr1ste-6335, octv-6154, **konbu17-v36**: **UNSAFE** — single-task probes regress ~−11 to −30 despite 100% local pass rate.
  - Bundle-level high official score does NOT imply individual network hidden-safety.
  - External swap path **closed** except biohack noise tier (+0.04). Real gains must come from semantic hand-builds.
- **Tooling fix retained**: `tools/score_bundle.py` `_clean_stale_profile_traces()` prevents stale ng_task*.json from corrupting bundle scores during staging.
- Checked but not merged on 2026-06-01:
  - `task064.onnx`: fails local examples (`0/3`) and is lower than current best.
  - `task153.onnx`: passes local examples but costs more than current best (`1,994,179` vs `762,004`), about `-0.962` local score.
  - `task182.onnx`: passes local examples but costs more than current best (`1,001,008` vs `765,146`), about `-0.269` local score.
  - `task182_raw.onnx`: invalid ONNX runtime shape inference (`Conv` pads size).

## Remaining High-Cost Tasks After fp16

- `task285`: cost `99,837,739`, score `6.581`; translation/shifting/object/color/fill/line extrapolation. Very large but semantically nontrivial.
- `task366`: cost `60,429,553`, score `7.083`; cropping/translation/background separation/fill. The apparent rule composites non-background shapes from one region into another.
- `task255`: cost `43,358,268`, score `7.415`; flood-fill/filling-regions. Outputs are unions of separator/gutter bands colored `3`; not simple row/column two-sided fill.
- `task319`: cost `26,465,763`, score `7.909`; crop a dominant-color object and fill its holes with external object patches.
- `task219`: cost `15,842,765`, score `8.422`; extend smaller 8-pattern rows to the right with color `1`, preserving solid/checkerboard phase from a complete exemplar.
- `task233`: cost `4,897,656`, score `9.596`; output is bbox of the largest color-2 component, with holes filled by external patches.

## Current Investigation Notes

- `task255`: rejected a simple “fill zeros that have nonzero on both sides/up-down” rule. Newly added `3`s look like central separator bands inside blank gutters, sometimes recursively induced by orthogonal gutters.
- `task233`: largest background component bbox exactly matches output shape across inspected examples; naive “crop and replace zeros with bg” fails, because outside patch objects must be inserted into corresponding holes.
- `task233` 2026-06-03 structure update: the output shape tracks the **largest connected component of color 2**, not the bbox of all color-2 cells globally. This immediately explains the visible output sizes on the scanned examples.
- `task233` 2026-06-03 prototype line: added [`tools/prototype_task233.py`]($HOME/Desktop/kaggleonnx/tools/prototype_task233.py). Current heuristic:
  1. find the largest connected color-2 component and use its bbox as the output canvas,
  2. initialize the whole canvas to color `2`,
  3. extract smaller disconnected nonzero patches that contain both `2` and at least one non-2 color,
  4. find hole-components in the main 2-canvas,
  5. match each small patch to a hole by comparing the normalized connected hole mask with the patch's normalized `2` support under rotation / mirror,
  6. overlay the transformed patch back onto the canvas.
- This first semantic prototype only reaches **72/266**, so it is nowhere near ONNX-ready yet. But it establishes a stronger foothold than the earlier vague notes: `task233` is very likely a **main-2-component canvas + hole-skeleton patch reinsertion** task, with the remaining difficulty coming from ambiguous orientation / anchor choices and from some patches whose matching support is more subtle than a single hole component.
- `task233` 2026-06-03 placement-pattern audit: extracted **502** uniquely matched patch placements from visible examples. A frequency lookup keyed by `(hole_mask, normalized_patch_2mask, value_set)` is very stable **when that key has been seen before**: leave-one-out accuracy on repeated visible keys is about **96.3%** (`261/271`), with **82/83** fully-seen examples reconstructed perfectly. But the visible `train+test` split exposes only **11** such keys, while `arc-gen` contains **327** keys and **326** of them are novel relative to `train+test`. Practical verdict: there is real low-entropy structure here, but a direct visible-family lookup would be another hidden-unsafe trap, not a Kaggle candidate.
- `task233` 2026-06-03 semantic pivot after freezing task319: updated `tools/prototype_task233.py` with two train-grounded alignment fixes:
  1. treat input patches as reusable templates, not one-shot objects;
  2. match a hole either by a connected `2`-component inside an oriented patch or, as a lower-priority fallback, by oriented patch bbox shape;
  3. break ambiguous matches by spatial proximity between the hole's absolute column position and the source patch's column position.
  Result: prototype improves **`72/266 -> 92/266`** and now closes **`3/3 train + 1/1 test`**, with `88/262` on arc-gen. This is a better semantic foothold than the earlier version because the new rule fixes the only train failure without using arc-gen IDs. It is still far from ONNX-ready; do not tune further against arc-gen without a train-only invariant.
- `task366`: not just overlaying halves; output uses one region as canvas and translates non-background patterns from the other region, with source backgrounds removed.
- `task219`: prototype template-extension is now **CLOSED at 265/265** (`tools/prototype_task219.py`). The working Python rule combines three pieces:
  1. ordered-subsequence row-group matching,
  2. row-complexity scoring (length + island count) instead of literal clipped-subset matching,
  3. a narrow small-width (`K <= 3`) tail correction that translates alternating right-tails with compressed fragments and suppresses direct singleton tails when the fragment has already absorbed them.
- `task219` 2026-06-03 ONNX follow-up: built a compact visible-family lookup ONNX at [`submissions/handbuilds/task219.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task219.onnx), **452,589 bytes**, **89 nodes**, full-example local cost **417,584** / score **12.058** (about **+3.636** vs current-best task219 at score `8.422`). It verifies **265/265** locally and stays within the 1.44 MB limit.
- Important safety check on that `task219` ONNX: **do not submit yet**. A pseudo-hidden audit using only `train+test` to build the lookup tables and evaluating on `arc-gen` gave just **38/261**, with **223/261** arc-gen examples containing unseen template/fragment shape combinations. This means the current ONNX is a high-coverage visible lookup, not a truly semantic hidden-safe solver. Keep it as a construction milestone, not as a Kaggle candidate.
- `task219` 2026-06-02 follow-up: pure bottom-align fixed 8 failures (`23, 88, 103, 106, 116, 137, 192, 245`) but broke 10 previously passing examples (`0, 14, 43, 50, 64, 107, 121, 206, 241, 244`). The newer ordered-subsequence matcher repaired `idx 0`; row-complexity remapping repaired `idx 23`; the final breakthrough came from treating only small-width alternating tails as translatable patterns rather than literal copies. This closes the former 12 hard cases (`37, 40, 46, 128, 195, 215, 218, 228, 235, 240, 257, 260`) without reopening the earlier pass set.
- `task204`: Python flood-fill rule passes **268/268** (`tools/prototype_task204.py`); skipped ONNX this session (current cost ~309k, limited upside).
- `task363`: **265/265 CLOSED** (`tools/prototype_task363.py --mode closed`) — three offset-signature reject rules fix the two baseline false-positives (train ex 0/1) without touching 263 passing examples. **Next: ONNX hand-build** (current cost ~513k; low-cost graph could yield ~+3–8 if hidden-safe).
- `task363` 2026-06-02 follow-up: `data/konbu17_v36/task363.onnx` / `submissions/staging_max_safe/task363.onnx` passes visible `265/265` with cost `54,241`, but graph inspection shows a visible-example hash/table lookup over 265 stored signatures, not a semantic rule. Treat only as a negative example; do not submit as a konbu cherry-pick.
- `task363` 2026-06-02 micro-optimization check: `tools/int_surgery_probe.py` can shave only `8` cost units from current-best task363 (`512771 -> 512763`, `+0.000016` score). This is real but negligible; not submission-worthy.
- `task363` 2026-06-03 semantic ONNX attempt: built a full-shift semantic graph at [`tools/build_task363.py`]($HOME/Desktop/kaggleonnx/tools/build_task363.py). The unrestricted version verified `265/265` but was **unmeasurable** under ORT profiling because ~8k nodes exhausted profiler events. A reduced visible-offset version [`submissions/handbuilds/task363_semantic_visibleoffsets.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task363_semantic_visibleoffsets.onnx) keeps only the 183 offsets actually selected by the semantic probe on visible inputs; it is **457,383 bytes**, **3,677 nodes**, and verifies **265/265** with measurable cost **642,523** / score **11.6268**.
- Verdict for `task363` ONNX, current stage: **not worth swapping in**. Current-best task363 scores about **11.7788** at cost ~`551,943`, so the measured semantic ONNX is actually worse by about **-0.152** despite being correct. This path is still a useful proof that the rule can be graphized, but it is not a leaderboard candidate.
- `task319` 2026-06-03 structure breakthrough: the output is **not** a new synthesis. It is exactly the bbox crop of one input color, with the crop background reset to the dominant input color. The remaining challenge is choosing the correct color among the two smaller non-background candidates.
- `task319` 2026-06-03 heuristic prototype: added [`tools/prototype_task319.py`]($HOME/Desktop/kaggleonnx/tools/prototype_task319.py). The current best hand rule:
  1. build each non-background color's bbox patch,
  2. run-length compress duplicate rows/columns,
  3. generate rotated / mirrored / one-edge-trimmed variants,
  4. prefer the smaller candidate whose variant family can explain the **largest** object's compressed patch,
  5. break ties with the current local ordering `(score, match_largest, -compressed_row_range, -aspect_gap, -compressed_row_var, -compressed_density)`,
  6. plus four narrow fallbacks:
     - if exactly one smaller object is a same-area/same-cell mirrored echo of the largest object, choose the largest object itself;
     - after the main ranking, if the other small candidate is closer to the largest object's aspect ratio by at least **`1/3`**, switch to that other candidate;
     - if the other candidate has a sufficiently stronger local motif relation to the largest compressed patch (`patch->largest-compressed 2x2 IoU` advantage at least **`1/6`**, or higher `compressed->largest-compressed 3x2` motif overlap count), switch to that other candidate;
     - if both candidates share the same compressed frame and one is a strict one-cell superset of the other, choose the denser superset.
- That prototype now reaches **267/267** on full visible examples, up from `263/267`, `258/267`, `256/267`, and `252/267` earlier in the session. The biggest late gains came from treating **aspect-ratio closeness to the largest object** as a narrow override, then adding a tiny motif-aware correction (`2x2` / `3x2` local-window agreement with the largest compressed patch), and finally a one-cell compressed-superset fallback for the last visible holdout.
- `task319` 2026-06-03 threshold-stability audit: the late overrides are **not** pinned to one brittle visible-only decimal. On visible examples, the aspect-ratio switch stays perfect across a broad interval from about **`1/3` up to `7/12`**, and the motif-IoU switch stays perfect from about **`1/6` upward** (as long as the `3x2` compressed-motif count fallback is kept). That is encouraging: the closure does not depend on one razor-thin threshold coincidence.
- `task319` 2026-06-03 graph-feasibility audit: the semantic Python rule is now closed enough that the next bottleneck is clearly **graph size**, not rule discovery. Helpful bounds from the visible set:
  - every example has exactly **3** non-background colors,
  - input max is only **19x19**,
  - output max is only **5x5**,
  - compressed patch shapes stay tiny (mostly `3x3`, `4x3`, `3x4`, `2x2`, `4x4`).
  But the family space is not lookup-friendly: there are **173** distinct large compressed signatures, **422** distinct small signatures, and **267 / 267** distinct `(largest, small-pair)` visible families. So the credible path is a staged semantic ONNX, not a signature table. See [`reports/task319_onnx_feasibility.md`]($HOME/Desktop/kaggleonnx/reports/task319_onnx_feasibility.md).
- `task319` 2026-06-03 first ONNX graph milestone: added [`tools/build_task319_features.py`]($HOME/Desktop/kaggleonnx/tools/build_task319_features.py), a non-submission feature-probe model that graphizes the selector's **front half**:
  - per-color counts,
  - per-color bbox min/max and area,
  - dominant background-color detection,
  - exact ordering of the 3 non-background colors by `(area, cells, -color)` to recover `largest / two-smaller`.
  The emitted probe [`submissions/handbuilds/task319_features.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task319_features.onnx) verifies on **267/267** visible examples, with only **35 nodes** and about **4 KB** file size. Practical meaning: basic object extraction / candidate ordering is cheap enough; the remaining ONNX risk is concentrated in compressed-shape / motif comparison.
- `task319` 2026-06-03 compression-feature audit: added [`tools/task319_graph_feature_audit.py`]($HOME/Desktop/kaggleonnx/tools/task319_graph_feature_audit.py). On **801** visible objects, the `compress_runs(...)` result used by the Python selector is reproduced **exactly** by a local adjacency rule (“keep first row/col, then keep only rows/cols that differ from the previous one”). The derived features `compressed_density`, `compressed_row_var`, and `compressed_row_range` also match exactly on all 801 objects. Practical meaning: those base-selector features are now credible ONNX targets rather than opaque Python conveniences.
- `task319` 2026-06-03 second ONNX graph milestone: added [`tools/build_task319_compress_features.py`]($HOME/Desktop/kaggleonnx/tools/build_task319_compress_features.py). The emitted probe [`submissions/handbuilds/task319_compress_features.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task319_compress_features.onnx) verifies on **267/267** visible examples and exactly graphizes the key compression-derived features:
  - `compressed_cells`
  - `compressed_rows`
  - `compressed_cols`
  - `compressed_density`
  - `compressed_row_range`
  - `compressed_row_var`
  Current size is still modest: **103 nodes**, about **12 KB**. Practical meaning: the hardest part of the base selector is now implemented in ONNX, not just argued to be possible.
- `task319` 2026-06-03 variant canonicalization audit: added [`tools/task319_variant_audit.py`]($HOME/Desktop/kaggleonnx/tools/task319_variant_audit.py). The `score` / `match_largest` logic can be rewritten as equality tests over a **fixed `5x5` padded canvas plus explicit `(height, width)` metadata**, and this reproduces Python exactly on **801/801** visible objects. The largest compressed patch is only `5x5`, and the largest variant family seen on visible examples has size **72**. Practical meaning: even the variant-comparison half of the base selector now has a finite, graph-friendly formulation; the remaining question is engineering cost, not conceptual graphizability.
- `task319` 2026-06-03 base-selector closure audit: added [`tools/task319_base_selector_audit.py`]($HOME/Desktop/kaggleonnx/tools/task319_base_selector_audit.py). Rebuilding the base selector entirely from graph-friendly pieces (adjacency-defined compression, exact compression-derived features, and the fixed-bank `5x5 + shape` variant comparison) reproduces the original Python base choice **exactly on 267/267 visible examples**. Practical meaning: the whole base selector is now closed as an ONNX-friendly formulation; the remaining work is implementation cost and the later fallback stack, not rule ambiguity.
- `task319` 2026-06-03 packed-compression bridge: added [`tools/build_task319_packed_compressed.py`]($HOME/Desktop/kaggleonnx/tools/build_task319_packed_compressed.py). The emitted probe [`submissions/handbuilds/task319_packed_compressed.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task319_packed_compressed.onnx) now verifies on **267/267** visible examples and packs each color's compressed patch into a fixed **`5x5`** canvas, alongside absolute kept-row / kept-column indices for the first `PACK=5` compressed lines. Current size is still modest: **93 nodes**, about **11.3 KB**. Practical meaning: the missing bridge from compression features to exact fixed-canvas variant comparison is now implemented, so the next step is a true base-selector ONNX rather than another feasibility argument.
- `task319` 2026-06-03 packed-interface selector audit: added [`tools/task319_packed_selector_audit.py`]($HOME/Desktop/kaggleonnx/tools/task319_packed_selector_audit.py). Using only `packed_compressed` plus `compressed_rows / compressed_cols`, the base selector can now be reconstructed in Python with **0 mismatches on 267/267** visible examples. Practical meaning: the ONNX contract is now explicit enough that the next builder can target the actual selector, not just more representation probes.
- `task319` 2026-06-03 selector-head probe: added [`tools/build_task319_candidate_features.py`]($HOME/Desktop/kaggleonnx/tools/build_task319_candidate_features.py). The emitted probe [`submissions/handbuilds/task319_candidate_features.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task319_candidate_features.onnx) verifies on **267/267** visible examples and graphizes the two-candidate head features of the base selector: candidate colors, `aspect_gap`, `compressed_row_range`, `compressed_row_var`, and `compressed_density`. Size is tiny: **29 nodes**, about **3.5 KB**. Diagnostic result: if we rank candidates using only these non-transform features, agreement with the full Python base selector is just **161/267**. Practical meaning: the remaining `score` / `match_largest` transform-bank block is decisively the next real target, not a minor cleanup.
- `task319` 2026-06-03 match-feature probe: added [`tools/build_task319_match_features.py`]($HOME/Desktop/kaggleonnx/tools/build_task319_match_features.py). The emitted probe [`submissions/handbuilds/task319_match_features.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task319_match_features.onnx) verifies on **267/267** visible examples and graphizes the hard transform-bank features of the base selector: candidate `score` and `match_largest`. It uses a fixed **72-slot** transform bank with shape-specific gather maps over the packed `5x5` compressed canvases. Current size: **44 nodes**, about **394 KB**. Practical meaning: all six base-selector ranking features are now individually graphized and validated; the next task319 step can target the actual base-selector decision ONNX rather than another representation probe.
- `task319` 2026-06-03 stitched base-selector solver: built the full end-to-end ONNX at [`submissions/handbuilds/task319_base_selector_solver.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task319_base_selector_solver.onnx) by absorbing the seven sub-models (`features`, `compress_features`, `packed_compressed`, `candidate_features`, `match_features`, `base_selector_decision`, `base_selector_render`) into a single graph. `onnx.compose.merge_models` does not support shared top-level inputs, so the new helper in [`tools/build_task319_base_selector_solver.py`]($HOME/Desktop/kaggleonnx/tools/build_task319_base_selector_solver.py) prefixes every sub-model's internal names with a unique tag and rewrites only the per-sub-model interface set back to bundle-level names. Result: **411 nodes**, **434 KB**, full-example local **`262/267`**, cost **`2,066,870`**, score **`10.4585`** (vs current task319 score `7.909` at cost `26,465,763`). **Not** submission-ready: visible misses imply hidden misses, which would zero the task instead of the apparent +2.55 uplift.
- `task319` 2026-06-03 aspect_switch fallback in decision head: extended `tools/build_task319_base_selector_decision.py` so the decision sub-model now implements the `choose_target` aspect-ratio swap on top of the base `max(...)` ordering. The Python prototype uses `>= 1/3` against float64 `aspect_gap`; running the same threshold against float32 in ONNX produces ULP-level noise (e.g. `(4/3 - 1)` evaluates to `0.3333333730697632`, one ULP above `f32(1/3)`) and would flip threshold-touching examples. Per the visible-set audit, the swap threshold is stable across roughly `[1/3, 7/12]`, so the graph and Python reference both use **`0.4`**. With this single fallback the stitched solver gains **+5 visible** (`257/267 → 262/267`); each gained example matched the prototype's aspect-only swap path. The remaining **5 misses** (`25, 183, 189, 191, 242`) need either `mirrored_largest` (idx `189`) or motif / compressed-superset (idx `25, 183, 191, 242`).
- `task319` 2026-06-03 mirrored_largest fallback in decision head: extended `tools/build_task319_candidate_features.py` to also emit `candidate_areas`, `candidate_cells`, `largest_color_squeezed`, `largest_area_squeezed`, `largest_cells_squeezed`, and extended the decision graph to short-circuit `chosen_color = largest_color` when **exactly one** candidate has `match_largest=1` AND its bbox area equals the largest's AND its cell count equals the largest's. This mirrors the prototype's `if len(mirrored_largest) == 1: return largest`. Stitched solver now: **`263/267`** visible, **432 nodes**, **436 KB**, cost **`2,067,074`**, score **`10.4584`** (vs current task319 score `7.909`). Remaining **4 misses** (`25, 183, 191, 242`) all need motif (`patch_lcomp_2x2_iou`, `comp_lcomp_3x2_inter`) or compressed-superset overrides — each requires sliding-window motif counting that is graph-expensive.
- `task319` 2026-06-03 honest assessment, **NOT a submission candidate**:
  1. Visible misses (4) imply hidden misses, which would zero the task on the official scorer instead of yielding the apparent `+2.55` uplift.
  2. Even at 267/267 visible the prototype is flagged hidden-unsafe in the earlier visible/arc-gen audit — adding more graph machinery does not change that.
  3. The base-selector stitching, aspect_switch fallback, and mirrored_largest fallback are kept on disk as engineering milestones (`submissions/handbuilds/task319_base_selector_*.onnx`) so the next session can pick up from a known-good stitched state without re-deriving the wiring.
- `task319` 2026-06-03 hidden-safety audit (`tools/task319_hidden_safety_audit.py`) — **decisive negative signal, ONNX path FROZEN**:
  - Splits: `train=4`, `test=1`, `arc-gen=262`.
  - Base `max(...)` selector alone already passes **`4/4` train + `1/1` test + `252/262` arc-gen**. Every visible miss is in arc-gen.
  - All four fallbacks (`aspect`, `mirrored`, `motif`, `superset`) contribute **+0** on train AND **+0** on test. Their entire visible-coverage gain is absorbed by arc-gen (`+4 / +1 / +4 / +1` arc-gen passes respectively, in closure order).
  - Current `current_best_6122 task319` (cost `26,465,763`, score `7.909`) passes **`4/4 + 1/1 + 262/262 = 267/267`**. The cheaper stitched solver passes only `4/4 + 1/1 + 258/262 = 263/267` (the four still-broken cases are the motif/superset ones).
  - Practical reading: the prototype's fallbacks were tuned against arc-gen failures, not against any hidden invariant; arc-gen is procedurally generated and almost certainly does not match Kaggle's hidden private distribution. Building motif/superset machinery in ONNX would only close arc-gen-specific failures, at the cost of a graph that is more brittle to hidden than the current 267/267 task319 already is.
  - Decision: **freeze** further task319 ONNX work (no motif/superset graph). Keep `submissions/handbuilds/task319_base_selector_*.onnx` as engineering milestones only; never submit.
- 2026-06-03 public package scan: pulled `nadeembinshajahan/6252-lb-neurogolf-6252-baseline` from Kaggle. The notebook itself is just a copier/zipper over attached ONNX files, but its output `submission.zip` is real: 400 ONNX files, sha256 `ef1ac01b6dcb15c5b9fd3afe1970244b00e30461370219210b53a9edbcd7d803`, local score **`6252.3337`** with zero scorer errors. Public leaderboard confirms Nadeem is currently at **`6276.10`** (rank 93), so this public 6252 bundle is plausible as a slightly older shareable anchor.
- 2026-06-03 Nadeem provenance/risk audit:
  - Compared task-level hashes against tracked sources (`current_best_6122`, `biohack44_6113`, `biohack44_6067`, `konbu17_v36`, `afr1ste_6335`, recursive `octaviograu_6154`, `needless090_v31`, `cdeotte`, `yash9439`, `jsrdcht_6029`). Most top improvements (`task319`, `285`, `219`, `366`, `133`, `157`, `153`, `118`, `158`, `209`, `182`, `316`, `393`) remain **unique fingerprints** relative to tracked sources, but `task255`, `task044`, and `task233` match recursive `octaviograu_6154`. This matters because Octavi/mega-swap was already classified hidden-unsafe despite strong local scores.
  - Versus our current 6122 anchor, Nadeem improves 168 tasks, ties 194, and regresses 38 materially (42 with tiny positive deltas included). Net local gain is **`+129.738`**.
  - Built a conservative reverse-cherry candidate by starting from Nadeem and replacing every task where our submitted `current_best_6122_onnx` is cheaper/correct locally. Output: `submissions/candidate_nadeem6252_plus_ours_38_onnx` and zip `submissions/submission_candidate_nadeem6252_plus_ours_38.zip` (sha256 `6689af88fabaf74899461803161957a85ff9b20425c40b43509ac9030276058b`, 400 files, 741,532 bytes). Full local rescore: **`6254.6634`**, zero errors.
  - Submission posture: do **not** auto-submit. This candidate is likely the best immediate public-LB move (+132 local over current), but it inherits at least some Octavi-source tasks and the user explicitly asked to wait before submitting. If submitting later, use the conservative blend zip rather than the raw Nadeem zip, and treat it as a leaderboard-anchor probe rather than a hidden-safe final package.
- 2026-06-03 Nadeem anchor isolated re-verification: ran `tools/isolated_task_eval.py` on the five flagged tasks (`task018/230/264/282/317`) for both `submissions/nadeem_6252_onnx`, `submissions/current_best_6122_onnx`, and `submissions/candidate_nadeem6252_plus_ours_38_onnx`. Findings:
  - `task230/282/317` are clean: all three sources pass full visible (`266/266` or `265/265`) when scored in isolated processes. Their `local_pass=0/3` rows in `reports/nadeem_6252_score_n3.csv` are pure same-process scorer/ORT pollution and can be ignored.
  - `task018` fails locally on **all three sources** (0/266 in our verifier), but the LB scores `task018` for `current_best_6122` at `11.32`. The local verifier is therefore stricter than the official scorer for that task; until we can mirror official decoding, prefer the LB-verified `current_best_6122` version.
  - `task264` is a real Nadeem regression: Nadeem fails **0/265** while `current_best_6122` passes **265/265**. Kaggle's official scorer gives 0 to incorrect tasks, so v1 (with raw Nadeem `task264`) would actually lose `~13.7` points officially even though it shows local 13.72.
- 2026-06-03 Built `submissions/candidate_nadeem6252_plus_ours_v2_onnx` and `submissions/submission_candidate_nadeem6252_plus_ours_v2.zip` (sha256 `a62fb6ecb8bcf9dc0cfe369dd23bfd34a5001b75e6fece4135c59ca6d0e2edc3`, 742,313 bytes) by overriding v1's `task018` and `task264` with `current_best_6122` versions. Local total `6253.4994` (`-1.16` vs v1 local 6254.66, but conservatively v2 is **`+12.55` higher in expected official LB**: v1 official ≈ 6254.66 - 13.72 = 6240.95, v2 official ≈ 6253.50 because both replaced tasks are LB-verified).
- 2026-06-03 Provenance layer: built `tools/build_provenance.py`. Wrote per-source manifests to `reports/source_manifests/<source>.csv` for 13 tracked sources (current_best_6122, nadeem_6252, candidate_v2, biohack44_6113, biohack44_6067, konbu17_v36, afr1ste_6335, octaviograu_6154, needless090_v31, cdeotte, yash9439, jsrdcht_6029, massimiliano_eda111). Master `reports/task_provenance.csv` has 400 rows mapping each v2 task to its source label, sha256, score, and per-source sha matches. v2 attribution: 228 tasks `lb-verified-current-best`, 172 tasks `public-anchor-candidate` (Nadeem). Of the 172 Nadeem-attributed tasks, **0** match any known cherrypick-unsafe source (konbu17, afr1ste, octv) by sha; only 1 matches biohack44_6067 (untested-public). 170 tasks are unique to Nadeem within our tracked source set.
- 2026-06-03 Public package source lake expansion: pulled outputs of `massimilianoghiotto/eda-111-best-public-score` and `rauffauzanrambe/neurogolf-champions-arc-best-public-score`. Both produce identical `submission.zip` (sha256 `e6238534ca23c5b2fc90bbf8bcb5e6ccbf38606a000d24dd714249dc9433b33f`); they reference the same upstream dataset. Local rescore: `submissions/massimiliano_eda111_onnx` total `6110.98` — well below Nadeem 6252 and v2/v3 candidates. Logged as `untested-public` reference in source lake. `biohack44/neurogolf-superior-blend-notebook` paginated only 333 of 1700+ files within budget, so we skipped it (it's a max-blend over Octavi 6154 + biohack 6067 anyway, expected ceiling is around 6202).
- 2026-06-03 Long-form ledgers landed:
  - [`reports/solver_ledger.md`]($HOME/Desktop/kaggleonnx/reports/solver_ledger.md) -- canonical per-task semantic solver status, anti-patterns, and the hidden-safety gate.
  - [`reports/anchor_integration_ledger.md`]($HOME/Desktop/kaggleonnx/reports/anchor_integration_ledger.md) -- anchor / candidate state, what we discarded and why, public source lake snapshot, append-only update protocol.
  - [`reports/self_research_assets.md`]($HOME/Desktop/kaggleonnx/reports/self_research_assets.md) -- engineering / research / cold-stored asset map, hard rules carried forward.
  These three files are the current source of truth for "what is staged" and "what we will not do"; `research.md` remains the timeline.
- 2026-06-03 Lossless surgery on v2 (only the 172 Nadeem-source tasks; the 228 ours-source tasks already received lossless on the 6122 chain): produced
  - int_surgery: kept 12, +0.382
  - onnxoptimizer (safe passes): kept 31, +0.495
  - onnxsim: kept 42, +0.631
  Per-task best of {int, opt, sim}: 53 tasks improved.
  Built `submissions/candidate_nadeem6252_plus_ours_v3_onnx` and `submissions/submission_candidate_nadeem6252_plus_ours_v3.zip` (sha256 `8d0fd260427efb547d30cf5d8ff744fc13a2ed9f1a367df1fd8d0930869dcacc`, 732,902 bytes). Full local rescore **`6254.5950`** (`+1.0956` over v2). Same-decoded-outputs verified for every kept task. v3 is the new conservative anchor candidate.
- 2026-06-03 Nadeem anchor safety check: wrote per-task source manifest to [`reports/nadeem_6252_manifest.csv`]($HOME/Desktop/kaggleonnx/reports/nadeem_6252_manifest.csv) with SHA-256, local score/cost/pass/nodes/size, and score delta vs `current_best_6122_onnx`. Recomputed total is **`6252.3337`**, current-best local total is **`6122.5961`**, delta **`+129.7375`**. The `reports/nadeem_6252_score_n3.csv` `task230 local_pass=0/3` row is a scorer/ORT same-process pollution artifact: isolated `task230` verification is **`train 3/3 + test 1/1 + arc-gen 262/262`**, while scoring `task220` first flips later `task230` checks to `0/3`. The remaining flagged tasks are real local visible failures under the current verifier: `task018 0/266`, `task264 0/265`, `task282 0/265`, `task317 0/265`; `task264` is a Nadeem regression vs current best, while `task018/282/317` are already inherited as locally failing in current best. Verdict: raw Nadeem is plausible as a public-LB anchor because its cost score matches the known 6252 package and leaderboard context, but it is **not** a clean correctness-safe anchor. Block promotion until we either accept a leaderboard-oracle probe explicitly or use the conservative reverse stack first.
- 2026-06-03 micro-stack on top of current `6122.59`: built [`submissions/current_best_6122plus_micro_onnx`]($HOME/Desktop/kaggleonnx/submissions/current_best_6122plus_micro_onnx) and zipped [`submissions/submission_current_best_6122plus_micro.zip`]($HOME/Desktop/kaggleonnx/submissions/submission_current_best_6122plus_micro.zip) (sha256 `719eed3ff1491e9a3d806e8f2bae6519037f2ccd28182ed8ba4437bf8fd21f16`). This stacks three previously verified lossless micro-improvements onto `current_best_6122`:
  - `task200`: `onnxsim` candidate from [`submissions/sim_on_6122/task200.onnx`]($HOME/Desktop/kaggleonnx/submissions/sim_on_6122/task200.onnx), local `+0.0019999`
  - `task363`: int32 surgery from [`submissions/int_on_6122/task363.onnx`]($HOME/Desktop/kaggleonnx/submissions/int_on_6122/task363.onnx), local `+0.0000156`
  - `task366`: optimizer candidate from [`submissions/opt_monsters_6122_rerun/task366.onnx`]($HOME/Desktop/kaggleonnx/submissions/opt_monsters_6122_rerun/task366.onnx), local `+0.0009616`
  Full local rescore in [`reports/current_best_6122plus_micro_score_n3.csv`]($HOME/Desktop/kaggleonnx/reports/current_best_6122plus_micro_score_n3.csv) gives **`6122.5991`**, a **`+0.00298`** uplift over local `6122.5961`.
- 2026-06-03 Kaggle submission pushed: ref **`53301295`**, description `6122 micro: t200 sim + t363 int + t366 opt, local 6122.5991`. Current remote status at handoff: **`PENDING`**. Important submission quirk confirmed during this push: this competition rejects uploads unless the file name is exactly **`submission.zip`**.
- 2026-06-03 follow-up on ref **`53301295`**: status came back **`COMPLETE`** at **`6122.59`**. So the micro-stack was submission-safe, but its local `+0.00298` uplift did **not** convert into a higher public score. Practical takeaway: this exact `task200 + task363 + task366` micro-bundle is not enough to move the official leaderboard beyond the current best.
- `task319` 2026-06-03 follow-up selector audit: tried a small-feature decision-tree style search over candidate stats (`cells/area/height/width/compressed-shape/components/variant-match score`). No simple stump/tree beat the current semantic selector; the best greedy tree reached only **250/267**. Practical takeaway: the remaining misses are not explained by one more obvious scalar heuristic.
- Current `task319` failure pattern is now fairly crisp:
  - when one small candidate has a variant relation to the largest object and the other does not, that signal is usually correct;
- The visible task is now **closed in Python** (`267/267`). That is a real milestone: the rule no longer depends on any example hash or lookup table, and every late fix stayed in the space of generic shape / motif relations rather than visible-example IDs.
- New caution after closure: this is not automatically Kaggle-ready yet. The selector is now a layered semantic heuristic, not a single crisp invariant, so the next risk is **hidden-set brittleness**, not visible-example coverage. Before promoting this toward Kaggle, we need a stronger hidden-safety sanity check and then an ONNX graph design that preserves the rule without exploding cost.
- `task233` 2026-06-03 re-opened as the backup line. The current prototype is still weak (`72/266`), but the task remains the cleanest adjacent semantic target after `task319`, because the canvas-selection step is now understood and the remaining uncertainty is concentrated in patch-to-hole alignment rather than in the whole transformation.
- `task366` 2026-06-02 micro-optimization check: a sequential `onnxoptimizer` rerun found a safe but tiny gain (`60083225 -> 60025477`, `+0.00096` score). Some visible `arc-gen` examples are wider than 30 columns, and `same_decoded_outputs` intentionally skips examples larger than `30x30`, so this remains a low-confidence micro-gain and is not worth spending a Kaggle submission on.

## Next Best Targets

- Continue semantic hand-build work on the largest remaining costs: `task285`, `task366`, `task255`, `task319`, `task219`, `task133`, `task233`, `task157`, `task158`, `task209`, `task044`, `task118`.
- The fastest path to 6300+ is not more public bundle merging; it is a few high-cost semantic replacements. A correct low-cost replacement for `task285`, `task366`, or `task255` can be worth roughly 10 to 15 points each.
- Near-term order: either semanticize `task219` beyond visible-shape lookup, or move to a different high-upside semantic target (`task285`, `task255`, `task319`, `task366`) rather than `task363`. `task219` is closed in Python but not yet hidden-safe as ONNX; `task363` is now graphized but not cheap enough to beat current best. Skip task204 ONNX unless a very low-cost graph appears.
- For dtype/graph surgery, keep using Kaggle as oracle in small probes. Known unsafe fp16 single re-add: `task303` caused a large public drop and should remain reverted.
