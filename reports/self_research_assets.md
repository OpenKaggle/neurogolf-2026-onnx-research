# Self-Research Assets

What we built ourselves and how each piece is used in the
post-Nadeem-anchor architecture. Three classes:

1. **Engineering assets**: keep using on every candidate. These are the
   verification + lossless backbone.
2. **Research assets**: keep on disk; not submission candidates today, but
   reusable as templates / inspiration / debugging context.
3. **Cold-stored**: kept for reference only; do NOT submit, do NOT extend.

## Engineering assets

| asset                                                            | role                              |
| ---------------------------------------------------------------- | --------------------------------- |
| `tools/neurogolf_local.py`                                       | local scorer + sanitizer          |
| `tools/score_bundle.py`                                          | bundle-level scoring              |
| `tools/single_task_score.py`                                     | per-task profile scoring          |
| `tools/isolated_task_eval.py`                                    | isolated full-example pass eval   |
| `tools/build_swap_zip.py`                                        | overlay one or more tasks         |
| `tools/build_provenance.py`                                      | source manifest + task provenance |
| `tools/build_v3_lossless.py`                                     | merge int/opt/sim into v3         |
| `tools/int_surgery_probe.py`                                     | int64 -> int32 lossless           |
| `tools/onnx_optimize_probe.py`                                   | onnxoptimizer safe passes         |
| `tools/onnxsim_probe.py`                                         | onnxsim simplification            |
| `tools/combo_optimize_probe.py`                                  | per-task best of pipelines        |
| `tools/fp16_surgery.py`, `tools/fp16_dual_surgery.py`            | fp16 surgery + same-output verify |

Notes:

- `tools/score_bundle.py` clears stale `ng_task*.json` profile traces at
  startup so concurrent probes do not corrupt each other's bundle scores.
- `tools/neurogolf_local.py` includes the PID in ORT profile prefix for the
  same reason.
- All lossless probes gate on `same_decoded_outputs` before keeping a
  candidate. Trust these as the fast safety check.

## Research assets

These are NOT current submission candidates but are too valuable to delete.

| asset                                                                                                | what it is                                |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `tools/prototype_task233.py`                                                                         | semantic prototype, currently 92/266      |
| `tools/prototype_task319.py`                                                                         | full Python rule, 267/267, frozen as ONNX |
| `tools/prototype_task219.py`                                                                         | three-rule semantic closure, 265/265      |
| `tools/prototype_task204.py`                                                                         | flood-fill rule, 268/268                  |
| `tools/prototype_task363.py`                                                                         | offset-signature rule, 265/265            |
| `tools/build_task319_*.py`                                                                           | seven graph builders for task319          |
| `submissions/handbuilds/task319_base_selector_solver.onnx`                                           | stitched 263/267 solver (engineering)     |
| `submissions/handbuilds/task319_features.onnx`                                                       | object-extraction probe                   |
| `submissions/handbuilds/task319_packed_compressed.onnx`                                              | packed-compression probe                  |
| `submissions/handbuilds/task319_match_features.onnx`                                                 | transform-bank features                   |
| `submissions/handbuilds/task219.onnx`                                                                | visible-family lookup ONNX (DO NOT SUBMIT)|
| `submissions/handbuilds/task363_semantic_visibleoffsets.onnx`                                        | semantic graph, not cheap enough          |
| `submissions/handbuilds/task243.onnx` etc.                                                           | semantic hand-builds already in 6122 best |

Reusable techniques worth keeping in mental cache:

- Fixed-bank variant comparison (transform-bank-on-padded-canvas) for
  small-grid pattern matching. Used in task319; cheap in graph form.
- Adjacency-defined run-length compression (keep first row/col, then keep
  rows/cols differing from previous). Reproduces `compress_runs` exactly.
- Connected-component analysis from `prototype_task233.py` is a clean ARC
  building block.
- Pseudo-hidden audit: build rules from `train + test` only, evaluate on
  arc-gen. If arc-gen drops materially below visible, the rule is a lookup
  in disguise (caught task219 ONNX, task363 konbu).

## Cold-stored

| asset                                                              | reason                                  |
| ------------------------------------------------------------------ | --------------------------------------- |
| `submissions/candidate_nadeem6252_plus_ours_38_onnx`               | superseded by v2 (task264 fail)         |
| `submissions/handbuilds/task319_base_selector_solver.onnx`         | hidden-safety audit failed              |
| `submissions/candidate_max_safe.zip`, `submission_candidate_konbu_t101.zip`, `submission_candidate_afr_monsters_*.zip`, `submission_candidate_mega_swap_*.zip` | confirmed cherrypick-unsafe sources |
| any "visible-only lookup" ONNX (e.g. `submissions/staging_max_safe/task363.onnx`) | hidden-overfit by construction |

## Hard rules carried forward

1. Never submit a candidate whose ONNX fails any visible example in the
   `isolated_task_eval` verifier without independently confirming Kaggle
   accepts that file (only acceptable when current best already does, e.g.
   the inherited `task018` situation).
2. Never call a public bundle "hidden-safe" just because it scores high on
   the LB. Bundle LB ≠ per-task hidden safety.
3. Never tune a fallback against arc-gen failures alone. If train+test does
   not get any benefit, treat the fallback as overfit.
4. Always verify lossless rewrites with `same_decoded_outputs` on every
   visible example before keeping the candidate.
5. Always log to `reports/anchor_integration_ledger.md`,
   `reports/solver_ledger.md`, and `reports/research.md`. The ledger files
   are the source of truth for what is current; `research.md` is the
   timeline. Do not collapse them.
