# Self-Research Semantic Solver Ledger

This is the canonical status board for every per-task semantic ONNX line we've
built or considered. Update this file when status changes; don't scatter the
truth across `reports/research.md` history.

## Status definitions

- `python_open`            Python prototype not yet closed on visible examples.
- `python_closed`          Python prototype passes 100 % of visible (`train + test + arc-gen`).
- `onnx_built`             ONNX implementation exists at the listed path; verifies on visible.
- `onnx_hidden_unsafe`     ONNX path is closed visibly but failed a hidden-safety audit
                           (e.g. only fixes arc-gen; or visible lookup table).
- `frozen`                 Decided not to invest more graph engineering on this line; existing
                           artifacts kept on disk for reference.
- `submission_candidate`   ONNX is hidden-safe AND cheaper than current best; ready for a probe.
- `submitted`              Has been submitted at least once; record outcomes inline.

## Hidden-safety gate

A solver may only graduate to `submission_candidate` when ALL of the following hold:

1. Python rule passes 100 % on `train + test + arc-gen`.
2. The rule is grounded in invariants visible from `train + test` ALONE. If a
   fallback/threshold was tuned only against arc-gen failures (no `train + test`
   delta), it is treated as overfit until proven otherwise.
3. `score_bundle.py --n-runs 0` and `tools/isolated_task_eval.py` both pass full
   visible on the candidate ONNX.
4. `same_decoded_outputs` against the source we'd be replacing on every visible
   example is verified for any lossless or graph-rewrite step.
5. ONNX cost is strictly cheaper than the current-best version, *and* visible
   correctness is at least equal.

## Per-task status

### task319 -- crop dominant-color object, fill holes with external patches
- python_state: `python_closed` (267/267) via [`tools/prototype_task319.py`]($HOME/Desktop/kaggleonnx/tools/prototype_task319.py).
- onnx_state: stitched base-selector + 2 fallbacks. Best built artifact:
  [`submissions/handbuilds/task319_base_selector_solver.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task319_base_selector_solver.onnx)
  passes 263/267 visible at cost ~2.07M / score ~10.46.
- hidden_audit: `frozen`. `tools/task319_hidden_safety_audit.py` shows base
  selector already closes `train + test` (4/4 + 1/1). All four fallbacks
  (`aspect`, `mirrored`, `motif`, `superset`) are arc-gen-only fixes (+0 on
  train AND +0 on test). They were tuned against arc-gen failures, not against
  any hidden invariant. Building motif/superset graph machinery would only
  close arc-gen-specific failures.
- decision: **frozen**. Do not submit any task319 hand-build. Current best
  task319 (cost 26.4M, score 7.91) passes 267/267 visible cleanly; replacing
  it gambles on arc-gen lookalikes matching Kaggle hidden distribution.
- artifacts to preserve as engineering reference:
  packed compression, transform-bank variant comparison, decision-head
  fallbacks (`aspect_switch`, `mirrored_largest`).

### task233 -- crop largest color-2 component, reinsert patches into holes
- python_state: `python_open`. [`tools/prototype_task233.py`]($HOME/Desktop/kaggleonnx/tools/prototype_task233.py)
  reaches **92/266** total, **train 3/3 + test 1/1**, **arc-gen 88/262**.
- onnx_state: not built.
- semantic foothold: largest-component canvas + hole-skeleton patch reinsertion.
  Visible-key lookup is hidden-unsafe (visible exposes only 11 keys; arc-gen
  has 327 mostly-novel keys).
- next_step_allowed: only train-grounded alignment fixes; never tune against
  arc-gen IDs. Look for invariants that improve `train + test` first; arc-gen
  improvements without train-grounded justification = overfit.
- next_step_disallowed: anything that resembles "leave-one-out arc-gen
  closure", "patch-shape lookup", or "rotation enumerator with frequency
  weights".
- target: get `train + test` to 4/4 + 1/1 stably under multiple invariants
  (current already at 4/4) AND raise arc-gen by at least 30 with NO new
  thresholds tuned against arc-gen-only failures. Then attempt graph build.

### task255 -- flood-fill / separator bands of color 3
- python_state: `python_partial`. `tools/prototype_task255_v2.py` is the current
  best semantic reference.
- onnx_state: not built.
- current `v4_plus10_compiler2` cost is ~1.59M; still a top monster and public
  tail-gap task.
- v2 result: train `3/3`, test `1/1`, arc-gen `209/261`; old stub was
  `0/3 + 0/1 + 0/261`.
- pseudo-hidden status: color permutation and far-noise checks preserve visible
  examples, but clipped zero-padded translation is a strong falsification
  (`train 3/24`, `test 0/8`, `arc-gen 245/2088`).
- decision: do **not** build final ONNX from v2 yet. The next step is semantic:
  add a train-justified recursive attachment rule for shorter corridors that
  attach to accepted major rectangles, and fix translation covariance. Do not
  lower the global run threshold merely because arc-gen improves.

### task285 -- translation/object/color/fill/line extrapolation
- python_state: `python_open`. No prototype; current_best cost ~99.8M under
  6122 anchor.
- onnx_state: not built.
- caveat: Nadeem 6252 has a much cheaper task285 (delta ~+5.5 vs our 6122).
  After v3 anchor swap, the *upside* of a hand-build collapses; retiring this
  as a primary semantic target unless a very specific rule pops up.

### task133 -- anchor-scaled tile stencil
- python_state: `semantic_closed`. `tools/prototype_task133_v2.py` recovers the
  rule: find the shared anchor color present in every component, use the most
  complete component as an anchor-scaled stencil, then stamp the stencil's
  payload-tile offsets onto every other component with its own payload color.
- evidence: train `4/4`, test `1/1`, arc-gen `262/262`; pseudo-hidden
  metamorphic `1785` passed, `0` failed, `84` skipped; leave-one-out `33`
  passed, `0` failed, `2` skipped.
- onnx_state: `not_worth_building_yet`. Current anchor graph already appears
  semantic and passes the same pseudo-hidden suite. The tempting cheaper
  `konbu17_v36` graph is only about `+0.014` local and fails `1293`
  non-skipped metamorphic cases.
- decision: no anchor action. Reuse this as a template for future
  anchor/payload tiled-component tasks, not as a standalone score branch.

### task366 -- cropping/translation/background separation/fill
- python_state: `semantic_closed`. `tools/prototype_task366_v2.py` closes a
  bounded-rectangle rule on train/test/arc-gen `266/266`.
- onnx_state: `tiny_gain_handbuild`. `submissions/handbuilds/task366_v2.onnx`
  matches the target on `255/255` ONNX-encodable rows; 11 arc-gen rows exceed
  fixed `30x30` width. Cost `794720` vs anchor `830720`; estimated gain
  `+0.0443`.
- pseudo-hidden: `1456` passed, `0` failed, `406` skipped by non-clipping
  preconditions.
- decision: keep as a construction asset, but do not spend an anchor slot or
  complicate compiler batches for `+0.044`.

### task191 -- clipped stamp completion
- python_state: `semantic_closed`. `tools/prototype_task191_v2.py` passes
  visible `267/267`, selected pseudo-hidden `90/90`, and global metamorphic
  `1602/1602`.
- rule: identify the source color-1 stamp with color-4 holes; complete every
  D4-transformed occurrence whose color-4 holes are fully visible, while the
  color-1 fill mask may be clipped by the canvas edge.
- onnx_state: `compile_open`. Known compact public family costs down to
  ~89k but only passes `227/267`; current exact anchor cost remains high
  (~691k after v4_plus10 terminal-cast narrowing). Next work is a dedicated
  boundary-aware ONNX compiler, not more semantic inference.

### task219 -- right-extend smaller 8-pattern rows
- python_state: `python_closed`. [`tools/prototype_task219.py`]($HOME/Desktop/kaggleonnx/tools/prototype_task219.py)
  passes 265/265 via three nested rules.
- onnx_state: `onnx_hidden_unsafe`.
  [`submissions/handbuilds/task219.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task219.onnx)
  is a visible-family lookup; pseudo-hidden audit (build tables only from
  `train + test`, evaluate on arc-gen) gave just **38/261**. **Do not submit.**
- decision: keep as construction milestone. Future ONNX must be derived from
  the three semantic rules, not the visible lookup.

### task204 -- flood-fill rule
- python_state: `python_closed` (268/268). [`tools/prototype_task204.py`]($HOME/Desktop/kaggleonnx/tools/prototype_task204.py).
- onnx_state: not built.
- decision: not invested. Current ONNX cost is already modest under 6122
  anchor (~309k). Under v3 anchor we should re-check; if biohack44_6113 or
  Nadeem already provides cheaper, no upside.

### task363 -- offset-signature pattern selection
- python_state: `python_closed` (265/265). [`tools/prototype_task363.py`]($HOME/Desktop/kaggleonnx/tools/prototype_task363.py).
- onnx_state: built but **not cheaper** than current best.
  [`submissions/handbuilds/task363_semantic_visibleoffsets.onnx`]($HOME/Desktop/kaggleonnx/submissions/handbuilds/task363_semantic_visibleoffsets.onnx)
  cost 642,523 / score 11.63 vs current best ~552k / score 11.78 = -0.15.
- decision: graphizable but not submission-worthy. Keep as semantic-to-graph
  template for future tasks; do not submit.

## Anti-patterns to keep frozen

- Visible-only lookup tables that pass 100 % on train+test+arc-gen but score
  far lower in a leave-one-out hidden-style audit (task219 ONNX, konbu task363).
- Cherry-picked single-task swaps from afr1ste / octv / konbu17 / Octavi 6154.
  Bundle-level high LB does not imply per-task hidden safety.
- Stacking arc-gen-tuned thresholds onto a base solver that already passes
  train+test (task319 fallback chain).
- Aggressive fp16 sweeps without per-task allowlist. `task303` is a known
  unsafe single re-add and must remain reverted.

## Where to look next

1. After v3 anchor lands, regenerate the cost-monster ranking against
   `reports/candidate_nadeem6252_plus_ours_v3_score_n3.csv`. Some current
   "monsters" (`task319`, `task285`, `task366`) drop dramatically once
   Nadeem's cheaper versions replace ours; the real next targets are
   whatever stays in the top 10 by cost AFTER the swap.
2. For task233, only invest if a train-grounded alignment fix raises arc-gen
   meaningfully without inventing arc-gen-only thresholds.
3. Do not re-open task319 unless we discover a stronger hidden-safety
   invariant (e.g. a rule that fixes train+test failures we previously
   ignored, not just arc-gen).
