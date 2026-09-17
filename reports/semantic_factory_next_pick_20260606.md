# Semantic Factory Next Pick - 2026-06-06

Current anchor at handoff update: `v4_plus13_task366`, public LB `6260.48`.

Allowed lane: `task158`, `task173`, `task286`, `task054`, `task025`.
Recommended next bounded worker after `task366`: **`task025`**.

## Why task025

`task025` is the best bounded semantic-factory pick because it has the strongest
combination of visible rule clarity and falsifiable hidden-safety contracts.
The visible task is a line/stray relocation rule: keep the full row/column line
colors, remove non-rule distractors, and move same-color stray cells to the
line-adjacent target side. That makes contract generation concrete: vary line
orientation, line position, stray side, color count, distractors, and border
cases.

The public-tail signal is real but must be handled carefully. The anchor has
score `13.149938` at cost `140093`, while the best seen public-family attempt is
`13.600400` at cost `89286`. That `+0.450462` local gap is larger than the
observed best-seen edge for the clean semantic-rebuild candidates in this lane.
However, the source class is explicitly tainted by `blend_v360` / konbu hidden
failure, so the value is in auditing/proving the rule, not in swapping the file.

The current repo already has `tools/build_task025.py`, and the current anchor
ONNX is a semantic graph rather than a blind public import. That makes this a
bounded worker: inspect the existing builder, define the contract, compare the
anchor and public cheaper candidate against generated cases, then produce a
verdict and a safer compact-builder target. No anchor directories, zips, or
ledgers should be touched.

## Why not the others first

- `task158`: highest clean semantic-rebuild EV in the ML triage (`0.542899`)
  and highest current cost in this lane (`191304`), but the visible rule is a
  richer motif/stamp placement problem. It is a good second pick after `task025`
  because the contract space is less immediately falsifiable.
- `task286`: visible semantics are very clear: seeded checkerboard fill of the
  reachable zero component through maze walls. Its contracts are good, but the
  engineering shape is closer to connected-component/flood-fill compiler work;
  current graph already uses many propagation ops and no better public graph is
  visible.
- `task173`: public-tail signal exists, but it sits in `audit_before_swap` with
  no better best-seen candidate and higher hidden risk than the clean rebuild
  tasks.
- `task054`: no best-seen improvement over anchor, and the visible rule is much
  less crisp than `task025` or `task286`.

## Expected Engineering Shape

1. Build a task-specific pseudo-hidden generator for `task025`.
   Cover horizontal and vertical full lines, one or more active colors, strays
   above/below or left/right of the line, unrelated singleton distractors,
   border-adjacent lines, and cases where a color has no legal move.
2. Run the generator against:
   `submissions/candidate_v4_plus13_task366_onnx/task025.onnx`,
   `submissions/probe_best6090_task025_semantic_onnx/task025.onnx` if useful,
   and the cheaper public-family file from `data/konbu17_v36/task025.onnx`.
3. If the cheaper graph fails generated contracts, use the failures to tighten
   `tools/build_task025.py` or a successor builder. Target a cost below the
   current `140093`; only treat sub-`90K` as materially interesting.
4. If a candidate passes all contracts and exact visible eval, hand it to the
   release lane for isolated scoring and possible single-task probe. The
   semantic-factory worker should not build a submission zip.

## Validation Gates

- Visible full-example equivalence against the `task025` training/test/arc-gen
  corpus, not just `3/3` train.
- ONNX checker and local scorer must pass; graph must stay cost-measurable.
- Generated pseudo-hidden suite must include negative controls for distractor
  colors and ambiguous/no-move cases.
- Candidate output must be decoded-equivalent to the Python reference on every
  generated case.
- Any public-source candidate must beat the current anchor by enough to matter;
  sub-`0.1` local score deltas are not worth a hidden-risk probe here.

## What Not To Try

- Do not raw-swap the konbu/blend `task025.onnx` into the anchor.
- Do not make a bundle or edit `v4_plus12` anchor directories, zips, ledgers, or
  manifests.
- Do not accept visible-clean parity as safety for this source class.
- Do not chase generic DSL depth search; this task wants a narrow line/stray
  contract and a compact hand builder.
- Do not spend the worker on `task054` unless new evidence appears; it has the
  weakest clarity/upside mix in this lane.
