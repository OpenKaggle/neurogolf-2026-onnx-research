# Task191 Compact V3

Date: 2026-06-06

## Scope

Owned only the task191 compact boundary-aware compiler lane. I did not touch
anchors, zips, or ledgers. New work is limited to:

- `tools/build_task191_compact_v3.py`
- `reports/task191_compact_v3.md`
- `reports/task191_compact_v3_verify.json`
- optional non-stage handbuild probes under `submissions/handbuilds/`

## Baseline

Current exact anchor remains `v4_plus10_compiler2` / task191 cost
`691342` with `267/267` visible
validation.

The public compact Massim-family graph is still cheap but non-stageable:

| model | visible | cost | memory | params | nodes |
|---|---:|---:|---:|---:|---:|
| public compact | 227/267 | 89072 | 88932 | 140 | 212 |
| holes-only rejected | 200/267 | 89072 | 88932 | 140 | 212 |
| BOOL-final rejected | 227/267 | 90872 | 90732 | 140 | 214 |
| zero-dependency rejected | 227/267 | 107077 | 106936 | 141 | 215 |

## Missing Subgraph

The public compact graph dynamically extracts the source stamp, normalizes it
to `5x5`, generates D4 masks, then runs:

```text
v286 = v284 - v285
v288 = Conv(color4_plane, v286)
v292 = v288 > (hole_count - 0.5)
v298 = ConvTranspose(v292, fill_masks)
```

Here `v284` is the transformed hole mask and `v285` is the transformed fill
mask. This is too strict for the failed large-stamp families because visible
color-4 anchors under another candidate's fill area cancel true hole matches.
The result is a match score below threshold and the whole stamp remains
background. The 40 public failures are concentrated in source stamps with
three holes and shape `5x4` or `5x5`.

Failure summary for the public compact graph:

- failures: `40` examples
- source shape counts: `{'5x4': 30, '5x5': 10}`
- source hole counts: `{'3': 40}`
- clipped-placement counts among failed examples: `{'0': 21, '1': 15, '2': 4}`

The exact compact predicate validated in Python is:

1. all transformed holes land on visible color 4;
2. every visible fill cell is color 0 or 1;
3. at least one visible fill cell is missing color 1;
4. holes must be fully visible, but fill cells may be clipped by the 23x23 canvas.

Python equivalence of this predicate against the closed semantic solver:

```json
{
  "hole_count_plus_visible_fill_guard_false_positive_placements": 0,
  "hole_count_plus_visible_fill_guard_false_negative_placements": 0
}
```

## Rejected Probes

`holes_only_rejected` replaces `v286 = v284 - v285` with `v286 = v284`.
It proves the diagnosis by changing the match predicate, but it overfills and
drops to `200/267`. The missing
piece is therefore not a one-node kernel replacement; it needs a visible-fill
guard.

`bool_final_rejected` keeps the public matcher but emits the final one-hot
tensor as BOOL using `And/Or/Concat`. It stays at
`227/267`, so the blocker is
match-mask generation rather than final output casting or background clearing.

`zero_dependency_rejected` keeps the public graph and adds a harmless zero
dependency on the dynamic kernel path. It also stays at
`227/267`, so materialization tricks
do not stage.

## Cost Target

A true compact compiler should keep the Massim-style D4 dynamic mask generator
but replace the single signed-kernel convolution with two guarded branches:

- `hole_hits = Conv(color4_plane, hole_mask)` and threshold at `hole_count - 0.5`;
- `bad_fill = Conv(non_0_or_1_plane, visible_fill_mask)` thresholded at `0.5`;
- optionally `missing_fill = Conv(color0_plane, visible_fill_mask)` thresholded at `0.5`;
- `match = hole_hits & !bad_fill & missing_fill`, with boundary handling that
keeps holes fully visible and lets fill cells clip.

The near-term target is an exact `267/267`
single-output model below roughly `150k` cost. The current compact family is
`89072` but fails; the current exact anchor is `691342`.

## Decision

No task191 compact-v3 ONNX can stage from this pass. The best exact model
remains the current anchor at `267/267`,
cost `691342`. The compact lane has a concrete missing subgraph
pattern and cost target, but none of the emitted probes is exact.
