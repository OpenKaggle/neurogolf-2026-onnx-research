# task066 Contract Audit - 2026-06-07

Scope: evaluate the low-cost konbu/blend task066 candidate against metamorphic contracts that the LB-verified anchor itself satisfies. No submission bundle was built and no anchor files were changed.

## Contract

- Full train/test/arc-gen visible identity must pass.
- Only transformations that the current anchor passes are used as replacement gates.
- Anchor-safe gates here are horizontal reflection, vertical reflection, 180-degree rotation, square-grid transpose, and one-cell top/left/top-left zero padding.
- These transformations preserve the rectilinear 3-to-2 connection geometry visible in the generated corpus while testing coordinate-specific memorization.

## Model Results

| model | visible | all metamorphic | anchor-safe | cost | score | delta vs anchor | checker | stageable | reason |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| anchor_v4_plus16 | 266/266 | 2128/2128 | 2128/2128 | 99734 | 13.489738 | +0.000000 | ok | no | not cheaper than anchor |
| konbu_blend_task066 | 266/266 | 266/2128 | 266/2128 | 21408 | 15.028480 | +1.538742 | ok | no | fails anchor-safe metamorphic contracts |

## Category Breakdown

| category | anchor | konbu/blend |
|---|---:|---:|
| half_turn_rotation | 266/266 | 0/266 |
| horizontal_reflection | 266/266 | 0/266 |
| main_diagonal_reflection | 266/266 | 0/266 |
| translate_down_one | 266/266 | 0/266 |
| translate_down_right_one | 266/266 | 0/266 |
| translate_right_one | 266/266 | 0/266 |
| vertical_reflection | 266/266 | 0/266 |
| visible_identity | 266/266 | 266/266 |

## First Failures

- anchor_v4_plus16: none on anchor-safe contracts.
- konbu_blend_task066: flip_h_000 (horizontal_reflection), flip_v_000 (vertical_reflection), rot180_000 (half_turn_rotation), transpose_000 (main_diagonal_reflection), pad_top_000 (translate_down_one), pad_left_000 (translate_right_one), pad_tl_000 (translate_down_right_one), flip_h_001 (horizontal_reflection).

## Verdict

- `data/konbu17_v36/task066.onnx` / `submissions/blend_v360/task066.onnx` is not stageable despite the visible score gain.
- The candidate passes only the original-coordinate visible corpus and fails every anchor-safe transformation bucket.
- Next minimal loop should move to the next queued candidate rather than submitting task066.
