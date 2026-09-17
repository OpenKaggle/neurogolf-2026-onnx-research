# Task285 Semantic V2

## Data closure

- Examples: 265 total; splits {'train': 3, 'test': 1, 'arc-gen': 261}.
- Shapes: {'12x12': 4, '13x13': 4, '14x14': 5, '15x15': 5, '16x16': 1, '17x17': 2, '18x18': 6, '19x19': 6, '20x20': 14, '21x21': 7, '22x22': 12, '23x23': 14, '24x24': 17, '25x25': 19, '26x26': 22, '27x27': 21, '28x28': 32, '29x29': 43, '30x30': 31}.
- All examples have unique majority-color bodies per foreground object: 265/265.
- Foreground objects: 524; anchor cells: 1382; reflected body-cell placements before overlap: 8173; visible added cells after zero-only painting: 6791.
- Semantic rule exact: 265/265, pixel errors 0.

## Rule

- Use 8-connected non-zero foreground components.
- In each component, the unique majority color is the source body; minority cells are anchors.
- Each anchor direction is determined by its position relative to the body bounding box.
- Reflect the body over the corresponding side/corner of the body bbox and paint zero target cells with the anchor color.
- Existing non-zero cells are preserved, so overlaps do not erase anchors or source bodies.

## Weaker hypotheses

- semantic_majority_reflect: exact 265/265, pixel errors 0.
- weaker_same_color_any_marker: exact 243/265, pixel errors 147.
- weaker_same_color_singleton_marker: exact 257/265, pixel errors 27.
- weaker_majority_no_diagonal: exact 9/265, pixel errors 2254.
- weaker_majority_paints_body_color: exact 0/265, pixel errors 6791.

## Pseudo-hidden

- Contract summary: {'total': 3445, 'passed': 3375, 'failed': 0, 'skipped': 70}.
- Leave-one-out visible summary: {'total': 52, 'passed': 50, 'failed': 0, 'skipped': 2}.
- Contracts used: original rows, two arbitrary color bijections fixing 0, transpose, left-right/up-down/180-degree symmetry, zero-border padding, in-frame translations when unclipped, and disconnected inert distractors.
- These contracts directly target hidden-plausible risks: color permutation, translations/borders, reflected direction changes under symmetry, irrelevant distractors, and avoiding a rule learned from any single visible row.

## Existing ONNX

- Anchor path: `submissions/candidate_v4_plus9_compiler_onnx/task285.onnx`.
- Anchor target exact: 265/265; semantic exact: 265/265.
- Anchor score/cost: 12.112175, cost 395468, memory 390848, params 4620, nodes 279, file size 36534.

## ONNX decision

- No `task285_v2` ONNX was built in this pass.
- Closure is strong in Python, but an honest compile of this object-level rule needs connected-component labeling, per-component majority color, body bbox recovery, and marker-directed scatter/reflection. That is likely larger than the current 279-node, 395468-cost local-neighborhood anchor unless a new compact primitive is found.
- A merely moderate semantic compile at cost 173596 would score about 12.936, gain +0.823 over the current anchor.
- The reported public top score 15.131 implies cost about 19322, gain +3.019; matching that requires a substantially more compact encoding than the straightforward semantic graph.
