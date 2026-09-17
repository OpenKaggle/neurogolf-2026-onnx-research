# DSL Frontier - 2026-06-06

Current comparison anchor: `v4_plus10_compiler2`.

## Readout

- The existing DSL is correct and useful, but current search hits do not expose a fresh v10 submission branch.
- Mature leaves (`transpose`, `swap_colors`, `flip_h`, `flip_v`) are already saturated or integrated.
- The next non-local-maximum step is typed object/mask IR, not deeper search over full-FLOAT grid primitives.
- Treat train-only hits as diagnostics. They are not candidates unless isolated ONNX full eval passes.

## Primitive Frontier

| primitive | category | hits | full | v10 cheaper/equal/loser | tasks | recommendation |
| --- | --- | ---: | ---: | --- | --- | --- |
| hollow_out | needs_object_ir | 1 | 0 | 0/0/0 | task171 | Promote to object/mask IR; depth search over current FLOAT builders is a local maximum. |
| shift_down | needs_typed_rewrite | 2 | 0 | 0/0/0 | task053 task261 | Rebuild as typed bbox/index primitive with BOOL masks and no full-grid mask multiply. |
| dominant_color | needs_typed_rewrite | 1 | 1 | 0/0/1 | task129 | Split into BOOL mask plus paint/cover primitives; avoid materializing full FLOAT one-hot until final output. |
| tile_h | needs_typed_rewrite | 1 | 1 | 0/0/1 | task249 | Rebuild as typed bbox/index primitive with BOOL masks and no full-grid mask multiply. |
| crop | partly_mature | 3 | 1 | 0/0/1 | task048 task326 task346 | Keep, but compare against v10 before staging; only tiny crops are near competitive. |
| rotate180_static | partly_mature | 2 | 2 | 0/0/2 | task087 task140 | Add lower-level static Slice/Pad templates per task; current generic version still loses to v10 rotate handbuilds. |
| flip_h | partly_mature | 1 | 1 | 0/0/1 | task150 | Already integrated through v4_plus5; use as sanity template, not a new score lane. |
| flip_v | partly_mature | 1 | 1 | 0/0/1 | task155 | Already integrated through v4_plus5; use as sanity template, not a new score lane. |
| rotate90_static | partly_mature | 1 | 1 | 0/0/1 | task380 | Add lower-level static Slice/Pad templates per task; current generic version still loses to v10 rotate handbuilds. |
| swap_colors | mature_cheap | 5 | 3 | 0/3/0 | task171 task261 task276 task309 task337 | Keep in depth-1/depth-2 search; no engineering needed unless a new exact hit beats v10. |
| replace_color | mature_cheap | 2 | 0 | 0/0/0 | task048 task346 | Keep in depth-1/depth-2 search; no engineering needed unless a new exact hit beats v10. |
| transpose | mature_cheap | 2 | 2 | 0/2/0 | task179 task241 | Keep in depth-1/depth-2 search; no engineering needed unless a new exact hit beats v10. |

## Next Typed Primitive Bets

1. `mask_color(c) -> BOOL[1,1,H,W]`: cheap channel extraction retained as BOOL.
2. `paint_mask(grid, mask, color) -> grid`: final-stage paint with no FLOAT one-hot until output.
3. `cover_mask(grid, mask) -> grid`: remove/zero selected cells as BOOL arithmetic.
4. `where_grid(mask, a, b) -> grid`: typed selection that avoids `[1,10,30,30]` FLOAT mask casts when possible.
5. `shift_mask_{up,down,left,right}` and `shift_grid_static`: Slice/Pad or Gather on BOOL/small spatial envelopes.
6. `bbox(mask) -> scalar coords`: reuse small INT64 vectors/scalars across crop, shift, and object rules.
7. `crop_static(r,c,h,w)`: preserve the cheap tiny-crop behavior and route dynamic bbox crop through object IR.
8. `holes_not_bordering(mask)`: needed by task233/task255-style hole/corridor tasks.
9. `majority_body_and_anchor_masks`: task285-like connected-object decomposition as a bounded template.
10. `bounded_rect_fill`: task366-like rectangle/fill primitive with static envelopes.

## Immediate Use

- Keep Hegel on `task285` compact compile and Hume on `task191` compact compile.
- Do not spend more time increasing depth over the current v4 primitive list; that is the local maximum.
- Open the next semantic-factory pass only after either `task285/task191` returns or the typed primitives above have stubs.
