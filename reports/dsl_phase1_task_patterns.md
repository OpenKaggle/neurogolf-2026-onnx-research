# DSL Phase 1: Sampled Task Patterns

Sampling driver: brute-force pattern check against `np.rot90 / np.flip / transpose / color-remap`
for every `task001..task400` (train+test split). Below: 10 task candidates picked to span
"pure transformation" (clean DSL targets) plus 3 high-cost monsters whose underlying logic
is plausibly simple but whose v4_plus3 baseline ONNXs are bloated.

## Primitive vocabulary observed

From the brute-force scan, the following pure 1-primitive tasks exist in the corpus:

| Primitive       | Tasks                          |
|-----------------|--------------------------------|
| rotate90        | task380                        |
| rotate180       | task087, task140               |
| transpose       | task179, task241               |
| flip_h          | task150                        |
| flip_v          | task155                        |
| color_remap     | task276 (6→2), task309 (7→5)   |
| color_swap      | task337 (8↔5)                  |
| (none found pure rotate270, identity)                |

So 9 tasks in the corpus are PURE single-op tasks under {rot, flip, transpose, color-remap}.

## 10 sampled tasks for DSL prototype

| task | baseline_cost | hypothesis                                 | DSL plan                  |
|------|---------------|---------------------------------------------|---------------------------|
| 087  | 376           | rot180 on square 3x3 grids                  | `[rotate180]`             |
| 140  | 376           | rot180 on square 3x3 grids                  | `[rotate180]`             |
| 150  | 1076          | fliph on square grids (3x3 .. 9x9)          | `[flip_h]`                |
| 155  | 1076          | flipv on square grids (3x3 .. 9x9)          | `[flip_v]`                |
| 179  | 0             | transpose on square 3x3                     | `[transpose]`             |
| 241  | 0             | transpose on square grids (3..9)            | `[transpose]`             |
| 380  | 737           | rot90 on 3x3                                | `[rotate90]`              |
| 276  | 10            | color remap 6→2                             | `[replace_color(6,2)]`    |
| 309  | 10            | color remap 7→5                             | `[replace_color(7,5)]`    |
| 337  | 10            | color swap 8↔5                              | `[swap_colors(8,5)]`      |

Three of these (179, 241, 337/276/309) already have **near-zero** baseline cost — they are
"trivial-baseline" anchors that DSL **must** match (not beat). The other seven have a small
but non-trivial baseline cost (376..1076); these are real DSL targets to either match or beat.

## Notes on monster tasks (not in sample but considered)

Looked briefly at:
- task255 (cost 1593763), task233 (cost 1456755), task366 (cost 830720) — these have
  >1M-byte intermediate working sets driven by per-task lookup / template logic. Not 1-primitive
  expressible by the geometric/color/object set proposed below. Deferred to a future "compound
  primitive" iteration.
- task018 (cost 476185, pass 0/3) — baseline is FAILING; DSL can't make it worse but also
  needs more reasoning machinery to express. Deferred.

## Key design constraint identified

The competition's grid encoding is `[1, 10, 30, 30]` one-hot **top-left aligned** — a 3x3 input
occupies channels at `(r=0..2, c=0..2)` and is zero elsewhere. This means:

1. Geometric ops like `flip_h` cannot be implemented by naive 30x30 flip — that would move
   content to the BOTTOM-RIGHT corner. The decoder reads top-left, so we must re-pin content.
2. Color ops (swap, remap) are **shape-preserving** and only permute channels — no bbox math
   needed, can be implemented with a single `Gather(axis=1)` (cost ≈ 0).
3. `transpose` on H==W is shape-preserving along the diagonal — no bbox math needed (cost 0).
4. Flips and rotations require dynamic `(H, W)` computation and an `area_mask` to zero
   out-of-grid cells. This adds 3-5 intermediate `[1,10,30,30]` BOOL tensors (~9000 bytes each),
   so a bbox-aware flip costs ~30-50K. **Higher than handcrafted baselines for cheap tasks**,
   but bounded and composable.

## v0 primitive set proposed

8 primitives, all `[1,10,30,30] BOOL → [1,10,30,30] BOOL`:

- **identity**: passthrough (Greater against zero) — cost 1
- **swap_colors(c1, c2)**: single Gather(axis=1) — cost ~0
- **replace_color(src, dst)**: 3-4 ops (mask src plane, OR into dst plane, clear src plane)
- **transpose**: single Transpose(perm=[0,1,3,2]) — cost 0
- **flip_h**: bbox-aware (compute W, gather with idx[c]=W-1-c, mask) — cost ~30K
- **flip_v**: bbox-aware analog — cost ~30K
- **rotate180**: bbox-aware composite of flip_h + flip_v — cost ~50K
- **rotate90**: bbox-aware composite (transpose + flip on H==W square) — cost ~30K

All compose left-to-right (output of op i is input of op i+1).
