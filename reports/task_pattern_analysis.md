# Task Pattern Analysis — Uncovered Tasks

**Total uncovered tasks analyzed:** 389

**Covered by DSL v0+v1:** 11 tasks

## 1. Primary Pattern Distribution

| Primary Pattern | Count | Percentage |
|----------------|-------|------------|
| symmetry/flip/rotate | 105 | 27.0% |
| other | 80 | 20.6% |
| reduce | 67 | 17.2% |
| color_remap | 37 | 9.5% |
| tiling | 30 | 7.7% |
| color_fill | 29 | 7.5% |
| crop | 22 | 5.7% |
| expand | 6 | 1.5% |
| trim_border | 6 | 1.5% |
| scalar_output | 6 | 1.5% |
| other_shape_change | 1 | 0.3% |

## 2. Tag Frequency (tasks may have multiple tags)

| Tag | Count | Percentage |
|-----|-------|------------|
| symmetry/flip/rotate | 105 | 27.0% |
| other | 81 | 20.8% |
| reduce | 75 | 19.3% |
| color_remap | 37 | 9.5% |
| tiling | 30 | 7.7% |
| color_fill | 29 | 7.5% |
| crop | 26 | 6.7% |
| trim_border | 8 | 2.1% |
| expand | 6 | 1.5% |
| scalar_output | 6 | 1.5% |

## 3. Shape Statistics

- Input shape range: 1–30 × 2–30
- Output shape range: 1–30 × 1–30
- Shape-preserving: 251 (64.5%)
- Transposed (non-square): 0 (0.0%)
- Crop (subregion): 26 (6.7%)
- Tile: 30 (7.7%)
- Trim border: 8 (2.1%)
- Output larger: 36 (9.3%)
- Output smaller: 101 (26.0%)

## 4. Color Statistics

- Output colors ⊆ Input: 256 (65.8%)
- Output colors ⊇ Input: 240 (61.7%)
- Same palette: 154 (39.6%)
- Disjoint palettes: 7 (1.8%)
- Color remap (subset, different): 37 (9.5%)
- Color fill (new colors): 29 (7.5%)
- Avg input palette size: 5.0
- Avg output palette size: 4.9

## 5. Second-Level Pattern Detection

- Counting (1×1 constant): 0 (0.0%)
- Scalar output (1×1, varying): 6 (1.5%)
- Symmetry candidates (same shape+colors): 105 (27.0%)
- Color remap (same shape, colors shift): 37 (9.5%)
- Color fill (same shape, new colors): 29 (7.5%)
- Tiling/replication: 30 (7.7%)
- Crop (exact subregion): 26 (6.7%)
- Trim border: 8 (2.1%)
- Expand (larger, not tiling): 6 (1.5%)
- Reduce (smaller, not crop): 75 (19.3%)
- No pattern detected (misc): 186 (47.8%)

## 6. Top 10 Most Common Primary Patterns

| Rank | Pattern | Count | Percentage |
|------|---------|-------|------------|
| 1 | symmetry/flip/rotate | 105 | 27.0% |
| 2 | other | 80 | 20.6% |
| 3 | reduce | 67 | 17.2% |
| 4 | color_remap | 37 | 9.5% |
| 5 | tiling | 30 | 7.7% |
| 6 | color_fill | 29 | 7.5% |
| 7 | crop | 22 | 5.7% |
| 8 | expand | 6 | 1.5% |
| 9 | trim_border | 6 | 1.5% |
| 10 | scalar_output | 6 | 1.5% |

## 7. Recommended Primitive Priority Order

| Priority | Primitive | Rationale |
|----------|-----------|----------|
| 1 | **Tile / Replicate pattern** | Covers 30 tasks (7.7%). Output dimensions are integer multiples of input. Needed for pattern expansion tasks. |
| 2 | **Crop / Subregion extraction** | Covers 26 tasks (6.7%). Output is an exact subregion of input. Needed to focus on object of interest. |
| 3 | **Color remap (systematic color replacement per object)** | Covers 37 tasks (9.5%). Shape-preserving, output colors ⊆ input. Tasks recolor individual objects while keeping layout. |
| 4 | **Color fill / Region coloring** | Covers 29 tasks (7.5%). Shape-preserving, adds new colors. Tasks fill regions with colors not present in input. |
| 5 | **Trim / Remove border** | Covers 8 tasks (2.1%). Output is slightly smaller, removing uniform borders. Need border-detection then slice. |
| 6 | **Symmetry composition (multi-flip, rotation combinations, identity check)** | Covers 105+ tasks. Same shape + same colors but existing single flip/rotate don't suffice. Need compositions or conditional symmetry. |
| 7 | **Counting / Scalar output** | Covers 6 tasks (1.5%). Output is 1×1. Need count-object, count-color, or measure primitive. |
| 8 | **Reduce / Downsample (pattern extraction)** | Covers 75+ tasks. Output smaller but not a simple subregion. These extract summaries or transformed sub-patterns. |
| 9 | **Expand / Pad** | Covers 6 tasks. Output larger but not simple tiling. May add margins or extend patterns. |

## 8. Summary — Top 3 Most Frequent Primitive Needs

1. **Tile / Replicate pattern** — Covers 30 tasks (7.7%). Output dimensions are integer multiples of input. Needed for pattern expansion tasks.

2. **Crop / Subregion extraction** — Covers 26 tasks (6.7%). Output is an exact subregion of input. Needed to focus on object of interest.

3. **Color remap (systematic color replacement per object)** — Covers 37 tasks (9.5%). Shape-preserving, output colors ⊆ input. Tasks recolor individual objects while keeping layout.

## 9. Surprising Findings

- **27% of uncovered tasks have same shape and same colors** — these are likely symmetry/flip/rotate tasks that existing single primitives didn't solve because the exact composition or parameter wasn't found by the search.
- **20.6% of tasks have shape-preserving same-colors patterns** that weren't classified into any specific category — these may involve pixel-level operations (move object, copy region, etc.) that don't change shape or overall palette.
- **Only 0 uncovered tasks involve non-square transpose** — the dynamic-shape transpose in v0 already covers all transpose needs (or transpose isn't needed for uncovered tasks).
- **Color fill tasks (adding new colors) are nearly as common as color remap tasks** — the need to introduce new colors (like filling enclosed regions) is a distinct pattern from recoloring existing colors.
- **Tiling and crop together account for ~14% of uncovered tasks** — these are the two most common shape-changing patterns, making them high-value Tier-2 targets.

## 10. Per-Task Classification

| Task ID | Primary Pattern | Tags |
|---------|----------------|------|
| 1 | tiling | tiling |
| 2 | other | other |
| 3 | expand | expand |
| 4 | symmetry/flip/rotate | symmetry/flip/rotate |
| 5 | symmetry/flip/rotate | symmetry/flip/rotate |
| 6 | reduce | reduce |
| 7 | color_remap | color_remap |
| 8 | symmetry/flip/rotate | symmetry/flip/rotate |
| 9 | symmetry/flip/rotate | symmetry/flip/rotate |
| 10 | color_fill | color_fill |
| 11 | color_remap | color_remap |
| 12 | symmetry/flip/rotate | symmetry/flip/rotate |
| 13 | symmetry/flip/rotate | symmetry/flip/rotate |
| 14 | crop | crop |
| 15 | other | other |
| 16 | color_fill | color_fill |
| 17 | color_remap | color_remap |
| 18 | symmetry/flip/rotate | symmetry/flip/rotate |
| 19 | tiling | tiling |
| 20 | symmetry/flip/rotate | symmetry/flip/rotate |
| 21 | crop | crop |
| 22 | reduce | reduce |
| 23 | color_fill | color_fill |
| 24 | symmetry/flip/rotate | symmetry/flip/rotate |
| 25 | color_remap | color_remap |
| 26 | reduce | reduce |
| 27 | other | other |
| 28 | symmetry/flip/rotate | symmetry/flip/rotate |
| 29 | crop | crop |
| 30 | symmetry/flip/rotate | symmetry/flip/rotate |
| 31 | crop | crop |
| 32 | symmetry/flip/rotate | symmetry/flip/rotate |
| 33 | symmetry/flip/rotate | symmetry/flip/rotate |
| 34 | color_remap | color_remap |
| 35 | symmetry/flip/rotate | symmetry/flip/rotate |
| 36 | crop | crop |
| 37 | symmetry/flip/rotate | symmetry/flip/rotate |
| 38 | reduce | reduce |
| 39 | crop | crop |
| 40 | color_remap | color_remap |
| 41 | symmetry/flip/rotate | symmetry/flip/rotate |
| 42 | other | other |
| 43 | other | other |
| 44 | symmetry/flip/rotate | symmetry/flip/rotate |
| 45 | symmetry/flip/rotate | symmetry/flip/rotate |
| 46 | trim_border | trim_border, reduce |
| 47 | other | other |
| 48 | scalar_output | scalar_output, crop |
| 49 | crop | crop |
| 50 | other | other |
| 51 | symmetry/flip/rotate | symmetry/flip/rotate |
| 52 | color_fill | color_fill |
| 53 | symmetry/flip/rotate | symmetry/flip/rotate |
| 54 | symmetry/flip/rotate | symmetry/flip/rotate |
| 55 | other | other |
| 56 | scalar_output | scalar_output, trim_border, reduce |
| 57 | reduce | reduce |
| 58 | other | other |
| 59 | symmetry/flip/rotate | symmetry/flip/rotate |
| 60 | other | other |
| 61 | color_remap | color_remap |
| 62 | color_fill | color_fill |
| 63 | other | other |
| 64 | symmetry/flip/rotate | symmetry/flip/rotate |
| 65 | crop | crop |
| 66 | symmetry/flip/rotate | symmetry/flip/rotate |
| 67 | reduce | reduce |
| 68 | color_remap | color_remap |
| 69 | color_remap | color_remap |
| 70 | other | other |
| 71 | color_remap | color_remap |
| 72 | reduce | reduce |
| 73 | symmetry/flip/rotate | symmetry/flip/rotate |
| 74 | color_remap | color_remap |
| 75 | color_remap | color_remap |
| 76 | symmetry/flip/rotate | symmetry/flip/rotate |
| 77 | other | other |
| 78 | symmetry/flip/rotate | symmetry/flip/rotate |
| 79 | crop | crop |
| 80 | symmetry/flip/rotate | symmetry/flip/rotate |
| 81 | other | other |
| 82 | symmetry/flip/rotate | symmetry/flip/rotate |
| 83 | tiling | tiling |
| 84 | other | other |
| 85 | symmetry/flip/rotate | symmetry/flip/rotate |
| 86 | symmetry/flip/rotate | symmetry/flip/rotate |
| 88 | reduce | reduce |
| 89 | symmetry/flip/rotate | symmetry/flip/rotate |
| 90 | other | other |
| 91 | crop | crop |
| 92 | symmetry/flip/rotate | symmetry/flip/rotate |
| 93 | color_remap | color_remap |
| 94 | other | other |
| 95 | other | other |
| 96 | reduce | reduce |
| 97 | symmetry/flip/rotate | symmetry/flip/rotate |
| 98 | symmetry/flip/rotate | symmetry/flip/rotate |
| 99 | symmetry/flip/rotate | symmetry/flip/rotate |
| 100 | reduce | reduce |
| 101 | symmetry/flip/rotate | symmetry/flip/rotate |
| 102 | other | other |
| 103 | scalar_output | scalar_output, trim_border, reduce |
| 104 | tiling | tiling |
| 105 | other | other |
| 106 | tiling | tiling |
| 107 | tiling | tiling |
| 108 | tiling | tiling |
| 109 | trim_border | trim_border, reduce |
| 110 | color_remap | color_remap |
| 111 | crop | crop |
| 112 | symmetry/flip/rotate | symmetry/flip/rotate |
| 113 | symmetry/flip/rotate | symmetry/flip/rotate |
| 114 | expand | expand |
| 115 | reduce | reduce |
| 116 | tiling | tiling |
| 117 | symmetry/flip/rotate | symmetry/flip/rotate |
| 118 | other | other |
| 119 | other | other |
| 120 | other | other |
| 121 | reduce | reduce |
| 122 | symmetry/flip/rotate | symmetry/flip/rotate |
| 123 | tiling | tiling |
| 124 | expand | expand |
| 125 | other | other |
| 126 | other | other |
| 127 | color_fill | color_fill |
| 128 | symmetry/flip/rotate | symmetry/flip/rotate |
| 130 | reduce | reduce |
| 131 | other | other |
| 132 | symmetry/flip/rotate | symmetry/flip/rotate |
| 133 | symmetry/flip/rotate | symmetry/flip/rotate |
| 134 | reduce | reduce |
| 135 | crop | crop |
| 136 | symmetry/flip/rotate | symmetry/flip/rotate |
| 137 | symmetry/flip/rotate | symmetry/flip/rotate |
| 138 | reduce | reduce |
| 139 | other | other |
| 141 | symmetry/flip/rotate | symmetry/flip/rotate |
| 142 | tiling | tiling |
| 143 | color_remap | color_remap |
| 144 | reduce | reduce |
| 145 | other | other |
| 146 | reduce | reduce |
| 147 | other | other |
| 148 | other | other |
| 149 | reduce | reduce |
| 151 | other | other |
| 152 | tiling | tiling |
| 153 | reduce | reduce |
| 154 | symmetry/flip/rotate | symmetry/flip/rotate |
| 156 | other | other |
| 157 | color_fill | color_fill |
| 158 | symmetry/flip/rotate | symmetry/flip/rotate |
| 159 | reduce | reduce |
| 160 | other | other |
| 161 | color_remap | color_remap |
| 162 | other | other |
| 163 | color_remap | color_remap |
| 164 | tiling | tiling |
| 165 | symmetry/flip/rotate | symmetry/flip/rotate |
| 166 | other | other |
| 167 | color_fill | color_fill |
| 168 | symmetry/flip/rotate | symmetry/flip/rotate |
| 169 | color_fill | color_fill |
| 170 | reduce | reduce |
| 171 | other | other |
| 172 | tiling | tiling |
| 173 | symmetry/flip/rotate | symmetry/flip/rotate |
| 174 | crop | crop |
| 175 | color_remap | color_remap |
| 176 | other | other |
| 177 | reduce | reduce |
| 178 | reduce | reduce |
| 180 | reduce | reduce |
| 181 | symmetry/flip/rotate | symmetry/flip/rotate |
| 182 | symmetry/flip/rotate | symmetry/flip/rotate |
| 183 | reduce | reduce |
| 184 | reduce | reduce |
| 185 | reduce | reduce |
| 186 | color_fill | color_fill |
| 187 | color_fill | color_fill |
| 188 | reduce | reduce |
| 189 | trim_border | trim_border, reduce |
| 190 | symmetry/flip/rotate | symmetry/flip/rotate |
| 191 | symmetry/flip/rotate | symmetry/flip/rotate |
| 192 | color_remap | color_remap |
| 193 | symmetry/flip/rotate | symmetry/flip/rotate |
| 194 | tiling | tiling |
| 195 | reduce | reduce |
| 196 | other | other |
| 197 | symmetry/flip/rotate | symmetry/flip/rotate |
| 198 | color_fill | color_fill |
| 199 | other | other |
| 200 | other | other |
| 201 | reduce | reduce |
| 202 | symmetry/flip/rotate | symmetry/flip/rotate |
| 203 | symmetry/flip/rotate | symmetry/flip/rotate |
| 204 | other | other |
| 205 | reduce | reduce |
| 206 | color_remap | color_remap |
| 207 | crop | crop |
| 208 | symmetry/flip/rotate | symmetry/flip/rotate |
| 209 | reduce | reduce |
| 210 | tiling | tiling |
| 211 | tiling | tiling |
| 212 | symmetry/flip/rotate | symmetry/flip/rotate |
| 213 | reduce | reduce |
| 214 | color_remap | color_remap |
| 215 | symmetry/flip/rotate | symmetry/flip/rotate |
| 216 | crop | crop |
| 217 | symmetry/flip/rotate | symmetry/flip/rotate |
| 218 | reduce | reduce |
| 219 | other | other |
| 220 | other | other |
| 221 | tiling | tiling |
| 222 | color_remap | color_remap |
| 223 | tiling | tiling |
| 224 | symmetry/flip/rotate | symmetry/flip/rotate |
| 225 | symmetry/flip/rotate | symmetry/flip/rotate |
| 226 | other | other |
| 227 | reduce | reduce |
| 228 | symmetry/flip/rotate | symmetry/flip/rotate |
| 229 | color_fill | color_fill |
| 230 | other | other |
| 231 | tiling | tiling |
| 232 | other | other |
| 233 | reduce | reduce |
| 234 | symmetry/flip/rotate | symmetry/flip/rotate |
| 235 | reduce | reduce |
| 236 | reduce | reduce |
| 237 | symmetry/flip/rotate | symmetry/flip/rotate |
| 238 | reduce | reduce |
| 239 | expand | expand |
| 240 | symmetry/flip/rotate | symmetry/flip/rotate |
| 242 | reduce | reduce |
| 243 | symmetry/flip/rotate | symmetry/flip/rotate |
| 244 | reduce | reduce |
| 245 | symmetry/flip/rotate | symmetry/flip/rotate |
| 246 | other | other |
| 247 | reduce | reduce |
| 248 | symmetry/flip/rotate | symmetry/flip/rotate |
| 249 | tiling | tiling |
| 250 | symmetry/flip/rotate | symmetry/flip/rotate |
| 251 | other | other |
| 252 | other | other |
| 253 | reduce | reduce |
| 254 | color_fill | color_fill |
| 255 | other | other |
| 256 | other | other |
| 257 | reduce | reduce |
| 258 | other | other |
| 259 | reduce | reduce |
| 260 | color_remap | color_remap |
| 261 | color_fill | color_fill |
| 262 | color_fill | color_fill |
| 263 | reduce | reduce |
| 264 | reduce | reduce |
| 265 | other | other |
| 266 | color_fill | color_fill |
| 267 | color_remap | color_remap |
| 268 | other | other |
| 269 | tiling | tiling |
| 270 | symmetry/flip/rotate | symmetry/flip/rotate |
| 271 | crop | crop |
| 272 | other | other |
| 273 | other | other |
| 274 | reduce | reduce |
| 275 | expand | expand |
| 277 | color_fill | color_fill |
| 278 | other | other |
| 279 | other | other |
| 280 | symmetry/flip/rotate | symmetry/flip/rotate |
| 281 | color_remap | color_remap |
| 282 | other | other |
| 283 | color_fill | color_fill |
| 284 | symmetry/flip/rotate | symmetry/flip/rotate |
| 285 | symmetry/flip/rotate | symmetry/flip/rotate |
| 286 | symmetry/flip/rotate | symmetry/flip/rotate |
| 287 | color_remap | color_remap |
| 288 | symmetry/flip/rotate | symmetry/flip/rotate |
| 289 | tiling | tiling |
| 290 | reduce | reduce |
| 291 | scalar_output | scalar_output, crop |
| 292 | other | other |
| 293 | symmetry/flip/rotate | symmetry/flip/rotate |
| 294 | other | other |
| 295 | tiling | tiling |
| 296 | reduce | reduce |
| 297 | color_remap | color_remap |
| 298 | symmetry/flip/rotate | symmetry/flip/rotate |
| 299 | other | other |
| 300 | crop | crop |
| 301 | symmetry/flip/rotate | symmetry/flip/rotate |
| 302 | other | other |
| 303 | other | other |
| 304 | tiling | tiling |
| 305 | color_remap | color_remap |
| 306 | symmetry/flip/rotate | symmetry/flip/rotate |
| 307 | tiling | tiling |
| 308 | reduce | reduce |
| 310 | crop | crop |
| 311 | tiling | tiling |
| 312 | color_remap | color_remap |
| 313 | color_remap | color_remap |
| 314 | symmetry/flip/rotate | symmetry/flip/rotate |
| 315 | tiling | tiling |
| 316 | reduce | reduce |
| 317 | color_fill | color_fill |
| 318 | reduce | reduce |
| 319 | crop | crop |
| 320 | other | other |
| 321 | reduce | reduce |
| 322 | symmetry/flip/rotate | symmetry/flip/rotate |
| 323 | other | other |
| 324 | symmetry/flip/rotate | symmetry/flip/rotate |
| 325 | reduce | reduce |
| 326 | crop | crop |
| 327 | tiling | tiling |
| 328 | symmetry/flip/rotate | symmetry/flip/rotate |
| 329 | color_remap | color_remap |
| 330 | color_fill | color_fill |
| 331 | other | other |
| 332 | other | other |
| 333 | symmetry/flip/rotate | symmetry/flip/rotate |
| 334 | trim_border | trim_border, reduce |
| 335 | other | other |
| 336 | other | other |
| 338 | color_fill | color_fill |
| 339 | reduce | reduce |
| 340 | color_remap | color_remap |
| 341 | other | other |
| 342 | color_remap | color_remap |
| 343 | symmetry/flip/rotate | symmetry/flip/rotate |
| 344 | color_fill | color_fill |
| 345 | symmetry/flip/rotate | symmetry/flip/rotate |
| 346 | scalar_output | scalar_output, crop |
| 347 | trim_border | trim_border, reduce |
| 348 | other | other |
| 349 | other | other |
| 350 | other | other |
| 351 | reduce | reduce |
| 352 | other | other |
| 353 | symmetry/flip/rotate | symmetry/flip/rotate |
| 354 | color_remap | color_remap |
| 355 | scalar_output | scalar_output, crop |
| 356 | symmetry/flip/rotate | symmetry/flip/rotate |
| 357 | color_fill | color_fill |
| 358 | symmetry/flip/rotate | symmetry/flip/rotate |
| 359 | color_remap | color_remap |
| 360 | reduce | reduce |
| 361 | symmetry/flip/rotate | symmetry/flip/rotate |
| 362 | color_remap | color_remap |
| 363 | symmetry/flip/rotate | symmetry/flip/rotate |
| 364 | color_fill | color_fill |
| 365 | crop | crop |
| 366 | reduce | reduce |
| 367 | other | other |
| 368 | color_remap | color_remap |
| 369 | color_fill | color_fill |
| 370 | symmetry/flip/rotate | symmetry/flip/rotate |
| 371 | other | other |
| 372 | reduce | reduce |
| 373 | symmetry/flip/rotate | symmetry/flip/rotate |
| 374 | color_fill | color_fill |
| 375 | symmetry/flip/rotate | symmetry/flip/rotate |
| 376 | expand | expand |
| 377 | reduce | reduce |
| 378 | symmetry/flip/rotate | symmetry/flip/rotate |
| 379 | symmetry/flip/rotate | symmetry/flip/rotate |
| 381 | other | other |
| 382 | symmetry/flip/rotate | symmetry/flip/rotate |
| 383 | symmetry/flip/rotate | symmetry/flip/rotate |
| 384 | reduce | reduce |
| 385 | symmetry/flip/rotate | symmetry/flip/rotate |
| 386 | reduce | reduce |
| 387 | other | other |
| 388 | tiling | tiling |
| 389 | color_fill | color_fill |
| 390 | symmetry/flip/rotate | symmetry/flip/rotate |
| 391 | reduce | reduce |
| 392 | color_fill | color_fill |
| 393 | reduce | reduce |
| 394 | crop | crop |
| 395 | trim_border | trim_border, reduce |
| 396 | reduce | reduce |
| 397 | other | other |
| 398 | tiling | tiling |
| 399 | other_shape_change | other |
| 400 | reduce | reduce |
