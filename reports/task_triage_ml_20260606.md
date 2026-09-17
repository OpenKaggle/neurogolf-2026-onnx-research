# Task Triage ML - 2026-06-06

## Snapshot

- Anchor: `v4_plus10_compiler2`, LB `6260.16`, local dossier sum `6260.1627`.
- Registry: `reports/task_dossiers_20260606.csv` and `reports/task_dossiers_20260606.json`.
- Parent priority queue is locked as ranks 1-13; ranks 14+ are the transparent model extension.
- Model balances score upside, pseudo-hidden status, family reuse, graphizability, dossier priority, and hidden risk.
- `sklearn`/`torch` were not used; available labels are too sparse for a model that would be more trustworthy than this auditable baseline.
- Top 50 has `2` stress-clean tasks, `2` stress-mixed tasks, and `5` tasks with hidden risk >= 0.70.

## Method

Gross upside uses public gap estimates when available, otherwise cost-model savings from `score=max(0,25-ln(cost+1))` and safe best-seen deltas. Success probability is a calibrated heuristic from pseudo-hidden status, known prototypes/builders, graphizability, and risk. The queue score locks the parent 13-task order and then ranks the rest by expected value per effort with reuse and graphizability bonuses.

## Top Families

| family | tasks | top tasks | top10 EV | reuse | graph | risk | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| symmetry/flip/rotate | 105 | task285,task191,task101,task018,task158,task076,task286,task243,task054,task066 | 8.547 | 1.000 | 0.718 | 0.571 | Prioritize invariant stress suites and reusable D4/placement compiler pieces. |
| reduce | 67 | task366,task233,task096,task396,task264,task209,task205,task153,task159,task088 | 7.712 | 0.989 | 0.526 | 0.534 | Invest in bbox/component selection primitives; require pseudo-hidden before swaps. |
| other | 80 | task255,task204,task219,task367,task077,task118,task251,task387,task145,task303 | 4.782 | 0.962 | 0.420 | 0.590 | Use dossier next_allowed_action per task. |
| color_remap | 37 | task025,task071,task110,task034,task175,task281,task313,task161,task206,task011 | 2.733 | 0.865 | 0.726 | 0.555 | Use color-bijection stress tests before trusting public or local-only gains. |
| color_fill | 29 | task023,task187,task198,task374,task392,task062,task364,task338,task010,task052 | 1.751 | 0.809 | 0.440 | 0.521 | Develop mask/flood-fill contracts before graph staging. |
| crop | 22 | task216,task036,task300,task014,task029,task091,task065,task021,task365,task049 | 1.585 | 0.752 | 0.580 | 0.563 | Validate bbox detection under translations and distractors. |
| tiling | 30 | task398,task019,task107,task221,task231,task289,task001,task123,task269,task295 | 0.828 | 0.736 | 0.680 | 0.500 | Use dossier next_allowed_action per task. |
| expand | 6 | task376,task275,task239,task003,task114,task124 | 0.409 | 0.417 | 0.500 | 0.543 | Use dossier next_allowed_action per task. |
| trim_border | 6 | task109,task046,task395,task334,task347,task189 | 0.286 | 0.417 | 0.620 | 0.513 | Use dossier next_allowed_action per task. |
| covered_or_unclassified | 11 | task179,task241,task276,task309,task337,task150,task155,task087,task140,task380 | 0.000 | 0.533 | 0.380 | 0.547 | Use dossier next_allowed_action per task. |

## Directions

| direction | tasks | top tasks | top12 EV | risk | graph | gate |
| --- | --- | --- | --- | --- | --- | --- |
| semantic_rebuild | 18 | task255,task101,task158,task076,task286,task054,task133,task096,task383,task358,task367,task396 | 8.031 | 0.618 | 0.577 | Prototype and stress-test first; no ONNX until pseudo-hidden story improves. |
| compact_compiler | 2 | task366,task285 | 5.974 | 0.220 | 0.790 | Set target cost, build exact graph, verify visible/stress equivalence. |
| pseudo_hidden_first | 11 | task233,task191,task018,task243,task204,task363,task182,task153,task387,task149,task200 | 5.019 | 0.595 | 0.671 | Write falsifiable contracts before considering any replacement. |
| family_dsl | 161 | task066,task382,task370,task284,task110,task023,task044,task175,task216,task128,task280,task234 | 4.352 | 0.530 | 0.640 | Mine family primitive, then isolated eval against anchor cost. |
| audit_before_swap | 142 | task025,task064,task173,task202,task005,task013,task085,task071,task034,task208,task020,task165 | 3.829 | 0.563 | 0.695 | No raw public-source swap; prove semantics and hidden safety. |
| defer | 55 | task219,task251,task131,task002,task027,task278,task196,task335,task252,task109,task055,task166 | 1.802 | 0.522 | 0.402 | Revisit through family-level compiler work or new evidence. |
| compiler_golf | 10 | task205,task215,task192,task092,task333,task138,task377,task374,task301,task376 | 0.909 | 0.546 | 0.568 | Exact-equivalence proof and structural audit before batching. |
| frozen | 1 | task319 | 0.127 | 0.980 | 0.740 | Leave untouched unless a new hidden-safety theory appears. |

## Ranked Queue 1-50

| rank | task | dir | family | cost | gross | EV | risk | graph | reuse | pseudo | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | task018 | pseudo_hidden_first | symmetry/flip/rotate | 476,185 | 3.520 | 0.584 | 0.98 | 0.82 | 1.00 | not_run | parent-priority / public gap +3.52 / cost rank 5 / best-seen +13.07 / builder / Do not cost-only replace; r... |
| 2 | task285 | compact_compiler | symmetry/flip/rotate | 395,468 | 3.620 | 2.729 | 0.24 | 0.86 | 1.00 | stress_clean | parent-priority / public gap +3.02 / cost rank 6 / stress clean / best-seen +0.82 / prototype / Compile a c... |
| 3 | task255 | semantic_rebuild | other | 1,592,864 | 5.071 | 1.678 | 0.86 | 0.40 | 0.99 | stress_mixed | parent-priority / public gap +4.57 / cost rank 1 / stress mixed / best-seen +2.39 / prototype / Retire brit... |
| 4 | task133 | semantic_rebuild | symmetry/flip/rotate | 196,680 | 1.787 | 0.554 | 0.86 | 0.64 | 1.00 | stress_mixed | parent-priority / public big-gap set / cost rank 9 / stress mixed / prototype / graph risk 2 / Revise seman... |
| 5 | task366 | compact_compiler | reduce | 830,720 | 4.370 | 3.245 | 0.20 | 0.72 | 1.00 | stress_clean | parent-priority / public gap +3.77 / cost rank 3 / stress clean / prototype / builder / Cost-audit builder ... |
| 6 | task233 | pseudo_hidden_first | reduce | 1,450,762 | 4.580 | 1.633 | 0.73 | 0.58 | 1.00 | not_run | parent-priority / public gap +3.98 / cost rank 2 / best-seen +2.96 / prototype / builder / graph risk 3 / R... |
| 7 | task025 | audit_before_swap | color_remap | 140,093 | 1.050 | 0.342 | 0.59 | 0.84 | 0.91 | not_run | parent-priority / public bottom15 / cost rank 17 / best-seen +0.45 / builder / Require semantic proof and p... |
| 8 | task173 | audit_before_swap | symmetry/flip/rotate | 175,243 | 1.718 | 0.393 | 0.63 | 0.66 | 1.00 | not_run | parent-priority / public bottom15 / cost rank 12 / Require semantic proof and pseudo-hidden audit; no raw p... |
| 9 | task243 | pseudo_hidden_first | symmetry/flip/rotate | 108,967 | 1.073 | 0.382 | 0.52 | 0.78 | 1.00 | not_run | parent-priority / public big-gap set / best-seen +0.47 / builder / Run or extend pseudo-hidden audit before... |
| 10 | task158 | semantic_rebuild | symmetry/flip/rotate | 191,304 | 1.771 | 0.543 | 0.52 | 0.66 | 1.00 | not_run | parent-priority / public bottom15 / cost rank 10 / Open semantic-factory pass: inspect examples, write prot... |
| 11 | task054 | semantic_rebuild | symmetry/flip/rotate | 150,548 | 1.627 | 0.499 | 0.52 | 0.66 | 1.00 | not_run | parent-priority / public bottom15 / cost rank 15 / Open semantic-factory pass: inspect examples, write prot... |
| 12 | task286 | semantic_rebuild | symmetry/flip/rotate | 160,931 | 1.667 | 0.511 | 0.52 | 0.66 | 1.00 | not_run | parent-priority / public bottom15 / cost rank 13 / Open semantic-factory pass: inspect examples, write prot... |
| 13 | task096 | semantic_rebuild | reduce | 184,630 | 2.438 | 0.746 | 0.68 | 0.48 | 0.98 | not_run | parent-priority / cost rank 11 / best-seen +1.84 / Open semantic-factory pass: inspect examples, write prot... |
| 14 | task191 | pseudo_hidden_first | symmetry/flip/rotate | 691,342 | 2.685 | 0.993 | 0.58 | 0.82 | 1.00 | not_run | cost rank 4 / best-seen +2.09 / prototype / builder / compiler +0.03 / Run or extend pseudo-hidden audit be... |
| 15 | task101 | semantic_rebuild | symmetry/flip/rotate | 374,431 | 2.338 | 0.788 | 0.58 | 0.66 | 1.00 | not_run | cost rank 7 / best-seen +1.74 / Open semantic-factory pass: inspect examples, write prototype, then stress-... |
| 16 | task076 | semantic_rebuild | symmetry/flip/rotate | 326,593 | 2.800 | 0.863 | 0.73 | 0.60 | 1.00 | not_run | cost rank 8 / best-seen +2.20 / graph risk 3 / Open semantic-factory pass: inspect examples, write prototyp... |
| 17 | task066 | family_dsl | symmetry/flip/rotate | 99,734 | 2.139 | 0.656 | 0.52 | 0.66 | 1.00 | not_run | best-seen +1.54 / Route to family DSL/compiler search; record any exact rule. |
| 18 | task064 | audit_before_swap | symmetry/flip/rotate | 114,153 | 1.461 | 0.416 | 0.59 | 0.78 | 1.00 | not_run | cost rank 27 / builder / Require semantic proof and pseudo-hidden audit; no raw public-source swap. |
| 19 | task204 | pseudo_hidden_first | other | 118,846 | 1.485 | 0.473 | 0.52 | 0.56 | 0.99 | not_run | cost rank 23 / prototype / builder / Run or extend pseudo-hidden audit before using the builder. |
| 20 | task383 | semantic_rebuild | symmetry/flip/rotate | 129,060 | 1.535 | 0.470 | 0.52 | 0.66 | 1.00 | not_run | cost rank 21 / Open semantic-factory pass: inspect examples, write prototype, then stress-test. |
| 21 | task358 | semantic_rebuild | symmetry/flip/rotate | 112,263 | 1.451 | 0.445 | 0.52 | 0.66 | 1.00 | not_run | cost rank 30 / Open semantic-factory pass: inspect examples, write prototype, then stress-test. |
| 22 | task202 | audit_before_swap | symmetry/flip/rotate | 125,720 | 1.519 | 0.367 | 0.55 | 0.66 | 1.00 | not_run | cost rank 22 / Require semantic proof and pseudo-hidden audit; no raw public-source swap. |
| 23 | task005 | audit_before_swap | symmetry/flip/rotate | 117,225 | 1.477 | 0.357 | 0.55 | 0.66 | 1.00 | not_run | cost rank 24 / Require semantic proof and pseudo-hidden audit; no raw public-source swap. |
| 24 | task013 | audit_before_swap | symmetry/flip/rotate | 116,882 | 1.475 | 0.357 | 0.55 | 0.66 | 1.00 | not_run | cost rank 25 / Require semantic proof and pseudo-hidden audit; no raw public-source swap. |
| 25 | task085 | audit_before_swap | symmetry/flip/rotate | 113,178 | 1.456 | 0.352 | 0.55 | 0.66 | 1.00 | not_run | cost rank 29 / Require semantic proof and pseudo-hidden audit; no raw public-source swap. |
| 26 | task071 | audit_before_swap | color_remap | 129,780 | 1.538 | 0.368 | 0.55 | 0.72 | 0.86 | not_run | cost rank 20 / Require semantic proof and pseudo-hidden audit; no raw public-source swap. |
| 27 | task219 | defer | other | 82,303 | 1.136 | 0.427 | 0.62 | 0.60 | 0.99 | not_run | best-seen +0.54 / prototype / builder / Avoid visible-family lookup staging; generalize the closed Python r... |
| 28 | task367 | semantic_rebuild | other | 143,603 | 1.559 | 0.498 | 0.52 | 0.36 | 0.94 | not_run | cost rank 16 / best-seen +0.96 / Open semantic-factory pass: inspect examples, write prototype, then stress... |
| 29 | task363 | pseudo_hidden_first | symmetry/flip/rotate | 87,112 | 1.074 | 0.388 | 0.62 | 0.80 | 1.00 | not_run | best-seen +0.47 / prototype / builder / graph risk 2 / Run or extend pseudo-hidden audit before using the b... |
| 30 | task382 | family_dsl | symmetry/flip/rotate | 107,354 | 1.424 | 0.380 | 0.52 | 0.66 | 1.00 | not_run | Route to family DSL/compiler search; record any exact rule. |
| 31 | task370 | family_dsl | symmetry/flip/rotate | 107,206 | 1.423 | 0.379 | 0.52 | 0.66 | 1.00 | not_run | Route to family DSL/compiler search; record any exact rule. |
| 32 | task396 | semantic_rebuild | reduce | 131,688 | 1.547 | 0.436 | 0.58 | 0.48 | 0.98 | not_run | cost rank 19 / compiler +0.07 / Open semantic-factory pass: inspect examples, write prototype, then stress-... |
| 33 | task264 | semantic_rebuild | reduce | 139,356 | 1.161 | 0.385 | 0.52 | 0.48 | 0.98 | not_run | cost rank 18 / best-seen +0.56 / Open semantic-factory pass: inspect examples, write prototype, then stress... |
| 34 | task284 | family_dsl | symmetry/flip/rotate | 101,133 | 1.388 | 0.370 | 0.52 | 0.66 | 1.00 | not_run | Route to family DSL/compiler search; record any exact rule. |
| 35 | task077 | semantic_rebuild | other | 157,494 | 1.654 | 0.463 | 0.52 | 0.36 | 0.94 | not_run | cost rank 14 / Open semantic-factory pass: inspect examples, write prototype, then stress-test. |
| 36 | task110 | family_dsl | color_remap | 107,857 | 1.427 | 0.377 | 0.52 | 0.72 | 0.86 | not_run | Route to family DSL/compiler search; record any exact rule. |
| 37 | task209 | semantic_rebuild | reduce | 113,356 | 0.984 | 0.326 | 0.52 | 0.48 | 0.98 | not_run | cost rank 28 / best-seen +0.38 / compiler +0.08 / Open semantic-factory pass: inspect examples, write proto... |
| 38 | task205 | compiler_golf | reduce | 103,313 | 1.401 | 0.352 | 0.52 | 0.48 | 0.98 | not_run | compiler +0.08 / Route to family DSL/compiler search; record any exact rule. |
| 39 | task034 | audit_before_swap | color_remap | 106,796 | 1.421 | 0.340 | 0.55 | 0.72 | 0.86 | not_run | Require semantic proof and pseudo-hidden audit; no raw public-source swap. |
| 40 | task023 | family_dsl | color_fill | 72,612 | 1.802 | 0.504 | 0.52 | 0.44 | 0.81 | not_run | best-seen +1.20 / Route to family DSL/compiler search; record any exact rule. |
| 41 | task044 | family_dsl | symmetry/flip/rotate | 43,947 | 0.923 | 0.283 | 0.52 | 0.66 | 1.00 | not_run | best-seen +0.32 / Route to family DSL/compiler search; record any exact rule. |
| 42 | task175 | family_dsl | color_remap | 87,620 | 1.586 | 0.435 | 0.68 | 0.66 | 0.86 | not_run | best-seen +0.99 / graph risk 2 / Route to family DSL/compiler search; record any exact rule. |
| 43 | task182 | pseudo_hidden_first | symmetry/flip/rotate | 47,587 | 0.546 | 0.156 | 0.52 | 0.78 | 1.00 | not_run | builder / Run or extend pseudo-hidden audit before using the builder. |
| 44 | task216 | family_dsl | crop | 103,812 | 1.404 | 0.357 | 0.48 | 0.58 | 0.75 | not_run | Route to family DSL/compiler search; record any exact rule. |
| 45 | task153 | pseudo_hidden_first | reduce | 21,618 | 0.771 | 0.240 | 0.52 | 0.60 | 1.00 | not_run | best-seen +0.63 / builder / Run or extend pseudo-hidden audit before using the builder. |
| 46 | task128 | family_dsl | symmetry/flip/rotate | 95,024 | 0.788 | 0.210 | 0.52 | 0.66 | 1.00 | not_run | Route to family DSL/compiler search; record any exact rule. |
| 47 | task280 | family_dsl | symmetry/flip/rotate | 90,506 | 0.771 | 0.206 | 0.52 | 0.66 | 1.00 | not_run | Route to family DSL/compiler search; record any exact rule. |
| 48 | task187 | semantic_rebuild | color_fill | 105,313 | 1.413 | 0.386 | 0.56 | 0.44 | 0.81 | not_run | public bottom15 / Open semantic-factory pass: inspect examples, write prototype, then stress-test. |
| 49 | task234 | family_dsl | symmetry/flip/rotate | 81,703 | 0.735 | 0.196 | 0.52 | 0.66 | 1.00 | not_run | Route to family DSL/compiler search; record any exact rule. |
| 50 | task208 | audit_before_swap | symmetry/flip/rotate | 92,408 | 0.778 | 0.188 | 0.55 | 0.66 | 1.00 | not_run | Require semantic proof and pseudo-hidden audit; no raw public-source swap. |

## Next 25-50 Focus

These are the extension slots after the most urgent public-tail and parent-priority work. They skew toward graphizable, reusable families unless hidden-safety risk says audit first.

| rank | task | dir | family | cost | best delta | EV | risk | graph | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | task085 | audit_before_swap | symmetry/flip/rotate | 113,178 | 0.000 | 0.352 | 0.55 | 0.66 | cost rank 29 / Require semantic proof and pseudo-hidden audit; no raw public-source swap. |
| 26 | task071 | audit_before_swap | color_remap | 129,780 | 0.000 | 0.368 | 0.55 | 0.72 | cost rank 20 / Require semantic proof and pseudo-hidden audit; no raw public-source swap. |
| 27 | task219 | defer | other | 82,303 | 0.536 | 0.427 | 0.62 | 0.60 | best-seen +0.54 / prototype / builder / Avoid visible-family lookup staging; generalize the closed Python r... |
| 28 | task367 | semantic_rebuild | other | 143,603 | 0.959 | 0.498 | 0.52 | 0.36 | cost rank 16 / best-seen +0.96 / Open semantic-factory pass: inspect examples, write prototype, then stress... |
| 29 | task363 | pseudo_hidden_first | symmetry/flip/rotate | 87,112 | 0.474 | 0.388 | 0.62 | 0.80 | best-seen +0.47 / prototype / builder / graph risk 2 / Run or extend pseudo-hidden audit before using the b... |
| 30 | task382 | family_dsl | symmetry/flip/rotate | 107,354 | 0.000 | 0.380 | 0.52 | 0.66 | Route to family DSL/compiler search; record any exact rule. |
| 31 | task370 | family_dsl | symmetry/flip/rotate | 107,206 | 0.000 | 0.379 | 0.52 | 0.66 | Route to family DSL/compiler search; record any exact rule. |
| 32 | task396 | semantic_rebuild | reduce | 131,688 | 0.000 | 0.436 | 0.58 | 0.48 | cost rank 19 / compiler +0.07 / Open semantic-factory pass: inspect examples, write prototype, then stress-... |
| 33 | task264 | semantic_rebuild | reduce | 139,356 | 0.561 | 0.385 | 0.52 | 0.48 | cost rank 18 / best-seen +0.56 / Open semantic-factory pass: inspect examples, write prototype, then stress... |
| 34 | task284 | family_dsl | symmetry/flip/rotate | 101,133 | 0.000 | 0.370 | 0.52 | 0.66 | Route to family DSL/compiler search; record any exact rule. |
| 35 | task077 | semantic_rebuild | other | 157,494 | 0.000 | 0.463 | 0.52 | 0.36 | cost rank 14 / Open semantic-factory pass: inspect examples, write prototype, then stress-test. |
| 36 | task110 | family_dsl | color_remap | 107,857 | 0.000 | 0.377 | 0.52 | 0.72 | Route to family DSL/compiler search; record any exact rule. |
| 37 | task209 | semantic_rebuild | reduce | 113,356 | 0.384 | 0.326 | 0.52 | 0.48 | cost rank 28 / best-seen +0.38 / compiler +0.08 / Open semantic-factory pass: inspect examples, write proto... |
| 38 | task205 | compiler_golf | reduce | 103,313 | 0.000 | 0.352 | 0.52 | 0.48 | compiler +0.08 / Route to family DSL/compiler search; record any exact rule. |
| 39 | task034 | audit_before_swap | color_remap | 106,796 | 0.005 | 0.340 | 0.55 | 0.72 | Require semantic proof and pseudo-hidden audit; no raw public-source swap. |
| 40 | task023 | family_dsl | color_fill | 72,612 | 1.202 | 0.504 | 0.52 | 0.44 | best-seen +1.20 / Route to family DSL/compiler search; record any exact rule. |
| 41 | task044 | family_dsl | symmetry/flip/rotate | 43,947 | 0.323 | 0.283 | 0.52 | 0.66 | best-seen +0.32 / Route to family DSL/compiler search; record any exact rule. |
| 42 | task175 | family_dsl | color_remap | 87,620 | 0.986 | 0.435 | 0.68 | 0.66 | best-seen +0.99 / graph risk 2 / Route to family DSL/compiler search; record any exact rule. |
| 43 | task182 | pseudo_hidden_first | symmetry/flip/rotate | 47,587 | 0.000 | 0.156 | 0.52 | 0.78 | builder / Run or extend pseudo-hidden audit before using the builder. |
| 44 | task216 | family_dsl | crop | 103,812 | 0.002 | 0.357 | 0.48 | 0.58 | Route to family DSL/compiler search; record any exact rule. |
| 45 | task153 | pseudo_hidden_first | reduce | 21,618 | 0.631 | 0.240 | 0.52 | 0.60 | best-seen +0.63 / builder / Run or extend pseudo-hidden audit before using the builder. |
| 46 | task128 | family_dsl | symmetry/flip/rotate | 95,024 | 0.000 | 0.210 | 0.52 | 0.66 | Route to family DSL/compiler search; record any exact rule. |
| 47 | task280 | family_dsl | symmetry/flip/rotate | 90,506 | 0.000 | 0.206 | 0.52 | 0.66 | Route to family DSL/compiler search; record any exact rule. |
| 48 | task187 | semantic_rebuild | color_fill | 105,313 | 0.000 | 0.386 | 0.56 | 0.44 | public bottom15 / Open semantic-factory pass: inspect examples, write prototype, then stress-test. |
| 49 | task234 | family_dsl | symmetry/flip/rotate | 81,703 | 0.000 | 0.196 | 0.52 | 0.66 | Route to family DSL/compiler search; record any exact rule. |
| 50 | task208 | audit_before_swap | symmetry/flip/rotate | 92,408 | 0.001 | 0.188 | 0.55 | 0.66 | Require semantic proof and pseudo-hidden audit; no raw public-source swap. |

## Missing Data For A Stronger Model

- Pseudo-hidden/metamorphic coverage is sparse: 5/400 tasks have any dossier status beyond not_run, and only 2/400 are stress_clean.
- Graph audit rows are incomplete or anchor-mismatched for model training: 302/400 tasks have risk stats.
- Only 10/400 tasks list known prototypes; semantic state should be structured for every high-priority task.
- Need branch-level training labels: submitted/not submitted, accepted/rejected, LB delta, local delta, and rollback reason.
- Need single-task or small-batch LB observations to map bundle outcomes back to task-level hidden risk.
- Need effort/cycle-time labels: human hours, generated graph size, failed attempts, and compile/test time.
- Need public competitor per-task scores beyond the bottom-15 list and five explicit gap estimates.
- Need machine-readable pseudo-hidden contract names and failure modes, not just aggregate pass/fail counts.
- Need graphizability labels from completed compilers: which family primitives compiled compactly and which blew up.
- Need source-trust labels for best_seen attempts; some best_seen deltas are useful evidence but not safe swap instructions.

## Output Files

- `reports/task_triage_ml_features_20260606.csv`
- `reports/task_triage_ml_top50_20260606.csv`
- `reports/task_triage_ml_next25_50_20260606.csv`
- `reports/task_triage_ml_families_20260606.csv`
- `reports/task_triage_ml_directions_20260606.csv`
- `reports/task_triage_ml_summary_20260606.json`
