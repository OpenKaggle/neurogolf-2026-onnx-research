# konbu17_v36 Swap Candidate Plan

## Source Background

- Kaggle public leaderboard rank 95, score **6196.55** (`konbu17/neurogolf-2026-v36`)
- Local re-score (n=3): 5992.77 — ~204 point gap vs official, similar magnitude to afr1ste/octv
- All 400 task ONNX files pass local n=3 examples
- Structural inspection: networks look algorithmic (Gather/ScatterElements/Conv/MatMul), not memorization lookups → modestly encouraging vs afr/octv pattern

## Risk Context

- biohack-6113 cherry-pick: SAFE (3/3 batches, 9 swaps, all matched local to ±0.01)
- afr1ste-6335 cherry-pick: UNSAFE (single-task probe regressed −30.32 with predicted +25)
- octv-6154 cherry-pick: UNSAFE (mega-swap probe regressed)
- konbu17-6196: UNTESTED, treat as risky until 1+ single probe confirms

## Full Local Verification (FULL example set, not n=3)

| Task | Pass | Total | Our Cost   | Konbu Cost | Reduction | Our Score | Konbu Score | Local Gain |
|------|------|-------|-----------:|-----------:|----------:|----------:|------------:|-----------:|
| 319  | 267  | 267   | 26,325,352 |     19,184 |   1372x   |     7.914 |      15.138 |     +7.224 |
| 285  | 265  | 265   | 99,576,167 |    173,596 |    574x   |     6.584 |      12.936 |     +6.352 |
| 219  | 265  | 265   | 15,842,765 |     48,138 |    329x   |     8.422 |      14.218 |     +5.796 |
| 255  | 265  | 265   | 42,917,144 |    146,404 |    293x   |     7.425 |      13.106 |     +5.681 |
| 044  | 266  | 266   |  2,380,303 |     31,805 |     75x   |    10.317 |      14.633 |     +4.317 |
| 153  | 265  | 265   |    755,558 |     11,500 |     66x   |    11.465 |      15.650 |     +4.184 |
| 233  | 266  | 266   |  3,682,865 |     75,542 |     49x   |     9.881 |      13.768 |     +3.887 |
| 133  | 267  | 267   |  9,344,642 |    193,923 |     48x   |     8.950 |      12.825 |     +3.876 |
| 157  | 265  | 265   |  3,947,893 |     85,241 |     46x   |     9.811 |      13.647 |     +3.836 |
| 076  | 266  | 266   |  1,418,521 |     36,184 |     39x   |    10.835 |      14.504 |     +3.669 |
| 118  | 267  | 267   |  2,155,775 |     61,093 |     35x   |    10.416 |      13.980 |     +3.564 |
| 023  | 266  | 266   |    605,720 |     21,828 |     28x   |    11.686 |      15.009 |     +3.323 |
| 209  | 266  | 266   |  2,053,296 |     77,236 |     27x   |    10.465 |      13.745 |     +3.280 |
| 066  | 266  | 266   |    499,535 |     21,408 |     23x   |    11.879 |      15.028 |     +3.149 |
| 101  | 266  | 266   |  1,261,888 |     65,879 |     19x   |    10.952 |      13.904 |     +2.952 |

Additional verified single-task wins (passes full local, smaller per-task gain):
158, 367, 096, 363, 200, 018, 387, 025, 182, 324, 243, 175, 084, 277, 204

**Excluded (fail local):**
- task191: konbu pass 227/267 (already known unsafe; biohack also failed)
- task366: konbu pass 255/266 → DO NOT swap

Total available cumulative local gain across 30 fully-passing wins: **~90 points** (5992→6210 in theory).

NOTE: task233 currently passes only **182/266** local in OUR 6120 best, so konbu's task233 (266/266) would also FIX a local-quality issue, not just lower cost.

## Biohack-6113 Untapped (proven-safe source!)

biohack-6113 swap was previously declared exhausted but a re-audit found 2 non-noise wins that weren't merged because we keep our own hand-build for those tasks:

- **task200**: our hand-build cost 81,842 (13.69) → biohack 10,511 (15.74), **+2.05**
- **task084**: our hand-build cost 30,168 (14.69) → biohack 19,513 (15.12), **+0.44**

Plus 3 noise wins (129/216/338) totalling +0.05.

biohack-6113 source has matched local prediction to ±0.01 in 9/9 prior swaps → near-zero risk for these.

## Pre-built Candidate Zips (validated locally)

| Zip | SHA256 (prefix) | Local | Δ vs 6120 | Risk |
|-----|-----------------|------:|----------:|------|
| submission_candidate_bio_t200_t084.zip       | 4aaab23d56f0b649 | 6122.60 | +2.49 | **low** (biohack proven safe) |
| submission_candidate_konbu_t101.zip          | 06aeed4113618af5 | 6123.06 | +2.95 | low (19x reduction, smallest konbu probe) |
| submission_candidate_konbu_3task_lowred.zip  | 8a03323f1bfa4df8 | 6129.53 | +9.43 | low (101+066+023, max 28x) |
| submission_candidate_konbu_5task_lowred.zip  | f69839fb4c394e42 | 6136.38 | +16.27 | med (+209+118, max 36x) |
| submission_candidate_combo_bio2_konbu5.zip   | 43d8d6ae1e96d356 | 6138.87 | +18.76 | med (bio 200+84 + konbu 5lowred) |
| submission_candidate_max_safe.zip            | a366a6cd60b54650 | 6208.25 | +88.14 | **high** (33 swaps incl monsters — afr-style risk if konbu unsafe) |

## OUTCOME (executed 2026-06-02, probes 1-2 of 5)

| Probe | Zip | Local Δ | Official | Result |
|-------|-----|--------:|---------:|--------|
| 1 bio_t200_t084 | `4aaab23d…` | +2.49 | **6122.59** (ref 53283290) | ✅ PERFECT — promoted to current best |
| 2 konbu_t101 on 6122 | `cadd8939…` | +2.95 | **6111.64** (ref 53283416) | ❌ REGRESSED −10.95 vs 6122.59 |

**Verdict: konbu17 cherry-pick is HIDDEN-UNSAFE** (same failure class as afr1ste/octv). Probes 3–5 and all staged konbu/combo/max_safe zips are **permanently rejected**. Do not submit.

Only biohack-6113 remains safe for external cherry-pick (now **11/11** official swaps, all matched local within 0.01). Remaining biohack wins on 6122 base: task338 (+0.03), task129 (+0.01) — noise tier only.

Current best: `submissions/submission_current_best_6122.zip` (sha `4aaab23d…`), official **6122.59**, rank **211 / 1540** (LB snapshot 2026-06-02T08:52:30).

---

## Recommended Probe Sequence (when quota refreshes) — SUPERSEDED

Each probe consumes 1 Kaggle quota. Plan adapts based on official feedback.

1. **Probe 1 = `submission_candidate_bio_t200_t084.zip`** (+2.49 local) — **DONE ✅**
   - Near-zero risk (biohack 9/9 prior matches). Locks in free score.
   - Officially matched → continue. (If it somehow regressed, biohack source assumptions wrong → reassess.)

2. **Probe 2 = `submission_candidate_konbu_t101.zip`** (+2.95 local on top of probe 1's promoted best) — **DONE ❌ official 6111.64**
   - Tests konbu cherry-pick safety at lowest cost-reduction ratio.
   - If official matches local → konbu low-reduction safe.
   - If official regresses → konbu cherry-pick abandoned; probe 1 result becomes new best (~6122).

3. **Probe 3 (if probe 2 safe)** = `submission_candidate_konbu_5task_lowred.zip` rebuilt on probe 2's base
   - Local +16.27 over 6120, scales konbu adoption.
   - Or equivalently: rebuild stack of probe 1 base + konbu 4 additional low-reduction tasks.

4. **Probe 4 (if probe 3 safe)**: medium-reduction stack — add konbu 044/076/153/157/158/367/233/118 (+30+ more).

5. **Probe 5**: monster bet — `task319` (+7.22) or `task285` (+6.35) single swap.

Total upside if konbu17 is fully cherry-pick-safe: **+50 to +90** over current 6120 best.

If konbu17 is hidden-UNSAFE, probe 1 still locks in +2.49 from biohack extras.

## Things NOT to do

- DO NOT swap task191 or task366 (fail local pass).
- DO NOT batch monster swaps before low-reduction baseline is proven.
- DO NOT replace 6120plus current best until any probe officially matches local prediction.
- DO NOT delete existing named best zips; always preserve a rollback path.
