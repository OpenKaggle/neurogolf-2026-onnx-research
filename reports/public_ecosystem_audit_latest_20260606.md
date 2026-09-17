# Latest Public Ecosystem Audit - NeuroGolf 2026 - 2026-06-06

Scope: public Kaggle ecosystem/discussion audit only. Current comparison anchor:
`v4_plus12_task285`, public LB `6260.36`, local n=3 `6260.364855`.
No anchor dirs, zips, or ledgers were edited. No Kaggle submission was made.

Post-run local state note: while this audit was running, another lane updated
`reports/HANDOFF_NEXT_AGENT.md` to `v4_plus13_task366`, public LB `6260.48`
(`2026-06-06 ~19:04 CST`). The public-source findings below were computed
against the user-requested `v4_plus12` anchor. The only material adjustment is
that `task366` is now partly improved locally (`830720 -> 734516` cost), but
the public `biohack44_superior` `task366` file is still worse and not a direct
candidate.

## Executive Findings

1. No newer public dataset supersedes the previous audit. `datasets list
   --search neurogolf --sort-by updated` still tops out at
   `massimilianoghiotto/neurogolf2026-6254`, updated `2026-06-03`.
2. The main unreflected public artifact is
   [biohack44/neurogolf-superior-blend-notebook](https://www.kaggle.com/code/biohack44/neurogolf-superior-blend-notebook).
   Earlier local notes marked this source as partial due to Kaggle output
   pagination failures. The local source dir now has 400 ONNX files and scores
   `6208.068760` against our scorer. It is far below anchor as a bundle, but it
   exposes small direct swap candidates and one large idea-only `task191` clue.
3. Fresh June 5-6 discussion facts reinforce the same frontier tasks:
   `task255`, `task233`, `task366`, `task018`, and `task191/285`-class
   shape/object tasks. These are not new packages, but they are useful public
   target-prior signals.
4. The only direct public-source candidates found are small. Excluding unsafe
   `task191`, the full-example-clean `biohack44_superior` positive deltas sum
   to about `+0.4068` local. This is not worth a standalone submission, but it
   is viable batch filler after a separate candidate build/audit.

## Public Items Not Reflected In The Prior Public Audit

| item | public ref | checked fact | action |
|---|---|---|---|
| Biohack Superior Blend Notebook | [kernel](https://www.kaggle.com/code/biohack44/neurogolf-superior-blend-notebook) | Last run `2026-06-02 12:46:00`; code blends `biohack44/notebook715db8358e` with [Massim Conv Part 1](https://www.kaggle.com/code/massimilianoghiotto/convolution-series-part-1). Kaggle log reports 400 tasks, 0 incorrect, local total `6208.07`. | Source lake only. Mine direct candidates below; do not use whole bundle. |
| Biohack Super Blend Best Public Score | [kernel](https://www.kaggle.com/code/biohack44/neurogolf-super-blend-best-public-score) | Already locally represented as `submissions/biohack44_super_onnx`; output download now also works via Kaggle API. | Already reflected in anchor lineage through `task018`; no new action. |
| Tail task score discussion | [704570](https://www.kaggle.com/competitions/neurogolf-2026/discussion/704570) | Poster names worst tasks `255`, `233`, `366`, `018`; reply says the same four can total `60` points for a stronger solution, vs much lower floors. | Confirms our bottom-task queue; no artifact to merge. |
| Corridor / empty-rectangle ONNX patterns | [704769](https://www.kaggle.com/competitions/neurogolf-2026/discussion/704769) | Public question asks about low-node patterns for empty corridors, border-connected empty regions, endpoint erosion, and `ReduceSum`/morphology/`Where` tradeoffs. No answers yet. | Idea-only lead for `255/233/366` hole/corridor templates. |
| Token efficiency discussion | [704568](https://www.kaggle.com/competitions/neurogolf-2026/discussion/704568) | Competitors report very high token spend and one claimed daily jump from `6867.96` to `7047.27`; another mentions `14B` tokens in a month. | Workflow signal only: prioritize reusable harness/ledgers over one-off prompting. |
| Overfit and timeout comments | [704762](https://www.kaggle.com/competitions/neurogolf-2026/discussion/704762) | New comments say ARC-gen can generate insoluble samples for complex tasks and synthetic private sets can reject Kaggle-passing models. | Keep pseudo-hidden as a gate, but do not blindly optimize to every generated sample. |

## Direct Merge Candidates

These are direct-candidate public source swaps from
`submissions/biohack44_superior_onnx` against `v4_plus12`. They passed isolated
train/test/arc-gen evaluation. They still require a normal candidate overlay,
full 400-task audit, and local scoring before any submission.

| task | local delta | source cost | anchor cost | isolated result | recommendation |
|---:|---:|---:|---:|---|---|
| 357 | `+0.085858` | `6581` | `7171` | `3/3`, `1/1`, `9/9` | Direct batch candidate. |
| 248 | `+0.085858` | `6581` | `7171` | `3/3`, `1/1`, `9/9` | Direct batch candidate. |
| 145 | `+0.050320` | `69757` | `73357` | `4/4`, `1/1`, `262/262` | Direct batch candidate. |
| 153 | `+0.037468` | `20823` | `21618` | `3/3`, `1/1`, `261/261` | Direct batch candidate. |
| 239 | `+0.018873` | `94477` | `96277` | `4/4`, `1/1`, `262/262` | Optional micro candidate. |
| 273 | `+0.016703` | `5937` | `6037` | `3/3`, `1/1`, `262/262` | Optional micro candidate. |
| 265 | `+0.016461` | `39162` | `39812` | `3/3`, `1/1`, `262/262` | Optional micro candidate. |
| 289 | `+0.014313` | `24972` | `25332` | `5/5`, `1/1`, `262/262` | Optional micro candidate. |
| 348 | `+0.013780` | `7207` | `7307` | `2/2`, `1/1`, `262/262` | Optional micro candidate. |
| 129 | `+0.013539` | `807` | `818` | `3/3`, `1/1`, `261/261` | Optional micro candidate. |

Remaining full-example-clean positive rows are tiny:
`281,383,089,247,397,055,034,216,208,224,059,360,175,263,003,212,275`.
All 27 safe-positive rows total about `+0.4068` local.

## Idea-Only Leads

| item | fact | why not direct |
|---|---|---|
| `task191` from `biohack44_superior` | Looks locally huge by n=3: score `13.638840` vs anchor `11.553610`, cost `85919` vs `691342`, delta `+2.085230`. | Full isolated audit fails arc-gen: `train 4/4`, `test 1/1`, `arc-gen 222/262`. Do not swap. Mine graph motifs for a guarded compact D4/boundary matcher. |
| Biohack notebook claim that selection blends preserve private behavior | The notebook says the blend cannot do worse than source submissions because it only selects existing task files. | Not enough under our hidden-safety rules. We need per-task source trust, full-example eval, pseudo-hidden contracts, and preferably single-task proof for non-anchor source classes. |
| Corridor / empty-rectangle discussion | Publicly points at `ReduceSum` row/column masks, morphology, and `Where` fills. | No code, no task-specific proof. Use to shape typed primitives, not as a source. |
| Token-use discussions | High-scoring competitors are spending large token budgets and using repeated prompting. | Workflow guidance only. Direct engineering value is harness discipline, attempt memory, and task dossiers. |

## Top 5 Safe Task-Level Leads

1. `task191`: highest public clue. Rebuild from the `biohack44_superior`
   compact graph idea, but require `262/262` arc-gen plus pseudo-hidden D4 and
   boundary contracts before candidate use. Target is a safe compact
   D4/stamp/boundary compiler path, not a raw source swap.
2. `task255`: current anchor score `10.718956`, cost `1,592,864`.
   Discussions identify it as a floor task; corridor/hole-fill patterns are
   public signals. Safe route: write contracts first, then engineer
   hole/enclosure/corridor templates with no visible-key lookup.
3. `task233`: current anchor score `10.812401`, cost `1,450,762`.
   Public floor discussions and the previous synthesis report both point here.
   Safe route: largest/salient component plus hole/patch reinsertion, with
   color-permutation and distractor contracts.
4. `task366`: current anchor score `11.369952`, cost `830,720`.
   Public source is worse, but discussion confirms it is a bottom-task target.
   Safe route: bounded-rectangle/corridor compiler, object summary first,
   no batch public-source absorption.
5. `task018`: current anchor score `11.926438`, visible local `0/3`, cost
   `476,185`. Public discussions keep naming it as a worst/overfit-risk task.
   Safe route is exact graph/cost surgery preserving the current hidden-safe
   behavior; do not replace semantics just because visible eval is already 0.

## Bottom Line

No new public package changes the anchor. The only actionable public-source
merge material is a small `biohack44_superior` micro-swap set worth about
`+0.4` local after normal candidate audit. The valuable new public information is
mostly target selection and pattern hints: attack `191/255/233/366/018` through
typed, contract-gated engineering rather than raw public bundle blending.
