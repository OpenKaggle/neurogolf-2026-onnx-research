# Public Ecosystem Audit - NeuroGolf 2026 - 2026-06-06

Scope: independent public Kaggle ecosystem audit for competition slug
`neurogolf-2026`. Current local anchor used for comparison:
`submissions/candidate_v4_plus8_task149_bool_onnx`, local `6257.0466`,
Kaggle ref `53415813`, LB `6257.04`.

Postscript from parent integration: after this audit started, the parent first
combined the uniform-scalar and BOOL-memory branches into `v4_plus9_compiler`
(ref `53416534`, LB `6258.14`), then added terminal-Cast output narrowing as
`v4_plus10_compiler2` (ref `53417133`, LB `6260.16`). `v4_plus10_compiler2` is
the current anchor. Public-ecosystem conclusions below are unchanged; direct
score comparisons should now use the v4_plus10 files.

No Kaggle submission was made. I wrote only this report.

## Sources and Access

Verified local context:

- `reports/research.md` and `reports/HANDOFF_NEXT_AGENT.md` identify the slug
  as `neurogolf-2026` and record the current LB-verified anchor.
- The workspace is not a git repo, so changed-file checks use filesystem
  inspection rather than `git status`.
- `kaggle` is not on `PATH`, but `.venv/bin/python -m kaggle` is available.

Live Kaggle access used on 2026-06-06 CST:

- `kernels list --competition neurogolf-2026` sorted by `dateCreated`,
  `dateRun`, `hotness`, and `voteCount`.
- `datasets list --search neurogolf --sort-by updated`.
- `datasets files` for high-signal public artifact datasets.
- `kernels pull` into `/tmp/neurogolf_public_audit_20260606` for selected
  notebooks not already present locally.
- `competitions topics list neurogolf-2026` and `topic-messages` for official
  updates, exploit threads, and recent workflow/performance discussions.

Limitations:

- `kernels status` returned Kaggle HTTP 500 for every tested public kernel, so
  status/run state is taken from `kernels list` `lastRunTime`, pulled notebook
  metadata, and dataset metadata instead.
- I did not execute public notebooks and did not download every older low-score
  dataset. Reproduction recommendations below are based on pulled code,
  metadata, local staged artifacts, and local reports.

## Executive Summary

Verified:

- The newly visible high-signal items after the existing handoff are mostly
  workflow and rewrite references, not immediately mergeable packages.
- The best normal measurable public artifact class remains the Massim/Nadeem
  source family already absorbed into the current anchor lineage. Massim
  `task149` was absorbed and then surpassed by our BOOL rewrite; Massim
  `task258` is already equal to the current anchor at cost `160`.
- Nadeem/Biohack FP16+prune notebooks are useful as public documentation of
  guarded graph surgery but do not expose a new safe artifact above the current
  anchor.
- `vyankteshdwivedi/neurogolf-multi-source-onnx-solver` is a notable not-yet
  digested notebook: it locks to Massim `6254.64`, then implements ReduceSum
  chain fusion, Cast-chain collapse, dim scrub, and disabled trusted overrides.
  It also contains explicit "GAMBLE" exceptions after validation failures, so
  it should be mined for rewrite ideas only, not run as a candidate.
- Recent discussions strongly support the strategic picture in the handoff:
  top progress is many repeated per-task micro-golfs under a harness with
  semantic search, cost introspection, staging, and selective Kaggle probes.

Hypotheses:

- The next public-ecosystem gain is more likely a small lossless graph rewrite
  pass, or a typed-IR compiler improvement, than another public bundle.
- The most valuable new public signal is not a file but a workflow: per-node
  cost breakdown, attempt memory, task dossiers, and repeated passes over the
  same task family.
- Any public package that depends on old scorer-boundary behavior, custom
  domains, or "visible pass but validation failure" should be treated as a
  research artifact until isolated Kaggle probes prove otherwise.

## High-Signal Public Inventory

| Item | URL | Author | Public date/update | Advertised score | Downloadable artifact | Verified facts | Novel techniques | Local reproduction recommendation | Scorer-boundary implications |
|---|---|---:|---:|---:|---|---|---|---|---|
| Surgical ONNX precision reduction | [kernel](https://www.kaggle.com/code/seddiktrk/surgical-onnx-precision-parameter-reduction) | seddik turki | last run `2026-06-05 20:07:42` | no whole-LB score advertised | notebook only; uses Massim Part 4 kernel output | Locally present in `public_kernels/seddik_precision`; notebook applies prune, exact initializer dedup, and uniform-initializer-to-scalar compression. Already summarized in `research.md`, but still worth keeping as an implementation reference. | Safe parameter-only rewrites; uniform scalar replacement with consumer-op broadcast whitelist. | Port ideas into a pass against current anchor; verify `same_decoded_outputs` and current scorer cost before any probe. Prioritize uniform scalar after BOOL typed IR. | Legitimate scorer optimization: params count every initializer element, so scalar broadcast can reduce cost without changing outputs. Watch shape-dependent consumers. |
| Trace Language DFA Solvers | [kernel](https://www.kaggle.com/code/scottweeden/neurogolf-trace-language-dfa-solvers) | Scott Weeden | last run `2026-06-05 20:57:17` | no LB score advertised | notebook output can package `submission.zip` | Pulled/local notebook defines 22-symbol pipeline alphabet and DFA: discover, analyze, build, optimize, verify, cost, blend, checksum, package, submit. Uses public sources `jsrdcht`, `afr1ste_6335`, `octavi_v205`, `konbu17_v117`. | Formal state machine for agent workflow; simple identity/recolor/label-propagation builders; banned-op/static-shape checks. | Do not use its generated bundle. Borrow the state-machine idea for our experiment gates and handoff tooling. | Reduces process mistakes, not model cost. Blending sources include known unsafe classes, so the scorer/hidden boundary risk remains. |
| Convolution Series Part 4 | [kernel](https://www.kaggle.com/code/massimilianoghiotto/convolution-series-part-4) | MassimilianoGhiotto | last run `2026-06-05 10:36:14` | tutorial task score `19.925` for `task258`; local Part 4 bundle `6255.1495` in repo | notebook + local staged `submissions/massim_part4_6255_onnx`; live dataset slug for `6255` was not listed, but `6254` is public | Part 4 demonstrates grouped Conv on `task258`: local current anchor already has `task258` cost `160`, score `19.924826`, same as Massim Part 4. `task149` in Part 4 is cost `313`, worse than our BOOL cost `172`. | `Conv(group=G)` as parameter reducer; active-color grouping. | No direct swap. Add grouped/depthwise Conv templates to DSL/typed compiler. | Groups are a clean, current-scorer-safe way to lower params when color interactions are block-local. |
| Convolution Series Part 3 | [kernel](https://www.kaggle.com/code/massimilianoghiotto/convolution-series-part-3) | MassimilianoGhiotto | last run `2026-06-03 14:02:49` | tutorial claims task149 `18.920`; packages `neurogolf2026-6254` | dataset [massimilianoghiotto/neurogolf2026-6254](https://www.kaggle.com/datasets/massimilianoghiotto/neurogolf2026-6254), updated `2026-06-03 11:51:12` | Pulled notebook explains stride-4 Conv for task149. Local `massim_6254/task149` cost `509`; Part 4 cost `313`; current BOOL rewrite cost `172`. | Strided Conv lands on separated blocks, then channel assembly with Neg/Concat/Pad. | Fully absorbed and surpassed. Keep as compiler-pattern reference only. | Shows memory can dominate even when params are tiny; BOOL lifetime beats FLOAT score maps on this task. |
| Nadeem 6252 baseline | [kernel](https://www.kaggle.com/code/nadeembinshajahan/6252-lb-neurogolf-6252-baseline) | Nadeem | last run `2026-06-03 18:05:11` | `6252 LB` | notebook zips attached ONNX dataset; local `submissions/nadeem_6252_onnx` | Already deeply absorbed by reports. Local raw total `6252.3337` but has known visible failures/pollution cases; conservative v4 lineage led to current anchor. | Packaging a strong public bundle; no code technique in notebook itself. | No raw use. Continue using provenance and audited integrated files only. | Public LB score does not imply each task is hidden-safe or visible-clean under our scorer. |
| Nadeem stable FP16 surgery | [kernel](https://www.kaggle.com/code/nadeembinshajahan/6151-lb-neurogolf-2026-stable-fp16-surgery) | Nadeem | last run `2026-06-01 14:15:53` | `6151 LB` | notebook output only; dataset source `afr1ste/neurogolf-5689-51-current-rules-open-artifact` | Pulled notebook documents FP16 cast-push plus dead-edge prune with risky-op exclusion and triple verification. | Risky op whitelist (`ScatterND`, `GridSample`, `GatherND`, `Resize`, etc.); ORT disabled/default double run. | Use as a validation pattern, not a source. Our anchor already has LB-verified type/optimizer surgeries. | Important boundary lesson: fp16 visible correctness can differ across ORT versions/platforms, so whitelist and dual-session checks matter. |
| Nadeem / Biohack FP16+prune blend | [Nadeem kernel](https://www.kaggle.com/code/nadeembinshajahan/neurogolf-2026-fp16-surgery-prune-blend-6130), [Biohack kernel](https://www.kaggle.com/code/biohack44/neurogolf-2026-fp16-surgery-prune-blend-6115) | Nadeem / Emre Cirak | Nadeem last run `2026-05-30 19:25:19`; Biohack last run `2026-05-31 14:39:58` | Nadeem `~6130`; Biohack title `~6115` but local `notebook.py` says `~6130` expected | notebooks output `submission.zip`; sources include `jsrdcht/neurogolf-6029-submission-bundle`, `biohack44/neurogolf-6067-locked`, `biohack44/neurogolf-6113-bundle` | Code applies fp16 surgery and prune when local examples match and cost decreases. Techniques are already mostly in our lossless-surgery lineage. | Cast-push fp16, prune unused initializers, bundle blending from public anchors. | No direct use. Mine for tests around FP16 exclusions if extending type narrowing. | Reinforces that value-preserving by examples is not enough on risky ops; cross-version/cross-hidden failures are real. |
| Multi-Source ONNX Solver | [kernel](https://www.kaggle.com/code/vyankteshdwivedi/neurogolf-multi-source-onnx-solver) | zorojuro / vyanktesh | last run `2026-06-04 15:47:27` | no stable LB title; notebook estimates score at runtime | notebook output only; public source fan-in includes `ngc26-public-v3`, `afr1ste`, `konbu17`, `massim_6254`, `sigmaborov/test-golf`; kernel sources include Beicicc, Needless, Jonathan Chan, Konbu | Pulled notebook contains a "PRIMARY artifact lock" to Massim `neurogolf2026-6254 -> 6254.64`, graph rewrites, dim scrub, optional trusted overrides disabled, and explicit `GAMBLE` keeps for tasks `096/101/133/178/185/234` after validation failure. | ReduceSum-chain fusion, Cast-chain collapse, dim scrub allowlist/blocklist, large public-source blender, some generated tensor-index templates. | Do not run as candidate. Extract only rewrite functions for isolated current-anchor evaluation. Start with ReduceSum-chain fusion and Cast-chain collapse on current anchor, gated by full examples and scorer cost. | High boundary risk: combines known unsafe sources and deliberately keeps some validation failures. Useful as an exploit/risk map, not as an anchor. |
| Dimple task111 deep dive | [kernel](https://www.kaggle.com/code/dimplebhardwaj536/neurogolf) | Dimple Bhardwaj | last run `2026-06-03 11:47:56` | task111 `18.319`; packages `6110` public bundle | notebook output only; dataset [massimilianoghiotto/neurogolf2026-6110](https://www.kaggle.com/datasets/massimilianoghiotto/neurogolf2026-6110), updated `2026-05-30 19:32:08` | Pulled notebook explains `task111` crop-by-gray-pixel with Slice, ReduceSum, ArgMax, arithmetic, Pad; final cell just zips Massim `6110`. | Good node-by-node crop-localization tutorial. | No source action; use as an educational pattern for ArgMax/Slice coordinate extraction. | Shows ArgMax/Slice coordinate solvers are current-scorer legal if shapes are static and outputs are padded. |

## Public Artifact Datasets

| Dataset | URL | Author | Last updated | Advertised score | Files verified | Recommendation |
|---|---|---:|---:|---:|---|---|
| `massimilianoghiotto/neurogolf2026-6254` | [dataset](https://www.kaggle.com/datasets/massimilianoghiotto/neurogolf2026-6254) | MassimilianoGhiotto | `2026-06-03 11:51:12` | `6254` | 400 ONNX under `submission/`; first files match local staged shape | Already mined; only task149 mattered before BOOL rewrite. |
| `massimilianoghiotto/neurogolf2026-6208` | [dataset](https://www.kaggle.com/datasets/massimilianoghiotto/neurogolf2026-6208) | MassimilianoGhiotto | `2026-06-02 09:06:12` | `6208` | flat 400 ONNX | Below current anchor; source lake only. |
| `massimilianoghiotto/neurogolf2026-6066-58-score` | [dataset](https://www.kaggle.com/datasets/massimilianoghiotto/neurogolf2026-6066-58-score) | MassimilianoGhiotto | `2026-06-01 08:04:51` | `6066.58` | 400 ONNX under `submission/` | Superseded. |
| `massimilianoghiotto/neurogolf2026-6112` | [dataset](https://www.kaggle.com/datasets/massimilianoghiotto/neurogolf2026-6112) | MassimilianoGhiotto | `2026-05-31 09:40:32` | `6112` | 400 ONNX under `submission/` | Superseded. |
| `massimilianoghiotto/neurogolf2026-6110` | [dataset](https://www.kaggle.com/datasets/massimilianoghiotto/neurogolf2026-6110) | MassimilianoGhiotto | `2026-05-30 19:32:08` | `6110` | 400 ONNX under `submission/` | Superseded; used by Dimple notebook. |
| `biohack44/neurogolf-6113-bundle` | [dataset](https://www.kaggle.com/datasets/biohack44/neurogolf-6113-bundle) | Emre Cirak | `2026-05-31 14:31:47` | `6113` | flat 400 ONNX | Already a trusted/safe cherry-pick source in reports; remaining wins are noise. |
| `biohack44/neurogolf-6067-locked` | [dataset](https://www.kaggle.com/datasets/biohack44/neurogolf-6067-locked) | Emre Cirak | `2026-05-25 11:01:23` | `6067` | flat 400 ONNX | Superseded except as FP16 tutorial anchor. |
| `octaviograu/neurogolf-manual-rewrites-v205` | [dataset](https://www.kaggle.com/datasets/octaviograu/neurogolf-manual-rewrites-v205) | Octavi Grau | `2026-06-01 10:50:41` | `6154.71` | README, provenance CSV, 400 ONNX under `submission/` | Known unsafe for broad cherry-pick in local reports; keep provenance only. |
| `afr1ste/neurogolf-5689-51-current-rules-open-artifact` | [dataset](https://www.kaggle.com/datasets/afr1ste/neurogolf-5689-51-current-rules-open-artifact) | Afr1ste | `2026-05-31 16:43:27` | `5689.51` | README, manifest, submission CSV, 400 ONNX | Older source; already mined and superseded. |
| `konbu17/neurogolf-2026-blend-source-v3-6-0` | [dataset](https://www.kaggle.com/datasets/konbu17/neurogolf-2026-blend-source-v3-6-0) | konbu17 | `2026-05-08 08:34:30` | blend source, no single title score in dataset list | flat 400 ONNX | Do not batch use; v4_plus6 proved hidden-catastrophic. |
| `jsrdcht/neurogolf-6029-submission-bundle` | [dataset](https://www.kaggle.com/datasets/jsrdcht/neurogolf-6029-submission-bundle) | Chet | `2026-05-18 10:59:56` | `6029.09` | flat 400 ONNX | Historic baseline only. |
| `kojimar/neurogolf-5800-55-minimal-onnx-blend-assets` | [dataset](https://www.kaggle.com/datasets/kojimar/neurogolf-5800-55-minimal-onnx-blend-assets) | islet | `2026-05-18 01:15:09` | `5800.55` | `base_submission/` ONNX | Historic source only. |

## Already Absorbed or Boundary Artifacts

Verified from local reports and live metadata:

- [Beicicc 6645.39 Open Solution](https://www.kaggle.com/code/beicicc/neurogolf-6645-39-public-score-open-solution), Kun Zhang, last run
  `2026-04-30 16:34:13`, artifact
  [beicicc/neurogolf-6645-39-open-submission-artifact](https://www.kaggle.com/datasets/beicicc/neurogolf-6645-39-open-submission-artifact).
  Local reports classify it as a scorer/environment-boundary artifact:
  many graphs use custom `golf.Identity`; not directly mergeable under the
  current local scorer. Reproduction recommendation: isolated tiny custom-domain
  boundary probes only, never anchor blending.
- [AgentZZ NeuroGolf Submit 6284 v2](https://www.kaggle.com/code/agentzz/neurogolf-submit-6284-v2),
  AgentZZ, last run `2026-05-04 22:29:47`, advertised `6284`.
  Local downloaded artifact is `downloads/afr1ste_6284/neurogolf-6284-93-open-submission-artifact.zip`.
  Local reports put it in the same boundary-artifact class: many excluded/custom
  behaviors and unmeasurable tasks. Reproduction recommendation: do not merge;
  compare only as scorer-boundary evidence.
- [Octavi 6042.85](https://www.kaggle.com/code/octaviograu/6042-85-per-task-hand-built-onnx-solvers),
  [Octavi 6154.71](https://www.kaggle.com/code/octaviograu/6154-71-onnx-rewrites-hand-built-solvers),
  [haoranran 6100](https://www.kaggle.com/code/haoranran/fp16-graph-surgery-v2-highest-public-6100),
  Biohack 6080/6113/super blends, and Massim EDA 111 are already represented in
  local source manifests or research notes. Current recommendation remains:
  cherry-pick only when there is an independent rule/safety story or prior
  source-specific LB proof.

## Lower-Signal Public Notebooks Screened

These appeared in live `kernels list` but did not expose a new high-score
downloadable artifact or technique beyond already known blending/tutorial paths:

- [ajayrao43/neuro-champ-2026](https://www.kaggle.com/code/ajayrao43/neuro-champ-2026),
  last run `2026-06-03 17:31:12`, low metadata signal.
- [ajayrao43/the-2026-neurogolf-championship11](https://www.kaggle.com/code/ajayrao43/the-2026-neurogolf-championship11),
  last run `2026-06-02 09:36:53`, low metadata signal.
- [rakshitverma16/arc-deep-dive-visualizing-400-reasoning-tasks](https://www.kaggle.com/code/rakshitverma16/arc-deep-dive-visualizing-400-reasoning-tasks),
  last run `2026-05-31 21:16:31`, visualization/EDA rather than artifact.
- [jsrdcht/glm-vs-opus-onnx-cost-opt-neurogolf-2026](https://www.kaggle.com/code/jsrdcht/glm-vs-opus-onnx-cost-opt-neurogolf-2026),
  last run `2026-05-31 11:28:47`, useful for model/workflow benchmarking but
  no public package to merge.
- Rau'uf Fauzan Rambe notebooks (`neurogolf-champions-arc-best-public-score`,
  `neurogolf-2026-the-best-score-kagglehub`, `neurogolf-training-submission-v40`)
  appear to be public-score/packaging notebooks. Existing local notes say the
  Champions notebook produced the same zip as Massim EDA 111.
- Older starter/EDA notebooks (`mmoffitt`, `yash9439`, `sigmaborov`,
  `karnakbaevarthur`, `cdeotte`, etc.) remain useful background but are below
  the current performance frontier.

## Discussion Audit

| Topic | URL | Date | Verified content | Implication |
|---|---|---:|---|---|
| NeuroGolf Update for April 21st | [discussion 693711](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711) | `2026-04-21 23:02:27` | Officially raised submissions/day to 100, ignored >30x30 cases, reported failing networks, patched negative memory, pinned `numpy 2.4.4`, `onnx 1.21.0`, `onnxruntime 1.24.4`, `onnx-tool 1.0.1`. Comments discuss dynamic-shape undercount and static-shape enforcement. | Old dynamic-shape and negative-memory tricks are stale; current scorer requires static positive dimensions. |
| NeuroGolf Update for April 28th | [discussion 695230](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230) | `2026-04-28 21:12:54` | Officially counted Constant values as params, enforced statically-defined shapes, changed memory to sum static shape bytes excluding input/output. Comments proposed adding `Compress` to excluded ops and rejecting nonstandard domains/subgraphs. | Constant-heavy public files must be rescored under current rules; parameter pruning/dedup is legitimate. |
| NeuroGolf Update for May 4th | [discussion 696953](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696953) | `2026-05-04 18:57:43` | Officially asserted positive tensor dimensions, improved memory using ORT-verified shapes, and eliminated MACs from the objective. Later comments document profiler trace filename clobbering and removal of `onnx_tool` from scorer. | Current objective is memory + params only. Local profiling must use unique trace prefixes; our tooling fix matches this. |
| There are still new exploits to be discovered | [discussion 699562](https://www.kaggle.com/competitions/neurogolf-2026/discussion/699562) | `2026-05-14 09:00:28` | A competitor reported a memory-cost exploit reducing task069 memory `44206 -> 625` and estimated `+400` to `+1000` possible if generalized. Host replied the exploit and another report were fixed on `2026-05-14`, with details in `neurogolf_utils.py`; undisclosed exploit use may be disqualifiable. | Treat pre-May-14 boundary packages as stale unless current probes prove them. Boundary research should be isolated and rules-aware. |
| Some thoughts on my agent workflow + sub-harness | [discussion 703914](https://www.kaggle.com/competitions/neurogolf-2026/discussion/703914) | `2026-06-02 16:26:22` | Competitor at about `6580` describes a multi-agent harness: terminal grid display, per-node cost breakdown, batch scans, candidate delta, initializer dumps, intermediate tensors, deployed-solution verification, semantic search over attempts/tools, task dossiers, claims, journals, staging, and Kaggle probe pipeline. Claims one 8-hour single-agent track gives about `+10` to `+15` LB/day, with accepted average `+0.36`, median `+0.21`; repeated passes up to about five times per task are needed. | Strong support for building an internal memory/search harness and repeated micro-golf workflow. This is probably more valuable than blind public package mining. |
| How are you using AI agents for Neurogolf? | [discussion 703854](https://www.kaggle.com/competitions/neurogolf-2026/discussion/703854) | `2026-06-02 07:49:06` | User at `6439.82` reports slow agent progress and high token use; reply points to top-model benchmarking. | Token efficiency and harness discipline are competitive differentiators. |
| My current bottom 15 NeuroGolf Tasks | [discussion 704006](https://www.kaggle.com/competitions/neurogolf-2026/discussion/704006) | `2026-06-02 20:46:49` | Listed hard/low tasks: `158,233,173,054,025,285,366,133,286,255,349,018,187,145,243`. Another competitor says their same 15 tasks score `232.16` vs poster's `227.716`; comments note BFS, copy-pasting, and shape recognition are hard. | Confirms our focus tasks (`233/255/285/366/018`) are public frontier tasks, but top teams still have headroom. |
| Token, Overfit, and Timeout | [discussion 704762](https://www.kaggle.com/competitions/neurogolf-2026/discussion/704762) | `2026-06-06 00:52:54` | Lists overfit-risk tasks `192,319,118,359,018,285,096,048,355,219`; slowest tasks `358,350,212,335,246,022,375,009,074,070`. Poster says synthetic private data can reduce overfit risk but sometimes rejects Kaggle-passing solutions. Comment notes no apparent private leaderboard and overfitting may be rational. | Matches our hidden-safety failures on 018/219/319-like tasks. Synthetic pseudo-hidden tests are useful but not authoritative. |
| How many tasks you got 20+? | [discussion 703232](https://www.kaggle.com/competitions/neurogolf-2026/discussion/703232) | `2026-05-29 12:45:04` | Competitors report distributions such as `51` tasks >=20 and another `101` tasks in the 20+ bands, with up to 3 tasks at 25. | Top teams are shrinking many tasks to tiny/static programs; our compiler should target 20+ score patterns. |
| Question for current #1 | [discussion 703431](https://www.kaggle.com/competitions/neurogolf-2026/discussion/703431) | `2026-05-31 05:48:28` | A competitor describes large gains from bypassing CNNs with pure functional tensor math and constant shift vectors, asking if leader uses DSL/rule-hardcoding. No substantive leader reply in pulled messages. | Inference, not fact: top path is likely program synthesis and hand-coded ONNX, consistent with local strategy docs. |

## Recommendations

1. Do not spend a Kaggle submission on any raw public bundle from this audit.
2. Extract and test only two new code ideas from the live ecosystem:
   ReduceSum-chain fusion and Cast-chain collapse from Vyanktesh's notebook.
3. Keep Seddik's uniform-scalar compression as a small lossless branch after
   the BOOL typed-IR work; ensure consumer-op broadcasting is explicit.
4. Add grouped/depthwise Conv templates to the DSL/compiler. Massim Part 4
   proves the pattern, and `task258` shows it can reach cost `160`.
5. Invest in the agent-harness features described publicly: per-node cost
   introspection, semantic search over attempts, task dossiers, and repeated
   passes. This is the most novel public signal not already in the handoff.
6. Treat scorer-boundary artifacts (`golf.Identity`, custom domains, old
   exploit-era public scores, validation-failure "GAMBLE" tasks) as research
   only. Any boundary probe must be isolated, tiny, and explicitly submitted
   only if the team chooses to spend a probe later.

## Changed Files

- `reports/public_ecosystem_audit_20260606.md`
