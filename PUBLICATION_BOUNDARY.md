# Publication-boundary review

Reviewed 2026-09-27 against the live
[NeuroGolf competition rules](https://www.kaggle.com/competitions/neurogolf-2026/rules).
The competition-specific data-security rule says participants must not
transmit, duplicate, publish, redistribute, or make Competition Data available
to people who have not agreed to the rules. The same rules describe an
Apache-2.0 data-use license and an open-source obligation for winners' code,
but neither statement turns a downloaded competition archive or a third-party
leaderboard export into a repository asset.

This repository is a voluntary research release, not a claim to be a winning
submission or to satisfy a winner's delivery obligation.

## Reviewed artifacts

| Artifact group | Classification | Publication decision | Evidence for the decision |
| --- | --- | --- | --- |
| `reports/task133_v2_*pseudo_hidden.csv`, `reports/task285_v2_pseudo_hidden.csv`, `reports/task366_v2_pseudo_hidden.csv` | First-party validation replay records | Retained | Their columns are source-order handle/label, contract, result, shape summary, mismatch count, and rule label; there are no serialized grids or task JSON records. |
| `reports/task255_v2_pseudo_hidden.csv`, `reports/task255_v2_pseudo_hidden.json`, and summary CSV | First-party validation replay records | Retained | 6,893 per-case result rows plus a 13-row summary; row fields are test variant and metrics. Diagnostic details contain no raw `train`, `test`, `input`, or `output` fields and no grid-shaped values. |
| `reports/leaderboard/neurogolf-2026.zip` and `reports/leaderboard/current/neurogolf-2026.zip` | Downloaded Kaggle leaderboard exports containing third-party participant records | Removed | Each ZIP is an external public-leaderboard CSV export. Its historic source, timestamp, and SHA-256 are retained in `DATA_SOURCES.md`; the copies themselves are not published. |
| `data/`, `submissions/`, downloaded ONNX bundles, and raw task files | Competition/third-party files | Excluded | No such files are tracked in this release. Tools may reference an ignored local path solely to support authorized reproduction. |

## Review rule

Keep detailed, useful evidence of work performed by this project: run records,
metrics, decision ledgers, compiled-code provenance, and validation/replay
results. Exclude literal downloaded files, copied third-party artifacts,
credentials, and anything the competition rules expressly restrict. Possible
inference from a research result alone is not a reason to discard the result.
