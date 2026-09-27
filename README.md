# NeuroGolf 2026 ONNX Research Archive

Sanitized research workspace for the Kaggle competition
`neurogolf-2026`. The project treats each ARC-style task as compact program
synthesis and compiles candidate rules into cost-conscious ONNX graphs.

## What this repository is

This is a working research record: small, inspectable tools, experiments, and
notes for studying compact ONNX programs for NeuroGolf tasks. It is designed
to be useful to people comparing ideas, extending a tool, or tracing how a
claim was tested—not as a mirror of the competition's files or of other
participants' artifacts.

The publication boundary is intentional. First-party code and original
research writing are shared here; competition data, leaderboard exports,
submission bundles, and third-party artifacts stay with their source. See
[DATA_SOURCES.md](DATA_SOURCES.md) for retrieval and provenance, and
[RELEASE_MANIFEST.md](RELEASE_MANIFEST.md) for the release boundary and
validation record. [PUBLICATION_BOUNDARY.md](PUBLICATION_BOUNDARY.md) records
the detailed review of retained research evidence versus excluded source files.

## Included

- first-party ONNX/DSL tooling and tests;
- task-analysis scripts and small reproducible diagnostics;
- research notes, score ledgers, and handoff documentation;
- build helpers that describe the local submission workflow.

## Intentionally excluded

- competition inputs, raw task data, and downloaded leaderboard exports;
- submission archives and generated ONNX bundles;
- copied public kernels and third-party artifacts;
- virtual environments, downloads, caches, profiler dumps, and multi-hundred-MB
  trace JSON files;
- credentials and machine-specific absolute paths.

For a source release, obtain data through the official competition after
accepting its terms, then pass its local directory to the relevant tool's
`--comp-dir` option. Do not commit that directory or its generated outputs.

## Research etiquette

Please share corrections, small reproductions, and alternative implementations
with clear provenance. Attribute upstream work, keep access-controlled material
out of issues and pull requests, and distinguish a local result from an
official leaderboard result. This archive keeps negative results on purpose:
they are often the shortest path to a better next experiment.

## Backup status

GitHub `main` is the verified remote copy of this sanitized archive. A separate
Kaggle Dataset upload was attempted, but it never appeared in the account's
dataset listing, so it is not counted as a verified backup.
