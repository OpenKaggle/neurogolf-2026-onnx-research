# NeuroGolf 2026 ONNX Research Archive

Sanitized research workspace for the Kaggle competition
`neurogolf-2026`. The project treats each ARC-style task as compact program
synthesis and compiles candidate rules into cost-conscious ONNX graphs.

## Included

- first-party ONNX/DSL tooling and tests;
- task-analysis scripts and small reproducible diagnostics;
- research notes, score ledgers, and handoff documentation;
- build helpers that describe the local submission workflow.

## Intentionally excluded

- competition inputs and raw task data;
- submission archives and generated ONNX bundles;
- copied public kernels and third-party artifacts;
- virtual environments, downloads, caches, profiler dumps, and multi-hundred-MB
  trace JSON files;
- credentials and machine-specific absolute paths.

The excluded artifacts remain in the private local workspace. See
`reports/HANDOFF_NEXT_AGENT.md` and `reports/research.md` for the technical
history and current direction.
