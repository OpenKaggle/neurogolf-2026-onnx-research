# Local workspace coverage (2026-09-28)

This note records the review of the local NeuroGolf ONNX workspace. The local
folder is an experiment lake; this repository is a curated source release, not
a byte-for-byte mirror.

## Reviewed inventory

At review time the local workspace contained approximately 2.6 GiB across
60,768 files. The public archive contains first-party tools, DSL code, tests,
research notes, and selected compact receipts. The large local artifact lake
is intentionally not committed into Git history.

| Local area | Review result | Public treatment |
| --- | --- | --- |
| `tools/` | First-party ONNX/DSL builders, audits, and tests | Publish after machine-path/credential scan; the reviewed source is already represented here. |
| `reports/` | Score ledgers, task diagnostics, profiler output, and some raw leaderboard exports | Publish compact derived receipts; pointer/hash only for raw exports and traces. |
| `data/` | Thousands of generated or downloaded ONNX task artifacts | Keep the bytes in an artifact release only after provenance and size review; Git receives an inventory/hash, not the whole bundle. |
| `submissions/` | Tens of thousands of candidate ONNX files and submission bundles | Do not copy wholesale into Git. Select first-party candidates for a separately versioned artifact package; retain the remainder locally until that package has fresh download/hash readback. |
| `downloads/`, `public_kernels/` | Downloaded competition/public-kernel material | Source-linked index only; upstream notebook and competition terms remain authoritative. |
| virtual environments and profiler caches | Machine-local dependencies and transient traces | Never publish. Recreate from documented dependencies. |

The official source remains the [NeuroGolf 2026 competition page](https://www.kaggle.com/competitions/neurogolf-2026).

## Why the ONNX bytes are not silently copied

An ONNX file may be our generated submission, a transformed public artifact,
or a third-party model. Filename and directory alone do not establish those
rights. The release process should therefore attach provenance, upstream
license, model lineage, byte hash, and a size-aware artifact URL to each model
family. A public source release can still reproduce the builders and tests
without redistributing every downloaded or generated model byte.

## What remains local

No local files were deleted during this review. Before freeing space, create a
manifest of excluded files (relative path, byte count, SHA-256, provenance
class), upload only the cleared artifact families to a separate host or Kaggle
Dataset, then fresh-download and compare hashes. Until that readback succeeds,
the local artifact lake is the recovery copy.
