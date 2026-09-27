# Release manifest

## Scope

This is a source-and-research release for `OpenKaggle/neurogolf-2026-onnx-research`.
It contains first-party tooling, research writing, aggregate diagnostics, and
provenance notes. It is not a data release, a submission-artifact release, or
a mirror of public kernels.

## Excluded categories

- Kaggle competition inputs and exact task records;
- downloaded leaderboard archives and private/hidden evaluation material;
- generated submission ZIPs, ONNX bundles, checkpoints, and caches;
- third-party kernels, model weights, and copied artifacts;
- credentials, environment files, and machine-specific paths;
- literal competition records, including downloaded input/output grids.

`DATA_SOURCES.md` is the canonical guide for retrieving authorized source
material and for the historic leaderboard-export checksums.

## Publication checks

Before this release was published, the tree was checked for tracked `data/`,
`submissions/`, `.onnx`, and `.zip` assets; none remain. The two downloaded
leaderboard ZIP exports were removed from the working tree and from reachable
`main` history. First-party per-case pseudo-hidden records were reviewed and
retained because they contain validation metrics and compact diagnostics, not
literal competition rows. Files at or above 10 MiB were also checked; none
remain.

The twelve JSON receipts in `evidence/derived-runs/` were added after the
history cleanup. They are first-party local evaluator metrics for two candidate
families. They contain no task grids, inputs, outputs, ONNX payloads, or
submission artifacts; the accompanying README explains their fields and limits.

The remaining code may refer to ignored local paths in order to support an
authorized reproducibility workflow. Such references are instructions, not
bundled data. Anyone reusing the code is responsible for complying with Kaggle
terms and the licenses of any upstream artifact they obtain.
