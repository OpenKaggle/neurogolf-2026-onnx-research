# Derived local-run receipts

This directory preserves twelve first-party, per-task local validation receipts
from two independently built candidate families: `anchor` and `konbu`.

Each receipt records only run-time metrics produced by the local evaluator
(score, resource counters, program size, and local pass count).  It does not
include competition inputs, target outputs, ONNX binaries, submitted ZIPs, or
downloaded leaderboard exports.  `task` is the evaluator task label, and
`source` is the original local candidate label retained for provenance rather
than a path to any released data.

These receipts are evidence of a historical local run, not a claim about a
private leaderboard result. Re-run the documented tools with officially
obtained competition access to produce a new receipt.
