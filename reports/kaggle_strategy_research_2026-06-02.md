# Kaggle Strategy Research Notes - 2026-06-02

## Why This Exists

We are currently at official `6122.59` on `neurogolf-2026`. External cherry-pick gains have mostly been exhausted or rejected, so the next improvements must be chosen carefully. This note captures general Kaggle competition tactics from public community sources and maps them to the current NeuroGolf workflow.

## Sources Checked

- Kaggle official competition docs: https://www.kaggle.com/docs/competitions
- Kaggle competition forum page for NeuroGolf: https://www.kaggle.com/competitions/neurogolf-2026/discussion
- Hacker News Algolia search for Kaggle/leaderboard/ensembling discussions:
  - https://hn.algolia.com/api/v1/search?query=Kaggle%20public%20leaderboard%20overfitting
  - https://hn.algolia.com/api/v1/search?query=Kaggle%20competitions%20tips
- HN thread surfaced by search: Kaggle Ensembling Guide, https://news.ycombinator.com/item?id=9720314
- HN thread surfaced by search: Machine learning isn't Kaggle competitions, https://news.ycombinator.com/item?id=8247389
- Neptune blog surfaced by HN search: Image segmentation tips from Kaggle competitions, https://neptune.ai/blog/image-segmentation-tips-and-tricks-from-kaggle-competitions

Reddit JSON search was blocked with HTTP 403 in the local script, so Reddit was not used as an automated source this round. Kaggle dynamic forum content also was not available through a simple unauthenticated API endpoint.

## General Tricks That Apply Here

### 1. Start With A Strong Public Baseline

For a normal Kaggle competition, this usually means public notebooks, reproduced baselines, and fast local validation. In NeuroGolf this already worked:

- Octavi 6042 anchor gave a reliable starting point.
- Public high-scoring bundles were useful as inspection material.
- The biggest early lift came from safe, measurable compression and a few verified single-task replacements.

Actionable rule for us: public sources are allowed as baselines or templates, but any copied ONNX must be treated as a source-specific risk until official single-task probes confirm it.

### 2. Keep A Trust Ledger Per Source

The current source ledger is now decisive:

- `biohack-6113`: 11/11 single-task probes matched local/official within about 0.01, now considered hidden-safe, but remaining gain is only noise-scale.
- `konbu17`, `afr1ste`, `octv`: hidden-unsafe. Even single-task or small monster swaps can pass all local examples and still regress official score badly.

Actionable rule for us: no more staged konbu/afr/octv zips. A high bundle score is not evidence that its individual networks are safe when cherry-picked.

### 3. When Stuck, Optimize The Scoring Surface Directly

Community advice around Kaggle competitions often converges on exploiting the exact evaluation metric and improving the validation loop. In NeuroGolf, this means:

- Cost reductions are often more reliable than algorithm changes when `same_decoded_outputs` holds.
- We already harvested int narrowing, onnxoptimizer, onnxsim, fp16 surgery, and safe biohack swaps.
- The residual compression gain is now below practical submission value.

Actionable rule for us: do not keep squeezing generic optimizers unless a new pass gives at least about `+0.05` local with full `same_decoded_outputs`.

### 4. Beware Visible-Set Memorization

This competition has many local examples, and ONNX can cheaply memorize them. That can produce perfect local pass and low cost while failing hidden/generalization.

Current concrete example:

- `data/konbu17_v36/task363.onnx` / `submissions/staging_max_safe/task363.onnx`
- Full visible validation: `265/265`
- Cost: `54,241`, much better than current `task363`
- Graph structure: slices channels 1/2/5, hashes/embeds the input, compares against 265 stored signatures, then emits a stored output table.

Conclusion: this is not a semantic solver. It should not be submitted despite attractive local score.

### 5. Spend Submissions On Questions, Not Hope

Good submission questions:

- Does a source already proven safe remain safe for a small positive candidate?
- Does a semantic hand-build with full local pass preserve official score?
- Does a lossless optimizer pass match official cost?

Bad submission questions:

- Maybe this unsafe source is safe for this one task.
- Maybe a visible-example memorizer generalizes.
- Maybe a bundle-level high score means cherry-pick safety.

## Direct Next Steps

1. `task219` semantic work is still the highest expected value. Current prototype is `245/265`; the bottom-align variant fixes 8 examples but breaks 10, proving at least two fragment alignment classes.
2. `task363` needs a true semantic ONNX, not the konbu memorizer. The Python rule is closed, but ONNX expression remains hard.
3. `task285/366/255/319` remain the real route to large jumps. Any full Python rule for one of these can be worth much more than all remaining compression.

