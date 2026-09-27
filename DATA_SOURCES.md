# Data sources and provenance

This repository documents methods and research conclusions. It does **not**
redistribute competition inputs, hidden-evaluation material, leaderboard
exports, submission archives, third-party model artifacts, or copies of other
participants' work.

## Official competition material

- Official competition entrypoint: <https://www.kaggle.com/competitions/neurogolf-2026>
- Official discussion area: <https://www.kaggle.com/competitions/neurogolf-2026/discussion>
- Kaggle competition documentation: <https://www.kaggle.com/docs/competitions>

Access requirements, terms, and availability are controlled by Kaggle. After
accepting the competition terms, download material through Kaggle and point a
tool at the resulting local directory (for example,
`--comp-dir /path/to/neurogolf-2026`). The expected local layout is described
only to make the tools runnable; it is ignored by Git and must not be added to
issues, pull requests, or releases.

## Historical leaderboard-export provenance

Two downloaded leaderboard ZIP snapshots were previously present in this
repository. They were removed from the working tree and rewritten out of the
public `main` history on 2026-09-27. The table retains only reproducibility
metadata, not the exports themselves.

| Former local path | Canonical source | Source snapshot date | SHA-256 before removal |
| --- | --- | --- | --- |
| `reports/leaderboard/neurogolf-2026.zip` | [NeuroGolf leaderboard](https://www.kaggle.com/competitions/neurogolf-2026/leaderboard) | 2026-09-17 (archive commit date) | `eccbd6ce25b2d98db2096c60e3ebc153d844de033e4de753a8f02978bdab9c2b` |
| `reports/leaderboard/current/neurogolf-2026.zip` | [NeuroGolf leaderboard](https://www.kaggle.com/competitions/neurogolf-2026/leaderboard) | 2026-09-17 (archive commit date) | `a7ad4f29fcf99ace76470a90de69ce316f0e1e86342605b328ffaaa55538c212` |

The checksums identify the historic local snapshots; they do not create a
right to redistribute them or guarantee that a signed-in Kaggle export remains
available.

## Derived research records

Aggregate metrics, source citations, and first-party analysis may be published
when they do not reveal exact competition records. Per-case pseudo-hidden test
matrices remain local-only. The code that generates those matrices is public,
so a participant with authorized data can reproduce the evaluation under the
competition's terms.
