# Trainium Frontier — Team Quanta research archive

Source, measurements and lessons from our 2026 AWS Trainium Frontier participation.
The active campaign closed on **1 November 2026**. This repository preserves useful
work and unsuccessful experiments; it does not claim a new state of the art.

[Resumen en español](README.es.md) · [Results](docs/results.md) ·
[Architecture](docs/architecture.md) · [Reproduction](docs/reproduction.md) ·
[Retrospective](docs/retrospective.md)

## What we achieved

- **Round 1:** our recorded final leaderboard observation placed Team Quanta third,
  with **0.9471819509 validation bits per byte**, using submission 48.
- **Round 2:** our best recorded combined-score submission, 250, obtained
  **0.7233577472 BPB / 0.2992382166 CORE**. Submission 263 improved BPB to
  **0.7166185501**, but its CORE fell to **0.2879724035**.
- Small local CORE improvements did not establish a recipe for a large competitive
  gain. We did **not** demonstrate CORE above 0.4. Round 2 was still open when we
  stopped, so this archive makes no claim about the competition's final ranking.

## What is here

| Path | Contents |
| --- | --- |
| [packages/](packages/README.md) | Eight source snapshots: Round 1 submission 48 and Round 2 packages 250, 263, 274, 302, 338, 339 and 356 |
| [experiments/](experiments/README.md) | Causal prefix cache and shared-prefix/two-tail proof of concept |
| [results/measurements.json](results/measurements.json) | Aggregate measurements with scope and interpretation limits |
| [docs/](docs/README.md) | Design, evaluation, negative results, reproducibility and publication policy |
| [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) | SHA-256 provenance for every copied source file |
| [scripts/verify_archive.py](scripts/verify_archive.py) | Offline integrity, Python syntax and local documentation-link checks |

The source snapshots are byte-identical to the selected local files. `envio/`
contains flattened submission files when the original package provided them;
the adjacent modules expose the same implementation in a more readable form.
See the package guide for the distinction between official submissions and local
experiments.

## Inspect without starting any compute

```bash
git clone https://github.com/CarSanoja/trainium-frontier-research.git
cd trainium-frontier-research
python3 scripts/verify_archive.py
```

The verifier uses the Python standard library, does not import the training code,
and makes no network or cloud calls. Reproducing training requires the matching
AWS Neuron environment, organizer data and evaluation harness; this archive does
not include a one-command cloud launcher.

## Scope and attribution

This is a curated public release. Private operational logs, account identifiers,
credentials, billing records, raw dataset/evaluation content and model weights
are excluded. Historical code can require unavailable or version-specific Neuron
components; current hardware compatibility has not been retested for publication.

Our code builds on the [AWS starter kit](https://github.com/aws-neuron/trainium-frontier).
See [LICENSE](LICENSE), [NOTICE](NOTICE), [third-party attribution](third_party/README.md)
and [publication policy](docs/publication.md). Original comments and identifiers
are often in Spanish. The license does not grant rights to datasets or software
dependencies that are not distributed here.
