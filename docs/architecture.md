# What the code explores

The project trained language models from scratch under a fixed hardware-time
budget. Round 1 used one Trainium2 chip for 30 minutes. Round 2 used the full
Trainium2 server for four hours and evaluated both language-model loss and CORE.
The published snapshots preserve the actual implementations; this document is
a navigation aid, not a claim that every optional switch improved quality.

## Transformer and kernels

The participant model extends the organizer's decoder-only Transformer. Source
files expose attention, positional encoding, residual paths, value embeddings,
the output head and optimizer logic. The selected recipes include reduced-
precision MLP operations, custom NKI kernels and distributed optimizer work.

Start with [phase2-250/train.py](../packages/phase2-250/train.py), its
[kernels](../packages/phase2-250/kernels/) and [BANDERAS](../packages/phase2-250/BANDERAS).
The flags identify the active recipe; the presence of an experimental module in
the package does not mean that every branch was enabled in the measured run.

## Host-resident learned memory

MEMHOST uses causal token-context keys to address a large memory table kept in
host memory. Trainable projections and gates combine retrieved features with
the Transformer. The implementation includes lookup, synchronization, optimizer
updates, checkpoint consolidation and evaluation loading.

Useful entry points in package 250:

- `qf2_memhost_hash.py`: causal context indices and hashing.
- `qf2_memhost_tabla.py`, `qf2_memhost_hilo.py`: table access and host work.
- `qf2_memhost_opt.py`, `qf2_memhost_sincro.py`: updates and coordination.
- `qf2_memhost_sitios.py`: projections and gated injection sites.
- `qf2_memhost_ckpt.py`: checkpoint materialization and evaluation access.

This design trades accelerator memory for host memory and communication. A table
being large is not evidence of reasoning capacity. Trained parameters and stored
data also have different meanings under the competition's checkpoint rules; see
the organizer's specification rather than inferring eligibility from tensor size.

## Document context and grouping

Package 274 adds document-summary context. Package 302 exposes the continuation
bank machinery, including related-document grouping. Packages 338 and 339 form
the later from-scratch related-versus-random pair. The document boundary, data
source, sampling order, training history and optimizer schedule all matter when
interpreting these comparisons. These are not interchangeable experimental units.

## Greater depth and optimizer precision

Package 356 preserves the D40 recipe with its corrected charged-time accounting.
The run completed 5,632 steps, with sampled median step time 2.49 seconds, and
passed its physical execution checks. Its quality result remained negative.
More layers consumed time that could otherwise train on more tokens. The run
does not isolate depth from all other recipe changes.

## Evaluation-only and adaptation experiments

The [causal prefix cache](../experiments/causal-prefix-cache/) uses only prior
tokens in each example to add sparse continuation support to logits. It resets
at document boundaries and does not retrieve another example's state. Its fixed
orders 2–8 and weight were tested as one policy, with mixed task-level effects.

The [two-tail pilot](../experiments/shared-prefix-tails/) tests shared prefix
computation and two adapted suffixes. The final implementation and equal-capacity
control are preserved. Its negative held-out NLL result is distinct from a CORE
result, and its auxiliary scripts assume a prepared model and local data files.

See [results](results.md) for what was measured and [reproduction](reproduction.md)
for the environment that these historical sources require.
