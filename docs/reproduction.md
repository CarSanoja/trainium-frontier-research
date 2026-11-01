# Reproduction boundaries

## Safe offline inspection

Run `python3 scripts/verify_archive.py` from the repository root. It verifies the
copied-source manifest, parses Python without importing it, reads aggregate result
JSON and checks local Markdown links. It does not install dependencies, create
cloud resources, download data, compile Neuron graphs or train a model.

This publication was checked this way. The historical experiments were not rerun
after shutdown. A passing archive check is not a hardware or quality validation.

## Recover the experiment environment

1. Select the package in the [package guide](../packages/README.md).
2. Obtain the organizer's [starter kit](https://github.com/aws-neuron/trainium-frontier)
   and the Neuron environment specified for the relevant round. Round 2's locally
   recorded kit revision is `2cdad297f0e4cffc39b2f650135a041224e3947e`.
   Historical official runs may have used the organizer's contemporaneous runner;
   the package source alone does not pin every server dependency.
3. Use the matching fixed `prepare.py`, tokenizer and evaluation code from the
   organizer. They are not copied into this release. Obtain data and any CORE
   bundle under their original access and license conditions.
4. Inspect `BANDERAS` and `COMANDO` for a Round 2 package, or `BUILD`, `SEMILLA` and
   `COMANDO` for Round 1. These preserve the recipe and launch form. No account,
   instance or reservation is supplied by this repository.
5. Place the modular package beside the matching fixed harness. Alternatively,
   use the original flattened `envio/train.py` and `envio/inference.py` where
   supplied. Preserve the flags and do not silently mix versions or packages.

The native Neuron runtime, compiler, PyTorch integration and hardware are required
for the accelerator paths. Installing ordinary CPU PyTorch is insufficient.
The organizer's dependency file complements its environment; it is not a complete
lockfile. We do not claim a fresh environment can reproduce these scores solely
from this archive.

## Keep the scientific comparison meaningful

- Preserve tokenizer, data-shard selection, packing, sequence length, seed, total
  training budget and charged-time convention for the comparison being tested.
- Evaluate BPB and CORE on the same model. Do not merge metrics from different
  checkpoints, add task detection or alter the evaluation harness.
- Distinguish continuation from training from scratch and local evaluation from
  official evaluation. Record all relevant changes rather than attributing their
  combined effect to one mechanism.
- Report timeouts, NaNs, invalid checkpoint reports and identity failures as such.
  Do not convert them into zero quality scores or evidence against a model family.
- Keep raw CORE content out of training. The aggregate measurements here contain
  no prompts, examples, answers or token sequences.

The shared-prefix pilot additionally requires its fitted checkpoint and frozen
training-document manifest to reproduce the exact values. They are deliberately
not distributed. Its source can be inspected, but the public archive is not a
self-contained replay of that run.

## Interpreting BPB and CORE

BPB averages predictive loss over evaluation text. CORE aggregates task scores
from a separate battery. Better average token likelihood does not mathematically
force better answers on each task. Changes to representation, context use and
calibration can affect the two differently. Training affects both; there is no
validated independent CORE adjustment knob in this work.
