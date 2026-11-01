# What the search process failed to establish

Our Round 1 result was competitive. In Round 2 we built working systems, found
some lower-BPB recipes and observed small local CORE gains, but did not validate
the large capability improvement we were seeking.

## Technical feasibility and quality were different questions

Compilation, finite gradients, causality, checkpoint identity, memory fit and
time-limit compliance were necessary checks. Passing them did not demonstrate
better CORE. Too much progress reporting treated enabling a mechanism as if it
already supplied evidence for a competitive improvement.

## Our fast measurements were not a faithful CORE simulator

Short continuation runs, QA panels and token-loss measurements answered narrower
questions. We did not establish that their rankings predicted the sign or size
of four-hour CORE improvements. A retrospective loss-curve calibration existed
for BPB in a limited family; it did not calibrate CORE across architectures.

A faster instrument is useful only for the decision it has been shown to support.
A physical failure can reject a particular execution. A failed training run is
not automatically a negative learning result. A held-out NLL gain is not proof
of a CORE gain. These distinctions should have constrained promotion decisions
and communication earlier.

## We overstated causal explanations

Some explanations for margin losses, document boundaries and context decay were
presented too confidently. Removing one intervention while other differences
remain does not establish a unique cause. Likewise, 25 times more effective
updates do not imply 25 times more CORE improvement. A small learning-rate
integral does not prove that a model cannot acquire new capabilities.

The short grouping bank and the from-scratch grouping pair are a concrete example:
the former suggested a small gain; the latter failed. Their different data and
training histories prevent attribution to one mechanism, but clearly refute the
confident extrapolation from one setting to the other.

## Parallel work did not guarantee more useful decisions

Parallel code review and implementation helped establish correctness. They did
not remove the bottleneck of full-hardware evaluation or ensure that proposals
tested distinct, decision-relevant hypotheses. Occupying a queue, adding another
seed or repairing another platform is not inherently valuable research progress.

The relevant quantity was uncertainty resolved per unit of time and cost. Each
line needed a concrete hypothesis, a control, a decision threshold and a spending
limit. Negative results should close the tested recipe without being reinterpreted
as universal impossibility or as an invitation to unlimited parameter rescue.

## Cost control belonged to the complete operation

Host-only compute and persistent storage needed the same inventory, ownership
and expiry discipline as the accelerator. At closure, both training and CPU
instances were terminated, and obsolete environments were removed. Prepaid
capacity is not refunded by terminating an instance; preserving historical
artifacts can still incur storage charges.

These are retrospective judgments about our process, not controlled experiments
on research management. The archive supports the narrower conclusion that the
tested recipes did not achieve the target. It does not prove that CORE above 0.4
was impossible under the competition constraints.
