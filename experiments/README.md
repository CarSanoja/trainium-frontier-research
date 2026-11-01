# Diagnostic experiments

## Causal prefix cache

[Source](causal-prefix-cache/) contains the sparse suffix-matching policy, model
wrapper and original CPU contract test. It imports the selected package's model
through `inference`; it is not an independent model distribution.

On a fixed local 274 checkpoint the policy added 0.001401 CORE and 0.001126 BPB.
The original wrapper also had an invalid checkpoint parameter-name report. It
was a diagnostic and was not promoted to an eligible submission. The small gain
does not establish a general improvement from inference-time copying.

## Shared frozen prefix and two adapted tails

[Source](shared-prefix-tails/) comes from the final 1 November pilot. Its modules
split the model, fit suffix adapters, evaluate held-out likelihood and apply the
predeclared result rule. Transport and cloud scripts, checkpoints and document
manifests are not published. Its entry points assume those local inputs exist.

The rank-matched mixture improved slightly over an adapted single-tail control,
but both were worse than the original checkpoint on held-out NLL. Length 1024
was tested; 2048 did not pass the preceding identity check. CORE was not measured.

See [results](../docs/results.md) for the complete numerical comparison and
[retrospective](../docs/retrospective.md) for the limits of extrapolating proxies.
