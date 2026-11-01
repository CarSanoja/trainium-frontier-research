# Results and limits

BPB is validation bits per byte: lower is better. CORE is a mean of centered task
scores: higher is better. They measure different properties of the same model.
The Round 2 competition combined both metrics after normalization across entries.
See the [organizer's rules](https://trainium-frontier.devpost.com/rules) and
[pinned Round 2 specification](https://github.com/aws-neuron/trainium-frontier/blob/2cdad297f0e4cffc39b2f650135a041224e3947e/UPDATED_RULES.md).

## Official scores recorded during participation

| Package | Scope | BPB | CORE |
| --- | --- | ---: | ---: |
| 48 | Round 1, final recorded third place | 0.9471819509 | Not scored |
| 250 | Round 2, best recorded combined score | 0.7233577472 | 0.2992382166 |
| 263 | Round 2, lower BPB but lower CORE | 0.7166185501 | 0.2879724035 |
| 274 | Round 2, document-summary variant | 0.7184405240 | 0.2936541564 |

These are recorded official results, not newly queried leaderboard values. The
two best component values belong to different models and cannot be combined into
a fictitious winning score. Round 1 and Round 2 budgets differ; their BPB values
are not a controlled comparison of one intervention.

## Local experiments

| Experiment | BPB | CORE | Decision |
| --- | ---: | ---: | --- |
| Short-bank control 01 | 0.718494 | 0.3004 | Reference, rounded record |
| Short-bank related grouping 02 | 0.719853 | 0.3060 | Small observed gain |
| Related grouping repeat 09 | — | 0.3066 | Small observed gain |
| 339, random grouping from scratch | 0.7350489796 | 0.2720526672 | Did not meet promotion target |
| 338, related grouping from scratch | 0.7353114247 | 0.2578696270 | Worse than paired random control |
| 356, D40 full-budget recipe | 0.7330519841 | 0.2857482746 | Did not meet promotion target |
| 274, fresh cache experiment control | 0.7223779412 | 0.3009508642 | Paired reference |
| 274 with causal prefix cache | 0.7235036895 | 0.3023522474 | CORE gain too small for preregistered gate |

The short-bank scores were continuation experiments. The later grouping pair
trained from scratch with a different data-supply protocol. Their different
outcomes invalidate a simple extrapolation; they do not isolate which change
caused the common deterioration. Two controls using different training shards
also differed by 0.0061 CORE. That single contrast is not a universal error bar or
a significance threshold for every task.

The cache added **0.0014013832 CORE** and **0.0011257483 BPB** on one fixed
checkpoint. Its original diagnostic wrapper failed the checkpoint parameter-name
accounting convention. It was not an eligible submission. This specific policy
was closed; the result does not rule out every causal cache design.

## Shared-prefix, two-tail proof of concept

The final pilot used the 356 checkpoint: a frozen 36-block prefix and two
four-block suffix adaptations. Two rank-4 LoRA branches were mixed with fixed
weight 0.5 and compared with one rank-8 branch of equal total trainable capacity.
Each arm received 200 updates; fitting used 64 training documents and held-out
evaluation used 32 other training documents. These were not CORE examples.

| Model | Held-out token NLL, lower is better |
| --- | ---: |
| Original model | 2.0535412887111306 |
| Split, unadapted model | 2.0535324020311236 |
| Single rank-8 adapted control | 2.1058224830776453 |
| Two rank-4 adapted branches, fixed mixture | 2.1012134682387114 |

The mixture was worse than the original by **0.0476721795 NLL**. It was better
than the adapted control by only **0.0046090148**, short of the declared 0.01
improvement requirement. Fitting loss decreased while held-out loss increased.
That is compatible with poor generalization, without identifying its sole cause.

The final pilot ran at length 1024. Length 2048 did not pass the preceding identity
check and was not validated. **No CORE or official BPB was measured for this POC.**
The final run took 519.42 seconds; this excludes the earlier engineering and
failed attempts needed to get there.

Exact machine-readable values, scope labels and record provenance are in
[measurements.json](../results/measurements.json). Aggregate scores alone do not
establish statistical significance, transfer to a different recipe or causality.
