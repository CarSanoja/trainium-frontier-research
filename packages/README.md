# Source snapshots

| Directory | Role | Status of the reported measurement |
| --- | --- | --- |
| [phase1-48](phase1-48/) | Round 1 selected submission | Recorded official result |
| [phase2-250](phase2-250/) | Best recorded combined-score Round 2 entry | Recorded official result |
| [phase2-263](phase2-263/) | Wider variant with the lowest recorded BPB in this selection | Recorded official result, lower CORE |
| [phase2-274](phase2-274/) | Document-summary variant and later diagnostic checkpoint family | Official result and separate local runs |
| [phase2-302](phase2-302/) | Short continuation bank and grouping machinery | Local experimental carrier |
| [phase2-338](phase2-338/) | Related-document grouping from scratch | Local full-budget negative result |
| [phase2-339](phase2-339/) | Random grouping control for 338 | Local full-budget result |
| [phase2-356](phase2-356/) | D40 recipe with corrected charged-time accounting | Local full-budget negative result |

`BANDERAS` records environment flags; `COMANDO` records the launch form. These are
historical experiment artifacts, not instructions to provision a machine. Round 1
uses `BUILD` and `SEMILLA` for additional configuration provenance.

The Round 2 packages include readable modules and, where present, `envio/` with
the flattened source used by the two-file submission interface. The latter writes
its embedded modules locally before importing them. This explains its size and
duplication; do not substitute one package's embedded modules for another's.

The organizer's `prepare.py`, data, tokenizer and evaluation bundle are external
requirements. Merely copying a package or running a syntax check does not validate
the hardware path, eligibility or reported score. See [reproduction](../docs/reproduction.md).
