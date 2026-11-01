# Publication and provenance

This public repository starts with a new Git history and an explicit source
allowlist. The original repositories remain private archives. Publication was
authorized by the project owner when closing participation.

Included:

- Selected participant training and inference implementations from both rounds.
- Two diagnostics that illustrate small or negative results.
- Aggregate numerical outcomes and their interpretation limits.
- SHA-256 digests of copied sources and references to historical evidence records.
- A retrospective and a guide to the missing requirements for reproduction.

Excluded:

- Credentials, tokens, account IDs, private hostnames, addresses and billing data.
- AWS launchers, reservation automation, resource inventories and private logs.
- Raw training shards, tokenizer artifacts, evaluation bundles, prompts or answers.
- Checkpoints, learned adapters and document/token manifests.
- Private conversations and copied third-party research caches.

The source files listed in `SOURCE_MANIFEST.json` are byte-identical snapshots;
documentation and aggregate result records were written for publication. Their
source identifiers are relative archive references, not links granting access to
the private repositories. Hashes allow an authorized holder of the original
records to compare provenance; they are not independent certification of scores.

The release was reviewed with an offline integrity checker, secret scanning and
a check for private infrastructure identifiers. These checks reduce disclosure
risk; they do not certify all historical code as secure or production-ready.

The campaign is closed. There are no scheduled workflows or cloud provisioning
scripts in this public repository. Model code is historical research software.
See [reproduction](reproduction.md) before attempting to run it.
