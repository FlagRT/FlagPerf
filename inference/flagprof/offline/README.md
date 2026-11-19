# Final capture and bounded offline import

A final capture seals the raw output of one normally closed SDK session. The
vendor adapter interprets the sealed export. The importer checks its source
and publishes a verified Store run. Capture loss and performance qualification
are independently recorded when evidence is available.

Choose the results directory with `--store`. Imports put validated scan and
prepared checkpoints under `.flagprof-<uid>/checkpoints` on that volume.
The key includes the capture occurrence, sealed content, import options and
the implementation and contract digest. Retry with the same capture and
settings to reuse valid stages. A changed capture fails validation. A damaged
derived checkpoint is rebuilt. The final run becomes visible after verification. Retrying a named run validates its completed result. A conflicting import using the same run ID fails.

FlagPerf owns its private attempt directories and excludes them from delivery
archives. Inactive checkpoints may be evicted after 30 days or under space
pressure. Recoverable intermediates use the result volume. Each inspection
and page query obtains a private system temporary directory. Reports use their
output volume, including SQLite temporary files. These operations can read
sealed evidence from a read-only mount.

See the [usage guide](../../docs/usage.md) for installation, commands and
the portable archive workflow.
