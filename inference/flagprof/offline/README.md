# Default final capture and bounded offline import

CLI record, Python record(), and attach() default to final capture. Windowed
capture requires an explicit mode. Plan B runs automatically use bounded
inspect/finalize and StoreView paging/iteration. Explicit full-result APIs keep
their materializing return semantics.

Production capture, Store, durable work, scratch, and official analysis work
copies require ext4. workspace.py checks the actual Linux mount before creating
artifacts, including bind mounts. record's store is a delivery root containing
runs; delivery/work is a valid sibling. import's store is the runs root. Work,
scratch, Store, and sealed capture must remain separate trees. CLI seal places
its temporary copy beside the checked output rather than the default /tmp.

capture.py orchestrates normal target exit, immutable raw delivery, official SDK
analysis on a work copy, and a separate standard-library import process.
vendors/final_session.py owns one start and one stop/export. Compatible handle
step calls do not drive the SDK. Plain atexit does not establish normal exit.
The exited-owner worker handles explicit embedded attach collections; CLI and
Python record orchestrate the complete chain.

bundle.py binds raw and trace to a capture occurrence. service.py validates the
source, restores identity-bound committed stages, creates a new Store run,
imports through the existing Writer, independently verifies conservation, then
finalizes and checks paging. Writer behavior and durability remain unchanged.
The source-to-ledger/native/completion/Reader checks preserve population,
identity, order, relationships, and tails. SDK capture loss remains independently
unknown when reliable counters are unavailable.

Verification streams array payloads directly and checks references during the
ledger/native traversal. The view keeps independently parsed metadata until it
closes. A successful finalized view publishes a rebuildable verification receipt
in query work, outside Store and capture. Reopening an unchanged local delivery
reuses that result; changed file identities, code/runtime versions, or missing or
damaged receipts trigger full verification. Read-only cache failures also fall
back to full verification. Event paging still uses Reader shard integrity checks.
Standalone bounded_verify.verify always performs the complete check and retains
its comparison digests. See docs/validation/t12.3/report.md.

pipeline.py and disk.py include the accepted flow join, exact integer interval,
and Dequeue query fixes. Scratch SQLite is rebuildable; checkpoints and capture
originals are durable. New v2 registration/checkpoints use checked relative paths
for whole-delivery migration. Historical v1 absolute paths stay read-compatible.
Never resume a failed Store run or edit old checkpoints to bypass identity.
Only successful owned temporary files are cleaned; failures are retained.

limits.py bounds metadata, file cardinality, and the one-trace Store contract.
finish.py supplies bounded dependencies to the finalizer; no global patching is
used. Historical helpers in context.py are not the production preparation path.
No production module depends on validation harnesses.

See docs/plan-b-usage.md for deployment and API boundaries, and
docs/validation/d33-default-integration-20261026/ for the integration evidence and
T9.7 handoff. The user accepted initial short-trace performance costs; historical
throughput failures remain recorded. Writer Gate, Stage 1, P0, and long-stability
are separate decisions.
