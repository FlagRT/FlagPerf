# Ascend Device-to-Host bandwidth

This configuration uses Torch-FL with an explicit `flagos:<local_rank>` device.
The timed loop copies one preallocated device source into one pinned host
destination, so allocation is outside the measurement and the byte formula is
one-way payload bytes.

The locked Torch-FL Ascend implementation currently lowers this `copy_` path to
blocking `aclrtMemcpy`; therefore `NON_BLOCKING` is deliberately `false`. The
result is blocking pinned-memory D2H bandwidth, not asynchronous overlap
performance. DMI remains a separate Toolkit test.
