# Ascend Host-to-Device bandwidth

This configuration uses Torch-FL with an explicit `flagos:<local_rank>` device.
The host source uses Torch-FL's pinned host allocator and the timed loop copies
into a preallocated device destination, so device allocation is not part of
the measured bandwidth.

The locked Torch-FL Ascend implementation currently lowers this `copy_` path to
blocking `aclrtMemcpy`; therefore `NON_BLOCKING` is deliberately `false`. The
result is blocking pinned-memory H2D bandwidth, not asynchronous overlap
performance. DMI remains a separate Toolkit test.
