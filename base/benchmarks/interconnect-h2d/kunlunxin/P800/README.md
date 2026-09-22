# Kunlunxin P800 H2D

The benchmark performs one-way host-to-device `Tensor.copy_` with preallocated
buffers. Pageable/pinned and blocking/non-blocking modes are separate contracts;
GB/s and GiB/s derive from the same one-way byte count.
