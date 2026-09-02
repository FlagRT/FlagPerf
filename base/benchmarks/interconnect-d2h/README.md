# Device-to-Host bandwidth

This Base benchmark measures one-way device-to-host tensor-copy bandwidth.
The timed loop reuses one device source tensor and one host destination tensor,
so allocation time is outside the measured interval. The reported payload is
`ITERS * Melements * 1024 * 1024 * sizeof(float32)`.

Vendor configuration selects whether the host destination is pageable or
pinned and whether `copy_` requests non-blocking execution. The final device
synchronization remains inside the measurement boundary.
