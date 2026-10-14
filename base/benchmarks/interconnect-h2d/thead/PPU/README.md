# interconnect-h2d / thead PPU - bounded copy variants

Chip assets for the single-card host-to-device copy case. The base config is the
pageable blocking variant; the measured-* overrides add pinned and
non-blocking combinations at the same 512 MiB payload. All variants run MODE
measured and enter the qualification account through the measured-mode gates.

Window policy: the day-5 calibration run (case_config.calibration.yaml, 64
pinned blocking iterations) fixes per-variant ITERS so every measured sample
stays above the 15 s qualification floor; ITERS is bounded by 200000 in the
transfer contract. A variant that cannot reach the floor within the bound is
recorded as a documented exception in the day-5 execution record and the
coverage account - it is never silently shortened.
