# Day 8 exploratory multicard extension

Date: 2026-09-28. This is an exploratory continuation of Day 7, authorized by the user to use cards with existing processes. No external process was terminated and no device reset, driver, or firmware change was performed.

## Code changes

- communication runtime scope now accepts MPI world sizes 2 through 8; P2P remains intentionally two rank because its benchmark semantics are rank0 to rank1;
- rank validation, bounded worker, stdout validation, AllReduce correctness reference, and bus bandwidth validation use the configured world size;
- occupied devices can proceed only with `P800_ALLOW_FOREIGN_HANDLES=1`, and the preflight summary records the authorization and observed occupancy; default behavior remains fail closed;
- sudo was used only through the existing user supplied password to permit the runner's documented `sudo -n` command set.

## Results

- `mpi-2-regression`: passed execution, correctness, measurement evidence, postflight, cleanup, and lease release on physical cards 5 and 6.
- `mpi-8-smoke-2`: all eight ranks passed runtime UUID and tensor binding checks and connected through Gloo; FlagCX collective initialization/workload timed out. The framework recorded failure, monitoring partial, cleanup passed, postflight passed, and lease release passed.

The eight card result is exploratory and does not qualify the runtime.
