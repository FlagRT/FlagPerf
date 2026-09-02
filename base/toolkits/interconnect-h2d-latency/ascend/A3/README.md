# Ascend A3 Host→Device 时延实验

按逻辑 Device 逐点执行 `ascend-dmi -l -t h2d -s <bytes> -d <id> -q --fmt json`。
默认负载为 512 B、4 KiB、64 KiB、1 MiB，可由 `--latency-sizes` 覆盖。
每个点独立保留命令和输出；结果单位为 ns。DMI 当前没有与该 Case 直接对应的
阈值诊断，因此诊断状态记录为 `not-run`，不影响测量状态。
