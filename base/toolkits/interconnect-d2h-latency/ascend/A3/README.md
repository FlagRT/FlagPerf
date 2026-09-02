# Ascend A3 Device→Host 时延实验

按逻辑 Device 逐点执行 `ascend-dmi -l -t d2h -s <bytes> -d <id> -q --fmt json`。
默认负载为 512 B、4 KiB、64 KiB、1 MiB，可由 `--latency-sizes` 覆盖。
结果保存 ns、传输大小、Device、命令和原始输出；无直接 DMI 阈值时诊断明确为
`not-run`，不会把“无诊断阈值”误写成测量失败。
