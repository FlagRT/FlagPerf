# Kunlunxin P800 BF16

This candidate uses native XPYTORCH `torch.mm` with BF16 input and output. CPU
reference values are computed from the quantized BF16 inputs in FP64. The
recorded result is limited to the locked M1 runtime, one P800, and the frozen
shape/configuration; internal accumulation precision is not claimed.

## Throughput limitation on the locked stack

The BF16 GEMM is a genuine device execution but is not bf16-accelerated: at 8192-cubed it
measures 118.97 TFLOPS, which equals the fp32 rate (117.05 TFLOPS) while FP16 reaches 258.9
TFLOPS on the same shapes. The vendor's own dense GEMM (`xtorch_ops._gemm.matmul`) returns
the same bf16 number as ATen, so this is a property of the vendor kernel rather than an ATen
dispatch choice. See the day-five review for the measurements.

## Throughput limitation on the locked stack

The BF16 GEMM is a genuine device execution but is not bf16-accelerated: at 8192-cubed it
measures 118.97 TFLOPS, which equals the fp32 rate (117.05 TFLOPS) while FP16 reaches 258.9
TFLOPS on the same shapes. The vendor's own dense GEMM (`xtorch_ops._gemm.matmul`) returns
the same bf16 number as ATen, so this is a property of the vendor kernel rather than an ATen
dispatch choice. See the day-five review for the measurements.
