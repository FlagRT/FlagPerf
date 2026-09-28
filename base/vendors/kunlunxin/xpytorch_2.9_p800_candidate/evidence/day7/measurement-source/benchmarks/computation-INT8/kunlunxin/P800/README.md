# Kunlunxin P800 INT8

The verified path is the vendor device kernel `xtorch_ops.gemm_I8_I8_bf16_nt`: a
block-quantized signed-INT8 GEMM with INT32 accumulation and bfloat16 output.

`torch.mm` rejects int8 on this stack, and `torch._int_mm` — while it returns int32
tensors on the device — executes on the host CPU. During the first qualification the
process CPU time of the "device" loop equalled the container thread count times wall
time, device telemetry stayed at idle (0% utilization, 39 C, 92 W) and the per-iteration
cost matched a PCIe round trip plus the CPU product. Those numbers were retracted; see
the [Day 5 review](../../../../vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day5/review.md).

## Contract

- Shape 8192x8192x8192 with seed 519, warmup 10 and 8000 iterations, matching the
  Ascend case scale. The iteration count is set by the 15 s measurement floor.
- `SCALE_A` and `SCALE_B` are dequantization scales in the symmetric int8 convention
  (`value = q * max / 127`). The kernel divides by 127 internally, so the driver passes
  `127 * SCALE_A` and `127 * SCALE_B` and the reported result is
  `(A @ B) * SCALE_A * SCALE_B` cast to `OUTPUT_DTYPE`.
- Correctness: five small shapes use full-output CPU float64 references over the
  quantized operands; the measurement shape checks fixed rows/columns over the complete
  reduction dimension before and after the timed window. Tolerances (atol 0.02,
  rtol 8e-3) are the bfloat16 output budget, not a relaxation of an exact-integer claim.
- Metric: TOPS, counting one multiply-add as two operations.

## Result

Five qualification runs on physical card 5 give a median of **504.591 TOPS** with
CV **0.3315%** and 17.37-17.52 s measurement windows. Device telemetry during the runs
shows 100% utilization, 61 C and 400 W, consistent with the FP16 path at the same scale.
