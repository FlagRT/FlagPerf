# Kunlunxin P800 INT8

The verified path is `torch._int_mm` with signed INT8 inputs, INT32 output,
scale 1 and zero point 0. CPU reference uses INT64 accumulation. Generic
`torch.mm(int8)` is rejected and no narrowing, saturation, or peak claim is
made.
