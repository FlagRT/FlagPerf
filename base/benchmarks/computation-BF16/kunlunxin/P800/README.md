# Kunlunxin P800 BF16

This candidate uses native XPYTORCH `torch.mm` with BF16 input and output. CPU
reference values are computed from the quantized BF16 inputs in FP64. The
recorded result is limited to the locked M1 runtime, one P800, and the frozen
shape/configuration; internal accumulation precision is not claimed.
