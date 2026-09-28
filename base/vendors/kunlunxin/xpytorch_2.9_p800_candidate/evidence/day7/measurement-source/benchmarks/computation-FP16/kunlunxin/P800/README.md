# Kunlunxin P800 FP16

This candidate uses native XPYTORCH `torch.mm` with FP16 input and output. CPU
reference values are computed from the quantized FP16 inputs in FP64. The
recorded result is limited to the locked M1 runtime, one P800, and the frozen
shape/configuration; internal accumulation precision is not claimed.
