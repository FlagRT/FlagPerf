# computation-FP8 / thead PPU - evidenced skip

The bounded capability probe recorded on the controlled result root fixed FP8
as the only computation precision this locked stack cannot execute:

- `torch.mm` on float8_e4m3fn / float8_e5m2 raises `NotImplementedError`
  ("addmm_cuda" not implemented for 'Float8_e4m3fn').
- `torch._scaled_mm` refuses the device through its compute-capability gate
  ("torch._scaled_mm is only supported on CUDA devices with compute
  capability >= 9.0 or 8.9, or ROCm MI300+").

`runtime_requirements.json` records `supported: false` with the reason, so
`run.py --case computation-FP8:PPU` fails closed as an applicability skip
instead of executing a fallback. FP32/FP16/BF16/INT8/FP64/TF32 carry their own
chip assets and are applicable.
