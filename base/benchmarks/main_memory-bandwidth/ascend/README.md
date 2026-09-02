# Ascend D2D/HBM copy bandwidth

This vendor configuration routes the existing FlagPerf
`main_memory-bandwidth` Case through the minimal Torch-FL adapter. The workload
remains the origin `tensor.clone()` loop; each iteration is reported as one
tensor read plus one tensor write (`2 * tensor_bytes`). Device selection uses
the torchrun local rank and synchronization uses `torch.flagos.synchronize()`.

The Gloo process group is control-plane only: it is used for rank barriers and
does not carry the measured device tensor. This Benchmark does not invoke DMI
and does not add an Oracle comparison.
