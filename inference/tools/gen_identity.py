# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Generate a sourced host identity JSON from the live torch device stack.

The identity file consumed by flagprof vendors (FLAGPROF_THEAD_IDENTITY) must
carry the real device UUIDs reported by torch, joined to vendor smi metadata.
This tool reads torch.cuda directly so the UUIDs are sourced from the same
framework the observer cross-checks against - never hand-written.

Usage (inside the training container):
  python tools/gen_identity.py --smi xpu-smi --driver 5.0.21.47 \
      --host-indices 3 --node-prefix /dev/xpu --pci 00000000:03:00.0 --out identity.json
Pass the host index and PCI BDF reported by smi for each framework-visible
device, in framework order.
"""
import argparse
import json
from pathlib import Path


def build_devices(props_list, driver, node_prefix, pci_list, host_indices):
    if not (len(props_list) == len(pci_list) == len(host_indices)) or not all(pci_list):
        raise ValueError('one PCI BDF and host index are required per visible device')
    if any(index < 0 for index in host_indices) or len(set(host_indices)) != len(host_indices):
        raise ValueError('host indices must be distinct and nonnegative')
    devices = []
    for i, props in enumerate(props_list):
        uuid = str(getattr(props, 'uuid', '') or '').lower().removeprefix('gpu-')
        devices.append(dict(
            model=getattr(props, 'name', ''),
            uuid=uuid,
            pci_bdf=pci_list[i],
            driver=driver,
            logical_device=i,
            physical=str(host_indices[i]),
            host_device_node='%s%d' % (node_prefix, host_indices[i]),
        ))
    return devices


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smi', required=True, help='vendor smi tool name, e.g. xpu-smi / ppu-smi / npu-smi')
    p.add_argument('--driver', required=True)
    p.add_argument('--node-prefix', required=True, help='host device path before its numeric index')
    p.add_argument('--host-indices', required=True, help='comma-separated smi host indices in visible order')
    p.add_argument('--pci', required=True, help='comma-separated PCI bus IDs in device order')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    import torch
    count = torch.cuda.device_count()
    if count < 1:
        raise SystemExit('no torch devices visible')
    props_list = [torch.cuda.get_device_properties(i) for i in range(count)]
    pci_list = [x.strip() for x in args.pci.split(',')]
    host_indices = [int(x) for x in args.host_indices.split(',')]
    identity = dict(
        source=args.smi,
        driver=args.driver,
        devices=build_devices(props_list, args.driver, args.node_prefix, pci_list, host_indices),
    )
    Path(args.out).write_text(json.dumps(identity, indent=1) + '\n', encoding='utf-8')
    print('GEN-IDENTITY-OK devices=%d out=%s' % (count, args.out))


if __name__ == '__main__':
    main()
