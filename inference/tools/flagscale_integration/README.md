# FlagScale 检索训练集成工件

本目录是"在 FlagScale `feat/retrieval-flagos` 的 `flagscale/train/train_retrieval.py`
里接入 FlagPerf 观察采集"的集成工件。补丁对应上游提交 `1de74e8a`。

- `train_retrieval_observe.patch` — 对 `train_retrieval.py` 的六处改动（可选
  import、相位小工具、`obs` 参数、五相标记、会话包裹、透传）。apply 方式：

  ```bash
  cd /path/to/FlagScale
  git switch --detach 1de74e8a3758bf7586b178f1658ef29a1a18a506
  git apply --check /path/to/train_retrieval_observe.patch
  git apply /path/to/train_retrieval_observe.patch
  cp /path/to/FlagPerf/inference/tools/flagscale_adapter.py flagscale/train/
  ```

- `fs_demo_config.yaml` — 可直接使用的 standalone 训练配置（bge-m3、ddp、
  bf16、tiny 数据、1 epoch），相对路径按容器内挂载写，照文档改路径即可。
- `fs_make_data.py` — 生成 120 行 triplet parquet 的脚本（anchor/positive/
  negative_1），用于没有 T2Ranking 时跑通链路。

复现（PPU docker，镜像 `1ba7ead15378`）：

```bash
docker run --rm --shm-size=16g \
  --device=/dev/alixpu_ctl --device=/dev/alixpu \
  --device=/dev/alixpu_ppu2 --device=/dev/alixpu_ppu3 \
  -v <FlagScale>:/workspace/FlagScale -w /workspace/FlagScale \
  -v <FlagPerf>:/workspace/FlagPerf \
  -v <数据+配置+store 目录>:/data \
  -e PYTHONPATH=/workspace/FlagScale:/workspace/FlagPerf/inference:/workspace/FlagPerf/inference/tools \
  -e FLAGPROF_SCRATCH=/data/scratch \
  -e FLAGPROF_TRAIN_OBSERVE=/data/fs-obs-store \
  1ba7ead15378 bash -c "torchrun --standalone --nproc-per-node=2 \
      flagscale/train/train_retrieval.py --config-file /data/fs_demo_config.yaml"
```

实跑结果（2026-11-18，PPU 双卡）：

- 训练：`train_loss 0.116918 / validation_loss 0.000107`，两 rank 正常收尾。
- 采集：store 1.6 GB，`manifest.status=complete`；
  `comm={'model/allreduce': 2}`（评测的 loss 归约；DDP 梯度 Reducer 按设计
  不可见）、`phases={data:28, forward:27, backward:27, optimizer:27, step:27}`。
- 零漂移：开/关采集两跑的 `model.safetensors` sha256 逐字节相同
  （`4b346bb79adcd23e…`）。

更多说明见 `docs/training/launcher.md` §10 与交付材料中的《FlagScale 使用
FlagPerf 最佳实践》。
