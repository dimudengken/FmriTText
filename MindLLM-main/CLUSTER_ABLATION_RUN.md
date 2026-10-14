# 第一阶段：单独检验聚类约束

这是一个单被试、单 seed 的探索性实验。比较原编码器、功能分簇、随机分簇；
三组共用官方 `ckpts/MindLLM/mindllm-base.ckpt` 初始化，LLM 冻结，辅助损失关闭。
聚类版本仅改变每组 attention query 可访问的体素集合。

## 冻结的实验设置

- 被试 1：原 HDF5 训练集 9000 图像；其中 900 留作验证，8100 用于微调与聚类。
- HDF5 测试集 1000 图像，均与微调训练/验证划分不重叠。
- `seed=42`；128 簇；每簇 8 个 attention query；ridge alpha=100。
- 三组同一个 base checkpoint；lr=1e-4；batch=4，累积 8；bf16；LM checkpointing off。
- 每组最多 20 epoch；验证损失连续 5 次不改善则早停；按最小 `val/token_loss` 选 checkpoint。
- 整个批次最多 24 小时；单张 RTX 4090，CPU 16-31，依次运行。
- 日志用 CSV，移除依赖 Wandb 的 WatchModel callback；不修改默认训练入口。
- 完整训练前，各模型用真实数据跑 8 个训练 batch 与 2 个验证 batch，检查训练可运行。

原数据模块的训练验证路径使用 `file['test']`。
本实验用独立 `ClusterAblationDataModule` 将验证改为 `file['val']`，
原代码与默认配置保留。三个条件都使用这一数据模块，因此测试不参与早停或选择。
官方 base 模型可能在预训练时见过后来留作验证的图像；这里的验证是相对于本次微调与
聚类拟合的留出集，不能据此声称它在所有模型训练阶段都未被使用。

## 判断标准

主指标是最终测试集真实 fMRI 下的 CIDEr（另记录 BLEU-4、ROUGE-L）。
对齐相同图像与参考描述，记录 functional-baseline、functional-random 差值及
2000 次 paired image bootstrap 的 95% 区间；CIDEr 的 IDF 固定在同一个参考集。
另外对每个模型生成零 fMRI 与同被试内打乱 fMRI 的描述，检查提升是否依赖图像匹配的脑信号。

如果功能分簇没有超过 baseline，或与随机分簇没有区别，则本次实验不支持功能聚类收益。
如果只有真实分数提升、打乱脑信号后表现同样提升，则不能把提升归因于更好的脑信息提取。
bootstrap 只衡量给定 checkpoint 的图像采样不确定性，不覆盖训练 seed 的波动。
单次实验结果将按探索性结果报告；不能自动扩张为普遍优于原编码器的结论。

## 执行与产物

```bash
export JAVA_HOME=/mnt/data2/yjs2026/jdk/jdk-11.0.32.1+1
export PATH="$JAVA_HOME/bin:$PATH"
env WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 \
  taskset -c 16-31 .venv-repaired/bin/python -u scripts/run_cluster_ablation.py \
  --output outputs/cluster_ablation_s1_seed42_20261013_r2
```

实验目录保存 `protocol.json`、`status.json`、两个簇 bank、每个阶段的日志。
各模型目录保存完整配置、split 图像列表与哈希、CSV 训练/验证指标、最佳 checkpoint 路径、
各条件的逐图预测和逐图 CIDEr。最终生成 `summary.json` 与 `metrics.csv`。
在测试前核对三组 split 哈希一致；对比指标前核对测试图像与参考描述顺序一致。
任一阶段失败则记录失败并停止后续阶段，不将失败的 run 当作有效结果。

本协议只允许创建本次实验的文件与输出；原模型代码、其他实验、原始数据和既有 checkpoint 均保留。

## 启动记录

2026-10-13：第一批完成簇构建后，模型初始化因后台 PATH 缺少 Java 而失败，
尚未进行正式训练或 checkpoint 选择。失败日志保留在原实验目录。
补齐已有 Temurin 11 后，原编码器的 8-batch CUDA 探测通过，峰值显存约 18.4 GiB。
正式批次使用 `_r2` 目录与相同实验设置，记录 Java 环境并继续三组对照。

## 监控与有限恢复

2026-10-13 晚：按用户要求增加无人值守监控。服务器 guardian 每 60 秒观察进程和进度；
本机隐藏 PowerShell 监控每 120 秒检查 SSH、错误、指标与阶段变化，并写入服务器 `progress.log`。
本机告警采用 Windows 桌面通知，平稳状态不反复通知。
服务器守护独立运行；本机监控重新连上服务器时会检查守护是否需要重建。

可恢复的进程故障最多重试两次／阶段。训练优先从该组 `last.ckpt` 恢复，
保留 optimizer、早停状态和 RNG callback；尚无 checkpoint 时从相同 base 重新开始。
重试用独立 CSV 目录，原始日志与部分结果保留。持续 30 分钟没有实验进度的所属 worker
可由守护停止并恢复；模型、损失、划分、batch size、学习率和原始 24 小时预算保持固定。
配置错误、重试上限或预算耗尽会停止并记录，不会自动更改研究设置。
恢复前检查 GPU 是否空闲，避免与其他训练进程冲突。

服务器保存 `guardian.json`、`recovery_policy.json`、`guard_events.jsonl`；
本机保存 `outputs/cluster_ablation_monitor/monitor.json`、`snapshot.json`、`events.jsonl`。
最终原批次输出仍为 `summary.json`、`metrics.csv` 和逐图分数，守护另写可读的 `RESULTS.md`。
守护模拟测试覆盖阶段接续、恢复次数上限、配置错误停机和最终报告生成；
没有为测试故意中断实际训练。不能将监控与有限恢复等同于修复服务器硬件故障。

## 60 轮续训

2026-10-14：20 轮批次完成后，按用户授权将三组续训上限统一提高至总计 60 轮，
保持验证早停 patience=5、学习率、数据划分、簇 bank 和其他研究设置。
新目录为 `outputs/cluster_ablation_s1_seed42_20261014_e60`，原 `_r2` 目录保留。
用 `scripts/run_cluster_ablation_continue.py --mode prepare` 复制各组最后检查点及此前最佳检查点，
只迁移 ModelCheckpoint 回调中的目录与路径，不改变参数、优化器、随机数、早停状态、epoch 或 step。
三组从各自第 20 轮末尾继续，已有最佳验证模型仍参与最终选择。
新批次顺序为 baseline、功能聚类、随机聚类、三组测试评估、统一比较；继续使用服务器守护。
`protocol.json` 保存来源检查点 SHA256，最终 `RESULTS.md` 同时列出续训与 20 轮指标差异。
续训批次独立设置 24 小时运行预算；可能先因验证早停停止，不要求必须训练满 60 轮。
这次延长是在查看 20 轮结果后决定的后续探索，不是独立重复实验。
