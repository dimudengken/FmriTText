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
