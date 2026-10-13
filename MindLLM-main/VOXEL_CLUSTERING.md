# 功能体素聚类：保留 MindLLM 原结构的第一阶段实验

分支：`codex/functional-voxel-clustering`。
`main` 的 `6f7a415` 提交保存改造前源码。原编码器、`main.py`、默认配置和已有损失函数没有改动。

## 改了什么

```text
原结构：坐标 + ROI → 每个 query 读取全部体素 → 1024 维聚合
                                  → 残差 MLP → 128 × 1024 token → projector → LLM

新结构：坐标 + ROI → 每组 query 只读取一个功能簇 → 1024 维聚合
                                  → 残差 MLP → 128 × 1024 token → projector → LLM
```

默认将体素分为 128 个功能簇，1024 个 attention query 连续分为 128 组，
每组 8 个 query。簇内仍使用原模型的 query/key 点积和 softmax，对体素活动值加权池化。
仅改变 query 可读取的体素集合，没有新增可训练参数，也没有改变其名称或形状。
坐标编码、五类 ROI embedding、残差 MLP、token 输出 head 和 LLM 接口均复用原实现。

这里的簇约束发生在最前面的体素聚合层。后续全连接 MLP/head 会混合簇信息，
**最终的第 k 个 LLM token 不等于第 k 个体素簇**。
这不是完整复现 Brain-IT-VQA；问题条件化 cross-attention、簇间 Transformer、
簇与最终 token 一一对应的结构属于后续实验，第一阶段先隔离聚类约束的效果。

## 聚类特征与数据边界

对每个被试，按 COCO 图像平均训练集的 fMRI 重复测量。
以冻结的 CLIP 图像特征为输入，以每个体素的训练响应为目标，拟合 ridge 编码模型：

```text
X: [训练图像数, CLIP维度]
Y: [训练图像数, 体素数]，仅在训练图像上逐体素标准化
W: [体素数, CLIP维度]，ridge 编码权重，逐体素 L2 归一化
```

将所有指定被试的 W 合并，拟合一个 MiniBatchKMeans。
各被试用相同中心编号，避免独立聚类后簇编号任意置换的问题。
这种共同特征空间提供对齐约束，不保证各被试的同编号簇具有完全相同的生物学功能。

聚类脚本按原数据模块的 COCO 划分顺序和 seed 排除验证图像；同时排除所有指定被试
HDF5 的 `val`、`test` 图像。多被试时先取所有被试的排除集合之并集，再拟合权重和中心。
读取验证/测试的 COCO ID 只用于排除，**不读取其体素响应，也不使用其 CLIP 特征拟合**。
CLIP bank 可以包含测试图像，但拟合只索引训练图像的特征。

训练入口在加载 tokenizer/LLM 之前审计 cluster bank 的 `fit_coco_ids` 与当前训练划分。
即使是 bank 中未参加当前训练的被试，其拟合图像也会检查是否与当前 held-out 图像重叠，
因为这些被试影响了共同中心。不同划分下产生泄漏的 bank 会拒绝加载。
下游训练仍沿用原 MindLLM 的逐被试数据划分；这里的审计限定于聚类拟合，
不能替代整个多被试实验的训练/验证刺激重叠检查。

cluster bank 保存簇编号、训练图像 ID、排除图像 ID、中心、seed、ridge 参数和 mask 顺序。
编码器核对 `nsdgeneral` mask 的形状及 C-order 非零索引，并将簇映射保存到 checkpoint。
空簇的聚合结果置零；不产生全 `-inf` softmax 的 NaN。
体素向量仍须遵循原项目的 mask 顺序；只有裸向量时，无法额外推断其内部是否被外部重排。

## 单被试运行

从项目根目录运行，依赖已在 `requirements.txt` 中。
本地源码目录没有 NSD 数据和 checkpoint，完整实验需在已配置的数据服务器执行。

如果已有对应的 CLIP 图像 bank，可直接复用。否则先生成：

```bash
python scripts/precompute_clip_features.py --modality image --subjects 1 --out data/clip_features/coco_clip_vitl14.pt
```

构建与 `data.split_val=true`、`seed=42` 一致的 bank：

```bash
python scripts/build_voxel_clusters.py --subjects 1 --feature-bank data/clip_features/coco_clip_vitl14.pt --n-clusters 128 --val-coco 900 --split-seed 42 --seed 42 --output data/voxel_clusters/functional_s1.pt
```

`config/clustered.yaml` 默认是 COCO caption、被试 1、验证划分开启。
使用新入口训练，保留原 checkpoint 的初始化能力：

```bash
python train_clustered.py checkpoint=/path/to/mindllm-base.ckpt lr=1e-4 early_stop=true
```

训练入口会沿用原来的 `main.py` 和 MindLLM；不需要更改原训练代码。
原 checkpoint 的所有 encoder 参数都可迁移，新增簇映射 buffer 由 cluster bank 提供。
加载原 checkpoint 时缺少这些新 buffer 的提示是预期情况。
checkpoint 的已有参数、训练过程和性能仍需用真实数据验证，不能将可加载等同于已适配。
恢复聚类 checkpoint 时，配置的 cluster bank 必须与 checkpoint 的体素簇映射一致；
checkpoint 的视觉 mask 也必须与构建 bank 时的 mask 一致。

输出默认写入 `outputs/clustered_attention_s1`，`resume=false`，避免读取原实验的 `last.ckpt`。
同一目录已经有聚类 checkpoint 时，应明确选择 resume 或换目录。
评测也使用新入口：

```bash
python train_clustered.py stage=test checkpoint=outputs/clustered_attention_s1/coco-caption/checkpoints/last.ckpt
```

服务器按 [SERVER_ENVIRONMENT.md](SERVER_ENVIRONMENT.md) 的 CPU affinity 要求，
将上述 `python ...` 放在 `./run_mindllm_server.sh` 后执行。

## 多被试与随机分簇对照

```bash
python scripts/build_voxel_clusters.py --subjects 1 2 3 4 5 6 7 --feature-bank data/clip_features/coco_clip_vitl14.pt --val-coco 900 --output data/voxel_clusters/functional_s1-7.pt
python train_clustered.py 'data.subjects=[1,2,3,4,5,6,7]' encoder.cluster_path=data/voxel_clusters/functional_s1-7.pt output_dir=outputs/clustered_attention_s1-7 checkpoint=/path/to/mindllm-base.ckpt
```

CLIP bank 必须覆盖指定被试所有用于拟合的训练图像；缺失图像会报错，不会悄悄跳过。
所有训练被试必须在 cluster bank 中。尚未拟合映射的新被试会被拒绝，
本版本不声称支持无需新被试数据的 zero-shot 跨被试聚类。

随机对照先做同一个功能拟合，再在各被试内部打乱体素的簇编号，保持每簇体素数量：

```bash
python scripts/build_voxel_clusters.py --subjects 1 --feature-bank data/clip_features/coco_clip_vitl14.pt --val-coco 900 --random-clusters --output data/voxel_clusters/random_s1.pt
python train_clustered.py encoder.cluster_path=data/voxel_clusters/random_s1.pt output_dir=outputs/random_cluster_attention_s1 checkpoint=/path/to/mindllm-base.ckpt lr=1e-4 early_stop=true
```

对照原编码器时，原命令仍然可用：

```bash
python main.py data.task=coco-caption data.split_val=true checkpoint=/path/to/mindllm-base.ckpt lr=1e-4 early_stop=true output_dir=outputs/original_attention_s1 resume=false
```

保持训练/验证划分、初始化 checkpoint、seed、训练超参数、辅助损失和评测集一致。
比较原始、随机分簇、功能分簇三组，并复用 `scripts/evaluate_neural_dependence.py`
检查预测对正确 fMRI、打乱 fMRI 和问题先验的依赖。

## 验证

新增测试覆盖簇外隔离、簇内均值、空簇、梯度、单簇退化与原 encoder 的一致性、
checkpoint 参数迁移和恢复、mask 顺序检查、训练数据边界、ridge 特征和完整 toy bank 构建。

```bash
python -m unittest discover -s tests -p 'test_clustered_attention.py' -v
python -m unittest discover -s tests -p 'test_voxel_clustering.py' -v
```

两个测试文件均使用合成数据，不下载模型、不启动完整 LLM 训练。
默认 `hidden_dim=1024` 必须能整除 `n_clusters`；改变簇数不改变最终 token 数。

本地验证环境是独立的 `.venv-clustering`：Python 3.12、CPU PyTorch 2.2.2、
torchvision 0.17.2、scikit-learn 1.5.2、h5py 3.11.0、Hydra 1.3.2、nibabel 4.0.2。
由于本地 Python 版本，测试环境使用 NumPy 1.26.4；项目的 Python 3.10 / NumPy 1.24.4
依赖声明保持原样。验证涵盖上述 16 个合成数据测试及两个 CLI 的参数/配置加载；
尚未执行真实 NSD 聚类、CUDA 验证或完整 LLM 训练。
