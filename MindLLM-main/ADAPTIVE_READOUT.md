# MindLLM 两轮自适应注意力实验（2026-10-14）

## 目的与版本范围

以 MindLLM 为 baseline，检验「根据一次脑信号汇聚结果，更新 Query 后再次读取
voxel」是否改善下游任务。只增加可选 Encoder 和实验入口；原 Encoder、默认
`main.py`、聚类、对比学习、JEPA、监控脚本及历史实验保留。

- 分支：`codex/adaptive-reread-attention`。
- 起点：`cd931c9`（现有聚类 60 轮续训版本）。
- 没有加入 BIT-LLM 的 LLM cross-attention 或奖励训练。
- 本次交付是实现与兼容性验证，**不是下游性能提升结果**。

## 新增机制

原来 `Q0` 是 `[1, S, R]` 的可学习参数，K 来自坐标/ROI，V 是标量 fMRI。
默认 `S=1024, R=128`。第一次汇聚保留原实现及未缩放的 QK 点积：

```text
h1 = softmax(Q0 Kᵀ) V                         # [B,S]
c  = tanh(Linear(LayerNorm(h1)))              # [B,r]
ΔQ = (U ⊙ c[:,None,:]) W                      # [B,S,R]
h2 = softmax((Q0 + ΔQ) Kᵀ) V
h  = h1 + tanh(gate) × (h2 − h1)
```

U 是 `[S,r]`、W 是 `[r,R]`，默认 `r=16`。每个样本得到自己的低秩 Query 更新。
`h` 继续经过原 post norm、残差 MLP、token head，输出仍为 `[B,128,1024]`。
原 Projector 和 prefix 接口继续使用，LLM 冻结；训练 Encoder 和 Projector，
新增参数自动包含在原优化器的 `fmri_encoder.parameters()` 中。

### 初始化与数值处理

- 保留 `neuro_informed_attn.query_embeddings` 及所有原有参数名称与形状。
- context projection 的权重/偏置初始化为零；U/W 随机初始化。
- `tanh(gate)` 初始化为 0.1，因此初始 ΔQ=0、输出与 baseline 一致，
  context projection 从第一个训练步起就能获得梯度。不能同时将门控也置零。
- U/W 与门控在初始 ΔQ=0 时可以是零梯度；context projection 更新后可继续学习。
- 新模块构造使用 `fork_rng`，不改变后续 Projector 初始化和全局随机流。
- 第二轮按 Query 分块（默认 128），训练时使用非重入 activation checkpointing。
  分块计算 `(softmax(logits+Δlogits)-softmax(logits))V`，避免初始时因不同
  GEMM 尺寸造成非零残差。此做法有额外重算成本；不承诺训练更快或显存降低。

## 三组对照

| 配置 | Encoder | 新增可训练参数（默认尺寸） |
|---|---|---:|
| `readout_baseline` | 原始注意力 | 0 |
| `readout_mlp_control` | 原汇聚后增加低秩残差 MLP | 35,857 |
| `readout_adaptive` | 根据 h1 更新 Query 并再次读取 voxel | 36,881 |

MLP control 为 `h1 + tanh(gate) * up(GELU(down(LN(h1))))`，up 零初始化。
它是**近似参数匹配**（新增参数相差约 2.9%），不是算力匹配：二次读取需要访问
全部 voxel。三组数据、初始化 checkpoint、学习率、训练预算和选择规则必须一致。

默认单被试 S1、seed=42、COCO-caption，lr=1e-4、batch=4、梯度累积 8、
bf16、最多 20 epochs、patience=5，按 `val/token_loss` 选最佳 checkpoint。
训练后测试最佳 checkpoint。验证复用已有 `ClusterAblationDataModule`：从训练中
留出验证图像，不拿测试集早停，不需要任何 cluster bank。该修改是三个实验共同的
评估协议，与默认入口区分；官方 base checkpoint 是否见过这些验证图像仍取决于其预训练来源。

## 运行

在项目根目录，使用已有 MindLLM 环境。三组**顺序执行**，不要抢占当前训练 GPU。
模型和数据路径沿用原项目。新增入口内部调用原 `main.py`。

```bash
export WANDB_MODE=disabled
python train_readout.py --config-name readout_baseline
python train_readout.py --config-name readout_mlp_control
python train_readout.py --config-name readout_adaptive
```

默认初始化为 `ckpts/MindLLM/mindllm-base.ckpt`。可用
`checkpoint=/absolute/path/to/mindllm-base.ckpt` 覆盖。默认输出各自独立：

```text
outputs/readout_baseline_s1_seed42/
outputs/readout_mlp_control_s1_seed42/
outputs/readout_adaptive_s1_seed42/
```

入口拒绝覆盖非空输出目录。每组记录 `readout_manifest.json`（配置、源码 SHA256、
初始 checkpoint SHA256）、`config.yaml`、`split.json`、CSV 指标和 checkpoint。
新 seed/新任务/新被试应显式指定新目录，特别是默认目录名称中的 s1 不会自动随被试列表改变。

QA 示例（三组都要使用相同覆盖参数）：

```bash
python train_readout.py --config-name readout_adaptive \
  data.task=coco-qa output_dir=outputs/readout_adaptive_cocoqa_s1_seed42
```

先检查小批次可运行时，可使用独立输出目录：

```bash
python train_readout.py --config-name readout_adaptive \
  output_dir=outputs/readout_adaptive_smoke_s1_seed42 \
  trainer.max_epochs=1 +trainer.limit_train_batches=8 \
  +trainer.limit_val_batches=2 test_after_fit=false
```

恢复同一个实验时使用原命令及原初始化 checkpoint，增加 `resume=true`；
入口要求 manifest、初始 checkpoint 哈希、配置及源码一致，并从该目录的
`last.ckpt` 恢复优化器和 RNG。新增 adapter 后不能直接恢复 baseline 的 optimizer。
本入口不支持偷偷更改 epoch 预算等配置后续训；需要单独设计并记录续训协议。

推理必须选择与训练时相同的 encoder 配置，例如：

```bash
python train_readout.py --config-name readout_adaptive stage=test \
  'checkpoint="outputs/readout_adaptive_s1_seed42/coco-caption/checkpoints/epoch=XX.ckpt"' \
  output_dir=outputs/readout_adaptive_test_s1_seed42
```

将 XX 换为验证选出的 epoch。不要把 base checkpoint 当作已经训练好的 adaptive
模型进行测试；入口会拒绝这种用法。variant marker 也会阻止 adaptive/MLP 权重混用。
直接使用旧入口的 `strict=False` 不能提供这些实验级检查，因此优先使用 `train_readout.py`。

## 检查与结果解释

```bash
python -m unittest discover -s tests -p 'test_adaptive_readout.py' -v
python -m unittest discover -s tests -p 'test_clustered_attention.py' -v
```

新测试覆盖：初始化输出与原分支梯度一致；新分支可学习；样本相关 Query；
分块与完整矩阵的输出/梯度一致；CPU bf16；完整 Encoder checkpoint 转移和回读；
随机流不变；三组配置；误加载与旧输出目录保护。

2026-10-14 本机验证：新增 8 项测试、原聚类注意力 9 项测试全部通过
（PyTorch 2.2.2 CPU）。这些测试不等同于真实数据上的完整训练或性能评估。

下游性能需要实际训练才能判断。先对比 adaptive 与 baseline，再对比 adaptive 与
MLP control；多个 seed 检验稳定性。Caption 关注 CIDEr/SPICE 等，QA 关注任务准确率。
额外训练本身不能被解释为结构收益，不能把仅初始化的 baseline 当成公平对照。

## GitHub 和服务器版本管理

Git 提交只包含本次新文件及 README 入口；不提交权重、数据、输出、虚拟环境或本地论文。
服务器采用独立 release 目录，目录名使用完整提交的短哈希：

```text
/mnt/data2/yjs2026/HLQ/MindLLM/releases/readout-<commit>/MindLLM-main/
```

release 从 Git 提交导出，共用原项目 `data`、`ckpts`、`.venv-repaired` 的符号链接，
输出放在 release 自己的 `outputs`；旧项目目录和运行中的聚类实验不被覆盖。
release 的上层保存源文件 SHA256 清单、提交号与检查记录，可按提交追溯。
在 release 项目目录运行已有的 `run_mindllm_server.sh python train_readout.py ...`，
它沿用服务器已验证的 CPU 16–31 与环境设置。部署并不自动启动训练。
