# MindLLM Table 2 复现记录

- 论文：Qiu et al., 2025, *MindLLM: A Subject-Agnostic and Versatile Model for fMRI-to-Text Decoding* (ICML 2025)
- 表：**Table 2 — Versatile decoding**
- 目标列：**MindLLM**（subject-agnostic，经 BIT 训练），**subject 1**
- 本地 PDF：`D:\BCI\Paper\Qiu 等 - 2025 - MindLLM A Subject-Agnostic and Versatile Model for fMRI-to-Text Decoding.pdf`（p.7 Table 2；p.16 Appendix B 超参）
- 论文里的下游微调配方：单卡，AdamW，**lr = 1e-4**，**batch size = 48**，贪婪解码；每个任务单独微调一次

---

## 0. 本次复现的运行信息（每次跑都填一遍）

| 项目 | 内容 |
|---|---|
| 日期 | |
| 任务 | |
| 完整命令 | |
| flags | |
| checkpoint | |
| 数据 / split | |
| 测试样本数 N | |
| 驱动脚本/代码版本 | |

---
  复现整张 Table 2

  论文是每个下游任务各微调一次、各测一次（Table 3 才是零样本）。task 名就是 DatasetSource.value（dataset.py:28-48）：coco-qa
  、visual-genome、vqa-v2、a-okvqa、st-vqa、ok-vqa、tally-qa、paragraph-caption、vqa-e、fsvqa、coco-caption-previous。照
  README 的模板换 data.task 即可：

  uv run python main.py data.task=<task> data.split_val=true early_stop=true \
    checkpoint=/mnt/data/yjs2026/MindLLM/MindLLM-main/ckpts/MindLLM/mindllm-base.ckpt lr=1e-4

  注意 test_dataloader 会跳过 llava/visdial/lvis（dataset.py:1724-1725），所以这三个任务测不了。Table 2 对应值（MindLLM
  列）：

## 1. 复现表

> 复现值一律填**仓库直接输出的数值**。注意：
> - `accuracy` 类指标仓库输出的是 **0–1 的小数**，论文是百分数（如仓库 `0.4819` ↔ 论文 `48.19%`）。
> - CIDEr 论文已 **×100**（Table 2 脚注）。
> - `accuracy` 若为空说明该任务没跑到 test（`stage=fit` 结尾会自动 `trainer.test(ckpt_path="best")`）。

| 任务 | 指标 | 日志键名 | 论文 MindLLM (subj 1) | 复现值 | 差值 | 备注 |
|---|---|---|---|---|---|---|
| COCO-QA | Accuracy | `test/coco-qa/accuracy` | 48.19% |52.38% | | |
| VG-QA | Accuracy | `test/visual-genome/accuracy` | 24.06% | | | |
| VQA-v2 | Accuracy | `test/vqa-v2/accuracy` | 52.14% | | | |
| A-OKVQA | Accuracy | `test/a-okvqa/accuracy` | 52.21% | | | |
| ST-VQA | ANLS | `test/st-vqa/ANLS` | 12.92% | | | |
| OK-VQA | Accuracy | `test/ok-vqa/accuracy` | 33.33% | | | |
| TallyQA | Accuracy | `test/tally-qa/accuracy` | 54.76% | | | |
| TallyQA | RMSE | `test/tally-qa/rmse` | 1.76 | | | |
| Paragraph Caption | BLEU-1 | `test/paragraph-caption/bleu@1` | 29.43 | | | |
| Paragraph Caption | BLEU-2 | `test/paragraph-caption/bleu@2` | 15.78 | | | |
| Paragraph Caption | BLEU-3 | `test/paragraph-caption/bleu@3` | 9.14 | | | |
| Paragraph Caption | BLEU-4 | `test/paragraph-caption/bleu@4` | 5.51 | | | |
| Paragraph Caption | METEOR | `test/paragraph-caption/meteor` | 13.18 | | | |
| Paragraph Caption | CIDEr | `test/paragraph-caption/cider` | 7.80 | | | |
| VQA-E | Accuracy | `test/vqa-e/accuracy` | 50.95% | | | |
| VQA-E | BLEU-1 | `test/vqa-e/bleu@1` | 37.70 | | | |
| VQA-E | BLEU-2 | `test/vqa-e/bleu@2` | 20.56 | | | |
| VQA-E | BLEU-3 | `test/vqa-e/bleu@3` | 12.34 | | | |
| VQA-E | BLEU-4 | `test/vqa-e/bleu@4` | 7.92 | | | |
| VQA-E | CIDEr | `test/vqa-e/cider` | 93.60 | | | |
| VQA-E | METEOR | `test/vqa-e/meteor` | 15.62 | | | |
| VQA-E | ROUGE | `test/vqa-e/rouge` | 35.88 | | | |
| FSVQA | VQA Acc. | `test/fsvqa/vqa-accuracy` | 48.03% | | | |
| FSVQA | FSVQA Acc. | `test/fsvqa/fsvqa-accuracy` | 43.00% | | | |
| FSVQA | BLEU-1 | `test/fsvqa/bleu@1` | 87.10 | | | |
| FSVQA | BLEU-2 | `test/fsvqa/bleu@2` | 83.03 | | | |
| FSVQA | BLEU-3 | `test/fsvqa/bleu@3` | 79.27 | | | |
| FSVQA | BLEU-4 | `test/fsvqa/bleu@4` | 75.50 | | | |
| FSVQA | METEOR | `test/fsvqa/meteor` | 49.05 | | | |
| FSVQA | CIDEr | `test/fsvqa/cider` | 666.26 | | | |
| Previous Caption | BLEU-1 | `test/coco-caption-previous/bleu@1` | 47.20 | | | |
| Previous Caption | BLEU-2 | `test/coco-caption-previous/bleu@2` | 25.16 | | | |
| Previous Caption | BLEU-3 | `test/coco-caption-previous/bleu@3` | 12.95 | | | |
| Previous Caption | BLEU-4 | `test/coco-caption-previous/bleu@4` | 7.49 | | | |
| Previous Caption | METEOR | `test/coco-caption-previous/meteor` | 11.96 | | | |
| Previous Caption | ROUGE | `test/coco-caption-previous/rouge` | 34.58 | | | |
| Previous Caption | CIDEr | `test/coco-caption-previous/cider` | 16.02 | | | |
| Previous Caption | SPICE | `test/coco-caption-previous/spice` | 3.93 | | | |

### Table 1 — Brain Captioning（COCO Caption）

> `data.task=coco-caption` 对的是 **Table 1**，不是 Table 2。论文 p.6 表注：CIDEr 已 ×100
> （"scaled by a factor of 100 for consistency with Table 1 and baselines"），所以这里 8 个指标全部 ×100。
> Table 1 的行是 subject-agnostic MindLLM，跟 Table 2 同列。

| 任务 | 指标 | 日志键名 | 论文 MindLLM (subj 1) | 复现值 | 差值 | 备注 |
|---|---|---|---|---|---|---|
| COCO Caption | BLEU-1 | `test/coco-caption/bleu@1` | 61.75 | 63.83 | +2.08 |  |
| COCO Caption | BLEU-2 | `test/coco-caption/bleu@2` | 42.84 | 44.82 | +1.98 |  |
| COCO Caption | BLEU-3 | `test/coco-caption/bleu@3` | 29.86 | 31.22 | +1.36 |  |
| COCO Caption | BLEU-4 | `test/coco-caption/bleu@4` | 21.24 | 21.97 | +0.73 |  |
| COCO Caption | METEOR | `test/coco-caption/meteor` | 19.54 | 20.44 | +0.90 |  |
| COCO Caption | ROUGE | `test/coco-caption/rouge` | 45.82 | 46.90 | +1.08 |  |
| COCO Caption | CIDEr | `test/coco-caption/cider` | 60.97 | 66.82 | +5.85 |  |
| COCO Caption | SPICE | `test/coco-caption/spice` | 11.79 | 12.65 | +0.86 |  |

---

## 2. 单任务命令模板

一律 **nohup 后台跑 + 重定向到日志**（SSH 断开不会把进程带走）：

```bash
# 微调 + 自动 test（stage=fit 结尾会跑 trainer.test(ckpt_path="best")）
nohup env WANDB_MODE=disabled uv run python main.py \
  data.task=<task> data.split_val=true early_stop=true data.num_workers=0 \
  checkpoint=/mnt/data/yjs2026/MindLLM/MindLLM-main/ckpts/MindLLM/mindllm-base.ckpt \
  lr=1e-4 > <task>_ft.log 2>&1 &

tail -f <task>_ft.log          # 看进度；Ctrl-C 只退出 tail，不影响训练

# 只重测（不重训）
nohup env WANDB_MODE=disabled uv run python main.py stage=test data.task=<task> \
  data.split_val=true data.num_workers=0 \
  checkpoint=outputs/<task>/checkpoints/epoch=NN.ckpt > <task>_test.log 2>&1 &
```

`WANDB_MODE=disabled` 是必须的：`src/models/mindllm.py:69-75` 拿它当开关，不设会强上
`flash_attention_2`。

`<task>` 取值见 `src/dataset.py:28-48`：`coco-qa`、`visual-genome`、`vqa-v2`、`a-okvqa`、`st-vqa`、`ok-vqa`、`tally-qa`、`paragraph-caption`、`vqa-e`、`fsvqa`、`coco-caption-previous`。

> `test_dataloader` 会跳过 `llava` / `visdial` / `lvis`（`src/dataset.py:1724-1725`），这三个任务测不了。

### 断点续训

中断后**原样重跑同一条命令**即可，会自动接着训（`resume: true` 默认开启）：

```bash
# 和上面 §2 的启动命令完全一样，只是再敲一遍
nohup env WANDB_MODE=disabled uv run python main.py \
  data.task=<task> data.split_val=true early_stop=true data.num_workers=0 \
  checkpoint=/mnt/data/yjs2026/MindLLM/MindLLM-main/ckpts/MindLLM/mindllm-base.ckpt \
  lr=1e-4 > <task>_ft.log 2>&1 &

grep '\[resume\]' <task>_ft.log   # 确认走的是哪条路
```

- checkpoint 落在 `outputs/<task>/checkpoints/`：`last.ckpt`（每个 epoch 覆盖，续训起点）+
  `epoch=NN.ckpt`（`val/token_loss` 最优，`trainer.test(ckpt_path="best")` 用它）。
- 日志里会有 `[resume] found ... resuming` / `[resume] no ... training from scratch`，
  以及续训时的 `[resume] restored RNG state from checkpoint`。后台跑看不到终端输出，用 `grep` 查。
- **想从头跑**：`resume=false`，或删掉 `outputs/<task>/checkpoints/last.ckpt`。
- **必须原样重跑的 flags**：`data.task` / `data.batch_size` / `data.split_val` / `data.split_seed` /
  `seed` / `lora` / `early_stop`。改了不会报错，但语义会悄悄偏掉。
- **续训时改 `lr` 无效**：ckpt 里的 `param_groups["lr"]` 会覆盖新值。
- 续训会恢复 optimizer / epoch / EarlyStopping 计数（所以 patience 不会清零）以及 RNG 状态，
  但**不等于一次不中断跑出来的跑**——`table.md` 记录时要在备注里写明"该值来自续训 run"。

---

## 3. 复现时的坑（对不上先查这里）

- **batch size**：论文 Appendix B 下游微调是 **batch 48**；仓库默认 `data.batch_size=8` × `trainer.accumulate_grad_batches=8` = 有效 64（那是 BIT 阶段的配方）。要严格对齐加上 `data.batch_size=48 trainer.accumulate_grad_batches=1`（显存不够就 `data.batch_size=6` 配累积 8）。
- **coco-qa 是精确匹配**：`ShortPhraseAccuracy` 只做 `lower().strip().rstrip(".")` 后逐字比对，不是 VQA 的 soft accuracy。跟 soft-acc 的数比会偏低。
- **CIDEr 量纲**：仓库 `COCOCIDEr` 返回原始值（约 0.6 量级），论文 **Table 1 和 Table 2 都已 ×100**
  （两张表的表注都写了 "The CIDEr metric is scaled by a factor of 100"）。其余 `accuracy`/BLEU/METEOR/ROUGE/SPICE
  也都是仓库给 0–1 小数、论文给百分数。
- **Java**：METEOR / SPICE 走 pycocoevalcap，需要 Java。纯 QA accuracy 不需要。
- **best ckpt**：`ModelCheckpoint` 监视 `val/token_loss`（`config/default.yaml:33-35`），所以要 `data.split_val=true` 才能选 best。
- **subject**：默认 `subjects: [1]`，即论文 Table 2 的 subject 1；不要改。
