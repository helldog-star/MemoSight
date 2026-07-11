# MemoSight

[English](README.md) | **中文**

[arXiv](https://arxiv.org/abs/2604.14889)

论文 **[MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration](https://arxiv.org/abs/2604.14889)**（[arXiv:2604.14889](https://arxiv.org/abs/2604.14889)）的官方 PyTorch 实现。

**MemoSight**（Memory-Foresight-Based Reasoning）将**上下文压缩**与**多token预测**统一用于思维链推理：压缩历史 token 以抑制 KV cache 增长，并并行预测未来 token 以加速解码，同时使推理精度接近 vanilla SFT 基线。

## 亮点

- 压缩与并行预测共用基于特殊 token 与 token 专属位置编码的极简设计。
- 相较 vanilla SFT，KV cache 最多降低 **66%**、推理加速 **56%**，在四个推理基准上平均精度下降不足 **3%**（详见[论文](https://arxiv.org/abs/2604.14889)）。

## 目录

- [环境要求](#环境要求)
- [安装](#安装)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [MTP 接受率分析](#mtp-接受率分析)
- [项目结构](#项目结构)
- [引用](#引用)
- [致谢](#致谢)
- [许可协议](#许可协议)

## 环境要求

- Python 3.9
- 训练与推理需支持 CUDA 的 GPU
- Python 依赖见 `[requirements.txt](requirements.txt)`（PyTorch 2.5.1、Transformers 4.46.3、DeepSpeed 0.15.3 等）

## 安装

```bash
git clone https://github.com/helldog-star/MemoSight.git
cd MemoSight

conda create -n memosight python=3.9 -y
conda activate memosight
pip install -r requirements.txt
```

## 数据准备

将训练数据放在 `data/` 下（例如 `data/train/train.jsonl`）。若使用仓库附带压缩包：

```bash
cd data && unzip data.zip && cd ..
```

## 快速开始

推荐使用统一入口脚本 `[scripts/pipeline.sh](scripts/pipeline.sh)` 完成训练、推理与评估，避免手动拼接多段命令。

查看帮助：

```bash
bash scripts/pipeline.sh -h
```

### 流水线阶段


| `--stage` | 说明                     |
| --------- | ---------------------- |
| `train`   | 仅训练                    |
| `infer`   | 仅推理（默认使用最新 checkpoint） |
| `eval`    | 仅评估                    |
| `all`     | 训练 → 推理 → 评估           |


### 常用参数

**通用（必填）**


| 参数                  | 说明              |
| ------------------- | --------------- |
| `--stage`           | 执行阶段            |
| `--exp_tag`         | 实验名（同时作为模型 tag） |
| `--output_base_dir` | 输出根目录           |


**训练**

- `--use_epl`、`--lr`、`--mode`
- `--tokenizer_path`、`--model_path`、`--train_data_path`
- `--train_gpus`：逗号分隔 GPU 编号，如 `0,1,2,3`

**推理**

- `--target_gpus`、`--process_per_gpu`
- `--datasets`：逗号分隔数据集名
- `--ckpt`：可选；省略时使用最新 checkpoint

**评估**

- `--eval_method`（默认 `normal`）
- `--datasets`、`--comp_config`
- `--interaction`：`true` / `false`

### 示例

请将 `OUTPUT_DIR`、`TOKENIZER_PATH`、`MODEL_PATH`、`TRAIN_DATA` 替换为本机路径。

**仅训练**

```bash
bash scripts/pipeline.sh \
  --stage train \
  --exp_tag vanilla_qwen \
  --output_base_dir OUTPUT_DIR \
  --use_epl false \
  --lr 1e-5 \
  --mode normal \
  --model_type qwen \
  --tokenizer_path TOKENIZER_PATH \
  --model_path MODEL_PATH \
  --train_data_path TRAIN_DATA \
  --train_gpus 0,1,2,3
```

**仅推理**

```bash
bash scripts/pipeline.sh \
  --stage infer \
  --exp_tag vanilla_qwen \
  --output_base_dir OUTPUT_DIR \
  --use_epl false \
  --model_type qwen \
  --tokenizer_path TOKENIZER_PATH \
  --target_gpus 0,1,2,3 \
  --process_per_gpu 1 \
  --datasets mmlu,gsm8k,gpqa,bbh
```

**全流程（train + infer + eval）**

```bash
bash scripts/pipeline.sh \
  --stage all \
  --exp_tag vanilla_qwen \
  --output_base_dir OUTPUT_DIR \
  --use_epl false \
  --lr 1e-5 \
  --mode normal \
  --model_type qwen \
  --tokenizer_path TOKENIZER_PATH \
  --model_path MODEL_PATH \
  --train_data_path TRAIN_DATA \
  --train_gpus 0,1,2,3 \
  --target_gpus 0,1,2,3 \
  --process_per_gpu 1 \
  --datasets mmlu,gsm8k,gpqa,bbh
```

### 输出目录

运行产物位于：

```text
<output_base_dir>/<exp_tag>/
```


| 路径              | 内容               |
| --------------- | ---------------- |
| `train/`        | 训练日志与 checkpoint |
| `inference/`    | 推理输出与子进程日志       |
| `eval/`         | 评估日志与结果          |
| `run_*.txt`     | 本次运行参数快照         |
| `pipeline_*.sh` | 运行时脚本快照（便于复现）    |


软链接便于追踪最近一次运行：`run_latest.txt`、`pipeline_latest.sh`、各阶段 `*_latest.log`。

### 常见问题

1. `**--stage infer` 找不到 checkpoint**
  先完成训练，或通过 `--ckpt` 显式指定路径。
2. **显存不足（OOM）**
  优先减小 `--micro_batch_size`，其次降低 `--max_length` 与 `--process_per_gpu`。
3. **参数错误**
  执行 `bash scripts/pipeline.sh -h` 核对参数名与取值。

## MTP 接受率分析

MemoSight 采用**自推测解码（self-speculative decoding）**：每个解码步用一次前向草拟 `γ + 1` 个 token（1 个必然接受的下一 token，加 `γ` 个投机性的 *register* token），随后用一次 *verify* 前向确认最长匹配前缀。**接受率**——即草稿 token 通过验证的比例——决定了额外 register 计算能否真正转化为加速。

### 运行扫描

只要以 `--spec_decode true` 运行推理，就会逐样本收集接受率统计。便捷脚本 `[scripts/run_mtp_acceptance.sh](scripts/run_mtp_acceptance.sh)` 会在数据集上扫描草稿长度 `γ` 并自动聚合：

```bash
DRAFT_LENS="1 2 3" DATASETS="gsm8k" GPU=0 WITH_BASELINE=1 \
  bash scripts/run_mtp_acceptance.sh
```


| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `DRAFT_LENS` | `1 2 3` | 每步投机 register token 数（`--mtp_draft_len`）。扫到超过训练 `max_offset` 可观察接受率何时崩塌。 |
| `DATASETS` | `gsm8k` | 空格分隔：`gsm8k mmlu bbh gpqa` |
| `WITH_BASELINE` | `0` | 设为 `1` 额外跑一次非投机解码作为墙钟对照 |
| `GPU` | `0` | CUDA 设备编号 |
| `MODEL_PATH` / `TOKENIZER_PATH` / `COMPRESS_CONFIG` | 见脚本头部 | 必须使用**含 `mtp` 块的 MTP 训练配置**，如 `[configs/LightThinker/qwen/adaptive_mtp_v1.json](configs/LightThinker/qwen/adaptive_mtp_v1.json)`；否则不会走投机分支。 |


> **前提：** 草稿长度 `γ` 通过 `--mtp_draft_len` 配置。接受率仅在不超过 checkpoint 训练时的 `max_offset` 范围内有意义（训练时 register offset 在 `[0, max_offset]` 中采样）。

### 手动分析

直接聚合已有推理输出（任何以 `--spec_decode true` 产生的结果）：

```bash
# 单次运行
python scripts/analyze_mtp_acceptance.py mtp_accept_results/dl2/**/*.jsonl

# 对比多个草稿长度，输出表格 + csv + 图
python scripts/analyze_mtp_acceptance.py \
  --group dl1=mtp_accept_results/dl1/**/*.jsonl \
  --group dl2=mtp_accept_results/dl2/**/*.jsonl \
  --group dl3=mtp_accept_results/dl3/**/*.jsonl \
  --csv mtp_accept_results/summary.csv \
  --plot mtp_accept_results/acceptance.png \
  --json mtp_accept_results/summary.json
```

### 报告指标


| 指标 | 定义 | 解读 |
| --- | --- | --- |
| **平均接受长度 τ** | 已提交 token 数 / 解码步数 | 每步等价产出的 token 数（加速比上界） |
| **整体接受率 α** | 接受的投机 token / 提议的投机 token | MTP 草稿被采纳的比例，`∈ [0, 1]` |
| **逐位置接受率 αₖ** | 第 k 个 register 位置的接受率（无条件 + 条件于"被触及"两种口径） | 预测质量随距离的衰减 |
| **token / 前向** | 已提交 token / 前向次数 | 计算受限下的加速代理（普通 AR 为 `1.0`） |
| **token / 秒** | 输出长度 / 推理耗时 | 实测墙钟吞吐 |
| **接受直方图** | 每步提交 1、2、… 个 token 的频率 | 接受分布的形状 |

准确率一并报告，避免脱离精度孤立地看加速。

### 产物

- 每条推理记录（`<output_tag>/<dataset>/<index>_<dataset>.jsonl`）新增 `mtp_stats` 块及 `mtp_draft_len`，携带原始计数，可跨样本、跨文件**无损**再聚合。
- `summary.csv` —— 每组一行：`draft_len, accuracy, mean_accept_len, overall_accept_rate, tokens_per_forward, tokens_per_sec …`
- `summary.json` —— 完整派生指标（含逐位置 αₖ 与接受直方图）。
- `acceptance.png` —— 两幅子图：逐位置接受率曲线，以及平均接受长度 / token-per-forward 随草稿长度的变化。

## 项目结构

```text
MemoSight/
├── LightThinker/           # 模型、训练、推理核心代码
├── configs/LightThinker/   # 模型与训练配置（JSON）
├── scripts/                # pipeline.sh、MTP 接受率扫描与分析等运行脚本
├── evaluation/             # 评估脚本
├── data/                   # 训练/评测数据
└── requirements.txt
```

传统 MTP 对照实验可使用 `[scripts/pipeline_traditional_MTP.sh](scripts/pipeline_traditional_MTP.sh)`。

## 引用

若本工作对您的研究有帮助，请引用：

```bibtex
@article{liu2026memosight,
  title   = {MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration},
  author  = {Liu, Xinyu and Liu, Xin and Jin, Bo and Zhao, Runsong and Huang, Pengcheng and Ruan, Junhao and Li, Bei and Xiao, Chunyang and Wang, Chenglong and Xiao, Tong and Zhu, Jingbo},
  journal = {arXiv preprint arXiv:2604.14889},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.14889}
}
```

## 致谢

本仓库基于 [LightThinker](https://github.com/ZJUNLP/LightThinker) 等开源工作扩展实现，感谢所有相关作者与贡献者。

## 许可协议

本项目采用 [MIT 许可证](LICENSE)。