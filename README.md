# MemoSight

**English** | [中文](README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2604.14889-b31b1b.svg)](https://arxiv.org/abs/2604.14889)

Official PyTorch implementation of **[MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration](https://arxiv.org/abs/2604.14889)** ([arXiv:2604.14889](https://arxiv.org/abs/2604.14889)).

**MemoSight** (Memory-Foresight-Based Reasoning) unifies **context compression** and **multi-token prediction (MTP)** for chain-of-thought reasoning: it compresses historical tokens to reduce KV-cache growth and predicts future tokens in parallel to speed up decoding, while keeping reasoning accuracy close to vanilla supervised fine-tuning (SFT).

## Highlights

- Shared minimalist design with special tokens and token-specific positional layouts for both compression and parallel prediction.
- Up to **66%** lower KV-cache usage and **56%** faster inference vs. vanilla SFT, with under **3%** average accuracy drop on four reasoning benchmarks (see the [paper](https://arxiv.org/abs/2604.14889) for full results).

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [MTP Acceptance-Rate Analysis](#mtp-acceptance-rate-analysis)
- [Runtime Breakdown](#runtime-breakdown)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Requirements

- Python 3.9
- CUDA-capable GPU(s) for training and inference
- Python dependencies in [`requirements.txt`](requirements.txt) (PyTorch 2.5.1, Transformers 4.46.3, DeepSpeed 0.15.3, etc.)

## Installation

```bash
git clone https://github.com/helldog-star/MemoSight.git
cd MemoSight

conda create -n memosight python=3.9 -y
conda activate memosight
pip install -r requirements.txt
```

## Data Preparation

Place training data under `data/` (e.g. `data/train/train.jsonl`). If you use the bundled archive:

```bash
cd data && unzip data.zip && cd ..
```

## Quick Start

We recommend the unified entry script [`scripts/pipeline.sh`](scripts/pipeline.sh) for training, inference, and evaluation instead of chaining commands manually.

Show help:

```bash
bash scripts/pipeline.sh -h
```

### Pipeline stages

| `--stage` | Description |
|-----------|-------------|
| `train` | Training only |
| `infer` | Inference only (latest checkpoint by default) |
| `eval` | Evaluation only |
| `all` | Train → infer → eval |

### Common arguments

**General (required)**

| Argument | Description |
|----------|-------------|
| `--stage` | Pipeline stage |
| `--exp_tag` | Experiment name (also used as model tag) |
| `--output_base_dir` | Root directory for outputs |

**Training**

- `--use_epl`, `--lr`, `--mode`
- `--tokenizer_path`, `--model_path`, `--train_data_path`
- `--train_gpus`: comma-separated GPU ids, e.g. `0,1,2,3`

**Inference**

- `--target_gpus`, `--process_per_gpu`
- `--datasets`: comma-separated dataset names
- `--ckpt`: optional; uses the latest checkpoint if omitted

**Evaluation**

- `--eval_method` (default: `normal`)
- `--datasets`, `--comp_config`
- `--interaction`: `true` / `false`

### Examples

Replace `OUTPUT_DIR`, `TOKENIZER_PATH`, `MODEL_PATH`, and `TRAIN_DATA` with paths on your machine.

**Training only**

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

**Inference only**

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

**Full pipeline (train + infer + eval)**

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

### Output layout

Artifacts are written under:

```text
<output_base_dir>/<exp_tag>/
```

| Path | Contents |
|------|----------|
| `train/` | Training logs and checkpoints |
| `inference/` | Inference outputs and worker logs |
| `eval/` | Evaluation logs and metrics |
| `run_*.txt` | Snapshot of run arguments |
| `pipeline_*.sh` | Snapshot of the invoked pipeline script |

Symlinks for the latest run: `run_latest.txt`, `pipeline_latest.sh`, and stage-specific `*_latest.log`.

### FAQ

1. **`--stage infer` cannot find a checkpoint**  
   Run training first, or pass an explicit path via `--ckpt`.

2. **Out-of-memory (OOM)**  
   Lower `--micro_batch_size` first, then `--max_length` and `--process_per_gpu`.

3. **Invalid arguments**  
   Run `bash scripts/pipeline.sh -h` to verify names and values.

## MTP Acceptance-Rate Analysis

MemoSight decodes with **self-speculative decoding**: at each step the model drafts `γ + 1` tokens in one forward (the mandatory next token plus `γ` speculative *register* tokens), then a single *verify* forward confirms the longest matching prefix. The **acceptance rate** — how many drafted tokens survive verification — is what turns the extra register compute into real speedup.

### Running the sweep

The convenience runner [`scripts/run_mtp_acceptance.sh`](scripts/run_mtp_acceptance.sh) chains *sweep draft length `γ` → run inference per dataset → aggregate* into one command. Open the script, edit the 5 lines in the **CONFIG** block at the top, then run:

```bash
# top of scripts/run_mtp_acceptance.sh
CKPT_PATH="/path/to/your/train_output/checkpoint-xxxx"   # MTP-trained checkpoint dir
TOKENIZER_PATH="/path/to/Qwen2.5-0.5B-Instruct"          # tokenizer dir
COMPRESS_CONFIG="./configs/LightThinker/qwen/adaptive_mtp_v1.json"  # config WITH an `mtp` block
DRAFT_LENS="1 2"      # speculative register tokens/step; used as-is, keep <= training max_offset
DATASETS="gsm8k"      # space-separated: gsm8k mmlu bbh gpqa
```

```bash
bash scripts/run_mtp_acceptance.sh
```

> **Prerequisites.** Run under an environment that has PyTorch + Transformers (the `LightThinker` deps). The script `cd`s to the repo root and auto-exports `PYTHONPATH=<repo root>`, because `inference.py` imports the package absolutely (`from LightThinker.utils import *`) and the repo ships no `setup.py` — **activating a conda env alone does not put the repo on the import path.** If you invoke `inference.py` directly instead of through this script, export `PYTHONPATH` yourself or you get `ModuleNotFoundError: No module named 'LightThinker'`.

The script: **sweeps exactly the `DRAFT_LENS` you pass** (it does not parse the config's `max_offset` — keep the values `<=` your training `max_offset` yourself, or the extra positions are untrained and their acceptance is a fake signal), picks the chat tokens from `MODEL_TYPE` (`qwen`/`llama`), runs a baseline for a real speedup reference, then aggregates into a CSV / plot / JSON and prints the result paths.

Every field is also overridable from the command line without editing the file:

```bash
CKPT_PATH=/my/ckpt DRAFT_LENS="1 2" DATASETS="gsm8k mmlu" GPU=0 \
  bash scripts/run_mtp_acceptance.sh
```

| Field / env var | Default | Meaning |
|---------|---------|---------|
| `CKPT_PATH` | *(fill in)* | MTP-trained checkpoint dir |
| `TOKENIZER_PATH` | *(fill in)* | tokenizer dir |
| `COMPRESS_CONFIG` | `configs/.../adaptive_mtp_v1.json` | Must have an `mtp` block or the speculative path is skipped. |
| `DRAFT_LENS` | `1 2` | Speculative register tokens per step (`--mtp_draft_len`); used as-is — keep `<=` your training `max_offset`. |
| `DATASETS` | `gsm8k` | Space-separated: `gsm8k mmlu bbh gpqa` |
| `MODEL_TYPE` | `qwen` | `qwen` / `llama`; picks `BOS_TOKEN` / `EOS_TOKEN`. |
| `WITH_BASELINE` | `1` | Also run a non-speculative pass for a wall-clock speedup reference. |
| `MAX_NEW_TOKENS` | `10240` | Generation cap **and** buffer size — see the sizing note below. |
| `SPLIT_SIZE` / `INDEX` | `1` / `1` | Data sharding: `SPLIT_SIZE=N INDEX=1` runs only the first `len/N` samples — handy for a quick smoke test. |
| `GPU` | `0` | CUDA device id |
| `RESULT_ROOT` | `mtp_accept_results` | Output directory |

> **Requirement & constraint:** the draft length `γ` is set via `--mtp_draft_len`. Acceptance is only meaningful up to the `max_offset` the checkpoint was trained with (register offset was sampled in `[0, max_offset]`) — **the script uses your `DRAFT_LENS` as-is, so keep it within the training value yourself.** A single `dl2` run already contains the per-position acceptance for **pos1 / pos2 / pos3** (`draft_len = γ + 1 = 3`) — no need to run each position separately.

> **Don't set `MAX_NEW_TOKENS` too small.** The token/KV buffer is preallocated to `MAX_NEW_TOKENS + max_prompt_len` (1100). With prompt compression on, the prefill *expands* — each prompt sentence appends ~300 register (`<|o_*|>`) tokens, so a single GSM8k prompt prefills to ≈4k tokens. If `MAX_NEW_TOKENS + 1100` is smaller than `prefill + generation`, you get `IndexError: index … is out of bounds` in `set_input_ids`. Keep the default `10240` (buffer ≈11.3k), or size it to `expected_prefill + expected_output` with headroom.

> **Does `max_offset` in `COMPRESS_CONFIG` need to match `DRAFT_LENS`? — No.** As long as `--mtp_draft_len` is passed (this script always does), it fully determines the inference draft length; the config's `max_offset` is overwritten and unused (`inference.py:1650`). The config's `mtp` block only needs to *exist* to trigger the speculative path. The real hard constraint is `DRAFT_LENS ≤ the max_offset the checkpoint was trained with` (baked into the weights). Still, keep the config's `max_offset` equal to the training value as a label for the checkpoint and for `inference_batched.py`'s fallback path.

### Analyzing manually

To aggregate existing inference outputs directly (any run produced with `--spec_decode true`):

```bash
# single run
python scripts/analyze_mtp_acceptance.py mtp_accept_results/dl2/**/*.jsonl

# compare several draft lengths, emit table + csv + plot
python scripts/analyze_mtp_acceptance.py \
  --group dl1=mtp_accept_results/dl1/**/*.jsonl \
  --group dl2=mtp_accept_results/dl2/**/*.jsonl \
  --group dl3=mtp_accept_results/dl3/**/*.jsonl \
  --csv mtp_accept_results/summary.csv \
  --plot mtp_accept_results/acceptance.png \
  --json mtp_accept_results/summary.json
```

### Reported metrics

| Metric | Definition | Reads as |
|--------|------------|----------|
| **mean accept length τ** | committed tokens / decode steps | tokens emitted per step (upper bound on speedup) |
| **overall acceptance α** | accepted speculative tokens / proposed speculative tokens | fraction of MTP drafts kept, `∈ [0, 1]` |
| **per-position αₖ** | acceptance of the k-th register position (unconditional and conditional-on-reached) | how prediction quality decays with distance |
| **tokens / forward** | committed tokens / forward passes | compute-bound speedup proxy vs. `1.0` for plain AR |
| **tokens / s** | output length / inference time | measured wall-clock throughput |
| **accept histogram** | frequency of committing 1, 2, … tokens per step | shape of the acceptance distribution |

Accuracy is reported alongside so speed gains are never read in isolation.

### Artifacts

- Each inference record (`<output_tag>/<dataset>/<index>_<dataset>.jsonl`) gains an `mtp_stats` block plus `mtp_draft_len`, carrying raw counters for exact, loss-free re-aggregation across samples and files.
- `summary.csv` — one row per group: `draft_len, accuracy, mean_accept_len, overall_accept_rate, tokens_per_forward, tokens_per_sec, …`
- `summary.json` — full derived metrics (including per-position αₖ and the accept histogram).
- `acceptance.png` — two panels: per-position acceptance curves, and mean accept length / tokens-per-forward vs. draft length.

## Runtime Breakdown

To answer *where the speedup comes from*, split each decode step's wall-clock time into **prediction / verification / compression / other**. Collect it by running inference with `--profile_breakdown true` (timing is GPU-synchronized so the attribution is real).

```bash
CKPT_PATH=/path/to/your/ckpt \
TOKENIZER_PATH=/path/to/Qwen2.5-0.5B-Instruct \
COMPRESS_CONFIG=./configs/LightThinker/qwen/adaptive_mtp_v1.json \
DRAFT_LEN=2 DATASETS="gsm8k" GPU=0 \
bash scripts/run_runtime_breakdown.sh
```

Phase → code: **prediction** = main forward + draft sampling; **verification** = verify forward + sampling + KV trims; **compression** = the compression branch's cache/input-ids reduction; **other** = `t_total` minus the three (mask construction, register bookkeeping, Python overhead). Seconds are *pooled* across samples before dividing (long samples weigh more — exactly what a "where does time go" figure wants).

> **Note:** profiling wraps each phase in `torch.cuda.synchronize()`, so **this run's tokens/s is not a valid throughput number** — keep it separate from the end-to-end speed / acceptance-rate runs. The compression forward is *fused* into the main forward, so the compression bucket only counts its unique overhead (cache reduction); the compression forward cost lands inside prediction. This is an accounting convention — state it in the write-up.

You can also aggregate existing profiled outputs directly:

```bash
python scripts/analyze_runtime_breakdown.py \
  --group gsm8k=runtime_breakdown_results/dl2/gsm8k/**/*.jsonl \
  --group bbh=runtime_breakdown_results/dl2/bbh/**/*.jsonl \
  --csv breakdown.csv --plot breakdown.png --json breakdown.json
```

Outputs: `breakdown.csv` (per-dataset per-phase seconds & %), `breakdown.png` (stacked bar), `breakdown.json` (full metrics).

**Run it per dataset?** The breakdown ratios are driven mostly by *mechanism* (forward-pass count, verify/compression trigger frequency), not by content the way accuracy is, so one representative dataset (e.g. GSM8k) is enough to support the "where time goes" claim. But this profiling rides the *same* inference path as the acceptance-rate run — if you're already sweeping benchmarks for acceptance, the breakdown comes at near-zero marginal cost. Recommendation: use GSM8k as the main-text figure, and use this script's per-dataset grouping in the appendix to show the split is stable across benchmarks (the compression-step fraction shifts a bit with CoT length/structure, worth showing). No need to add datasets *just* for the breakdown.

## Project Structure

```text
MemoSight/
├── LightThinker/           # Core model, training, and inference code
├── configs/LightThinker/   # Model and training configs (JSON)
├── scripts/                # pipeline.sh, MTP acceptance sweep & analysis, runners
├── evaluation/             # Evaluation scripts
├── data/                   # Training and benchmark data
└── requirements.txt
```

For a traditional MTP baseline, use [`scripts/pipeline_traditional_MTP.sh`](scripts/pipeline_traditional_MTP.sh).

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2026memosight,
  title   = {MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration},
  author  = {Liu, Xinyu and Liu, Xin and Jin, Bo and Zhao, Runsong and Huang, Pengcheng and Ruan, Junhao and Li, Bei and Xiao, Chunyang and Wang, Chenglong and Xiao, Tong and Zhu, Jingbo},
  journal = {arXiv preprint arXiv:2604.14889},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.14889}
}
```

## Acknowledgments

This repository extends [LightThinker](https://github.com/ZJUNLP/LightThinker) and related open-source work. We thank the original authors and contributors.

## License

This project is licensed under the [MIT License](LICENSE).
