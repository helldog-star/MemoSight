#!/usr/bin/env bash
# Sweep the MTP draft length and measure self-speculative-decoding acceptance
# rate across datasets, then aggregate with scripts/analyze_mtp_acceptance.py.
#
# Prereq: an MTP-trained checkpoint (config must carry an `mtp` block, e.g.
# configs/LightThinker/qwen/adaptive_mtp_v1.json) — the spec_decode path is
# only taken when comp_config.mtp_cfg is set.
#
# Usage:
#   bash scripts/run_mtp_acceptance.sh
#   DRAFT_LENS="1 2 3" DATASETS="gsm8k" GPU=0 bash scripts/run_mtp_acceptance.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# ---- edit these to match your model ----------------------------------------
MODEL_PATH="${MODEL_PATH:-/mnt/jinbo/RLRM/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/jinbo/RLRM/model/Qwen/Qwen2.5-0.5B-Instruct}"
COMPRESS_CONFIG="${COMPRESS_CONFIG:-./configs/LightThinker/qwen/adaptive_mtp_v1.json}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
BOS_TOKEN="${BOS_TOKEN:-<|im_start|>}"
EOS_TOKEN="${EOS_TOKEN:-<|im_end|>}"
CKPT="${CKPT:-0}"
MODEL_TAG="${MODEL_TAG:-mtp_model}"
# ----------------------------------------------------------------------------

DRAFT_LENS="${DRAFT_LENS:-1 2 3}"          # speculative register tokens per step
DATASETS="${DATASETS:-gsm8k}"              # space-separated: gsm8k mmlu bbh gpqa
GPU="${GPU:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10240}"
UPDATE_ATTN="${UPDATE_ATTN:-local}"
INDEX="${INDEX:-1}"
SPLIT_SIZE="${SPLIT_SIZE:-1}"
RESULT_ROOT="${RESULT_ROOT:-mtp_accept_results}"

ROOT_DIR="./LightThinker"
mkdir -p "$RESULT_ROOT"
GROUP_ARGS=()

run_one () {          # $1 = draft_len ("baseline" -> spec_decode off)
    local dl="$1" spec out_tag extra_args
    if [ "$dl" = "baseline" ]; then
        spec="false"; out_tag="${RESULT_ROOT}/baseline"; extra_args=()
    else
        spec="true";  out_tag="${RESULT_ROOT}/dl${dl}"; extra_args=(--mtp_draft_len "$dl")
    fi
    echo ">>> running ${out_tag} (spec_decode=${spec})"
    CUDA_VISIBLE_DEVICES="$GPU" python "${ROOT_DIR}/inference.py" \
        --model_tag "$MODEL_TAG" \
        --model_short_tag "mtp_dl_${dl}" \
        --ckpt "$CKPT" \
        --model_path "$MODEL_PATH" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --compress_config "$COMPRESS_CONFIG" \
        --model_type "$MODEL_TYPE" \
        --bos_token "$BOS_TOKEN" \
        --eos_token "$EOS_TOKEN" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --update_attention_method "$UPDATE_ATTN" \
        --output_tag "$out_tag" \
        --datasets $DATASETS \
        --split_size "$SPLIT_SIZE" \
        --index "$INDEX" \
        --spec_decode "$spec" \
        "${extra_args[@]}"
    # collect this run's jsonls into an analyzer group
    GROUP_ARGS+=(--group "${dl}=${out_tag}/**/*.jsonl")
}

# Optional wall-clock baseline (no speculation) for real speedup numbers.
if [ "${WITH_BASELINE:-0}" = "1" ]; then
    run_one baseline
fi

for dl in $DRAFT_LENS; do
    run_one "$dl"
done

echo ">>> aggregating acceptance stats"
python scripts/analyze_mtp_acceptance.py \
    "${GROUP_ARGS[@]}" \
    --csv "${RESULT_ROOT}/summary.csv" \
    --plot "${RESULT_ROOT}/acceptance.png" \
    --json "${RESULT_ROOT}/summary.json"

echo ">>> done. see ${RESULT_ROOT}/summary.csv and ${RESULT_ROOT}/acceptance.png"
