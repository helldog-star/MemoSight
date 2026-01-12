#!/bin/bash


# eval "$(/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/miniconda/bin/conda shell.bash hook)"
# conda activate lightthinker
# echo "Using python: $(which python)"


# ROOT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/RRcot"
# cd $ROOT_DIR
# export PYTHONPATH=$PYTHONPATH:$(pwd)

# echo "[INFO] Working directory: $(pwd)"


echo "=======🚀 lightthinker开始训练 ======="
bash ./our_train_lighthinker.sh
echo "=======🚀 lightthinker结束训练 ======="

echo "=======🚀 lightthinker开始推理 ======="
bash ./our_inference_repe.sh
echo "=======🚀 lightthinker结束推理 ======="


echo "=======🚀 lightthinker_EPL开始训练 ======="
bash ./our_train_lighthinker_epl.sh
echo "=======🚀 lightthinker_EPL结束训练 ======="


echo "=======🚀 lightthinker_EPL开始推理 ======="
bash ./our_inference_epl_repe.sh
echo "=======🚀 lightthinker_EPL结束推理 ======="


echo "=======🚀 lightthinker_EPL_MTP开始训练 ======="
bash ./our_train_lighthinker_epl_mtp.sh
echo "=======🚀 lightthinker_EPL_MTP结束训练 ======="

echo "=======🚀 lightthinker_EPL_MTP开始推理 ======="
bash ./our_inference_epl_mtp_repe.sh
echo "=======🚀 lightthinker_EPL_MTP结束推理 ======="


echo "=======🚀 vanilla开始训练 ======="
bash ./our_train_baseline.sh
echo "=======🚀 vanilla结束训练 ======="

echo "=======🚀 vanilla开始推理 ======="
bash ./our_sglang_infer.sh
echo "=======🚀 vanilla结束推理 ======="




echo "=========================================="
echo "      🚀 lightthinker评估开始     "
echo "=========================================="
output_path="/tmp/hx/rrcot/lightthinker"
model_tag="lighthinker"
ckpt="1305"
output_tag="${output_path}/${model_tag}/inference"
bash our_evaluate.sh --method anchor-thought --dataset bbh --base_path ${output_tag}/bbh/${ckpt}
bash our_evaluate.sh --method anchor-thought --dataset gpqa --base_path ${output_tag}/gpqa/${ckpt}
bash our_evaluate.sh --method anchor-thought --dataset gsm8k --base_path ${output_tag}/gsm8k/${ckpt}
bash our_evaluate.sh --method anchor-thought --dataset mmlu --base_path ${output_tag}/mmlu/${ckpt}
echo "=========================================="
echo "      🚀 lightthinker评估完成      "
echo "=========================================="



echo "=========================================="
echo "      🚀 lightthinker_epl评估开始     "
echo "=========================================="
output_path="/tmp/hx/rrcot/lightthinker_epl"
model_tag="lighthinker_epl"
ckpt="1305"
output_tag="${output_path}/${model_tag}/inference"
bash our_evaluate.sh --method anchor-thought --dataset bbh --base_path ${output_tag}/bbh/${ckpt}
bash our_evaluate.sh --method anchor-thought --dataset gpqa --base_path ${output_tag}/gpqa/${ckpt}
bash our_evaluate.sh --method anchor-thought --dataset gsm8k --base_path ${output_tag}/gsm8k/${ckpt}
bash our_evaluate.sh --method anchor-thought --dataset mmlu --base_path ${output_tag}/mmlu/${ckpt}
echo "=========================================="
echo "      🚀 lightthinker_epl评估完成      "
echo "=========================================="


echo "=========================================="
echo "      🚀 lightthinker_epl_mtp评估开始     "
echo "=========================================="
output_path="/tmp/hx/rrcot/lightthinker_epl_mtp"
model_tag="lighthinker_epl_mtp"
ckpt="1305"
output_tag="${output_path}/${model_tag}/inference"
bash our_evaluate.sh --method anchor-thought --dataset bbh --base_path ${output_tag}/bbh/${ckpt}
bash our_evaluate.sh --method anchor-thought --dataset gpqa --base_path ${output_tag}/gpqa/${ckpt}
bash our_evaluate.sh --method anchor-thought --dataset gsm8k --base_path ${output_tag}/gsm8k/${ckpt}
bash our_evaluate.sh --method anchor-thought --dataset mmlu --base_path ${output_tag}/mmlu/${ckpt}
echo "=========================================="
echo "      🚀 lightthinker_epl_mtp评估完成      "
echo "=========================================="



echo "=========================================="
echo "      🚀 vanilla评估开始     "
echo "=========================================="
output_path="/tmp/hx/rrcot/vanilla"
model_tag="vanilla_inference"
ckpt="1305"
output_tag="${output_path}/${model_tag}/inference"
bash our_evaluate.sh --method anchor-thought --dataset bbh --base_path ${output_tag}/bbh/${ckpt}
bash our_evaluate.sh --method anchor-thought --dataset gpqa --base_path ${output_tag}/gpqa/${ckpt}
bash our_evaluate.sh --method anchor-thought --dataset gsm8k --base_path ${output_tag}/gsm8k/${ckpt}
bash our_evaluate.sh --method anchor-thought --dataset mmlu --base_path ${output_tag}/mmlu/${ckpt}
echo "=========================================="
echo "      🚀 vanilla评估完成      "
echo "=========================================="