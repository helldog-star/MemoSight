#!/bin/bash


# eval "$(/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/miniconda/bin/conda shell.bash hook)"
# conda activate lightthinker
# echo "Using python: $(which python)"


# ROOT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/RRcot"
# cd $ROOT_DIR
# export PYTHONPATH=$PYTHONPATH:$(pwd)

# echo "[INFO] Working directory: $(pwd)"


echo "================ TRAINING ================"
bash ./our_train_lighthinker_epl_mtp.sh
echo "=========== TRAINING DONE ==============="


echo "================ INFERENCE ================"
bash ./our_inference_repe.sh
echo "=========== INFERENCE DONE ==============="


output_path="/mnt/zhaorunsong/lx/rrcot_test"
model_tag="lighthinker_epl_mtp_lambda_1d0_7b_aug-wo-pc"
ckpt="1305"
output_tag="${output_path}/${model_tag}_ckpt${ckpt}_fix_infer"
evaluation_data_path="评估数据集的路径末尾是eval文件夹 zhaorunsong/data/eval"

echo "================ EVALUATION ================"
bash our_evaluate.sh --method anchor-thought --dataset bbh --base_path ${output_tag}/bbh/${ckpt} --evaluation_data_path ${evaluation_data_path}
bash our_evaluate.sh --method anchor-thought --dataset gpqa --base_path ${output_tag}/gpqa/${ckpt} --evaluation_data_path ${evaluation_data_path}
bash our_evaluate.sh --method anchor-thought --dataset gsm8k --base_path ${output_tag}/gsm8k/${ckpt} --evaluation_data_path ${evaluation_data_path}
bash our_evaluate.sh --method anchor-thought --dataset mmlu --base_path ${output_tag}/mmlu/${ckpt} --evaluation_data_path ${evaluation_data_path}
echo "=========== EVALUATION DONE ==============="

echo ""
echo "=========================================="
echo "      🚀 所有流程完成！一次性执行结束      "
echo "=========================================="
