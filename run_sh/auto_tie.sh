#!/bin/bash

# ==================== 路径配置 ====================
# 所有路径统一在此设置，便于在不同服务器上运行
ROOT_DIR="/zhaorunsong/RRcot"  # 项目根目录
INFERENCE_ROOT_DIR="${ROOT_DIR}/LightThinker"  # 推理脚本使用的代码根目录

# 输出路径配置
OUTPUT_BASE_DIR="/tmp/hx/rrcot"  # 所有输出（训练、推理）的基础目录

# 模型和Tokenizer路径配置
TOKENIZER_PATH="/tmp/hx/Qwen/Qwen2.5-1.5B-Instruct"  # Tokenizer路径
MODEL_PATH="/tmp/hx/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 预训练模型路径

# 训练数据路径配置
TRAIN_DATA_PATH="/home/zhaorunsong.zrs/repo/RRcot/data/train/train.jsonl"  # 训练数据路径

# Conda环境配置（用于sglang_inference.sh）
CONDA_SH_PATH="/mnt/zhaorunsong/anaconda3/etc/profile.d/conda.sh"  # Conda初始化脚本路径
CONDA_ENV_NAME="niah"  # Conda环境名称

# ==================== 推理和评估配置 ====================
# 设置推理和评估的默认参数
REPETITION_PENALTY="1.1"  # 重复惩罚系数
CKPT="1305"  # 检查点编号，可以根据实际情况修改
EVAL_METHOD="anchor-thought"  # 评估方法：anchor-thought 或 normal
DATASETS=("bbh" "gpqa" "gsm8k" "mmlu")  # 要评估的数据集

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.sh"
INFERENCE_SCRIPT="${SCRIPT_DIR}/inference.sh"
SGLANG_INFERENCE_SCRIPT="${SCRIPT_DIR}/sglang_inference.sh"
EVALUATE_SCRIPT="${SCRIPT_DIR}/evaluate.sh"


# 1. vanilla (baseline, 不使用压缩方法)
echo "=======🚀 vanilla开始训练 ======="
bash ${TRAIN_SCRIPT} "${ROOT_DIR}" "vanilla" "False" "1e-5" "normal" "None" "${OUTPUT_BASE_DIR}" "${TOKENIZER_PATH}" "${MODEL_PATH}" "${TRAIN_DATA_PATH}"
if [ $? -ne 0 ]; then
    echo "❌ vanilla训练失败"
    exit 1
fi
echo "=======✅ vanilla结束训练 ======="

# 2. lightthinker (baseline, 使用压缩方法)
echo "=======🚀 lightthinker开始训练 ======="
bash ${TRAIN_SCRIPT} "${ROOT_DIR}" "lightthinker" "False" "2e-5" "aug-wo-pc" "None" "${OUTPUT_BASE_DIR}" "${TOKENIZER_PATH}" "${MODEL_PATH}" "${TRAIN_DATA_PATH}"
if [ $? -ne 0 ]; then
    echo "❌ lightthinker训练失败"
    exit 1
fi
echo "=======✅ lightthinker结束训练 ======="

# 3. lightthinker_epl (使用压缩方法+EPL，不使用MTP)
echo "=======🚀 lightthinker_epl开始训练 ======="
bash ${TRAIN_SCRIPT} "${ROOT_DIR}" "lightthinker_epl" "True" "2e-5" "aug-wo-pc" "None" "${OUTPUT_BASE_DIR}" "${TOKENIZER_PATH}" "${MODEL_PATH}" "${TRAIN_DATA_PATH}"
if [ $? -ne 0 ]; then
    echo "❌ lightthinker_epl训练失败"
    exit 1
fi
echo "=======✅ lightthinker_epl结束训练 ======="

# 4. lightthinker_epl_mtp (使用EPL和MTP normal模式)
echo "=======🚀 lightthinker_epl_mtp开始训练 ======="
bash ${TRAIN_SCRIPT} "${ROOT_DIR}" "lightthinker_epl_mtp" "True" "2e-5" "aug-wo-pc" "configs/mtp_aux_config.json" "${OUTPUT_BASE_DIR}" "${TOKENIZER_PATH}" "${MODEL_PATH}" "${TRAIN_DATA_PATH}"
if [ $? -ne 0 ]; then
    echo "❌ lightthinker_epl_mtp训练失败"
    exit 1
fi
echo "=======✅ lightthinker_epl_mtp结束训练 ======="

# 5. lightthinker_epl_mtp_midlayer (使用EPL和MTP midlayer模式)
echo "=======🚀 lightthinker_epl_mtp_midlayer开始训练 ======="
bash ${TRAIN_SCRIPT} "${ROOT_DIR}" "lightthinker_epl_mtp_midlayer" "True" "2e-5" "aug-wo-pc" "configs/mtp_aux_midlayer_config.json" "${OUTPUT_BASE_DIR}" "${TOKENIZER_PATH}" "${MODEL_PATH}" "${TRAIN_DATA_PATH}"
if [ $? -ne 0 ]; then
    echo "❌ lightthinker_epl_mtp_midlayer训练失败"
    exit 1
fi
echo "=======✅ lightthinker_epl_mtp_midlayer结束训练 ======="

# 6. lightthinker_epl_mtp_cross_attn (使用EPL和MTP cross-attention模式)
echo "=======🚀 lightthinker_epl_mtp_cross_attn开始训练 ======="
bash ${TRAIN_SCRIPT} "${ROOT_DIR}" "lightthinker_epl_mtp_cross_attn" "True" "2e-5" "aug-wo-pc" "configs/mtp_aux_cross_attn_config.json" "${OUTPUT_BASE_DIR}" "${TOKENIZER_PATH}" "${MODEL_PATH}" "${TRAIN_DATA_PATH}"
if [ $? -ne 0 ]; then
    echo "❌ lightthinker_epl_mtp_cross_attn训练失败"
    exit 1
fi
echo "=======✅ lightthinker_epl_mtp_cross_attn结束训练 ======="

echo "=========================================="
echo "      ✅ 所有模型训练完成     "
echo "=========================================="

# ==================== 推理和评估部分 ====================
# 定义模型列表和对应的评估方法、推理脚本
# 格式: "model_tag:eval_method:inference_script"
# inference_script: "inference" 或 "sglang_inference"
declare -a models=(
    "vanilla:normal:sglang_inference"
    "lightthinker:anchor-thought:inference"
    "lightthinker_epl:anchor-thought:inference"
    "lightthinker_epl_mtp:anchor-thought:inference"
    "lightthinker_epl_mtp_midlayer:anchor-thought:inference"
    "lightthinker_epl_mtp_cross_attn:anchor-thought:inference"
)

# 为每个模型运行推理和评估
for model_config in "${models[@]}"; do
    IFS=':' read -r model_tag eval_method inference_script_type <<< "$model_config"
    
    echo ""
    echo "=========================================="
    echo "      🚀 ${model_tag} 开始推理     "
    echo "=========================================="
    
    # 根据模型类型选择推理脚本
    if [ "$inference_script_type" = "sglang_inference" ]; then
        INFERENCE_CMD="${SGLANG_INFERENCE_SCRIPT}"
        echo "使用 sglang_inference.sh 进行推理"
    else
        INFERENCE_CMD="${INFERENCE_SCRIPT}"
        echo "使用 inference.sh 进行推理"
    fi
    
    # 运行推理（两个脚本使用相同的参数格式）
    if [ "$inference_script_type" = "sglang_inference" ]; then
        # sglang_inference需要额外的conda路径参数
        bash ${INFERENCE_CMD} "${model_tag}" "${REPETITION_PENALTY}" "${CKPT}" "${INFERENCE_ROOT_DIR}" "${OUTPUT_BASE_DIR}" "${CONDA_SH_PATH}" "${CONDA_ENV_NAME}"
    else
        # inference.sh需要output_base_dir和tokenizer_path
        bash ${INFERENCE_CMD} "${model_tag}" "${REPETITION_PENALTY}" "${CKPT}" "${INFERENCE_ROOT_DIR}" "${OUTPUT_BASE_DIR}" "${TOKENIZER_PATH}"
    fi
    if [ $? -ne 0 ]; then
        echo "❌ ${model_tag} 推理失败"
        continue  # 继续下一个模型，不退出
    fi
    
    echo "=======✅ ${model_tag} 推理完成 ======="
    
    # 等待推理完全完成（确保所有后台进程结束）
    # sglang_inference 是同步的，inference 是异步的，统一等待
    sleep 10
    
    # 运行评估
    echo ""
    echo "=========================================="
    echo "      🚀 ${model_tag} 评估开始     "
    echo "=========================================="
    
    output_path="${OUTPUT_BASE_DIR}/${model_tag}"
    output_tag="${output_path}/inference"
    
    for dataset in "${DATASETS[@]}"; do
        base_path="${output_tag}/${dataset}"
        
        # 检查推理结果是否存在
        if [ ! -d "$base_path" ]; then
            echo "⚠️  警告: 推理结果路径不存在: ${base_path}"
            echo "跳过 ${dataset} 数据集的评估"
            continue
        fi
        
        echo "评估数据集: ${dataset}"
        # 使用位置参数调用评估脚本: method tokenizer_path dataset base_path
        bash ${EVALUATE_SCRIPT} "${eval_method}" "${TOKENIZER_PATH}" "${dataset}" "${base_path}"
        
        if [ $? -ne 0 ]; then
            echo "❌ ${model_tag} 在 ${dataset} 数据集上评估失败"
        else
            echo "✅ ${model_tag} 在 ${dataset} 数据集上评估完成"
        fi
    done
    
    echo "=========================================="
    echo "      ✅ ${model_tag} 评估完成     "
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "      ✅ 所有模型推理和评估完成     "
echo "=========================================="