eval "$(/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/miniconda/bin/conda shell.bash hook)"
which conda
conda activate lightinfer
which python

cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/RRcot

export PYTHONPATH="$(pwd):${PYTHONPATH}"

# model_path="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/RRcot/output/baseline_7b_normal/checkpoint-1305"
# datasets="mmlu gsm8k gpqa bbh"

# model_path="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/models/DeepSeek-R1-Distill-Qwen-1.5B"
# datasets="gpqa_cot bbh_cot" # mmlu_cot gsm8k_cot gpqa_cot bbh_cot

# model_path="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/RRcot/output/baseline_1.5b_normal/checkpoint-1305"
# datasets="mmlu gsm8k gpqa bbh"

model_path="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/models/DeepSeek-R1-Distill-Qwen-7B"
datasets="mmlu_cot gsm8k_cot gpqa_cot bbh_cot"

batch_size=16
output_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/RRcot/sglang_inference_results"
# extend_name="inf_baseline_r1distillqwen7b"
# extend_name="inf_cot_r1distillqwen1.5b"
# extend_name="inf_baseline_r1distillqwen1.5b_again"
extend_name="inf_cot_r1distillqwen7b"

root_dir="./LightThinker"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python "${root_dir}/sglang_inference.py" \
  --model_path $model_path \
  --datasets $datasets \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --extend_name $extend_name
