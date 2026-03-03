**Big Picture**
- **目标**: 本仓库（RRcot）主要包含模型推理/评估、Inference 结果整理及轻量化/压缩相关的脚本与配置。核心流程是把模型/结果以.jsonl 格式产出，再用评估脚本批量计算指标。
- **主要组件**: 模型/权重目录、推理/训练脚本（shell 脚本集合）、评估代码（`evaluation/`）。把数据、配置、脚本分离：推理产物放在 `infer_result*`/`inference_results` 等目录，评估从这些目录里读取 `.jsonl` 文件。

**Key Scripts & Workflows**
- **批量评估**: 使用 [RRcot/our_evaluate.sh](RRcot/our_evaluate.sh#L1)。该脚本自动扫描 `--base_path` 下的 `.jsonl` 文件并调用 `evaluation/eval_file.py`。
  - 重要参数：`--method`（anchor-token|normal|kvcache|anchor-thought）、`--dataset`（gsm8k|gpqa|mmlu|bbh）、`--tokenizer_path`、`--comp_config`（默认 `configs/LightThinker/qwen/v1.json`）、`--cache_size`、`--interaction`。
  - 典型命令示例（仓库根目录下运行）:

```bash
bash RRcot/our_evaluate.sh --method anchor-thought --dataset bbh --base_path /path/to/your/inf_results_dir
```

- **评估实现入口**: [evaluation/eval_file.py](evaluation/eval_file.py)（`our_evaluate.sh` 会调用），查看该文件以了解 `--method` 如何被调度和实现。

- **推理/训练脚本**: 仓库根目录下有一系列 shell 脚本（`train.sh`, `inference.sh`, `our_inference.sh`, `inference_full.sh` 等）。它们通常：
  - 设置环境变量（conda/miniconda 环境在仓库中存在）
  - 生成/指定输出目录（推理结果以 `.jsonl` 保存）
  - 使用特定的配置文件（`configs/LightThinker/...`）

**Project Conventions & Patterns**
- 输入/输出格式：评估端以每个案例一行的 `.jsonl` 文件为输入（单文件可包含多个样本）。脚本按目录扫描并按文件名排序处理。
- 参数传递：多数 shell 脚本将参数拼接成 `python` 命令数组并直接执行（见 `our_evaluate.sh`），因此在添加参数时保持一致的 `--key value` 风格。
- 配置优先级：脚本会带默认值（例如 tokenizer path、comp_config），如需覆盖请使用对应 `--` 参数；脚本会检查 `--base_path` 是否存在并在不存在时退出。
- 交互模式：`--interaction` 为布尔 flag，启用后会将该参数传给 Python 评估入口。

**Integration Points & External Dependencies**
- 外部 tokenizer/模型路径：默认指向集群或共享存储（脚本中的 `DEFAULT_TOKENIZER_PATH`），在本地测试时请改为可访问的路径。
- 压缩/配置 JSON：`--comp_config` 指向 `configs/LightThinker/...` 下的 JSON，修改/扩展算法或超参时优先更新这些配置。
- Python 环境：脚本将 `PYTHONPATH` 加入当前工作目录（`export PYTHONPATH=$PYTHONPATH:$(pwd)`），推荐在仓库根（`RRcot`）执行，且先激活合适的 conda/venv。

**Where To Look (quick links)**
- 评估入口脚本: [RRcot/our_evaluate.sh](RRcot/our_evaluate.sh#L1)
- Python 评估实现: [evaluation/eval_file.py](evaluation/eval_file.py)
- 配置示例: [RRcot/configs/LightThinker/qwen/v1.json](RRcot/configs/LightThinker/qwen/v1.json)
- 推理脚本集合: RRcot 根目录下的 `*.sh`（如 `inference.sh`, `train.sh`）

**How AI agents should operate here**
- 先在 `RRcot` 根目录运行小规模本地试验：准备一个包含 1-2 个样本的 `.jsonl` 文件放到临时目录，运行 `our_evaluate.sh --base_path <tmpdir>` 来验证端到端路径是否正确。
- 修改或添加评估方法时：优先在 `evaluation/eval_file.py` 中查找 `--method` 的分发点并添加小而明确的单元（易于本地验证）。
- 当需要更改默认路径/配置，更新 `our_evaluate.sh` 的默认常量并保留原来的注释示例命令以便开发者复制使用。

**Short Checklist for common tasks**
- 运行评估：`bash RRcot/our_evaluate.sh --base_path /abs/path/to/jsonl_dir`。
- 调试评估：在仓库根手动运行 `python evaluation/eval_file.py --help` 查看参数并用单文件调试。
- 添加新方法：在 `evaluation/` 中定位 method 分发，添加实现并更新 `our_evaluate.sh` 的 `--method` 注释。

如需我把某一节扩展为更详细的示例（例如逐行解释 `evaluation/eval_file.py` 的关键分支，或把常用 conda 环境/requirements.txt 写成运行步骤），告诉我想要的部分，我会继续迭代。