# MTP 投机解码 Debug 指南

排查 **MTP + 投机采样解码**（`--spec_decode true`）下答案格式被破坏的问题，
典型现象：`\boxed{D}` 生成成 `\boxedD}`（`\boxed` 后少一个 `{`），导致大量
answer 解析 error。**自回归推理（`--spec_decode false`）不复现**。

---

## 1. 背景：问题定位到哪一步

已排除的原因：

- **不是 repetition penalty**：`--repetition_penalty 1.0` 下仍系统性出错，
  说明不是「数值噪声 + 惩罚放大」（那只会偶发）。
- **不是 output 记账丢 token**：`show_output_input_ids` 的 append 逐分支核对过是连续的。
- **非压缩步在 rp=1.0 下可证明无损**：d0 取自 anchor 干净因果 logit（= AR argmax），
  verify 用普通因果 mask + 连续 position，被接受的投机 token 一定"验证正确"，
  trim 的 cache 索引无 off-by-one。

因此系统性错误只可能来自**绕过上述无损保证**的路径（`use_EPL=true` + 压缩开启时）：

1. **压缩分支（`IS_COMP_MODE`）的 register + verify 交互**——压缩步 cache 索引复杂
   （先删原文再删 register），`\boxed{...}` 常紧跟一次压缩的 `<|continue|>`。
2. **`<|continue|>` 后的 d0 无验证提交**——压缩步 d0 = `continue` 位置 argmax，
   同样无条件接受、无 verify 兜底；`continue` logit 有系统性偏差就直接错。
3. **EPL position 重排**——主 forward 给 register 的位置走 EPL 压缩坐标
   （`continue_epl_pos + 1..`），而 verify 给 draft 的位置是「原始顺序位置 −
   `use_compression_all_count`」。两套坐标在压缩步一旦对不齐，verify 就在错误
   位置算 `verify_preds`，系统性接受"跳过结构 token"的投机，并把错位 KV 写进 cache。

核心可疑代码位置（`LightThinker/inference.py`）：

- register 位置（主 forward，EPL）：`_sentence_level_mtp_register_generate` 内
  `if register_token_count > 0: ... register_position_ids = torch.arange(last_pos+1, ...)`
- verify 位置：`verify_position_ids = verify_position_ids - use_compression_all_count`
- d0（mandatory，无验证）：`accepted_len = 1` 起步，只 verify `d2..dk`

---

## 2. 诊断埋点（已内置，env 门控）

`_sentence_level_mtp_register_generate` 里已加 `MTP_DBG` 门控的诊断打印，
**不设 `MTP_DBG=1` 时零开销、不影响正常跑**。它在每个进入 verify 的步打印：

```
[MTP_DBG step=<n> COMP=<bool> ucc=<use_compression_all_count> rc=<register_count>]
  main_pos   =[...]     # 主 forward 的 position_ids
  main_ids   =[...]     # 主 forward 的 input_ids（已 decode）
  verify_pos =[...]     # verify forward 给 draft 的 position_ids
  drafts     =[...]     # draft tokens（d0=mandatory, d1..=register 投机；已 decode）
  vpreds     =[...]     # verify 的逐位预测（已 decode）
  accepted_len=<n>      # 实际接受长度（含 d0）
```

---

## 3. 运行

用现成的 MTP 命令，加 `MTP_DBG=1`，**单进程**（`--split_size 1 --index 1`），
把 stdout 落盘：

```bash
cd /mnt/lxy/MemoSight
MTP_DBG=1 CUDA_VISIBLE_DEVICES=0 python LightThinker/inference.py \
  --model_path /你的/mtp-checkpoint \
  --tokenizer_path /你的/tokenizer \
  --compress_config configs/LightThinker/qwen/adaptive_mtp_v1.json \
  --model_type qwen --bos_token '<|im_start|>' --eos_token '<|im_end|>' \
  --max_new_tokens 2048 --update_attention_method local \
  --use_EPL true --spec_decode true \
  --datasets gsm8k --split_size 1 --index 1 \
  --model_tag dbg --model_short_tag dbg --ckpt 0 --output_tag /tmp/mtp_dbg \
  2>&1 | tee /tmp/mtp_dbg.log
```

> 日志会很大，跑出一两个含 `\boxed` 的样本即可 Ctrl-C。

定位答案区：

```bash
grep -n boxed /tmp/mtp_dbg.log
```

看命中行**附近的 `COMP=True` 步**。

---

## 4. 判读（关键判据）

压缩步主 forward 的布局是 `[..split.., <o_0..o_k>, continue, reg_1..reg_rc]`：

- **register 槽位** = `main_pos[-rc:]`（主 forward 就是在这些位置预测出 `drafts[1:]`）
- **anchor（d0 的来源）= `continue`** = `main_pos[-(rc+1)]`

**应当成立的不变式：**

1. `verify_pos` 连续：`[continue_pos+1, +2, +3]`
2. `verify_pos[0] == continue_pos + 1 == main_pos[-rc]`
   （verify 里 d0 的位置 = 主 forward 里第一个 register 的位置 = continue 的下一格）

**判定：**

| 现象 | 结论 |
|---|---|
| `COMP=True` 步里 `verify_pos[0] != main_pos[-rc]`，或两套差了约 `ucc` 的常数 | **EPL-verify 位置错位**（hypothesis 3）——主 forward 用压缩坐标放 register，verify 用「原始−ucc」放 draft，压缩步对不齐 |
| `drafts[0]`（d0）已不是 `{`，但 `vpreds`/后续是对的 | **压缩步 d0 无验证翻车**（hypothesis 2）——`continue` 位置 logit 系统性偏差 |
| `drafts` 对，但 `accepted_len` 接受了不该接受的 | **verify 位置/mask 问题**（hypothesis 1/3） |

把答案区附近几条 `COMP=True` 的四行（`main_pos` / `verify_pos` / `drafts` / `vpreds`）
拎出来对照，即可区分是「位置错位」还是「d0 翻车」。

---

## 5. 对应修法（确认后再动）

- **EPL-verify 位置错位** → 让 verify 的 `verify_position_ids` 复用主 forward 给
  register 的那套 EPL 坐标（`continue_epl_pos + 1..`），而不是「原始 − ucc」。
- **压缩步 d0 翻车** → 给 d0 也加验证：verify forward（干净因果）里本就有 anchor
  位置的干净单步预测，d0 与其不一致时以 verify 版为准（等于不再无条件接受 mandatory）。
- 快速缓解验证：先跑 `--repetition_penalty 1.0` 排除惩罚干扰（已知不能根治，仅用于隔离变量）。

---

## 附：相关文件

- 主逻辑：`LightThinker/inference.py` → `_sentence_level_mtp_register_generate`
- 自回归对照：同文件 `_sentence_level_generate`（不含 register/verify/EPL-verify）
- 分支选择：`generate()` 内 `if comp_config.mtp_cfg and spec_decode:`
- 接受率聚合（非本问题，但相关）：`scripts/analyze_mtp_acceptance.py`
