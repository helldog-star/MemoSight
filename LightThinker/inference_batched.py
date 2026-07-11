"""
Batched (batch>1) inference for the MemoSight / LightThinker compression decoder.

This is a PARALLEL implementation to the single-sequence engine in `inference.py`.
The bs=1 loops in `inference.py` (`_sentence_level_generate`, `_sentence_level_mtp_register_generate`)
are left untouched and used as the correctness ORACLE.

Design (see plan): per-sample bookkeeping arrays + one LEFT-PADDED batched KV cache of
width L_pad = max_s valid_len[s]. Physical compression is preserved: on a compression step
(or MTP ragged-accept) we gather-rebuild the cache keeping only each sample's live columns,
so attention compute + peak memory actually shrink -- which is what the speedup measurement needs.

Scope of THIS file (milestone: vanilla / lightthinker):
    - update_attention_method == 'local'
    - use_EPL == False, rolling_rope == False, compress_prompt == False
    - greedy decoding (repetition_penalty == 1.0)
    - attn_implementation forced to 'sdpa'
MTP (`_sentence_level_mtp_register_generate_batched`) is added in a later step.

Everything else asserts out so we never silently produce wrong numbers.
"""

import os
import sys
import time
import json
import argparse
import torch
import jsonlines
from typing import *
from tqdm import tqdm

# reuse everything from the single-sequence engine / project
from config import Config
from tokenizer import Tokenizer
from inference import (
    InferenceUtils,
    get_model_and_tokenizer,
    generate as oracle_generate,
    AttentionUtils,
    TokenUtils,
    MTPStats,
)
from dataset_reader import GPQAReader, MMLUReader, BBHReader, GSM8KReader, Reader
from utils import str2bool


# --------------------------------------------------------------------------- #
# Per-sample control state (parallels the scalars in _sentence_level_generate) #
# --------------------------------------------------------------------------- #
class BatchState:
    """Per-sample bookkeeping for the *active* working set.

    All arrays are Python lists indexed by the WORKING row (0..B_active-1); `orig_index`
    maps a working row back to its original sample id so outputs land in the right slot.
    Finished samples are removed from the working set (active-row compaction).
    """

    def __init__(self, orig_index: List[int], prompt_ids: List[List[int]],
                 valid_len: List[int], last_pos: List[int]):
        n = len(orig_index)
        self.orig_index: List[int] = list(orig_index)        # working row -> original sample id
        self.prompt_ids: List[List[int]] = prompt_ids        # per working row (shown prompt ids)
        self.out_ids: List[List[int]] = [[] for _ in range(n)]  # committed output ids per working row
        self.valid_len: List[int] = list(valid_len)          # #real tokens in cache per row
        self.last_pos: List[int] = list(last_pos)            # last position_id emitted per row
        # local-compression bookkeeping (logical coords, 0..valid_len)
        self.local_start_off: List[int] = list(valid_len)    # where current CoT begins (= end of prompt initially)
        self.van_cot_start: List[int] = list(last_pos)       # position where current CoT segment starts
        for i in range(n):
            self.van_cot_start[i] = last_pos[i] + 1           # first generated position
        self.step_count: List[int] = [0] * n                 # committed new tokens per row

    def __len__(self):
        return len(self.orig_index)


# --------------------------------------------------------------------------- #
# Batched KV cache with gather-based compaction (replaces KVUtils.reduce_cache) #
# --------------------------------------------------------------------------- #
class BatchKVUtils:
    def __init__(self):
        from transformers import DynamicCache
        self.past_key_values = DynamicCache()

    def get_cache(self):
        return self.past_key_values

    @property
    def width(self) -> int:
        if len(self.past_key_values.key_cache) == 0:
            return 0
        return self.past_key_values.key_cache[0].shape[2]

    @torch.no_grad()
    def rebuild(self, row_keep: List[int], col_keep: List[List[int]], device):
        """Gather-rebuild the cache: keep working rows `row_keep`; for kept row i keep cache
        columns `col_keep[i]` (ordered). Produces a LEFT-PADDED [R, H, L_new, D] cache where
        each row's kept columns are RIGHT-aligned (newest at the far right) and pad slots
        (index 0, masked by attention) sit on the left.

        Returns new_valid_len: List[int] for the kept rows.
        """
        pkv = self.past_key_values
        n_layers = len(pkv.key_cache)
        assert n_layers > 0
        R = len(row_keep)
        new_valid_len = [len(col_keep[i]) for i in range(R)]
        L_new = max(new_valid_len) if R > 0 else 0

        # gather_index[R, L_new]: right-aligned real cols, left pad -> index 0
        gather_index = torch.zeros((R, L_new), dtype=torch.long, device=device)
        for i in range(R):
            cols = col_keep[i]
            n = len(cols)
            if n > 0:
                gather_index[i, L_new - n:] = torch.tensor(cols, dtype=torch.long, device=device)

        row_keep_t = torch.tensor(row_keep, dtype=torch.long, device=device)
        for l in range(n_layers):
            K = pkv.key_cache[l]                       # [B, H, L_old, D]
            V = pkv.value_cache[l]
            B0, Hh, L_old, Dd = K.shape
            K = K.index_select(0, row_keep_t)          # [R, H, L_old, D]
            V = V.index_select(0, row_keep_t)
            idx = gather_index[:, None, :, None].expand(R, Hh, L_new, Dd)
            pkv.key_cache[l] = torch.gather(K, 2, idx).contiguous()
            pkv.value_cache[l] = torch.gather(V, 2, idx).contiguous()
        pkv._seen_tokens = L_new
        return new_valid_len


# --------------------------------------------------------------------------- #
# Batched additive attention masks ([B,1,W,L_pad+W]) -- local method          #
# --------------------------------------------------------------------------- #
class BatchAttnUtils:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device
        self.min_dtype = torch.finfo(dtype).min

    @torch.no_grad()
    def build_local_step(
        self,
        L_pad: int,
        W: int,
        valid_len: List[int],          # per row, real tokens currently in cache
        step_keep: List[int],          # per row, #real step tokens (right-aligned in W)
        comp_indicator: List[Optional[Tuple[int, int, int]]],  # per row: (text_start_off, text_end_off, continue_rel) or None
        reg_count: int = 0,            # MTP: trailing reg_count step tokens are registers (block-right-aligned)
    ) -> torch.Tensor:
        """Build additive mask; 0 visible, min_dtype hidden.
        Column layout: [0:L_pad] = cache (left-padded per row), [L_pad:L_pad+W] = new query block.
        comp_indicator offsets are LOGICAL (relative to the row's real-token start).
        """
        R = len(valid_len)
        mv = self.min_dtype
        mask = torch.zeros((R, 1, W, L_pad + W), dtype=self.dtype, device=self.device)

        base_tri = torch.triu(
            torch.full((W, W), mv, dtype=self.dtype, device=self.device), diagonal=1
        )
        for b in range(R):
            pad_left_kv = L_pad - valid_len[b]           # masked cache columns on the left
            pad_left_q = W - step_keep[b]                # dead query rows on top (left pad of step)
            # 1. causal on new-query block
            mask[b, 0, :, L_pad:L_pad + W] = base_tri
            # 1b. in-block LEFT PAD: when W>1 (some other row compressed), this row's step is
            #     right-aligned; its leading (W-step_keep) block columns are PAD tokens and must
            #     be hidden from ALL query rows, else the real token attends pad via causality.
            if pad_left_q > 0:
                mask[b, 0, :, L_pad:L_pad + pad_left_q] = mv
            # 2. left-pad of KV columns -> hidden
            if pad_left_kv > 0:
                mask[b, 0, :, 0:pad_left_kv] = mv
            # 3. dead (left-pad) query rows: keep them from being all -inf via diagonal guard below;
            #    also they must not see real content (irrelevant, discarded) -- leave as-is.
            # 4. compression indicator: continue-token (and after) must not attend the raw CoT text.
            ind = comp_indicator[b]
            if ind is not None:
                text_start_off, text_end_off, continue_rel = ind
                # absolute cache columns of the CoT text span
                real_base = L_pad - valid_len[b]
                text_start_abs = real_base + text_start_off
                text_end_abs = real_base + text_end_off
                # query rows from `continue_rel` onward (within the real step block) mask the text.
                # real step block occupies query rows [pad_left_q, W); continue is at pad_left_q + continue_rel.
                q_from = pad_left_q + continue_rel
                if text_end_abs > text_start_abs and q_from < W:
                    mask[b, 0, q_from:, text_start_abs:text_end_abs] = mv
            # 4b. MTP register-vs-register: registers are the trailing reg_count real step
            #     tokens -> block cols [W-reg_count, W). A register query must NOT attend the
            #     OTHER registers of the same step (parallels inference.py:667-672).
            if reg_count > 0:
                reg_lo = W - reg_count
                for q in range(reg_lo, W):
                    for kk in range(reg_lo, W):
                        if kk != q:
                            mask[b, 0, q, L_pad + kk] = mv
            # 5. _unmask_unattended guard: every query row must attend >=1 key. Force self-diagonal
            #    on the new block (row q attends its own new-block column q). Real rows already do via
            #    causal; this also covers dead/left-pad rows harmlessly.
            diag = torch.arange(W, device=self.device)
            mask[b, 0, diag, L_pad + diag] = 0.0
        return mask


# --------------------------------------------------------------------------- #
# Batched greedy argmax (parallels InferenceUtils.get_predicted_token_ids)     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batched_argmax(model_output, idx: int = -1) -> torch.Tensor:
    logits = model_output.logits            # [B, seq, vocab]
    return torch.argmax(logits[:, idx, :], dim=-1)   # [B]


# --------------------------------------------------------------------------- #
# Batched prefill (compress_prompt == False)                                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batched_prefill(
    model,
    tokenizer: Tokenizer,
    comp_config: Config,
    system_prompt: str,
    questions: List[str],
    kv_utils: BatchKVUtils,
    attn_utils: BatchAttnUtils,
    device: str,
) -> Tuple[torch.Tensor, List[List[int]], List[int]]:
    """Returns (pred[B], prompt_ids_per_sample, real_len_per_sample)."""
    B = len(questions)
    per_ids: List[List[int]] = []
    for q in questions:
        prompt = tokenizer.bos_token + comp_config.template_cfg['complete'].format(
            system=system_prompt, question=q
        )
        ids = tokenizer.tokenizer(prompt, return_tensors=None, add_special_tokens=False)['input_ids']
        per_ids.append(ids)

    real_len = [len(x) for x in per_ids]
    Lp = max(real_len)
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    input_ids = torch.full((B, Lp), pad_id, dtype=torch.long, device=device)
    position_ids = torch.zeros((B, Lp), dtype=torch.long, device=device)
    for b in range(B):
        n = real_len[b]
        input_ids[b, Lp - n:] = torch.tensor(per_ids[b], dtype=torch.long, device=device)
        position_ids[b, Lp - n:] = torch.arange(n, dtype=torch.long, device=device)

    # attention mask [B,1,Lp,Lp]: causal + left-pad hidden + diagonal guard
    mv = attn_utils.min_dtype
    mask = torch.zeros((B, 1, Lp, Lp), dtype=attn_utils.dtype, device=device)
    base_tri = torch.triu(torch.full((Lp, Lp), mv, dtype=attn_utils.dtype, device=device), diagonal=1)
    for b in range(B):
        mask[b, 0] = base_tri
        pad_left = Lp - real_len[b]
        if pad_left > 0:
            mask[b, 0, :, 0:pad_left] = mv
    diag = torch.arange(Lp, device=device)
    mask[:, 0, diag, diag] = 0.0

    model_output = model(
        input_ids=input_ids,
        attention_mask=mask,
        position_ids=position_ids,
        past_key_values=kv_utils.get_cache(),
        use_cache=True,
        return_dict=True,
    )
    pred = batched_argmax(model_output, idx=-1)     # last col is last real token for every row
    return pred, per_ids, real_len


# --------------------------------------------------------------------------- #
# Vanilla / LightThinker batched loop (parallels _sentence_level_generate)     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _sentence_level_generate_batched(
    model,
    tokenizer: Tokenizer,
    comp_config: Config,
    max_new_tokens: int,
    kv_utils: BatchKVUtils,
    attn_utils: BatchAttnUtils,
    state: BatchState,
    pred: torch.Tensor,          # [B] first predicted token from prefill
    device: str,
    force_split_schedule: Optional[List[set]] = None,   # TEST-ONLY: per orig-sample set of step_counts to force a split
) -> None:
    eos_id = tokenizer.eos_token_id
    split_id = comp_config.split_token_id
    continue_id = comp_config.continue_token_id
    pad_id = eos_id if eos_id is not None else 0

    pred_list: List[int] = pred.tolist()

    while len(state) > 0:
        R = len(state)
        L_pad = kv_utils.width

        # ---- 1. plan per-row step tokens ----
        step_tokens: List[List[int]] = []
        is_comp: List[bool] = []
        comp_indicator: List[Optional[Tuple[int, int, int]]] = []
        for b in range(R):
            p = pred_list[b]
            if p == split_id:
                # cot_length (non-EPL): last_pos + 2 - van_cot_start  (parallels inference.py:1455)
                cot_length = state.last_pos[b] + 2 - state.van_cot_start[b]
                comp_tokens = comp_config.get_output_comp_token_id(cot_length=cot_length)
                tokens = [split_id] + list(comp_tokens) + [continue_id]
                step_tokens.append(tokens)
                is_comp.append(True)
                # indicator: text span = [local_start_off, split_off+1); continue is last real step token.
                split_off = state.valid_len[b]              # split goes right after current real tokens
                text_start_off = state.local_start_off[b]
                text_end_off = split_off + 1
                continue_rel = len(tokens) - 1              # continue is the last token in the block
                comp_indicator.append((text_start_off, text_end_off, continue_rel))
            else:
                step_tokens.append([p])
                is_comp.append(False)
                comp_indicator.append(None)

        step_keep = [len(t) for t in step_tokens]
        W = max(step_keep)

        # ---- 2. input_ids / position_ids [R, W], right-aligned ----
        input_ids = torch.full((R, W), pad_id, dtype=torch.long, device=device)
        position_ids = torch.zeros((R, W), dtype=torch.long, device=device)
        for b in range(R):
            toks = step_tokens[b]
            wk = len(toks)
            input_ids[b, W - wk:] = torch.tensor(toks, dtype=torch.long, device=device)
            # non-EPL: sequential positions from last_pos+1
            base = state.last_pos[b]
            position_ids[b, W - wk:] = torch.arange(base + 1, base + 1 + wk, dtype=torch.long, device=device)
            # dead query rows (left pad): give them a benign position (0); masked out anyway.

        # ---- 3. mask ----
        valid_len = list(state.valid_len)
        mask = attn_utils.build_local_step(L_pad, W, valid_len, step_keep, comp_indicator)

        # ---- 4. forward (cache grows by W) ----
        model_output = model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=kv_utils.get_cache(),
            use_cache=True,
            return_dict=True,
        )

        # ---- 5. next-token predictions (last real col = rightmost) ----
        next_pred = batched_argmax(model_output, idx=-1).tolist()

        # ---- 6. commit tokens, update per-row bookkeeping ----
        # After forward, cache width = L_pad + W. For row b, the W new columns occupy
        # [L_pad, L_pad+W); its real step tokens are right-aligned at [L_pad + (W-wk), L_pad+W).
        newly_finished_rows: List[int] = []
        for b in range(R):
            toks = step_tokens[b]
            wk = len(toks)
            # Output string parity with oracle: it appends only the DRIVER token
            # (`predicted_token_id`, i.e. the split token or the plain token), NOT the
            # internal comp/continue tokens (inference.py:1442). And it counts loop
            # iterations, not tokens (inference.py:1595).
            state.out_ids[b].append(toks[0])
            state.step_count[b] += 1
            # positions DO advance over the whole block (split/comp/continue occupy cache slots)
            state.last_pos[b] = state.last_pos[b] + wk
            if is_comp[b]:
                # continue-token position = last_pos now; next CoT starts after it
                state.van_cot_start[b] = state.last_pos[b] + 1

        # ---- 6b. TEST-ONLY forced split injection (deterministic per orig sample) ----
        if force_split_schedule is not None:
            for b in range(R):
                if state.step_count[b] in force_split_schedule[state.orig_index[b]]:
                    next_pred[b] = split_id

        # ---- 7. compaction: build row_keep + col_keep, then rebuild ----
        # Determine per-row termination on the NEXT predicted token / length.
        row_keep: List[int] = []
        col_keep: List[List[int]] = []
        keep_state_idx: List[int] = []
        for b in range(R):
            wk = step_keep[b]
            real_new_start = L_pad + (W - wk)            # abs col of this row's first real new token
            real_base = L_pad - valid_len[b]             # abs col of this row's first real old token
            if is_comp[b]:
                # keep [real_base, local_start_abs) U [split_abs+1, cache_end)
                local_start_abs = real_base + state.local_start_off[b]
                split_abs = real_new_start               # split is first real new token
                kept = list(range(real_base, local_start_abs)) + \
                       list(range(split_abs + 1, L_pad + W))
                # update valid_len / local_start after compression
                new_vlen = (local_start_abs - real_base) + (L_pad + W - (split_abs + 1))
                state.valid_len[b] = new_vlen
                state.local_start_off[b] = new_vlen      # next CoT begins at the (new) end
            else:
                # keep all old real cols + the single new real col
                kept = list(range(real_base, L_pad)) + list(range(real_new_start, L_pad + W))
                state.valid_len[b] = valid_len[b] + wk    # wk == 1 here

            # termination check on next predicted token
            np_tok = next_pred[b]
            finished = False
            if np_tok == eos_id:
                finished = True
            elif state.step_count[b] >= max_new_tokens:
                finished = True
            elif state.valid_len[b] + 4 >= attn_utils_max_len(kv_utils, attn_utils):
                finished = True

            if finished:
                # commit the final predicted token (parallels oracle appending final token)
                state.out_ids[b].append(np_tok)
                newly_finished_rows.append(b)
            else:
                row_keep.append(b)
                col_keep.append(kept)
                keep_state_idx.append(b)

        # flush finished rows to a side store on the state object
        if newly_finished_rows:
            _flush_finished(state, newly_finished_rows)

        if len(row_keep) == 0:
            break

        # rebuild cache to keep only surviving rows + compressed columns
        new_valid = kv_utils.rebuild(row_keep, col_keep, device)

        # compact the state arrays to the surviving rows (in row_keep order)
        _compact_state(state, keep_state_idx, new_valid)
        pred_list = [next_pred[b] for b in keep_state_idx]


# ---- state compaction helpers (active-row removal) ----
def attn_utils_max_len(kv_utils, attn_utils) -> int:
    # max_length stored on attn_utils
    return getattr(attn_utils, "max_length", 10**9)


def _flush_finished(state: BatchState, rows: List[int]):
    if not hasattr(state, "_done"):
        state._done = {}   # orig_index -> out_ids
    for b in rows:
        state._done[state.orig_index[b]] = state.out_ids[b]


def _compact_state(state: BatchState, keep_idx: List[int], new_valid: List[int]):
    state.orig_index = [state.orig_index[i] for i in keep_idx]
    state.prompt_ids = [state.prompt_ids[i] for i in keep_idx]
    state.out_ids = [state.out_ids[i] for i in keep_idx]
    state.last_pos = [state.last_pos[i] for i in keep_idx]
    state.van_cot_start = [state.van_cot_start[i] for i in keep_idx]
    state.local_start_off = [state.local_start_off[i] for i in keep_idx]
    state.step_count = [state.step_count[i] for i in keep_idx]
    state.valid_len = list(new_valid)


# --------------------------------------------------------------------------- #
# MemoSight (MTP register + self-speculative) batched loop                     #
# parallels _sentence_level_mtp_register_generate (inference.py:1608)          #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _sentence_level_mtp_register_generate_batched(
    model, tokenizer, comp_config, max_new_tokens: int,
    kv_utils: BatchKVUtils, attn_utils: BatchAttnUtils, state: BatchState,
    pred: torch.Tensor, device: str, register_count: int,
    mtp_stats: Optional[List[Optional[MTPStats]]] = None,
) -> None:
    eos_id = tokenizer.eos_token_id
    split_id = comp_config.split_token_id
    continue_id = comp_config.continue_token_id
    reg_id = comp_config.register_token_id
    pad_id = eos_id if eos_id is not None else 0
    rc = register_count
    draft_len = rc + 1
    assert rc >= 1, "MTP batched requires register_count>=1"
    max_len = attn_utils.max_length
    pred_list: List[int] = pred.tolist()

    def _stat(b):
        return mtp_stats[state.orig_index[b]] if mtp_stats is not None else None

    while len(state) > 0:
        R = len(state)
        L_pad = kv_utils.width

        # ---- phase 1: plan step tokens (driver [+comp/continue] + registers) ----
        step_tokens, is_comp, comp_ind, n_noreg = [], [], [], []
        for b in range(R):
            p = pred_list[b]
            state.out_ids[b].append(p)                     # commit driver (parallels inference.py:1666)
            if p == split_id:
                cot_length = state.last_pos[b] + 2 - state.van_cot_start[b]
                comp = list(comp_config.get_output_comp_token_id(cot_length=cot_length))
                toks = [split_id] + comp + [continue_id] + [reg_id] * rc
                is_comp.append(True)
                n_noreg.append(1 + len(comp) + 1)
                split_off = state.valid_len[b]
                # oracle update_attention_local_for_mtp_register: n_prefix includes the trailing
                # registers, so comp_end_r == new_length-1 -> only the LAST step token (last register)
                # masks the CoT text. So the text-mask query-row start = last step token = len(toks)-1.
                comp_ind.append((state.local_start_off[b], split_off + 1, len(toks) - 1))
            else:
                toks = [p] + [reg_id] * rc
                is_comp.append(False)
                n_noreg.append(1)
                comp_ind.append(None)
            step_tokens.append(toks)
        step_keep = [len(t) for t in step_tokens]
        W = max(step_keep)

        input_ids = torch.full((R, W), pad_id, dtype=torch.long, device=device)
        position_ids = torch.zeros((R, W), dtype=torch.long, device=device)
        for b in range(R):
            wk = step_keep[b]; base = state.last_pos[b]
            input_ids[b, W - wk:] = torch.tensor(step_tokens[b], dtype=torch.long, device=device)
            position_ids[b, W - wk:] = torch.arange(base + 1, base + 1 + wk, dtype=torch.long, device=device)

        valid_len = list(state.valid_len)
        mask = attn_utils.build_local_step(L_pad, W, valid_len, step_keep, comp_ind, reg_count=rc)
        mo = model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                   past_key_values=kv_utils.get_cache(), use_cache=True, return_dict=True)

        # ---- phase 2: drafts (anchor + registers = last draft_len cols, uniform) ----
        drafts = [[int(mo.logits[b, -(draft_len - j), :].argmax()) for j in range(draft_len)] for b in range(R)]

        # ---- phase 2b: rebuild #1 -> remove registers + CoT + in-block pad ----
        row_keep = list(range(R)); col_keep = []
        for b in range(R):
            real_base = L_pad - valid_len[b]
            wk = step_keep[b]; real_new_start = L_pad + (W - wk)
            nn = n_noreg[b]
            if is_comp[b]:
                local_start_abs = real_base + state.local_start_off[b]
                split_abs = real_new_start
                kept = list(range(real_base, local_start_abs)) + list(range(split_abs + 1, real_new_start + nn))
                new_vlen = (local_start_abs - real_base) + (nn - 1)
                state.local_start_off[b] = new_vlen
            else:
                kept = list(range(real_base, L_pad)) + [real_new_start]
                new_vlen = valid_len[b] + 1
            col_keep.append(kept)
            split_pos = state.last_pos[b] + 1                # split is the first new token
            state.last_pos[b] += nn
            state.valid_len[b] = new_vlen
            if is_comp[b]:
                # MTP oracle uses cot_start = split_pos + 2 (position_ids[0]+1, then +1) at
                # inference.py:1795 -- NOT continue_pos+1 like the vanilla loop (inference.py:1519).
                state.van_cot_start[b] = split_pos + 2
        kv_utils.rebuild(row_keep, col_keep, device)
        L_pad2 = kv_utils.width
        valid_len2 = list(state.valid_len)

        # ---- phase 3: classify control; decide verify ----
        is_control = [drafts[b][0] == split_id or drafts[b][0] == eos_id for b in range(R)]
        need_verify = any((not is_control[b]) for b in range(R))

        predicted = [None] * R
        accepted_count = [1] * R
        keep_n = [0] * R    # #verify-block tokens to keep in cache per row

        if need_verify:
            v_pos = torch.zeros((R, draft_len), dtype=torch.long, device=device)
            v_ids = torch.zeros((R, draft_len), dtype=torch.long, device=device)
            for b in range(R):
                v_ids[b] = torch.tensor(drafts[b], dtype=torch.long, device=device)
                base = state.last_pos[b]
                v_pos[b] = torch.arange(base + 1, base + 1 + draft_len, dtype=torch.long, device=device)
            v_mask = attn_utils.build_local_step(L_pad2, draft_len, valid_len2,
                                                 [draft_len] * R, [None] * R, reg_count=0)
            vo = model(input_ids=v_ids, attention_mask=v_mask, position_ids=v_pos,
                       past_key_values=kv_utils.get_cache(), use_cache=True, return_dict=True)
            vpreds = [[int(vo.logits[b, -(draft_len - j), :].argmax()) for j in range(draft_len)] for b in range(R)]

            for b in range(R):
                st = _stat(b)
                if is_control[b]:
                    predicted[b] = drafts[b][0]; accepted_count[b] = 1; keep_n[b] = 0
                    if st is not None:
                        st.record_control_step()
                    continue
                acc = 1
                for i in range(draft_len - 1):
                    if vpreds[b][i] == drafts[b][i + 1]:
                        acc += 1
                    else:
                        break
                if st is not None:
                    st.record_verify(gamma=draft_len - 1, raw_accepted_len=acc)
                control_token = None; extra = []
                for tok in drafts[b][1:acc]:
                    if tok == split_id or tok == eos_id:
                        control_token = tok; break
                    extra.append(tok)
                effective = 1 + len(extra)
                keep_n[b] = effective
                state.out_ids[b].append(drafts[b][0])
                state.out_ids[b].extend(extra)
                accepted_count[b] = effective
                predicted[b] = control_token if control_token is not None else vpreds[b][acc - 1]
                state.last_pos[b] += effective
                if st is not None:
                    st.record_step(forward_passes=2, committed=effective)

            # rebuild #2: keep first keep_n[b] verify-block cols per row (ragged)
            rk2 = list(range(R)); ck2 = []
            for b in range(R):
                real_cols = list(range(L_pad2 - valid_len2[b], L_pad2))
                keep_cols = list(range(L_pad2, L_pad2 + keep_n[b]))
                ck2.append(real_cols + keep_cols)
                state.valid_len[b] = valid_len2[b] + keep_n[b]
            kv_utils.rebuild(rk2, ck2, device)
        else:
            for b in range(R):
                predicted[b] = drafts[b][0]; accepted_count[b] = 1
                st = _stat(b)
                if st is not None:
                    st.record_control_step()

        # ---- phase 4: termination + active-row compaction ----
        keep_idx, row_keep_final, col_keep_final = [], [], []
        newly_finished = []
        cur_w = kv_utils.width
        for b in range(R):
            state.step_count[b] += accepted_count[b]
            np_tok = predicted[b]
            finished = (np_tok == eos_id) or (state.step_count[b] >= max_new_tokens) \
                or (state.valid_len[b] + draft_len + 4 >= max_len)
            if finished:
                state.out_ids[b].append(np_tok)
                newly_finished.append(b)
            else:
                keep_idx.append(b)
                row_keep_final.append(b)
                col_keep_final.append(list(range(cur_w - state.valid_len[b], cur_w)))
        if newly_finished:
            _flush_finished(state, newly_finished)
        if not keep_idx:
            break
        new_valid = kv_utils.rebuild(row_keep_final, col_keep_final, device)
        _compact_state(state, keep_idx, new_valid)
        pred_list = [predicted[b] for b in keep_idx]


# --------------------------------------------------------------------------- #
# Public batched generate                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batched_generate(
    model,
    tokenizer: Tokenizer,
    comp_config: Config,
    system_prompt: str,
    questions: List[str],
    max_new_tokens: int,
    max_length: int,
    device: str,
    dtype,
    update_attention_method: str = "local",
    use_EPL: bool = False,
    spec_decode: bool = False,
    mtp_draft_len: Optional[int] = None,
    mtp_stats: Optional[List[Optional[MTPStats]]] = None,
    force_split_schedule: Optional[List[set]] = None,   # TEST-ONLY (vanilla path)
    _debug_return_ids: bool = False,                     # TEST-ONLY
) -> List[str]:
    """Returns decoded output string per sample (in the original order)."""
    assert update_attention_method == "local", "batched path currently supports local only"
    assert not use_EPL, "batched path: use_EPL not yet supported"
    assert getattr(model.config, "_attn_implementation", "sdpa") != "flash_attention_2", \
        "batched 4D masks require sdpa/eager, not flash_attention_2"
    use_mtp = bool(comp_config.mtp_cfg) and spec_decode
    if spec_decode and not comp_config.mtp_cfg:
        # parity with oracle: spec_decode on a non-MTP config falls back to plain sentence loop
        use_mtp = False

    B = len(questions)
    kv_utils = BatchKVUtils()
    attn_utils = BatchAttnUtils(dtype=dtype, device=device)
    attn_utils.max_length = max_length

    pred, prompt_ids, real_len = batched_prefill(
        model, tokenizer, comp_config, system_prompt, questions, kv_utils, attn_utils, device
    )
    last_pos = [n - 1 for n in real_len]
    state = BatchState(
        orig_index=list(range(B)),
        prompt_ids=prompt_ids,
        valid_len=list(real_len),
        last_pos=last_pos,
    )

    if comp_config.output_comp_level != "sentence":
        raise NotImplementedError("batched path currently supports output_comp_level='sentence'")

    if use_mtp:
        rc = int(mtp_draft_len) if mtp_draft_len is not None else int(comp_config.mtp_cfg.get("max_offset", 2))
        rc = max(1, rc)
        _sentence_level_mtp_register_generate_batched(
            model, tokenizer, comp_config, max_new_tokens, kv_utils, attn_utils, state, pred, device,
            register_count=rc, mtp_stats=mtp_stats,
        )
    else:
        _sentence_level_generate_batched(
            model, tokenizer, comp_config, max_new_tokens, kv_utils, attn_utils, state, pred, device,
            force_split_schedule=force_split_schedule,
        )

    # gather outputs in original order
    done: Dict[int, List[int]] = getattr(state, "_done", {})
    # any rows still in state at loop end (shouldn't normally happen) also flush
    for i in range(len(state)):
        done.setdefault(state.orig_index[i], state.out_ids[i])
    if _debug_return_ids:
        return [done.get(i, []) for i in range(B)]
    outputs: List[str] = []
    for i in range(B):
        ids = done.get(i, [])
        # strip a trailing eos for readability parity with oracle decode
        outputs.append(tokenizer.decode(ids))
    return outputs


# --------------------------------------------------------------------------- #
# Fair speedup benchmark (parallels eval_dataset)                              #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def bench_dataset(
    model, tokenizer, reader: Reader, comp_config: Config,
    dataset_name: str, batch_size: int, max_new_tokens: int, max_prompt_len: int,
    device: str, dtype, bench_mode: str, warmup_batches: int = 1,
    limit: Optional[int] = None,
) -> Dict:
    """Run the batched engine over a dataset in chunks of `batch_size`, timing throughput
    and peak memory. Accuracy is computed via the reader to prove it matches bs=1.
    """
    spec_decode = (bench_mode == "memosight")
    assert bench_mode in ("vanilla", "lightthinker", "memosight")
    n = len(reader) if limit is None else min(limit, len(reader))
    system_prompt = reader.get_system_prompt()

    total_out_tokens = 0
    total_time = 0.0
    peak_mem = 0
    total = 0
    acc = 0
    on_cuda = (device == "cuda" and torch.cuda.is_available())

    idx_all = list(range(n))
    batches = [idx_all[i:i + batch_size] for i in range(0, n, batch_size)]
    pbar = tqdm(total=len(batches), desc=f"{dataset_name}/{bench_mode}/bs{batch_size}")
    for bi, chunk in enumerate(batches):
        questions = [reader.get_prompt(idx=i) for i in chunk]
        if on_cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        outputs = batched_generate(
            model=model, tokenizer=tokenizer, comp_config=comp_config,
            system_prompt=system_prompt, questions=questions, max_new_tokens=max_new_tokens,
            max_length=max_new_tokens + max_prompt_len, device=device, dtype=dtype,
            update_attention_method="local", use_EPL=False, spec_decode=spec_decode,
        )
        if on_cuda:
            torch.cuda.synchronize()
        dt = time.time() - t0

        # warmup batches excluded from timing/memory aggregates
        if bi >= warmup_batches:
            total_time += dt
            for out in outputs:
                total_out_tokens += len(tokenizer.tokenizer(out, add_special_tokens=False)['input_ids'])
            if on_cuda:
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())

        for j, i in enumerate(chunk):
            model_answer = reader.extract_answer(outputs[j])
            gt = reader.get_answer(i)
            acc_state, _ = reader.compare_answer(model_answer, gt, i)
            total += 1
            acc += int(bool(acc_state))
        pbar.update(1)
    pbar.close()

    tps = (total_out_tokens / total_time) if total_time > 0 else 0.0
    return dict(
        dataset=dataset_name, bench_mode=bench_mode, batch_size=batch_size,
        num_samples=total, accuracy=(acc / total if total else 0.0),
        tokens_per_sec=tps, total_out_tokens=total_out_tokens,
        total_time_s=total_time, peak_mem_bytes=int(peak_mem),
        peak_mem_gb=round(peak_mem / 1024**3, 3),
    )


def _bench_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--model_type', type=str, default='qwen', choices=['qwen', 'llama'])
    p.add_argument('--tokenizer_path', type=str, default=None)
    p.add_argument('--compress_config', type=str, required=True)
    p.add_argument('--bos_token', type=str, default="<|im_start|>")
    p.add_argument('--eos_token', type=str, default="<|im_end|>")
    p.add_argument('--bench_mode', type=str, default='lightthinker',
                   choices=['vanilla', 'lightthinker', 'memosight'])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_new_tokens', type=int, default=1024)
    p.add_argument('--max_prompt_len', type=int, default=1100)
    p.add_argument('--warmup_batches', type=int, default=1)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--datasets', type=str, nargs='+', default=['gsm8k'],
                   choices=['mmlu', 'gsm8k', 'gpqa', 'bbh'])
    p.add_argument('--bench_report', type=str, default=None)
    # unused-but-accepted for get_model_and_tokenizer parity
    p.add_argument('--model_tag', type=str, default=None)
    p.add_argument('--ckpt', type=int, default=None)
    return p.parse_args()


def main():
    args = _bench_parser()
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    comp_config = Config.from_file(args.compress_config)
    model, tokenizer = get_model_and_tokenizer(args, comp_config)
    if model.get_input_embeddings().weight.shape[0] < len(tokenizer.tokenizer):
        model.resize_token_embeddings(len(tokenizer.tokenizer))

    all_tasks = {"mmlu": MMLUReader, "gsm8k": GSM8KReader, "gpqa": GPQAReader, "bbh": BBHReader}
    reports = []
    for name in args.datasets:
        reader = all_tasks[name]()
        rep = bench_dataset(
            model=model, tokenizer=tokenizer, reader=reader, comp_config=comp_config,
            dataset_name=name, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            max_prompt_len=args.max_prompt_len, device=device, dtype=dtype,
            bench_mode=args.bench_mode, warmup_batches=args.warmup_batches, limit=args.limit,
        )
        print(json.dumps(rep, ensure_ascii=False))
        reports.append(rep)
    if args.bench_report:
        with open(args.bench_report, 'w') as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        print(f"[bench] report written to {args.bench_report}")


if __name__ == '__main__':
    main()
