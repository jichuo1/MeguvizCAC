"""Train, QAT fine-tune, calibrate early-exit, and dual-graph INT8/Edge ONNX export MeguvizCAC v0.3-Loop."""

import argparse
import inspect
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .train_student import (
    CODE_LIST,
    PRAG_TAGS,
    build_train_records,
    compute_eval_metrics,
    find_val_thresholds,
    fit_temperature,
    load_pairs,
    load_qwen_fused,
    load_rescored_corpus,
    sha256_file,
)
from .v03_runtime import MeguvizV03Engine, quantize_embedding_w4_opset21
from .v03_tokenizer import (
    BASE_VOCAB_SIZE,
    BYTE_TOKENS,
    ByteFallbackWordPieceTokenizer,
    HEAD_CHARS,
    MAX_LEN,
    TAIL_CHARS,
    TOTAL_VOCAB_SIZE,
    prepare_text,
)


DEFAULT_BACKBONE = "BAAI/bge-small-zh-v1.5"
# Indices into PRAG_TAGS: surface [recap, contrast_thesis, assistant_leak, community_slang],
# deep [contextual_ellipsis, external_personal_exp, irony_parody, lyrical_emotion].
SURFACE_PRAG_IDXS = (0, 1, 2, 7)
DEEP_PRAG_IDXS = (3, 4, 5, 6)


def compute_weighted_svd_factors(
    word_weight: np.ndarray,
    train_counts: np.ndarray,
    embed_dim: int,
    unk_id: int,
    seed: int = 20261123,
) -> tuple[np.ndarray, np.ndarray]:
    """Factorize pretrained (V_base, H) via frequency+uniform weighted SVD and append +256 UTF-8 byte slots."""
    v_base, h_dim = word_weight.shape
    max_log = max(1e-8, float(np.log1p(train_counts[:v_base]).max()))
    weights = 0.35 + 0.65 * (np.log1p(train_counts[:v_base]) / max_log)
    sqrt_w = np.sqrt(weights).astype(np.float64)

    scaled = sqrt_w[:, None] * word_weight.astype(np.float64)
    u, s, vh = np.linalg.svd(scaled, full_matrices=False)
    r = min(int(embed_dim), len(s))
    sqrt_s = np.sqrt(np.maximum(s[:r], 0.0))

    a_base = ((u[:, :r] / sqrt_w[:, None]) * sqrt_s[None, :]).astype(np.float32)
    p_mat = (sqrt_s[:, None] * vh[:r, :]).astype(np.float32)

    if r < embed_dim:
        a_pad = np.zeros((v_base, embed_dim), dtype=np.float32)
        p_pad = np.zeros((embed_dim, h_dim), dtype=np.float32)
        a_pad[:, :r] = a_base
        p_pad[:r, :] = p_mat
        a_base, p_mat = a_pad, p_pad

    rng = np.random.default_rng(seed)
    a_std = float(np.std(a_base)) or 0.02
    byte_slots = np.broadcast_to(a_base[unk_id : unk_id + 1], (BYTE_TOKENS, embed_dim)).copy()
    byte_slots += rng.normal(0.0, 0.02 * a_std, size=byte_slots.shape).astype(np.float32)

    a_full = np.concatenate([a_base, byte_slots], axis=0).astype(np.float32)
    return a_full, p_mat


def _seed_ascii_bytes_into_a(a_full: np.ndarray, tokenizer: ByteFallbackWordPieceTokenizer, byte_offset: int) -> np.ndarray:
    out = a_full.copy()
    for b in range(0x21, 0x7F):
        ch = chr(b)
        if ch in tokenizer.vocab and tokenizer.vocab[ch] < byte_offset:
            out[byte_offset + b] = out[tokenizer.vocab[ch]]
        elif ch.lower() in tokenizer.vocab and tokenizer.vocab[ch.lower()] < byte_offset:
            out[byte_offset + b] = out[tokenizer.vocab[ch.lower()]]
    return out


def run_dry(data_path: Path, qwen_path: Path | None, pairs_path: Path | None, vocab_path: Path | None) -> None:
    """Verify dataset splits, zero leakage, and zero-[UNK] tokenization locally without requiring torch."""
    t0 = time.monotonic()
    rows = load_rescored_corpus(data_path)
    qwen_map = load_qwen_fused(qwen_path) if qwen_path else None
    forbidden = {prepare_text(r["text"]) for r in rows if r["split"] != "train"}
    pairs = load_pairs(pairs_path, forbidden) if pairs_path else []
    tr_5k = build_train_records(rows, synth_cap=5000, qwen_map=qwen_map)
    val_gold = [r for r in rows if r["split"] == "validation" and r["score_source"] == "hydrogen_xhigh_direct"]
    test_gold = [r for r in rows if r["split"] == "test" and r["score_source"] == "hydrogen_xhigh_direct"]
    probe = [r for r in rows if r["cohort"] == "user_confirmed"]

    tok_report: dict | str = "skipped_no_vocab"
    if vocab_path and vocab_path.is_file():
        tok = ByteFallbackWordPieceTokenizer(vocab_path)
        sample_rows = tr_5k[:5000]
        counts = np.zeros(BASE_VOCAB_SIZE, dtype=np.float64)
        byte_fallbacks = 0
        unks = 0
        lengths = []
        for item in sample_rows:
            ids, st = tok.encode_ids(item["text"], max_len=MAX_LEN, return_stats=True)
            byte_fallbacks += st["byte_fallback_tokens"]
            unks += st["unk_tokens"]
            lengths.append(st["seq_len"])
            for tid in ids:
                if tid < BASE_VOCAB_SIZE:
                    counts[tid] += 1.0
        extra_tests = [
            "(˘ω˘)ｽﾔｧ",
            "𠮷野家🫠",
            "asyncio.gather(*tasks)",
            "§①→②ℝⁿ",
        ]
        extra_unks = sum(tok.encode_ids(x, return_stats=True)[1]["unk_tokens"] for x in extra_tests)
        extra_bytes = sum(tok.encode_ids(x, return_stats=True)[1]["byte_fallback_tokens"] for x in extra_tests)
        lengths_np = np.array(lengths, dtype=np.int32)
        tok_report = {
            "vocab_size": tok.vocab_size,
            "sampled_train_rows": len(sample_rows),
            "unk_tokens": unks + extra_unks,
            "byte_fallback_tokens_in_5k": byte_fallbacks,
            "byte_fallback_tokens_in_extra_traps": extra_bytes,
            "distinct_base_tokens_seen": int((counts > 0).sum()),
            "seq_len_p50": int(np.percentile(lengths_np, 50)),
            "seq_len_p95": int(np.percentile(lengths_np, 95)),
        }

    print(
        json.dumps(
            {
                "dry_run": "PASS",
                "total_rows": len(rows),
                "train_items_synth5k": len(tr_5k),
                "val_gold": len(val_gold),
                "test_gold": len(test_gold),
                "probe_u001_u017": len(probe),
                "qwen_rows": len(qwen_map) if qwen_map else 0,
                "pairs_rows": len(pairs),
                "tokenizer_check": tok_report,
                "elapsed_s": round(time.monotonic() - t0, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _export_and_quantize_onnx(module, dummy_inputs: tuple, input_names: list[str], output_names: list[str], dynamic_axes: dict, int8_path: Path) -> None:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    fp32_path = int8_path.with_suffix(".fp32.tmp.onnx")
    export_kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    try:
        torch.onnx.export(module, dummy_inputs, str(fp32_path), **export_kwargs)
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    finally:
        fp32_path.unlink(missing_ok=True)


def train_and_export(args) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import Dataset, Sampler, DataLoader
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 进行 MeguvizCAC v0.3-Loop 训练。")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else (256 if vram_gb >= 35.0 else 128)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    # Save and load ByteFallbackWordPieceTokenizer directly from AutoTokenizer assets for zero train/serve skew.
    hf_tok = AutoTokenizer.from_pretrained(args.backbone)
    tok_tmp = out_dir / "_tok_tmp"
    hf_tok.save_pretrained(tok_tmp)
    for fn in ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src_f = tok_tmp / fn
        if src_f.is_file():
            shutil.copy2(src_f, out_dir / fn)
    if not (out_dir / "vocab.txt").is_file():
        vocab_items = sorted(hf_tok.get_vocab().items(), key=lambda kv: kv[1])
        (out_dir / "vocab.txt").write_text("\n".join(t for t, _ in vocab_items) + "\n", encoding="utf-8")
    shutil.rmtree(tok_tmp, ignore_errors=True)
    tokenizer = ByteFallbackWordPieceTokenizer(out_dir / "vocab.txt")

    # ------------------------------------------------------------------
    # Neural building blocks (defined inside train_and_export for lazy torch)
    # ------------------------------------------------------------------
    class QATEmbedding(nn.Module):
        def __init__(self, num_embeddings: int, embedding_dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
            self.qat_enabled = False

        def forward(self, input_ids):
            w = self.weight
            if self.qat_enabled:
                scale = w.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
                w_q = torch.clamp(torch.round(w / scale), -128.0, 127.0) * scale
                w = w + (w_q - w).detach()
            return F.embedding(input_ids, w)

    class QATLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
            self.qat_enabled = False
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        def forward(self, x):
            w = self.weight
            if self.qat_enabled:
                scale_w = w.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
                w_q = torch.clamp(torch.round(w / scale_w), -128.0, 127.0) * scale_w
                w = w + (w_q - w).detach()

                x_min = x.detach().amin().clamp(max=0.0)
                x_max = x.detach().amax().clamp(min=0.0)
                scale_x = ((x_max - x_min) / 255.0).clamp(min=1e-6)
                zp = torch.clamp(torch.round(-x_min / scale_x), 0.0, 255.0)
                x_q = (torch.clamp(torch.round(x / scale_x + zp), 0.0, 255.0) - zp) * scale_x
                x = x + (x_q - x).detach()
            return F.linear(x, w, self.bias)

    def copy_linear_(dst: QATLinear, src: nn.Linear) -> None:
        with torch.no_grad():
            dst.weight.copy_(src.weight)
            if dst.bias is not None and src.bias is not None:
                dst.bias.copy_(src.bias)

    def copy_ln_(dst: nn.LayerNorm, src: nn.LayerNorm) -> None:
        with torch.no_grad():
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)

    class WeightedSVDEmbedding(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, max_len: int, eps: float):
            super().__init__()
            self.word_emb_A = QATEmbedding(vocab_size, embed_dim)
            self.word_emb_P = QATLinear(embed_dim, hidden_size, bias=False)
            self.position_weight = nn.Parameter(torch.empty(max_len, hidden_size))
            self.token_type_bias = nn.Parameter(torch.zeros(hidden_size))
            self.ln_weight = nn.Parameter(torch.ones(hidden_size))
            self.ln_bias = nn.Parameter(torch.zeros(hidden_size))
            self.eps = float(eps)
            self.register_buffer("position_ids", torch.arange(max_len, dtype=torch.long).unsqueeze(0))

        def load_from_bert(self, bert_emb, a_full: np.ndarray, p_mat: np.ndarray) -> None:
            with torch.no_grad():
                self.word_emb_A.weight.copy_(torch.from_numpy(a_full))
                self.word_emb_P.weight.copy_(torch.from_numpy(p_mat.T))
                self.position_weight.copy_(bert_emb.position_embeddings.weight[:MAX_LEN])
                self.token_type_bias.copy_(bert_emb.token_type_embeddings.weight[0])
                self.ln_weight.copy_(bert_emb.LayerNorm.weight)
                self.ln_bias.copy_(bert_emb.LayerNorm.bias)

        def forward(self, input_ids):
            seq_len = input_ids.shape[1]
            word_vec = self.word_emb_P(self.word_emb_A(input_ids))
            pos_ids = self.position_ids[:, :seq_len]
            pos_vec = F.embedding(pos_ids, self.position_weight)
            x = word_vec + pos_vec + self.token_type_bias
            mean = x.mean(dim=-1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
            return (x - mean) * torch.rsqrt(var + self.eps) * self.ln_weight + self.ln_bias

    class StaticEncoderBlock(nn.Module):
        """Untied Prelude Transformer block matching BertLayer bit-for-bit."""

        def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, eps: float, drop_p: float = 0.1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.scale = 1.0 / math.sqrt(self.head_dim)
            self.eps = float(eps)

            self.q_proj = QATLinear(hidden_size, hidden_size)
            self.k_proj = QATLinear(hidden_size, hidden_size)
            self.v_proj = QATLinear(hidden_size, hidden_size)
            self.o_proj = QATLinear(hidden_size, hidden_size)
            self.attn_ln = nn.LayerNorm(hidden_size, eps=eps)
            self.ffn_up = QATLinear(hidden_size, intermediate_size)
            self.ffn_down = QATLinear(intermediate_size, hidden_size)
            self.ffn_ln = nn.LayerNorm(hidden_size, eps=eps)
            self.drop = nn.Dropout(drop_p)

        def load_from_bert_layer(self, layer) -> None:
            copy_linear_(self.q_proj, layer.attention.self.query)
            copy_linear_(self.k_proj, layer.attention.self.key)
            copy_linear_(self.v_proj, layer.attention.self.value)
            copy_linear_(self.o_proj, layer.attention.output.dense)
            copy_ln_(self.attn_ln, layer.attention.output.LayerNorm)
            copy_linear_(self.ffn_up, layer.intermediate.dense)
            copy_linear_(self.ffn_down, layer.output.dense)
            copy_ln_(self.ffn_ln, layer.output.LayerNorm)

        def forward(self, x, attn_bias_4d):
            bsz, seqlen, _ = x.shape
            q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale + attn_bias_4d
            probs = self.drop(torch.softmax(scores, dim=-1))
            ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)
            h = self.attn_ln(x + self.drop(self.o_proj(ctx)))
            ff = self.ffn_down(F.gelu(self.ffn_up(h)))
            return self.ffn_ln(h + self.drop(ff))

    class QKNormRMS(nn.Module):
        def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, num_heads, 1, head_dim))
            self.eps = eps

        def forward(self, x):
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x * inv_rms * self.weight

    class StepIndexedLoRALinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, num_steps: int, lora_rank: int):
            super().__init__()
            self.base = QATLinear(in_features, out_features)
            self.lora_A = nn.Parameter(torch.zeros(num_steps, in_features, lora_rank))
            self.lora_B = nn.Parameter(torch.zeros(num_steps, lora_rank, out_features))

        def load_base_from(self, src: nn.Linear) -> None:
            copy_linear_(self.base, src)

        def init_lora_from_target(self, step_idx: int, target: nn.Linear, scale: float = 1.0) -> None:
            with torch.no_grad():
                delta = (target.weight.detach().float() - self.base.weight.detach().float()) * float(scale)
                u, s, vh = torch.linalg.svd(delta, full_matrices=False)
                r = min(self.lora_A.shape[2], len(s))
                sqrt_s = torch.sqrt(s[:r].clamp(min=0.0))
                self.lora_A[step_idx, :, :r].copy_(vh[:r, :].T * sqrt_s.unsqueeze(0))
                self.lora_B[step_idx, :r, :].copy_(sqrt_s.unsqueeze(1) * u[:, :r].T)

        def init_lora_normal(self, step_idx: int, std: float = 0.02) -> None:
            with torch.no_grad():
                nn.init.normal_(self.lora_A[step_idx], mean=0.0, std=std)
                self.lora_B[step_idx].zero_()

        def forward(self, x, step_id):
            out = self.base(x)
            a_t = self.lora_A[step_id]
            b_t = self.lora_B[step_id]
            return out + torch.matmul(torch.matmul(x, a_t), b_t)

    class StepIndexedLayerNorm(nn.Module):
        def __init__(self, num_steps: int, hidden_size: int, eps: float):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(num_steps, hidden_size))
            self.bias = nn.Parameter(torch.zeros(num_steps, hidden_size))
            self.eps = float(eps)

        def init_all_from(self, src: nn.LayerNorm) -> None:
            with torch.no_grad():
                for s in range(self.weight.shape[0]):
                    self.weight[s].copy_(src.weight)
                    self.bias[s].copy_(src.bias)

        def init_step_from(self, step_idx: int, src: nn.LayerNorm) -> None:
            with torch.no_grad():
                self.weight[step_idx].copy_(src.weight)
                self.bias[step_idx].copy_(src.bias)

        def forward(self, x, step_id):
            mean = x.mean(dim=-1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
            w_t = self.weight[step_id]
            b_t = self.bias[step_id]
            return (x - mean) * torch.rsqrt(var + self.eps) * w_t + b_t

    class CoreStepBlock(nn.Module):
        """Recurrent Transformer block with step-indexed LoRA, QK-Norm, and step-indexed LayerNorm."""

        def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, num_steps: int, lora_rank: int, eps: float, drop_p: float = 0.1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.scale = 1.0 / math.sqrt(self.head_dim)

            self.q_proj = StepIndexedLoRALinear(hidden_size, hidden_size, num_steps, lora_rank)
            self.k_proj = StepIndexedLoRALinear(hidden_size, hidden_size, num_steps, lora_rank)
            self.v_proj = StepIndexedLoRALinear(hidden_size, hidden_size, num_steps, lora_rank)
            self.o_proj = StepIndexedLoRALinear(hidden_size, hidden_size, num_steps, lora_rank)
            self.q_norm = QKNormRMS(num_heads, self.head_dim)
            self.k_norm = QKNormRMS(num_heads, self.head_dim)
            self.attn_ln = StepIndexedLayerNorm(num_steps, hidden_size, eps)
            self.ffn_up = StepIndexedLoRALinear(hidden_size, intermediate_size, num_steps, lora_rank)
            self.ffn_down = StepIndexedLoRALinear(intermediate_size, hidden_size, num_steps, lora_rank)
            self.ffn_ln = StepIndexedLayerNorm(num_steps, hidden_size, eps)
            self.drop = nn.Dropout(drop_p)

        def load_base_from_bert_layer(self, layer) -> None:
            self.q_proj.load_base_from(layer.attention.self.query)
            self.k_proj.load_base_from(layer.attention.self.key)
            self.v_proj.load_base_from(layer.attention.self.value)
            self.o_proj.load_base_from(layer.attention.output.dense)
            self.attn_ln.init_all_from(layer.attention.output.LayerNorm)
            self.ffn_up.load_base_from(layer.intermediate.dense)
            self.ffn_down.load_base_from(layer.output.dense)
            self.ffn_ln.init_all_from(layer.output.LayerNorm)

        def init_step_lora_from_layer(self, step_idx: int, layer, scale: float = 1.0) -> None:
            self.q_proj.init_lora_from_target(step_idx, layer.attention.self.query, scale=scale)
            self.k_proj.init_lora_from_target(step_idx, layer.attention.self.key, scale=scale)
            self.v_proj.init_lora_from_target(step_idx, layer.attention.self.value, scale=scale)
            self.o_proj.init_lora_from_target(step_idx, layer.attention.output.dense, scale=scale)
            self.ffn_up.init_lora_from_target(step_idx, layer.intermediate.dense, scale=scale)
            self.ffn_down.init_lora_from_target(step_idx, layer.output.dense, scale=scale)
            self.attn_ln.init_step_from(step_idx, layer.attention.output.LayerNorm)
            self.ffn_ln.init_step_from(step_idx, layer.output.LayerNorm)

        def init_step_lora_normal(self, step_idx: int, std: float = 0.02) -> None:
            for mod in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.ffn_up, self.ffn_down):
                mod.init_lora_normal(step_idx, std=std)

        def calibrate_qk_rms(self, x_sample) -> None:
            with torch.no_grad():
                bsz, seqlen, _ = x_sample.shape
                step0 = torch.tensor(0, dtype=torch.long, device=x_sample.device)
                q = self.q_proj(x_sample, step0).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(x_sample, step0).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
                q_rms = torch.sqrt(q.pow(2).mean(dim=(0, 2), keepdim=True).mean(dim=-1, keepdim=True).clamp(min=1e-6))
                k_rms = torch.sqrt(k.pow(2).mean(dim=(0, 2), keepdim=True).mean(dim=-1, keepdim=True).clamp(min=1e-6))
                self.q_norm.weight.copy_(q_rms.expand_as(self.q_norm.weight))
                self.k_norm.weight.copy_(k_rms.expand_as(self.k_norm.weight))

        def forward(self, x, attn_bias_4d, step_id):
            bsz, seqlen, _ = x.shape
            q = self.q_norm(self.q_proj(x, step_id).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2))
            k = self.k_norm(self.k_proj(x, step_id).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2))
            v = self.v_proj(x, step_id).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale + attn_bias_4d
            probs = self.drop(torch.softmax(scores, dim=-1))
            ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)
            h = self.attn_ln(x + self.drop(self.o_proj(ctx, step_id)), step_id)
            ff = self.ffn_down(F.gelu(self.ffn_up(h, step_id)), step_id)
            return self.ffn_ln(h + self.drop(ff), step_id)

    class MeguvizV03LoopModel(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            core_layers: int,
            num_loops: int,
            lora_rank: int,
            eps: float,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.core_layers = core_layers
            self.num_loops = num_loops

            self.embeddings = WeightedSVDEmbedding(vocab_size, embed_dim, hidden_size, MAX_LEN, eps)
            self.emb_drop = nn.Dropout(0.1)
            self.prelude = StaticEncoderBlock(hidden_size, num_heads, intermediate_size, eps)

            # Gated Input Injection: u_t = (1 - g_t) * z_prev + g_t * LN(h0, step_id)
            self.inject_gate = nn.Parameter(torch.full((num_loops, hidden_size), -2.0))
            with torch.no_grad():
                self.inject_gate[0].fill_(-6.0)
            self.inject_norm = StepIndexedLayerNorm(num_loops, hidden_size, eps)

            self.core_blocks = nn.ModuleList(
                [
                    CoreStepBlock(hidden_size, num_heads, intermediate_size, num_loops, lora_rank, eps)
                    for _ in range(core_layers)
                ]
            )

            # Coda readout & Ouro exit gate (code_head & prag_head pruned on ONNX export)
            self.coda_proj = QATLinear(hidden_size * 2, 256)
            self.coda_ln = nn.LayerNorm(256, eps=eps)
            self.coda_drop = nn.Dropout(0.1)
            self.score_head = QATLinear(256, 1)
            self.code_head = nn.Linear(256, len(CODE_LIST))
            self.prag_head = nn.Linear(256, len(PRAG_TAGS))

            self.exit_feat_proj = nn.Linear(256, 32)
            self.exit_logit_proj = nn.Linear(1, 32, bias=False)
            self.exit_step_bias = nn.Parameter(torch.zeros(num_loops, 32))
            self.exit_out = nn.Linear(32, 1)
            with torch.no_grad():
                self.exit_out.bias.fill_(-1.0)

            self.register_buffer("step_temperatures", torch.ones(num_loops, dtype=torch.float32))
            self.register_buffer("step_idxs", torch.arange(num_loops, dtype=torch.long))

        def set_qat_enabled(self, enabled: bool) -> None:
            for mod in self.modules():
                if isinstance(mod, (QATLinear, QATEmbedding)):
                    mod.qat_enabled = bool(enabled)

        def forward_prelude(self, input_ids, attention_mask):
            attn_bias = (1.0 - attention_mask[:, None, None, :].to(self.prelude.q_proj.weight.dtype)) * (-10000.0)
            emb = self.emb_drop(self.embeddings(input_ids))
            return self.prelude(emb, attn_bias)

        def forward_core_step(self, z_prev, h0, attention_mask, step_id, apply_aux: bool = True):
            g_t = torch.sigmoid(self.inject_gate[step_id])
            u_t = (1.0 - g_t) * z_prev + g_t * self.inject_norm(h0, step_id)
            attn_bias = (1.0 - attention_mask[:, None, None, :].to(u_t.dtype)) * (-10000.0)

            z = u_t
            for blk in self.core_blocks:
                z = blk(z, attn_bias, step_id)

            mask_f = attention_mask.to(z.dtype).unsqueeze(-1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            mean_pool = (z * mask_f).sum(dim=1) / denom
            cls_vec = z[:, 0, :]
            pooled = torch.cat([cls_vec, mean_pool], dim=-1)
            feat = self.coda_drop(F.gelu(self.coda_ln(self.coda_proj(pooled))))

            raw_logit = self.score_head(feat).reshape(-1)
            temp_t = self.step_temperatures[step_id]
            calib_logit = raw_logit / temp_t
            prob = torch.sigmoid(calib_logit)

            exit_hidden = F.gelu(
                self.exit_feat_proj(feat)
                + self.exit_logit_proj(calib_logit.abs().unsqueeze(-1))
                + self.exit_step_bias[step_id]
            )
            halt_prob = torch.sigmoid(self.exit_out(exit_hidden)).reshape(-1)

            if apply_aux:
                return z, prob, calib_logit, halt_prob, self.code_head(feat), self.prag_head(feat)
            return z, prob, calib_logit, halt_prob

    class PreludeExportWrapper(nn.Module):
        def __init__(self, loop_model: MeguvizV03LoopModel):
            super().__init__()
            self.m = loop_model

        def forward(self, input_ids, attention_mask):
            return self.m.forward_prelude(input_ids, attention_mask)

    class CoreStepExportWrapper(nn.Module):
        def __init__(self, loop_model: MeguvizV03LoopModel):
            super().__init__()
            self.m = loop_model

        def forward(self, z_prev, h0, attention_mask, step_id):
            return self.m.forward_core_step(z_prev, h0, attention_mask, step_id, apply_aux=False)

    # ------------------------------------------------------------------
    # Data prep & pre-tokenization (zero val/test/probe leakage)
    # ------------------------------------------------------------------
    rows = load_rescored_corpus(args.data.resolve())
    qwen_map = load_qwen_fused(args.qwen_data.resolve()) if args.qwen_data else None
    forbidden = {prepare_text(r["text"]) for r in rows if r["split"] != "train"}
    pairs = load_pairs(args.pairs_data.resolve(), forbidden) if args.pairs_data else []

    train_items_raw = build_train_records(rows, synth_cap=args.synth_cap, qwen_map=qwen_map)
    if args.smoke_test:
        rng_smoke = random.Random(20261125)
        train_items_raw = rng_smoke.sample(train_items_raw, min(400, len(train_items_raw)))
        pairs = pairs[:32]

    # Pre-tokenize train rows once AND gather strictly train-split token frequencies for Weighted SVD
    train_counts = np.zeros(BASE_VOCAB_SIZE, dtype=np.float64)
    for item in train_items_raw:
        ids = tokenizer.encode_ids(item["text"], max_len=MAX_LEN)
        item["ids"] = ids
        item["length"] = len(ids)
        for tid in ids:
            if tid < BASE_VOCAB_SIZE:
                train_counts[tid] += 1.0

    # Initialize backbone + Weighted SVD + RRT LoRA + empirical QK-Norm
    bert_src = AutoModel.from_pretrained(args.backbone)
    cfg = bert_src.config
    a_full, p_mat = compute_weighted_svd_factors(
        bert_src.embeddings.word_embeddings.weight.detach().cpu().numpy(),
        train_counts=train_counts,
        embed_dim=args.embed_dim,
        unk_id=tokenizer.unk_id,
    )
    a_full = _seed_ascii_bytes_into_a(a_full, tokenizer, tokenizer.byte_offset)

    def make_fresh_model() -> MeguvizV03LoopModel:
        mdl = MeguvizV03LoopModel(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            intermediate_size=cfg.intermediate_size,
            core_layers=args.core_layers,
            num_loops=args.num_loops,
            lora_rank=args.lora_rank,
            eps=cfg.layer_norm_eps,
        )
        mdl.embeddings.load_from_bert(bert_src.embeddings, a_full, p_mat)
        mdl.prelude.load_from_bert_layer(bert_src.encoder.layer[0])

        src_layers = bert_src.encoder.layer
        if args.core_layers == 2:
            mdl.core_blocks[0].load_base_from_bert_layer(src_layers[1])
            mdl.core_blocks[1].load_base_from_bert_layer(src_layers[2])
            if args.num_loops >= 2:
                mdl.core_blocks[0].init_step_lora_from_layer(1, src_layers[2], scale=0.25)
                mdl.core_blocks[1].init_step_lora_from_layer(1, src_layers[3], scale=1.0)
            for s in range(2, args.num_loops):
                mdl.core_blocks[0].init_step_lora_normal(s, std=0.02)
                mdl.core_blocks[1].init_step_lora_normal(s, std=0.02)
        elif args.core_layers == 1:
            mdl.core_blocks[0].load_base_from_bert_layer(src_layers[1])
            if args.num_loops >= 2:
                mdl.core_blocks[0].init_step_lora_from_layer(1, src_layers[2], scale=1.0)
            if args.num_loops >= 3:
                mdl.core_blocks[0].init_step_lora_from_layer(2, src_layers[3], scale=1.0)
            for s in range(3, args.num_loops):
                mdl.core_blocks[0].init_step_lora_normal(s, std=0.02)
        else:
            raise ValueError(f"--core-layers must be 1 or 2, got {args.core_layers}")

        mdl = mdl.to(device)
        # Measure empirical per-head Q/K RMS over a 32-comment real sample for zero step-0 shock
        sample_enc = [x["ids"] for x in train_items_raw[:32]]
        max_s = max(len(x) for x in sample_enc)
        s_ids = torch.full((len(sample_enc), max_s), tokenizer.pad_id, dtype=torch.long, device=device)
        s_mask = torch.zeros((len(sample_enc), max_s), dtype=torch.long, device=device)
        for i, seq in enumerate(sample_enc):
            s_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            s_mask[i, : len(seq)] = 1
        mdl.eval()
        with torch.no_grad():
            h_cal = mdl.forward_prelude(s_ids, s_mask)
            attn_b = (1.0 - s_mask[:, None, None, :].to(h_cal.dtype)) * (-10000.0)
            step0 = torch.tensor(0, dtype=torch.long, device=device)
            for blk in mdl.core_blocks:
                blk.calibrate_qk_rms(h_cal)
                h_cal = blk(h_cal, attn_b, step0)
        return mdl

    class PreTokenizedDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    class LengthBucketBatchSampler(Sampler):
        def __init__(self, lengths: list[int], bs: int, seed: int = 20261123):
            self.lengths = lengths
            self.bs = bs
            self.seed = seed
            self.epoch = 0

        def __iter__(self):
            rng = random.Random(self.seed + self.epoch)
            self.epoch += 1
            indices = list(range(len(self.lengths)))
            rng.shuffle(indices)
            batches = []
            mega = max(self.bs * 16, 2048)
            for start in range(0, len(indices), mega):
                chunk = indices[start : start + mega]
                chunk.sort(key=lambda i: self.lengths[i])
                for b_start in range(0, len(chunk), self.bs):
                    batches.append(chunk[b_start : b_start + self.bs])
            rng.shuffle(batches)
            for b in batches:
                yield b

        def __len__(self):
            return (len(self.lengths) + self.bs - 1) // self.bs

    def collate_tokenized(batch):
        max_seq = max(x["length"] for x in batch)
        ids = torch.full((len(batch), max_seq), tokenizer.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), max_seq), dtype=torch.long)
        for i, x in enumerate(batch):
            n = x["length"]
            ids[i, :n] = torch.tensor(x["ids"], dtype=torch.long)
            mask[i, :n] = 1
        return (
            ids,
            mask,
            torch.tensor([x["y_score"] for x in batch], dtype=torch.float32),
            torch.tensor([x["code_idx"] for x in batch], dtype=torch.long),
            torch.tensor([x["weight"] for x in batch], dtype=torch.float32),
            torch.tensor([x["prag_vec"] for x in batch], dtype=torch.float32),
        )

    def collate_texts(texts: list[str]):
        ids_np, mask_np, _ = tokenizer.encode_batch(texts, max_len=MAX_LEN)
        return torch.from_numpy(ids_np), torch.from_numpy(mask_np)

    def pre_encode_eval(subset: list[dict]) -> tuple[list[list[int]], np.ndarray]:
        enc = [tokenizer.encode_ids(r["text"], max_len=MAX_LEN) for r in subset]
        scores = np.array([float(r["score"]) for r in subset], dtype=np.float32)
        return enc, scores

    def predict_all_steps_pt(mdl: MeguvizV03LoopModel, encoded_ids: list[list[int]], bs: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Return (calibrated_logits[T, N], halt_probs[T, N]) across all loop steps on GPU."""
        mdl.eval()
        num_t = mdl.num_loops
        n_total = len(encoded_ids)
        logits_out = np.zeros((num_t, n_total), dtype=np.float32)
        halts_out = np.zeros((num_t, n_total), dtype=np.float32)
        if n_total == 0:
            return logits_out, halts_out
        with torch.no_grad():
            for start in range(0, n_total, bs):
                chunk = encoded_ids[start : start + bs]
                max_seq = max(len(s) for s in chunk)
                ids = torch.full((len(chunk), max_seq), tokenizer.pad_id, dtype=torch.long, device=device)
                mask = torch.zeros((len(chunk), max_seq), dtype=torch.long, device=device)
                for i, seq in enumerate(chunk):
                    ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                    mask[i, : len(seq)] = 1
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    h0 = mdl.forward_prelude(ids, mask)
                    z = h0
                    for s in range(num_t):
                        z, _, c_logit, h_prob, _, _ = mdl.forward_core_step(z, h0, mask, mdl.step_idxs[s], apply_aux=True)
                        logits_out[s, start : start + len(chunk)] = c_logit.float().cpu().numpy()
                        halts_out[s, start : start + len(chunk)] = h_prob.float().cpu().numpy()
        return logits_out, halts_out

    def apply_exit_policy(probs_steps: np.ndarray, halts_steps: np.ndarray, max_loops: int, policy: dict | None) -> tuple[np.ndarray, np.ndarray]:
        n = probs_steps.shape[1]
        max_loops = max(1, min(probs_steps.shape[0], max_loops))
        out_probs = np.empty(n, dtype=np.float32)
        steps_used = np.empty(n, dtype=np.int32)
        if policy is None or max_loops == 1:
            out_probs[:] = probs_steps[max_loops - 1]
            steps_used[:] = max_loops
            return out_probs, steps_used

        e_low = float(policy["exit_low"])
        e_high = float(policy["exit_high"])
        h_thr = float(policy["halt_threshold"])
        undecided = np.ones(n, dtype=bool)
        surv = np.ones(n, dtype=np.float32)
        for s in range(max_loops):
            p = probs_steps[s]
            surv = surv * (1.0 - np.clip(halts_steps[s], 0.0, 1.0))
            cum_h = 1.0 - surv
            if s < max_loops - 1:
                take = undecided & ((p <= e_low) | (p >= e_high) | (cum_h >= h_thr))
            else:
                take = undecided
            out_probs[take] = p[take]
            steps_used[take] = s + 1
            undecided &= ~take
            if not undecided.any():
                break
        return out_probs, steps_used

    # ------------------------------------------------------------------
    # Prepare splits & train
    # ------------------------------------------------------------------
    val_gold = [r for r in rows if r["split"] == "validation" and r["score_source"] == "hydrogen_xhigh_direct"]
    val_dsh_gold = [r for r in val_gold if r["dsh"]]
    test_gold = [r for r in rows if r["split"] == "test" and r["score_source"] == "hydrogen_xhigh_direct"]
    test_dsh_gold = [r for r in test_gold if r["dsh"]]
    test_bili_all = [r for r in rows if r["cohort"] == "bili" and r["split"] == "test"]
    test_dsh_all = [r for r in test_bili_all if r["dsh"]]
    probe_rows = [r for r in rows if r["cohort"] == "user_confirmed"]

    val_enc, val_scores = pre_encode_eval(val_gold)
    val_dsh_enc, val_dsh_scores = pre_encode_eval(val_dsh_gold)
    test_enc, test_scores = pre_encode_eval(test_gold)
    test_dsh_enc, test_dsh_scores = pre_encode_eval(test_dsh_gold)
    test_all_enc, test_all_scores = pre_encode_eval(test_bili_all)
    test_dsh_all_enc, test_dsh_all_scores = pre_encode_eval(test_dsh_all)
    probe_enc, probe_scores = pre_encode_eval(probe_rows)

    mdl = make_fresh_model()
    del bert_src
    torch.cuda.empty_cache()

    dataset = PreTokenizedDataset(train_items_raw)
    sampler = LengthBucketBatchSampler([x["length"] for x in train_items_raw], bs=batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_tokenized, num_workers=0, pin_memory=True)

    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    ce_code = nn.CrossEntropyLoss(ignore_index=-100)
    margin_fn = nn.MarginRankingLoss(margin=0.25)

    # Pre-build progressive pragmatic tag weights per step [T, 8]
    prag_step_weights = []
    for s in range(args.num_loops):
        alpha = s / max(1, args.num_loops - 1)
        w_vec = np.ones(len(PRAG_TAGS), dtype=np.float32)
        for idx in SURFACE_PRAG_IDXS:
            w_vec[idx] = 1.3 - 0.6 * alpha
        for idx in DEEP_PRAG_IDXS:
            w_vec[idx] = 0.5 + 1.0 * alpha
        prag_step_weights.append(torch.tensor(w_vec, dtype=torch.float32, device=device))

    def run_training_phase(num_epochs: int, lr: float, qat: bool, phase_name: str, max_steps_cap: int = 0) -> None:
        mdl.set_qat_enabled(qat)
        opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
        total_steps = min(max_steps_cap, len(loader) * num_epochs) if max_steps_cap > 0 else max(1, len(loader) * num_epochs)
        sched = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * 0.08)), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        rng = random.Random(20261124 + (99 if qat else 0))
        pair_bs = 12
        log_every = max(50, len(loader) // 4)
        t_p = time.monotonic()
        global_step = 0

        for ep in range(num_epochs):
            mdl.train()
            running_loss = 0.0
            n_seen = 0
            for step, (ids, mask, y_sc, y_cd, w, prag) in enumerate(loader):
                ids, mask = ids.to(device), mask.to(device)
                y_sc, y_cd, w, prag = y_sc.to(device), y_cd.to(device), w.to(device), prag.to(device)
                opt.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    h0 = mdl.forward_prelude(ids, mask)
                    z = h0
                    surv = torch.ones_like(y_sc)
                    step_q_list = []
                    step_sc_losses = []
                    loss_cd_total = 0.0
                    loss_pr_total = 0.0
                    prag_mask = prag >= 0.0

                    for s in range(args.num_loops):
                        z, _, sc_logit, halt_p, cd_logit, pr_logit = mdl.forward_core_step(
                            z, h0, mask, mdl.step_idxs[s], apply_aux=True
                        )
                        q_s = surv * halt_p if s < args.num_loops - 1 else surv
                        surv = surv * (1.0 - halt_p)
                        step_q_list.append(q_s)

                        per_sample_sc = bce_none(sc_logit, y_sc) * w
                        step_sc_losses.append(per_sample_sc)
                        loss_cd_total = loss_cd_total + ce_code(cd_logit, y_cd) * (0.7 if s == 0 else 1.1)

                        if prag_mask.any():
                            pr_mat = bce_none(pr_logit, prag.clamp(min=0.0)) * prag_step_weights[s].unsqueeze(0)
                            loss_pr_total = loss_pr_total + pr_mat[prag_mask].mean()

                    q_mat = torch.stack(step_q_list, dim=-1)  # (B, T)
                    sc_mat = torch.stack(step_sc_losses, dim=-1)  # (B, T)
                    loss_ouro = (q_mat * sc_mat).sum(dim=-1).mean()
                    loss_uniform = sc_mat.mean()
                    prior = 1.0 / args.num_loops
                    kl_prior = (q_mat * (torch.log(q_mat.clamp(min=1e-6)) - math.log(prior))).sum(dim=-1).mean()

                    loss = 0.5 * loss_ouro + 0.5 * loss_uniform + 0.05 * kl_prior
                    loss = loss + 0.15 * (loss_cd_total / args.num_loops)
                    if prag_mask.any():
                        loss = loss + 0.25 * (loss_pr_total / args.num_loops)

                    if pairs:
                        p_batch = rng.sample(pairs, min(pair_bs, len(pairs)))
                        ai_ids, ai_mask = collate_texts([p["ai_text"] for p in p_batch])
                        hu_ids, hu_mask = collate_texts([p["human_text"] for p in p_batch])
                        ai_ids, ai_mask = ai_ids.to(device), ai_mask.to(device)
                        hu_ids, hu_mask = hu_ids.to(device), hu_mask.to(device)

                        h0_ai = mdl.forward_prelude(ai_ids, ai_mask)
                        h0_hu = mdl.forward_prelude(hu_ids, hu_mask)
                        z_ai, z_hu = h0_ai, h0_hu
                        pair_loss_acc = 0.0
                        pair_w_sum = 0.0
                        for s in range(args.num_loops):
                            z_ai, _, l_ai, _, _, _ = mdl.forward_core_step(z_ai, h0_ai, ai_mask, mdl.step_idxs[s], apply_aux=True)
                            z_hu, _, l_hu, _, _, _ = mdl.forward_core_step(z_hu, h0_hu, hu_mask, mdl.step_idxs[s], apply_aux=True)
                            sw = 0.25 if s == 0 else 1.0
                            p_ai = torch.sigmoid(l_ai)
                            p_hu = torch.sigmoid(l_hu)
                            l_margin = margin_fn(p_ai, p_hu, torch.ones_like(p_ai))
                            l_end = 0.5 * (
                                bce_none(l_ai, torch.full_like(l_ai, 0.85)).mean()
                                + bce_none(l_hu, torch.full_like(l_hu, 0.15)).mean()
                            )
                            pair_loss_acc = pair_loss_acc + sw * (l_margin + 0.5 * l_end)
                            pair_w_sum += sw
                        loss = loss + 0.35 * (pair_loss_acc / max(1e-6, pair_w_sum))

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

                running_loss += float(loss.detach())
                n_seen += 1
                global_step += 1
                if (step + 1) % log_every == 0:
                    print(f"[{phase_name} ep={ep+1}/{num_epochs} step={step+1}/{len(loader)}] loss={running_loss/n_seen:.4f} elapsed={time.monotonic()-t_p:.1f}s", flush=True)
                if max_steps_cap > 0 and global_step >= max_steps_cap:
                    break

            v_logits, _ = predict_all_steps_pt(mdl, val_enc)
            val_last_probs = expit(v_logits[min(args.num_loops - 1, 2)])
            m_ep = compute_eval_metrics(val_scores, val_last_probs)
            print(f"[{phase_name} ep={ep+1} DONE] val_normal_last_step: {json.dumps(m_ep)}", flush=True)
            if max_steps_cap > 0 and global_step >= max_steps_cap:
                break

    run_training_phase(
        num_epochs=1 if args.smoke_test else args.epochs,
        lr=args.lr,
        qat=False,
        phase_name="FP32-Train",
    )
    if args.qat_epochs > 0:
        run_training_phase(
            num_epochs=1 if args.smoke_test else args.qat_epochs,
            lr=args.qat_lr,
            qat=True,
            phase_name="INT8-QAT",
            max_steps_cap=10 if args.smoke_test else 0,
        )

    # Disable QAT fake-quant before calibration and clean FP32 -> Dynamic INT8 ONNX export
    mdl.set_qat_enabled(False)
    mdl.eval()

    # Fit per-step temperatures on val_bili_gold
    with torch.no_grad():
        mdl.step_temperatures.fill_(1.0)
    raw_val_logits, val_halts = predict_all_steps_pt(mdl, val_enc)
    fitted_temps = []
    for s in range(args.num_loops):
        t_val = fit_temperature(raw_val_logits[s], val_scores / 100.0)
        fitted_temps.append(round(float(t_val), 6))
        with torch.no_grad():
            mdl.step_temperatures[s] = float(t_val)
    print(f"[Calibration] Fitted step_temperatures = {fitted_temps}", flush=True)

    # Re-evaluate calibrated val predictions & sweep early-exit policy for normal mode
    cal_val_logits, val_halts = predict_all_steps_pt(mdl, val_enc)
    val_probs_steps = expit(cal_val_logits)
    normal_max = max(1, min(args.num_loops, args.normal_max_loops))
    audit_max = max(normal_max, args.num_loops)

    val_bin = (val_scores >= 70.0).astype(bool)
    base_norm_auroc = float(roc_auc_score(val_bin, val_probs_steps[normal_max - 1]))
    best_policy = {"exit_low": 0.25, "exit_high": 0.85, "halt_threshold": 0.85}
    best_avg_steps = float(normal_max) + 1.0
    fallback_policy = best_policy
    fallback_auc = -1.0

    for e_low in (0.15, 0.20, 0.25, 0.30):
        for e_high in (0.80, 0.85, 0.90):
            for h_thr in (0.70, 0.80, 0.90, 0.99):
                cand = {"exit_low": e_low, "exit_high": e_high, "halt_threshold": h_thr}
                p_cand, st_cand = apply_exit_policy(val_probs_steps, val_halts, normal_max, cand)
                auc_cand = float(roc_auc_score(val_bin, p_cand))
                avg_s = float(st_cand.mean())
                if auc_cand > fallback_auc:
                    fallback_auc = auc_cand
                    fallback_policy = cand
                if auc_cand >= base_norm_auroc - 0.0010 and avg_s < best_avg_steps:
                    best_avg_steps = avg_s
                    best_policy = cand
    if best_avg_steps > normal_max:
        best_policy = fallback_policy
    print(f"[Calibration] Selected normal exit_policy={best_policy} (base_norm_auroc={base_norm_auroc:.4f})", flush=True)

    # Calibrate f05 / fpr05 thresholds on val_dsh_gold under selected normal adaptive policy
    val_dsh_logits, val_dsh_halts = predict_all_steps_pt(mdl, val_dsh_enc)
    val_dsh_norm_p, _ = apply_exit_policy(expit(val_dsh_logits), val_dsh_halts, normal_max, best_policy)
    t_f05, t_fpr05 = find_val_thresholds(val_dsh_scores, val_dsh_norm_p)

    # Collect comprehensive PyTorch metrics across per-step, normal adaptive, and audit adaptive
    def summarize_split(enc: list[list[int]], scores: np.ndarray) -> tuple[dict, dict, dict, np.ndarray]:
        l_steps, h_steps = predict_all_steps_pt(mdl, enc)
        p_steps = expit(l_steps)
        per_step = {
            f"step_{s}": compute_eval_metrics(scores, p_steps[s], t_f05, t_fpr05)
            for s in range(args.num_loops)
        }
        p_norm, st_norm = apply_exit_policy(p_steps, h_steps, normal_max, best_policy)
        m_norm = {
            **compute_eval_metrics(scores, p_norm, t_f05, t_fpr05),
            "avg_steps": round(float(st_norm.mean()), 3),
            "exit_hist": dict(sorted((str(k), int(v)) for k, v in Counter(st_norm.tolist()).items())),
        }
        p_aud, st_aud = apply_exit_policy(p_steps, h_steps, audit_max, best_policy)
        m_aud = {
            **compute_eval_metrics(scores, p_aud, t_f05, t_fpr05),
            "avg_steps": round(float(st_aud.mean()), 3),
            "exit_hist": dict(sorted((str(k), int(v)) for k, v in Counter(st_aud.tolist()).items())),
        }
        return per_step, m_norm, m_aud, p_norm

    _, val_norm_m, _, _ = summarize_split(val_enc, val_scores)
    _, val_dsh_norm_m, _, _ = summarize_split(val_dsh_enc, val_dsh_scores)
    test_step_m, test_norm_m, test_aud_m, _ = summarize_split(test_enc, test_scores)
    _, test_dsh_norm_m, test_dsh_aud_m, _ = summarize_split(test_dsh_enc, test_dsh_scores)
    _, test_all_norm_m, _, _ = summarize_split(test_all_enc, test_all_scores)
    _, test_dsh_all_norm_m, _, _ = summarize_split(test_dsh_all_enc, test_dsh_all_scores)
    probe_step_m, probe_norm_m, probe_aud_m, probe_norm_p = summarize_split(probe_enc, probe_scores)

    probe_norm_m.update(
        {
            "pred_mean_100": round(float(probe_norm_p.mean() * 100.0), 2),
            "pred_median_100": round(float(np.median(probe_norm_p) * 100.0), 2),
            "ge_060": int((probe_norm_p >= 0.60).sum()),
            "ge_070": int((probe_norm_p >= 0.70).sum()),
            "ge_val_f05": int((probe_norm_p >= t_f05).sum()),
            "ge_val_fpr05": int((probe_norm_p >= t_fpr05).sum()),
            "details": [
                {"id": r["sample_id"], "hx_score": r["score"], "student_prob": round(float(p), 4)}
                for r, p in sorted(zip(probe_rows, probe_norm_p), key=lambda x: x[0]["sample_id"])
            ],
        }
    )

    sample_texts = [prepare_text(r["text"]) for r in test_gold[:256]]
    sample_enc = [tokenizer.encode_ids(t, max_len=MAX_LEN) for t in sample_texts]
    _, _, _, pt_sample_norm_p = summarize_split(sample_enc, test_scores[: len(sample_texts)])

    # ------------------------------------------------------------------
    # Export Zero-Duplication Dual ONNX Graphs (prelude.onnx + core_step.onnx)
    # ------------------------------------------------------------------
    mdl_cpu = mdl.cpu().eval()
    prelude_onnx = out_dir / "prelude.onnx"
    core_onnx = out_dir / "core_step.onnx"

    dummy_ids = torch.ones((2, 32), dtype=torch.long)
    dummy_mask = torch.ones((2, 32), dtype=torch.long)
    _export_and_quantize_onnx(
        PreludeExportWrapper(mdl_cpu),
        (dummy_ids, dummy_mask),
        input_names=["input_ids", "attention_mask"],
        output_names=["h0"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "h0": {0: "batch", 1: "seq"}},
        int8_path=prelude_onnx,
    )

    dummy_z = torch.zeros((2, 32, cfg.hidden_size), dtype=torch.float32)
    dummy_h0 = torch.zeros((2, 32, cfg.hidden_size), dtype=torch.float32)
    dummy_step = torch.tensor(0, dtype=torch.long)
    _export_and_quantize_onnx(
        CoreStepExportWrapper(mdl_cpu),
        (dummy_z, dummy_h0, dummy_mask, dummy_step),
        input_names=["z_prev", "h0", "attention_mask", "step_id"],
        output_names=["z_next", "prob", "logit", "halt_prob"],
        dynamic_axes={
            "z_prev": {0: "batch", 1: "seq"},
            "h0": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "z_next": {0: "batch", 1: "seq"},
            "prob": {0: "batch"},
            "logit": {0: "batch"},
            "halt_prob": {0: "batch"},
        },
        int8_path=core_onnx,
    )

    manifest = {
        "name": "MeguvizCAC",
        "version": args.version,
        "architecture": f"Prelude(1)+RRT_Cycle({args.core_layers}x{args.num_loops})+SVDEmb(E={args.embed_dim})+ByteFallback",
        "license": "Apache-2.0",
        "base_model": args.backbone,
        "quant_method": "qat_plus_dynamic_int8",
        "vocab_size": tokenizer.vocab_size,
        "byte_offset": tokenizer.byte_offset,
        "embed_dim": int(args.embed_dim),
        "hidden_size": int(cfg.hidden_size),
        "core_layers": int(args.core_layers),
        "num_loops": int(args.num_loops),
        "lora_rank": int(args.lora_rank),
        "normal_max_loops": int(normal_max),
        "audit_max_loops": int(audit_max),
        "max_len": MAX_LEN,
        "head_chars": HEAD_CHARS,
        "tail_chars": TAIL_CHARS,
        "step_temperatures": fitted_temps,
        "exit_policy": best_policy,
        "thresholds": {"f05": round(t_f05, 4), "fpr05": round(t_fpr05, 4), "score70": 0.70},
        "prelude_file": prelude_onnx.name,
        "core_file": core_onnx.name,
        "prelude_bytes": int(prelude_onnx.stat().st_size),
        "core_bytes": int(core_onnx.stat().st_size),
        "total_onnx_bytes": int(prelude_onnx.stat().st_size + core_onnx.stat().st_size),
        "prelude_sha256": sha256_file(prelude_onnx),
        "core_sha256": sha256_file(core_onnx),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # CPU ORT parity + latency verification on 256 test_gold rows
    engine = MeguvizV03Engine(out_dir, threads=1, mode="normal")
    t_single = time.monotonic()
    ort_single = [engine.predict_one(txt, mode="normal") for txt in sample_texts]
    single_ms = (time.monotonic() - t_single) * 1000.0 / max(1, len(sample_texts))

    t_batch = time.monotonic()
    ort_batch = engine.predict_batch(sample_texts, mode="normal", max_batch=32)
    batch_ms = (time.monotonic() - t_batch) * 1000.0 / max(1, len(sample_texts))

    ort_probs = np.array([x["prob"] for x in ort_batch], dtype=np.float32)
    ort_single_probs = np.array([x["prob"] for x in ort_single], dtype=np.float32)
    parity = {
        "mae_vs_pytorch": round(float(np.mean(np.abs(pt_sample_norm_p - ort_probs))), 5),
        "max_abs_diff_vs_pytorch": round(float(np.max(np.abs(pt_sample_norm_p - ort_probs))), 5),
        "batch_vs_single_mae": round(float(np.mean(np.abs(ort_single_probs - ort_probs))), 6),
        "single_1thread_ms": round(single_ms, 2),
        "compact_batch32_1thread_ms": round(batch_ms, 2),
        "ort_avg_steps": round(float(np.mean([x["steps_used"] for x in ort_batch])), 3),
    }

    # Build Main v0.3-CPU-INT8 bundle zip
    metrics_file = out_dir / f"metrics-{args.version}.json"
    metrics = {
        "version": args.version,
        "backbone": args.backbone,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "batch_size_used": batch_size,
        "arch": {
            "embed_dim": args.embed_dim,
            "core_layers": args.core_layers,
            "num_loops": args.num_loops,
            "lora_rank": args.lora_rank,
            "normal_max_loops": normal_max,
            "audit_max_loops": audit_max,
            "vocab_size": tokenizer.vocab_size,
        },
        "step_temperatures": fitted_temps,
        "exit_policy": best_policy,
        "thresholds": {"val_f05": round(t_f05, 4), "val_fpr05": round(t_fpr05, 4)},
        "val_bili_gold": val_norm_m,
        "val_dsh_gold": val_dsh_norm_m,
        "test_bili_gold": test_norm_m,
        "test_bili_gold_audit": test_aud_m,
        "test_bili_gold_by_step": test_step_m,
        "test_dsh_gold": test_dsh_norm_m,
        "test_dsh_gold_audit": test_dsh_aud_m,
        "test_all_bili_distilled": test_all_norm_m,
        "test_all_dsh_distilled": test_dsh_all_norm_m,
        "probe_u001_u017": probe_norm_m,
        "probe_u001_u017_audit": {k: v for k, v in probe_aud_m.items() if k != "details"},
        "probe_u001_u017_by_step": probe_step_m,
        "onnx_int8": {
            **parity,
            "prelude_bytes": manifest["prelude_bytes"],
            "core_bytes": manifest["core_bytes"],
            "total_onnx_bytes": manifest["total_onnx_bytes"],
        },
        "qwen_rows_used": len(qwen_map) if qwen_map else 0,
        "literary_pairs_used": len(pairs),
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }

    # Optional W4-Emb / W8-Body Edge Bundle (opset-21 GatherBlockQuantized)
    prelude_w4 = out_dir / "prelude_w4.onnx"
    w4_res = quantize_embedding_w4_opset21(prelude_onnx, prelude_w4, block_size=32)
    metrics["edge_w4_w8"] = w4_res
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    zip_main = out_dir / f"meguvizcac-{args.version}-int8.zip"
    common_assets = ["vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", metrics_file.name]
    with ZipFile(zip_main, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
        for fn in (prelude_onnx.name, core_onnx.name, "manifest.json", *common_assets):
            fp = out_dir / fn
            if fp.is_file():
                zf.write(fp, fn)
    metrics["onnx_int8"]["zip_bytes"] = int(zip_main.stat().st_size)

    if w4_res.get("ok") and prelude_w4.is_file():
        edge_dir = out_dir / "_edge_tmp"
        edge_dir.mkdir(parents=True, exist_ok=True)
        edge_manifest = dict(manifest)
        edge_manifest["quant_method"] = "w4_emb_opset21_plus_w8_body"
        edge_manifest["prelude_file"] = prelude_w4.name
        edge_manifest["prelude_bytes"] = int(prelude_w4.stat().st_size)
        edge_manifest["total_onnx_bytes"] = int(prelude_w4.stat().st_size + core_onnx.stat().st_size)
        edge_manifest["prelude_sha256"] = sha256_file(prelude_w4)
        (edge_dir / "manifest.json").write_text(json.dumps(edge_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        zip_edge = out_dir / f"meguvizcac-{args.version}-edge-mixed.zip"
        with ZipFile(zip_edge, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
            zf.write(prelude_w4, prelude_w4.name)
            zf.write(core_onnx, core_onnx.name)
            zf.write(edge_dir / "manifest.json", "manifest.json")
            for fn in common_assets:
                fp = out_dir / fn
                if fp.is_file():
                    zf.write(fp, fn)
        shutil.rmtree(edge_dir, ignore_errors=True)
        w4_res["edge_zip_bytes"] = int(zip_edge.stat().st_size)
        w4_res["edge_total_onnx_bytes"] = edge_manifest["total_onnx_bytes"]

    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"=== MeguvizCAC {args.version} COMPLETE ({metrics['elapsed_seconds']}s) ===", flush=True)
    print(
        json.dumps(
            {
                "test_bili_gold": metrics["test_bili_gold"],
                "test_dsh_gold": metrics["test_dsh_gold"],
                "probe_u001_u017": {k: v for k, v in probe_norm_m.items() if k != "details"},
                "onnx_int8": metrics["onnx_int8"],
                "edge_w4_w8": w4_res,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--qwen-data", type=Path, default=None)
    parser.add_argument("--pairs-data", type=Path, default=None)
    parser.add_argument("--vocab", type=Path, default=None, help="Optional vocab.txt/tokenizer.json dir for --dry-run")
    parser.add_argument("--out", type=Path, default=Path("./out_v03"))
    parser.add_argument("--version", default="v0.3-loop-2x4-e128")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--core-layers", type=int, default=2)
    parser.add_argument("--num-loops", type=int, default=4)
    parser.add_argument("--normal-max-loops", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--qat-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto (256 for >=35GB A100, else 128)")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--qat-lr", type=float, default=5e-6)
    parser.add_argument("--synth-cap", type=int, default=5000)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_dry(
            args.data.resolve(),
            args.qwen_data.resolve() if args.qwen_data else None,
            args.pairs_data.resolve() if args.pairs_data else None,
            args.vocab.resolve() if args.vocab else None,
        )
        return
    train_and_export(args)


if __name__ == "__main__":
    main()
