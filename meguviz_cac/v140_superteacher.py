"""MeguvizCAC v1.4.0-SuperTeacher-Ultra Core Library (SenseNova-Loop-9B 3-Paradigm Zero-PII Super-Teacher).

Upgrades v1.3.0's 3x300M Chinese-only Super-Teacher to a strictly Zero-PII (`diff === 0.0`),
Dual-Axis (AI + Commercial Promo), Multilingual (ZH/EN/JA + Homoglyph), 3-Paradigm Super-Teacher
Ensemble anchored by SenseTime's 2026 Looped Transformer (`sensenova/SenseNova-Loop-9B`) on 96 GB Blackwell:
  1. `electra_large`     (`325M`, 24L Chinese RTD)                    : `hfl/chinese-electra-180g-large-discriminator`
  2. `bge_m3`            (`568M`, 24L Bidirectional ML Contrastive)   : `BAAI/bge-m3`
  3. `sensenova_loop_9b` (`9.0B`, 32L->56L Looped Hybrid-Attn + mHC)  : `sensenova/SenseNova-Loop-9B` (BF16/FP32-mHC + LoRA r=32)
  (Registry also supports `bge_reranker_m3`, `sensenova_loop_9b_preview`, and custom HF repos.)

Key v1.4.0 engineering upgrades over v1.3.0 (strictly zero OjinjinBench contact!):
  - Native SenseNova-Loop-9B VLM text-backbone unwrapping (`language_model` extraction, drops unused
    vision tower & `lm_head` to save VRAM!) + `dtype="auto"` preserving FP32 mHC Sinkhorn stability!
  - Native zero-dependency PyTorch `LoRALinearV140` covering BOTH Full-Attention (`q/k/v/o_proj`),
    Linear-Attention (`in_proj/in_proj_qkv/in_proj_qkvz/in_proj_z/in_proj_ba/out_proj`), and FFN
    (`gate/up/down_proj`) with `min(in,out) >= 64` guard (never touches 2-stream mHC routing!) + exact
    `merge_lora_inplace()` for full-speed 250K distillation-pool inference & compact ~240 MB checkpoints.
  - Per-stat normalized 3-stat poolers:
      * Bidirectional Encoders : `[LN(max), LN(mean), LN(attn)]`
      * Looped / Causal LLMs   : `[LN(last_valid_eos), LN(comment_mean), LN(comment_attn)]`
  - Fixed XLM-RoBERTa `cls_token_id=0` / `pad_token_id=1` handling (never uses truthy `or`!).
  - Full ingestion of newly merged `+57,585` `v123-sibling-delta` gold rows (incl. `1,422` R1c-rescued
    text-verifiable real AI, `+28,991` >1y old-video hard humans, `1,200` Q2 dual-pos, `200` no-period
    short stealth AI, `813` real decorator humans) + `15,903` Counterfactual Mirror Pairs (incl. `1,200`
    micro-short `12-30c` & `500` decorator `1:1` pairs) + clean `EN/JA` multilingual gold anchors!
  - Counterfactual Mirror Pairwise Margin Loss directly inside Super-Teacher training.
  - Mega-batch length-bucketed batching (~3x speedup on 9B Looped backbone with zero shuffling bias).
"""

from __future__ import annotations

import gc
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BertConfig, BertModel, Qwen2Config, Qwen2Model

from .v07_features import canonicalize_zero_pii_v07
from .v11_features import _CONTACT_ID_PII_RE
from .v123_data_refinery import load_sibling_and_delta_v123, sanitize_era5_v123


# ---------------------------------------------------------------------------
# 0. Backbone Registry (100% Public Non-Gated HuggingFace Repos)
# ---------------------------------------------------------------------------

SUPERTEACHER_BACKBONES_V140: dict[str, dict[str, Any]] = {
    "electra_large": {
        "hf_name": "hfl/chinese-electra-180g-large-discriminator",
        "arch_type": "encoder",
        "ft_mode": "full",
        "trust_remote_code": False,
        "preserve_auto_dtype": False,
        "param_tag": "325M-ZH-RTD",
        "batch_size": 64,
        "grad_accum": 2,
        "eval_batch_size": 256,
        "backbone_lr": 2.5e-5,
        "head_lr": 2.5e-4,
        "llrd": 0.90,
        "lora_r": 0,
        "lora_alpha": 0,
        "grad_ckpt": False,
    },
    "bge_m3": {
        "hf_name": "BAAI/bge-m3",
        "arch_type": "encoder",
        "ft_mode": "full",
        "trust_remote_code": False,
        "preserve_auto_dtype": False,
        "param_tag": "568M-ML-Contrastive",
        "batch_size": 48,
        "grad_accum": 3,
        "eval_batch_size": 192,
        "backbone_lr": 2.0e-5,
        "head_lr": 2.5e-4,
        "llrd": 0.90,
        "lora_r": 0,
        "lora_alpha": 0,
        "grad_ckpt": False,
    },
    "bge_reranker_m3": {
        "hf_name": "BAAI/bge-reranker-v2-m3",
        "arch_type": "encoder",
        "ft_mode": "full",
        "trust_remote_code": False,
        "preserve_auto_dtype": False,
        "param_tag": "568M-ML-Reranker",
        "batch_size": 48,
        "grad_accum": 3,
        "eval_batch_size": 192,
        "backbone_lr": 2.0e-5,
        "head_lr": 2.5e-4,
        "llrd": 0.90,
        "lora_r": 0,
        "lora_alpha": 0,
        "grad_ckpt": False,
    },
    "sensenova_loop_9b": {
        "hf_name": "sensenova/SenseNova-Loop-9B",
        "arch_type": "causal_lm",
        "ft_mode": "lora",
        "trust_remote_code": True,
        "preserve_auto_dtype": True,  # Preserves FP32 mHC Sinkhorn routing + BF16 backbone weights!
        "param_tag": "9.0B-56L-SenseNova-Loop-mHC",
        "batch_size": 24,
        "grad_accum": 5,
        "eval_batch_size": 96,
        "backbone_lr": 1.0e-4,
        "head_lr": 2.5e-4,
        "llrd": 1.0,
        "lora_r": 32,
        "lora_alpha": 64,
        "grad_ckpt": True,
    },
    "sensenova_loop_9b_preview": {
        "hf_name": "sensenova/SenseNova-Loop-9B-Preview",
        "arch_type": "causal_lm",
        "ft_mode": "lora",
        "trust_remote_code": True,
        "preserve_auto_dtype": True,
        "param_tag": "9.0B-56L-SenseNova-Loop-Preview",
        "batch_size": 24,
        "grad_accum": 5,
        "eval_batch_size": 96,
        "backbone_lr": 1.0e-4,
        "head_lr": 2.5e-4,
        "llrd": 1.0,
        "lora_r": 32,
        "lora_alpha": 64,
        "grad_ckpt": True,
    },
}

DEFAULT_MODELS_V140 = "electra_large,bge_m3,sensenova_loop_9b"

CAUSAL_TASK_PREFIX_V140 = "判断评论是否为AI生成或商业推广：\n"

LORA_TARGET_LEAF_NAMES_V140 = frozenset({
    # Standard Full-Attention & FFN projections
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # Hybrid Linear-Attention / Gated-DeltaNet projections (SenseNova-Loop-9B!)
    "in_proj",
    "in_proj_qkv",
    "in_proj_qkvz",
    "in_proj_z",
    "in_proj_ba",
    "out_proj",
})

SUB4_MAP_V140: dict[str, int] = {
    "hard_contact": 0,
    "affiliate_link": 1,
    "soft_shill": 2,
    "self_promo": 3,
    "coupon_commerce": 1,
    "self_promo_funnel": 3,
    "tool_boilerplate": 2,
}

NOISY_BEHAVIOR_ORIGINS_V140 = frozenset({
    "Coordinated-Latent-Cluster",
    "Human-Assisted",
})
NOISY_SCRIPTBOT_SOURCES_V140 = frozenset({
    "sibling_god_tier_calibrated_gold",
    "sibling_conflict_adjudicated",
})


def canonicalize_zero_pii_v140(text: str, max_chars: int = 8192) -> str:
    """Strictly origin/main-compatible Zero-PII canonicalizer (`@user` & `<CONTACT_ID>` -> diff === 0.0)."""
    cl, _ = canonicalize_zero_pii_v07(str(text or ""), max_chars=max_chars)
    return _CONTACT_ID_PII_RE.sub(r"\1<CONTACT_ID>", cl)


def text_sha16_v140(text: str) -> str:
    norm = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()[:16]


def read_jsonl_any_v140(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.is_file():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    out: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------------
# 1. Online Shortcut Breakers (ZH Punctuation + EN Homoglyph/ZW Jitter)
# ---------------------------------------------------------------------------

_RE_TRAIL_PERIOD = re.compile(r"[。\.]+(\s*)$")
_RE_EMDASH = re.compile(r"——|—|--")
_RE_CORNER_Q = re.compile(r"[「」『』]")
_HOMOGLYPH_EN_MAP = str.maketrans({
    "a": "\u0430",
    "e": "\u0435",
    "o": "\u043e",
    "i": "\u0456",
    "c": "\u0441",
    "p": "\u0440",
})


def jitter_ai_text_v140(text: str, rng: random.Random, is_ml_en: bool = False, prob: float = 0.35) -> str:
    """Online punctuation-shortcut breaker (ZH) + Cyrillic homoglyph/ZW jitter (EN) on AI positives."""
    if not text or rng.random() >= prob:
        return text
    if is_ml_en:
        if rng.random() < 0.65:
            chars = list(text)
            for i in range(len(chars)):
                if chars[i] in "aeoicp" and rng.random() < 0.22:
                    chars[i] = chars[i].translate(_HOMOGLYPH_EN_MAP)
            return "".join(chars)
        out = []
        for ch in text:
            out.append(ch)
            if ch == " " and rng.random() < 0.35:
                out.append("\u200b")
        return "".join(out)
    s = text
    if rng.random() < 0.75:
        s = _RE_TRAIL_PERIOD.sub(r"\1", s)
    if rng.random() < 0.60 and _RE_EMDASH.search(s):
        s = _RE_EMDASH.sub(rng.choice(["，", " ", "，"]), s)
    if rng.random() < 0.55 and _RE_CORNER_Q.search(s):
        s = _RE_CORNER_Q.sub("", s)
    return s.strip() or text


# ---------------------------------------------------------------------------
# 2. Universal Zero-PII Head-192 + Tail-80 Tokenizer (Encoder + Causal LLM)
# ---------------------------------------------------------------------------

def _safe_tok_id(tokenizer: Any, attr: str, fallback: int) -> int:
    val = getattr(tokenizer, attr, None)
    return int(val) if val is not None else int(fallback)


def encode_universal_v140(
    tokenizer: Any,
    texts: list[str],
    arch_type: str = "encoder",
    max_len: int = 288,
    head_len: int = 192,
    prefix_ids: list[int] | None = None,
    already_canonical: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode batch with strict Zero-PII canonicalization (`C1 === 0.0`) & Head+Tail truncation.

    Returns `(input_ids, attention_mask, pool_mask)` where `pool_mask` excludes padding and any
    constant instruction prefix so mean/attention pooling focuses purely on the comment + EOS.
    """
    clean_texts = [
        str(t) if already_canonical else canonicalize_zero_pii_v140(str(t))
        for t in texts
    ]

    enc = tokenizer(
        clean_texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    seqs: list[list[int]] = []
    pool_starts: list[int] = []

    if arch_type == "causal_lm":
        eos_id = _safe_tok_id(tokenizer, "eos_token_id", _safe_tok_id(tokenizer, "sep_token_id", 2))
        pad_id = _safe_tok_id(tokenizer, "pad_token_id", eos_id)
        bos_val = getattr(tokenizer, "bos_token_id", None)
        pref: list[int] = []
        if bos_val is not None:
            pref.append(int(bos_val))
        if prefix_ids is None:
            p_enc = tokenizer(
                CAUSAL_TASK_PREFIX_V140,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            p_ids = p_enc["input_ids"]
            if p_ids and isinstance(p_ids[0], list):
                p_ids = p_ids[0]
            pref.extend(int(x) for x in p_ids)
        else:
            pref.extend(int(x) for x in prefix_ids)

        body_budget = max(16, int(max_len) - len(pref) - 1)
        h_cap = min(int(head_len), body_budget - 8)
        t_cap = max(8, body_budget - h_cap)
        p_start = len(pref)
        for raw_ids in enc["input_ids"]:
            if len(raw_ids) > body_budget:
                body = raw_ids[:h_cap] + raw_ids[-t_cap:]
            else:
                body = raw_ids
            seqs.append(pref + body + [eos_id])
            pool_starts.append(p_start)
    else:
        # Bidirectional Encoder (BERT, Electra, MacBERT, XLM-RoBERTa / BGE-M3 where cls=0, sep=2, pad=1!)
        cls_id = _safe_tok_id(tokenizer, "cls_token_id", _safe_tok_id(tokenizer, "bos_token_id", 101))
        sep_id = _safe_tok_id(tokenizer, "sep_token_id", _safe_tok_id(tokenizer, "eos_token_id", 102))
        pad_id = _safe_tok_id(tokenizer, "pad_token_id", 0)
        body_budget = max(16, int(max_len) - 2)
        h_cap = min(int(head_len), body_budget - 8)
        t_cap = max(8, body_budget - h_cap)
        for raw_ids in enc["input_ids"]:
            if len(raw_ids) > body_budget:
                body = raw_ids[:h_cap] + raw_ids[-t_cap:]
            else:
                body = raw_ids
            seqs.append([cls_id] + body + [sep_id])
            pool_starts.append(0)

    ml = max(4, max(len(s) for s in seqs))
    ids = torch.full((len(seqs), ml), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), ml), dtype=torch.long)
    pmask = torch.zeros((len(seqs), ml), dtype=torch.long)
    for i, s in enumerate(seqs):
        n_s = len(s)
        ids[i, :n_s] = torch.tensor(s, dtype=torch.long)
        mask[i, :n_s] = 1
        ps = min(pool_starts[i], n_s - 1)
        pmask[i, ps:n_s] = 1
    return ids, mask, pmask


# ---------------------------------------------------------------------------
# 3. Native Zero-Dependency PyTorch LoRA (`LoRALinearV140`) with Exact Merge
# ---------------------------------------------------------------------------

class LoRALinearV140(nn.Module):
    """Zero-dependency PyTorch LoRA adapter wrapping a frozen `nn.Linear` with exact inference merge."""

    def __init__(self, base: nn.Linear, r: int = 32, alpha: int = 64, dropout: float = 0.05) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(max(1, r))
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        # Keep LoRA weights in FP32 for clean AdamW precision; cast on-the-fly in forward
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty((self.r, self.in_features), dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.r), dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        if not torch.is_autocast_enabled() and x_in.dtype != self.base.weight.dtype:
            x_in = x_in.to(dtype=self.base.weight.dtype)
        out = self.base(x_in)
        if self.merged:
            return out
        xd = self.drop(x_in)
        h = F.linear(xd, self.lora_A.to(dtype=xd.dtype))
        delta = F.linear(h, self.lora_B.to(dtype=xd.dtype)) * self.scaling
        return out + delta

    @torch.no_grad()
    def merge_inplace(self) -> None:
        if self.merged:
            return
        dw = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        self.base.weight.data.add_(dw.to(dtype=self.base.weight.dtype, device=self.base.weight.device))
        self.merged = True

    @torch.no_grad()
    def unmerge_inplace(self) -> None:
        if not self.merged:
            return
        dw = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        self.base.weight.data.sub_(dw.to(dtype=self.base.weight.dtype, device=self.base.weight.device))
        self.merged = False


def inject_lora_v140(
    module: nn.Module,
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
    target_names: frozenset[str] = LORA_TARGET_LEAF_NAMES_V140,
    min_dim: int = 64,
) -> int:
    """Freeze all parameters in `module` and wrap every matching `nn.Linear` (`min(in,out) >= 64`) with LoRA."""
    for p in module.parameters():
        p.requires_grad_(False)
    replaced = 0
    for parent_name, parent in list(module.named_modules()):
        # Never wrap vision towers or mHC Sinkhorn routing heads if any survived unwrapping!
        pnl = parent_name.lower()
        if any(skip in pnl for skip in ("visual", "vision", "mhc", "hyper_conn", "router")):
            continue
        for child_name, child in list(parent.named_children()):
            if (
                child_name in target_names
                and isinstance(child, nn.Linear)
                and min(int(child.in_features), int(child.out_features)) >= min_dim
            ):
                setattr(parent, child_name, LoRALinearV140(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    return replaced


# ---------------------------------------------------------------------------
# 4. Universal Dual-Axis Multi-Billion Super-Teacher Network
# ---------------------------------------------------------------------------

def _unwrap_text_backbone_v140(loaded: nn.Module) -> nn.Module:
    """Extract pure text decoder from multimodal VLMs (e.g. SenseNova-Loop-9B) & drop unused vision/lm_head!"""
    cand: nn.Module | None = None
    # Pattern 1: SenseNovaLoopForConditionalGeneration -> .model.language_model
    if hasattr(loaded, "model") and hasattr(loaded.model, "language_model") and isinstance(loaded.model.language_model, nn.Module):
        cand = loaded.model.language_model
    # Pattern 2: SenseNovaLoopModel -> .language_model
    elif hasattr(loaded, "language_model") and isinstance(loaded.language_model, nn.Module):
        cand = loaded.language_model
    # Pattern 3: AutoModelForCausalLM text-only -> .model (drops lm_head)
    elif hasattr(loaded, "lm_head") and hasattr(loaded, "model") and isinstance(loaded.model, nn.Module):
        cand = loaded.model

    if cand is not None:
        has_emb = hasattr(cand, "embed_tokens") or (
            hasattr(cand, "get_input_embeddings") and cand.get_input_embeddings() is not None
        )
        if has_emb:
            return cand
    return loaded


def _resolve_hidden_size_v140(cfg: Any, encoder: nn.Module) -> int:
    for obj in (getattr(encoder, "config", None), getattr(cfg, "text_config", None), cfg):
        if obj is not None and getattr(obj, "hidden_size", None):
            return int(obj.hidden_size)
    emb = encoder.get_input_embeddings() if hasattr(encoder, "get_input_embeddings") else None
    if emb is not None and hasattr(emb, "weight"):
        return int(emb.weight.shape[-1])
    raise ValueError(f"Unable to determine hidden_size from config={type(cfg)} encoder={type(encoder)}")


def _load_automodel_safe(
    hf_name: str,
    cfg: Any,
    load_dtype: torch.dtype | str | None,
    trust_remote_code: bool = False,
) -> nn.Module:
    kwargs: dict[str, Any] = {"trust_remote_code": bool(trust_remote_code)}
    if load_dtype is not None:
        major_v = int(str(transformers.__version__).split(".")[0])
        if major_v >= 5:
            kwargs["dtype"] = load_dtype
        else:
            kwargs["torch_dtype"] = load_dtype

    loaders: list[Any] = [AutoModel, AutoModelForCausalLM]
    am_i2t = getattr(transformers, "AutoModelForImageTextToText", None)
    if am_i2t is not None:
        loaders.append(am_i2t)

    last_err: Exception | None = None
    for loader_cls in loaders:
        extras: list[dict[str, Any]] = [{"config": cfg, "attn_implementation": "sdpa"}, {"config": cfg}, {}]
        if trust_remote_code and loader_cls is not AutoModel:
            # Custom VLM auto_map classes (like SenseNova-Loop-9B) load cleanest without forcing config override
            extras = [{}, {"config": cfg}]
        for extra in extras:
            try:
                loaded = loader_cls.from_pretrained(hf_name, **kwargs, **extra)
                text_enc = _unwrap_text_backbone_v140(loaded)
                if text_enc is not loaded:
                    del loaded
                    gc.collect()
                return text_enc
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Failed to load backbone {hf_name}: {last_err}")


class UniversalDualAxisSuperTeacherV140(nn.Module):
    """Unified Dual-Axis Super-Teacher supporting Encoders & Looped/Causal LLMs (incl. SenseNova-Loop-9B)."""

    def __init__(
        self,
        backbone_name_or_path: str,
        arch_type: str = "encoder",
        ft_mode: str = "full",
        lora_r: int = 32,
        lora_alpha: int = 64,
        dropout: float = 0.15,
        load_dtype: torch.dtype | str | None = None,
        trust_remote_code: bool = False,
        smoke_tiny: bool = False,
        vocab_size: int = 2048,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone_name_or_path)
        self.arch_type = str(arch_type)
        self.ft_mode = str(ft_mode)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.trust_remote_code = bool(trust_remote_code)
        self.smoke_tiny = bool(smoke_tiny)
        self.lora_modules_count = 0

        if smoke_tiny:
            if self.arch_type == "causal_lm":
                cfg = Qwen2Config(
                    vocab_size=vocab_size,
                    hidden_size=96,
                    intermediate_size=192,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=512,
                )
                self.encoder = Qwen2Model(cfg)
                h_dim = int(cfg.hidden_size)
            else:
                cfg = BertConfig(
                    vocab_size=vocab_size,
                    hidden_size=96,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    intermediate_size=192,
                    max_position_embeddings=512,
                )
                self.encoder = BertModel(cfg, add_pooling_layer=False)
                h_dim = int(cfg.hidden_size)
        else:
            cfg = AutoConfig.from_pretrained(backbone_name_or_path, trust_remote_code=self.trust_remote_code)
            if hasattr(cfg, "use_cache"):
                cfg.use_cache = False
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "use_cache"):
                cfg.text_config.use_cache = False
            self.encoder = _load_automodel_safe(
                backbone_name_or_path,
                cfg,
                load_dtype if ft_mode == "lora" else None,
                trust_remote_code=self.trust_remote_code,
            )
            h_dim = _resolve_hidden_size_v140(cfg, self.encoder)

        # Inspect forward signature once so `use_cache=False` is passed safely on any custom backbone
        try:
            fwd_params = inspect.signature(self.encoder.forward).parameters
            self._supports_use_cache = "use_cache" in fwd_params
        except Exception:
            self._supports_use_cache = self.arch_type == "causal_lm"

        if self.ft_mode == "lora":
            self.lora_modules_count = inject_lora_v140(
                self.encoder,
                r=max(4, self.lora_r if not smoke_tiny else 8),
                alpha=max(8, self.lora_alpha if not smoke_tiny else 16),
                dropout=0.05,
            )

        self.h_dim = h_dim
        attn_mid = min(256, max(32, h_dim // 8))
        self.attn_pool = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, attn_mid),
            nn.Tanh(),
            nn.Linear(attn_mid, 1),
        )
        # Per-stat LayerNorm balances scale across last-token/max/mean/attn branches (critical for Qwen LLMs!)
        self.ln_s1 = nn.LayerNorm(h_dim)
        self.ln_s2 = nn.LayerNorm(h_dim)
        self.ln_s3 = nn.LayerNorm(h_dim)

        pool_dim = h_dim * 3
        mid_dim = min(768, max(64, h_dim // 4))

        # Strictly separate AI and Commercial-Promo heads (structural zero-leakage between axes!)
        self.ai_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pool_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )
        self.promo_trunk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pool_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.promo_head = nn.Linear(mid_dim, 1)
        self.promo_sub4_head = nn.Linear(mid_dim, 4)

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            try:
                self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.encoder.gradient_checkpointing_enable()
        if self.ft_mode == "lora":
            if hasattr(self.encoder, "enable_input_require_grads"):
                self.encoder.enable_input_require_grads()
            else:
                emb = self.encoder.get_input_embeddings()
                if emb is not None:
                    def _make_grads(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> torch.Tensor:
                        if self.training and isinstance(out, torch.Tensor):
                            out.requires_grad_(True)
                        return out
                    emb.register_forward_hook(_make_grads)

    def merge_lora_inplace(self) -> int:
        cnt = 0
        for m in self.encoder.modules():
            if isinstance(m, LoRALinearV140):
                m.merge_inplace()
                cnt += 1
        return cnt

    def unmerge_lora_inplace(self) -> int:
        cnt = 0
        for m in self.encoder.modules():
            if isinstance(m, LoRALinearV140):
                m.unmerge_inplace()
                cnt += 1
        return cnt

    def trainable_state_dict(self, to_cpu: bool = True) -> dict[str, torch.Tensor]:
        """Compact state dict containing ONLY trainable params (~220 MB for 8B LoRA, ~1.1 GB for BGE-M3!)."""
        out: dict[str, torch.Tensor] = {}
        for k, p in self.named_parameters():
            if p.requires_grad:
                t = p.detach()
                out[k] = t.cpu().clone() if to_cpu else t.clone()
        return out

    def load_trainable_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        param_map = dict(self.named_parameters())
        with torch.no_grad():
            for k, v in sd.items():
                if k in param_map:
                    param_map[k].copy_(v.to(device=param_map[k].device, dtype=param_map[k].dtype))

    def pool_tokens(
        self,
        h: torch.Tensor,
        attention_mask: torch.Tensor,
        pool_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pm = attention_mask if pool_mask is None else pool_mask
        h_f = h.float()
        m_u = pm.float().unsqueeze(-1)
        denom = m_u.sum(dim=1).clamp_min(1.0)
        mean_p = (h_f * m_u).sum(dim=1) / denom

        a_logits = self.attn_pool(h_f).squeeze(-1)
        a_logits = a_logits.masked_fill(pm == 0, -1e4)
        a_w = torch.softmax(a_logits, dim=-1).unsqueeze(-1)
        attn_p = (h_f * a_w).sum(dim=1)

        if self.arch_type == "causal_lm":
            # Exact last valid token (EOS) attended over instruction prefix + entire comment!
            last_idx = (attention_mask.long().sum(dim=1) - 1).clamp_min(0)
            b_idx = torch.arange(h_f.size(0), device=h_f.device)
            first_stat = h_f[b_idx, last_idx]
        else:
            neg_inf = -1e4
            h_masked = h_f.masked_fill(pm.unsqueeze(-1) == 0, neg_inf)
            first_stat = h_masked.max(dim=1).values

        return torch.cat([self.ln_s1(first_stat), self.ln_s2(mean_p), self.ln_s3(attn_p)], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pool_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self._supports_use_cache:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = self.pool_tokens(h, attention_mask, pool_mask=pool_mask)
        logit_ai = self.ai_head(pooled).squeeze(-1)
        pr_h = self.promo_trunk(pooled)
        logit_promo = self.promo_head(pr_h).squeeze(-1)
        logit_sub4 = self.promo_sub4_head(pr_h)
        return {
            "logit_ai": logit_ai,
            "logit_promo": logit_promo,
            "logit_sub4": logit_sub4,
        }


def build_optimizer_param_groups_v140(
    model: UniversalDualAxisSuperTeacherV140,
    backbone_lr: float = 2.5e-5,
    head_lr: float = 2.5e-4,
    llrd: float = 0.90,
    weight_decay: float = 0.01,
) -> list[dict[str, Any]]:
    """LLRD param groups for full-FT encoders OR clean LoRA/Head LR split for multi-billion LLMs."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm", "norm.weight", "ln_s")
    if model.ft_mode == "lora":
        lora_decay, lora_nodecay, head_decay, head_nodecay = [], [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_nd = any(nd in name for nd in no_decay)
            if name.startswith("encoder."):
                (lora_nodecay if is_nd else lora_decay).append(p)
            else:
                (head_nodecay if is_nd else head_decay).append(p)
        out = []
        for ps, lr, wd in [
            (lora_decay, backbone_lr, weight_decay),
            (lora_nodecay, backbone_lr, 0.0),
            (head_decay, head_lr, weight_decay),
            (head_nodecay, head_lr, 0.0),
        ]:
            if ps:
                out.append({"params": ps, "lr": lr, "base_lr": lr, "weight_decay": wd})
        return out

    num_layers = 24
    cfg = getattr(model.encoder, "config", None)
    if cfg is not None and hasattr(cfg, "num_hidden_layers"):
        num_layers = int(cfg.num_hidden_layers)

    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    layer_re = re.compile(r"(?:encoder\.layer|layers)\.(\d+)\.")

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        if not name.startswith("encoder."):
            lr = head_lr
        elif "embeddings." in name or "embed_tokens." in name:
            lr = backbone_lr * (llrd ** (num_layers + 1))
        else:
            m = layer_re.search(name)
            if m:
                l_idx = int(m.group(1))
                depth_from_top = max(0, num_layers - 1 - l_idx)
                lr = backbone_lr * (llrd ** depth_from_top)
            else:
                lr = backbone_lr
        key = (round(lr, 10), wd)
        groups.setdefault(key, []).append(p)

    return [
        {"params": ps, "lr": lr, "base_lr": lr, "weight_decay": wd}
        for (lr, wd), ps in groups.items()
    ]


# ---------------------------------------------------------------------------
# 5. Losses: Top-Tail Pairwise Margin + Counterfactual Mirror Margin Loss
# ---------------------------------------------------------------------------

def compute_top_tail_pairwise_loss_v140(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    hard_neg_boost: torch.Tensor | None = None,
    margin: float = 1.8,
    top_neg_frac: float = 0.25,
) -> torch.Tensor:
    m = valid_mask > 0.5
    if int(m.sum().item()) < 4:
        return logits.sum() * 0.0
    z = logits[m].float()
    y = targets[m].float()
    pos_m = y >= 0.70
    neg_m = y <= 0.25
    n_pos = int(pos_m.sum().item())
    n_neg = int(neg_m.sum().item())
    if n_pos == 0 or n_neg == 0:
        return logits.sum() * 0.0

    z_pos = z[pos_m]
    z_neg = z[neg_m]
    k = max(1, min(n_neg, int(math.ceil(n_neg * top_neg_frac))))
    top_vals, top_idx = torch.topk(z_neg, k=k, largest=True)

    diff = z_pos.unsqueeze(1) - top_vals.unsqueeze(0)
    pair_loss = F.softplus(margin - diff)

    if hard_neg_boost is not None:
        hn = hard_neg_boost[m][neg_m][top_idx].float()
        pair_loss = pair_loss * (1.0 + 0.85 * hn).unsqueeze(0)
    return pair_loss.mean()


def compute_dual_axis_loss_v140(
    out: dict[str, torch.Tensor],
    y_ai: torch.Tensor,
    has_ai: torch.Tensor,
    y_pr: torch.Tensor,
    has_pr: torch.Tensor,
    sub4: torch.Tensor,
    w_ai: torch.Tensor,
    w_pr: torch.Tensor,
    hard_neg_flag: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    lg_ai = out["logit_ai"].float()
    lg_pr = out["logit_promo"].float()
    lg_s4 = out["logit_sub4"].float()

    ai_m = has_ai > 0.5
    if ai_m.any():
        ya_s = y_ai[ai_m].float() * 0.96 + 0.02
        l_ai_bce = (F.binary_cross_entropy_with_logits(lg_ai[ai_m], ya_s, reduction="none") * w_ai[ai_m].float()).mean()
        l_ai_tail = compute_top_tail_pairwise_loss_v140(lg_ai, y_ai, has_ai, hard_neg_boost=hard_neg_flag, margin=1.8, top_neg_frac=0.25)
        hn_m = ai_m & (y_ai <= 0.20) & (hard_neg_flag > 0.5)
        l_ai_hn = F.relu(lg_ai[hn_m] + 0.55).pow(2).mean() if hn_m.any() else lg_ai.sum() * 0.0
    else:
        l_ai_bce = lg_ai.sum() * 0.0
        l_ai_tail = lg_ai.sum() * 0.0
        l_ai_hn = lg_ai.sum() * 0.0

    pr_m = has_pr > 0.5
    if pr_m.any():
        yp_s = y_pr[pr_m].float() * 0.96 + 0.02
        l_pr_bce = (F.binary_cross_entropy_with_logits(lg_pr[pr_m], yp_s, reduction="none") * w_pr[pr_m].float()).mean()
        l_pr_tail = compute_top_tail_pairwise_loss_v140(lg_pr, y_pr, has_pr, margin=1.6, top_neg_frac=0.25)
    else:
        l_pr_bce = lg_pr.sum() * 0.0
        l_pr_tail = lg_pr.sum() * 0.0

    s4_m = pr_m & (sub4 >= 0) & (sub4 < 4)
    l_s4 = F.cross_entropy(lg_s4[s4_m], sub4[s4_m].long()) if s4_m.any() else lg_pr.sum() * 0.0

    loss = l_ai_bce + 0.35 * l_ai_tail + 0.22 * l_ai_hn + 0.72 * l_pr_bce + 0.20 * l_pr_tail + 0.12 * l_s4
    return loss, {
        "ai_bce": float(l_ai_bce.item()),
        "ai_tail": float(l_ai_tail.item()),
        "ai_hn": float(l_ai_hn.item()),
        "pr_bce": float(l_pr_bce.item()),
    }


def compute_cf_mirror_loss_v140(
    logit_ai_pos: torch.Tensor,
    logit_ai_neg: torch.Tensor,
    pair_weights: torch.Tensor,
    margin: float = 2.4,
) -> torch.Tensor:
    """Counterfactual same-topic/same-length/same-decorator mirror margin loss (kills style confounders!)."""
    zp = logit_ai_pos.float()
    zn = logit_ai_neg.float()
    pw = pair_weights.float()
    l_margin = F.softplus(margin - (zp - zn))
    l_anchor = 0.25 * (F.relu(0.6 - zp).pow(2) + F.relu(zn + 0.6).pow(2))
    return ((l_margin + l_anchor) * pw).mean()


# ---------------------------------------------------------------------------
# 6. Upgraded v1.4.0 Corpus Builder (Sibling + 57.5K Delta + 15.9K CF + EN/JA!)
# ---------------------------------------------------------------------------

def _extract_origin_v140(row: dict[str, Any]) -> str:
    bm = row.get("bucket_meta") or {}
    return str(row.get("origin_6way") or bm.get("origin_6way") or bm.get("origin") or "")


def _is_pure_text_ai_pos_v140(
    source_tag: str,
    origin_6way: str,
    text_only_verifiable: Any = None,
    is_synthetic: bool = False,
) -> tuple[bool, float, bool]:
    """Return `(admit_as_binary_ai_pos, weight, add_to_distill_pool)`."""
    st = str(source_tag or "")
    stl = st.lower()
    orig = str(origin_6way or "")

    # Explicitly allow the 1,200 targeted delta_d5 Q2 (AI=1, Promo=1) quadrant-balancing anchors into training
    # (never into the real-only distillation pool!) so the teacher learns AI & Promo are independent axes!
    if "delta_q2_dual_pos" in stl:
        return True, 0.85, False

    if is_synthetic or "synth" in stl:
        return False, 0.0, False

    if text_only_verifiable is False:
        return False, 0.0, True

    # Newly rescued R1c pure-text verifiable real AI (1,422 rows: 273 short stealth + 87 Q2 + 1,062 textverif)
    if "r1c_rescued" in stl:
        if "short_stealth" in stl or "q2_dual_pos" in stl:
            return True, 1.40, True
        return True, 1.25, True

    # Newly merged v4B / v123ext real no-period short stealth AI & micro-short real AI
    if "short_stealth_ai_noperiod" in stl or "micro_short_real_ai" in stl:
        return True, 1.45, True
    if "q2_dual_pos" in stl or "verified_real_ai" in stl:
        return True, 1.35, True

    if orig in NOISY_BEHAVIOR_ORIGINS_V140:
        return False, 0.0, True
    if st in NOISY_SCRIPTBOT_SOURCES_V140 and orig == "Script-Plugin-Bot":
        return False, 0.0, True
    if st == "sibling_short_stealth_ai_15_60":
        return (orig == "Autonomous-Persona-Agent"), (1.35 if orig == "Autonomous-Persona-Agent" else 0.0), True
    if st == "sibling_neon_committee_3r_gold":
        return True, 1.25, True
    if st == "sibling_god_tier_calibrated_gold":
        return (orig == "Autonomous-Persona-Agent"), (1.15 if orig == "Autonomous-Persona-Agent" else 0.0), True
    if st == "v12_gold_v2_train":
        return True, 1.30, True
    if st in ("v121_wave3_train", "v123_delta"):
        return True, 1.25, True
    if st in ("v11_wave2_gold", "v11_wave1_gold", "v11_wave25_gold", "v11_wave2_longform"):
        if orig == "Autonomous-Persona-Agent":
            return True, 1.20, True
        if orig == "Script-Plugin-Bot":
            return True, 0.70, True
        return True, 0.90, True
    if orig == "Autonomous-Persona-Agent":
        return True, 1.15, True
    if orig == "Script-Plugin-Bot":
        return True, 0.65, True
    return True, 1.00, True


def _load_v04_multilingual_gold_v140(
    v04_path: Path | None,
    blocked_hashes: set[str],
    smoke_test: bool,
    cap_each: int = 1200,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Harvest clean EN/JA multilingual AI+ and Human- gold anchors from `v04-merged-1m4-r3.jsonl.gz`."""
    if v04_path is None or not v04_path.is_file():
        return [], []
    ml_pos: list[dict[str, Any]] = []
    ml_neg: list[dict[str, Any]] = []
    cap = 40 if smoke_test else int(cap_each)
    seen: set[str] = set()
    opener = gzip.open if v04_path.suffix == ".gz" else open
    with opener(v04_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("split") != "train":
                continue
            co = str(r.get("cohort", ""))
            if co not in ("inkling_wk", "v04_1m", "bili"):
                continue
            txt = str(r.get("text", "")).strip()
            if len(txt) < 12:
                continue
            head = txt[:200]
            n_cjk = sum(0x4E00 <= ord(c) <= 0x9FFF for c in head)
            n_lat = sum(("a" <= c <= "z") or ("A" <= c <= "Z") for c in head)
            n_kana = sum(0x3040 <= ord(c) <= 0x30FF for c in head)
            is_en = (co == "inkling_wk") or (n_lat >= 25 and n_cjk <= 5)
            is_ja = (n_kana >= 6 and n_cjk <= 20)
            if not (is_en or is_ja):
                continue
            h = text_sha16_v140(txt)
            if h in blocked_hashes or h in seen:
                continue
            seen.add(h)
            sc = float(r.get("score", 50.0))
            sid = str(r.get("sample_id") or f"v04ml_{h}")
            if sc >= 70.0 and len(ml_pos) < cap:
                ml_pos.append({
                    "sid": sid, "text": txt, "y_ai": 1.0, "has_ai": 1.0,
                    "y_pr": 0.0, "has_pr": 0.0, "sub4": -1,
                    "w_ai": 1.15, "w_pr": 0.0, "hard_neg": 0.0, "era5": -1,
                    "source_tag": f"v04_ml_{co}", "is_ml_en": bool(is_en), "is_multilingual": True,
                })
            elif sc <= 25.0 and len(ml_neg) < cap:
                ml_neg.append({
                    "sid": sid, "text": txt, "y_ai": 0.0, "has_ai": 1.0,
                    "y_pr": 0.0, "has_pr": 0.0, "sub4": -1,
                    "w_ai": 1.15, "w_pr": 0.0, "hard_neg": 1.0, "era5": -1,
                    "source_tag": f"v04_ml_{co}", "is_ml_en": bool(is_en), "is_multilingual": True,
                })
            if len(ml_pos) >= cap and len(ml_neg) >= cap:
                break
    return ml_pos, ml_neg


def build_v140_superteacher_corpus(
    sibling_dir: Path | None,
    delta_dir: Path | None,
    gold_v2_dir: Path | None,
    wave3_dir: Path | None,
    v11_dir: Path | None,
    v04_data_path: Path | None = None,
    extra_delta_dirs: list[Path] | None = None,
    smoke_test: bool = False,
    seed: int = 20261128,
) -> dict[str, Any]:
    """Assemble strictly purified, zero-leakage v1.4.0 train / val / test / CF-pair / 250K distill pools."""
    rng = random.Random(seed)

    # 1. Load STRICT held-out validation & test sets FIRST to build an absolute dedup firewall!
    val_gold_v2: list[dict[str, Any]] = []
    test_gold_v2: list[dict[str, Any]] = []
    val_wave3: list[dict[str, Any]] = []
    val_promo_v11: list[dict[str, Any]] = []
    blocked_hashes: set[str] = set()

    def _register_holdout(rows: list[dict[str, Any]], dst: list[dict[str, Any]], src_name: str, limit: int | None = None) -> None:
        for r in rows[: (limit if limit is not None else len(rows))]:
            txt = str(r.get("text", "")).strip()
            if not txt:
                continue
            h = text_sha16_v140(txt)
            blocked_hashes.add(h)
            ai_sc = r.get("ai_score", r.get("score"))
            pr_sc = r.get("promo_score")
            has_ai = 1.0 if ai_sc is not None and (float(ai_sc) <= 30.0 or float(ai_sc) >= 70.0) else 0.0
            has_pr = 1.0 if pr_sc is not None and (float(pr_sc) <= 30.0 or float(pr_sc) >= 70.0) else 0.0
            dst.append({
                "sid": str(r.get("sample_id") or r.get("sid") or h),
                "text": txt,
                "y_ai": 1.0 if (ai_sc is not None and float(ai_sc) >= 70.0) else 0.0,
                "has_ai": has_ai,
                "y_pr": 1.0 if (pr_sc is not None and float(pr_sc) >= 70.0) else 0.0,
                "has_pr": has_pr,
                "sub4": SUB4_MAP_V140.get(str(r.get("promo_type", "none")), -1),
                "source_tag": src_name,
                "era5": sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts")),
            })

    if gold_v2_dir and gold_v2_dir.is_dir():
        _register_holdout(read_jsonl_any_v140(gold_v2_dir / "v12_gold_v2_val.jsonl.gz"), val_gold_v2, "v12_gold_v2_val", 64 if smoke_test else None)
        _register_holdout(read_jsonl_any_v140(gold_v2_dir / "v12_gold_v2_test.jsonl.gz"), test_gold_v2, "v12_gold_v2_test", 64 if smoke_test else None)
    if wave3_dir and wave3_dir.is_dir():
        _register_holdout(read_jsonl_any_v140(wave3_dir / "v121_wave3_val.jsonl.gz"), val_wave3, "v121_wave3_val", 64 if smoke_test else None)
    if v11_dir and v11_dir.is_dir():
        for fn in ("v11_wave2_dualaxis_gold_val.jsonl.gz", "v11_wave2_dualaxis_gold_cert.jsonl.gz"):
            _register_holdout(read_jsonl_any_v140(v11_dir / fn), val_promo_v11, fn.replace(".jsonl.gz", ""), 48 if smoke_test else None)

    # 2. Load Sibling + v123-sibling-delta packs via verified v1.2.3 refinery
    sib_pack = load_sibling_and_delta_v123(sibling_dir, delta_dir, smoke_test=smoke_test)
    if extra_delta_dirs:
        for ed in extra_delta_dirs:
            if ed and ed.is_dir():
                ep = load_sibling_and_delta_v123(None, ed, smoke_test=smoke_test)
                sib_pack["dual_train_rows"].extend(ep["dual_train_rows"])
                sib_pack["ai_only_real_rows"].extend(ep["ai_only_real_rows"])
                sib_pack["cf_mirror_pairs"].extend(ep["cf_mirror_pairs"])

    for r in sib_pack["god_tier_500_holdout"]:
        blocked_hashes.add(text_sha16_v140(str(r.get("text", ""))))

    # 3. Prepare purified training pools + counterfactual pairs + full real distillation pool
    train_ai_pos: list[dict[str, Any]] = []
    train_hu_natural: list[dict[str, Any]] = []
    train_hu_hard: list[dict[str, Any]] = []
    train_promo_extra: list[dict[str, Any]] = []
    distill_pool_by_hash: dict[str, dict[str, Any]] = {}

    stats: dict[str, int] = {
        "purged_noisy_ai_pos": 0,
        "purged_synth_ai_pos": 0,
        "blocked_holdout_dedup": 0,
        "admitted_pure_ai_pos": 0,
        "admitted_rescued_ai_pos": 0,
        "admitted_q2_dual_pos": 0,
        "admitted_hu_natural": 0,
        "admitted_hu_hard": 0,
        "admitted_promo_pos": 0,
        "admitted_ml_train": 0,
        "admitted_cf_pairs": 0,
    }

    def _add_distill(
        sid: str,
        txt: str,
        src: str,
        era5: int,
        orig: str,
        gold_ai: float | None,
        gold_pr: float | None,
        is_ml: bool = False,
    ) -> None:
        h = text_sha16_v140(txt)
        if h in blocked_hashes or h in distill_pool_by_hash:
            return
        distill_pool_by_hash[h] = {
            "sid": sid,
            "sha16": h,
            "text": txt,
            "source_tag": src,
            "era5": int(era5),
            "origin_6way": orig,
            "gold_ai": gold_ai,
            "gold_pr": gold_pr,
            "is_multilingual": bool(is_ml),
        }

    def _ingest_row(
        txt: str,
        sid: str,
        src: str,
        orig: str,
        era5: int,
        ai_sc: float | None,
        pr_sc: float | None,
        ptype: str = "none",
        is_hard_neg_hint: bool = False,
        is_pre_llm: bool = False,
        text_only_verifiable: Any = None,
        is_synthetic: bool = False,
        add_to_distill: bool = True,
    ) -> None:
        txt = str(txt or "").strip()
        if len(txt) < 2:
            return
        h = text_sha16_v140(txt)
        if h in blocked_hashes:
            stats["blocked_holdout_dedup"] += 1
            return

        g_ai = float(ai_sc) if ai_sc is not None else None
        g_pr = float(pr_sc) if pr_sc is not None else None
        stl = str(src or "").lower()

        has_pr = 1.0 if (g_pr is not None and (g_pr <= 25.0 or g_pr >= 70.0)) else 0.0
        y_pr = 1.0 if (g_pr is not None and g_pr >= 70.0) else 0.0
        s4 = SUB4_MAP_V140.get(str(ptype or "none"), -1) if (has_pr > 0.5 and y_pr > 0.5) else -1

        # Case A: Candidate AI Positive
        if g_ai is not None and g_ai >= 70.0:
            admit, w_a, allow_dist = _is_pure_text_ai_pos_v140(src, orig, text_only_verifiable, is_synthetic)
            if add_to_distill and allow_dist and not is_synthetic and ("synth" not in stl):
                _add_distill(sid, txt, src, era5, orig, g_ai, g_pr)
            if not admit:
                if is_synthetic or "synth" in stl:
                    stats["purged_synth_ai_pos"] += 1
                else:
                    stats["purged_noisy_ai_pos"] += 1
                if has_pr > 0.5 and y_pr > 0.5 and not is_synthetic:
                    train_promo_extra.append({
                        "sid": sid, "text": txt, "y_ai": 0.0, "has_ai": 0.0,
                        "y_pr": 1.0, "has_pr": 1.0, "sub4": s4,
                        "w_ai": 0.0, "w_pr": 1.5, "hard_neg": 0.0, "era5": era5,
                        "source_tag": src, "is_ml_en": False,
                    })
                    stats["admitted_promo_pos"] += 1
                return
            train_ai_pos.append({
                "sid": sid, "text": txt, "y_ai": 1.0, "has_ai": 1.0,
                "y_pr": y_pr, "has_pr": has_pr, "sub4": s4,
                "w_ai": w_a, "w_pr": 1.55 if y_pr > 0.5 else 0.70, "hard_neg": 0.0, "era5": era5,
                "source_tag": src, "is_ml_en": False,
            })
            stats["admitted_pure_ai_pos"] += 1
            if "r1c_rescued" in stl:
                stats["admitted_rescued_ai_pos"] += 1
            if has_pr > 0.5 and y_pr > 0.5:
                stats["admitted_q2_dual_pos"] += 1
                stats["admitted_promo_pos"] += 1
            return

        # Real-only distillation pool admission for non-AI-pos rows
        if add_to_distill and not is_synthetic and ("synth" not in stl):
            _add_distill(sid, txt, src, era5, orig, g_ai, g_pr)

        # Case B: Verified Human Negative (`ai_score <= 25`)
        if g_ai is not None and g_ai <= 25.0:
            is_q4_promo = bool(has_pr > 0.5 and y_pr > 0.5)
            is_old_gt1y = bool(is_pre_llm or era5 in (0, 1, 2) or "oldvideo" in stl or "pre_llm" in stl)
            is_decor = bool("decorator" in stl)
            is_brand_hn = bool("organic_brand" in stl)
            is_hard = bool(
                is_hard_neg_hint or is_q4_promo or is_old_gt1y or is_decor or is_brand_hn
                or "hard_human" in stl or "trap" in stl or "longform" in stl or len(txt) >= 110
            )
            w_a = (
                1.65 if is_q4_promo
                else (1.45 if (is_pre_llm or "oldvideo" in stl or is_decor or "trap" in stl)
                      else (1.20 if is_hard else 1.0))
            )
            if is_synthetic:
                w_a *= 0.85
            item = {
                "sid": sid, "text": txt, "y_ai": 0.0, "has_ai": 1.0,
                "y_pr": y_pr, "has_pr": has_pr, "sub4": s4,
                "w_ai": w_a,
                "w_pr": 1.85 if is_q4_promo else (1.40 if is_brand_hn else 0.75),
                "hard_neg": 1.0 if is_hard else 0.0, "era5": era5,
                "source_tag": src, "is_ml_en": False,
            }
            if is_hard:
                train_hu_hard.append(item)
                stats["admitted_hu_hard"] += 1
            else:
                train_hu_natural.append(item)
                stats["admitted_hu_natural"] += 1
            if is_q4_promo:
                stats["admitted_promo_pos"] += 1
            return

        # Case C: Promo-only supervision
        if has_pr > 0.5 and not is_synthetic:
            train_promo_extra.append({
                "sid": sid, "text": txt, "y_ai": 0.0, "has_ai": 0.0,
                "y_pr": y_pr, "has_pr": 1.0, "sub4": s4,
                "w_ai": 0.0, "w_pr": 1.6 if y_pr > 0.5 else 1.0, "hard_neg": 0.0, "era5": era5,
                "source_tag": src, "is_ml_en": False,
            })
            if y_pr > 0.5:
                stats["admitted_promo_pos"] += 1

    # 4. Ingest Sibling + Delta Dual-Axis Rows (`140,513` rows!)
    for r in sib_pack["dual_train_rows"]:
        bm = r.get("bucket_meta") or {}
        src = str(r.get("source_tag") or ("v123_delta" if r.get("is_delta") else "sibling_dual"))
        orig = _extract_origin_v140(r)
        ai_sc = float(r.get("ai_score", 0.0)) if r.get("has_ai", True) else None
        pr_sc = float(r.get("promo_score", 0.0)) if r.get("has_promo", True) else None
        is_pre = str(r.get("era_bucket", "")).lower().startswith("pre")
        is_syn = bool(bm.get("is_synthetic", False)) or ("synth" in src.lower())
        e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), is_pre_llm=is_pre, score_source=src, ai_score_0_100=ai_sc)
        _ingest_row(
            str(r.get("text", "")), str(r.get("sample_id") or r.get("sid") or ""), src, orig, e5,
            ai_sc, pr_sc, str(r.get("promo_type", "none")),
            is_pre_llm=is_pre,
            text_only_verifiable=r.get("text_only_verifiable", bm.get("text_only_verifiable")),
            is_synthetic=is_syn,
        )

    # 5. Ingest Sibling + Delta AI-Only Real Rows (`82,154` rows!)
    for r in sib_pack["ai_only_real_rows"]:
        bm = r.get("bucket_meta") or {}
        src = str(r.get("source_tag") or "sibling_ai_only")
        orig = _extract_origin_v140(r)
        ai_sc = float(r.get("ai_score", 0.0))
        is_pre = str(r.get("era_bucket", "")).lower().startswith("pre")
        is_syn = bool(bm.get("is_synthetic", False)) or ("synth" in src.lower())
        e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), is_pre_llm=is_pre, score_source=src, ai_score_0_100=ai_sc)
        _ingest_row(
            str(r.get("text", "")), str(r.get("sample_id") or ""), src, orig, e5, ai_sc, None,
            is_pre_llm=is_pre,
            text_only_verifiable=r.get("text_only_verifiable", bm.get("text_only_verifiable")),
            is_synthetic=is_syn,
        )

    # 6. Ingest `v12_gold_v2_train` + `v121_wave3_train` + `v11` Gold/UOT Promo
    if gold_v2_dir and gold_v2_dir.is_dir():
        for r in read_jsonl_any_v140(gold_v2_dir / "v12_gold_v2_train.jsonl.gz")[: (96 if smoke_test else 999999)]:
            ai_sc = float(r.get("ai_score", r.get("score", 0.0)))
            pr_sc = float(r["promo_score"]) if r.get("promo_score") is not None else None
            e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=ai_sc)
            _ingest_row(str(r.get("text", "")), str(r.get("sample_id") or r.get("sid") or ""), "v12_gold_v2_train", _extract_origin_v140(r), e5, ai_sc, pr_sc, str(r.get("promo_type", "none")))
    if wave3_dir and wave3_dir.is_dir():
        for r in read_jsonl_any_v140(wave3_dir / "v121_wave3_train.jsonl.gz")[: (96 if smoke_test else 999999)]:
            ai_sc = float(r.get("ai_score", 0.0))
            pr_sc = float(r.get("promo_score", 0.0))
            e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=ai_sc)
            _ingest_row(str(r.get("text", "")), str(r.get("sample_id") or r.get("sid") or ""), "v121_wave3_train", _extract_origin_v140(r), e5, ai_sc, pr_sc, str(r.get("promo_type", "none")))
    if v11_dir and v11_dir.is_dir():
        for fn, tag in [
            ("v11_wave1_dualaxis_gold_verified.jsonl.gz", "v11_wave1_gold"),
            ("v11_wave2_dualaxis_gold_verified.jsonl.gz", "v11_wave2_gold"),
            ("v11_wave25_dualaxis_gold_verified.jsonl.gz", "v11_wave25_gold"),
            ("v11_wave2_longform_gold_1200.jsonl.gz", "v11_wave2_longform"),
        ]:
            for r in read_jsonl_any_v140(v11_dir / fn)[: (64 if smoke_test else 999999)]:
                if r.get("split", "train") != "train":
                    continue
                ai_sc = float(r.get("ai_score", 0.0))
                pr_sc = float(r.get("promo_score", 0.0))
                e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=ai_sc)
                _ingest_row(str(r.get("text", "")), str(r.get("sample_id") or r.get("sid") or ""), tag, _extract_origin_v140(r), e5, ai_sc, pr_sc, str(r.get("promo_type", "none")))
        for r in read_jsonl_any_v140(v11_dir / "v11_uot_wild_promo_pos.jsonl")[: (32 if smoke_test else 999999)]:
            _ingest_row(str(r.get("text", "")), str(r.get("sample_id") or ""), "v11_uot_wild_promo", "", -1, None, 95.0, str(r.get("promo_type", "soft_shill")))
        for r in read_jsonl_any_v140(v11_dir / "v11_bge_knn_promo_hard_negatives_2k_uot_gated.jsonl")[: (48 if smoke_test else 999999)]:
            _ingest_row(str(r.get("text", "")), str(r.get("sample_id") or ""), "v11_promo_hn", "", -1, None, 0.0, "none")

    # 7. Clean EN/JA Multilingual Gold Anchors (85% train + 15% held-out `val_multilingual`!)
    val_multilingual: list[dict[str, Any]] = []
    ml_pos, ml_neg = _load_v04_multilingual_gold_v140(v04_data_path, blocked_hashes, smoke_test=smoke_test, cap_each=1200)
    rng.shuffle(ml_pos)
    rng.shuffle(ml_neg)
    n_ml_val_p = max(4, int(len(ml_pos) * 0.15)) if len(ml_pos) >= 8 else 0
    n_ml_val_n = max(4, int(len(ml_neg) * 0.15)) if len(ml_neg) >= 8 else 0
    for it in ml_pos[:n_ml_val_p] + ml_neg[:n_ml_val_n]:
        blocked_hashes.add(text_sha16_v140(it["text"]))
        val_multilingual.append(it)
    for it in ml_pos[n_ml_val_p:]:
        train_ai_pos.append(it)
        _add_distill(it["sid"], it["text"], it["source_tag"], -1, "", 95.0, None, is_ml=True)
        stats["admitted_ml_train"] += 1
    for it in ml_neg[n_ml_val_n:]:
        train_hu_hard.append(it)
        _add_distill(it["sid"], it["text"], it["source_tag"], -1, "", 5.0, None, is_ml=True)
        stats["admitted_ml_train"] += 1

    # 8. Filter Counterfactual Mirror Pairs (`15,903` pairs, incl. `1,200` micro-short & `500` decorator 1:1!)
    cf_pairs_micro: list[dict[str, Any]] = []
    cf_pairs_decor: list[dict[str, Any]] = []
    cf_pairs_general: list[dict[str, Any]] = []
    for pr in sib_pack["cf_mirror_pairs"]:
        at = str(pr.get("ai_text", "")).strip()
        ht = str(pr.get("human_text", "")).strip()
        if len(at) < 4 or len(ht) < 4:
            continue
        if text_sha16_v140(at) in blocked_hashes or text_sha16_v140(ht) in blocked_hashes:
            continue
        mode = str(pr.get("mode", ""))
        rec = {
            "ai_text": at,
            "human_text": ht,
            "pair_weight": float(pr.get("pair_weight", 1.15)),
            "mode": mode,
        }
        if pr.get("is_micro_short") or "micro" in mode.lower():
            cf_pairs_micro.append(rec)
        elif "decor" in mode.lower():
            cf_pairs_decor.append(rec)
        else:
            cf_pairs_general.append(rec)
    stats["admitted_cf_pairs"] = len(cf_pairs_micro) + len(cf_pairs_decor) + len(cf_pairs_general)

    # 9. Carve dedicated held-out `val_old_video_neg` (`>1y` & `pre_llm` human FPR guard, 2,000 rows!)
    old_cand = [
        i for i, it in enumerate(train_hu_hard)
        if it["era5"] in (0, 1, 2) or "oldvideo" in it["source_tag"].lower()
    ]
    rng.shuffle(old_cand)
    n_old_val = min(48 if smoke_test else 2000, len(old_cand) // 5)
    old_val_idx = set(old_cand[:n_old_val])
    val_old_video_neg = [train_hu_hard[i] for i in sorted(old_val_idx)]
    train_hu_hard = [it for i, it in enumerate(train_hu_hard) if i not in old_val_idx]

    # 10. Real-wild unlabeled Bilibili comments from `v04` added STRICTLY to `distill_pool`
    if v04_data_path and v04_data_path.is_file():
        max_v04_distill = 200 if smoke_test else 45000
        added_v04 = 0
        opener = gzip.open if v04_data_path.suffix == ".gz" else open
        with opener(v04_data_path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if added_v04 >= max_v04_distill:
                    break
                if not line.strip():
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                src = str(o.get("score_source") or o.get("source") or "").lower()
                if any(k in src for k in ("synth", "plugin", "deepseek", "qwen", "inkling")):
                    continue
                txt = str(o.get("text", "")).strip()
                if len(txt) < 4:
                    continue
                e5 = sanitize_era5_v123(o.get("era_bucket"), o.get("created_ts"), is_pre_llm=bool(o.get("is_pre_llm")))
                _add_distill(str(o.get("sample_id") or f"v04_{added_v04}"), txt, "v04_real_wild_unlabeled", e5, "", None, None)
                added_v04 += 1

    # 11. Diagnostic `god_tier_500_holdout` (uses explicit `text_only_verifiable` from delta_d1!)
    diag_god500 = []
    for r in sib_pack["god_tier_500_holdout"][: (48 if smoke_test else 999)]:
        bm = r.get("bucket_meta") or {}
        orig = _extract_origin_v140(r)
        tv = r.get("text_only_verifiable", bm.get("text_only_verifiable"))
        if tv is None:
            tv = orig not in NOISY_BEHAVIOR_ORIGINS_V140 and orig != "Script-Plugin-Bot"
        diag_god500.append({
            "sid": str(r.get("sample_id") or ""),
            "text": str(r.get("text", "")),
            "y_ai": 1.0 if float(r.get("ai_score", 0.0)) >= 70.0 else 0.0,
            "is_text_verifiable": bool(tv),
        })

    def _dedup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen_h: set[str] = set()
        out_r: list[dict[str, Any]] = []
        for x in rows:
            h = text_sha16_v140(x["text"])
            if h in seen_h:
                continue
            seen_h.add(h)
            out_r.append(x)
        return out_r

    train_ai_pos = _dedup(train_ai_pos)
    train_hu_natural = _dedup(train_hu_natural)
    train_hu_hard = _dedup(train_hu_hard)
    train_promo_extra = _dedup(train_promo_extra)

    return {
        "train_ai_pos": train_ai_pos,
        "train_hu_natural": train_hu_natural,
        "train_hu_hard": train_hu_hard,
        "train_promo_extra": train_promo_extra,
        "cf_pairs_micro": cf_pairs_micro,
        "cf_pairs_decor": cf_pairs_decor,
        "cf_pairs_general": cf_pairs_general,
        "val_gold_v2": val_gold_v2,
        "test_gold_v2": test_gold_v2,
        "val_wave3": val_wave3,
        "val_promo_v11": val_promo_v11,
        "val_old_video_neg": val_old_video_neg,
        "val_multilingual": val_multilingual,
        "diag_god500": diag_god500,
        "distill_pool": list(distill_pool_by_hash.values()),
        "stats": {
            **stats,
            "dedup_train_ai_pos": len(train_ai_pos),
            "dedup_train_hu_natural": len(train_hu_natural),
            "dedup_train_hu_hard": len(train_hu_hard),
            "dedup_train_promo_extra": len(train_promo_extra),
            "cf_pairs_micro": len(cf_pairs_micro),
            "cf_pairs_decor": len(cf_pairs_decor),
            "cf_pairs_general": len(cf_pairs_general),
            "val_gold_v2": len(val_gold_v2),
            "test_gold_v2": len(test_gold_v2),
            "val_wave3": len(val_wave3),
            "val_old_video_neg": len(val_old_video_neg),
            "val_multilingual": len(val_multilingual),
            "distill_pool_total": len(distill_pool_by_hash),
        },
    }


def sample_antidilution_epoch_v140(
    corpus: dict[str, Any],
    rng: random.Random,
    neg_per_pos: float = 4.2,
    natural_frac: float = 0.55,
    smoke_test: bool = False,
) -> list[dict[str, Any]]:
    """Build one anti-dilution epoch with boosted >1y old-video + Q4 promo + multilingual hard-neg coverage."""
    pos = list(corpus["train_ai_pos"])
    hu_nat = list(corpus["train_hu_natural"])
    hu_hd = list(corpus["train_hu_hard"])
    pr_ex = list(corpus["train_promo_extra"])

    if smoke_test:
        out = pos[:32] + hu_nat[:40] + hu_hd[:40] + pr_ex[:16]
        rng.shuffle(out)
        return out

    n_neg_target = max(len(pos) * 2, int(round(len(pos) * neg_per_pos)))
    n_nat = int(round(n_neg_target * natural_frac))
    n_hd = max(1, n_neg_target - n_nat)

    def _draw(pool: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        if not pool or k <= 0:
            return []
        if len(pool) >= k:
            return rng.sample(pool, k)
        return list(pool) + rng.choices(pool, k=k - len(pool))

    # Always include ALL Q4 human-promo firewall rows + ALL EN/JA multilingual human negatives!
    must_hd = [
        x for x in hu_hd
        if (x.get("has_pr", 0.0) > 0.5 and x.get("y_pr", 0.0) > 0.5) or x.get("is_multilingual")
    ]
    other_hd = [
        x for x in hu_hd
        if not ((x.get("has_pr", 0.0) > 0.5 and x.get("y_pr", 0.0) > 0.5) or x.get("is_multilingual"))
    ]
    chosen_hd = must_hd + _draw(other_hd, max(0, n_hd - len(must_hd)))
    chosen_nat = _draw(hu_nat, n_nat)
    chosen_pr = _draw(pr_ex, min(len(pr_ex), max(1500, len(pos) // 3)))

    epoch_rows = pos + chosen_nat + chosen_hd + chosen_pr
    rng.shuffle(epoch_rows)
    return epoch_rows


def iter_length_bucketed_batches_v140(
    rows: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
    mega_mult: int = 16,
) -> list[list[dict[str, Any]]]:
    """Mega-batch length bucketing (~3x faster 7B/8B training with zero batch-order bias!)."""
    shuffled = list(rows)
    rng.shuffle(shuffled)
    mega_sz = max(batch_size, batch_size * mega_mult)
    batches: list[list[dict[str, Any]]] = []
    for st in range(0, len(shuffled), mega_sz):
        chunk = shuffled[st : st + mega_sz]
        chunk.sort(key=lambda r: len(r["text"]))
        for b_st in range(0, len(chunk), batch_size):
            batches.append(chunk[b_st : b_st + batch_size])
    rng.shuffle(batches)
    return batches


def sample_cf_mirror_batch_v140(
    corpus: dict[str, Any],
    k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample `k` counterfactual pairs prioritizing micro-short (35%), decorator (20%), general (45%)."""
    micro = corpus.get("cf_pairs_micro") or []
    decor = corpus.get("cf_pairs_decor") or []
    gen = corpus.get("cf_pairs_general") or []
    all_p = micro + decor + gen
    if not all_p:
        return []
    out: list[dict[str, Any]] = []
    for _ in range(k):
        u = rng.random()
        if u < 0.35 and micro:
            out.append(rng.choice(micro))
        elif u < 0.55 and decor:
            out.append(rng.choice(decor))
        elif gen:
            out.append(rng.choice(gen))
        else:
            out.append(rng.choice(all_p))
    return out


# ---------------------------------------------------------------------------
# 7. Calibration + Strict Zero-PII & LoRA-Merge-Parity Gates
# ---------------------------------------------------------------------------

def fit_scalar_temperature_v140(logits: np.ndarray, y: np.ndarray) -> float:
    if len(logits) < 10 or y.sum() == 0 or y.sum() == len(y):
        return 1.0
    best_t, best_nll = 1.0, 1e9
    for t in np.linspace(0.4, 3.0, 53):
        z = np.clip(logits / t, -30.0, 30.0)
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1.0 - 1e-6)
        nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return round(best_t, 4)


@torch.no_grad()
def run_v140_superteacher_zero_pii_gate(
    model: UniversalDualAxisSuperTeacherV140,
    tokenizer: Any,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Strict Zero-PII verification (`V140_ST_C1`): swapping @username & contact IDs MUST yield diff === 0.0!"""
    model.eval()
    ta = ["回复 @测试用户A : 关注工种号交流2群 168688884 助理老师：xhs54199 领取资料[doge]"]
    tb = ["回复 @RandomName99 : 关注工种号交流2群 999888777 助理老师：abc98765 领取资料[doge]"]
    ia, ma, pa = encode_universal_v140(tokenizer, ta, arch_type=model.arch_type)
    ib, mb, pb = encode_universal_v140(tokenizer, tb, arch_type=model.arch_type)
    use_amp = device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        oa = model(ia.to(device), ma.to(device), pa.to(device))
        ob = model(ib.to(device), mb.to(device), pb.to(device))
    d_ai = float((oa["logit_ai"].float() - ob["logit_ai"].float()).abs().max().item())
    d_pr = float((oa["logit_promo"].float() - ob["logit_promo"].float()).abs().max().item())
    mx = max(d_ai, d_pr)
    return {"max_abs_diff": mx, "ai_diff": d_ai, "promo_diff": d_pr, "pass": mx == 0.0}
