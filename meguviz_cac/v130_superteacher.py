"""MeguvizCAC v1.3.0-ApexKD Super-Teacher Core Library.

Builds a strictly Zero-PII (`diff === 0.0`), Dual-Axis (AI + Commercial Promo), 3-Model
Heterogeneous 24L Chinese Pretrained Super-Teacher Ensemble:
  1. `hfl/chinese-electra-180g-large-discriminator` (RTD pretraining -> micro-cadence anomaly detection)
  2. `hfl/chinese-macbert-large` (MLM-as-correction -> semantic & stylistic fluency)
  3. `BAAI/bge-large-zh-v1.5` (contrastive Chinese encoder -> subtle register separation)

Key anti-dilution & precision-tail engineering (zero OjinjinBench contact!):
  - Strict Zero-PII canonicalization (`canonicalize_zero_pii_v11`) BEFORE tokenization (`C1 === 0.0`).
  - Head-180 + Tail-74 token preservation (`max_len = 256`) so both hooks & AI closings survive.
  - Label purification: removes noisy behavior-only labels (`Coordinated-Latent-Cluster`,
    `Human-Assisted`, `god_tier/conflict Script-Plugin-Bot`, midband `30-70`, & `v0.4` synth) from
    binary `y_ai=1` supervision, routing them to the unlabeled committee distillation pool instead.
  - Anti-dilution 65% natural-traffic / 35% hard-negative real humans across all 5 eras
    (`pre_llm`, `2023`, `2024`, `2025_gt1y`, `2026`), heavily anchoring `>1y` old-video long
    humans, organic brand/promo hard negatives, and Q4 human-commercial-promo firewall anchors.
  - Online 35% punctuation-jitter AI augmentation (breaks the `ends_period` & `——/「」` shortcut!).
  - Differentiable Top-Tail Pairwise Margin Loss directly maximizing `PR-AUC`, `R@P95`, & `R@P99`.
"""

from __future__ import annotations

import gzip
import hashlib
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
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel

from .v11_features import canonicalize_zero_pii_v11
from .v123_data_refinery import load_sibling_and_delta_v123, sanitize_era5_v123

SUPERTEACHER_BACKBONES_V130: dict[str, str] = {
    "electra_large": "hfl/chinese-electra-180g-large-discriminator",
    "macbert_large": "hfl/chinese-macbert-large",
    "bge_large": "BAAI/bge-large-zh-v1.5",
}

SUB4_MAP_V130: dict[str, int] = {
    "hard_contact": 0,
    "affiliate_link": 1,
    "soft_shill": 2,
    "self_promo": 3,
}

# Origins that are frequently behavior/cluster-labeled or human-written-with-light-edit
# and therefore MUST NOT enter binary pure-text `y_ai = 1.0` supervision!
NOISY_BEHAVIOR_ORIGINS_V130 = frozenset({
    "Coordinated-Latent-Cluster",
    "Human-Assisted",
})
NOISY_SCRIPTBOT_SOURCES_V130 = frozenset({
    "sibling_god_tier_calibrated_gold",
    "sibling_conflict_adjudicated",
})


def text_sha16_v130(text: str) -> str:
    norm = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()[:16]


def read_jsonl_any_v130(path: Path) -> list[dict[str, Any]]:
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
# 1. Online Punctuation-Shortcut Breaker (35% on AI positives during training)
# ---------------------------------------------------------------------------

_RE_TRAIL_PERIOD = re.compile(r"[。\.]+(\s*)$")
_RE_EMDASH = re.compile(r"——|—|--")
_RE_CORNER_Q = re.compile(r"[「」『』]")


def jitter_ai_punctuation_v130(text: str, rng: random.Random, prob: float = 0.35) -> str:
    """Break formal-punctuation shortcuts (`。`, `——`, `「」`) on AI positives during training."""
    if not text or rng.random() >= prob:
        return text
    s = text
    if rng.random() < 0.75:
        s = _RE_TRAIL_PERIOD.sub(r"\1", s)
    if rng.random() < 0.60 and _RE_EMDASH.search(s):
        repl = rng.choice(["，", " ", "，"])
        s = _RE_EMDASH.sub(repl, s)
    if rng.random() < 0.55 and _RE_CORNER_Q.search(s):
        s = _RE_CORNER_Q.sub("", s)
    return s.strip() or text


# ---------------------------------------------------------------------------
# 2. Strict Zero-PII + Head-180 + Tail-74 Tokenization (`max_len = 256`)
# ---------------------------------------------------------------------------

def encode_head_tail_hf_v130(
    tokenizer: Any,
    texts: list[str],
    max_len: int = 256,
    head_len: int = 180,
    already_canonical: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode batch with strict `canonicalize_zero_pii_v11` + Head+Tail truncation (`C1 === 0.0`)."""
    cls_id = int(getattr(tokenizer, "cls_token_id", None) or 101)
    sep_id = int(getattr(tokenizer, "sep_token_id", None) or 102)
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or 0)
    body_budget = max(8, int(max_len) - 2)
    h_cap = min(int(head_len), body_budget - 4)
    t_cap = body_budget - h_cap

    clean_texts: list[str] = []
    for t in texts:
        if already_canonical:
            clean_texts.append(str(t))
        else:
            cl, _ = canonicalize_zero_pii_v11(str(t))
            clean_texts.append(cl)

    enc = tokenizer(
        clean_texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    seqs: list[list[int]] = []
    for raw_ids in enc["input_ids"]:
        if len(raw_ids) > body_budget:
            body = raw_ids[:h_cap] + raw_ids[-t_cap:]
        else:
            body = raw_ids
        seqs.append([cls_id] + body + [sep_id])

    ml = max(4, max(len(s) for s in seqs))
    ids = torch.full((len(seqs), ml), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), ml), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask


# ---------------------------------------------------------------------------
# 3. Dual-Axis 3-Stat Pooled Pretrained Super-Teacher Model
# ---------------------------------------------------------------------------

class DualAxisSuperTeacherNet(nn.Module):
    """24L Chinese Pretrained Encoder + Masked 3-Stat Pool (Mean+Max+Attn) + Disentangled AI/Promo Heads."""

    def __init__(
        self,
        backbone_name_or_path: str,
        dropout: float = 0.15,
        smoke_tiny: bool = False,
        vocab_size: int = 21128,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone_name_or_path)
        self.smoke_tiny = bool(smoke_tiny)
        if smoke_tiny:
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
            cfg = AutoConfig.from_pretrained(backbone_name_or_path)
            # Disable pooler warning / save VRAM
            self.encoder = AutoModel.from_pretrained(backbone_name_or_path, config=cfg)
            h_dim = int(cfg.hidden_size)

        self.h_dim = h_dim
        self.attn_pool = nn.Sequential(
            nn.Linear(h_dim, h_dim // 4),
            nn.Tanh(),
            nn.Linear(h_dim // 4, 1),
        )
        pool_dim = h_dim * 3
        mid_dim = max(64, h_dim // 2)

        # Strictly separate AI and Commercial-Promo heads (structural zero-leakage between axes!)
        self.ai_head = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Dropout(dropout),
            nn.Linear(pool_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )
        self.promo_trunk = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Dropout(dropout),
            nn.Linear(pool_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.promo_head = nn.Linear(mid_dim, 1)
        self.promo_sub4_head = nn.Linear(mid_dim, 4)

    def pool_tokens(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        m = attention_mask.float()  # (B, L)
        m_u = m.unsqueeze(-1)  # (B, L, 1)
        denom = m_u.sum(dim=1).clamp_min(1.0)
        mean_p = (h * m_u).sum(dim=1) / denom

        # Masked max pool (safe in bf16/fp16)
        neg_inf = torch.finfo(h.dtype).min / 4.0
        h_masked = h.masked_fill(attention_mask.unsqueeze(-1) == 0, neg_inf)
        max_p = h_masked.max(dim=1).values

        # Masked learned attention pool
        a_logits = self.attn_pool(h).squeeze(-1)
        a_logits = a_logits.masked_fill(attention_mask == 0, -1e4)
        a_w = torch.softmax(a_logits.float(), dim=-1).to(h.dtype).unsqueeze(-1)
        attn_p = (h * a_w).sum(dim=1)

        return torch.cat([mean_p, max_p, attn_p], dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        pooled = self.pool_tokens(h, attention_mask)
        logit_ai = self.ai_head(pooled).squeeze(-1)
        pr_h = self.promo_trunk(pooled)
        logit_promo = self.promo_head(pr_h).squeeze(-1)
        logit_sub4 = self.promo_sub4_head(pr_h)
        return {
            "logit_ai": logit_ai,
            "logit_promo": logit_promo,
            "logit_sub4": logit_sub4,
        }


def build_llrd_param_groups_v130(
    model: DualAxisSuperTeacherNet,
    backbone_lr: float = 2.5e-5,
    head_lr: float = 2.5e-4,
    llrd: float = 0.90,
    weight_decay: float = 0.01,
) -> list[dict[str, Any]]:
    """Layer-wise LR decay (LLRD) for 24L Chinese Transformers + higher LR for freshly initialized heads."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm")
    # Detect number of transformer layers
    num_layers = 24
    cfg = getattr(model.encoder, "config", None)
    if cfg is not None and hasattr(cfg, "num_hidden_layers"):
        num_layers = int(cfg.num_hidden_layers)

    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    layer_re = re.compile(r"encoder\.layer\.(\d+)\.")

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        if not name.startswith("encoder."):
            lr = head_lr
        elif "embeddings." in name:
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
# 4. Top-Tail Precision Pairwise Margin Loss (Direct PR-AUC & R@P95/99 Optimizer)
# ---------------------------------------------------------------------------

def compute_top_tail_pairwise_loss_v130(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    hard_neg_boost: torch.Tensor | None = None,
    margin: float = 1.6,
    top_neg_frac: float = 0.25,
) -> torch.Tensor:
    """Push top hardest negatives below all positives with smooth softplus margin (maximizes R@P95/99!)."""
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

    # Pairwise margin: want z_pos - z_neg >= margin
    diff = z_pos.unsqueeze(1) - top_vals.unsqueeze(0)  # (n_pos, k)
    pair_loss = F.softplus(margin - diff)

    if hard_neg_boost is not None:
        hn = hard_neg_boost[m][neg_m][top_idx].float()  # (k,)
        w_col = 1.0 + 0.8 * hn
        pair_loss = pair_loss * w_col.unsqueeze(0)

    return pair_loss.mean()


def compute_dual_axis_superteacher_loss_v130(
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
        # Mild label smoothing (0.02) prevents logit explosion while preserving tail ordering
        ya_s = y_ai[ai_m].float() * 0.96 + 0.02
        l_ai_bce = (F.binary_cross_entropy_with_logits(lg_ai[ai_m], ya_s, reduction="none") * w_ai[ai_m].float()).mean()
        l_ai_tail = compute_top_tail_pairwise_loss_v130(lg_ai, y_ai, has_ai, hard_neg_boost=hard_neg_flag, margin=1.8, top_neg_frac=0.25)
        # Hard-negative ceiling penalty (pre_llm / >1y old-video / Q4 human promo MUST stay < -0.5!)
        hn_m = ai_m & (y_ai <= 0.20) & (hard_neg_flag > 0.5)
        l_ai_hn = F.relu(lg_ai[hn_m] + 0.5).pow(2).mean() if hn_m.any() else lg_ai.sum() * 0.0
    else:
        l_ai_bce = lg_ai.sum() * 0.0
        l_ai_tail = lg_ai.sum() * 0.0
        l_ai_hn = lg_ai.sum() * 0.0

    pr_m = has_pr > 0.5
    if pr_m.any():
        yp_s = y_pr[pr_m].float() * 0.96 + 0.02
        l_pr_bce = (F.binary_cross_entropy_with_logits(lg_pr[pr_m], yp_s, reduction="none") * w_pr[pr_m].float()).mean()
        l_pr_tail = compute_top_tail_pairwise_loss_v130(lg_pr, y_pr, has_pr, margin=1.6, top_neg_frac=0.25)
    else:
        l_pr_bce = lg_pr.sum() * 0.0
        l_pr_tail = lg_pr.sum() * 0.0

    s4_m = pr_m & (sub4 >= 0) & (sub4 < 4)
    l_s4 = F.cross_entropy(lg_s4[s4_m], sub4[s4_m].long()) if s4_m.any() else lg_pr.sum() * 0.0

    loss = l_ai_bce + 0.35 * l_ai_tail + 0.20 * l_ai_hn + 0.70 * l_pr_bce + 0.20 * l_pr_tail + 0.12 * l_s4
    return loss, {
        "ai_bce": float(l_ai_bce.item()),
        "ai_tail": float(l_ai_tail.item()),
        "ai_hn": float(l_ai_hn.item()),
        "pr_bce": float(l_pr_bce.item()),
    }


# ---------------------------------------------------------------------------
# 5. Purified Real-Only Corpus Builder + Full Distillation Pool Aggregator
# ---------------------------------------------------------------------------

def _extract_origin_v130(row: dict[str, Any]) -> str:
    bm = row.get("bucket_meta") or {}
    return str(row.get("origin_6way") or bm.get("origin_6way") or bm.get("origin") or "")


def _is_pure_text_ai_pos_v130(source_tag: str, origin_6way: str) -> tuple[bool, float]:
    """Return `(admit_as_binary_ai_pos, weight)` based on our empirical label-purity audit."""
    st = str(source_tag or "")
    orig = str(origin_6way or "")
    # Never train binary y_ai=1 on synth or noisy behavior-only labels!
    if "synth" in st.lower():
        return False, 0.0
    if orig in NOISY_BEHAVIOR_ORIGINS_V130:
        return False, 0.0
    if st in NOISY_SCRIPTBOT_SOURCES_V130 and orig == "Script-Plugin-Bot":
        return False, 0.0
    if st == "sibling_short_stealth_ai_15_60":
        return (orig == "Autonomous-Persona-Agent"), (1.35 if orig == "Autonomous-Persona-Agent" else 0.0)
    if st == "sibling_neon_committee_3r_gold":
        return True, 1.25
    if st == "sibling_god_tier_calibrated_gold":
        return (orig == "Autonomous-Persona-Agent"), (1.15 if orig == "Autonomous-Persona-Agent" else 0.0)
    if st == "v12_gold_v2_train":
        return True, 1.30
    if st in ("v121_wave3_train", "v123_delta"):
        return True, 1.25
    if st in ("v11_wave2_gold", "v11_wave1_gold", "v11_wave25_gold", "v11_wave2_longform"):
        if orig == "Autonomous-Persona-Agent":
            return True, 1.20
        if orig == "Script-Plugin-Bot":
            return True, 0.70
        return True, 0.90
    if orig == "Autonomous-Persona-Agent":
        return True, 1.15
    if orig == "Script-Plugin-Bot":
        return True, 0.65
    return True, 1.00


def build_v130_superteacher_corpus(
    sibling_dir: Path | None,
    delta_dir: Path | None,
    gold_v2_dir: Path | None,
    wave3_dir: Path | None,
    v11_dir: Path | None,
    v04_data_path: Path | None = None,
    smoke_test: bool = False,
    seed: int = 20261128,
) -> dict[str, Any]:
    """Assemble strictly purified, zero-leakage train / val / test / 180k distillation pools."""
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
            h = text_sha16_v130(txt)
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
                "sub4": SUB4_MAP_V130.get(str(r.get("promo_type", "none")), -1),
                "source_tag": src_name,
                "era5": sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts")),
            })

    if gold_v2_dir and gold_v2_dir.is_dir():
        _register_holdout(read_jsonl_any_v130(gold_v2_dir / "v12_gold_v2_val.jsonl.gz"), val_gold_v2, "v12_gold_v2_val", 64 if smoke_test else None)
        _register_holdout(read_jsonl_any_v130(gold_v2_dir / "v12_gold_v2_test.jsonl.gz"), test_gold_v2, "v12_gold_v2_test", 64 if smoke_test else None)
    if wave3_dir and wave3_dir.is_dir():
        _register_holdout(read_jsonl_any_v130(wave3_dir / "v121_wave3_val.jsonl.gz"), val_wave3, "v121_wave3_val", 64 if smoke_test else None)
    if v11_dir and v11_dir.is_dir():
        for fn in ("v11_wave2_dualaxis_gold_val.jsonl.gz", "v11_wave2_dualaxis_gold_cert.jsonl.gz"):
            _register_holdout(read_jsonl_any_v130(v11_dir / fn), val_promo_v11, fn.replace(".jsonl.gz", ""), 48 if smoke_test else None)

    # 2. Load Sibling + Delta packs via our existing verified v1.2.3 loader (automatically quarantines god500 & hygiene!)
    sib_pack = load_sibling_and_delta_v123(sibling_dir, delta_dir, smoke_test=smoke_test)
    # Block god_tier_500_holdout texts from training too!
    for r in sib_pack["god_tier_500_holdout"]:
        blocked_hashes.add(text_sha16_v130(str(r.get("text", ""))))

    # Containers for purified training + complete real distillation pool
    train_ai_pos: list[dict[str, Any]] = []
    train_hu_natural: list[dict[str, Any]] = []
    train_hu_hard: list[dict[str, Any]] = []
    train_promo_extra: list[dict[str, Any]] = []
    val_old_video_neg: list[dict[str, Any]] = []
    distill_pool_by_hash: dict[str, dict[str, Any]] = {}
    stats = {
        "purged_noisy_ai_pos": 0,
        "purged_synth_ai_pos": 0,
        "blocked_holdout_dedup": 0,
        "admitted_pure_ai_pos": 0,
        "admitted_hu_natural": 0,
        "admitted_hu_hard": 0,
        "admitted_promo_pos": 0,
    }

    def _add_distill(sid: str, txt: str, src: str, era5: int, orig: str, gold_ai: float | None, gold_pr: float | None) -> None:
        h = text_sha16_v130(txt)
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
        add_to_distill: bool = True,
    ) -> None:
        txt = str(txt or "").strip()
        if len(txt) < 2:
            return
        h = text_sha16_v130(txt)
        if h in blocked_hashes:
            stats["blocked_holdout_dedup"] += 1
            return

        g_ai = float(ai_sc) if ai_sc is not None else None
        g_pr = float(pr_sc) if pr_sc is not None else None
        if add_to_distill and ("synth" not in src.lower()):
            _add_distill(sid, txt, src, era5, orig, g_ai, g_pr)

        has_pr = 1.0 if (g_pr is not None and (g_pr <= 25.0 or g_pr >= 70.0)) else 0.0
        y_pr = 1.0 if (g_pr is not None and g_pr >= 70.0) else 0.0
        s4 = SUB4_MAP_V130.get(ptype, -1) if (has_pr > 0.5 and y_pr > 0.5) else -1

        # Case A: Candidate AI Positive
        if g_ai is not None and g_ai >= 70.0:
            admit, w_a = _is_pure_text_ai_pos_v130(src, orig)
            if not admit:
                if "synth" in src.lower():
                    stats["purged_synth_ai_pos"] += 1
                else:
                    stats["purged_noisy_ai_pos"] += 1
                # Keep promo supervision if it is a strong promo row!
                if has_pr > 0.5 and y_pr > 0.5:
                    train_promo_extra.append({
                        "sid": sid, "text": txt, "y_ai": 0.0, "has_ai": 0.0,
                        "y_pr": 1.0, "has_pr": 1.0, "sub4": s4,
                        "w_ai": 0.0, "w_pr": 1.5, "hard_neg": 0.0, "era5": era5, "source_tag": src,
                    })
                    stats["admitted_promo_pos"] += 1
                return
            train_ai_pos.append({
                "sid": sid, "text": txt, "y_ai": 1.0, "has_ai": 1.0,
                "y_pr": y_pr, "has_pr": has_pr, "sub4": s4,
                "w_ai": w_a, "w_pr": 1.5 if y_pr > 0.5 else 0.7, "hard_neg": 0.0, "era5": era5, "source_tag": src,
            })
            stats["admitted_pure_ai_pos"] += 1
            if has_pr > 0.5 and y_pr > 0.5:
                stats["admitted_promo_pos"] += 1
            return

        # Case B: Verified Human Negative (`ai_score <= 25`)
        if g_ai is not None and g_ai <= 25.0:
            is_q4_promo = bool(has_pr > 0.5 and y_pr > 0.5)
            is_old_gt1y = bool(is_pre_llm or era5 in (0, 1, 2) or "oldvideo" in src.lower() or "pre_llm" in src.lower())
            is_hard = bool(
                is_hard_neg_hint or is_q4_promo or is_old_gt1y
                or "trap" in src.lower() or "organic_brand" in src.lower() or "longform" in src.lower()
                or len(txt) >= 110
            )
            w_a = 1.55 if is_q4_promo else (1.35 if (is_pre_llm or "oldvideo" in src.lower() or "trap" in src.lower()) else (1.15 if is_hard else 1.0))
            item = {
                "sid": sid, "text": txt, "y_ai": 0.0, "has_ai": 1.0,
                "y_pr": y_pr, "has_pr": has_pr, "sub4": s4,
                "w_ai": w_a, "w_pr": 1.8 if is_q4_promo else (1.2 if "organic_brand" in src.lower() else 0.75),
                "hard_neg": 1.0 if is_hard else 0.0, "era5": era5, "source_tag": src,
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
        if has_pr > 0.5:
            train_promo_extra.append({
                "sid": sid, "text": txt, "y_ai": 0.0, "has_ai": 0.0,
                "y_pr": y_pr, "has_pr": 1.0, "sub4": s4,
                "w_ai": 0.0, "w_pr": 1.6 if y_pr > 0.5 else 1.0, "hard_neg": 0.0, "era5": era5, "source_tag": src,
            })
            if y_pr > 0.5:
                stats["admitted_promo_pos"] += 1

    # 3. Ingest Sibling Dual-Axis + Delta Dual-Axis (already merged by load_sibling_and_delta_v123!)
    for r in sib_pack["dual_train_rows"]:
        src = str(r.get("source_tag") or ("v123_delta" if r.get("is_delta") else "sibling_dual"))
        orig = _extract_origin_v130(r)
        ai_sc = float(r.get("ai_score", 0.0)) if r.get("has_ai", True) else None
        pr_sc = float(r.get("promo_score", 0.0)) if r.get("has_promo", True) else None
        is_pre = str(r.get("era_bucket", "")).lower() == "pre_llm"
        e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), is_pre_llm=is_pre, score_source=src, ai_score_0_100=ai_sc)
        _ingest_row(
            str(r.get("text", "")), str(r.get("sample_id") or r.get("sid") or ""), src, orig, e5,
            ai_sc, pr_sc, str(r.get("promo_type", "none")), is_pre_llm=is_pre,
        )

    # 4. Ingest Sibling AI-Only Real (`pack1_A_wild_human`, `pack2_B`, `pack3_D`, `delta_ai_real`)
    for r in sib_pack["ai_only_real_rows"]:
        src = str(r.get("source_tag") or "sibling_ai_only")
        orig = _extract_origin_v130(r)
        ai_sc = float(r.get("ai_score", 0.0))
        is_pre = str(r.get("era_bucket", "")).lower() == "pre_llm"
        e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), is_pre_llm=is_pre, score_source=src, ai_score_0_100=ai_sc)
        _ingest_row(str(r.get("text", "")), str(r.get("sample_id") or ""), src, orig, e5, ai_sc, None, is_pre_llm=is_pre)

    # 5. Ingest `v12_gold_v2_train` + `v121_wave3_train`
    if gold_v2_dir and gold_v2_dir.is_dir():
        for r in read_jsonl_any_v130(gold_v2_dir / "v12_gold_v2_train.jsonl.gz")[: (96 if smoke_test else 999999)]:
            ai_sc = float(r.get("ai_score", r.get("score", 0.0)))
            pr_sc = float(r["promo_score"]) if r.get("promo_score") is not None else None
            e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=ai_sc)
            _ingest_row(str(r.get("text", "")), str(r.get("sample_id") or r.get("sid") or ""), "v12_gold_v2_train", _extract_origin_v130(r), e5, ai_sc, pr_sc, str(r.get("promo_type", "none")))
    if wave3_dir and wave3_dir.is_dir():
        for r in read_jsonl_any_v130(wave3_dir / "v121_wave3_train.jsonl.gz")[: (96 if smoke_test else 999999)]:
            ai_sc = float(r.get("ai_score", 0.0))
            pr_sc = float(r.get("promo_score", 0.0))
            e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=ai_sc)
            _ingest_row(str(r.get("text", "")), str(r.get("sample_id") or r.get("sid") or ""), "v121_wave3_train", _extract_origin_v130(r), e5, ai_sc, pr_sc, str(r.get("promo_type", "none")))

    # 6. Ingest `v11` verified dual-axis gold + Wave-2.5 + UOT Promo Positives & Hard Negatives
    if v11_dir and v11_dir.is_dir():
        for fn, tag in [
            ("v11_wave1_dualaxis_gold_verified.jsonl.gz", "v11_wave1_gold"),
            ("v11_wave2_dualaxis_gold_verified.jsonl.gz", "v11_wave2_gold"),
            ("v11_wave25_dualaxis_gold_verified.jsonl.gz", "v11_wave25_gold"),
            ("v11_wave2_longform_gold_1200.jsonl.gz", "v11_wave2_longform"),
        ]:
            for r in read_jsonl_any_v130(v11_dir / fn)[: (64 if smoke_test else 999999)]:
                if r.get("split", "train") != "train":
                    continue
                ai_sc = float(r.get("ai_score", 0.0))
                pr_sc = float(r.get("promo_score", 0.0))
                e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=ai_sc)
                _ingest_row(str(r.get("text", "")), str(r.get("sample_id") or r.get("sid") or ""), tag, _extract_origin_v130(r), e5, ai_sc, pr_sc, str(r.get("promo_type", "none")))
        for r in read_jsonl_any_v130(v11_dir / "v11_uot_wild_promo_pos.jsonl")[: (32 if smoke_test else 999999)]:
            _ingest_row(str(r.get("text", "")), str(r.get("sample_id") or ""), "v11_uot_wild_promo", "", -1, None, 95.0, str(r.get("promo_type", "soft_shill")))
        for r in read_jsonl_any_v130(v11_dir / "v11_bge_knn_promo_hard_negatives_2k_uot_gated.jsonl")[: (48 if smoke_test else 999999)]:
            _ingest_row(str(r.get("text", "")), str(r.get("sample_id") or ""), "v11_promo_hn", "", -1, None, 0.0, "none")

    # 7. Carve a dedicated held-out `old_video_gt1y` + `pre_llm` human FPR-guard validation set (1,200 rows!)
    old_cand = [
        i for i, it in enumerate(train_hu_hard)
        if it["era5"] in (0, 1, 2) or "oldvideo" in it["source_tag"].lower()
    ]
    rng.shuffle(old_cand)
    n_old_val = min(48 if smoke_test else 1200, len(old_cand) // 5)
    old_val_idx = set(old_cand[:n_old_val])
    val_old_video_neg = [train_hu_hard[i] for i in sorted(old_val_idx)]
    train_hu_hard = [it for i, it in enumerate(train_hu_hard) if i not in old_val_idx]

    # 8. Optional real-wild unlabeled Bilibili comments from `v04` added STRICTLY to `distill_pool` (NEVER trained with noisy v04 labels!)
    if v04_data_path and v04_data_path.is_file():
        max_v04_distill = 200 if smoke_test else 40000
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

    # Also add god_tier_500_holdout as pure diagnostic slice (NOT in train, NOT in distill!)
    diag_god500 = []
    for r in sib_pack["god_tier_500_holdout"][: (48 if smoke_test else 999)]:
        orig = _extract_origin_v130(r)
        diag_god500.append({
            "sid": str(r.get("sample_id") or ""),
            "text": str(r.get("text", "")),
            "y_ai": 1.0 if float(r.get("ai_score", 0.0)) >= 70.0 else 0.0,
            "is_text_verifiable": orig not in NOISY_BEHAVIOR_ORIGINS_V130 and orig != "Script-Plugin-Bot",
        })

    # Deduplicate train pools by hash
    def _dedup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out_r: list[dict[str, Any]] = []
        for x in rows:
            h = text_sha16_v130(x["text"])
            if h in seen:
                continue
            seen.add(h)
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
        "val_gold_v2": val_gold_v2,
        "test_gold_v2": test_gold_v2,
        "val_wave3": val_wave3,
        "val_promo_v11": val_promo_v11,
        "val_old_video_neg": val_old_video_neg,
        "diag_god500": diag_god500,
        "distill_pool": list(distill_pool_by_hash.values()),
        "stats": {
            **stats,
            "dedup_train_ai_pos": len(train_ai_pos),
            "dedup_train_hu_natural": len(train_hu_natural),
            "dedup_train_hu_hard": len(train_hu_hard),
            "dedup_train_promo_extra": len(train_promo_extra),
            "val_gold_v2": len(val_gold_v2),
            "test_gold_v2": len(test_gold_v2),
            "val_wave3": len(val_wave3),
            "val_old_video_neg": len(val_old_video_neg),
            "distill_pool_total": len(distill_pool_by_hash),
        },
    }


def sample_antidilution_epoch_v130(
    corpus: dict[str, Any],
    rng: random.Random,
    neg_per_pos: float = 4.5,
    natural_frac: float = 0.65,
    smoke_test: bool = False,
) -> list[dict[str, Any]]:
    """Build one anti-dilution epoch: 100% pure AI+ + 65% natural-traffic / 35% hard-neg humans + extra promo."""
    pos = list(corpus["train_ai_pos"])
    hu_nat = list(corpus["train_hu_natural"])
    hu_hd = list(corpus["train_hu_hard"])
    pr_ex = list(corpus["train_promo_extra"])

    if smoke_test:
        pos = pos[:32]
        hu_nat = hu_nat[:48]
        hu_hd = hu_hd[:32]
        pr_ex = pr_ex[:16]
        out = pos + hu_nat + hu_hd + pr_ex
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

    # Always include ALL Q4 human-promo rows inside hard negatives!
    q4_rows = [x for x in hu_hd if x.get("has_pr", 0.0) > 0.5 and x.get("y_pr", 0.0) > 0.5]
    other_hd = [x for x in hu_hd if not (x.get("has_pr", 0.0) > 0.5 and x.get("y_pr", 0.0) > 0.5)]
    chosen_hd = q4_rows + _draw(other_hd, max(0, n_hd - len(q4_rows)))
    chosen_nat = _draw(hu_nat, n_nat)
    chosen_pr = _draw(pr_ex, min(len(pr_ex), max(1500, len(pos) // 3)))

    epoch_rows = pos + chosen_nat + chosen_hd + chosen_pr
    rng.shuffle(epoch_rows)
    return epoch_rows


def fit_scalar_temperature_v130(logits: np.ndarray, y: np.ndarray) -> float:
    """Grid + golden-section NLL temperature fit for clean calibrated Super-Teacher probabilities."""
    if len(logits) < 10 or y.sum() == 0 or y.sum() == len(y):
        return 1.0
    best_t, best_nll = 1.0, 1e9
    for t in np.linspace(0.4, 3.0, 53):
        z = np.clip(logits / t, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return round(best_t, 4)


def run_v130_superteacher_zero_pii_gate(model: DualAxisSuperTeacherNet, tokenizer: Any, device: torch.device) -> dict[str, Any]:
    """Strict Zero-PII verification (`V130_ST_C1`): swapping @username & contact IDs MUST yield diff === 0.0!"""
    model.eval()
    ta = ["回复 @测试用户A : 关注工种号交流2群 168688884 助理老师：xhs54199 领取资料[doge]"]
    tb = ["回复 @RandomName99 : 关注工种号交流2群 999888777 助理老师：abc98765 领取资料[doge]"]
    with torch.no_grad():
        ia, ma = encode_head_tail_hf_v130(tokenizer, ta)
        ib, mb = encode_head_tail_hf_v130(tokenizer, tb)
        oa = model(ia.to(device), ma.to(device))
        ob = model(ib.to(device), mb.to(device))
        d_ai = float((oa["logit_ai"] - ob["logit_ai"]).abs().max().item())
        d_pr = float((oa["logit_promo"] - ob["logit_promo"]).abs().max().item())
    mx = max(d_ai, d_pr)
    return {"max_abs_diff": mx, "ai_diff": d_ai, "promo_diff": d_pr, "pass": mx == 0.0}
