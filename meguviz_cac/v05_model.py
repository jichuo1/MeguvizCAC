"""MeguvizCAC v0.5.0-ApexLoop Neural Architecture:
Modern mmBERT-inherited RoPE+RMSNorm+GeGLU Prelude + Static K=2 Long-Comment Merge +
32-d Zero-PII PIFE Stylometrics + Optional 4-d Zero-PII Context Echo +
Static 16-Slot Workbench (4 LOTUS + 4 Task Registers + 8 Evidence Anchors) +
Zero-LoRA-MatMul Single-Stream InfiLoop + SCORE Continuous-Progress Core +
256-d L2 Embedding Coda + Full-Trajectory LoopQ QAT + Dual-Graph Static ONNX Export.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v05_features import MAX_LEN_V05, STYLE_FEAT_DIM


class QATEmbedding(nn.Module):
    """Fake-quantized per-row INT8/W4 STE embedding (FP16-safe)."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.qat_enabled = False
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.qat_enabled:
            scale = w.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
            w_q = torch.clamp(torch.round(w / scale), -128.0, 127.0) * scale
            w = w + (w_q - w).detach()
        return F.embedding(input_ids, w)


class QATLinear(nn.Module):
    """Fake-quantized per-channel W8 + per-tensor A8 STE linear matching ORT INT8."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.qat_enabled = False
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class NoBiasLayerNorm(nn.Module):
    """Exact mmBERT-compatible mean-centered LayerNorm without bias (ONNX-friendly, zero init mismatch)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
        return ((xf - mean) * torch.rsqrt(var + self.eps)).to(x.dtype) * self.weight


class AdaRMSNorm(nn.Module):
    """Zero-LoRA-MatMul step-modulated LayerNorm driven by continuous budget progress tau (mmBERT-compatible)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, scale_delta: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
        normed = ((xf - mean) * torch.rsqrt(var + self.eps)).to(x.dtype)
        return normed * (self.weight + scale_delta) + shift


class QKNorm(nn.Module):
    """Per-head RMS QK-Norm preventing FP16/INT8 attention logit spikes."""

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(1, num_heads, 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)
        return x * inv_rms * self.weight


def precompute_rope(max_len: int, head_dim: int, theta: float = 160000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute ModernBERT-style interleaved RoPE cos/sin tables (ONNX-friendly)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)))
    pos = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return cos[None, None, :, :], sin[None, None, :, :]


def apply_rope_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply standard half-split RoPE (100% compatible with ModernBERT / mmBERT pretrained weights)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin


def build_tau_features(tau: float, dt: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """8-d continuous progress encoding (tau = t/T, dt = 1/T + Fourier harmonics) replacing discrete step_id."""
    return torch.tensor(
        [
            tau,
            dt,
            math.sin(math.pi * tau),
            math.cos(math.pi * tau),
            math.sin(2.0 * math.pi * tau),
            math.cos(2.0 * math.pi * tau),
            math.sin(4.0 * math.pi * tau),
            math.cos(4.0 * math.pi * tau),
        ],
        device=device,
        dtype=dtype,
    )


class ModernPreludeBlock(nn.Module):
    """Modern RoPE + NoBiasLayerNorm + GeGLU + DWConv Prelude layer (zero-mismatch mmBERT-small inheritance)."""

    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 6,
        intermediate_size: int = 1152,
        max_len: int = MAX_LEN_V05,
        eps: float = 1e-5,
        drop_p: float = 0.08,
        skip_attn_norm: bool = False,
        rope_theta: float = 160000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.skip_attn_norm = bool(skip_attn_norm)

        self.attn_norm = nn.Identity() if skip_attn_norm else NoBiasLayerNorm(hidden_size, eps=eps)
        self.wqkv = QATLinear(hidden_size, 3 * hidden_size, bias=False)
        self.wo = QATLinear(hidden_size, hidden_size, bias=False)

        self.mlp_norm = NoBiasLayerNorm(hidden_size, eps=eps)
        self.wi = QATLinear(hidden_size, 2 * intermediate_size, bias=False)
        self.wdown = QATLinear(intermediate_size, hidden_size, bias=False)
        # Topic-invariant multi-scale local phrasing DWConv (k=3, 5, 7 — <0.02 MB, near-zero init to preserve mmBERT!)
        self.dw3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size, bias=False)
        self.dw5 = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size, bias=False)
        self.dw7 = nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3, groups=hidden_size, bias=False)
        self.dw_gate = nn.Parameter(torch.full((1, 1, hidden_size), -3.5))
        self.drop = nn.Dropout(drop_p)
        self.use_sdpa = False

        cos, sin = precompute_rope(max_len, self.head_dim, theta=rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def load_from_mmbert_layer(self, layer) -> None:
        with torch.no_grad():
            if (not self.skip_attn_norm) and hasattr(layer, "attn_norm") and hasattr(layer.attn_norm, "weight"):
                self.attn_norm.weight.copy_(layer.attn_norm.weight)
            self.wqkv.weight.copy_(layer.attn.Wqkv.weight)
            self.wo.weight.copy_(layer.attn.Wo.weight)
            if hasattr(layer, "mlp_norm") and hasattr(layer.mlp_norm, "weight"):
                self.mlp_norm.weight.copy_(layer.mlp_norm.weight)
            self.wi.weight.copy_(layer.mlp.Wi.weight)
            self.wdown.weight.copy_(layer.mlp.Wo.weight)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, mask_f: torch.Tensor | None = None) -> torch.Tensor:
        b, l, _ = x.shape
        h = self.attn_norm(x)
        # Local multi-scale DWConv cadence branch (zero self-attention overhead, masked cleanly)
        ht = (h * mask_f if mask_f is not None else h).transpose(1, 2)
        dw_out = ((self.dw3(ht) + self.dw5(ht) + self.dw7(ht)) / 3.0).transpose(1, 2)
        qkv = self.wqkv(h).view(b, l, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        cos = self.rope_cos[:, :, :l, :].to(q.dtype)
        sin = self.rope_sin[:, :, :l, :].to(q.dtype)
        q = apply_rope_half(q, cos, sin)
        k = apply_rope_half(k, cos, sin)

        if self.use_sdpa:
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.to(q.dtype), dropout_p=self.drop.p if self.training else 0.0)
        else:
            sc = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            sc = 30.0 * torch.tanh(sc / 30.0) + attn_bias
            att = self.drop(torch.softmax(sc, dim=-1))
            ctx = torch.matmul(att, v)
        ctx = ctx.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
        x = x + self.drop(self.wo(ctx)) + torch.sigmoid(self.dw_gate).to(x.dtype) * dw_out

        h2 = self.mlp_norm(x)
        u, g = self.wi(h2).chunk(2, dim=-1)
        x = x + self.drop(self.wdown(F.gelu(u) * g))
        return x


class StaticSlotCompressor(nn.Module):
    """Static 16-Slot Cross-Attention Workbench: 4 LOTUS + 4 Task Registers + 8 Evidence Anchors -> fixed [B, 16, D]."""

    def __init__(self, hidden_size: int = 384, num_heads: int = 6, num_slots: int = 16, eps: float = 1e-5, drop_p: float = 0.08):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_slots = num_slots
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, hidden_size) * 0.02)
        self.q_proj = QATLinear(hidden_size, hidden_size, bias=False)
        self.k_proj = QATLinear(hidden_size, hidden_size, bias=False)
        self.v_proj = QATLinear(hidden_size, hidden_size, bias=False)
        self.o_proj = QATLinear(hidden_size, hidden_size, bias=False)
        self.q_norm = QKNorm(num_heads, self.head_dim)
        self.k_norm = QKNorm(num_heads, self.head_dim)
        self.out_norm = NoBiasLayerNorm(hidden_size, eps=eps)
        self.drop = nn.Dropout(drop_p)
        self.use_sdpa = False

    def forward(self, h_pre: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        b, l, _ = h_pre.shape
        slots = self.slot_queries.expand(b, -1, -1)
        q = self.q_norm(self.q_proj(slots).view(b, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2))
        k = self.k_norm(self.k_proj(h_pre).view(b, l, self.num_heads, self.head_dim).transpose(1, 2))
        v = self.v_proj(h_pre).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_sdpa:
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.to(q.dtype), dropout_p=self.drop.p if self.training else 0.0)
        else:
            sc = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            sc = 30.0 * torch.tanh(sc / 30.0) + attn_bias
            att = self.drop(torch.softmax(sc, dim=-1))
            ctx = torch.matmul(att, v)
        ctx = ctx.transpose(1, 2).contiguous().view(b, self.num_slots, self.hidden_size)
        return self.out_norm(slots + self.drop(self.o_proj(ctx)))


class ApexCoreBlock(nn.Module):
    """Contractive Shared Core running on fixed [B, 16, 384] with AdaLayerNorm + SCORE + Post-Norm stability lock."""

    def __init__(self, hidden_size: int = 384, num_heads: int = 6, intermediate_size: int = 1152, eps: float = 1e-5, drop_p: float = 0.08):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.attn_norm = AdaRMSNorm(hidden_size, eps=eps)
        self.wqkv = QATLinear(hidden_size, 3 * hidden_size, bias=False)
        self.wo = QATLinear(hidden_size, hidden_size, bias=False)

        self.mlp_norm = AdaRMSNorm(hidden_size, eps=eps)
        self.wi = QATLinear(hidden_size, 2 * intermediate_size, bias=False)
        self.wdown = QATLinear(intermediate_size, hidden_size, bias=False)
        self.post_norm = NoBiasLayerNorm(hidden_size, eps=eps)

        # Continuous budget progress tau = t/T_budget -> AdaLayerNorm mods + OperLoop decay/eta + GRT gate
        self.tau_mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.SiLU(),
            nn.Linear(64, 4 * hidden_size + 3),
        )
        nn.init.normal_(self.tau_mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.tau_mlp[-1].bias)

        # Lightweight GRT elementwise update gate (no extra heavy MatMul on 16 slots)
        self.grt_gate = QATLinear(2 * hidden_size, hidden_size, bias=True)
        nn.init.zeros_(self.grt_gate.weight)
        nn.init.constant_(self.grt_gate.bias, 0.5)

        self.drop = nn.Dropout(drop_p)
        self.use_sdpa = False

    def load_from_mmbert_layer(self, layer) -> None:
        with torch.no_grad():
            if hasattr(layer, "attn_norm") and hasattr(layer.attn_norm, "weight"):
                self.attn_norm.weight.copy_(layer.attn_norm.weight)
            self.wqkv.weight.copy_(layer.attn.Wqkv.weight)
            self.wo.weight.copy_(layer.attn.Wo.weight)
            if hasattr(layer, "mlp_norm") and hasattr(layer.mlp_norm, "weight"):
                self.mlp_norm.weight.copy_(layer.mlp_norm.weight)
            self.wi.weight.copy_(layer.mlp.Wi.weight)
            self.wdown.weight.copy_(layer.mlp.Wo.weight)

    def forward(self, h: torch.Tensor, z0: torch.Tensor, tau_feat: torch.Tensor) -> torch.Tensor:
        """One strictly norm-bounded contractive SCORE + OperLoop + GRT step on fixed [B, 16, D]."""
        b, n_slots, d = h.shape
        mods = self.tau_mlp(tau_feat)
        s1, sh1, s2, sh2 = mods[: 4 * d].chunk(4)
        s1 = 0.20 * torch.tanh(s1)
        sh1 = 0.20 * torch.tanh(sh1)
        s2 = 0.20 * torch.tanh(s2)
        sh2 = 0.20 * torch.tanh(sh2)

        dt_in = tau_feat[1]  # explicit 1 / T_budget SCORE contractive step size
        eta = (0.35 + 0.45 * torch.sigmoid(mods[4 * d])) * dt_in
        decay = 0.15 * torch.sigmoid(mods[4 * d + 1]) * dt_in
        orth_w = 0.20 * torch.sigmoid(mods[4 * d + 2])

        # Orthogonal Prelude Input Injection (arXiv:2610.06833): strip component parallel to h
        h_f = h.float()
        z0_f = z0.float()
        proj_coeff = (z0_f * h_f).sum(dim=-1, keepdim=True) / (h_f.pow(2).sum(dim=-1, keepdim=True) + 1e-5)
        e_orth = (z0_f - proj_coeff * h_f).to(h.dtype)

        u = self.attn_norm(h + orth_w * e_orth, s1, sh1)
        qkv = self.wqkv(u).view(b, n_slots, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        if self.use_sdpa:
            ctx = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        else:
            sc = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            sc = 30.0 * torch.tanh(sc / 30.0)
            att = self.drop(torch.softmax(sc, dim=-1))
            ctx = torch.matmul(att, v)
        ctx = ctx.transpose(1, 2).contiguous().view(b, n_slots, d)

        h_mid = h + (eta * self.drop(self.wo(ctx)))

        u2 = self.mlp_norm(h_mid, s2, sh2)
        a, g = self.wi(u2).chunk(2, dim=-1)
        ff_delta = self.drop(self.wdown(F.gelu(a) * g))
        grt_g = torch.sigmoid(self.grt_gate(torch.cat([h_mid, z0], dim=-1)))
        # Post-step NoBiasLayerNorm guarantees state norm stays strictly O(1) across R=2/4/6!
        h_next = self.post_norm((1.0 - decay) * h_mid + grt_g * (eta * ff_delta))
        return h_next


class MeguvizV05ApexLoopModel(nn.Module):
    """MeguvizCAC v0.5.0-ApexLoop full model supporting variable SCORE budgets R=2/4/6 & dual-graph ONNX export."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 384,
        num_heads: int = 6,
        intermediate_size: int = 1152,
        prelude_layers: int = 2,
        core_layers: int = 1,
        num_slots: int = 16,
        style_dim: int = STYLE_FEAT_DIM,
        intent_dim: int = 1024,
        num_codes: int = 8,
        num_prag: int = 6,
        num_surf: int = 4,
        num_deep: int = 4,
        emb_drop_p: float = 0.10,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.prelude_layers = int(prelude_layers)
        self.core_layers = int(core_layers)
        self.num_slots = num_slots
        self.style_dim = style_dim

        # Factorized W4-ready embedding
        self.word_emb_A = QATEmbedding(vocab_size, embed_dim)
        self.word_emb_P = QATLinear(embed_dim, hidden_size, bias=False)
        self.emb_norm = NoBiasLayerNorm(hidden_size, eps=eps)
        self.emb_drop = nn.Dropout(emb_drop_p)

        # Multi-layer Modern RoPE + NoBiasLayerNorm + GeGLU + DWConv Prelude (inherits mmBERT L0 + L3 global!)
        self.prelude_blocks = nn.ModuleList(
            [
                ModernPreludeBlock(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    max_len=MAX_LEN_V05,
                    eps=eps,
                    skip_attn_norm=(i == 0),
                    rope_theta=160000.0,
                )
                for i in range(self.prelude_layers)
            ]
        )

        # 32-d Zero-PII Native Stylometric + PIFE Forensics Branch
        self.style_net = nn.Sequential(
            QATLinear(style_dim, 64, bias=True),
            nn.SiLU(),
            QATLinear(64, hidden_size, bias=True),
        )
        # Optional 4-d Zero-PII Context Echo Branch (video_echo / root_echo; zero-init for pure-text parity)
        self.ctx_echo_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, hidden_size),
        )
        nn.init.zeros_(self.ctx_echo_net[-1].weight)
        nn.init.zeros_(self.ctx_echo_net[-1].bias)

        # Prelude Fast Exit-0 Head (runs after full 2L Prelude — only exits ultra-confident comments!)
        self.exit0_norm = NoBiasLayerNorm(hidden_size, eps=eps)
        self.exit0_head = QATLinear(hidden_size, 1, bias=True)
        self.register_buffer("exit0_temperature", torch.tensor(1.0, dtype=torch.float32))

        # Static 16-Slot Workbench Compressor (4 LOTUS + 4 Task Registers + 8 Evidence Anchors)
        self.compressor = StaticSlotCompressor(hidden_size, num_heads, num_slots, eps=eps)

        # Shared Looped Core on fixed [B, 16, 384]
        self.core_blocks = nn.ModuleList(
            [ApexCoreBlock(hidden_size, num_heads, intermediate_size, eps=eps) for _ in range(core_layers)]
        )

        # InfiLoop content-weighted time-decayed residual attention over step history
        self.infiloop_q = nn.Linear(hidden_size, 32, bias=False)
        self.infiloop_k = nn.Linear(hidden_size, 32, bias=False)
        self.infiloop_decay = nn.Parameter(torch.tensor(0.35))

        # Coda Readout + 256-d L2 Embedding Export for L2 Prototype/Mahalanobis OOD
        self.coda_norm = NoBiasLayerNorm(hidden_size, eps=eps)
        self.coda_proj = QATLinear(2 * hidden_size, 256, bias=True)
        self.ai_head = QATLinear(256, 1, bias=True)
        self.register_buffer("coda_temperature", torch.tensor(1.0, dtype=torch.float32))

        # Training-Only Aux Heads (stripped completely at ONNX export -> 0 ms inference overhead!)
        self.aux_code_head = nn.Linear(256, num_codes)
        self.aux_prag_head = nn.Linear(256, num_prag)
        self.aux_lotus_intent = nn.Linear(hidden_size, intent_dim)
        self.aux_lotus_surf = nn.Linear(hidden_size, num_surf)
        self.aux_lotus_deep = nn.Linear(hidden_size, num_deep)
        self.aux_lotus_code = nn.Linear(hidden_size, num_codes)
        # Stage-0 Fast-ELECTRA 100% RTD head (stripped at export!)
        self.rtd_head = nn.Linear(hidden_size, 1)

    def load_from_mmbert(self, mmbert_model, a_mat: np.ndarray, p_mat: np.ndarray) -> None:
        with torch.no_grad():
            self.word_emb_A.weight.copy_(torch.from_numpy(a_mat))
            self.word_emb_P.weight.copy_(torch.from_numpy(p_mat.T))
            if hasattr(mmbert_model.embeddings, "norm") and hasattr(mmbert_model.embeddings.norm, "weight"):
                self.emb_norm.weight.copy_(mmbert_model.embeddings.norm.weight)
            # Inherit mmBERT-small global layers: Prelude -> [0, 3], Core -> [9] (or [6, 15] if 2L core)
            pre_src_idxs = [0, 3, 6][: self.prelude_layers]
            for blk, s_idx in zip(self.prelude_blocks, pre_src_idxs):
                blk.load_from_mmbert_layer(mmbert_model.layers[min(s_idx, len(mmbert_model.layers) - 1)])
            core_src_idxs = [9] if self.core_layers == 1 else [9, 15][: self.core_layers]
            for blk, s_idx in zip(self.core_blocks, core_src_idxs):
                blk.load_from_mmbert_layer(mmbert_model.layers[min(s_idx, len(mmbert_model.layers) - 1)])
            if hasattr(mmbert_model, "final_norm") and hasattr(mmbert_model.final_norm, "weight"):
                self.coda_norm.weight.copy_(mmbert_model.final_norm.weight)

    def set_sdpa(self, enabled: bool) -> None:
        for blk in self.prelude_blocks:
            blk.use_sdpa = bool(enabled)
        self.compressor.use_sdpa = bool(enabled)
        for blk in self.core_blocks:
            blk.use_sdpa = bool(enabled)

    def set_qat_enabled(self, enabled: bool) -> None:
        for m in self.modules():
            if isinstance(m, (QATEmbedding, QATLinear)):
                m.qat_enabled = bool(enabled)

    def forward_prelude(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        ctx_echo: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb_drop(self.emb_norm(self.word_emb_P(self.word_emb_A(input_ids))))
        attn_bias = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * (-10000.0)
        m_f = attention_mask.to(x.dtype).unsqueeze(-1)
        h_seq = x
        for blk in self.prelude_blocks:
            h_seq = blk(h_seq, attn_bias, mask_f=m_f)

        st_vec = self.style_net(style_feats.to(x.dtype))
        if ctx_echo is not None:
            st_vec = st_vec + self.ctx_echo_net(ctx_echo.to(x.dtype))

        mean_h = (h_seq * m_f).sum(dim=1) / m_f.sum(dim=1).clamp(min=1.0)
        exit0_feat = self.exit0_norm(0.5 * h_seq[:, 0, :] + 0.5 * mean_h + 0.25 * st_vec)
        exit0_logit = self.exit0_head(exit0_feat).squeeze(-1) / self.exit0_temperature.to(h_seq.dtype).clamp(min=0.5)

        z0 = self.compressor(h_seq, attn_bias)
        return z0, st_vec, exit0_logit, h_seq

    def run_core_step(self, h: torch.Tensor, z0: torch.Tensor, tau_feat: torch.Tensor) -> torch.Tensor:
        for blk in self.core_blocks:
            h = blk(h, z0, tau_feat)
        return h

    def infiloop_aggregate(self, history: list[torch.Tensor]) -> torch.Tensor:
        """InfiLoop content-weighted + time-decayed residual attention across loop history [h_0 .. h_t]."""
        if len(history) == 1:
            return history[0]
        h_cur = history[-1]  # (B, 16, D)
        stack = torch.stack(history, dim=1)  # (B, T, 16, D)
        t_len = len(history)
        q = self.infiloop_q(h_cur.mean(dim=1))  # (B, 32)
        k = self.infiloop_k(stack.mean(dim=2))  # (B, T, 32)
        scores = (k * q.unsqueeze(1)).sum(dim=-1) / math.sqrt(32.0)  # (B, T)
        # Learnable recency bias so newer refined steps get slight natural priority while preserving early anchors
        ages = torch.arange(t_len - 1, -1, -1, device=h_cur.device, dtype=scores.dtype)
        scores = scores - F.softplus(self.infiloop_decay).to(scores.dtype) * ages.unsqueeze(0)
        w = torch.softmax(scores, dim=-1)[:, :, None, None]  # (B, T, 1, 1)
        return (stack * w).sum(dim=1)

    def read_coda(self, h_agg: torch.Tensor, st_vec: torch.Tensor, return_aux: bool = False):
        h_c = self.coda_norm(h_agg)
        # Slots 0..3 = LOTUS reasoning slots; 4..7 = Task Registers; 8..15 = Evidence Anchors
        lotus_pool = h_c[:, :4, :].mean(dim=1)
        anchor_pool = h_c[:, 8:, :].mean(dim=1)
        pooled = 0.65 * lotus_pool + 0.35 * anchor_pool
        emb256_raw = F.silu(self.coda_proj(torch.cat([pooled, st_vec], dim=-1)))
        emb256_norm = F.normalize(emb256_raw.float(), p=2, dim=-1).to(emb256_raw.dtype)
        logit = self.ai_head(emb256_raw).squeeze(-1) / self.coda_temperature.to(emb256_raw.dtype).clamp(min=0.5)
        if not return_aux:
            return logit, emb256_norm
        aux = {
            "code_logit": self.aux_code_head(emb256_raw),
            "prag_logit": self.aux_prag_head(emb256_raw),
            "lotus_surf": self.aux_lotus_surf(h_c[:, 0, :]),
            "lotus_intent": self.aux_lotus_intent(h_c[:, 1, :]),
            "lotus_code": self.aux_lotus_code(h_c[:, 2, :]),
            "lotus_deep": self.aux_lotus_deep(h_c[:, 3, :]),
        }
        return logit, emb256_norm, aux

    def forward_full(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        num_steps: int = 2,
        ctx_echo: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> dict:
        z0, st_vec, exit0_logit, _ = self.forward_prelude(input_ids, attention_mask, style_feats, ctx_echo=ctx_echo)
        h = z0
        hist = [z0]
        step_logits = []
        step_deltas = []
        step_aux = []
        last_emb = None
        for t in range(1, num_steps + 1):
            tf = build_tau_features(float(t) / float(num_steps), 1.0 / float(num_steps), z0.device, z0.dtype)
            h_prev = h
            h = self.run_core_step(h, z0, tf)
            hist.append(h)
            # Zero-parameter trajectory velocity delta_t
            delta_t = (h.float() - h_prev.float()).norm(dim=(1, 2)) / (h.float().norm(dim=(1, 2)) + 1e-5)
            step_deltas.append(delta_t)
            h_agg = self.infiloop_aggregate(hist)
            if return_aux:
                lg, emb, ax = self.read_coda(h_agg, st_vec, return_aux=True)
                step_aux.append(ax)
            else:
                lg, emb = self.read_coda(h_agg, st_vec, return_aux=False)
            step_logits.append(lg)
            last_emb = emb
        return {
            "exit0_logit": exit0_logit,
            "step_logits": step_logits,
            "step_deltas": step_deltas,
            "emb256": last_emb,
            "z0": z0,
            "h_last": h,
            "h_prev": hist[-2] if len(hist) >= 2 else z0,
            "step_aux": step_aux,
        }


class V05PreludeExportWrapper(nn.Module):
    def __init__(self, model: MeguvizV05ApexLoopModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, style_feats: torch.Tensor):
        z0, st_vec, exit0_logit, _ = self.model.forward_prelude(input_ids, attention_mask, style_feats, ctx_echo=None)
        exit0_prob = torch.sigmoid(exit0_logit)
        return z0, st_vec, exit0_logit, exit0_prob


class V05CoreStepExportWrapper(nn.Module):
    """Single-step SCORE + OperLoop + GRT Core graph on fixed [B, 16, 384] — 100% INT8 quantized with zero weight duplication, supports R=2/4/6!"""

    def __init__(self, model: MeguvizV05ApexLoopModel):
        super().__init__()
        self.model = model

    def forward(self, h_prev: torch.Tensor, z0: torch.Tensor, tau_feat: torch.Tensor):
        h_next = self.model.run_core_step(h_prev, z0, tau_feat)
        return h_next


class V05CodaExportWrapper(nn.Module):
    """InfiLoop history aggregator + Coda readout + 256-d L2 embedding exporter supporting any history length T=R+1."""

    def __init__(self, model: MeguvizV05ApexLoopModel):
        super().__init__()
        self.model = model

    def forward(self, h_stack: torch.Tensor, st_vec: torch.Tensor):
        # h_stack: (B, T, 16, D)
        h_cur = h_stack[:, -1, :, :]
        t_len = h_stack.shape[1]
        q = self.model.infiloop_q(h_cur.mean(dim=1))
        k = self.model.infiloop_k(h_stack.mean(dim=2))
        scores = (k * q.unsqueeze(1)).sum(dim=-1) / math.sqrt(32.0)
        ages = torch.arange(t_len - 1, -1, -1, device=h_cur.device, dtype=scores.dtype)
        scores = scores - F.softplus(self.model.infiloop_decay).to(scores.dtype) * ages.unsqueeze(0)
        w = torch.softmax(scores, dim=-1)[:, :, None, None]
        h_agg = (h_stack * w).sum(dim=1)
        logit, emb256 = self.model.read_coda(h_agg, st_vec, return_aux=False)
        prob = torch.sigmoid(logit)
        return logit, prob, emb256


def export_v05_onnx_bundles(model_cpu: MeguvizV05ApexLoopModel, out_dir: Path, vocab_size: int) -> dict:
    """Export Prelude (INT8 + W4-Emb Edge), Single-Step Core (100% INT8, R=2/4/6), and Coda (INT8)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cpu = model_cpu.cpu().eval()
    model_cpu.set_sdpa(False)
    model_cpu.set_qat_enabled(False)

    def _export(mod, dummy, in_names, out_names, dyn, fp32_p: Path, int8_p: Path):
        kw = dict(input_names=in_names, output_names=out_names, dynamic_axes=dyn, opset_version=17, do_constant_folding=True)
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            kw["dynamo"] = False
        torch.onnx.export(mod, dummy, str(fp32_p), **kw)
        quantize_dynamic(str(fp32_p), str(int8_p), weight_type=QuantType.QInt8)

    d_ids = torch.ones((2, 32), dtype=torch.long)
    d_mask = torch.ones((2, 32), dtype=torch.long)
    d_style = torch.zeros((2, STYLE_FEAT_DIM), dtype=torch.float32)

    pre_fp32 = out_dir / "prelude.fp32.tmp.onnx"
    pre_int8 = out_dir / "prelude.onnx"
    pre_w4 = out_dir / "prelude_w4.onnx"
    core_fp32 = out_dir / "core_step.fp32.tmp.onnx"
    core_int8 = out_dir / "core_step.onnx"
    coda_fp32 = out_dir / "coda.fp32.tmp.onnx"
    coda_int8 = out_dir / "coda.onnx"

    _export(
        V05PreludeExportWrapper(model_cpu),
        (d_ids, d_mask, d_style),
        ["input_ids", "attention_mask", "style_feats"],
        ["z0", "st_vec", "exit0_logit", "exit0_prob"],
        {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "style_feats": {0: "batch"},
            "z0": {0: "batch"},
            "st_vec": {0: "batch"},
            "exit0_logit": {0: "batch"},
            "exit0_prob": {0: "batch"},
        },
        pre_fp32,
        pre_int8,
    )

    d_h = torch.zeros((2, model_cpu.num_slots, model_cpu.hidden_size), dtype=torch.float32)
    d_tau = build_tau_features(0.5, 0.5, torch.device("cpu"), torch.float32)
    d_stack = torch.zeros((2, 3, model_cpu.num_slots, model_cpu.hidden_size), dtype=torch.float32)
    d_sv = torch.zeros((2, model_cpu.hidden_size), dtype=torch.float32)
    try:
        _export(
            V05CoreStepExportWrapper(model_cpu),
            (d_h, d_h, d_tau),
            ["h_prev", "z0", "tau_feat"],
            ["h_next"],
            {"h_prev": {0: "batch"}, "z0": {0: "batch"}, "h_next": {0: "batch"}},
            core_fp32,
            core_int8,
        )
        _export(
            V05CodaExportWrapper(model_cpu),
            (d_stack, d_sv),
            ["h_stack", "st_vec"],
            ["logit", "prob", "emb256"],
            {"h_stack": {0: "batch", 1: "hist_len"}, "st_vec": {0: "batch"}, "logit": {0: "batch"}, "prob": {0: "batch"}, "emb256": {0: "batch"}},
            coda_fp32,
            coda_int8,
        )
    finally:
        core_fp32.unlink(missing_ok=True)
        coda_fp32.unlink(missing_ok=True)

    # Ordered W4-Emb / W8-Body Edge Prelude (shrinks 24k x 128 embedding from ~12.6 MB FP32 -> ~1.5 MB W4!)
    try:
        w4_res = _quantize_v05_prelude_w4_w8(pre_fp32, pre_w4, target_vocab=vocab_size)
    finally:
        pre_fp32.unlink(missing_ok=True)

    return {
        "prelude_int8": pre_int8.name,
        "prelude_w4": pre_w4.name if w4_res.get("ok") else None,
        "core_step": core_int8.name,
        "coda": coda_int8.name,
        "prelude_int8_bytes": int(pre_int8.stat().st_size),
        "core_step_bytes": int(core_int8.stat().st_size),
        "coda_bytes": int(coda_int8.stat().st_size),
        "edge_w4_w8": w4_res,
    }


def _quantize_v05_prelude_w4_w8(prelude_fp32_in: Path, prelude_out: Path, target_vocab: int = 24576, block_size: int = 32) -> dict:
    """Quantize v0.5 Prelude embedding Gather to W4 GatherBlockQuantized FIRST, then dynamic INT8 MatMuls."""
    try:
        import onnx
        import onnxruntime as ort
        from onnx import version_converter
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        from onnxruntime.quantization.quant_utils import QuantFormat
    except Exception as exc:
        return {"ok": False, "error": f"import_failed: {exc}"}

    tmp_named = prelude_out.with_suffix(".tmp_named.onnx")
    tmp_w4 = prelude_out.with_suffix(".tmp_w4only.onnx")
    try:
        model = onnx.load(str(prelude_fp32_in), load_external_data=True)
        default_opset = max((op.version for op in model.opset_import if op.domain in ("", "ai.onnx")), default=0)
        if default_opset < 21:
            model = version_converter.convert_version(model, 21)
            model.ir_version = max(model.ir_version, 10)
        init_shapes = {init.name: tuple(init.dims) for init in model.graph.initializer}
        target_gathers: list[str] = []
        used_names = {n.name for n in model.graph.node if n.name}
        for idx, node in enumerate(model.graph.node):
            if node.op_type != "Gather" or not node.input:
                continue
            dims = init_shapes.get(node.input[0])
            if dims and len(dims) == 2 and dims[0] == int(target_vocab):
                if not node.name:
                    cand = f"v05_embed_gather_{idx}"
                    while cand in used_names:
                        cand += "_x"
                    node.name = cand
                    used_names.add(cand)
                target_gathers.append(node.name)
        if not target_gathers:
            return {"ok": False, "error": "no_embedding_gather_found"}
        onnx.save(model, str(tmp_named))
        quant = MatMulNBitsQuantizer(
            model=str(tmp_named),
            bits=4,
            block_size=int(block_size),
            is_symmetric=False,
            nodes_to_include=target_gathers,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("Gather",),
            quant_axes=(("Gather", 1),),
        )
        quant.process()
        quant.model.save_model_to_file(str(tmp_w4))
        quantize_dynamic(
            str(tmp_w4),
            str(prelude_out),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        ref_s = ort.InferenceSession(str(prelude_fp32_in), sess_options=opts, providers=["CPUExecutionProvider"])
        w4_s = ort.InferenceSession(str(prelude_out), sess_options=opts, providers=["CPUExecutionProvider"])
        feed = {
            "input_ids": np.array([[2, 10, 50, 200, 3]], dtype=np.int64),
            "attention_mask": np.ones((1, 5), dtype=np.int64),
            "style_feats": np.zeros((1, STYLE_FEAT_DIM), dtype=np.float32),
        }
        diff = float(np.mean(np.abs(ref_s.run(None, feed)[0] - w4_s.run(None, feed)[0])))
        return {"ok": True, "bytes": int(prelude_out.stat().st_size), "z0_mae_vs_fp32": round(diff, 6), "target_gathers": target_gathers}
    except Exception as exc:
        prelude_out.unlink(missing_ok=True)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        tmp_named.unlink(missing_ok=True)
        tmp_w4.unlink(missing_ok=True)
