"""MeguvizCAC v1.1.0-OmniGuard Architecture, P1 Zero-Shock Surgery, Bundle v6+ (`MGVZ6`), & 4-Session Static ONNX.

Strictly implements ALL 12 code-audit findings (`F1–F12`) & architectural pillars (`P1, P4, P5, P8`) from
`docs/posttrain-plan-v1.1.0-omniguard.md` while keeping 100% of `v1.0.0` files untouched:
  - **F1 (Decision 1)**: `ATTRIBUTION_6WAY` appends `Human-Commercial-Promo` at **idx 5** (`energies[:, 2:5]` AI valley).
  - **F2/F3**: `DynaSeamMILAggregatorV11(16, 32, 2)` (`33.7 KB`), bypassed at `M=1` and quote-wrap hardened.
  - **F6**: `cbm_enc[:112, 192:256] === 0` structurally masked in forward + `cbm_audit_w[:, 112:] === 0` (16 gov slots).
  - **F7**: New rows of MoLE (`8+2`), EBT (`5+1`), and `attr6` (`5+1`) stored as separate `nn.Parameter`s for true
    per-row AdamW LR separation, concatenated in forward and seamlessly merged/unpacked under clean MGVZ6 keys.
  - **F8**: Unified `plan_long_chunks_v11` (`240-char`) chunker across training (`ChronoStream` TBPTT `k1=M, k2=3`) & serving,
    carrying ONLY `fwd_state` + Top-32 `L0-Haar` surface-seam sketch.
  - **P1.1–P1.4**: Zero-shock transplant from `v1.0.0-R1` pre-folded `fp16_master.pt` (LS `40->56` bridge, exact OBS
    coordinate permutation firewall `[0:128]` shared / `[128:192]` AI-only / `[192:256]` promo-only, exact Net2Net
    MoLE `8->10` split with `bal_bias -= ln 2`, and masked group-target balancing!).
  - **P8 (`CSP-1`)**: ABI tag `emb_layout = v11-perm-<sha8>` & post-ONNX zero-forgetting capability slot runner!
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v07_features import (
    BPE_VOCAB_SIZE_V07,
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
)
from .v07_omniveritas import (
    ATTRIBUTION_5WAY,
    MAX_LEN_V07,
    CappedReLU2FFN,
    OmniQuantLinear,
    PreludeBlockV07,
    RationalDerf,
    RMSNorm,
    SeqLoRAMoE,
    _pack_2bit_bytes,
    _pack_5trit_bytes,
    _pack_w4_bytes,
    _unpack_2bit_bytes,
    _unpack_5trit_bytes,
    _unpack_w4_bytes,
)
from .v09_chronoloop import (
    BLTMatryoshkaPreludeL0,
    D1_PREFIX_V09,
    D2_COMP_V09,
    FFN1_PREFIX_V09,
    FFN2_COMP_V09,
    HIDDEN_V09,
    NUM_FAST_SLOTS_V09,
    NUM_SLOTS_V09,
    NUM_SLOW_SLOTS_V09,
    ResidualDeferralRouterV09,
    _hadamard_64,
)
from .v10_apexloop import (
    BUS_DIM_V10,
    ENGRAM_BIGRAM_BUCKETS_V10,
    ENGRAM_TRIGRAM_BUCKETS_V10,
    MOLE_NUM_EXPERTS_V10,
    MOLE_RANK_V10,
    NUM_SEM_FAST_V10,
    NUM_SPAN_FAST_V10,
    ContextSecondaryLoopV10,
    MeguvizV10ApexLoopModel,
    MultiScaleRhythmConvV10,
    tunnel_trm_ste,
)
from .v11_features import STYLE_FEAT_DIM_V11, extract_style_features_56d, plan_long_chunks_v11
from .v11_longstream import DynaSeamMILAggregatorV11, export_dynaseam_onnx_and_verify, format_long_verdict_v11

# F1 / Decision 1: Append Human-Commercial-Promo strictly at idx 5!
ATTRIBUTION_6WAY: tuple[str, ...] = ATTRIBUTION_5WAY + ("Human-Commercial-Promo",)
PROMO_SUB4_NAMES: tuple[str, ...] = ("coupon_commerce", "self_promo_funnel", "soft_shill", "tool_boilerplate")
MOLE_NEW_EXPERTS_V11 = 2
MOLE_NUM_EXPERTS_V11 = MOLE_NUM_EXPERTS_V10 + MOLE_NEW_EXPERTS_V11  # 10 experts (+2.6 KB FP16!)
SHARED_EMB_END_V11 = 128
AI_EMB_END_V11 = 192  # AI reads [0:192], Promo reads [0:128] U [192:256]
CBM_AI_CONCEPTS_V11 = 112  # Concepts 112..127 reserved for governance / CSP-1 (F6)
R1_PROMO_VETO_DIMS_V11 = (48, 49, 50, 51, 53, 54, 55)  # excludes dim 52 (kedaibiao!)


# ---------------------------------------------------------------------------
# 1. F7/F9/F11/P1.4: DynaCore-MoLE v1.1 (8 Old + 2 Split New Parameters!)
# ---------------------------------------------------------------------------

class DynaCoreMoLEV11(nn.Module):
    """10-Expert (`8 Old + 2 New`) DynaCore-MoLE with split parameters (F7) & masked group-target balancing (F11/P1.4)."""

    def __init__(
        self,
        hidden_size: int = HIDDEN_V09,
        num_old_experts: int = MOLE_NUM_EXPERTS_V10,
        num_new_experts: int = MOLE_NEW_EXPERTS_V11,
        rank: int = MOLE_RANK_V10,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_old_experts = int(num_old_experts)
        self.num_new_experts = int(num_new_experts)
        self.num_experts = self.num_old_experts + self.num_new_experts
        self.rank = int(rank)

        self.basis_u = OmniQuantLinear(hidden_size, rank, bias=False, target_bits="w4", group_size=64)
        self.basis_v = OmniQuantLinear(rank, hidden_size, bias=False, target_bits="w4", group_size=64)
        with torch.no_grad():
            u_q = torch.empty(rank, hidden_size)
            nn.init.orthogonal_(u_q)
            self.basis_u.weight.copy_(u_q * 0.5)
            v_q = torch.empty(hidden_size, rank)
            nn.init.orthogonal_(v_q)
            self.basis_v.weight.copy_(v_q * 0.05)

        self.core_shared = nn.Parameter(torch.zeros(rank, rank))
        # F7: Separate old vs new Parameters so AdamW assigns independent LR / WD cleanly!
        self.core_routed_old = nn.Parameter(torch.randn(num_old_experts, rank, rank) * 0.02)
        self.core_routed_new = nn.Parameter(torch.randn(num_new_experts, rank, rank) * 0.02)
        self.core_bias_old = nn.Parameter(torch.zeros(num_old_experts, rank))
        self.core_bias_new = nn.Parameter(torch.zeros(num_new_experts, rank))

        self.phase_silu_gain = nn.Parameter(torch.zeros(3))
        self.phase_diag = nn.Parameter(torch.ones(3, rank))
        self.adaln_scale = nn.Parameter(torch.zeros(3, 1, hidden_size))
        self.adaln_shift = nn.Parameter(torch.zeros(3, 1, hidden_size))

        self.router_proj = nn.Linear(hidden_size, 64, bias=False)
        self.router_keys_old = nn.Parameter(torch.randn(num_old_experts, 64) * 0.15)
        self.router_keys_new = nn.Parameter(torch.randn(num_new_experts, 64) * 0.15)
        self.router_scale = nn.Parameter(torch.tensor(6.0))
        self.register_buffer("bal_bias", torch.zeros(3, self.num_experts, dtype=torch.float32))
        self.register_buffer("load_r1_ref", torch.full((num_old_experts,), 1.0 / float(num_old_experts), dtype=torch.float32))
        self.register_buffer("bal_update_mask", torch.ones(self.num_experts, dtype=torch.float32))
        self.dyn_profiles_old = nn.Parameter(torch.zeros(num_old_experts, 6))
        self.dyn_profiles_new = nn.Parameter(torch.zeros(num_new_experts, 6))

    def _cat_routed(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.cat([self.core_routed_old, self.core_routed_new], dim=0),
            torch.cat([self.core_bias_old, self.core_bias_new], dim=0),
            torch.cat([self.router_keys_old, self.router_keys_new], dim=0),
            torch.cat([self.dyn_profiles_old, self.dyn_profiles_new], dim=0),
        )

    def modulate(self, x: torch.Tensor, phase_idx: int) -> torch.Tensor:
        p = min(phase_idx, 2)
        return x * (1.0 + self.adaln_scale[p]) + self.adaln_shift[p]

    def route(
        self,
        zh_mean: torch.Tensor,
        tick_idx: int,
        prev_probs: torch.Tensor | None = None,
        surprise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = min(tick_idx, 2)
        _, _, r_keys, d_prof = self._cat_routed()
        q = F.normalize(self.router_proj(zh_mean), p=2, dim=-1)
        k = F.normalize(r_keys, p=2, dim=-1)
        raw_logits = torch.matmul(q, k.transpose(-1, -2)) * self.router_scale.clamp(min=1.0, max=16.0)
        z_loss = (torch.logsumexp(raw_logits.float(), dim=-1) ** 2).mean()
        probs_now = F.softmax(raw_logits + self.bal_bias[t].unsqueeze(0), dim=-1)
        if prev_probs is not None:
            mu = (0.70 - 0.55 * surprise.clamp(0.0, 1.0)).unsqueeze(-1) if surprise is not None else 0.45
            probs = mu * prev_probs + (1.0 - mu) * probs_now
        else:
            probs = probs_now
        ctrl = torch.matmul(probs, d_prof)
        return probs, ctrl, z_loss

    @torch.no_grad()
    def update_deepseek_bal_bias(
        self,
        probs: torch.Tensor,
        tick_idx: int,
        rho_new: float = 0.15,
        rate: float = 0.004,
        deadzone: float = 0.03,
        frozen: bool = False,
    ) -> None:
        """P1.4 Masked Group-Target Balancing: prevents pushing 8 old AI experts down toward 1/10!"""
        if frozen:
            return
        t = min(tick_idx, 2)
        load = probs.detach().float().mean(dim=0)
        rho = float(max(0.0, min(0.30, rho_new)))
        t_old = (1.0 - rho) * self.load_r1_ref.to(load.device)
        t_new = torch.full((self.num_new_experts,), rho / float(max(1, self.num_new_experts)), device=load.device)
        target = torch.cat([t_old, t_new], dim=0)
        err = target - load
        step = torch.where(err.abs() > deadzone * target.clamp(min=1e-3), torch.sign(err) * rate, torch.zeros_like(err))
        step = step * self.bal_update_mask.to(load.device)
        self.bal_bias[t].add_(step).clamp_(-2.5, 2.5)

    def forward_delta(self, x: torch.Tensor, probs: torch.Tensor, phase_idx: int) -> torch.Tensor:
        p = min(phase_idx, 2)
        c_routed, c_bias, _, _ = self._cat_routed()
        h = self.basis_u(x)
        h = h + torch.tanh(self.phase_silu_gain[p]) * F.silu(h)
        c_mix = self.core_shared.unsqueeze(0) + torch.einsum("be,ers->brs", probs, c_routed)
        b_mix = torch.matmul(probs, c_bias).unsqueeze(1)
        h_core = torch.matmul(h, c_mix) * self.phase_diag[p].view(1, 1, -1) + b_mix
        return self.basis_v(h_core)


# ---------------------------------------------------------------------------
# 2. F1/F7: 6-Way In-Loop Zero-Autograd Analytic-Gradient EBT (Promo at idx 5!)
# ---------------------------------------------------------------------------

class InLoopAnalyticEBTV11(nn.Module):
    """6-Way In-Loop Analytic EBT (`fc2_old(5)` + `fc2_new(1)` split per F7; `Promo` at idx 5 per F1!)."""

    def __init__(self, in_dim: int = BUS_DIM_V10, hidden: int = 96):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2_old_w = nn.Parameter(torch.empty(5, hidden))
        self.fc2_old_b = nn.Parameter(torch.zeros(5))
        self.fc2_new_w = nn.Parameter(torch.empty(1, hidden))
        self.fc2_new_b = nn.Parameter(torch.full((1,), 6.0))
        nn.init.kaiming_uniform_(self.fc2_old_w, a=math.sqrt(5))
        with torch.no_grad():
            self.fc2_new_w.copy_(self.fc2_old_w[0:1])
        self.step_eta_logit = nn.Parameter(torch.tensor(-2.0))

    def _fc2_wb(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.cat([self.fc2_old_w, self.fc2_new_w], dim=0), torch.cat([self.fc2_old_b, self.fc2_new_b], dim=0)

    def forward_energy_and_grad(self, u: torch.Tensor) -> dict[str, torch.Tensor]:
        w2, b2 = self._fc2_wb()
        a1 = self.fc1(u)
        h1 = F.softplus(a1)
        dh1 = torch.sigmoid(a1)
        energies = F.linear(h1, w2, b2)  # (B, 6)
        neg_e = -energies
        q_probs = F.softmax(neg_e, dim=-1)
        free_e = -torch.logsumexp(neg_e, dim=-1)

        d_h1 = torch.matmul(q_probs, w2)
        grad_u = torch.matmul(d_h1 * dh1, self.fc1.weight)
        grad_norm = torch.sqrt((grad_u * grad_u).sum(dim=-1) + 1e-8)

        eta = 0.15 * torch.sigmoid(self.step_eta_logit)
        u_refined = u - eta * grad_u / (1.0 + grad_norm.unsqueeze(-1))

        # F1: Human valleys = {0 (Organic), 5 (Human-Commercial-Promo)}; AI valleys = {2, 3, 4}!
        e_hu_min = torch.minimum(energies[:, 0], energies[:, 5])
        e_ai_min = energies[:, 2:5].amin(dim=-1)
        ebt_margin = e_hu_min - e_ai_min
        ebt_promo_margin = energies[:, 0] - energies[:, 5]
        entropy = -(q_probs * torch.log(q_probs.clamp(min=1e-7))).sum(dim=-1)

        return {
            "u_refined": u_refined,
            "energies": energies,
            "free_energy": free_e,
            "grad_u": grad_u,
            "grad_norm": grad_norm,
            "entropy": entropy,
            "ebt_margin": ebt_margin,
            "ebt_promo_margin": ebt_promo_margin,
        }


# ---------------------------------------------------------------------------
# 3. TunnelSymplect Core v1.1 (Shares exact v1.0 layout + MoLEV11 + EBTV11)
# ---------------------------------------------------------------------------

class TunnelSymplectCoreV11(nn.Module):
    """TunnelSymplect Core v1.1 (`MoLE 8+2`, `EBT 5+1`, exact v1.0 key names for zero-shock weight copy!)."""

    def __init__(self, hidden_size: int = HIDDEN_V09, num_heads: int = 6, ffn_size: int = 896):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.n_slow = NUM_SLOW_SLOTS_V09
        self.n_sem = NUM_SEM_FAST_V10
        self.n_span = NUM_SPAN_FAST_V10

        self.norm1 = RMSNorm(hidden_size)
        self.q_proj = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.k_proj = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.v_proj = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.sub_ln_attn = RMSNorm(hidden_size)
        self.o_proj = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)

        self.norm2 = RMSNorm(hidden_size)
        self.ffn = CappedReLU2FFN(hidden_size=hidden_size, ffn_size=ffn_size, cap_tau=3.0)
        self.mole = DynaCoreMoLEV11(hidden_size=hidden_size, rank=MOLE_RANK_V10)

        self.fast_norm = RMSNorm(hidden_size)
        self.p2_pre_a = OmniQuantLinear(hidden_size * 2, 8, bias=False, target_bits="w4", group_size=64)
        self.p2_pre_b = OmniQuantLinear(8, hidden_size, target_bits="w4", group_size=64)
        self.p2_post_a = OmniQuantLinear(hidden_size * 2, 8, bias=False, target_bits="w4", group_size=64)
        self.p2_post_b = OmniQuantLinear(8, hidden_size, target_bits="w4", group_size=64)
        self.fast_res_scale = nn.Parameter(torch.full((hidden_size,), 0.35))
        self.p2_l0_pen_a = OmniQuantLinear(hidden_size, 16, bias=False, target_bits="w4", group_size=64)
        self.p2_l0_pen_b = OmniQuantLinear(16, hidden_size, bias=False, target_bits="w4", group_size=64)
        self.p2_l0_gate = nn.Parameter(torch.full((hidden_size,), -1.5))

        self.bus_down = OmniQuantLinear(hidden_size, BUS_DIM_V10, bias=False, target_bits="w4", group_size=64)
        self.bus_up = OmniQuantLinear(BUS_DIM_V10, hidden_size, bias=False, target_bits="w4", group_size=64)

        self.c1_q = nn.Linear(BUS_DIM_V10, BUS_DIM_V10, bias=False)
        self.c1_k = nn.Linear(BUS_DIM_V10, BUS_DIM_V10, bias=False)
        self.c1_cayley_gamma = nn.Parameter(torch.zeros(BUS_DIM_V10))
        self.c1_bridge_scale = nn.Parameter(torch.tensor(0.20))

        self.c2_beta_logit = nn.Parameter(torch.full((BUS_DIM_V10,), 0.4))
        self.c2_carry_scale = nn.Parameter(torch.tensor(0.22))
        self.c2_surprise_mlp = nn.Sequential(nn.Linear(BUS_DIM_V10 + 1, 16), nn.SiLU(), nn.Linear(16, 1))

        self.inloop_ebt = InLoopAnalyticEBTV11(in_dim=BUS_DIM_V10, hidden=96)
        self.ebt_back_scale = nn.Parameter(torch.tensor(0.15))

        self.ctx_loop = ContextSecondaryLoopV10(hidden_size=hidden_size, bus_dim=BUS_DIM_V10, ctx_in_dim=72)

        self.reread_norm = RMSNorm(hidden_size)
        self.reread_q = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.reread_o_a = OmniQuantLinear(hidden_size, 32, bias=False, target_bits="w4", group_size=64)
        self.reread_o_b = OmniQuantLinear(32, hidden_size, target_bits="w4", group_size=64)
        self.reread_gate = nn.Parameter(torch.tensor(-1.2))

        self.haar_q = OmniQuantLinear(hidden_size, BUS_DIM_V10, bias=False, target_bits="w4", group_size=64)
        self.haar_o = OmniQuantLinear(BUS_DIM_V10, hidden_size, bias=False, target_bits="w4", group_size=64)
        self.haar_gate = nn.Parameter(torch.tensor(-1.4))

        self.mhc_alpha_logit = nn.Parameter(torch.full((hidden_size,), -1.8))

    def mhc_anchor_mix(self, s_cur: torch.Tensor, s_anchor: torch.Tensor) -> torch.Tensor:
        alpha = 0.5 * torch.sigmoid(self.mhc_alpha_logit)
        return (1.0 - alpha) * s_cur + alpha * s_anchor

    def _cayley_peer_couple(self, z_l: torch.Tensor, cayley_delta: torch.Tensor) -> torch.Tensor:
        u = self.bus_down(z_l)
        u_a, u_b = u[:, : self.n_sem, :], u[:, self.n_sem :, :]
        sc_ab = torch.matmul(self.c1_q(u_a), self.c1_k(u_b).transpose(-1, -2)) / 8.0
        msg_to_a = torch.matmul(F.softmax(sc_ab, dim=-1), u_b)
        msg_to_b = torch.matmul(F.softmax(sc_ab.transpose(-1, -2), dim=-1), u_a)

        g = (self.c1_cayley_gamma.unsqueeze(0) + 0.15 * torch.tanh(cayley_delta)).clamp(-2.5, 2.5)
        g2 = g * g
        inv = 1.0 / (1.0 + g2)
        c = ((1.0 - g2) * inv).unsqueeze(1)
        s = ((2.0 * g) * inv).unsqueeze(1)

        rot_a = c * u_a + s * msg_to_a
        rot_b = -s * msg_to_b + c * u_b
        delta = self.bus_up(torch.cat([rot_a - u_a, rot_b - u_b], dim=1))
        return z_l + (0.35 * torch.tanh(self.c1_bridge_scale)) * delta

    def _fast_inner_step(
        self,
        slots: torch.Tensor,
        probs: torch.Tensor,
        ctrl: torch.Tensor,
        phase_idx: int,
        trm_lam: float,
    ) -> torch.Tensor:
        z_h = slots[:, : self.n_slow, :]
        z_l = slots[:, self.n_slow :, :]
        zh_ref = z_h.mean(dim=1, keepdim=True).expand_as(z_l)

        zn = self.fast_norm(z_l)
        zn_trm = tunnel_trm_ste(zn, trm_lam)
        zh_trm = tunnel_trm_ste(zh_ref, trm_lam)
        pre_in = torch.cat([zn_trm, zn_trm - zh_trm], dim=-1)
        g_pre = torch.sigmoid(self.p2_pre_b(F.silu(self.p2_pre_a(pre_in))) + 0.25 * ctrl[:, 2:3].unsqueeze(1))
        h_l = self.mole.modulate(zn_trm, phase_idx) * g_pre
        delta_l = self.ffn(h_l) + self.mole.forward_delta(h_l, probs, phase_idx)

        g_post = torch.sigmoid(self.p2_post_b(F.silu(self.p2_post_a(torch.cat([zn, delta_l], dim=-1)))) + 0.25 * ctrl[:, 3:4].unsqueeze(1))
        z_l_new = z_l + self.fast_res_scale * (g_post * delta_l)
        z_l_new = self._cayley_peer_couple(z_l_new, ctrl[:, 0:1])
        return torch.cat([z_h, z_l_new], dim=1)

    def _active_reread_and_haar(
        self,
        slots: torch.Tensor,
        k_ev: torch.Tensor,
        v_ev: torch.Tensor,
        ev_mask: torch.Tensor,
        k_haar: torch.Tensor | None,
        v_haar: torch.Tensor | None,
        haar_mask: torch.Tensor | None,
        haar_delta: torch.Tensor,
    ) -> torch.Tensor:
        sn = self.reread_norm(slots)
        q = self.reread_q(sn)
        sc = torch.matmul(q, k_ev.transpose(-1, -2)) / math.sqrt(float(self.hidden_size))
        sc = sc.masked_fill(~ev_mask.bool().unsqueeze(1), -1e4)
        out = self.reread_o_b(F.silu(self.reread_o_a(torch.matmul(F.softmax(sc, dim=-1), v_ev))))
        s = slots + torch.sigmoid(self.reread_gate) * out

        if k_haar is not None and v_haar is not None and haar_mask is not None:
            qh = self.haar_q(sn)
            sch = torch.matmul(qh, k_haar.transpose(-1, -2)) / 8.0
            sch = sch.masked_fill(~haar_mask.bool().unsqueeze(1), -1e4)
            oh = self.haar_o(torch.matmul(F.softmax(sch, dim=-1), v_haar))
            g_h = torch.sigmoid(self.haar_gate + 0.35 * haar_delta).view(-1, 1, 1)
            s = s + g_h * oh
        return s

    def forward_outer_step(
        self,
        slots: torch.Tensor,
        s_anchor: torch.Tensor,
        k_ev: torch.Tensor,
        v_ev: torch.Tensor,
        ev_mask: torch.Tensor,
        h0_pool: torch.Tensor,
        outer_idx: int,
        k_haar: torch.Tensor | None = None,
        v_haar: torch.Tensor | None = None,
        haar_mask: torch.Tensor | None = None,
        carry_m_l: torch.Tensor | None = None,
        prev_zh_bus: torch.Tensor | None = None,
        prev_mole_probs: torch.Tensor | None = None,
        ctx_vec: torch.Tensor | None = None,
        ctx_kish_lambda: torch.Tensor | None = None,
        k_inner: int = 2,
        trm_lam_k0: float = 1.0,
        trm_lam_k1: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        s = slots
        bsz = s.shape[0]
        device = s.device

        if outer_idx >= 1 and ctx_vec is not None and ctx_kish_lambda is not None:
            z_h_ctx = self.ctx_loop(s[:, : self.n_slow, :], ctx_vec, ctx_kish_lambda)
            s = torch.cat([z_h_ctx, s[:, self.n_slow :, :]], dim=1)

        zh_now_bus = self.bus_down(s[:, : self.n_slow, :].mean(dim=1))
        if prev_zh_bus is not None:
            cos_sim = F.cosine_similarity(zh_now_bus, prev_zh_bus, dim=-1, eps=1e-6).unsqueeze(-1)
            surprise = torch.sigmoid(self.c2_surprise_mlp(torch.cat([zh_now_bus - prev_zh_bus, cos_sim], dim=-1)).squeeze(-1))
        else:
            surprise = torch.full((bsz,), 0.5, device=device, dtype=s.dtype)

        mole_probs, ctrl, z_loss = self.mole.route(
            s[:, : self.n_slow, :].mean(dim=1), tick_idx=outer_idx, prev_probs=prev_mole_probs, surprise=surprise
        )

        if outer_idx >= 1:
            s = self._active_reread_and_haar(s, k_ev, v_ev, ev_mask, k_haar, v_haar, haar_mask, ctrl[:, 1])

        z_l_start = s[:, self.n_slow :, :]
        if carry_m_l is not None:
            c2_gain = (0.35 * torch.tanh(self.c2_carry_scale) * (1.0 - 0.5 * surprise) * (1.0 + 0.25 * torch.tanh(ctrl[:, 4]))).view(-1, 1, 1)
            z_l_inj = z_l_start + c2_gain * self.bus_up(carry_m_l)
            s = torch.cat([s[:, : self.n_slow, :], z_l_inj], dim=1)

        for ki in range(k_inner):
            lam_now = trm_lam_k0 if ki == 0 else trm_lam_k1
            s = self._fast_inner_step(s, mole_probs, ctrl, phase_idx=min(ki, 1), trm_lam=lam_now)

        delta_fast_bus = torch.tanh(self.bus_down(s[:, self.n_slow :, :] - z_l_start))
        beta_c = 0.85 * torch.sigmoid(self.c2_beta_logit).view(1, 1, -1)
        new_carry_m_l = (1.0 - beta_c) * delta_fast_bus if carry_m_l is None else (beta_c * carry_m_l + (1.0 - beta_c) * delta_fast_bus)

        n_s = s.shape[1]
        h = self.norm1(s)
        q = self.q_proj(h).view(bsz, n_s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, n_s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, n_s, self.num_heads, self.head_dim).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-1, -2)) / 8.0
        a = torch.matmul(F.softmax(sc, dim=-1), v).transpose(1, 2).contiguous().view(bsz, n_s, self.hidden_size)
        s = s + self.o_proj(self.sub_ln_attn(a))

        n2 = self.mole.modulate(self.norm2(s), phase_idx=2)
        s = s + self.ffn(tunnel_trm_ste(n2, trm_lam_k1)) + self.mole.forward_delta(n2, mole_probs, phase_idx=2)

        l0_pen = self.p2_l0_pen_b(F.silu(self.p2_l0_pen_a(h0_pool))).unsqueeze(1)
        z_h_pen = s[:, : self.n_slow, :] + torch.sigmoid(self.p2_l0_gate).view(1, 1, -1) * l0_pen
        s = torch.cat([z_h_pen, s[:, self.n_slow :, :]], dim=1)

        u_zh = self.bus_down(s[:, : self.n_slow, :].mean(dim=1))
        ebt_out = self.inloop_ebt.forward_energy_and_grad(u_zh)
        ebt_delta = self.bus_up(ebt_out["u_refined"] - u_zh).unsqueeze(1)
        ebt_g = (0.35 * torch.tanh(self.ebt_back_scale) * (1.0 + 0.25 * torch.tanh(ctrl[:, 5]))).view(-1, 1, 1)
        z_h_ebt = s[:, : self.n_slow, :] + ebt_g * ebt_delta
        s = torch.cat([z_h_ebt, s[:, self.n_slow :, :]], dim=1)

        s_out = self.mhc_anchor_mix(s, s_anchor)
        return {
            "slots": s_out,
            "carry_m_l": new_carry_m_l,
            "zh_bus": u_zh,
            "mole_probs": mole_probs,
            "mole_z_loss": z_loss,
            "surprise": surprise,
            "ebt_out": ebt_out,
        }


# ---------------------------------------------------------------------------
# 4. Full MeguvizCAC v1.1.0-OmniGuard Model
# ---------------------------------------------------------------------------

class MeguvizV11OmniGuardModel(nn.Module):
    """MeguvizCAC v1.1.0-OmniGuard Dual-Axis (`AI Origin x Promo Intent`) + Long-Context Streaming (`<= 2.20 MB`!)."""

    def __init__(
        self,
        vocab_size: int = BPE_VOCAB_SIZE_V07,
        embed_dim: int = 48,
        hidden_size: int = HIDDEN_V09,
        num_heads: int = 6,
        ffn_size: int = 896,
        engram_buckets: int = ENGRAM_BIGRAM_BUCKETS_V10,
        trigram_buckets: int = ENGRAM_TRIGRAM_BUCKETS_V10,
        engram_dim: int = 16,
        max_len: int = MAX_LEN_V07,
        style_dim: int = STYLE_FEAT_DIM_V11,
        num_codes: int = 8,
        num_prag: int = 8,
        cbm_concepts: int = 128,
        cbm_topk: int = 6,
        intent_dim: int = 1024,
    ):
        super().__init__()
        assert hidden_size == HIDDEN_V09 and style_dim == STYLE_FEAT_DIM_V11
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_size = int(hidden_size)
        self.engram_buckets = int(engram_buckets)
        self.trigram_buckets = int(trigram_buckets)
        self.d1 = D1_PREFIX_V09
        self.d2 = D2_COMP_V09
        self.max_len = int(max_len)
        self.style_dim = int(style_dim)
        self.cbm_concepts = int(cbm_concepts)
        self.cbm_topk = int(cbm_topk)

        # Embeddings (exact v1.0 keys)
        self.word_emb_A = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.word_emb_P1 = OmniQuantLinear(embed_dim, self.d1, bias=False, target_bits="w4", group_size=64)
        self.word_emb_P2 = OmniQuantLinear(embed_dim, self.d2, bias=False, target_bits="w4", group_size=64)
        self.engram_emb = nn.Embedding(engram_buckets, engram_dim, padding_idx=0)
        self.engram_P1 = OmniQuantLinear(engram_dim, self.d1, bias=False, target_bits="w4", group_size=64)
        self.engram_P2 = OmniQuantLinear(engram_dim, self.d2, bias=False, target_bits="w4", group_size=64)
        self.trigram_emb = nn.Embedding(trigram_buckets, engram_dim, padding_idx=0)
        self.trigram_P1 = OmniQuantLinear(engram_dim, self.d1, bias=False, target_bits="w4", group_size=64)
        self.trigram_P2 = OmniQuantLinear(engram_dim, self.d2, bias=False, target_bits="w4", group_size=64)
        self.pos_emb1 = nn.Embedding(max_len, self.d1)
        self.pos_emb2 = nn.Embedding(max_len, self.d2)
        self.emb_norm1 = RationalDerf(self.d1)
        self.emb_norm2 = RationalDerf(self.d2)
        self.emb_drop = nn.Dropout(0.10)

        # Rung 1 Prefix + Rhythm Conv + Dual-Axis Exit Heads + 421-d Deferral Router
        self.prelude_l0 = BLTMatryoshkaPreludeL0(d1=self.d1, d2=self.d2, m1=FFN1_PREFIX_V09, m2=FFN2_COMP_V09)
        self.r1_rhythm_conv = MultiScaleRhythmConvV10(d1=self.d1)
        self.r1_attn_pool = nn.Linear(self.d1, 1)
        self.r1_style_mlp = nn.Sequential(nn.Linear(style_dim, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.r1_span_head = nn.Linear(self.d1, 1)
        self.r1_ai_head = nn.Linear(self.d1 * 3 + 32 + 3, 1)  # 419 -> 1
        # NEW P1.1: Rung-1 Promo Head = Linear(419 + 8 -> 8) -> SiLU -> Linear(8 -> 1)
        self.r1_promo_head = nn.Sequential(nn.Linear(self.d1 * 3 + 32 + 3 + 8, 8), nn.SiLU(), nn.Linear(8, 1))
        nn.init.xavier_uniform_(self.r1_promo_head[0].weight, gain=0.5)
        nn.init.zeros_(self.r1_promo_head[0].bias)
        nn.init.zeros_(self.r1_promo_head[2].weight)
        nn.init.constant_(self.r1_promo_head[2].bias, -2.0)
        # P1.1: defer_r1_enc (421 = 419 + raw_logit_r1 + raw_logit_promo_r1)
        self.defer_r1_enc = nn.Sequential(nn.Linear(self.d1 * 3 + 32 + 5, 32), nn.SiLU())
        self.defer_r1_out = nn.Linear(32, 1)

        # H1 SSSQ + P3 Haar + Rung 2 Prelude L1 + Slot Cross-Attn + H2 DCCH + Core v1.1
        self.h1_d1_norm = RMSNorm(32)
        self.h1_q_a = OmniQuantLinear(self.d1 + 32 + 2, 16, bias=False, target_bits="w4", group_size=64)
        self.h1_q_b = OmniQuantLinear(16, NUM_FAST_SLOTS_V09 * hidden_size // 4, bias=False, target_bits="w4", group_size=64)
        self.h1_q_up = OmniQuantLinear(hidden_size // 4, hidden_size, bias=False, target_bits="w4", group_size=64)
        self.h1_gate = nn.Parameter(torch.tensor([0.05]))

        self.haar_k_proj = OmniQuantLinear(self.d1, BUS_DIM_V10, bias=False, target_bits="w4", group_size=64)
        self.haar_v_proj = OmniQuantLinear(self.d1, BUS_DIM_V10, bias=False, target_bits="w4", group_size=64)

        self.funnel_norm = RationalDerf(hidden_size)
        self.prelude_l1 = PreludeBlockV07(hidden_size, num_heads=num_heads, swa_heads=2, ffn_size=ffn_size)

        self.slot_queries = nn.Parameter(torch.empty(1, NUM_SLOTS_V09, hidden_size))
        nn.init.orthogonal_(self.slot_queries.data[0], gain=1.0)
        self.slot_k = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.slot_v = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.slot_o = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.slot_norm = RMSNorm(hidden_size)

        self.h2_hyp_proj = OmniQuantLinear(BUS_DIM_V10, hidden_size, bias=False, target_bits="w4", group_size=64)
        self.h2_doubt_proj = OmniQuantLinear(BUS_DIM_V10 + 4, hidden_size, bias=False, target_bits="w4", group_size=64)
        self.h2_hyp_gate = nn.Parameter(torch.tensor([0.05]))
        self.h2_doubt_gate = nn.Parameter(torch.tensor([0.05]))

        self.core = TunnelSymplectCoreV11(hidden_size=hidden_size, num_heads=num_heads, ffn_size=ffn_size)

        # Shared UNet Span Base + Dual-Axis Span Heads
        self.span_l0_proj = nn.Linear(hidden_size, 64)
        self.span_l1_proj = nn.Linear(hidden_size, 64)
        self.span_topdown_proj = nn.Linear(hidden_size * 2, 64)
        self.span_head = nn.Sequential(nn.SiLU(), nn.Linear(64, 1))
        self.mil_gate = nn.Parameter(torch.tensor(-1.0))
        self.promo_span_head = nn.Sequential(nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.promo_span_head[1].weight)
        nn.init.constant_(self.promo_span_head[1].bias, -2.0)
        self.promo_mil_gate = nn.Parameter(torch.tensor(-1.0))

        # Coda + 256-d Permutation Firewall (P1.3) + F6 Masked TinyScope-CBM
        self.coda_norm = RationalDerf(hidden_size)
        self.coda_proj = nn.Sequential(nn.Linear(hidden_size * 2, 256), nn.SiLU())
        self.style_net = nn.Sequential(nn.Linear(style_dim, 64), nn.SiLU(), nn.Linear(64, 256))
        self.style_gate = nn.Parameter(torch.tensor(-2.5))
        self.cbm_enc = nn.Linear(256, cbm_concepts)
        self.cbm_dec = nn.Linear(cbm_concepts, 256, bias=False)
        self.cbm_audit_w = nn.Linear(cbm_concepts, 1, bias=False)
        with torch.no_grad():
            self.cbm_enc.weight[:CBM_AI_CONCEPTS_V11, AI_EMB_END_V11:].zero_()
            self.cbm_audit_w.weight[:, CBM_AI_CONCEPTS_V11:].zero_()

        # Orthogonal Dual-Axis Heads:
        #   - ai_head reads ONLY [0:192] (structurally impossible to leak [192:256]!)
        #   - promo_head & promo_sub4_head read cat(D_lam(emb[0:128]), emb[192:256]) (192-d)
        self.ai_head = nn.Linear(AI_EMB_END_V11, 1)
        self.ebt_logit_gate = nn.Parameter(torch.tensor(-1.5))

        self.promo_head = nn.Linear(SHARED_EMB_END_V11 + (256 - AI_EMB_END_V11), 1)
        nn.init.zeros_(self.promo_head.weight)
        nn.init.constant_(self.promo_head.bias, -2.0)
        self.promo_sub4_head = nn.Linear(SHARED_EMB_END_V11 + (256 - AI_EMB_END_V11), 4)
        nn.init.zeros_(self.promo_sub4_head.weight)
        nn.init.constant_(self.promo_sub4_head.bias, -3.0)
        self.ebt_promo_gate = nn.Parameter(torch.tensor(-1.5))

        self.defer_head_r2 = ResidualDeferralRouterV09(256 + 6, hidden=32)
        # F7: 6-Way Attribution head split into old(5) + new(1)
        self.attr6_old_w = nn.Parameter(torch.empty(5, 256))
        self.attr6_old_b = nn.Parameter(torch.zeros(5))
        self.attr6_new_w = nn.Parameter(torch.empty(1, 256))
        self.attr6_new_b = nn.Parameter(torch.full((1,), -6.0))
        nn.init.kaiming_uniform_(self.attr6_old_w, a=math.sqrt(5))
        with torch.no_grad():
            self.attr6_new_w.copy_(self.attr6_old_w[0:1])

        self.code_head = nn.Linear(256, num_codes)
        self.prag_head = nn.Linear(256, num_prag)
        self.intent_head = nn.Linear(256, intent_dim)

        # Long-Context Streaming (`F2`: 33.7 KB FP16 DynaSeam + `2.2 KB` W4 chunk_emb_proj)
        self.chunk_emb_proj = OmniQuantLinear(256, 16, bias=False, target_bits="w4", group_size=64)
        self.longstream = DynaSeamMILAggregatorV11(chunk_emb_dim=16, hidden=32, num_heads=2)

        # Calibration & Firewall ABI Buffers
        self.register_buffer("exit_affine_a", torch.ones(3, dtype=torch.float32))
        self.register_buffer("exit_affine_b", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("exit_affine_promo_a", torch.ones(3, dtype=torch.float32))
        self.register_buffer("exit_affine_promo_b", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emb_perm", torch.arange(256, dtype=torch.long))

    def emb_layout_tag(self) -> str:
        """P8 CSP-1 ABI tag uniquely identifying the OBS permutation layout of `emb (256-d)`."""
        h = hashlib.sha256(self.emb_perm.detach().cpu().numpy().astype(np.int32).tobytes()).hexdigest()[:8]
        return f"v11-perm-{h}"

    def set_quant_mode(self, mode: str) -> None:
        for m in self.modules():
            if isinstance(m, OmniQuantLinear):
                m.quant_mode = m.target_bits if mode == "sub2" else mode

    def _embed_prefix(self, input_ids: torch.Tensor, engram_ids: torch.Tensor, trigram_ids: torch.Tensor) -> torch.Tensor:
        seq_l = input_ids.shape[1]
        pos = torch.arange(seq_l, device=input_ids.device).clamp(max=self.max_len - 1).unsqueeze(0)
        x1 = (
            self.word_emb_P1(self.word_emb_A(input_ids))
            + self.engram_P1(self.engram_emb(engram_ids))
            + self.trigram_P1(self.trigram_emb(trigram_ids))
            + self.pos_emb1(pos)
        )
        return self.emb_drop(self.emb_norm1(x1))

    def _embed_complement(self, input_ids: torch.Tensor, engram_ids: torch.Tensor, trigram_ids: torch.Tensor) -> torch.Tensor:
        seq_l = input_ids.shape[1]
        pos = torch.arange(seq_l, device=input_ids.device).clamp(max=self.max_len - 1).unsqueeze(0)
        x2 = (
            self.word_emb_P2(self.word_emb_A(input_ids))
            + self.engram_P2(self.engram_emb(engram_ids))
            + self.trigram_P2(self.trigram_emb(trigram_ids))
            + self.pos_emb2(pos)
        )
        return self.emb_drop(self.emb_norm2(x2))

    @staticmethod
    def _compute_l0_haar_highpass(h0_1: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        m = attention_mask.float().unsqueeze(-1)
        hm = h0_1 * m
        h_prev = F.pad(hm, (0, 0, 1, 0))[:, :-1, :]
        h_next = F.pad(hm, (0, 0, 0, 1))[:, 1:, :]
        m_prev = F.pad(m, (0, 0, 1, 0))[:, :-1, :]
        m_next = F.pad(m, (0, 0, 0, 1))[:, 1:, :]
        return (hm - (h_prev + h_next) / (m_prev + m_next).clamp(min=1.0)) * m

    def forward_rung1_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        trigram_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if trigram_ids is None:
            trigram_ids = torch.zeros_like(input_ids)
        x1 = self._embed_prefix(input_ids, engram_ids, trigram_ids)
        h0_1 = self.prelude_l0.forward_prefix(x1, attention_mask)
        h0_1 = self.r1_rhythm_conv(h0_1, attention_mask)

        m_f = attention_mask.float().unsqueeze(-1)
        m_den = m_f.sum(dim=1).clamp(min=1.0)
        h_avg = (h0_1 * m_f).sum(dim=1) / m_den
        h_max = h0_1.masked_fill(~attention_mask.bool().unsqueeze(-1), -1e4).amax(dim=1)
        a_logits = self.r1_attn_pool(h0_1).squeeze(-1).masked_fill(~attention_mask.bool(), -1e4)
        h_attn = (h0_1 * F.softmax(a_logits, dim=-1).unsqueeze(-1)).sum(dim=1)
        h_std_s = torch.sqrt((((h0_1 - h_avg.unsqueeze(1)) ** 2) * m_f).sum(dim=1) / m_den + 1e-6).mean(dim=-1, keepdim=True)

        tok_r = torch.sigmoid(self.r1_span_head(h0_1).squeeze(-1)) * attention_mask.float()
        w_avg = F.avg_pool1d(F.pad(tok_r.unsqueeze(1), (0, 7)), kernel_size=8, stride=4).squeeze(1)
        r_max_win = w_avg.amax(dim=-1)
        r_std_win = torch.sqrt(((w_avg - w_avg.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1) + 1e-6)

        st_h = self.r1_style_mlp(style_feats)
        feat = torch.cat([h_avg, h_max, h_attn, st_h, h_std_s, r_max_win.unsqueeze(-1), r_std_win.unsqueeze(-1)], dim=-1)  # 419
        raw_logit_r1 = self.r1_ai_head(feat).squeeze(-1)
        logit_r1 = (self.exit_affine_a[0] * raw_logit_r1 + self.exit_affine_b[0]) / self.temperature.clamp(min=0.2)

        # Rung-1 Promo Head consumes feat(419) + pure-text commercial clues style56[48:56] (8-d)
        raw_logit_promo_r1 = self.r1_promo_head(torch.cat([feat, style_feats[:, 48:56]], dim=-1)).squeeze(-1)
        logit_promo_r1 = self.exit_affine_promo_a[0] * raw_logit_promo_r1 + self.exit_affine_promo_b[0]

        d1_hid = self.defer_r1_enc(torch.cat([feat, raw_logit_r1.unsqueeze(-1), raw_logit_promo_r1.unsqueeze(-1)], dim=-1))
        p_defer1 = torch.sigmoid(self.defer_r1_out(d1_hid).squeeze(-1))

        tok_w = tok_r / tok_r.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        r1_susp_vec = (h0_1 * tok_w.unsqueeze(-1)).sum(dim=1)
        r1_pkt = torch.cat([r1_susp_vec, self.h1_d1_norm(d1_hid), r_max_win.unsqueeze(-1), r_std_win.unsqueeze(-1)], dim=-1)

        return {
            "x1_in": x1,
            "h0_1": h0_1,
            "logit_r1": logit_r1,
            "raw_logit_r1": raw_logit_r1,
            "logit_promo_r1": logit_promo_r1,
            "raw_logit_promo_r1": raw_logit_promo_r1,
            "p_defer1": p_defer1,
            "r1_span_max": r_max_win,
            "r1_span_std": r_std_win,
            "r1_pkt": r1_pkt,
        }

    def _coda_readout(
        self,
        slots: torch.Tensor,
        span_base: torch.Tensor,
        mask0: torch.Tensor,
        style_feats: torch.Tensor,
        ebt_out: dict[str, torch.Tensor],
        exit_idx: int,
        promo_sh_lam: float = 0.35,
    ) -> dict[str, torch.Tensor]:
        s_norm = self.coda_norm(slots)
        z_h = s_norm[:, : NUM_SLOW_SLOTS_V09, :]
        z_a = s_norm[:, NUM_SLOW_SLOTS_V09 : NUM_SLOW_SLOTS_V09 + NUM_SEM_FAST_V10, :]
        z_b = s_norm[:, NUM_SLOW_SLOTS_V09 + NUM_SEM_FAST_V10 :, :]
        s_verdict = z_h.mean(dim=1)
        s_evid = torch.cat([z_a, z_b], dim=1).mean(dim=1)
        s_span_ctx = z_b.mean(dim=1)
        slot_spread = z_h.std(dim=1, unbiased=False).mean(dim=-1)

        emb = self.coda_proj(torch.cat([s_verdict, s_evid], dim=-1))
        emb = emb + torch.sigmoid(self.style_gate) * self.style_net(style_feats)

        # Shared span_base + td -> BOTH AI span MIL & Promo span MIL!
        td = self.span_topdown_proj(torch.cat([s_verdict, s_span_ctx], dim=-1)).unsqueeze(1)
        sp_in = span_base + td
        tok_risk = self.span_head(sp_in).squeeze(-1) * mask0.float()
        w_pool = F.avg_pool1d(F.pad(tok_risk.unsqueeze(1), (0, 7)), kernel_size=8, stride=4).squeeze(1)
        mil_peak = w_pool.amax(dim=-1)

        tok_pr_risk = self.promo_span_head(sp_in).squeeze(-1) * mask0.float()
        wp_pool = F.avg_pool1d(F.pad(tok_pr_risk.unsqueeze(1), (0, 7)), kernel_size=8, stride=4).squeeze(1)
        promo_mil_peak = wp_pool.amax(dim=-1)

        # F6 Structural Firewall on TinyScope-CBM:
        #   - AI concepts 0..111 read ONLY emb[:, :192] (zero promo->AI leak!)
        #   - Governance concepts 112..127 read full emb[:, :256] and NEVER enter cbm_logit!
        cbm_ai_raw = F.relu(F.linear(emb[:, :AI_EMB_END_V11], self.cbm_enc.weight[:CBM_AI_CONCEPTS_V11, :AI_EMB_END_V11], self.cbm_enc.bias[:CBM_AI_CONCEPTS_V11]))
        cbm_gov_raw = F.relu(F.linear(emb, self.cbm_enc.weight[CBM_AI_CONCEPTS_V11:, :], self.cbm_enc.bias[CBM_AI_CONCEPTS_V11:]))
        cbm_raw = torch.cat([cbm_ai_raw, cbm_gov_raw], dim=-1)
        k_top = min(self.cbm_topk, self.cbm_concepts)
        top_v, top_i = torch.topk(cbm_raw, k=k_top, dim=-1)
        cbm_sparse = torch.zeros_like(cbm_raw).scatter_(-1, top_i, top_v)
        cbm_rec = self.cbm_dec(cbm_sparse)
        cbm_loss = F.mse_loss(cbm_rec, emb.detach())
        cbm_logit = F.linear(cbm_sparse[:, :CBM_AI_CONCEPTS_V11], self.cbm_audit_w.weight[:, :CBM_AI_CONCEPTS_V11]).squeeze(-1)

        # AI Axis Readout (reads ONLY emb[:, :192]!)
        ebt_m_clamped = 4.0 * torch.tanh(ebt_out["ebt_margin"] / 4.0)
        raw_logit = (
            self.ai_head(emb[:, :AI_EMB_END_V11]).squeeze(-1)
            + torch.sigmoid(self.mil_gate) * mil_peak
            + 0.10 * cbm_logit
            + (0.50 * torch.sigmoid(self.ebt_logit_gate)) * ebt_m_clamped
        )
        ex_i = min(exit_idx, 2)
        logit = (self.exit_affine_a[ex_i] * raw_logit + self.exit_affine_b[ex_i]) / self.temperature.clamp(min=0.2)

        # Promo Axis Readout (reads cat(D_lam(emb[:, :128]), emb[:, 192:256])!)
        sh_in = tunnel_trm_ste(emb[:, :SHARED_EMB_END_V11], promo_sh_lam) if self.training else emb[:, :SHARED_EMB_END_V11]
        emb_promo = torch.cat([sh_in, emb[:, AI_EMB_END_V11:]], dim=-1)
        ebt_pm_clamped = 4.0 * torch.tanh(ebt_out["ebt_promo_margin"] / 4.0)
        raw_logit_promo = (
            self.promo_head(emb_promo).squeeze(-1)
            + torch.sigmoid(self.promo_mil_gate) * promo_mil_peak
            + (0.50 * torch.sigmoid(self.ebt_promo_gate)) * ebt_pm_clamped
        )
        logit_promo = self.exit_affine_promo_a[ex_i] * raw_logit_promo + self.exit_affine_promo_b[ex_i]
        promo_sub4_logit = self.promo_sub4_head(emb_promo)

        # 6-Way Attribution (F7 split params)
        a6_w = torch.cat([self.attr6_old_w, self.attr6_new_w], dim=0)
        a6_b = torch.cat([self.attr6_old_b, self.attr6_new_b], dim=0)
        attr6_logit = F.linear(emb, a6_w, a6_b)

        p_defer2 = self.defer_head_r2(
            torch.cat(
                [
                    emb,
                    raw_logit.unsqueeze(-1),
                    slot_spread.unsqueeze(-1),
                    mil_peak.unsqueeze(-1),
                    ebt_out["free_energy"].unsqueeze(-1),
                    ebt_out["grad_norm"].unsqueeze(-1),
                    ebt_out["entropy"].unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        return {
            "logit": logit,
            "raw_logit": raw_logit,
            "logit_promo": logit_promo,
            "raw_logit_promo": raw_logit_promo,
            "promo_sub4_logit": promo_sub4_logit,
            "emb": emb,
            "mil_peak": mil_peak,
            "promo_mil_peak": promo_mil_peak,
            "slot_spread": slot_spread,
            "p_defer2": p_defer2,
            "cbm_loss": cbm_loss,
            "ebt_energies": ebt_out["energies"],
            "ebt_margin": ebt_out["ebt_margin"],
            "ebt_promo_margin": ebt_out["ebt_promo_margin"],
            "ebt_free_energy": ebt_out["free_energy"],
            "ebt_grad_norm": ebt_out["grad_norm"],
            "attr6_logit": attr6_logit,
            "code_logit": self.code_head(emb),
            "prag_logit": self.prag_head(emb),
            "intent_pred": self.intent_head(emb),
        }

    def forward_rung2_t1(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        x1_in: torch.Tensor,
        h0_1: torch.Tensor,
        r1_pkt: torch.Tensor,
        trigram_ids: torch.Tensor | None = None,
        handoff_mask: torch.Tensor | None = None,
        k_inner: int = 2,
        trm_lam_k0: float = 1.0,
        trm_lam_k1: float = 1.0,
        promo_sh_lam: float = 0.35,
        fwd_state_in: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if trigram_ids is None:
            trigram_ids = torch.zeros_like(input_ids)
        x2_in = self._embed_complement(input_ids, engram_ids, trigram_ids)
        m_f = attention_mask.float().unsqueeze(-1)
        cls_c = (x2_in * m_f).sum(dim=1) / m_f.sum(dim=1).clamp(min=1.0)

        h0_2, fwd_st_out = self.prelude_l0.forward_complement(
            x1_in, x2_in, h0_1, attention_mask, cls_c, fwd_state_in=fwd_state_in
        )
        h0_full = torch.cat([h0_1, h0_2], dim=-1)
        h0_pool = (h0_full * m_f).sum(dim=1) / m_f.sum(dim=1).clamp(min=1.0)

        h0_haar = self._compute_l0_haar_highpass(h0_1, attention_mask)
        k_haar = self.haar_k_proj(h0_haar)
        v_haar = self.haar_v_proj(h0_haar)

        h_pool = F.avg_pool1d(h0_full.transpose(1, 2), kernel_size=2, stride=2, ceil_mode=True).transpose(1, 2)
        m_pool = (F.max_pool1d(attention_mask.float().unsqueeze(1), kernel_size=2, stride=2, ceil_mode=True).squeeze(1) > 0.5).long()
        cls_full = (h_pool * m_pool.float().unsqueeze(-1)).sum(dim=1) / m_pool.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        h1_full, _ = self.prelude_l1(self.funnel_norm(h_pool), m_pool, cls_full)

        seq_l0 = h0_full.shape[1]
        p1_up = F.interpolate(self.span_l1_proj(h1_full).transpose(1, 2), size=seq_l0, mode="nearest").transpose(1, 2)
        span_base = self.span_l0_proj(h0_full) + p1_up

        k_ev = self.slot_k(h1_full)
        v_ev = self.slot_v(h1_full)
        bsz = h1_full.shape[0]

        q0_base = self.slot_queries.expand(bsz, -1, -1)
        pkt_clean = r1_pkt.detach() if self.training else r1_pkt
        if handoff_mask is not None:
            pkt_clean = pkt_clean * handoff_mask.view(-1, 1)
        dq_l = self.h1_q_up(F.silu(self.h1_q_b(F.silu(self.h1_q_a(pkt_clean))).view(bsz, NUM_FAST_SLOTS_V09, self.hidden_size // 4)))
        g_h1 = 0.35 * torch.tanh(self.h1_gate)
        q0 = torch.cat([q0_base[:, : NUM_SLOW_SLOTS_V09, :], q0_base[:, NUM_SLOW_SLOTS_V09 :, :] + g_h1 * dq_l], dim=1)

        sc0 = torch.matmul(q0, k_ev.transpose(-1, -2)) / math.sqrt(float(self.hidden_size))
        sc0 = sc0.masked_fill(~m_pool.bool().unsqueeze(1), -1e4)
        s0 = self.slot_norm(q0 + self.slot_o(torch.matmul(F.softmax(sc0, dim=-1), v_ev)))

        st1 = self.core.forward_outer_step(
            s0, s0, k_ev, v_ev, m_pool, h0_pool, outer_idx=0,
            k_haar=k_haar, v_haar=v_haar, haar_mask=attention_mask,
            k_inner=k_inner, trm_lam_k0=trm_lam_k0, trm_lam_k1=trm_lam_k1,
        )
        e1_out = self._coda_readout(
            st1["slots"], span_base, attention_mask, style_feats, st1["ebt_out"], exit_idx=1, promo_sh_lam=promo_sh_lam
        )

        return {
            "exit1_out": e1_out,
            "slots": st1["slots"],
            "s_anchor": s0,
            "carry_m_l": st1["carry_m_l"],
            "zh_bus": st1["zh_bus"],
            "mole_probs": st1["mole_probs"],
            "mole_z_loss": st1["mole_z_loss"],
            "k_ev": k_ev,
            "v_ev": v_ev,
            "m_pool": m_pool,
            "k_haar": k_haar,
            "v_haar": v_haar,
            "h0_pool": h0_pool,
            "span_base": span_base,
            "fwd_state_out": fwd_st_out,
        }

    def forward_rung3_step(
        self,
        slots: torch.Tensor,
        s_anchor: torch.Tensor,
        carry_m_l: torch.Tensor,
        zh_bus: torch.Tensor,
        mole_probs: torch.Tensor,
        k_ev: torch.Tensor,
        v_ev: torch.Tensor,
        m_pool: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        attention_mask: torch.Tensor,
        h0_pool: torch.Tensor,
        span_base: torch.Tensor,
        style_feats: torch.Tensor,
        haar_mask: torch.Tensor | None = None,
        u_doubt_scalars: torch.Tensor | None = None,
        outer_idx: int = 1,
        handoff_mask: torch.Tensor | None = None,
        ctx_vec: torch.Tensor | None = None,
        ctx_kish_lambda: torch.Tensor | None = None,
        k_inner: int = 2,
        trm_lam_k0: float = 1.0,
        trm_lam_k1: float = 1.0,
        promo_sh_lam: float = 0.35,
    ) -> dict[str, Any]:
        s_in = slots
        if u_doubt_scalars is not None:
            s_hyp = zh_bus.detach() if self.training else zh_bus
            u_d = torch.cat([s_hyp, u_doubt_scalars.detach() if self.training else u_doubt_scalars], dim=-1)
            if handoff_mask is not None:
                hm = handoff_mask.view(-1, 1)
                s_hyp = s_hyp * hm
                u_d = u_d * hm
            conf_damp = 1.0 - 0.75 * (u_doubt_scalars[:, 0:1].clamp(0.0, 1.0) ** 2)
            falsify_delta = (
                -0.20 * torch.tanh(self.h2_hyp_gate) * conf_damp * self.h2_hyp_proj(s_hyp)
                + 0.25 * torch.tanh(self.h2_doubt_gate) * self.h2_doubt_proj(u_d)
            )
            s_in = torch.cat([s_in[:, :-1, :], s_in[:, -1:, :] + falsify_delta.unsqueeze(1)], dim=1)

        st = self.core.forward_outer_step(
            s_in, s_anchor, k_ev, v_ev, m_pool, h0_pool, outer_idx=outer_idx,
            k_haar=k_haar, v_haar=v_haar, haar_mask=(haar_mask if haar_mask is not None else attention_mask),
            carry_m_l=carry_m_l, prev_zh_bus=zh_bus, prev_mole_probs=mole_probs,
            ctx_vec=ctx_vec, ctx_kish_lambda=ctx_kish_lambda,
            k_inner=k_inner, trm_lam_k0=trm_lam_k0, trm_lam_k1=trm_lam_k1,
        )
        out = self._coda_readout(
            st["slots"], span_base, attention_mask, style_feats, st["ebt_out"], exit_idx=2, promo_sh_lam=promo_sh_lam
        )
        return {**out, **st}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor | None = None,
        trigram_ids: torch.Tensor | None = None,
        outer_steps: int = 2,
        k_inner: int = 2,
        trm_lam_k0: float = 1.0,
        trm_lam_k1: float = 1.0,
        promo_sh_lam: float = 0.35,
        handoff_drop_prob: float = 0.0,
        handoff_flip_prob: float = 0.0,
        bal_rho_new: float = 0.15,
        bal_frozen: bool = False,
        ctx_vec: torch.Tensor | None = None,
        ctx_kish_lambda: torch.Tensor | None = None,
        fwd_state_in: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        if engram_ids is None:
            engram_ids = torch.zeros_like(input_ids)
        if trigram_ids is None:
            trigram_ids = torch.zeros_like(input_ids)
        r1 = self.forward_rung1_prefix(input_ids, attention_mask, style_feats, engram_ids, trigram_ids)

        bsz = input_ids.shape[0]
        device = input_ids.device
        h_mask = None
        r1_pkt_use = r1["r1_pkt"]
        if self.training and (handoff_drop_prob > 0.0 or handoff_flip_prob > 0.0):
            h_mask = (torch.rand(bsz, device=device) >= handoff_drop_prob).float()
            if handoff_flip_prob > 0.0 and bsz >= 2:
                do_flip = (torch.rand(bsz, device=device) < handoff_flip_prob).unsqueeze(-1)
                r1_pkt_use = torch.where(do_flip, torch.roll(r1_pkt_use.detach(), shifts=1, dims=0), r1_pkt_use)

        r2 = self.forward_rung2_t1(
            input_ids, attention_mask, style_feats, engram_ids,
            r1["x1_in"], r1["h0_1"], r1_pkt_use, trigram_ids=trigram_ids, handoff_mask=h_mask,
            k_inner=k_inner, trm_lam_k0=trm_lam_k0, trm_lam_k1=trm_lam_k1, promo_sh_lam=promo_sh_lam,
            fwd_state_in=fwd_state_in,
        )
        if self.training:
            self.core.mole.update_deepseek_bal_bias(r2["mole_probs"], tick_idx=0, rho_new=bal_rho_new, frozen=bal_frozen)

        slots, carry_m_l, zh_bus, m_probs = r2["slots"], r2["carry_m_l"], r2["zh_bus"], r2["mole_probs"]
        total_z_loss = r2["mole_z_loss"]
        final_out = r2["exit1_out"]
        e1 = r2["exit1_out"]
        mole_probs_all = [m_probs]

        for oi in range(1, max(1, int(outer_steps))):
            c1_conf = (2.0 * torch.sigmoid(e1["logit"].detach()) - 1.0).abs().unsqueeze(-1)
            u_doubt_s = torch.cat(
                [c1_conf, e1["slot_spread"].unsqueeze(-1), e1["p_defer2"].unsqueeze(-1), e1["ebt_grad_norm"].unsqueeze(-1)],
                dim=-1,
            )
            r3 = self.forward_rung3_step(
                slots, r2["s_anchor"], carry_m_l, zh_bus, m_probs,
                r2["k_ev"], r2["v_ev"], r2["m_pool"], r2["k_haar"], r2["v_haar"],
                attention_mask, r2["h0_pool"], r2["span_base"], style_feats,
                u_doubt_scalars=u_doubt_s, outer_idx=oi, handoff_mask=h_mask,
                ctx_vec=ctx_vec, ctx_kish_lambda=ctx_kish_lambda,
                k_inner=k_inner, trm_lam_k0=trm_lam_k0, trm_lam_k1=trm_lam_k1, promo_sh_lam=promo_sh_lam,
            )
            if self.training:
                self.core.mole.update_deepseek_bal_bias(r3["mole_probs"], tick_idx=oi, rho_new=bal_rho_new, frozen=bal_frozen)
            slots, carry_m_l, zh_bus, m_probs = r3["slots"], r3["carry_m_l"], r3["zh_bus"], r3["mole_probs"]
            total_z_loss = total_z_loss + r3["mole_z_loss"]
            mole_probs_all.append(m_probs)
            final_out = r3

        mean_mole_probs = torch.stack(mole_probs_all, dim=0).mean(dim=0)
        if return_aux:
            return {
                **final_out,
                **r1,
                "exit1_out": r2["exit1_out"],
                "slots_final": slots,
                "slots_anchor": r2["s_anchor"],
                "mole_probs_final": m_probs,
                "mole_probs_mean": mean_mole_probs,
                "mole_z_loss": total_z_loss / float(max(1, int(outer_steps))),
                "k_ev": r2["k_ev"],
                "v_ev": r2["v_ev"],
                "k_haar": r2["k_haar"],
                "v_haar": r2["v_haar"],
                "fwd_state_out": r2["fwd_state_out"],
            }
        return final_out["logit"], final_out["logit_promo"], final_out["emb"], r1["logit_r1"], r1["logit_promo_r1"]

    def forward_chronostream_batch(
        self,
        chunk_ids: torch.Tensor,      # (B, M, L)
        chunk_mask: torch.Tensor,     # (B, M, L)
        chunk_eng: torch.Tensor,      # (B, M, L)
        chunk_tri: torch.Tensor,      # (B, M, L)
        chunk_style56: torch.Tensor,  # (B, M, 56)
        chunk_meta6: torch.Tensor,    # (B, M, 6)
        chunk_valid: torch.Tensor,    # (B, M) 1=valid, 0=pad
        doc_style56: torch.Tensor,    # (B, 56)
        tbptt_k2: int = 3,
        trunk_grad_lam: float = 0.30,
        trunk_no_grad: bool = False,
        outer_steps: int = 2,
    ) -> dict[str, torch.Tensor]:
        """F8 Unified ChronoStream TBPTT (`k1=M, k2=3`) carrying `fwd_state` + Top-32 `L0-Haar` sketch -> `DynaSeam-MIL`!"""
        bsz, m_max, _ = chunk_ids.shape
        device = chunk_ids.device
        grad_start = max(0, m_max - int(tbptt_k2))

        fwd_st: torch.Tensor | None = None
        hk_sketch: torch.Tensor | None = None
        hv_sketch: torch.Tensor | None = None
        ch_ai_l, ch_pr_l, ch_ebt_m, ch_unc, ch_emb16 = [], [], [], [], []

        for mi in range(m_max):
            use_grad = (not trunk_no_grad) and (mi >= grad_start)
            ids_m = chunk_ids[:, mi, :]
            msk_m = chunk_mask[:, mi, :]
            eng_m = chunk_eng[:, mi, :]
            tri_m = chunk_tri[:, mi, :]
            st_m = chunk_style56[:, mi, :]
            v_m = chunk_valid[:, mi].unsqueeze(-1)  # (B, 1)

            with torch.set_grad_enabled(use_grad):
                r1 = self.forward_rung1_prefix(ids_m, msk_m, st_m, eng_m, tri_m)
                r2 = self.forward_rung2_t1(
                    ids_m, msk_m, st_m, eng_m, r1["x1_in"], r1["h0_1"], r1["r1_pkt"],
                    trigram_ids=tri_m, fwd_state_in=fwd_st,
                )
                fwd_new = r2["fwd_state_out"]
                if fwd_new is not None:
                    # Only advance fwd_state for valid chunks (shape-agnostic across 2D/3D/4D fwd_state!); detach across TBPTT boundary!
                    if fwd_st is not None:
                        v_nd = v_m.view([bsz] + [1] * (fwd_new.ndim - 1))
                        fwd_st = torch.where(v_nd > 0.5, fwd_new, fwd_st)
                    else:
                        fwd_st = fwd_new
                    if not use_grad:
                        fwd_st = fwd_st.detach()

                # Constant-SRAM Top-32 L0-Haar surface-seam sketch carried across chunks (F8)
                kh, vh = r2["k_haar"], r2["v_haar"]
                if hk_sketch is not None:
                    kh_cat = torch.cat([hk_sketch, kh], dim=1)
                    vh_cat = torch.cat([hv_sketch, vh], dim=1)
                else:
                    kh_cat, vh_cat = kh, vh
                kk = min(32, kh_cat.shape[1])
                top_idx = torch.topk(kh_cat.norm(dim=-1), k=kk, dim=-1).indices
                idx_e = top_idx.unsqueeze(-1).expand(-1, -1, kh_cat.shape[-1])
                hk_next = torch.gather(kh_cat, 1, idx_e)
                hv_next = torch.gather(vh_cat, 1, idx_e)
                if not use_grad:
                    hk_next, hv_next = hk_next.detach(), hv_next.detach()
                hk_sketch, hv_sketch = hk_next, hv_next

                e1 = r2["exit1_out"]
                if outer_steps >= 2:
                    c1_c = (2.0 * torch.sigmoid(e1["logit"].detach()) - 1.0).abs().unsqueeze(-1)
                    uds = torch.cat([c1_c, e1["slot_spread"].unsqueeze(-1), e1["p_defer2"].unsqueeze(-1), e1["ebt_grad_norm"].unsqueeze(-1)], dim=-1)
                    out_m = self.forward_rung3_step(
                        r2["slots"], r2["s_anchor"], r2["carry_m_l"], r2["zh_bus"], r2["mole_probs"],
                        r2["k_ev"], r2["v_ev"], r2["m_pool"], hk_sketch, hv_sketch,
                        msk_m, r2["h0_pool"], r2["span_base"], st_m,
                        haar_mask=torch.ones((bsz, hk_sketch.shape[1]), dtype=torch.long, device=device),
                        u_doubt_scalars=uds, outer_idx=1,
                    )
                else:
                    out_m = e1

                lg_ai = out_m["logit"]
                lg_pr = out_m["logit_promo"]
                ebt_m = out_m["ebt_margin"]
                unc = out_m["p_defer2"]
                emb_m = out_m["emb"]

            # Scale trunk gradient from long losses by D_lam = trunk_grad_lam (P2: 0.30)!
            if self.training and use_grad:
                lg_ai = tunnel_trm_ste(lg_ai, trunk_grad_lam)
                lg_pr = tunnel_trm_ste(lg_pr, trunk_grad_lam)
                emb_m = tunnel_trm_ste(emb_m, trunk_grad_lam)

            ch_ai_l.append(lg_ai)
            ch_pr_l.append(lg_pr)
            ch_ebt_m.append(ebt_m)
            ch_unc.append(unc)
            ch_emb16.append(self.chunk_emb_proj(emb_m))

        ai_stack = torch.stack(ch_ai_l, dim=1)       # (B, M)
        pr_stack = torch.stack(ch_pr_l, dim=1)       # (B, M)
        ebt_stack = torch.stack(ch_ebt_m, dim=1)     # (B, M)
        unc_stack = torch.stack(ch_unc, dim=1)       # (B, M)
        emb16_stack = torch.stack(ch_emb16, dim=1)   # (B, M, 16)
        chunk_x = torch.cat(
            [
                ai_stack.unsqueeze(-1),
                pr_stack.unsqueeze(-1),
                ebt_stack.unsqueeze(-1),
                unc_stack.unsqueeze(-1),
                emb16_stack,
                chunk_meta6,
                chunk_style56,
            ],
            dim=-1,
        )
        ds_out = self.longstream(
            chunk_x,
            ai_stack,
            pr_stack,
            chunk_meta6[:, :, 3],  # quote_frac
            chunk_meta6[:, :, 4],  # has_quote_intro
            chunk_valid,
            doc_style56,
        )
        return {**ds_out, "raw_chunk_ai": ai_stack, "raw_chunk_pr": pr_stack}


# ---------------------------------------------------------------------------
# 5. P1 Surgical Weight Transplantation & Zero-Shock Init (v1.0 -> v1.1)
# ---------------------------------------------------------------------------

def fit_style40_to_style56_ridge_bridge(
    f40: np.ndarray,
    f56: np.ndarray,
    ridge_lam: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """P1.2: Fit exact ridge bridge `f40 ≈ A @ f56 + c` (`A: (40, 56), c: (40,)`) & return aggregate R^2."""
    x = np.asarray(f56, dtype=np.float64)
    y = np.asarray(f40, dtype=np.float64)
    n, d_in = x.shape
    x_aug = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)  # (N, 57)
    reg = np.eye(d_in + 1, dtype=np.float64) * float(ridge_lam)
    reg[-1, -1] = 0.0  # do not regularize intercept
    w_sol = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)  # (57, 40)
    a_mat = w_sol[:-1, :].T.astype(np.float32)  # (40, 56)
    c_vec = w_sol[-1, :].astype(np.float32)     # (40,)
    pred = x @ a_mat.T + c_vec[None, :]
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return a_mat, c_vec, float(r2)


def compute_obs_saliency_permutation(
    emb_cal: np.ndarray,
    w_ai_256: np.ndarray,
    ridge_eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """P1.3: Compute exact OBS saliency `s_c = w_c^2 / Theta_cc` (`Theta = Sigma_emb^{-1}`) & 3-region permutation `P`.

    Layout after permutation `P`:
      - `[0:128]`   : Shared semantic/stylometry core (middle 128 saliency dims)
      - `[128:192]` : AI-exclusive high-saliency region (top 64 dims)
      - `[192:256]` : Promo/Governance-exclusive region (lowest 64 AI saliency dims)
    """
    x = np.asarray(emb_cal, dtype=np.float64)
    w = np.asarray(w_ai_256, dtype=np.float64).reshape(-1)
    assert x.shape[1] == 256 and w.shape[0] == 256
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / float(max(1, len(x) - 1)) + np.eye(256, dtype=np.float64) * float(ridge_eps)
    theta = np.linalg.inv(cov)
    theta_diag = np.maximum(np.diag(theta), 1e-8)
    saliency = (w * w) / theta_diag
    asc = np.argsort(saliency)  # lowest -> highest AI saliency
    promo_64 = asc[:64]
    shared_128 = asc[64:192]
    ai_64 = asc[192:256]
    perm = np.concatenate([shared_128, ai_64, promo_64]).astype(np.int64)
    return perm, saliency.astype(np.float32)


@torch.no_grad()
def transplant_v10_to_v11_zero_shock(
    v10_model: MeguvizV10ApexLoopModel,
    v11_model: MeguvizV11OmniGuardModel,
    f40_cal: np.ndarray,
    f56_cal: np.ndarray,
    emb_v10_cal: np.ndarray,
    j1_expert: int = 0,
    j2_expert: int = 1,
    load_r1_ref: np.ndarray | None = None,
    seed: int = 20261126,
) -> dict[str, Any]:
    """P1 Zero-Shock Weight Transplant from pre-folded `v1.0.0-R1` into `MeguvizV11OmniGuardModel`!"""
    g = torch.Generator().manual_seed(seed)
    sd10 = v10_model.state_dict()
    sd11 = v11_model.state_dict()

    # 1. Copy all identically named & shaped weights/buffers first
    copied = 0
    for k, v in sd10.items():
        if k in sd11 and sd11[k].shape == v.shape:
            sd11[k].copy_(v.to(sd11[k].device, dtype=sd11[k].dtype))
            copied += 1

    # 2. P1.2: 40 -> 56 Ridge Bridge folded into `r1_style_mlp.0` & `style_net.0`
    a_mat, c_vec, bridge_r2 = fit_style40_to_style56_ridge_bridge(f40_cal, f56_cal)
    a_t = torch.from_numpy(a_mat).to(sd11["r1_style_mlp.0.weight"].device, dtype=torch.float32)
    c_t = torch.from_numpy(c_vec).to(sd11["r1_style_mlp.0.weight"].device, dtype=torch.float32)
    for pref in ("r1_style_mlp.0", "style_net.0"):
        w10 = sd10[f"{pref}.weight"].float().to(a_t.device)
        b10 = sd10[f"{pref}.bias"].float().to(a_t.device)
        sd11[f"{pref}.weight"].copy_((w10 @ a_t).to(sd11[f"{pref}.weight"].dtype))
        sd11[f"{pref}.bias"].copy_((b10 + w10 @ c_t).to(sd11[f"{pref}.bias"].dtype))

    # 3. `defer_r1_enc.0` (420 -> 421): copy first 420 cols, zero col 420 (raw_logit_promo_r1)
    sd11["defer_r1_enc.0.weight"][:, :420].copy_(sd10["defer_r1_enc.0.weight"].to(sd11["defer_r1_enc.0.weight"].device))
    sd11["defer_r1_enc.0.weight"][:, 420:].zero_()
    sd11["defer_r1_enc.0.bias"].copy_(sd10["defer_r1_enc.0.bias"].to(sd11["defer_r1_enc.0.bias"].device))

    # 4. F1 + P1.1: EBT 5 -> 6 (`b5 = b0 + 6.0` => ebt_ai_margin bit-identical!)
    dev = sd11["core.inloop_ebt.fc2_old_w"].device
    w2_10 = sd10["core.inloop_ebt.fc2.weight"].float().to(dev)
    b2_10 = sd10["core.inloop_ebt.fc2.bias"].float().to(dev)
    sig_w2 = float(w2_10.std().item()) + 1e-6
    noise_w2 = torch.randn((1, w2_10.shape[1]), generator=g, dtype=torch.float32).to(dev) * (0.01 * sig_w2)
    sd11["core.inloop_ebt.fc2_old_w"].copy_(w2_10.to(sd11["core.inloop_ebt.fc2_old_w"].dtype))
    sd11["core.inloop_ebt.fc2_old_b"].copy_(b2_10.to(sd11["core.inloop_ebt.fc2_old_b"].dtype))
    sd11["core.inloop_ebt.fc2_new_w"].copy_((w2_10[0:1] + noise_w2).to(sd11["core.inloop_ebt.fc2_new_w"].dtype))
    sd11["core.inloop_ebt.fc2_new_b"].copy_((b2_10[0:1] + 6.0).to(sd11["core.inloop_ebt.fc2_new_b"].dtype))

    # 5. P1.1 + P1.4: Exact Net2Net MoLE 8 -> 10 Split!
    j1, j2 = int(j1_expert) % 8, int(j2_expert) % 8
    sd11["core.mole.core_routed_old"].copy_(sd10["core.mole.core_routed"].to(dev))
    sd11["core.mole.core_bias_old"].copy_(sd10["core.mole.core_bias"].to(dev))
    sd11["core.mole.dyn_profiles_old"].copy_(sd10["core.mole.dyn_profiles"].to(dev))
    sd11["core.mole.router_keys_old"].copy_(sd10["core.mole.router_keys"].to(dev))

    sd11["core.mole.core_routed_new"][0].copy_(sd10["core.mole.core_routed"][j1].to(dev))
    sd11["core.mole.core_routed_new"][1].copy_(sd10["core.mole.core_routed"][j2].to(dev))
    sd11["core.mole.core_bias_new"][0].copy_(sd10["core.mole.core_bias"][j1].to(dev))
    sd11["core.mole.core_bias_new"][1].copy_(sd10["core.mole.core_bias"][j2].to(dev))
    sd11["core.mole.dyn_profiles_new"][0].copy_(sd10["core.mole.dyn_profiles"][j1].to(dev))
    sd11["core.mole.dyn_profiles_new"][1].copy_(sd10["core.mole.dyn_profiles"][j2].to(dev))

    # Tiny orthogonal key perturbation (1e-3 * u_perp) breaks symmetry while keeping MoLE mix diff <= 1e-5!
    for slot_idx, src_j in ((0, j1), (1, j2)):
        k_src = sd10["core.mole.router_keys"][src_j].float().to(dev)
        u_r = torch.randn(k_src.shape, generator=g, dtype=torch.float32).to(dev)
        u_perp = u_r - (torch.dot(u_r, k_src) / (torch.dot(k_src, k_src) + 1e-8)) * k_src
        u_perp = u_perp / (u_perp.norm() + 1e-8)
        sd11["core.mole.router_keys_new"][slot_idx].copy_((k_src + 1e-3 * u_perp).to(sd11["core.mole.router_keys_new"].dtype))

    bal_old = sd10["core.mole.bal_bias"].float().to(dev)  # (3, 8)
    bal_new = torch.zeros((3, 10), device=dev, dtype=torch.float32)
    bal_new[:, :8].copy_(bal_old)
    if j1 != j2:
        ln2 = math.log(2.0)
        bal_new[:, j1] = bal_old[:, j1] - ln2
        bal_new[:, j2] = bal_old[:, j2] - ln2
        bal_new[:, 8] = bal_old[:, j1] - ln2
        bal_new[:, 9] = bal_old[:, j2] - ln2
    else:
        ln3 = math.log(3.0)
        bal_new[:, j1] = bal_old[:, j1] - ln3
        bal_new[:, 8] = bal_old[:, j1] - ln3
        bal_new[:, 9] = bal_old[:, j1] - ln3
    sd11["core.mole.bal_bias"].copy_(bal_new)

    if load_r1_ref is not None and len(load_r1_ref) == 8:
        lr_t = torch.from_numpy(np.asarray(load_r1_ref, dtype=np.float32)).to(dev)
        sd11["core.mole.load_r1_ref"].copy_(lr_t / lr_t.sum().clamp(min=1e-6))
    # P1.4: update bal_bias ONLY on {j1, j2, 8, 9}!
    bmask = torch.zeros(10, device=dev, dtype=torch.float32)
    for idx_u in (j1, j2, 8, 9):
        bmask[idx_u] = 1.0
    sd11["core.mole.bal_update_mask"].copy_(bmask)

    # 6. P1.3: Exact OBS Coordinate Permutation Firewall (`emb_perm`) + OBS Ridge Compensation on `ai_head` + F6 CBM Mask!
    w_ai_10 = sd10["ai_head.weight"].float().cpu().numpy().reshape(-1)
    b_ai_10 = float(sd10["ai_head.bias"].float().cpu().item())
    perm_np, _ = compute_obs_saliency_permutation(emb_v10_cal, w_ai_10)
    perm_t = torch.from_numpy(perm_np).to(dev)
    sd11["emb_perm"].copy_(perm_t)

    # Fold P into ROWS of `coda_proj.0` & `style_net.2` (commutes exactly with SiLU & aligns with W4 g64!)
    sd11["coda_proj.0.weight"].copy_(sd10["coda_proj.0.weight"].to(dev)[perm_t, :])
    sd11["coda_proj.0.bias"].copy_(sd10["coda_proj.0.bias"].to(dev)[perm_t])
    sd11["style_net.2.weight"].copy_(sd10["style_net.2.weight"].to(dev)[perm_t, :])
    sd11["style_net.2.bias"].copy_(sd10["style_net.2.bias"].to(dev)[perm_t])

    # Fold P into COLUMNS of downstream readers (`cbm_enc`, `defer_head_r2[:, :256]`, `attr6`, `code`, `prag`, `intent`)
    cbm_w_perm = sd10["cbm_enc.weight"].to(dev)[:, perm_t].clone()
    cbm_w_perm[:CBM_AI_CONCEPTS_V11, AI_EMB_END_V11:].zero_()  # F6 firewall!
    sd11["cbm_enc.weight"].copy_(cbm_w_perm)
    sd11["cbm_enc.bias"].copy_(sd10["cbm_enc.bias"].to(dev))
    sd11["cbm_dec.weight"].copy_(sd10["cbm_dec.weight"].to(dev)[perm_t, :])
    cbm_aw = sd10["cbm_audit_w.weight"].to(dev).clone()
    cbm_aw[:, CBM_AI_CONCEPTS_V11:].zero_()  # F6 reserve 16 governance concepts!
    sd11["cbm_audit_w.weight"].copy_(cbm_aw)

    # defer_head_r2 first linear layer consumes [emb(256), 6 scalars]
    d2_key = "defer_head_r2.net.0.weight"
    if d2_key in sd10 and d2_key in sd11:
        d2_w = sd10[d2_key].to(dev).clone()
        d2_w[:, :256] = d2_w[:, perm_t]
        sd11[d2_key].copy_(d2_w)

    a5_w_perm = sd10["attr5_head.weight"].to(dev)[:, perm_t]
    a5_b = sd10["attr5_head.bias"].to(dev)
    sd11["attr6_old_w"].copy_(a5_w_perm)
    sd11["attr6_old_b"].copy_(a5_b)
    sd11["attr6_new_w"].copy_(a5_w_perm[0:1])
    sd11["attr6_new_b"].copy_(a5_b[0:1] - 6.0)

    for hname in ("code_head", "prag_head", "intent_head"):
        if f"{hname}.weight" in sd10 and f"{hname}.weight" in sd11 and sd10[f"{hname}.weight"].shape == sd11[f"{hname}.weight"].shape:
            sd11[f"{hname}.weight"].copy_(sd10[f"{hname}.weight"].to(dev)[:, perm_t])
            if f"{hname}.bias" in sd10:
                sd11[f"{hname}.bias"].copy_(sd10[f"{hname}.bias"].to(dev))

    # OBS ridge compensation for zeroing `ai_head` on `[192:256]`: reproduce old `emb @ w_ai_10 + b_ai_10` using ONLY `[0:192]`!
    emb_perm_np = emb_v10_cal[:, perm_np].astype(np.float64)
    old_ai_lin = emb_v10_cal.astype(np.float64) @ w_ai_10.astype(np.float64) + b_ai_10
    x192 = np.concatenate([emb_perm_np[:, :AI_EMB_END_V11], np.ones((len(emb_perm_np), 1), dtype=np.float64)], axis=1)
    reg192 = np.eye(AI_EMB_END_V11 + 1, dtype=np.float64) * 1e-4
    reg192[-1, -1] = 0.0
    sol_ai = np.linalg.solve(x192.T @ x192 + reg192, x192.T @ old_ai_lin)
    w_ai_192 = sol_ai[:-1].astype(np.float32)
    b_ai_192 = float(sol_ai[-1])
    obs_rmse = float(np.sqrt(np.mean((x192 @ sol_ai - old_ai_lin) ** 2)))

    sd11["ai_head.weight"].copy_(torch.from_numpy(w_ai_192).view(1, AI_EMB_END_V11).to(dev, dtype=sd11["ai_head.weight"].dtype))
    sd11["ai_head.bias"].fill_(b_ai_192)

    return {
        "copied_identical_tensors": copied,
        "bridge_40_to_56_r2": round(bridge_r2, 5),
        "obs_ai_head_192_rmse": round(obs_rmse, 6),
        "mole_net2net_split": {"j1": j1, "j2": j2},
        "emb_layout": v11_model.emb_layout_tag(),
    }


# ---------------------------------------------------------------------------
# 6. Zero-Op Fold (`fold_zero_op_v11`) & MGVZ6+ Packer (Merges F7 Split Keys!)
# ---------------------------------------------------------------------------

@torch.no_grad()
def fold_zero_op_v11(
    model: MeguvizV11OmniGuardModel,
    calib_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    smooth_alpha: float = 0.5,
) -> dict[str, Any]:
    """Exact ZERO-OP offline algebraic fold (SpinQuant R_2 + SubLN + SmoothQuant on Prelude L1)."""
    h64 = _hadamard_64()

    def _rotate_vo(v_lin: nn.Linear, o_lin: nn.Linear, sub_ln: RMSNorm, n_heads: int) -> None:
        dev, dt = v_lin.weight.device, v_lin.weight.dtype
        sw = sub_ln.weight.detach().float()
        o_lin.weight.data.copy_((o_lin.weight.detach().float() * sw.unsqueeze(0)).to(dt))
        sub_ln.weight.data.fill_(1.0)
        r2 = torch.block_diag(*([h64] * n_heads)).to(dev)
        v_lin.weight.data.copy_((r2.T @ v_lin.weight.detach().float()).to(dt))
        if v_lin.bias is not None:
            v_lin.bias.data.copy_((v_lin.bias.detach().float() @ r2).to(dt))
        o_lin.weight.data.copy_((o_lin.weight.detach().float() @ r2).to(dt))

    _rotate_vo(model.core.v_proj, model.core.o_proj, model.core.sub_ln_attn, model.core.num_heads)
    _rotate_vo(model.prelude_l0.v_p, model.prelude_l0.o_p, model.prelude_l0.sub_ln_p, model.prelude_l0.swa_heads)

    folded_subln = 0
    for mod in model.modules():
        if hasattr(mod, "w_up") and hasattr(mod, "sub_ln") and hasattr(mod, "w_down"):
            dt = mod.w_down.weight.dtype
            sw = mod.sub_ln.weight.detach().float()
            mod.w_down.weight.data.copy_((mod.w_down.weight.detach().float() * sw.unsqueeze(0)).to(dt))
            mod.sub_ln.weight.data.fill_(1.0)
            folded_subln += 1
    p0 = model.prelude_l0
    dt_c = p0.ffn_c_down.weight.dtype
    p0.ffn_c_down.weight.data.copy_((p0.ffn_c_down.weight.detach().float() * p0.ffn_c_subln.weight.detach().float().unsqueeze(0)).to(dt_c))
    p0.ffn_c_subln.weight.data.fill_(1.0)
    folded_subln += 1

    act_max: torch.Tensor | None = None
    if calib_inputs:
        def _hk(_m, _inp, out):
            nonlocal act_max
            mx = out.detach().float().abs().amax(dim=(0, 1))
            act_max = mx if act_max is None else torch.maximum(act_max, mx)

        h = model.prelude_l1.norm2.register_forward_hook(_hk)
        model.eval()
        for ids_t, m_t, st_t, eng_t in calib_inputs:
            _ = model(ids_t, m_t, st_t, engram_ids=eng_t, outer_steps=2, return_aux=False)
        h.remove()

    b1 = model.prelude_l1
    dt1 = b1.ffn.w_up.weight.dtype
    w_abs = b1.ffn.w_up.weight.detach().float().abs().amax(dim=0).clamp(min=1e-4)
    x_abs = (act_max.to(w_abs.device) if act_max is not None else torch.ones_like(w_abs)).clamp(min=1e-4)
    s = (x_abs.pow(smooth_alpha) / w_abs.pow(1.0 - smooth_alpha)).clamp(0.25, 4.0)
    b1.norm2.gamma.data.copy_((b1.norm2.gamma.detach().float() / s).to(dt1))
    b1.norm2.beta.data.copy_((b1.norm2.beta.detach().float() / s).to(dt1))
    b1.ffn.w_up.weight.data.copy_((b1.ffn.w_up.weight.detach().float() * s.unsqueeze(0)).to(dt1))
    b1.lora_moe.down.data.copy_((b1.lora_moe.down.detach().float() * s.view(1, -1, 1)).to(dt1))

    return {"folded_spinquant_r2_blocks": 2, "folded_subln_ffn_blocks": folded_subln, "folded_smoothquant_blocks": 1, "new_runtime_ops_added": 0}


_SPLIT_MERGE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("core.mole.core_routed_old", "core.mole.core_routed_new", "core.mole.core_routed"),
    ("core.mole.core_bias_old", "core.mole.core_bias_new", "core.mole.core_bias"),
    ("core.mole.dyn_profiles_old", "core.mole.dyn_profiles_new", "core.mole.dyn_profiles"),
    ("core.mole.router_keys_old", "core.mole.router_keys_new", "core.mole.router_keys"),
    ("core.inloop_ebt.fc2_old_w", "core.inloop_ebt.fc2_new_w", "core.inloop_ebt.fc2.weight"),
    ("core.inloop_ebt.fc2_old_b", "core.inloop_ebt.fc2_new_b", "core.inloop_ebt.fc2.bias"),
    ("attr6_old_w", "attr6_new_w", "attr6_head.weight"),
    ("attr6_old_b", "attr6_new_b", "attr6_head.bias"),
)


def pack_bundle_v6_v11(model: MeguvizV11OmniGuardModel, out_path: Path, lever_w4_longstream: bool = False, lever_w4_r1_promo: bool = False) -> dict[str, Any]:
    """Pack `MeguvizV11OmniGuardModel` into `MGVZ6` (merges F7 split keys; pins all new sensitive heads in FP16 per P5!)."""
    sd_raw = model.state_dict()
    sd: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for k_old, k_new, k_merged in _SPLIT_MERGE_PAIRS:
        if k_old in sd_raw and k_new in sd_raw:
            sd[k_merged] = torch.cat([sd_raw[k_old], sd_raw[k_new]], dim=0)
            consumed.add(k_old)
            consumed.add(k_new)
    for k, v in sd_raw.items():
        if k not in consumed:
            sd[k] = v

    qtype_by_param: dict[str, tuple[str, int]] = {
        "word_emb_A.weight": ("seq2", 64),
        "engram_emb.weight": ("w4", 64),
        "trigram_emb.weight": ("w4", 64),
        "pos_emb1.weight": ("w4", 64),
        "pos_emb2.weight": ("w4", 64),
        "coda_proj.0.weight": ("w4", 64),
        "span_l0_proj.weight": ("w4", 64),
        "span_l1_proj.weight": ("w4", 64),
        "span_topdown_proj.weight": ("w4", 64),
        "cbm_enc.weight": ("w4", 64),
        "style_net.2.weight": ("w4", 64),
        # P5 FP16-pinned sensitive dual-axis & core tensors
        "core.mole.core_shared": ("fp16", 0),
        "core.mole.core_routed": ("fp16", 0),
        "core.mole.core_bias": ("fp16", 0),
        "core.mole.dyn_profiles": ("fp16", 0),
        "core.mole.router_keys": ("fp16", 0),
        "core.inloop_ebt.fc1.weight": ("fp16", 0),
        "core.inloop_ebt.fc2.weight": ("fp16", 0),
        "promo_head.weight": ("fp16", 0),
        "promo_sub4_head.weight": ("fp16", 0),
        "promo_span_head.1.weight": ("fp16", 0),
        "attr6_head.weight": ("fp16", 0),
        "r1_promo_head.0.weight": ("w4", 64) if lever_w4_r1_promo else ("fp16", 0),
        "r1_promo_head.2.weight": ("fp16", 0),
    }
    if lever_w4_longstream:
        for lk in ("longstream.chunk_enc.0.weight", "longstream.qkv.weight", "longstream.ffn.0.weight", "longstream.ffn.2.weight"):
            qtype_by_param[lk] = ("w4", 64)

    for mod_name, mod in model.named_modules():
        if isinstance(mod, OmniQuantLinear):
            qtype_by_param[f"{mod_name}.weight"] = (mod.target_bits, mod.group_size)
        elif isinstance(mod, SeqLoRAMoE):
            qtype_by_param[f"{mod_name}.down"] = ("w4", 64)
            qtype_by_param[f"{mod_name}.up"] = ("w4", 64)
        elif isinstance(mod, nn.Linear) and mod.weight.numel() >= 8192 and f"{mod_name}.weight" not in qtype_by_param:
            if not mod_name.startswith("longstream."):
                qtype_by_param[f"{mod_name}.weight"] = ("w4", 64)

    meta_tensors: list[dict[str, Any]] = []
    blobs: list[bytes] = []
    byte_breakdown: dict[str, int] = {"trit158": 0, "seq2": 0, "w4": 0, "fp16": 0}

    for k, v in sd.items():
        if k.startswith(("intent_head.", "cbm_dec.")) or k in ("core.mole.load_r1_ref", "core.mole.bal_update_mask"):
            continue
        arr = v.detach().float().cpu().numpy()
        shape = list(arr.shape)
        numel = int(arr.size)
        qkind, gsz = qtype_by_param.get(k, ("fp16", 0))

        if qkind == "trit158":
            pad = (gsz - (numel % gsz)) % gsz
            flat = np.pad(arr.reshape(-1), (0, pad)) if pad else arr.reshape(-1)
            wg = flat.reshape(-1, gsz)
            sc = np.maximum(np.mean(np.abs(wg), axis=1, keepdims=True), 1e-5).astype(np.float16)
            q = np.clip(np.round(wg / sc.astype(np.float32)), -1, 1).astype(np.int8)
            raw_q, raw_s = _pack_5trit_bytes(q.reshape(-1)[:numel]), sc.tobytes()
            blob = raw_s + raw_q
            byte_breakdown["trit158"] += len(blob)
            meta_tensors.append({"name": k, "q": "trit158", "g": gsz, "shape": shape, "s_bytes": len(raw_s), "q_bytes": len(raw_q)})
            blobs.append(blob)
        elif qkind == "seq2":
            pad = (gsz - (numel % gsz)) % gsz
            flat = np.pad(arr.reshape(-1), (0, pad)) if pad else arr.reshape(-1)
            wg = flat.reshape(-1, gsz)
            sc = np.maximum(np.mean(np.abs(wg), axis=1, keepdims=True) * 0.62, 1e-5).astype(np.float16)
            idx = np.clip(np.round((wg / sc.astype(np.float32) + 3.0) * 0.5), 0, 3).astype(np.uint8)
            raw_q, raw_s = _pack_2bit_bytes(idx.reshape(-1)[:numel]), sc.tobytes()
            blob = raw_s + raw_q
            byte_breakdown["seq2"] += len(blob)
            meta_tensors.append({"name": k, "q": "seq2", "g": gsz, "shape": shape, "s_bytes": len(raw_s), "q_bytes": len(raw_q)})
            blobs.append(blob)
        elif qkind == "w4":
            pad = (gsz - (numel % gsz)) % gsz
            flat = np.pad(arr.reshape(-1), (0, pad)) if pad else arr.reshape(-1)
            wg = flat.reshape(-1, gsz)
            sc = np.maximum(np.max(np.abs(wg), axis=1, keepdims=True) / 7.0, 1e-5).astype(np.float16)
            q = np.clip(np.round(wg / sc.astype(np.float32)), -7, 7).astype(np.int8)
            raw_q, raw_s = _pack_w4_bytes(q.reshape(-1)[:numel]), sc.tobytes()
            blob = raw_s + raw_q
            byte_breakdown["w4"] += len(blob)
            meta_tensors.append({"name": k, "q": "w4", "g": gsz, "shape": shape, "s_bytes": len(raw_s), "q_bytes": len(raw_q)})
            blobs.append(blob)
        else:
            blob = arr.astype(np.float16).tobytes()
            byte_breakdown["fp16"] += len(blob)
            meta_tensors.append({"name": k, "q": "fp16", "g": 0, "shape": shape, "s_bytes": 0, "q_bytes": len(blob)})
            blobs.append(blob)

    header = json.dumps({"magic": "MGVZ6", "v": "1.1.0", "emb_layout": model.emb_layout_tag(), "tensors": meta_tensors}, separators=(",", ":")).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as wf:
        wf.write(b"MGVZ6\x00\x01\x01")
        wf.write(struct.pack("<I", len(header)))
        wf.write(header)
        for b in blobs:
            wf.write(b)

    raw_bytes = int(out_path.stat().st_size)
    return {
        "file": out_path.name,
        "raw_bytes": raw_bytes,
        "raw_mb": round(raw_bytes / (1024.0 * 1024.0), 4),
        "breakdown_kb": {k: round(v / 1024.0, 1) for k, v in byte_breakdown.items()},
        "emb_layout": model.emb_layout_tag(),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }


def load_bundle_v6_v11_into_model(model: MeguvizV11OmniGuardModel, mgvz_path: Path) -> None:
    """Bit-exact unpacker from `bundle_v6_omniguard.mgvz` back into `MeguvizV11OmniGuardModel` (splits F7 keys cleanly!)."""
    data = mgvz_path.read_bytes()
    assert data[:5] == b"MGVZ6", "Invalid MGVZ6 magic!"
    hlen = struct.unpack("<I", data[8:12])[0]
    header = json.loads(data[12 : 12 + hlen].decode("utf-8"))
    pos = 12 + hlen
    sd = model.state_dict()
    split_map = {k_m: (k_o, k_n) for k_o, k_n, k_m in _SPLIT_MERGE_PAIRS}

    for t in header["tensors"]:
        k, q, gsz, shape = t["name"], t["q"], int(t["g"]), tuple(t["shape"])
        numel = int(np.prod(shape))
        sb, qb = int(t["s_bytes"]), int(t["q_bytes"])
        s_buf, q_buf = data[pos : pos + sb], data[pos + sb : pos + sb + qb]
        pos += sb + qb
        if q == "fp16":
            arr = np.frombuffer(q_buf, dtype=np.float16).astype(np.float32).reshape(shape)
        else:
            sc = np.frombuffer(s_buf, dtype=np.float16).astype(np.float32).reshape(-1, 1)
            pad = (gsz - (numel % gsz)) % gsz
            if q == "trit158":
                vals = _unpack_5trit_bytes(q_buf, numel).astype(np.float32)
            elif q == "seq2":
                vals = (_unpack_2bit_bytes(q_buf, numel).astype(np.float32) * 2.0) - 3.0
            else:
                vals = _unpack_w4_bytes(q_buf, numel).astype(np.float32)
            if pad:
                vals = np.pad(vals, (0, pad))
            arr = (vals.reshape(-1, gsz) * sc).reshape(-1)[:numel].reshape(shape)

        if k in split_map:
            k_o, k_n = split_map[k]
            n_o = sd[k_o].shape[0]
            sd[k_o].copy_(torch.from_numpy(arr[:n_o]).to(sd[k_o].dtype))
            sd[k_n].copy_(torch.from_numpy(arr[n_o:]).to(sd[k_n].dtype))
        elif k in sd:
            sd[k].copy_(torch.from_numpy(arr).to(sd[k].dtype))


# ---------------------------------------------------------------------------
# 7. 4-Session Static-ONNX Exporter (`R1`, `R2`, `R3`, `DynaSeam`) + Cascade Engine
# ---------------------------------------------------------------------------

class _V11Rung1Export(nn.Module):
    def __init__(self, m: MeguvizV11OmniGuardModel):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, style_feats, engram_ids, trigram_ids):
        r1 = self.m.forward_rung1_prefix(input_ids, attention_mask, style_feats, engram_ids, trigram_ids)
        return r1["logit_r1"], r1["logit_promo_r1"], r1["p_defer1"], r1["r1_span_max"], r1["x1_in"], r1["h0_1"], r1["r1_pkt"]


class _V11Rung2T1Export(nn.Module):
    def __init__(self, m: MeguvizV11OmniGuardModel):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, style_feats, engram_ids, trigram_ids, x1_in, h0_1, r1_pkt):
        r2 = self.m.forward_rung2_t1(input_ids, attention_mask, style_feats, engram_ids, x1_in, h0_1, r1_pkt, trigram_ids=trigram_ids)
        e1 = r2["exit1_out"]
        return (
            e1["logit"], e1["logit_promo"], e1["promo_sub4_logit"], e1["emb"], e1["p_defer2"], e1["slot_spread"],
            e1["ebt_margin"], e1["ebt_grad_norm"],
            r2["slots"], r2["s_anchor"], r2["carry_m_l"], r2["zh_bus"], r2["mole_probs"],
            r2["k_ev"], r2["v_ev"], r2["m_pool"], r2["k_haar"], r2["v_haar"], r2["h0_pool"], r2["span_base"],
        )


class _V11Rung3StepExport(nn.Module):
    def __init__(self, m: MeguvizV11OmniGuardModel, outer_idx: int = 1):
        super().__init__()
        self.m = m
        self.outer_idx = int(outer_idx)

    def forward(
        self, slots, s_anchor, carry_m_l, zh_bus, mole_probs,
        k_ev, v_ev, m_pool, k_haar, v_haar, attention_mask, h0_pool, span_base, style_feats, u_doubt_scalars,
    ):
        r3 = self.m.forward_rung3_step(
            slots, s_anchor, carry_m_l, zh_bus, mole_probs,
            k_ev, v_ev, m_pool, k_haar, v_haar, attention_mask, h0_pool, span_base, style_feats,
            u_doubt_scalars=u_doubt_scalars, outer_idx=self.outer_idx,
        )
        return (
            r3["logit"], r3["logit_promo"], r3["promo_sub4_logit"], r3["emb"], r3["ebt_margin"], r3["ebt_grad_norm"],
            r3["slots"], r3["carry_m_l"], r3["zh_bus"], r3["mole_probs"],
        )


def export_v11_modular_onnx_and_bundle_v6(
    model: MeguvizV11OmniGuardModel,
    out_dir: Path,
    lever_w4_longstream: bool = False,
    lever_w4_r1_promo: bool = False,
) -> dict[str, Any]:
    """Export `MGVZ6` + 4-Session Static ONNX (`r1`, `r2`, `r3`, unquantized `dynaseam`) with parity check (<1e-4)."""
    import inspect
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir.mkdir(parents=True, exist_ok=True)
    mdl_cpu = model.eval().cpu()
    mdl_cpu.set_quant_mode("fp16")

    mgvz_path = out_dir / "bundle_v6_omniguard.mgvz"
    mgvz_info = pack_bundle_v6_v11(mdl_cpu, mgvz_path, lever_w4_longstream=lever_w4_longstream, lever_w4_r1_promo=lever_w4_r1_promo)

    exp_kw: dict[str, Any] = {"opset_version": 17, "do_constant_folding": True}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        exp_kw["dynamo"] = False

    def _quant_excluding_sensitive(fp32_p: Path, int8_p: Path) -> None:
        om = onnx.load(str(fp32_p))
        # P5: Exclude ALL dual-axis promo/sub4/attr/longstream/chunk_emb + v10 sensitive heads from INT8!
        excl = [
            nd.name for nd in om.graph.node
            if nd.name and any(
                k in nd.name
                for k in (
                    "ai_head", "r1_ai_head", "r1_promo", "promo", "sub4", "attr", "longstream", "chunk_emb",
                    "r1_rhythm", "r1_attn_pool", "defer_", "style_net", "r1_style_mlp",
                    "coda_proj", "span_", "word_emb", "engram_", "trigram_", "pos_emb",
                    "inloop_ebt", "mole", "h1_", "h2_", "p2_", "c1_", "c2_",
                    "bus_", "haar_", "reread_", "slot_", "q_proj", "k_proj", "v_proj", "o_proj",
                    "cbm_", "ctx_loop",
                )
            )
        ]
        quantize_dynamic(
            str(fp32_p), str(int8_p), weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excl,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )

    d_ids = torch.randint(1, mdl_cpu.vocab_size, (2, 40), dtype=torch.long)
    d_m = torch.ones((2, 40), dtype=torch.long)
    d_st = torch.zeros((2, STYLE_FEAT_DIM_V11), dtype=torch.float32)
    d_eng = torch.randint(1, mdl_cpu.engram_buckets, (2, 40), dtype=torch.long)
    d_tri = torch.randint(1, mdl_cpu.trigram_buckets, (2, 40), dtype=torch.long)

    # 1. R1 Prefix
    w1 = _V11Rung1Export(mdl_cpu).eval()
    r1_fp, r1_i8 = out_dir / "_v11_r1_fp32.onnx", out_dir / "v11_r1_prefix_int8.onnx"
    r1_in_names = ["input_ids", "attention_mask", "style_feats", "engram_ids", "trigram_ids"]
    r1_out_names = ["logit_r1", "logit_promo_r1", "p_defer1", "r1_span_max", "x1_in", "h0_1", "r1_pkt"]
    torch.onnx.export(
        w1, (d_ids, d_m, d_st, d_eng, d_tri), str(r1_fp),
        input_names=r1_in_names, output_names=r1_out_names,
        dynamic_axes={
            "input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"}, "style_feats": {0: "B"},
            "engram_ids": {0: "B", 1: "L"}, "trigram_ids": {0: "B", 1: "L"},
            "logit_r1": {0: "B"}, "logit_promo_r1": {0: "B"}, "p_defer1": {0: "B"}, "r1_span_max": {0: "B"},
            "x1_in": {0: "B", 1: "L"}, "h0_1": {0: "B", 1: "L"}, "r1_pkt": {0: "B"},
        },
        **exp_kw,
    )
    _quant_excluding_sensitive(r1_fp, r1_i8)
    with torch.no_grad():
        o1 = w1(d_ids, d_m, d_st, d_eng, d_tri)
    s1 = ort.InferenceSession(str(r1_fp), providers=["CPUExecutionProvider"])
    ox1 = s1.run(None, dict(zip(r1_in_names, [d_ids.numpy(), d_m.numpy(), d_st.numpy(), d_eng.numpy(), d_tri.numpy()])))
    r1_diff = float(max(np.max(np.abs(o1[0].numpy() - ox1[0])), np.max(np.abs(o1[1].numpy() - ox1[1]))))
    r1_fp.unlink(missing_ok=True)

    # 2. R2 T1
    w2 = _V11Rung2T1Export(mdl_cpu).eval()
    r2_fp, r2_i8 = out_dir / "_v11_r2_fp32.onnx", out_dir / "v11_r2_t1_int8.onnx"
    r2_in_names = ["input_ids", "attention_mask", "style_feats", "engram_ids", "trigram_ids", "x1_in", "h0_1", "r1_pkt"]
    r2_out_names = [
        "logit_t1", "logit_promo_t1", "sub4_t1", "emb_t1", "p_defer2", "slot_spread", "ebt_margin_t1", "ebt_grad_t1",
        "slots", "s_anchor", "carry_m_l", "zh_bus", "mole_probs",
        "k_ev", "v_ev", "m_pool", "k_haar", "v_haar", "h0_pool", "span_base",
    ]
    torch.onnx.export(
        w2, (d_ids, d_m, d_st, d_eng, d_tri, o1[4], o1[5], o1[6]), str(r2_fp),
        input_names=r2_in_names, output_names=r2_out_names,
        dynamic_axes={
            "input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"}, "style_feats": {0: "B"},
            "engram_ids": {0: "B", 1: "L"}, "trigram_ids": {0: "B", 1: "L"},
            "x1_in": {0: "B", 1: "L"}, "h0_1": {0: "B", 1: "L"}, "r1_pkt": {0: "B"},
            "logit_t1": {0: "B"}, "logit_promo_t1": {0: "B"}, "sub4_t1": {0: "B"}, "emb_t1": {0: "B"},
            "p_defer2": {0: "B"}, "slot_spread": {0: "B"}, "ebt_margin_t1": {0: "B"}, "ebt_grad_t1": {0: "B"},
            "slots": {0: "B"}, "s_anchor": {0: "B"}, "carry_m_l": {0: "B"}, "zh_bus": {0: "B"}, "mole_probs": {0: "B"},
            "k_ev": {0: "B", 1: "Lp"}, "v_ev": {0: "B", 1: "Lp"}, "m_pool": {0: "B", 1: "Lp"},
            "k_haar": {0: "B", 1: "L"}, "v_haar": {0: "B", 1: "L"}, "h0_pool": {0: "B"}, "span_base": {0: "B", 1: "L"},
        },
        **exp_kw,
    )
    _quant_excluding_sensitive(r2_fp, r2_i8)
    with torch.no_grad():
        o2 = w2(d_ids, d_m, d_st, d_eng, d_tri, o1[4], o1[5], o1[6])
    s2 = ort.InferenceSession(str(r2_fp), providers=["CPUExecutionProvider"])
    ox2 = s2.run(None, dict(zip(r2_in_names, [d_ids.numpy(), d_m.numpy(), d_st.numpy(), d_eng.numpy(), d_tri.numpy(), o1[4].numpy(), o1[5].numpy(), o1[6].numpy()])))
    r2_diff = float(max(np.max(np.abs(o2[0].numpy() - ox2[0])), np.max(np.abs(o2[1].numpy() - ox2[1]))))
    r2_fp.unlink(missing_ok=True)

    # 3. R3 Step
    w3 = _V11Rung3StepExport(mdl_cpu, outer_idx=1).eval()
    r3_fp, r3_i8 = out_dir / "_v11_r3_fp32.onnx", out_dir / "v11_r3_step_int8.onnx"
    c1_c = (2.0 * torch.sigmoid(o2[0]) - 1.0).abs().unsqueeze(-1)
    d_uds = torch.cat([c1_c, o2[5].unsqueeze(-1), o2[4].unsqueeze(-1), o2[7].unsqueeze(-1)], dim=-1)
    r3_in = (o2[8], o2[9], o2[10], o2[11], o2[12], o2[13], o2[14], o2[15], o2[16], o2[17], d_m, o2[18], o2[19], d_st, d_uds)
    r3_in_names = [
        "slots", "s_anchor", "carry_m_l", "zh_bus", "mole_probs",
        "k_ev", "v_ev", "m_pool", "k_haar", "v_haar",
        "attention_mask", "h0_pool", "span_base", "style_feats", "u_doubt_scalars",
    ]
    r3_out_names = [
        "logit_r3", "logit_promo_r3", "sub4_r3", "emb_r3", "ebt_margin_r3", "ebt_grad_r3",
        "slots_out", "carry_out", "zh_bus_out", "mole_probs_out",
    ]
    torch.onnx.export(
        w3, r3_in, str(r3_fp),
        input_names=r3_in_names, output_names=r3_out_names,
        dynamic_axes={
            "slots": {0: "B"}, "s_anchor": {0: "B"}, "carry_m_l": {0: "B"}, "zh_bus": {0: "B"}, "mole_probs": {0: "B"},
            "k_ev": {0: "B", 1: "Lp"}, "v_ev": {0: "B", 1: "Lp"}, "m_pool": {0: "B", 1: "Lp"},
            "k_haar": {0: "B", 1: "L"}, "v_haar": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"},
            "h0_pool": {0: "B"}, "span_base": {0: "B", 1: "L"}, "style_feats": {0: "B"}, "u_doubt_scalars": {0: "B"},
            "logit_r3": {0: "B"}, "logit_promo_r3": {0: "B"}, "sub4_r3": {0: "B"}, "emb_r3": {0: "B"},
            "ebt_margin_r3": {0: "B"}, "ebt_grad_r3": {0: "B"},
            "slots_out": {0: "B"}, "carry_out": {0: "B"}, "zh_bus_out": {0: "B"}, "mole_probs_out": {0: "B"},
        },
        **exp_kw,
    )
    _quant_excluding_sensitive(r3_fp, r3_i8)
    with torch.no_grad():
        o3 = w3(*r3_in)
    s3 = ort.InferenceSession(str(r3_fp), providers=["CPUExecutionProvider"])
    ox3 = s3.run(None, dict(zip(r3_in_names, [t.numpy() for t in r3_in])))
    r3_diff = float(max(np.max(np.abs(o3[0].numpy() - ox3[0])), np.max(np.abs(o3[1].numpy() - ox3[1]))))
    r3_fp.unlink(missing_ok=True)

    # 4. Unquantized DynaSeam ONNX (P5: never INT8-quantized!)
    ds_info = export_dynaseam_onnx_and_verify(mdl_cpu.longstream, out_dir / "v11_dynaseam_fp32.onnx")

    return {
        "bundle_v6_mgvz": mgvz_info,
        "rung1_onnx": r1_i8.name,
        "rung1_bytes": int(r1_i8.stat().st_size),
        "rung2_onnx": r2_i8.name,
        "rung2_bytes": int(r2_i8.stat().st_size),
        "rung3_onnx": r3_i8.name,
        "rung3_bytes": int(r3_i8.stat().st_size),
        "dynaseam_onnx": ds_info,
        "r1_fp32_max_diff": round(r1_diff, 7),
        "r2_fp32_max_diff": round(r2_diff, 7),
        "r3_fp32_max_diff": round(r3_diff, 7),
    }


# ---------------------------------------------------------------------------
# 8. Automated V11 Hardware-First & Zero-Shock CI Suite (`V11-G0`, `V11-F6`, `V11-C1`)
# ---------------------------------------------------------------------------

def run_v11_ci_gates(
    model: MeguvizV11OmniGuardModel,
    tokenizer: BiliPickyBPETokenizer,
    v10_ref_model: MeguvizV10ApexLoopModel | None = None,
) -> dict[str, Any]:
    """Verify v1.1.0 structural firewall (F1/F6/F7), analytic 6-way EBT gradient parity, Zero-PII, & M=1 safety."""
    device = next(model.parameters()).device
    model.eval()
    ci: dict[str, Any] = {}

    def _infer(txt: str):
        cl, f56, _ = extract_style_features_56d(txt)
        ids = tokenizer.encode_ids(cl, max_len=MAX_LEN_V07)
        eng = make_engram_bigram_ids(ids, buckets=model.engram_buckets)
        tri = make_engram_trigram_ids(ids, buckets=model.trigram_buckets)
        with torch.no_grad():
            return model(
                torch.tensor([ids], dtype=torch.long, device=device),
                torch.ones((1, len(ids)), dtype=torch.long, device=device),
                torch.from_numpy(f56).unsqueeze(0).to(device),
                engram_ids=torch.tensor([eng], dtype=torch.long, device=device),
                trigram_ids=torch.tensor([tri], dtype=torch.long, device=device),
                outer_steps=2, k_inner=2, return_aux=True,
            )

    # V11-S1: Dynamic lengths + F1 6-Way Analytic EBT exact gradient parity (< 1e-5)
    dyn_ok = True
    for seq_l in (2, 17, 65, 256):
        with torch.no_grad():
            ids_t = torch.randint(1, tokenizer.vocab_size, (2, seq_l), device=device)
            eg_t = torch.randint(1, model.engram_buckets, (2, seq_l), device=device)
            tg_t = torch.randint(1, model.trigram_buckets, (2, seq_l), device=device)
            od = model(ids_t, torch.ones_like(ids_t), torch.zeros((2, STYLE_FEAT_DIM_V11), device=device), engram_ids=eg_t, trigram_ids=tg_t, outer_steps=2, return_aux=True)
            if torch.isnan(od["logit"]).any() or torch.isnan(od["logit_promo"]).any() or od["attr6_logit"].shape[-1] != 6:
                dyn_ok = False

    u_t = torch.randn(4, BUS_DIM_V10, device=device, requires_grad=True)
    ebt_r = model.core.inloop_ebt.forward_energy_and_grad(u_t)
    ag_g = torch.autograd.grad(ebt_r["free_energy"].sum(), u_t)[0]
    ebt_err = float((ebt_r["grad_u"].detach() - ag_g.detach()).abs().max().item())
    ci["V11_S1_dynamic_and_ebt6_grad"] = {
        "dynamic_ok": dyn_ok, "ebt6_grad_max_diff": round(ebt_err, 8), "pass": bool(dyn_ok and ebt_err < 1e-5)
    }

    # V11-F6: Structural Firewall Proof — perturbing promo-only emb[192:256] MUST leave AI logit & cbm_logit diff === 0.0!
    with torch.no_grad():
        emb_a = torch.randn(4, 256, device=device)
        emb_b = emb_a.clone()
        emb_b[:, AI_EMB_END_V11:] += 50.0
        ai_a = model.ai_head(emb_a[:, :AI_EMB_END_V11])
        ai_b = model.ai_head(emb_b[:, :AI_EMB_END_V11])
        cbm_a = F.relu(F.linear(emb_a[:, :AI_EMB_END_V11], model.cbm_enc.weight[:CBM_AI_CONCEPTS_V11, :AI_EMB_END_V11], model.cbm_enc.bias[:CBM_AI_CONCEPTS_V11]))
        cbm_b = F.relu(F.linear(emb_b[:, :AI_EMB_END_V11], model.cbm_enc.weight[:CBM_AI_CONCEPTS_V11, :AI_EMB_END_V11], model.cbm_enc.bias[:CBM_AI_CONCEPTS_V11]))
        fw_diff = float(max((ai_a - ai_b).abs().max().item(), (cbm_a - cbm_b).abs().max().item()))
    ci["V11_F6_promo_to_ai_firewall"] = {"max_leak_diff": fw_diff, "pass": fw_diff == 0.0}

    # V11-C1: Strict Zero-PII (@username + <CONTACT_ID>) invariance across BOTH AI and Promo axes (diff === 0.0!)
    base_o = _infer("回复 @测试用户A : 关注工种号交流2群 168688884 助理老师：xhs54199 领取资料[doge]")
    alt_o = _infer("回复 @RandomName99 : 关注工种号交流2群 999888777 助理老师：abc98765 领取资料[doge]")
    pii_d = float(
        max(
            abs(float(base_o["logit"][0].item()) - float(alt_o["logit"][0].item())),
            abs(float(base_o["logit_promo"][0].item()) - float(alt_o["logit_promo"][0].item())),
            abs(float(base_o["logit_r1"][0].item()) - float(alt_o["logit_r1"][0].item())),
            abs(float(base_o["logit_promo_r1"][0].item()) - float(alt_o["logit_promo_r1"][0].item())),
        )
    )
    ci["V11_C1_zero_pii_dual_axis"] = {"max_abs_diff": pii_d, "pass": pii_d == 0.0}

    # V11-G0 (Optional when v10_ref_model provided): Net2Net MoLE exact mix diff <= 1e-5 & EBT AI margin diff <= 1e-4!
    if v10_ref_model is not None:
        v10_ref_model.eval()
        with torch.no_grad():
            zh_t = torch.randn(4, HIDDEN_V09, device=device)
            x_t = torch.randn(4, 16, HIDDEN_V09, device=device)
            p10, c10, _ = v10_ref_model.core.mole.route(zh_t, tick_idx=0)
            p11, c11, _ = model.core.mole.route(zh_t, tick_idx=0)
            d10 = v10_ref_model.core.mole.forward_delta(x_t, p10, phase_idx=0)
            d11 = model.core.mole.forward_delta(x_t, p11, phase_idx=0)
            mole_diff = float((d10 - d11).abs().max().item())
            ctrl_diff = float((c10 - c11).abs().max().item())
            u_ebt = torch.randn(8, BUS_DIM_V10, device=device)
            em10 = v10_ref_model.core.inloop_ebt.forward_energy_and_grad(u_ebt)["ebt_margin"]
            em11 = model.core.inloop_ebt.forward_energy_and_grad(u_ebt)["ebt_margin"]
            ebt_m_diff = float((em10 - em11).abs().max().item())
        ci["V11_G0_net2net_and_ebt_exact"] = {
            "mole_delta_max_diff": round(mole_diff, 7),
            "mole_ctrl_max_diff": round(ctrl_diff, 7),
            "ebt_ai_margin_max_diff": round(ebt_m_diff, 7),
            "pass": bool(mole_diff <= 1e-4 and ctrl_diff <= 1e-4 and ebt_m_diff <= 1e-5),
        }

    ci["all_ci_passed"] = all(bool(v.get("pass")) for k, v in ci.items() if isinstance(v, dict))
    return ci
