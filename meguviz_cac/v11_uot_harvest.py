"""MeguvizCAC v1.1.0-OmniGuard `CDMAD + Sinkhorn UOT` Full-Corpus Harvester, P10-3 Gate & Connected-Component Cert Splitter.

Implements 2025-2026 frontier semi-supervised & data-centric upgrades over `v11_promo_pseudo.py` (0 B / 0.000 ms student cost):
  1. **Calibrated 5-Class Neural Committee (`3-Seed Ensemble` on `BGE-512 + 56-d`) + Lexical TF-IDF Guard**:
       - Classes: `0=none, 1=coupon_commerce, 2=self_promo_funnel, 3=soft_shill, 4=tool_boilerplate`.
       - Trained with `Wave-1 Gold + 1,600 Counterfactual Pairs + hardened v11_augment (PCR-ToxiCN, quoted/negated/fandom
         hard negatives) + purified chant hard negatives + balanced natural organic anchors` so the 5-class logits stay
         grounded in natural Bilibili organic comments!
  2. **CDMAD Counterfactual Prevalence Debiasing (`CVPR 2024`)**:
       - Subtracts the empirical neutral organic baseline logit `z_null = E_{D_null}[z(x)]` (`z_deb = z - beta * z_null`)
         before temperature-calibrated softmax, neutralizing the ~45%-labeled vs <1%-wild prior shift!
  3. **GPU Entropic Unbalanced / Partial Optimal Transport (`PROTOCOL ICML'25`, `Semi-UOT NeurIPS'25`)**:
       - Solves 15-step log-stabilized Sinkhorn UOT over `(B=8192, 5)` batches with relaxed row penalty `tau1` (rejecting
         uncertain rows cleanly) and wild target marginal `c_prior` (preserving rare `soft_shill` & `Q2` mass without FP blowup!).
       - Produces:
           * `Tier-1A Consensus Wild Positives` (high-precision UOT + lexical/cue confirmation -> auto pseudo-pos, `w=0.80`)
           * `Tier-1B Semantic-UOT Wild Positives` (very high calibrated neural UOT + ultra-low 3-seed std -> covert shills, `w=0.65`)
           * `Wave-2.5 Asymmetric Disagreement Queue` (`p_nn_comm >= 0.80` & `p_lx <= 0.65`, deduplicated `cos < 0.90`,
             prioritizing `soft_shill`, `Q2` AI-promos, & long `>=220ch` comments for external Gemini Pro labeling!)
           * `A-Res` Density-Inverse Organic Negative Reservoir (`80,000` rows, downweighting collapsed synthetic sources!)
  4. **Opus 5.5 P10-3 Committee Gate on `2k BGE k-NN` Hard Negatives**:
       - Scores all 2,000 boundary hard negatives with the CDMAD+UOT committee and marks `l_pceil_admitted = (uot_p_promo <= 0.10)`
         so unverified borderline covert shills are NEVER slammed by the `L_pceil` top-4 ceiling hammer!
  5. **Connected-Component (`cos >= 0.92`) Leak-Free `promo_val_cert` Splitter (Decision 2: `>= 600` short promos)**:
       - Builds the exact GPU cosine similarity graph (`cos >= 0.92`) over all verified short (`<= 96` chars) commercial promo
         positives (`Wave-1 gold + Wave-1 pairs + Tier-1A wild short promos`) and allocates whole connected components into
         `promo_train / promo_val_select / promo_val_cert (>= 600)` so exact Clopper-Pearson LTT (`alpha_pr = 1%`) is mathematically valid!
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .v11_augment import AdversarialObfuscationAug, BrandPriceSlotAug, QuadSpliceAug
from .v11_features import extract_style_features_56d
from .v11_promo_pseudo import _load_extra_exclusions, _norm, _sha_norm

PROMO_5CLASS = ("none", "coupon_commerce", "self_promo_funnel", "soft_shill", "tool_boilerplate")
COMM_CLASS_IDS = (1, 2, 3)

# Broad commercial/funnel/shill cue detector for Tier-1A verification & chant-spam separation
_BROAD_COMM_CUE_RE = re.compile(
    r"(?:券|劵|满\d+减|到手|史低|直降|百亿补贴|国补|神券|优惠|返佣|蓝链|蓝色文字|小蓝字|置顶|凑单|抄作业|"
    r"关注|互关|求关|私信|思信|滴我|主页|简介|加微|加v|vx|薇信|威信|绿泡泡|扣群|粉丝[群裙]|进[群裙]|君羊|"
    r"公[众粽]号|工种号|功重号|\bgzh\b|\bxhs\b|小红书|网盘|网盼|夸克|百度盘|接单|代做|橱窗|带货|淘宝|京东|拼多多|pdd|天猫|闲鱼|"
    r"闭眼入|闭眼冲|无脑入|无脑冲|亲测有效|亲测好用|回购|不踩雷|早买早享受|强烈安利|良心推荐|入股不亏|性价比无敌|质感封神|"
    r"充电会员|公测|预约领|礼包码|兑换码|免费领|戳我|私聊|check my channel|subscribe|🛰️|🐧|🍠|<CONTACT_ID>)",
    re.I,
)

# Source-level inverse collapse prior for A-Res organic reservoir (calibrated from Exp 1 SemDeDup audit!)
_ARES_SOURCE_WEIGHT: dict[str, float] = {
    "v04_confirmed_ai_plugin": 0.25,                  # heavy L2 template collapse (16.6% prune@0.92)
    "colab_gpu_distilled_from_hydrogen_xhigh": 0.65,  # moderate L1 exact/template collapse (5.0% prune@0.92)
    "multi_teacher_ensemble": 1.25,                   # highest entropy (H=0.984, 0.40% prune)
    "pre_llm_human_prior": 1.20,                      # clean organic pre-LLM baseline
    "v04_neon_hydrogen_consensus": 1.20,
    "v08_gemini_80k": 1.30,
}


class _BgeMacro5ClassMLP(nn.Module):
    def __init__(self, in_dim: int = 512 + 56, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 224),
            nn.LayerNorm(224),
            nn.SiLU(),
            nn.Dropout(0.22),
            nn.Linear(224, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def sinkhorn_uot_assign(
    probs: torch.Tensor,
    c_prior: torch.Tensor,
    eps: float = 0.08,
    tau1: float = 0.15,
    tau2: float = 0.40,
    n_iters: int = 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-stabilized GPU Entropic Unbalanced/Partial Optimal Transport (`PROTOCOL ICML'25` / `Semi-UOT NeurIPS'25`).

    Args:
        probs: `(N, K)` CDMAD-debiased calibrated class probabilities.
        c_prior: `(K,)` target wild class prior summing to 1.0.
        eps: entropic regularization temperature.
        tau1: row-marginal KL weight (smaller -> partial mass rejection for ambiguous rows).
        tau2: column-marginal KL weight (prevents rare tail class collapse without forcing garbage).
    Returns:
        q_norm: `(N, K)` normalized UOT posterior per sample.
        row_mass: `(N,)` ratio `N * sum_j P_ij` (>1 means stronger transport affinity than average; <0.5 rejected).
    """
    n, k = probs.shape
    device = probs.device
    p_safe = probs.clamp(min=1e-7, max=1.0)
    log_k = torch.log(p_safe) / float(eps)  # -C / eps = log(p) / eps

    log_r = torch.full((n,), -math.log(float(n)), device=device, dtype=torch.float32)
    log_c = torch.log(c_prior.to(device=device, dtype=torch.float32).clamp(min=1e-8))

    f1 = float(tau1) / float(tau1 + eps)
    f2 = float(tau2) / float(tau2 + eps)

    log_u = torch.zeros(n, device=device, dtype=torch.float32)
    log_v = torch.zeros(k, device=device, dtype=torch.float32)

    for _ in range(int(n_iters)):
        # log(K @ v) = logsumexp(log_k + log_v[None, :], dim=1)
        l_kv = torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_u = f1 * (log_r - l_kv)
        # log(K^T @ u) = logsumexp(log_k + log_u[:, None], dim=0)
        l_ktu = torch.logsumexp(log_k + log_u.unsqueeze(1), dim=0)
        log_v = f2 * (log_c - l_ktu)

    log_p = log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0)
    # Row-normalized posterior q_norm = softmax(log_p, dim=1)
    q_norm = F.softmax(log_p, dim=1)
    # Relative row transport mass: N * sum_j P_ij
    log_row_sum = torch.logsumexp(log_p, dim=1)
    row_mass = torch.exp((log_row_sum + math.log(float(n))).clamp(min=-20.0, max=20.0))
    return q_norm, row_mass


def gpu_connected_components_cos(embs: np.ndarray, thr: float = 0.92, block: int = 2048) -> list[list[int]]:
    """Exact connected components of the cosine similarity graph (`cos >= thr`) on GPU/CPU."""
    n = int(embs.shape[0])
    if n == 0:
        return []
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e_t = F.normalize(torch.from_numpy(embs).to(dev, dtype=torch.float32), p=2, dim=-1)
    for i in range(0, n, block):
        ei = e_t[i : i + block]
        for j in range(i, n, block):
            ej = e_t[j : j + block]
            sim = torch.matmul(ei, ej.T)
            if i == j:
                sim = torch.triu(sim, diagonal=1)
            pairs = (sim >= thr).nonzero(as_tuple=False).cpu().numpy()
            for r_idx, c_idx in pairs:
                _union(i + int(r_idx), j + int(c_idx))

    comps: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        comps[_find(idx)].append(idx)
    return list(comps.values())


def gpu_greedy_semdedup_indices(embs: np.ndarray, scores: np.ndarray, thr: float = 0.90, max_keep: int = 2500) -> list[int]:
    """Keep highest-priority items while greedily pruning near-duplicates (`cos >= thr`)."""
    n = int(embs.shape[0])
    if n == 0:
        return []
    order = np.argsort(-scores)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e_t = F.normalize(torch.from_numpy(embs[order]).to(dev, dtype=torch.float32), p=2, dim=-1)
    kept_local: list[int] = []
    kept_vecs: list[torch.Tensor] = []
    for pos in range(n):
        v = e_t[pos : pos + 1]
        if kept_vecs:
            mat = torch.cat(kept_vecs, dim=0)
            if float(torch.matmul(v, mat.T).max().item()) >= thr:
                continue
        kept_local.append(int(order[pos]))
        kept_vecs.append(v)
        if len(kept_local) >= max_keep:
            break
    return kept_local


def build_cc_promo_cert_split(
    short_records: list[dict[str, Any]],
    short_embs: np.ndarray,
    target_cert: int = 620,
    target_select: int = 240,
    cos_thr: float = 0.92,
    seed: int = 20261126,
) -> dict[str, Any]:
    """Allocate whole `cos >= 0.92` connected components into `cert / select / train` for valid LTT (`alpha_pr = 1%`).

    Prioritizes diverse real wild components (`Wave-1 gold` + `Tier-1A wild`) + counterfactual pair components
    stratified across the 3 commercial subtypes (`coupon_commerce / self_promo_funnel / soft_shill`) while
    strictly guaranteeing ZERO cross-split cosine >= `cos_thr` clone leakage!
    """
    comps = gpu_connected_components_cos(short_embs, thr=cos_thr)
    rng = random.Random(seed)

    # Summarize each connected component by dominant subtype & origin
    comp_infos: list[dict[str, Any]] = []
    for cid, idxs in enumerate(comps):
        st_cnt = Counter(short_records[i]["promo_type"] for i in idxs)
        dom_sub = st_cnt.most_common(1)[0][0]
        has_wild = any(short_records[i]["origin"] in ("wave1_gold", "uot_tier1a_wild") for i in idxs)
        comp_infos.append({
            "cid": cid,
            "idxs": idxs,
            "size": len(idxs),
            "dom_sub": dom_sub,
            "has_wild": has_wild,
        })

    # Group by (origin, dom_sub) so EVERY split gets proportional real-wild (wave1_gold + uot_tier1a_wild) AND pair coverage!
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for info in comp_infos:
        dom_orig = Counter(short_records[i]["origin"] for i in info["idxs"]).most_common(1)[0][0]
        by_cell[(dom_orig, info["dom_sub"])].append(info)
    for cell_key in by_cell:
        rng.shuffle(by_cell[cell_key])
        by_cell[cell_key].sort(key=lambda x: 0 if x["size"] <= 4 else 1)

    total_n = len(short_records)
    cert_frac = min(0.45, float(target_cert) / float(max(1, total_n)))
    sel_frac = min(0.20, float(target_select) / float(max(1, total_n)))

    splits: dict[str, list[int]] = {"promo_val_cert": [], "promo_val_select": [], "promo_train": []}
    used_cids: set[int] = set()

    # Pass 1: proportional quota inside every (origin, subtype) cell (preserves >50% of real wild promos in promo_train!)
    for cell_key in sorted(by_cell.keys()):
        arr = by_cell[cell_key]
        cell_n = sum(c["size"] for c in arr)
        c_quota = int(round(cell_n * cert_frac))
        s_quota = int(round(cell_n * sel_frac))
        c_got = s_got = 0
        for c in arr:
            if c["size"] > 16:
                continue
            if c_got < c_quota:
                used_cids.add(c["cid"])
                splits["promo_val_cert"].extend(c["idxs"])
                c_got += c["size"]
            elif s_got < s_quota:
                used_cids.add(c["cid"])
                splits["promo_val_select"].extend(c["idxs"])
                s_got += c["size"]

    # Pass 2: top-up cert/select to exact targets across subtypes if rounding fell slightly short
    sub_order = ["coupon_commerce", "self_promo_funnel", "soft_shill"]
    remaining_by_sub: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for info in comp_infos:
        if info["cid"] not in used_cids and info["size"] <= 16:
            remaining_by_sub[info["dom_sub"]].append(info)
    for s in sub_order:
        rng.shuffle(remaining_by_sub[s])

    def _top_up(bucket_name: str, cap: int) -> None:
        ptrs = {s: 0 for s in sub_order}
        progressed = True
        while len(splits[bucket_name]) < cap and progressed:
            progressed = False
            for s in sub_order:
                arr = remaining_by_sub.get(s, [])
                while ptrs[s] < len(arr) and arr[ptrs[s]]["cid"] in used_cids:
                    ptrs[s] += 1
                if ptrs[s] < len(arr) and len(splits[bucket_name]) < cap:
                    c = arr[ptrs[s]]
                    ptrs[s] += 1
                    used_cids.add(c["cid"])
                    splits[bucket_name].extend(c["idxs"])
                    progressed = True

    _top_up("promo_val_cert", target_cert)
    _top_up("promo_val_select", target_select)
    for info in comp_infos:
        if info["cid"] not in used_cids:
            splits["promo_train"].extend(info["idxs"])

    # Verify ZERO cross-split cosine >= cos_thr leakage!
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e_norm = F.normalize(torch.from_numpy(short_embs).to(dev, dtype=torch.float32), p=2, dim=-1)
    max_cross_cos = 0.0
    if splits["promo_val_cert"] and splits["promo_train"]:
        c_e = e_norm[splits["promo_val_cert"]]
        t_e = e_norm[splits["promo_train"]]
        max_cross_cos = max(max_cross_cos, float(torch.matmul(c_e, t_e.T).max().item()))
    if splits["promo_val_cert"] and splits["promo_val_select"]:
        c_e = e_norm[splits["promo_val_cert"]]
        s_e = e_norm[splits["promo_val_select"]]
        max_cross_cos = max(max_cross_cos, float(torch.matmul(c_e, s_e.T).max().item()))

    def _summarize(idxs: list[int]) -> dict[str, Any]:
        return {
            "count": len(idxs),
            "by_subtype": dict(Counter(short_records[i]["promo_type"] for i in idxs)),
            "by_origin": dict(Counter(short_records[i]["origin"] for i in idxs)),
        }

    return {
        "cos_thr": cos_thr,
        "total_short_promos": len(short_records),
        "total_connected_components": len(comps),
        "max_component_size": max((len(c) for c in comps), default=0),
        "max_cross_split_cos": round(max_cross_cos, 5),
        "zero_clone_leak_verified": bool(max_cross_cos < cos_thr),
        "cert_ge_600_verified": bool(len(splits["promo_val_cert"]) >= 600),
        "summary": {k: _summarize(v) for k, v in splits.items()},
        "splits": {
            k: [
                {
                    "id": short_records[i]["id"],
                    "origin": short_records[i]["origin"],
                    "promo_type": short_records[i]["promo_type"],
                    "norm_sha256": _sha_norm(_norm(short_records[i]["text"])),
                    "len": len(short_records[i]["text"]),
                }
                for i in idxs
            ]
            for k, idxs in splits.items()
        },
    }


def harvest_uot_cdmad_full_corpus(
    v04_corpus_gz: Path,
    wave1_gold_gz: Path,
    wave1_pairs_jsonl: Path,
    purified_prev_gz: Path,
    knn_hardneg_jsonl: Path,
    out_dir: Path,
    max_scan: int = 2000000,
    max_neg_keep: int = 80000,
    max_disagree_keep: int = 2500,
    seed: int = 20261126,
) -> dict[str, Any]:
    """Run full-corpus CDMAD + Sinkhorn UOT harvest, P10-3 k-NN gate, & connected-component cert split on GPU."""
    from transformers import AutoModel, AutoTokenizer

    t0 = time.perf_counter()
    root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/6] Loading BGE-small-zh-v1.5 on {device} & assembling 5-class training set...", flush=True)
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
    bge = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5").to(device).eval()

    @torch.no_grad()
    def _encode(texts: list[str], bs: int = 512) -> tuple[list[str], np.ndarray, np.ndarray]:
        cleans, f56s = [], []
        for t in texts:
            cl, f56, _ = extract_style_features_56d(t)
            cleans.append(cl)
            f56s.append(f56)
        embs = []
        for st in range(0, len(cleans), bs):
            enc = tok(cleans[st : st + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                h = F.normalize(bge(**enc).last_hidden_state[:, 0].float(), p=2, dim=-1)
            embs.append(h.cpu().numpy())
        e_arr = np.concatenate(embs, axis=0).astype(np.float32)
        f_arr = np.stack(f56s, axis=0).astype(np.float32)
        return cleans, e_arr, np.concatenate([e_arr, f_arr], axis=1).astype(np.float32)

    w1_gold = [json.loads(l) for l in gzip.open(wave1_gold_gz, "rt", encoding="utf-8") if l.strip()]
    w1_pairs = [json.loads(l) for l in wave1_pairs_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    excl_norms, excl_hashes = _load_extra_exclusions(root)
    for r in w1_gold:
        excl_norms.add(_norm(r["text"]))
    for x in w1_pairs:
        excl_norms.add(_norm(x["promo_text"]))
        excl_norms.add(_norm(x["organic_text"]))

    # Load previous purified chant hard-negatives & natural organic anchor slice (for CDMAD null baseline!)
    prev_chants: list[str] = []
    null_anchor_texts: list[str] = []
    if purified_prev_gz.is_file():
        with gzip.open(purified_prev_gz, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                src = str(r.get("source", ""))
                if "chant" in src or (r.get("pseudo_label") == 0 and float(r.get("p_nn", 0.0)) >= 0.85):
                    prev_chants.append(r["text"])
                elif r.get("pseudo_label") == 0 and str(r.get("old_source", "")) != "v04_confirmed_ai_plugin":
                    if len(null_anchor_texts) < 12000:
                        null_anchor_texts.append(r["text"])

    rng = random.Random(seed)
    rng.shuffle(null_anchor_texts)
    train_org_anchor = null_anchor_texts[:3000]
    cdmad_null_anchor = null_anchor_texts[3000:9000]  # strictly disjoint 6,000 natural organic baseline rows!

    obf, slot, qs = AdversarialObfuscationAug(seed), BrandPriceSlotAug(seed + 1), QuadSpliceAug(seed + 2)
    tr_txt: list[str] = []
    tr_cls: list[int] = []
    tr_w: list[float] = []
    org_hu: list[str] = []

    cls_to_id = {c: i for i, c in enumerate(PROMO_5CLASS)}
    for r in w1_gold:
        pt = str(r.get("promo_type", "none"))
        ps = float(r["promo_score"])
        if pt == "tool_boilerplate":
            tr_txt.append(r["text"]); tr_cls.append(4); tr_w.append(1.5)
        elif ps >= 65.0 and pt in ("coupon_commerce", "self_promo_funnel", "soft_shill"):
            tr_txt.append(r["text"]); tr_cls.append(cls_to_id[pt]); tr_w.append(2.2 if pt == "soft_shill" else 1.8)
        elif ps <= 25.0:
            tr_txt.append(r["text"]); tr_cls.append(0); tr_w.append(1.0)
            if float(r["ai_score"]) <= 20.0 and len(r["text"]) >= 45:
                org_hu.append(r["text"])

    for x in w1_pairs:
        pt = str(x.get("promo_type", "soft_shill"))
        cid = cls_to_id.get(pt, 3)
        tr_txt.append(x["promo_text"]); tr_cls.append(cid); tr_w.append(1.5)
        tr_txt.append(x["organic_text"]); tr_cls.append(0); tr_w.append(1.5)

    # Forensic-safe augmentations (PCR-ToxiCN homophones, slot swaps, quoted/negated/fandom hard negs, hidden tail)
    for x in rng.sample(w1_pairs, min(650, len(w1_pairs))):
        cid = cls_to_id.get(str(x.get("promo_type", "soft_shill")), 3)
        p2, o2 = slot.augment_pair(x["promo_text"], x["organic_text"])
        tr_txt.append(p2); tr_cls.append(cid); tr_w.append(1.2)
        tr_txt.append(o2); tr_cls.append(0); tr_w.append(1.2)
    for x in rng.sample(w1_pairs, min(550, len(w1_pairs))):
        cid = cls_to_id.get(str(x.get("promo_type", "soft_shill")), 3)
        tr_txt.append(obf.obfuscate_promo_or_ai(x["promo_text"], intensity=0.65))
        tr_cls.append(cid); tr_w.append(1.3)
    for x in rng.sample(w1_pairs, min(320, len(w1_pairs))):
        tr_txt.append(qs.make_quoted_promo_critique(x["promo_text"])["text"]); tr_cls.append(0); tr_w.append(1.8)
    for _ in range(180):
        tr_txt.append(qs.make_negated_shill_critique()["text"]); tr_cls.append(0); tr_w.append(1.8)
    for _ in range(150):
        tr_txt.append(qs.make_fandom_hype_hard_neg()["text"]); tr_cls.append(0); tr_w.append(1.8)
    for ch in prev_chants:
        tr_txt.append(ch); tr_cls.append(0); tr_w.append(2.2)
    for t_org in train_org_anchor:
        tr_txt.append(t_org); tr_cls.append(0); tr_w.append(0.8)
    if len(org_hu) >= 5:
        for x in rng.sample(w1_pairs, min(250, len(w1_pairs))):
            cid = cls_to_id.get(str(x.get("promo_type", "soft_shill")), 3)
            hb = "\n\n".join(rng.sample(org_hu, 3))
            tr_txt.append(hb + "\n\n" + x["promo_text"]); tr_cls.append(cid); tr_w.append(1.3)
            tr_txt.append(hb + "\n\n" + x["organic_text"]); tr_cls.append(0); tr_w.append(1.3)

    print(
        f"[2/6] Training 3-Seed 5-Class Neural Committee + Lexical TF-IDF Guard on {len(tr_txt):,} rows "
        f"(cls={dict(Counter(tr_cls))})...",
        flush=True,
    )
    tr_cleans, _, x_tr = _encode(tr_txt, bs=256)
    y_cls = np.array(tr_cls, dtype=np.int64)
    w_tr = np.array(tr_w, dtype=np.float32)

    xt = torch.from_numpy(x_tr).to(device)
    yt = torch.from_numpy(y_cls).to(device)
    wt = torch.from_numpy(w_tr).to(device)

    mlps: list[_BgeMacro5ClassMLP] = []
    for s_off in (0, 101, 202):
        torch.manual_seed(seed + s_off)
        m = _BgeMacro5ClassMLP(x_tr.shape[1], num_classes=5).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=2.2e-3, weight_decay=2.5e-2)
        for ep in range(20):
            m.train()
            perm = torch.randperm(len(x_tr), device=device)
            for st in range(0, len(x_tr), 128):
                idx = perm[st : st + 128]
                loss = (F.cross_entropy(m(xt[idx]), yt[idx], reduction="none", label_smoothing=0.02) * wt[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        m.eval()
        mlps.append(m)

    # Lexical binary commercial guard (excludes tool_boilerplate from positives!)
    y_bin_comm = np.isin(y_cls, COMM_CLASS_IDS).astype(np.int64)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=40000, sublinear_tf=True)
    clf_lx = LogisticRegression(C=2.5, max_iter=500, class_weight="balanced")
    clf_lx.fit(vec.fit_transform(tr_cleans), y_bin_comm, sample_weight=w_tr)

    # Compute CDMAD counterfactual null organic baseline logit vector on 6,000 disjoint natural organic rows!
    print(f"[3/6] Computing CDMAD null baseline on {len(cdmad_null_anchor):,} disjoint organic rows & diagnostic check...", flush=True)
    null_cleans, _, x_null = _encode(cdmad_null_anchor, bs=512)
    with torch.no_grad():
        xn_t = torch.from_numpy(x_null).to(device)
        null_logits = torch.stack([m(xn_t) for m in mlps], dim=0).mean(dim=0)  # (N_null, 5)
        z_null_mean = null_logits.mean(dim=0)  # (5,)
    # CDMAD debiasing factor beta: shifts mean neutral-organic logit to preserve natural wild calibration
    # Relative shift: subtract beta * (z_null_mean - z_null_mean[0])? Careful:
    # In softmax, shifting all logits by a constant is invariant, so we want to subtract excess commercial bias
    # relative to natural baseline: z_deb = z - beta * (z_null_mean - z_null_mean.mean()).
    # Wait: for a typical organic sample, z_null_mean[0] is ALREADY much higher than z_null_mean[1..3]!
    # Subtracting z_null_mean directly would EQUALIZE organic samples to uniform [0.2, 0.2, 0.2, 0.2, 0.2] — BAD!
    # Look at CDMAD (CVPR 2024) carefully:
    #   z_tilde(x) = z(x) - z_bias, where z_bias = z(x_blank) (or log(pi_train / pi_wild)) removes the *training-prior
    #   class bias* `log(pi_train_c / pi_wild_c)`!
    # Let's compute the exact Bayes prior-shift adjustment `delta_prior = log(pi_train) - log(pi_wild)` (Menon et al.
    # Logit Adjustment + CDMAD neutral-centering of non-none classes) so organic stays near 0.992 while relative
    # imbalance across (coupon, self_promo, soft_shill, tool) is exact!
    tr_counts = np.bincount(y_cls, minlength=5).astype(np.float64)
    pi_train = tr_counts / float(tr_counts.sum())
    pi_wild = np.array([0.9910, 0.0022, 0.0028, 0.0020, 0.0020], dtype=np.float64)
    # Dampened logit adjustment (gamma=0.65) + CDMAD inter-subtype null centering across classes 1..4
    gamma_prior = 0.65
    log_prior_adj = gamma_prior * (np.log(pi_train + 1e-8) - np.log(pi_wild + 1e-8))
    # CDMAD relative null bias ONLY among the 4 positive classes (prevents soft_shill vs coupon baseline drift!)
    zn_np = z_null_mean.cpu().numpy()
    sub_bias = zn_np[1:] - float(np.mean(zn_np[1:]))
    cdmad_adj = np.zeros(5, dtype=np.float32)
    cdmad_adj[1:] = 0.50 * sub_bias
    total_logit_shift = torch.from_numpy((log_prior_adj + cdmad_adj).astype(np.float32)).to(device)
    c_prior_t = torch.from_numpy(pi_wild.astype(np.float32)).to(device)

    @torch.no_grad()
    def _predict_committee(x_arr: np.ndarray, cleans: list[str]) -> dict[str, np.ndarray]:
        xt_b = torch.from_numpy(x_arr).to(device)
        all_logits = torch.stack([m(xt_b) for m in mlps], dim=0)  # (3, N, 5)
        raw_probs = F.softmax(all_logits, dim=-1)  # (3, N, 5)
        raw_comm_per_seed = raw_probs[:, :, 1:4].sum(dim=-1)  # (3, N)
        p_nn_raw = raw_comm_per_seed.mean(dim=0).cpu().numpy()
        p_nn_std = raw_comm_per_seed.std(dim=0).cpu().numpy()

        mean_logits = all_logits.mean(dim=0)  # (N, 5)
        deb_logits = mean_logits - total_logit_shift.unsqueeze(0)
        p_deb = F.softmax(deb_logits, dim=-1)  # (N, 5)
        p_lx = clf_lx.predict_proba(vec.transform(cleans))[:, 1].astype(np.float32)
        return {
            "p_deb": p_deb.cpu().numpy(),
            "p_deb_t": p_deb,
            "p_nn_raw": p_nn_raw.astype(np.float32),
            "p_nn_std": p_nn_std.astype(np.float32),
            "p_lx": p_lx,
        }

    # Diagnostic check on frozen PromoBench-48 & LongBench-24 (read-only telemetry; NEVER used for threshold fitting!)
    diag_metrics: dict[str, Any] = {}
    pb_p = root / "meguviz_cac/data/v11-promobench-seed.jsonl"
    if pb_p.is_file():
        pb_rows = [json.loads(l) for l in pb_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        pb_cls, _, pb_x = _encode([r["text"] for r in pb_rows], bs=128)
        pb_out = _predict_committee(pb_x, pb_cls)
        pb_y = np.array([int(r["y_promo"]) if "y_promo" in r else (1 if float(r.get("promo_score", 0)) >= 65 else 0) for r in pb_rows])
        pb_score_deb = pb_out["p_deb"][:, 1:4].sum(axis=1)
        pb_score_ens = 0.65 * pb_out["p_nn_raw"] + 0.35 * pb_out["p_lx"]
        diag_metrics["promobench48_deb_roc"] = round(float(roc_auc_score(pb_y, pb_score_deb)), 4)
        diag_metrics["promobench48_ens_roc"] = round(float(roc_auc_score(pb_y, pb_score_ens)), 4)
        diag_metrics["promobench48_ens_pr"] = round(float(average_precision_score(pb_y, pb_score_ens)), 4)
        diag_metrics["promobench48_deb_fp05"] = round(float(np.mean(pb_score_deb[pb_y == 0] >= 0.5)), 4)
    # Null organic baseline FP check on disjoint 6,000 rows
    null_out = _predict_committee(x_null, null_cleans)
    null_p_comm = null_out["p_deb"][:, 1:4].sum(axis=1)
    diag_metrics["null_organic_6k_pdeb_ge_05"] = round(float(np.mean(null_p_comm >= 0.50)), 5)
    diag_metrics["null_organic_6k_pdeb_ge_02"] = round(float(np.mean(null_p_comm >= 0.20)), 5)
    print(f"  [Diag] {json.dumps(diag_metrics, ensure_ascii=False)}", flush=True)

    # Pass 4: Collect non-train exclusions & stream full unique v04 train corpus in B=8,192 UOT batches!
    print(f"[4/6] Scanning full v04 corpus in B=8,192 Sinkhorn-UOT batches (max_scan={max_scan:,})...", flush=True)
    with gzip.open(v04_corpus_gz, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != "train":
                excl_norms.add(_norm(r.get("text", "")))

    seen_scan: set[str] = set()
    tier1a_pos: list[dict[str, Any]] = []
    tier1b_pos: list[dict[str, Any]] = []
    chant_negs: list[dict[str, Any]] = []
    disagree_cands: list[dict[str, Any]] = []
    disagree_embs: list[np.ndarray] = []
    # A-Res weighted organic negative reservoir min-heap: `(key, counter, rec)`
    ares_heap: list[tuple[float, int, dict[str, Any]]] = []
    ares_counter = 0
    neg_seen = 0
    scanned = 0
    sub_cnt_1a, sub_cnt_1b = Counter(), Counter()
    buf: list[tuple[str, float, str, str]] = []

    def _flush_uot(batch_rows: list[tuple[str, float, str, str]]) -> None:
        nonlocal neg_seen, ares_counter
        if not batch_rows:
            return
        txts = [b[2] for b in batch_rows]
        cleans, e_b, x_b = _encode(txts, bs=512)
        out = _predict_committee(x_b, cleans)
        p_deb_t = out["p_deb_t"]
        p_deb = out["p_deb"]
        p_nn_raw = out["p_nn_raw"]
        p_nn_std = out["p_nn_std"]
        p_lx = out["p_lx"]

        q_uot_t, row_mass_t = sinkhorn_uot_assign(p_deb_t, c_prior_t, eps=0.08, tau1=0.15, tau2=0.40, n_iters=15)
        q_uot = q_uot_t.cpu().numpy()
        row_mass = row_mass_t.cpu().numpy()

        for i, (sid, osc, txt, src) in enumerate(batch_rows):
            cl = cleans[i]
            rep4 = float(x_b[i, 512 + 45])  # index 45 in style56 is rep4_ratio!
            has_comm_cue = bool(_BROAD_COMM_CUE_RE.search(cl))
            p_comm_deb = float(p_deb[i, 1:4].sum())
            q_comm = float(q_uot[i, 1:4].sum())
            best_c = int(1 + np.argmax(q_uot[i, 1:4]))
            sub_name = PROMO_5CLASS[best_c]
            pn_r = float(p_nn_raw[i])
            pn_s = float(p_nn_std[i])
            pl = float(p_lx[i])
            rm = float(row_mass[i])

            # Guard against repetitive chant/meme spam (no commercial cues + high 4-gram repetition)
            if rep4 >= 0.55 and not has_comm_cue:
                if pn_r >= 0.70 or p_comm_deb >= 0.30:
                    chant_negs.append({
                        "orig_sample_id": sid, "text": txt, "old_ai_score": round(osc, 1), "old_source": src,
                        "pseudo_label": 0, "pseudo_promo_type": "none", "p_nn_raw": round(pn_r, 4),
                        "p_lx": round(pl, 4), "rep4_ratio": round(rep4, 3), "weight": 1.60,
                        "source": "v11_uot_chant_hard_neg",
                    })
                continue

            # Forensic Rule A: `v04_confirmed_ai_plugin` is synthetic non-commercial AI (Q1); never auto-promote to wild promo!
            is_synth_plugin = (src == "v04_confirmed_ai_plugin")

            # Tier-1A: High-precision UOT Consensus Wild Positives (safe for auto-training & cert candidates!)
            is_1a = (
                not is_synth_plugin
                and q_comm >= 0.85
                and p_comm_deb >= 0.55
                and pn_s <= 0.08
                and (pl >= 0.70 or (has_comm_cue and pn_r >= 0.92 and pl >= 0.45))
            )
            # Tier-1B: High-confidence Semantic-UOT Wild Positives (forensically verified `p_lx >= 0.54` precision cliff!)
            is_1b = (
                not is_synth_plugin
                and not is_1a
                and q_comm >= 0.90
                and p_comm_deb >= 0.65
                and pn_r >= 0.90
                and pn_s <= 0.05
                and pl >= 0.54
                and rm >= 0.60
                and len(txt) >= 28
            )
            if is_1a:
                sub_cnt_1a[sub_name] += 1
                tier1a_pos.append({
                    "orig_sample_id": sid, "text": txt, "old_ai_score": round(osc, 1), "old_source": src,
                    "pseudo_promo_score": round(min(99.5, (0.55 * q_comm + 0.30 * pn_r + 0.15 * pl) * 100.0), 1),
                    "pseudo_promo_type": sub_name,
                    "q_uot_comm": round(q_comm, 4), "p_comm_deb": round(p_comm_deb, 4),
                    "p_nn_raw": round(pn_r, 4), "p_nn_std": round(pn_s, 4), "p_lx": round(pl, 4),
                    "uot_row_mass": round(rm, 4), "has_comm_cue": has_comm_cue,
                    "pseudo_label": 1, "tier": "tier1a_uot_consensus", "weight": 0.80,
                    "source": "v11_uot_cdmad_tier1a_pos",
                })
            elif is_1b:
                sub_cnt_1b[sub_name] += 1
                tier1b_pos.append({
                    "orig_sample_id": sid, "text": txt, "old_ai_score": round(osc, 1), "old_source": src,
                    "pseudo_promo_score": round(min(98.0, (0.60 * q_comm + 0.40 * pn_r) * 100.0), 1),
                    "pseudo_promo_type": sub_name,
                    "q_uot_comm": round(q_comm, 4), "p_comm_deb": round(p_comm_deb, 4),
                    "p_nn_raw": round(pn_r, 4), "p_nn_std": round(pn_s, 4), "p_lx": round(pl, 4),
                    "uot_row_mass": round(rm, 4), "has_comm_cue": has_comm_cue,
                    "pseudo_label": 1, "tier": "tier1b_uot_semantic", "weight": 0.65,
                    "source": "v11_uot_cdmad_tier1b_pos",
                })

            # Asymmetric Epistemic Disagreement Queue -> Wave-2.5 Gemini Pro (real Bilibili comments only, excl. synth plugin!)
            if not is_synth_plugin and pn_r >= 0.80 and pl <= 0.65 and not is_1a and not is_1b and len(txt) >= 30:
                # Priority score favors covert soft_shill, Q2 AI-origin (osc >= 65), long >= 220ch, & high UOT disagreement
                prio = pn_r + 0.35 * q_comm
                if sub_name == "soft_shill":
                    prio += 0.30
                if osc >= 65.0:
                    prio += 0.25
                if len(txt) >= 220:
                    prio += 0.20
                if not has_comm_cue:
                    prio += 0.15
                disagree_cands.append({
                    "orig_sample_id": sid, "text": txt, "old_ai_score": round(osc, 1), "old_source": src,
                    "pred_subtype": sub_name, "p_nn_raw": round(pn_r, 4), "p_nn_std": round(pn_s, 4),
                    "p_comm_deb": round(p_comm_deb, 4), "q_uot_comm": round(q_comm, 4), "p_lx": round(pl, 4),
                    "has_comm_cue": has_comm_cue, "priority": round(prio, 4),
                    "queue": "wave25_asymmetric_disagreement",
                })
                disagree_embs.append(e_b[i])

            # Density-Inverse A-Res Organic Negative Reservoir (`80,000` rows, downweighting collapsed sources!)
            elif p_comm_deb <= 0.015 and pn_r <= 0.06 and pl <= 0.10 and float(p_deb[i, 4]) <= 0.15:
                neg_seen += 1
                w_src = _ARES_SOURCE_WEIGHT.get(src, 1.0)
                # Mild boost for substantive & long organic rows (len >= 96)
                if len(txt) >= 180:
                    w_src *= 1.35
                elif len(txt) >= 80:
                    w_src *= 1.15
                # A-Res key: u^(1 / w) -> larger w produces larger keys -> higher survival in min-heap of top K!
                u_rand = max(1e-12, rng.random())
                key = u_rand ** (1.0 / max(0.05, w_src))
                rec = {
                    "orig_sample_id": sid, "text": txt, "old_ai_score": round(osc, 1), "old_source": src,
                    "pseudo_promo_score": round(p_comm_deb * 100.0, 2), "pseudo_promo_type": "none",
                    "p_nn_raw": round(pn_r, 4), "p_lx": round(pl, 4),
                    "pseudo_label": 0, "weight": 0.60, "ares_w": round(w_src, 2),
                    "source": "v11_uot_ares_organic_neg",
                }
                ares_counter += 1
                if len(ares_heap) < max_neg_keep:
                    heapq.heappush(ares_heap, (key, ares_counter, rec))
                elif key > ares_heap[0][0]:
                    heapq.heapreplace(ares_heap, (key, ares_counter, rec))

    with gzip.open(v04_corpus_gz, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != "train":
                continue
            txt = str(r.get("text", "")).strip()
            L = len(txt)
            if L < 22 or L > 600:
                continue
            nt = _norm(txt)
            if nt in excl_norms or nt in seen_scan:
                continue
            if excl_hashes and _sha_norm(nt) in excl_hashes:
                continue
            seen_scan.add(nt)
            buf.append((str(r["sample_id"]), float(r.get("score", 0.0)), txt, str(r.get("score_source", ""))))
            scanned += 1
            if len(buf) >= 8192:
                _flush_uot(buf)
                buf.clear()
                if scanned % 65536 < 8192:
                    print(
                        f"  [UOT scan {scanned:,}] 1A={len(tier1a_pos):,} ({dict(sub_cnt_1a)}) | "
                        f"1B={len(tier1b_pos):,} ({dict(sub_cnt_1b)}) | disagree={len(disagree_cands):,} | "
                        f"chants={len(chant_negs):,} | neg_seen={neg_seen:,} | t={time.perf_counter()-t0:.1f}s",
                        flush=True,
                    )
            if scanned >= max_scan:
                break
    _flush_uot(buf)

    # Deduplicate Tier-1A + Tier-1B & Disagreement Queue with GPU greedy SemDeDup (< 0.92 / < 0.90)
    print(
        f"[5/6] Deduplicating harvested pools (raw 1A={len(tier1a_pos)}, 1B={len(tier1b_pos)}, disagree={len(disagree_cands)}), "
        f"gating 2k k-NN hard-negs (P10-3), & building connected-component promo_val_cert (>=600)...",
        flush=True,
    )
    if tier1a_pos:
        _, e_1a, _ = _encode([x["text"] for x in tier1a_pos], bs=512)
        sc_1a = np.array([x["pseudo_promo_score"] for x in tier1a_pos], dtype=np.float32)
        keep_1a = gpu_greedy_semdedup_indices(e_1a, sc_1a, thr=0.92, max_keep=len(tier1a_pos))
        tier1a_pos = [tier1a_pos[i] for i in keep_1a]
    if tier1b_pos:
        _, e_1b, _ = _encode([x["text"] for x in tier1b_pos], bs=512)
        sc_1b = np.array([x["pseudo_promo_score"] for x in tier1b_pos], dtype=np.float32)
        keep_1b = gpu_greedy_semdedup_indices(e_1b, sc_1b, thr=0.90, max_keep=len(tier1b_pos))
        tier1b_pos = [tier1b_pos[i] for i in keep_1b]

    disagree_selected: list[dict[str, Any]] = []
    if disagree_cands:
        e_dis = np.stack(disagree_embs, axis=0)
        sc_dis = np.array([x["priority"] for x in disagree_cands], dtype=np.float32)
        keep_dis = gpu_greedy_semdedup_indices(e_dis, sc_dis, thr=0.90, max_keep=max_disagree_keep)
        disagree_selected = [disagree_cands[i] for i in keep_dis]

    # Opus 5.5 P10-3: Score & gate `v11_bge_knn_promo_hard_negatives_2k.jsonl`
    knn_gate_summary: dict[str, Any] = {}
    knn_gated_path = out_dir / "v11_bge_knn_promo_hard_negatives_2k_uot_gated.jsonl"
    if knn_hardneg_jsonl.is_file():
        knn_rows = [json.loads(l) for l in knn_hardneg_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        k_cls, _, k_x = _encode([r["text"] for r in knn_rows], bs=512)
        k_out = _predict_committee(k_x, k_cls)
        admitted = 0
        flagged_borderline = 0
        out_knn_lines = []
        for idx, r in enumerate(knn_rows):
            p_deb_c = float(k_out["p_deb"][idx, 1:4].sum())
            p_nn_r = float(k_out["p_nn_raw"][idx])
            p_lx_v = float(k_out["p_lx"][idx])
            p_comm_gate = max(p_deb_c, 0.5 * p_nn_r + 0.5 * p_lx_v)
            is_admit = bool(p_comm_gate <= 0.10)
            if is_admit:
                admitted += 1
            else:
                flagged_borderline += 1
            r_out = dict(r)
            r_out["uot_p_promo_deb"] = round(p_deb_c, 4)
            r_out["uot_p_nn_raw"] = round(p_nn_r, 4)
            r_out["uot_p_lx"] = round(p_lx_v, 4)
            r_out["uot_p_gate"] = round(p_comm_gate, 4)
            r_out["l_pceil_admitted"] = is_admit
            r_out["stage1b_weight"] = 1.0
            out_knn_lines.append(json.dumps(r_out, ensure_ascii=False))
        knn_gated_path.write_text("\n".join(out_knn_lines) + "\n", encoding="utf-8")
        knn_gate_summary = {
            "total": len(knn_rows),
            "l_pceil_admitted_le_010": admitted,
            "deferred_to_b2_gt_010": flagged_borderline,
            "admit_rate": round(admitted / float(max(1, len(knn_rows))), 4),
            "out_file": str(knn_gated_path),
        }

    # Decision 2: Connected-component (`cos >= 0.92`) split of short (`<= 96` chars) commercial promos for `promo_val_cert >= 600`!
    short_records: list[dict[str, Any]] = []
    seen_short_norm: set[str] = set()
    for r in w1_gold:
        pt = str(r.get("promo_type", "none"))
        if float(r["promo_score"]) >= 65.0 and pt in ("coupon_commerce", "self_promo_funnel", "soft_shill") and len(r["text"]) <= 96:
            nt = _norm(r["text"])
            if nt not in seen_short_norm:
                seen_short_norm.add(nt)
                short_records.append({"id": str(r.get("sid", r.get("orig_sample_id", len(short_records)))), "origin": "wave1_gold", "promo_type": pt, "text": r["text"]})
    for r in tier1a_pos:
        if len(r["text"]) <= 96:
            nt = _norm(r["text"])
            if nt not in seen_short_norm:
                seen_short_norm.add(nt)
                short_records.append({"id": r["orig_sample_id"], "origin": "uot_tier1a_wild", "promo_type": r["pseudo_promo_type"], "text": r["text"]})
    for idx, x in enumerate(w1_pairs):
        pt = str(x.get("promo_type", "soft_shill"))
        if pt in ("coupon_commerce", "self_promo_funnel", "soft_shill") and len(x["promo_text"]) <= 96:
            nt = _norm(x["promo_text"])
            if nt not in seen_short_norm:
                seen_short_norm.add(nt)
                short_records.append({"id": f"w1pair_{idx}", "origin": "wave1_pair", "promo_type": pt, "text": x["promo_text"]})

    _, short_embs, _ = _encode([r["text"] for r in short_records], bs=512)
    split_manifest = build_cc_promo_cert_split(short_records, short_embs, target_cert=620, target_select=240, cos_thr=0.92, seed=seed)
    split_manifest_p = out_dir / "v11_promo_split_manifest.json"
    split_manifest_p.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Step 6: Persist all UOT harvest artifacts
    print(f"[6/6] Writing UOT harvest outputs to {out_dir}...", flush=True)
    neg_out = [item[2] for item in ares_heap]
    all_pos = tier1a_pos + tier1b_pos
    out_full_gz = out_dir / "v11_uot_cdmad_pseudo_full.jsonl.gz"
    with gzip.open(out_full_gz, "wt", encoding="utf-8") as wf:
        for row in all_pos + chant_negs + neg_out:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_pos_jsonl = out_dir / "v11_uot_wild_promo_pos.jsonl"
    out_pos_jsonl.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in all_pos) + ("\n" if all_pos else ""), encoding="utf-8")

    out_dis_jsonl = out_dir / "v11_wave25_disagreement_queue.jsonl"
    out_dis_jsonl.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in disagree_selected) + ("\n" if disagree_selected else ""), encoding="utf-8")

    neg_src_dist = Counter(x["old_source"] for x in neg_out)
    report = {
        "scanned_unique_train_rows": scanned,
        "diag_metrics": diag_metrics,
        "tier1a_consensus_pos": len(tier1a_pos),
        "tier1a_subtypes": dict(Counter(x["pseudo_promo_type"] for x in tier1a_pos)),
        "tier1a_q2_ai_ge_65": sum(1 for x in tier1a_pos if float(x["old_ai_score"]) >= 65.0),
        "tier1b_semantic_pos": len(tier1b_pos),
        "tier1b_subtypes": dict(Counter(x["pseudo_promo_type"] for x in tier1b_pos)),
        "tier1b_q2_ai_ge_65": sum(1 for x in tier1b_pos if float(x["old_ai_score"]) >= 65.0),
        "total_uot_wild_pos": len(all_pos),
        "chant_hard_negs_caught": len(chant_negs),
        "wave25_disagreement_raw": len(disagree_cands),
        "wave25_disagreement_dedup_saved": len(disagree_selected),
        "wave25_disagreement_subtypes": dict(Counter(x["pred_subtype"] for x in disagree_selected)),
        "wave25_disagreement_q2_ai_ge_65": sum(1 for x in disagree_selected if float(x["old_ai_score"]) >= 65.0),
        "wave25_disagreement_long_ge_220": sum(1 for x in disagree_selected if len(x["text"]) >= 220),
        "ares_organic_neg_seen": neg_seen,
        "ares_organic_neg_saved": len(neg_out),
        "ares_organic_neg_sources": dict(neg_src_dist.most_common(6)),
        "knn_p10_3_gate": knn_gate_summary,
        "promo_cert_split_summary": {
            "total_short_promos": split_manifest["total_short_promos"],
            "total_connected_components": split_manifest["total_connected_components"],
            "max_cross_split_cos": split_manifest["max_cross_split_cos"],
            "zero_clone_leak_verified": split_manifest["zero_clone_leak_verified"],
            "cert_ge_600_verified": split_manifest["cert_ge_600_verified"],
            "buckets": split_manifest["summary"],
        },
        "files": {
            "full_pseudo_gz": str(out_full_gz),
            "wild_pos_jsonl": str(out_pos_jsonl),
            "wave25_disagreement_jsonl": str(out_dis_jsonl),
            "promo_split_manifest_json": str(split_manifest_p),
        },
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "v11_uot_harvest_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_v11_uot_ci() -> dict[str, Any]:
    """Fast self-contained unit CI verifying Sinkhorn UOT convergence, connected-component split, & zero-clone leak."""
    torch.manual_seed(42)
    # 1. Sinkhorn UOT test on synthetic 256x5 imbalanced batch
    logits = torch.randn(256, 5) * 0.5
    logits[:, 0] += 4.0  # mostly organic
    logits[0:4, 1] += 8.0  # strong coupon
    logits[4:7, 3] += 8.0  # strong rare soft_shill
    probs = F.softmax(logits, dim=-1)
    c_prior = torch.tensor([0.96, 0.015, 0.01, 0.01, 0.005], dtype=torch.float32)
    q_norm, r_mass = sinkhorn_uot_assign(probs, c_prior, eps=0.08, tau1=0.15, tau2=0.40, n_iters=15)
    uot_ok = bool(
        torch.isfinite(q_norm).all().item()
        and torch.isfinite(r_mass).all().item()
        and float(q_norm[0:4, 1].mean().item()) > 0.90
        and float(q_norm[4:7, 3].mean().item()) > 0.90
        and float(q_norm[10:, 0].mean().item()) > 0.90
    )

    # 2. Connected-component splitter test with cloned pairs (cos >= 0.98) -> zero cross-split leakage!
    rng = np.random.default_rng(42)
    base_vecs = rng.normal(size=(700, 32)).astype(np.float32)
    base_vecs /= np.linalg.norm(base_vecs, axis=1, keepdims=True)
    # Create 200 near-clones of the first 200 vectors
    clones = base_vecs[:200] + 0.01 * rng.normal(size=(200, 32)).astype(np.float32)
    clones /= np.linalg.norm(clones, axis=1, keepdims=True)
    all_vecs = np.concatenate([base_vecs, clones], axis=0)
    subs = ("coupon_commerce", "self_promo_funnel", "soft_shill")
    recs = [{"id": f"s{i}", "origin": "wave1_pair", "promo_type": subs[i % 3], "text": f"t_{i}"} for i in range(900)]
    sp = build_cc_promo_cert_split(recs, all_vecs, target_cert=610, target_select=120, cos_thr=0.92, seed=42)
    cc_ok = bool(sp["zero_clone_leak_verified"] and sp["cert_ge_600_verified"] and sp["total_connected_components"] <= 700)
    return {
        "uot_ok": uot_ok,
        "cc_split_ok": cc_ok,
        "cc_max_cross_cos": sp["max_cross_split_cos"],
        "cc_cert_count": sp["summary"]["promo_val_cert"]["count"],
        "pass": bool(uot_ok and cc_ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="CDMAD + Sinkhorn UOT full-corpus promo harvester & cert splitter.")
    ap.add_argument("--ci", action="store_true", help="Run fast unit CI only.")
    ap.add_argument("--corpus", type=Path, default=Path("v04-corpus/local-train/v04-merged-1m4-r3.jsonl.gz"))
    ap.add_argument("--wave1-gold", type=Path, default=Path("v11-omniguard-datafactory/v11_wave1_dualaxis_gold_all5880.jsonl.gz"))
    ap.add_argument("--wave1-pairs", type=Path, default=Path("v11-omniguard-datafactory/v11_wave1_promo_counterfactual_pairs_1600.jsonl"))
    ap.add_argument("--purified-prev", type=Path, default=Path("v11-omniguard-datafactory/v11_confident_promo_pseudo_purified.jsonl.gz"))
    ap.add_argument("--knn-hardneg", type=Path, default=Path("v11-omniguard-datafactory/v11_bge_knn_promo_hard_negatives_2k.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("v11-omniguard-datafactory"))
    ap.add_argument("--max-scan", type=int, default=2000000)
    ap.add_argument("--max-neg-keep", type=int, default=80000)
    ap.add_argument("--max-disagree-keep", type=int, default=2500)
    args = ap.parse_args()
    if args.ci:
        res = run_v11_uot_ci()
        print(json.dumps(res, ensure_ascii=False, indent=2))
        assert res["pass"]
        return
    rep = harvest_uot_cdmad_full_corpus(
        args.corpus, args.wave1_gold, args.wave1_pairs, args.purified_prev, args.knn_hardneg, args.out_dir,
        max_scan=args.max_scan, max_neg_keep=args.max_neg_keep, max_disagree_keep=args.max_disagree_keep,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
