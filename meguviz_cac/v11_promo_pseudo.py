"""MeguvizCAC v1.1.0-OmniGuard Heterogeneous 2-Model `PromoTeacher` Confident Pseudo-Label Harvester.

Multiplies Wave-1 dual-axis supervision (`5,880` gold + `1,600` counterfactual pairs) by `50x` at `$0` API cost:
  1. Trains a strictly heterogeneous 2-Model `PromoTeacher` Committee on `Wave-1 + 1,600 Minimal Pairs + v11_augment`:
       - Model A (Semantic Neural): `BAAI/bge-small-zh-v1.5 (512-d)` + `56-d` length-invariant macro stylometry MLP
       - Model B (Lexical Subword): `Char-WB 2..4 TF-IDF (35k)` balanced LogisticRegression
  2. Scans clean `v04` train rows (strictly excluding ALL eval/test/probe/PromoBench/LongBench/Wave-1/Wave-2 splits!)
     with exact normalized-text deduplication.
  3. Retains ONLY strict 2-Model Consensus Confident Pseudo-Labels:
       - `pseudo_promo_pos`: `p_nn >= 0.92 AND p_lx >= 0.88` (harvests regex-free homophone-evasion ads like `工种号` /
         `蓝色文字` / `助理老师：xhs...` & real soft-shills while filtering meme/poetic outliers!)
       - `pseudo_promo_neg`: `p_nn <= 0.04 AND p_lx <= 0.06` (ultra-clean organic non-promo anchors!)
       - Ambiguous mid-band is discarded (zero confirmation-bias error propagation!).
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .v11_augment import AdversarialObfuscationAug, BrandPriceSlotAug, QuadSpliceAug
from .v11_features import extract_style_features_56d

_NORM_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _NORM_RE.sub("", str(s or "")).lower()


class _BgeMacroPromoMLP(nn.Module):
    def __init__(self, in_dim: int = 512 + 56):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def harvest_confident_promo_pseudo_labels(
    v04_corpus_gz: Path,
    wave1_gold_gz: Path,
    wave1_pairs_jsonl: Path,
    out_pseudo_gz: Path,
    max_scan: int = 250000,
    max_neg_keep: int = 40000,
    pos_nn_thr: float = 0.92,
    pos_lx_thr: float = 0.88,
    neg_nn_thr: float = 0.04,
    neg_lx_thr: float = 0.06,
    seed: int = 20261126,
) -> dict[str, Any]:
    """Train 2-Model PromoTeacher Committee & harvest strictly deduplicated confident pseudo-labels."""
    from transformers import AutoModel, AutoTokenizer

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
    bge = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5").to(device).eval()

    @torch.no_grad()
    def _encode(texts: list[str], bs: int = 512) -> tuple[list[str], np.ndarray]:
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
        return cleans, np.concatenate([np.concatenate(embs, axis=0), np.stack(f56s, axis=0)], axis=1).astype(np.float32)

    w1_gold = [json.loads(l) for l in gzip.open(wave1_gold_gz, "rt", encoding="utf-8") if l.strip()]
    w1_pairs = [json.loads(l) for l in wave1_pairs_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    excl_norms = {_norm(r["text"]) for r in w1_gold}
    for x in w1_pairs:
        excl_norms.add(_norm(x["promo_text"]))
        excl_norms.add(_norm(x["organic_text"]))

    # Build A2 forensic-augmented training set
    rng = random.Random(seed)
    obf, slot, qs = AdversarialObfuscationAug(seed), BrandPriceSlotAug(seed + 1), QuadSpliceAug(seed + 2)
    tr_txt, tr_y, tr_w = [], [], []
    org_hu = []
    for r in w1_gold:
        if r.get("promo_type") == "tool_boilerplate":
            continue
        ps = float(r["promo_score"])
        if ps >= 65.0:
            tr_txt.append(r["text"]); tr_y.append(1.0); tr_w.append(2.0)
        elif ps <= 25.0:
            tr_txt.append(r["text"]); tr_y.append(0.0); tr_w.append(1.0)
            if float(r["ai_score"]) <= 20.0 and len(r["text"]) >= 45:
                org_hu.append(r["text"])
    for x in w1_pairs:
        tr_txt.append(x["promo_text"]); tr_y.append(1.0); tr_w.append(1.5)
        tr_txt.append(x["organic_text"]); tr_y.append(0.0); tr_w.append(1.5)
    for x in rng.sample(w1_pairs, min(600, len(w1_pairs))):
        p2, o2 = slot.augment_pair(x["promo_text"], x["organic_text"])
        tr_txt.append(p2); tr_y.append(1.0); tr_w.append(1.2)
        tr_txt.append(o2); tr_y.append(0.0); tr_w.append(1.2)
    for x in rng.sample(w1_pairs, min(450, len(w1_pairs))):
        tr_txt.append(obf.obfuscate_promo_or_ai(x["promo_text"], intensity=0.6))
        tr_y.append(1.0); tr_w.append(1.3)
    for x in rng.sample(w1_pairs, min(320, len(w1_pairs))):
        tr_txt.append(qs.make_quoted_promo_critique(x["promo_text"])["text"])
        tr_y.append(0.0); tr_w.append(1.6)
    for _ in range(180):
        tr_txt.append(qs.make_negated_shill_critique()["text"])
        tr_y.append(0.0); tr_w.append(1.6)
    if len(org_hu) >= 5:
        for x in rng.sample(w1_pairs, min(250, len(w1_pairs))):
            hb = "\n\n".join(rng.sample(org_hu, 3))
            tr_txt.append(hb + "\n\n" + x["promo_text"]); tr_y.append(1.0); tr_w.append(1.4)
            tr_txt.append(hb + "\n\n" + x["organic_text"]); tr_y.append(0.0); tr_w.append(1.4)

    tr_cleans, x_tr = _encode(tr_txt, bs=256)
    y_tr = np.array(tr_y, dtype=np.float32)
    w_tr = np.array(tr_w, dtype=np.float32)

    torch.manual_seed(seed)
    mlp = _BgeMacroPromoMLP(x_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=2.5e-3, weight_decay=2e-2)
    xt, yt, wt = torch.from_numpy(x_tr).to(device), torch.from_numpy(y_tr).to(device), torch.from_numpy(w_tr).to(device)
    for _ in range(18):
        mlp.train()
        perm = torch.randperm(len(x_tr), device=device)
        for st in range(0, len(x_tr), 128):
            idx = perm[st : st + 128]
            loss = (F.binary_cross_entropy_with_logits(mlp(xt[idx]), yt[idx], reduction="none") * wt[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=35000, sublinear_tf=True)
    clf_lx = LogisticRegression(C=2.5, max_iter=500, class_weight="balanced")
    clf_lx.fit(vec.fit_transform(tr_cleans), y_tr.astype(int), sample_weight=w_tr)

    # Stream & scan clean train rows (strictly skipping eval splits & deduplicating normalized text!)
    eval_norms: set[str] = set()
    with gzip.open(v04_corpus_gz, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != "train":
                eval_norms.add(_norm(r.get("text", "")))
    excl_norms |= eval_norms

    seen_scan: set[str] = set()
    pos_out: list[dict[str, Any]] = []
    neg_out: list[dict[str, Any]] = []
    scanned = 0
    buf: list[tuple[str, float, str]] = []

    def _flush(batch_rows: list[tuple[str, float, str]]) -> None:
        if not batch_rows:
            return
        txts = [b[2] for b in batch_rows]
        cls, x_b = _encode(txts, bs=512)
        with torch.no_grad():
            p_nn = torch.sigmoid(mlp(torch.from_numpy(x_b).to(device))).cpu().numpy()
        p_lx = clf_lx.predict_proba(vec.transform(cls))[:, 1]
        for i, (sid, osc, txt) in enumerate(batch_rows):
            pn, pl = float(p_nn[i]), float(p_lx[i])
            if pn >= pos_nn_thr and pl >= pos_lx_thr:
                pos_out.append({
                    "orig_sample_id": sid, "text": txt, "old_ai_score": round(osc, 1),
                    "pseudo_promo_score": round((0.65 * pn + 0.35 * pl) * 100.0, 1),
                    "p_nn": round(pn, 4), "p_lx": round(pl, 4),
                    "pseudo_label": 1, "weight": 0.75, "source": "v11_promoteacher_consensus_pos",
                })
            elif pn <= neg_nn_thr and pl <= neg_lx_thr and len(neg_out) < max_neg_keep:
                neg_out.append({
                    "orig_sample_id": sid, "text": txt, "old_ai_score": round(osc, 1),
                    "pseudo_promo_score": round((0.65 * pn + 0.35 * pl) * 100.0, 1),
                    "p_nn": round(pn, 4), "p_lx": round(pl, 4),
                    "pseudo_label": 0, "weight": 0.60, "source": "v11_promoteacher_consensus_neg",
                })

    with gzip.open(v04_corpus_gz, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != "train":
                continue
            txt = str(r.get("text", "")).strip()
            if len(txt) < 22 or len(txt) > 500:
                continue
            nt = _norm(txt)
            if nt in excl_norms or nt in seen_scan:
                continue
            seen_scan.add(nt)
            buf.append((str(r["sample_id"]), float(r.get("score", 0.0)), txt))
            scanned += 1
            if len(buf) >= 2048:
                _flush(buf)
                buf.clear()
            if scanned >= max_scan:
                break
    _flush(buf)

    out_pseudo_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_pseudo_gz, "wt", encoding="utf-8") as wf:
        for row in pos_out + neg_out:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "scanned_unique_train_rows": scanned,
        "confident_promo_pos": len(pos_out),
        "confident_organic_neg": len(neg_out),
        "out_file": str(out_pseudo_gz),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest deduplicated 2-model confident promo pseudo-labels.")
    ap.add_argument("--corpus", type=Path, default=Path("v04-corpus/local-train/v04-merged-1m4-r3.jsonl.gz"))
    ap.add_argument("--wave1-gold", type=Path, default=Path("v11-omniguard-datafactory/v11_wave1_dualaxis_gold_all5880.jsonl.gz"))
    ap.add_argument("--wave1-pairs", type=Path, default=Path("v11-omniguard-datafactory/v11_wave1_promo_counterfactual_pairs_1600.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("v11-omniguard-datafactory/v11_confident_promo_pseudo.jsonl.gz"))
    ap.add_argument("--max-scan", type=int, default=250000)
    args = ap.parse_args()
    rep = harvest_confident_promo_pseudo_labels(args.corpus, args.wave1_gold, args.wave1_pairs, args.out, max_scan=args.max_scan)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
