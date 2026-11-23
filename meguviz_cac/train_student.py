"""Train, temperature-calibrate, and INT8-ONNX export MeguvizCAC (BAAI/bge-small-zh-v1.5)."""

import argparse
import gzip
import hashlib
import json
import math
import random
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


CODE_LIST = ("A", "R", "E", "T", "P", "H", "S", "Q")
CODE_TO_IDX = {c: i for i, c in enumerate(CODE_LIST)}
PRAG_TAGS = (
    "recap",
    "contrast_thesis",
    "assistant_leak",
    "contextual_ellipsis",
    "external_personal_exp",
    "irony_parody",
    "lyrical_emotion",
    "community_slang",
)
DEFAULT_BACKBONE = "BAAI/bge-small-zh-v1.5"
MAX_LEN = 192
HEAD_CHARS = 140
TAIL_CHARS = 50


def prepare_text(text: str) -> str:
    """Normalize and apply head+tail truncation keeping both openers and closers within 192 tokens."""
    t = " ".join(unicodedata.normalize("NFKC", text).split())
    if len(t) > MAX_LEN:
        t = t[:HEAD_CHARS] + " " + t[-TAIL_CHARS:]
    return t


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_rescored_corpus(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_qwen_fused(path: Path) -> dict[str, dict]:
    fused = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") not in (None, "train"):
                raise ValueError(f"防泄漏校验失败: stage2-fused 发现非 train 样本 {r.get('sample_id')} ({r.get('split')})")
            fused[r["sample_id"]] = r
    return fused


def load_pairs(path: Path, forbidden_texts: set[str]) -> list[dict]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        ai_t = prepare_text(p["ai_text"])
        hu_t = prepare_text(p["human_text"])
        if ai_t in forbidden_texts or hu_t in forbidden_texts:
            continue
        pairs.append({
            "ai_text": ai_t,
            "human_text": hu_t,
            "weight": float(p.get("weight", 1.0)),
            "category": p.get("category", "literary"),
        })
    return pairs


def build_train_records(rows: list[dict], synth_cap: int, qwen_map: dict[str, dict] | None, seed: int = 20261123):
    """Build strictly split=='train' weighted dataset (zero validation/test/probe leakage)."""
    rng = random.Random(seed)
    train_items = []
    bili_train = [r for r in rows if r["cohort"] == "bili" and r["split"] == "train"]
    synth_train = [
        r for r in rows
        if r["cohort"] in ("deepseek", "other_models") and r["split"] == "train"
        and len(r["text"]) >= 26 and float(r["score"]) >= 70.0
    ]
    if synth_cap > 0 and len(synth_train) > synth_cap:
        synth_train = rng.sample(synth_train, synth_cap)
    elif synth_cap == 0:
        synth_train = []

    for r in bili_train:
        uid = r["sample_id"]
        score = float(r["score"])
        code = r.get("reason_code", "")
        code_idx = CODE_TO_IDX.get(code, -100)
        unc = float(r.get("uncertainty", 10.0))
        is_direct = r.get("score_source") == "hydrogen_xhigh_direct"
        dsh_boost = 1.5 if r.get("dsh") else 1.0

        if is_direct:
            w = 3.0
        elif unc < 15.0 and (score <= 28.0 or score >= 72.0):
            w = 1.0 * dsh_boost
        else:
            w = 0.35 * dsh_boost

        prag_vec = [-1.0] * len(PRAG_TAGS)
        if qwen_map and uid in qwen_map:
            qr = qwen_map[uid]
            qw = float(qr.get("quality_weight", w))
            if qw <= 0.0:
                continue
            score = float(qr.get("fused_score", score))
            if qr.get("reason_code") in CODE_TO_IDX:
                code_idx = CODE_TO_IDX[qr["reason_code"]]
            tags = qr.get("pragmatic_tags")
            if isinstance(tags, list) and len(tags) == len(PRAG_TAGS):
                prag_vec = [float(x) for x in tags]
            w = qw * (1.2 if r.get("dsh") else 1.0)

        train_items.append({
            "sample_id": uid,
            "text": prepare_text(r["text"]),
            "y_score": float(np.clip(score / 100.0, 0.0, 1.0)),
            "code_idx": code_idx,
            "weight": float(w),
            "prag_vec": prag_vec,
        })

    for r in synth_train:
        uid = r["sample_id"]
        score = float(r["score"])
        code_idx = CODE_TO_IDX.get(r.get("reason_code", ""), -100)
        w = 0.15
        prag_vec = [-1.0] * len(PRAG_TAGS)
        if qwen_map and uid in qwen_map:
            qr = qwen_map[uid]
            qw = float(qr.get("quality_weight", 0.0))
            if qw <= 0.0:
                continue
            score = float(qr.get("fused_score", score))
            tags = qr.get("pragmatic_tags")
            if isinstance(tags, list) and len(tags) == len(PRAG_TAGS):
                prag_vec = [float(x) for x in tags]
            w = 0.25
        train_items.append({
            "sample_id": uid,
            "text": prepare_text(r["text"]),
            "y_score": float(np.clip(score / 100.0, 0.0, 1.0)),
            "code_idx": code_idx,
            "weight": float(w),
            "prag_vec": prag_vec,
        })

    return train_items


def compute_eval_metrics(y_true_score: np.ndarray, y_prob: np.ndarray, threshold_f05: float | None = None, threshold_fpr05: float | None = None) -> dict:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    y_bin = (y_true_score >= 70.0).astype(bool)
    pred_100 = y_prob * 100.0
    mae = float(np.mean(np.abs(y_true_score - pred_100)))
    rmse = float(np.sqrt(np.mean((y_true_score - pred_100) ** 2)))
    sp = float(spearmanr(y_true_score, y_prob).statistic) if len(np.unique(y_true_score)) > 1 else 0.0
    pe = float(pearsonr(y_true_score, y_prob).statistic) if len(np.unique(y_true_score)) > 1 else 0.0
    out = {
        "n": int(len(y_true_score)),
        "positives_ge70": int(y_bin.sum()),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "spearman": round(sp, 4),
        "pearson": round(pe, 4),
    }
    if 0 < y_bin.sum() < len(y_bin):
        out["auroc"] = round(float(roc_auc_score(y_bin, y_prob)), 4)
        out["ap"] = round(float(average_precision_score(y_bin, y_prob)), 4)

    def at_cutoff(cut: float) -> dict:
        pred = y_prob >= cut
        tp = int((y_bin & pred).sum())
        fp = int((~y_bin & pred).sum())
        fn = int((y_bin & ~pred).sum())
        tn = int((~y_bin & ~pred).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        f05 = 1.25 * tp / (1.25 * tp + fp + 0.25 * fn) if (1.25 * tp + fp + 0.25 * fn) else 0.0
        return {
            "threshold": round(float(cut), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "fpr": round(fpr, 4), "f05": round(f05, 4),
        }

    out["at_070"] = at_cutoff(0.70)
    if threshold_f05 is not None:
        out["at_val_f05"] = at_cutoff(threshold_f05)
    if threshold_fpr05 is not None:
        out["at_val_fpr05"] = at_cutoff(threshold_fpr05)
    return out


def find_val_thresholds(y_true_score: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import precision_recall_curve

    y_bin = (y_true_score >= 70.0).astype(bool)
    prec, rec, vals = precision_recall_curve(y_bin, y_prob)
    prec, rec = prec[:-1], rec[:-1]
    f05 = 1.25 * prec * rec / np.maximum(0.25 * prec + rec, 1e-12)
    t_f05 = float(vals[int(np.argmax(f05))]) if len(vals) else 0.70

    neg = np.sort(y_prob[~y_bin])[::-1]
    max_fp = max(0, int(math.floor(len(neg) * 0.05)))
    if len(neg) and max_fp < len(neg):
        t_fpr05 = float(np.nextafter(neg[max_fp], np.inf))
    else:
        t_fpr05 = 0.70
    return t_f05, t_fpr05


def fit_temperature(logits: np.ndarray, y_soft: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    from scipy.special import expit

    def nll(temp: float) -> float:
        p = np.clip(expit(logits / temp), 1e-6, 1.0 - 1e-6)
        return float(-np.mean(y_soft * np.log(p) + (1.0 - y_soft) * np.log(1.0 - p)))

    res = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
    return float(res.x)


def run_dry(data_path: Path, qwen_path: Path | None, pairs_path: Path | None) -> None:
    t0 = time.monotonic()
    rows = load_rescored_corpus(data_path)
    qwen_map = load_qwen_fused(qwen_path) if qwen_path else None
    val_texts = {prepare_text(r["text"]) for r in rows if r["split"] != "train"}
    pairs = load_pairs(pairs_path, val_texts) if pairs_path else []
    tr_0 = build_train_records(rows, synth_cap=0, qwen_map=qwen_map)
    tr_5k = build_train_records(rows, synth_cap=5000, qwen_map=qwen_map)
    val_gold = [r for r in rows if r["split"] == "validation" and r["score_source"] == "hydrogen_xhigh_direct"]
    test_gold = [r for r in rows if r["split"] == "test" and r["score_source"] == "hydrogen_xhigh_direct"]
    probe = [r for r in rows if r["cohort"] == "user_confirmed"]
    print(json.dumps({
        "dry_run": "PASS",
        "total_rows": len(rows),
        "train_items_synth0": len(tr_0),
        "train_items_synth5k": len(tr_5k),
        "val_gold": len(val_gold),
        "test_gold": len(test_gold),
        "probe_u001_u017": len(probe),
        "qwen_rows": len(qwen_map) if qwen_map else 0,
        "pairs_rows": len(pairs),
        "sample_truncated_len": len(prepare_text("甲" * 240)),
        "elapsed_s": round(time.monotonic() - t0, 2),
    }, ensure_ascii=False, indent=2))


def train_and_export(args) -> None:
    import inspect
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
    from scipy.special import expit
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 进行学生模型蒸馏训练。")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    rows = load_rescored_corpus(args.data.resolve())
    qwen_map = load_qwen_fused(args.qwen_data.resolve()) if args.qwen_data else None
    forbidden = {prepare_text(r["text"]) for r in rows if r["split"] != "train"}
    pairs = load_pairs(args.pairs_data.resolve(), forbidden) if args.pairs_data else []

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    class StudentDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    def collate_main(batch):
        enc = tokenizer([x["text"] for x in batch], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return (
            enc["input_ids"],
            enc["attention_mask"],
            torch.tensor([x["y_score"] for x in batch], dtype=torch.float32),
            torch.tensor([x["code_idx"] for x in batch], dtype=torch.long),
            torch.tensor([x["weight"] for x in batch], dtype=torch.float32),
            torch.tensor([x["prag_vec"] for x in batch], dtype=torch.float32),
        )

    def collate_eval(texts):
        enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    class StudentModel(nn.Module):
        def __init__(self, backbone_name: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(backbone_name)
            h = self.encoder.config.hidden_size
            self.drop = nn.Dropout(0.1)
            self.score_head = nn.Linear(h, 1)
            self.code_head = nn.Linear(h, len(CODE_LIST))
            self.prag_head = nn.Linear(h, len(PRAG_TAGS))

        def encode(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state[:, 0]

        def forward(self, input_ids, attention_mask):
            cls = self.drop(self.encode(input_ids, attention_mask))
            return self.score_head(cls).squeeze(-1), self.code_head(cls), self.prag_head(cls)

    class ExportWrapper(nn.Module):
        def __init__(self, encoder, score_head, temp: float):
            super().__init__()
            self.encoder = encoder
            self.score_head = score_head
            self.register_buffer("temperature", torch.tensor(temp, dtype=torch.float32))

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0]
            logit = self.score_head(cls).squeeze(-1) / self.temperature
            prob = torch.sigmoid(logit)
            return prob, logit

    val_gold = [r for r in rows if r["split"] == "validation" and r["score_source"] == "hydrogen_xhigh_direct"]
    val_dsh_gold = [r for r in val_gold if r["dsh"]]
    test_gold = [r for r in rows if r["split"] == "test" and r["score_source"] == "hydrogen_xhigh_direct"]
    test_dsh_gold = [r for r in test_gold if r["dsh"]]
    test_bili_all = [r for r in rows if r["cohort"] == "bili" and r["split"] == "test"]
    test_dsh_all = [r for r in test_bili_all if r["dsh"]]
    probe_rows = [r for r in rows if r["cohort"] == "user_confirmed"]

    val_texts = [prepare_text(r["text"]) for r in val_gold]
    val_scores = np.array([float(r["score"]) for r in val_gold], dtype=np.float32)

    def predict_logits(mdl, texts_list: list[str], bs: int = 512) -> np.ndarray:
        mdl.eval()
        logits = []
        with torch.no_grad():
            for i in range(0, len(texts_list), bs):
                ids, mask = collate_eval(texts_list[i:i + bs])
                ids, mask = ids.to(device), mask.to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    sc_logit, _, _ = mdl(ids, mask)
                    logits.append(sc_logit.float().cpu().numpy())
        return np.concatenate(logits) if logits else np.array([], dtype=np.float32)

    def train_one(synth_cap: int):
        torch.manual_seed(20261123)
        items = build_train_records(rows, synth_cap=synth_cap, qwen_map=qwen_map)
        loader = DataLoader(StudentDataset(items), batch_size=args.batch_size, shuffle=True, collate_fn=collate_main, num_workers=0, pin_memory=True)
        mdl = StudentModel(args.backbone).to(device)
        opt = torch.optim.AdamW(mdl.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = max(1, len(loader) * args.epochs)
        sched = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * 0.08)), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        bce_none = nn.BCEWithLogitsLoss(reduction="none")
        ce_code = nn.CrossEntropyLoss(ignore_index=-100)
        margin_loss_fn = nn.MarginRankingLoss(margin=0.25)

        rng = random.Random(20261124)
        pair_bs = 16

        t_tr = time.monotonic()
        for ep in range(args.epochs):
            mdl.train()
            for step, (ids, mask, y_sc, y_cd, w, prag) in enumerate(loader):
                ids, mask = ids.to(device), mask.to(device)
                y_sc, y_cd, w, prag = y_sc.to(device), y_cd.to(device), w.to(device), prag.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    l_sc, l_cd, l_pr = mdl(ids, mask)
                    loss_sc = (bce_none(l_sc, y_sc) * w).mean()
                    loss_cd = ce_code(l_cd, y_cd)
                    loss = loss_sc + 0.20 * loss_cd
                    prag_mask = (prag >= 0.0)
                    if prag_mask.any():
                        loss_pr = bce_none(l_pr, prag.clamp(min=0.0))[prag_mask].mean()
                        loss = loss + 0.25 * loss_pr
                    if pairs:
                        p_batch = rng.sample(pairs, min(pair_bs, len(pairs)))
                        ai_ids, ai_mask = collate_eval([p["ai_text"] for p in p_batch])
                        hu_ids, hu_mask = collate_eval([p["human_text"] for p in p_batch])
                        ai_ids, ai_mask = ai_ids.to(device), ai_mask.to(device)
                        hu_ids, hu_mask = hu_ids.to(device), hu_mask.to(device)
                        l_ai, _, _ = mdl(ai_ids, ai_mask)
                        l_hu, _, _ = mdl(hu_ids, hu_mask)
                        p_ai = torch.sigmoid(l_ai)
                        p_hu = torch.sigmoid(l_hu)
                        ones = torch.ones_like(p_ai)
                        loss_pair = margin_loss_fn(p_ai, p_hu, ones)
                        # Also gently anchor pair endpoints
                        loss_end = 0.5 * (bce_none(l_ai, torch.full_like(l_ai, 0.85)).mean() + bce_none(l_hu, torch.full_like(l_hu, 0.15)).mean())
                        loss = loss + 0.40 * (loss_pair + 0.5 * loss_end)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            print(f"[synth_cap={synth_cap}] Epoch {ep+1}/{args.epochs} elapsed={time.monotonic()-t_tr:.1f}s", flush=True)

        val_logits = predict_logits(mdl, val_texts)
        temp = fit_temperature(val_logits, val_scores / 100.0)
        val_probs = expit(val_logits / temp)
        m_val = compute_eval_metrics(val_scores, val_probs)
        dsh_mask = np.array([r["dsh"] for r in val_gold], dtype=bool)
        m_val_dsh = compute_eval_metrics(val_scores[dsh_mask], val_probs[dsh_mask])
        print(f"[synth_cap={synth_cap}] Val Gold: {json.dumps(m_val)} | Val DSH Gold: {json.dumps(m_val_dsh)} | T={temp:.4f}", flush=True)
        return mdl, temp, m_val, m_val_dsh, val_probs

    ablation_caps = (0, 5000) if args.ablate_synth else (args.synth_cap,)
    best_mdl = None
    best_cap = ablation_caps[0]
    best_temp = 1.0
    best_rank = -1.0
    ablation_results = {}

    for cap in ablation_caps:
        mdl, temp, m_val, m_val_dsh, _ = train_one(cap)
        rank = m_val_dsh.get("auroc", 0.0) + m_val_dsh["spearman"] + m_val.get("auroc", 0.0)
        ablation_results[f"synth_cap_{cap}"] = {"val_gold": m_val, "val_dsh_gold": m_val_dsh, "temperature": round(temp, 4)}
        if rank > best_rank:
            if best_mdl is not None:
                del best_mdl
                torch.cuda.empty_cache()
            best_rank = rank
            best_mdl = mdl
            best_cap = cap
            best_temp = temp
        else:
            del mdl
            torch.cuda.empty_cache()

    # Calibration thresholds on validation DSH Gold
    val_dsh_probs = expit(predict_logits(best_mdl, [prepare_text(r["text"]) for r in val_dsh_gold]) / best_temp)
    val_dsh_scores = np.array([float(r["score"]) for r in val_dsh_gold], dtype=np.float32)
    t_f05, t_fpr05 = find_val_thresholds(val_dsh_scores, val_dsh_probs)

    def eval_group(subset: list[dict]) -> dict:
        probs = expit(predict_logits(best_mdl, [prepare_text(r["text"]) for r in subset]) / best_temp)
        true_sc = np.array([float(r["score"]) for r in subset], dtype=np.float32)
        return compute_eval_metrics(true_sc, probs, t_f05, t_fpr05)

    probe_probs = expit(predict_logits(best_mdl, [prepare_text(r["text"]) for r in probe_rows]) / best_temp)
    probe_true = np.array([float(r["score"]) for r in probe_rows], dtype=np.float32)
    probe_metrics = {
        **compute_eval_metrics(probe_true, probe_probs, t_f05, t_fpr05),
        "pred_mean_100": round(float(probe_probs.mean() * 100.0), 2),
        "pred_median_100": round(float(np.median(probe_probs) * 100.0), 2),
        "ge_060": int((probe_probs >= 0.60).sum()),
        "ge_070": int((probe_probs >= 0.70).sum()),
        "ge_val_f05": int((probe_probs >= t_f05).sum()),
        "ge_val_fpr05": int((probe_probs >= t_fpr05).sum()),
        "details": [
            {"id": r["sample_id"], "hx_score": r["score"], "student_prob": round(float(p), 4)}
            for r, p in sorted(zip(probe_rows, probe_probs), key=lambda x: x[0]["sample_id"])
        ],
    }

    # Evaluate all splits on GPU BEFORE moving encoder to CPU for ONNX export
    eval_val_gold = eval_group(val_gold)
    eval_val_dsh_gold = eval_group(val_dsh_gold)
    eval_test_gold = eval_group(test_gold)
    eval_test_dsh_gold = eval_group(test_dsh_gold)
    eval_test_bili_all = eval_group(test_bili_all)
    eval_test_dsh_all = eval_group(test_dsh_all)
    sample_texts = [prepare_text(r["text"]) for r in test_gold[:256]]
    pt_probs = expit(predict_logits(best_mdl, sample_texts) / best_temp)

    # Export FP32 -> Dynamic INT8 ONNX (auxiliary heads pruned for zero runtime overhead!)
    best_mdl.eval()
    wrapper = ExportWrapper(best_mdl.encoder, best_mdl.score_head, best_temp).cpu().eval()
    dummy_ids = torch.ones((2, 32), dtype=torch.long)
    dummy_mask = torch.ones((2, 32), dtype=torch.long)
    fp32_onnx = out_dir / "model_fp32.onnx"
    int8_onnx = out_dir / f"meguvizcac-{args.version}-int8.onnx"
    export_kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["prob", "logit"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "prob": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(wrapper, (dummy_ids, dummy_mask), str(fp32_onnx), **export_kwargs)
    quantize_dynamic(str(fp32_onnx), str(int8_onnx), weight_type=QuantType.QInt8)
    fp32_onnx.unlink(missing_ok=True)

    # Parity & CPU latency check on 256 test samples
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(int8_onnx), sess_options=opts, providers=["CPUExecutionProvider"])
    ort_probs = []
    t_lat = time.monotonic()
    for txt in sample_texts:
        enc = tokenizer(txt, truncation=True, max_length=MAX_LEN, return_tensors="np")
        p_out, _ = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)})
        ort_probs.append(float(p_out[0]))
    cpu_ms = (time.monotonic() - t_lat) * 1000.0 / max(1, len(sample_texts))
    ort_probs = np.array(ort_probs, dtype=np.float32)
    parity = {
        "mae": round(float(np.mean(np.abs(pt_probs - ort_probs))), 5),
        "max_abs_diff": round(float(np.max(np.abs(pt_probs - ort_probs))), 5),
        "cpu_1thread_ms_per_comment": round(cpu_ms, 2),
    }

    # Save tokenizer assets & manifest
    tok_dir = out_dir / "tok_tmp"
    tokenizer.save_pretrained(tok_dir)
    for fn in ("vocab.txt", "tokenizer_config.json", "special_tokens_map.json"):
        src = tok_dir / fn
        if src.is_file():
            shutil.copy2(src, out_dir / fn)
    shutil.rmtree(tok_dir, ignore_errors=True)

    onnx_sha = sha256_file(int8_onnx)
    manifest = {
        "name": "MeguvizCAC",
        "version": args.version,
        "license": "Apache-2.0",
        "base_model": args.backbone,
        "quant_method": "dynamic_int8",
        "max_len": MAX_LEN,
        "head_chars": HEAD_CHARS,
        "tail_chars": TAIL_CHARS,
        "temperature": round(best_temp, 6),
        "thresholds": {"f05": round(t_f05, 4), "fpr05": round(t_fpr05, 4), "score70": 0.70},
        "onnx_file": int8_onnx.name,
        "onnx_bytes": int8_onnx.stat().st_size,
        "onnx_sha256": onnx_sha,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = {
        "version": args.version,
        "backbone": args.backbone,
        "gpu_name": gpu_name,
        "best_synth_cap": best_cap,
        "ablation": ablation_results,
        "temperature": round(best_temp, 4),
        "thresholds": {"val_f05": round(t_f05, 4), "val_fpr05": round(t_fpr05, 4)},
        "val_bili_gold": eval_val_gold,
        "val_dsh_gold": eval_val_dsh_gold,
        "test_bili_gold": eval_test_gold,
        "test_dsh_gold": eval_test_dsh_gold,
        "test_all_bili_distilled": eval_test_bili_all,
        "test_all_dsh_distilled": eval_test_dsh_all,
        "probe_u001_u017": probe_metrics,
        "onnx_int8": {**parity, "bytes": int8_onnx.stat().st_size, "sha256": onnx_sha},
        "qwen_rows_used": len(qwen_map) if qwen_map else 0,
        "literary_pairs_used": len(pairs),
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    metrics_file = out_dir / f"metrics-{args.version}.json"
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    zip_path = out_dir / f"meguvizcac-{args.version}-int8.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
        for fn in (int8_onnx.name, "manifest.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json", metrics_file.name):
            fp = out_dir / fn
            if fp.is_file():
                zf.write(fp, fn)

    print(f"=== MeguvizCAC {args.version} COMPLETE ({metrics['elapsed_seconds']}s) ===", flush=True)
    print(json.dumps({
        "test_bili_gold": metrics["test_bili_gold"],
        "test_dsh_gold": metrics["test_dsh_gold"],
        "probe_u001_u017": {k: v for k, v in probe_metrics.items() if k != "details"},
        "onnx_int8": metrics["onnx_int8"],
    }, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--qwen-data", type=Path, default=None)
    parser.add_argument("--pairs-data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("./out_student"))
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--synth-cap", type=int, default=5000)
    parser.add_argument("--ablate-synth", action="store_true", default=True)
    parser.add_argument("--no-ablate-synth", dest="ablate_synth", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_dry(args.data.resolve(), args.qwen_data.resolve() if args.qwen_data else None, args.pairs_data.resolve() if args.pairs_data else None)
        return
    train_and_export(args)


if __name__ == "__main__":
    main()
