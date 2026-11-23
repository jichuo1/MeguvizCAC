"""Colab A100 multi-teacher pseudo-labeling, Hard Human Negative mining, active audit pool export & semantic_intent cache builder for MeguvizCAC v0.4."""

import argparse
from collections import Counter
from contextlib import closing
import gzip
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time

import numpy as np

from .train_student import CODE_LIST, CODE_TO_IDX, PRAG_TAGS, build_train_records, load_qwen_fused, load_rescored_corpus
from .v03_tokenizer import ByteFallbackWordPieceTokenizer, MAX_LEN, prepare_text


REPO_ROOT = Path(__file__).resolve().parent.parent


def _open_read(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def load_new_real_rows(
    real_jsonl: Path | None = None,
    real_db: Path | None = None,
    limit: int = 0,
) -> list[dict]:
    """Load non-legacy real Bilibili rows from exported .jsonl.gz or read-only WAL sqlite."""
    rows: list[dict] = []
    if real_jsonl is not None and real_jsonl.is_file():
        with _open_read(real_jsonl) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("source_cohort") == "legacy_200k":
                    continue
                rows.append(
                    {
                        "sample_id": r["sample_id"],
                        "video_id": r.get("video_id", ""),
                        "partition": r.get("partition", "未分类"),
                        "era_bucket": r.get("era_bucket", "mid_llm"),
                        "comment_level": r.get("comment_level", "top"),
                        "text": r["text"],
                    }
                )
                if limit > 0 and len(rows) >= limit:
                    break
        return rows

    if real_db is not None and real_db.is_file():
        with closing(sqlite3.connect(f"file:{real_db.as_posix()}?mode=ro", uri=True, timeout=30)) as con:
            con.execute("PRAGMA busy_timeout=30000")
            q = "SELECT sample_id, video_id, partition, era_bucket, comment_level, json_extract(metadata_json, '$.text') FROM comments"
            if limit > 0:
                q += f" LIMIT {int(limit)}"
            for sid, vid, part, era, lvl, txt in con.execute(q):
                if txt:
                    rows.append(
                        {
                            "sample_id": sid,
                            "video_id": vid,
                            "partition": part,
                            "era_bucket": era,
                            "comment_level": lvl,
                            "text": txt,
                        }
                    )
        return rows

    raise FileNotFoundError("Either --real-jsonl or --real-db must exist.")


def collect_intent_texts(qwen_distill_path: Path | None, qwen_fused_path: Path | None) -> tuple[dict[str, str], list[str]]:
    """Collect unique Qwen semantic_intent strings and sample_id -> intent mapping (split=='train' only)."""
    sid_to_intent: dict[str, str] = {}
    for p in (qwen_distill_path, qwen_fused_path):
        if p is None or not p.is_file():
            continue
        with _open_read(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("split") not in (None, "train"):
                    continue
                intent = str(r.get("semantic_intent") or "").strip()
                if intent:
                    sid_to_intent[r["sample_id"]] = intent
    uniq_intents = sorted(set(sid_to_intent.values()))
    return sid_to_intent, uniq_intents


def build_intent_embedding_cache(
    qwen_distill_path: Path | None,
    qwen_fused_path: Path | None,
    out_npz: Path,
    encoder_model: str = "BAAI/bge-small-zh-v1.5",
    batch_size: int = 256,
) -> dict:
    """Encode unique Qwen semantic_intent strings with frozen BGE encoder into float16 normalized NPZ."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    sid_to_intent, uniq_intents = collect_intent_texts(qwen_distill_path, qwen_fused_path)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    if not uniq_intents:
        np.savez_compressed(
            out_npz,
            vectors=np.zeros((0, 512), dtype=np.float16),
            sample_ids=np.array([], dtype=object),
            intent_indices=np.zeros((0,), dtype=np.int32),
        )
        return {"unique_intents": 0, "mapped_samples": 0, "path": str(out_npz)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(encoder_model)
    mdl = AutoModel.from_pretrained(encoder_model).to(device).eval()

    vecs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(uniq_intents), batch_size):
            batch = uniq_intents[start : start + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=48, return_tensors="pt").to(device)
            out = mdl(**enc).last_hidden_state[:, 0, :]
            out = torch.nn.functional.normalize(out.float(), p=2, dim=-1)
            vecs.append(out.cpu().numpy().astype(np.float16))

    mat = np.concatenate(vecs, axis=0) if vecs else np.zeros((0, 512), dtype=np.float16)
    intent_to_idx = {s: i for i, s in enumerate(uniq_intents)}
    sids = sorted(sid_to_intent.keys())
    idxs = np.array([intent_to_idx[sid_to_intent[s]] for s in sids], dtype=np.int32)

    np.savez_compressed(
        out_npz,
        vectors=mat,
        sample_ids=np.array(sids, dtype=str),
        intent_indices=idxs,
    )
    return {
        "encoder_model": encoder_model,
        "unique_intents": len(uniq_intents),
        "mapped_samples": len(sids),
        "dim": int(mat.shape[1]),
        "bytes": int(out_npz.stat().st_size),
        "path": str(out_npz),
    }


def train_and_score_12l_teacher(
    stage1_path: Path,
    qwen_fused_path: Path | None,
    synth_path: Path | None,
    new_real_rows: list[dict],
    teacher_backbone: str = "hfl/chinese-macbert-base",
    epochs: int = 2,
    batch_size: int = 256,
    max_synth_train: int = 40000,
    seed: int = 20261124,
) -> tuple[np.ndarray, dict]:
    """Train a 12-layer BF16 teacher on GPU over Stage1+Qwen+pre_llm+250k synth, then score all new_real_rows."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("--train-12l-teacher requires CUDA GPU.")

    t0 = time.monotonic()
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True

    stage1_rows = load_rescored_corpus(stage1_path)
    qwen_map = load_qwen_fused(qwen_fused_path) if qwen_fused_path else None
    forbidden = {prepare_text(r["text"]) for r in stage1_rows if r["split"] != "train"}
    train_items = build_train_records(stage1_rows, synth_cap=8000, qwen_map=qwen_map, seed=seed)

    rng = random.Random(seed)
    pre_llm_rows = [r for r in new_real_rows if r["era_bucket"] == "pre_llm" and prepare_text(r["text"]) not in forbidden]
    for r in pre_llm_rows[:40000]:
        train_items.append(
            {
                "sample_id": r["sample_id"],
                "text": prepare_text(r["text"]),
                "y_score": 0.03,
                "code_idx": CODE_TO_IDX["H"],
                "weight": 0.65,
                "prag_vec": [-1.0] * len(PRAG_TAGS),
            }
        )

    if synth_path is not None and synth_path.is_file():
        s_cands: list[dict] = []
        with _open_read(synth_path) as f:
            for line in f:
                if not line.strip():
                    continue
                sr = json.loads(line)
                pt = prepare_text(sr["text"])
                if pt in forbidden:
                    continue
                s_cands.append({"sample_id": sr["sample_id"], "text": pt})
        rng.shuffle(s_cands)
        for sr in s_cands[:max_synth_train]:
            train_items.append(
                {
                    "sample_id": sr["sample_id"],
                    "text": sr["text"],
                    "y_score": 0.94,
                    "code_idx": CODE_TO_IDX["A"],
                    "weight": 0.45,
                    "prag_vec": [-1.0] * len(PRAG_TAGS),
                }
            )

    tok = AutoTokenizer.from_pretrained(teacher_backbone)
    enc_train = tok([x["text"] for x in train_items], truncation=True, max_length=MAX_LEN)
    for item, ids in zip(train_items, enc_train["input_ids"]):
        item["ids"] = ids

    class TeacherDS(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    def collate(batch):
        max_s = max(len(x["ids"]) for x in batch)
        ids = torch.full((len(batch), max_s), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(batch), max_s), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["ids"])
            ids[i, :n] = torch.tensor(x["ids"], dtype=torch.long)
            mask[i, :n] = 1
        return (
            ids,
            mask,
            torch.tensor([x["y_score"] for x in batch], dtype=torch.float32),
            torch.tensor([x["weight"] for x in batch], dtype=torch.float32),
        )

    class DualTeacher12L(nn.Module):
        def __init__(self, backbone_id: str):
            super().__init__()
            self.enc = AutoModel.from_pretrained(backbone_id)
            h = self.enc.config.hidden_size
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h * 2, 256), nn.GELU(), nn.Linear(256, 1))

        def forward(self, input_ids, attention_mask):
            out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mf = attention_mask.to(out.dtype).unsqueeze(-1)
            mp = (out * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
            feat = torch.cat([out[:, 0, :], mp], dim=-1)
            return self.head(feat).reshape(-1)

    mdl = DualTeacher12L(teacher_backbone).to(device)
    loader = DataLoader(TeacherDS(train_items), batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=2.5e-5, weight_decay=0.01)
    tot = max(1, len(loader) * epochs)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(tot * 0.08)), tot)
    bce_none = nn.BCEWithLogitsLoss(reduction="none")

    for ep in range(epochs):
        mdl.train()
        for ids, mask, y, w in loader:
            ids, mask, y, w = ids.to(device), mask.to(device), y.to(device), w.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = mdl(ids, mask)
                loss = (bce_none(logits, y) * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()
            sched.step()
        print(f"[12L Teacher ep={ep + 1}/{epochs}] done elapsed={time.monotonic() - t0:.1f}s", flush=True)

    # Validate 12L teacher on Stage-1 val_gold
    val_gold = [r for r in stage1_rows if r["split"] == "validation" and r["score_source"] == "hydrogen_xhigh_direct"]
    mdl.eval()

    def score_texts(texts: list[str], bs: int = 512) -> np.ndarray:
        probs = np.empty(len(texts), dtype=np.float32)
        with torch.no_grad():
            for st in range(0, len(texts), bs):
                chunk = texts[st : st + bs]
                enc = tok([prepare_text(t) for t in chunk], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
                ids = enc["input_ids"].to(device, non_blocking=True)
                mask = enc["attention_mask"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    p = torch.sigmoid(mdl(ids, mask))
                probs[st : st + len(chunk)] = p.float().cpu().numpy()
        return probs

    val_p = score_texts([r["text"] for r in val_gold], bs=512)
    val_auc = float(roc_auc_score([float(r["score"]) >= 70.0 for r in val_gold], val_p))
    print(f"[12L Teacher] val_bili_gold AUROC={val_auc:.4f}", flush=True)

    real_probs = score_texts([r["text"] for r in new_real_rows], bs=512)
    meta = {
        "teacher_backbone": teacher_backbone,
        "train_items": len(train_items),
        "val_bili_gold_auroc": round(val_auc, 4),
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    return real_probs * 100.0, meta


def score_with_onnx_bundles(
    bundle_paths: list[Path],
    texts: list[str],
    threads: int = 8,
    max_batch: int = 128,
) -> list[tuple[str, np.ndarray]]:
    """Run existing ONNX student/teacher bundles (e.g., v0.3 audit mode 9L + v0.2) to compute multi-model agreement."""
    from .v04_runtime import MeguvizV04Engine

    outputs: list[tuple[str, np.ndarray]] = []
    for bp in bundle_paths:
        t0 = time.monotonic()
        eng = MeguvizV04Engine(bp, threads=threads, mode="audit")
        preds = eng.predict_batch(texts, mode="audit", max_batch=max_batch)
        arr = np.array([p["prob"] * 100.0 for p in preds], dtype=np.float32)
        outputs.append((eng.manifest.get("version", bp.stem), arr))
        print(f"[ONNX Teacher {eng.manifest.get('version', bp.stem)}] scored {len(texts):,} rows in {time.monotonic() - t0:.1f}s", flush=True)
    return outputs


def run_dry(args) -> None:
    real_rows = load_new_real_rows(args.real_jsonl, args.real_db, limit=args.limit)
    eras = Counter(r["era_bucket"] for r in real_rows)
    sid_to_intent, uniq_intents = collect_intent_texts(args.qwen_distill, args.qwen_fused)
    print(
        json.dumps(
            {
                "dry_run": "PASS",
                "new_real_rows": len(real_rows),
                "by_era_bucket": dict(eras),
                "pre_llm_human_priors": eras.get("pre_llm", 0),
                "needs_pseudo_score": len(real_rows) - eras.get("pre_llm", 0),
                "qwen_intent_samples": len(sid_to_intent),
                "unique_qwen_intents": len(uniq_intents),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-jsonl", type=Path, default=None)
    parser.add_argument("--real-db", type=Path, default=REPO_ROOT / "v04-corpus" / "bili-live-1m.sqlite3")
    parser.add_argument("--stage1-data", type=Path, default=REPO_ROOT / "relabel-hydrogen" / "bili-hydrogen-rescored-350387.jsonl.gz")
    parser.add_argument("--qwen-fused", type=Path, default=REPO_ROOT / "stage2-qwen" / "stage2-fused-train.jsonl.gz")
    parser.add_argument("--qwen-distill", type=Path, default=REPO_ROOT / "stage2-qwen" / "qwen38_27b_distill.jsonl")
    parser.add_argument("--synth-path", type=Path, default=REPO_ROOT / "v04-corpus" / "synthetic-250k" / "bili-v04-synthetic-250k.jsonl.gz")
    parser.add_argument("--onnx-bundle", type=Path, action="append", default=[])
    parser.add_argument("--train-12l-teacher", action="store_true")
    parser.add_argument("--teacher-backbone", default="hfl/chinese-macbert-base")
    parser.add_argument("--teacher-epochs", type=int, default=2)
    parser.add_argument("--onnx-threads", type=int, default=min(12, os.cpu_count() or 4))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-scores", type=Path, default=REPO_ROOT / "v04-corpus" / "v04-teacher-scores.jsonl.gz")
    parser.add_argument("--active-pool-out", type=Path, default=REPO_ROOT / "v04-corpus" / "v04-active-audit-pool.jsonl")
    parser.add_argument("--active-pool-cap", type=int, default=15000)
    parser.add_argument("--build-intent-cache", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_dry(args)
        return

    summary: dict = {}
    if args.build_intent_cache is not None:
        summary["intent_cache"] = build_intent_embedding_cache(
            args.qwen_distill,
            args.qwen_fused,
            args.build_intent_cache,
        )

    real_rows = load_new_real_rows(args.real_jsonl, args.real_db, limit=args.limit)
    texts = [r["text"] for r in real_rows]

    score_tracks: list[tuple[str, np.ndarray]] = []
    if args.train_12l_teacher:
        sc12, meta12 = train_and_score_12l_teacher(
            args.stage1_data,
            args.qwen_fused,
            args.synth_path,
            real_rows,
            teacher_backbone=args.teacher_backbone,
            epochs=args.teacher_epochs,
        )
        score_tracks.append(("teacher_12l", sc12))
        summary["teacher_12l"] = meta12

    if args.onnx_bundle:
        score_tracks.extend(score_with_onnx_bundles(args.onnx_bundle, texts, threads=args.onnx_threads))

    if not score_tracks and args.build_intent_cache is None:
        raise ValueError("Specify --train-12l-teacher, --onnx-bundle, or --build-intent-cache (or --dry-run).")

    if score_tracks:
        mat = np.stack([arr for _, arr in score_tracks], axis=0)  # [M, N]
        mean_raw = mat.mean(axis=0)
        std_raw = mat.std(axis=0) if mat.shape[0] > 1 else np.zeros(len(real_rows), dtype=np.float32)

        args.out_scores.parent.mkdir(parents=True, exist_ok=True)
        staging = args.out_scores.with_suffix(".writing")
        hard_human_cnt = 0
        cands_active: list[tuple[float, dict]] = []

        with gzip.open(staging, "wt", encoding="utf-8", newline="\n", compresslevel=5) as f:
            for i, r in enumerate(real_rows):
                era = r["era_bucket"]
                raw_s = float(mean_raw[i])
                disagree = float(std_raw[i])
                if era == "pre_llm":
                    is_hard_neg = raw_s >= 55.0
                    if is_hard_neg:
                        hard_human_cnt += 1
                    t_score = 3.0
                    t_unc = 2.0
                    src = "pre_llm_human_prior"
                else:
                    is_hard_neg = False
                    t_score = raw_s
                    t_unc = max(3.0, disagree * 1.5)
                    src = "multi_teacher_ensemble"

                rec = {
                    "sample_id": r["sample_id"],
                    "era_bucket": era,
                    "partition": r["partition"],
                    "teacher_score": round(t_score, 2),
                    "raw_teacher_score": round(raw_s, 2),
                    "teacher_uncertainty": round(t_unc, 2),
                    "teacher_disagreement": round(disagree, 2),
                    "hard_human_neg": is_hard_neg,
                    "label_source": src,
                }
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

                # Rank high-leverage active-audit candidates (suspected 2026 AI, boundary disagreement, hard pre_llm FN/FP)
                prio = 0.0
                if era == "current_2026" and raw_s >= 50.0:
                    prio += 40.0 + raw_s * 0.4
                if era == "pre_llm" and is_hard_neg:
                    prio += 35.0 + raw_s * 0.3
                if 32.0 <= raw_s <= 72.0:
                    prio += 25.0 + disagree
                if prio > 0.0:
                    cands_active.append((prio, {**r, **rec}))

        staging.replace(args.out_scores)
        cands_active.sort(key=lambda kv: kv[0], reverse=True)
        args.active_pool_out.parent.mkdir(parents=True, exist_ok=True)
        with args.active_pool_out.open("w", encoding="utf-8", newline="\n") as pf:
            for prio, item in cands_active[: args.active_pool_cap]:
                item["active_priority"] = round(prio, 2)
                pf.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

        summary["scores"] = {
            "total_scored": len(real_rows),
            "tracks": [name for name, _ in score_tracks],
            "hard_human_negatives_in_pre_llm": hard_human_cnt,
            "active_pool_exported": min(len(cands_active), args.active_pool_cap),
            "out_scores": str(args.out_scores),
            "active_pool_out": str(args.active_pool_out),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
