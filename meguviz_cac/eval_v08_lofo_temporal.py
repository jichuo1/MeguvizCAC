"""MeguvizCAC v0.8.0-Zenith Standalone LOFO, Temporal Holdout, Extreme-Low-FPR & Mondrian 9-Cell Audit Harness.

Runs strictly zero-leakage evaluation on ANY exported `v0.7.0-OmniVeritas` or `v0.8.0-Zenith` Bundle v4 directory:
  1. Overall `test_bili_gold`, `test_dsh_gold`, `test_all_bili_distilled`, and `probe_u001_u017` scorecards
  2. Extreme-Low-FPR (`TPR@1% / 0.5% / 0.1% / 0.01% FPR`) + 15-bin Adaptive ECE + Brier calibration audit
  3. Per-Model-Family & Per-Cohort Slice Breakdown (simulating Leave-One-Family-Out / generator generalization)
  4. Temporal Era-Shift Audit (`|FPR(2026_modern) - FPR(pre_llm)|` at deployment threshold + `score >= 70`)
  5. 9-Cell Mondrian (`{Short, Med, Long} x {Casual, Structured-Tech, Multilingual}`) subgroup FPR verification
  6. Built-in Zero-PII & Anti-Spoof Trap Stress Suite (`_eval_traps_v07`)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .train_student import compute_eval_metrics
from .train_v04_dualloop import _open_read, compute_high_precision_metrics
from .train_v07_omniveritas import _eval_traps_v07
from .v07_features import extract_style_features_40d_batch
from .v07_omniveritas import MeguvizV07Engine
from .v08_zenith_ops import (
    assign_mondrian_cell_9,
    compute_zenith_calibration_and_lowfpr_metrics,
)
from .v09_chronoloop import MeguvizV09CascadeEngine


def _load_bundle_engine(bundle_dir: Path, threads: int = 2):
    manifest = json.loads((Path(bundle_dir) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") == "meguviz.bundle.v5" or "rung1_onnx" in manifest:
        return MeguvizV09CascadeEngine(bundle_dir, threads=threads), True
    return MeguvizV07Engine(bundle_dir, threads=threads, enable_nano_cascade=True), False


def _infer_family_tag(row: dict[str, Any]) -> str:
    """Derive generator/cohort family tag strictly from metadata (never modifying texts)."""
    for key in ("generator_family", "model_family", "teacher_family", "source_model", "cohort"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return "unknown"


def _eval_subset(engine, rows: list[dict], thresh: float = 0.70, max_batch: int = 32, is_v09: bool = False) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    txts = [str(r["text"]) for r in rows]
    scores = np.array([float(r["score"]) for r in rows], dtype=np.float32)
    preds = engine.predict_batch(txts, max_batch=max_batch, force_full=False) if is_v09 else engine.predict_batch(txts, max_batch=max_batch, force_main=False)
    probs = np.array([x["prob"] for x in preds], dtype=np.float32)
    m_all = {**compute_eval_metrics(scores, probs, thresh, thresh), **compute_high_precision_metrics(scores, probs)}
    m_cal = compute_zenith_calibration_and_lowfpr_metrics(scores, probs)
    exited = np.array([1.0 if (x.get("exit_rung") == 1 if is_v09 else x.get("exited_nano")) else 0.0 for x in preds], dtype=np.float32)
    return {
        "n": len(rows),
        "auroc": m_all.get("auroc"),
        "pr_auc": m_all.get("pr_auc"),
        "best_f05": m_all.get("best_f05"),
        "recall_at_p95": m_all.get("recall_at_p95"),
        "recall_at_p99": m_all.get("recall_at_p99"),
        "early_exit_rate": round(float(np.mean(exited)), 4),
        "early_exit_missed_ai": int(np.sum((exited > 0.5) & (scores >= 70.0))),
        "calibration_and_lowfpr": m_cal,
        "_probs": probs,
        "_scores": scores,
    }


def run_lofo_and_temporal_audit(
    bundle_dir: Path,
    data_path: Path,
    threads: int = 2,
    max_test_all: int = 20000,
    out_json: Path | None = None,
) -> dict[str, Any]:
    t0 = time.monotonic()
    engine, is_v09 = _load_bundle_engine(bundle_dir, threads=threads)
    manifest = engine.manifest
    t_deploy = float((manifest.get("thresholds") or {}).get("f05", 0.70))
    m9_root = manifest.get("selection_safe_mondrian_9cell") or manifest.get("mondrian_9cell") or {}
    m9_cfg = (m9_root.get("alphas") or {}).get("alpha_003") or {}
    m9_cells = m9_cfg.get("cells") or {}

    test_gold: list[dict] = []
    test_dsh_gold: list[dict] = []
    test_all: list[dict] = []
    probe_rows: list[dict] = []
    pre_llm_hu: list[dict] = []
    modern_hu: list[dict] = []
    by_family_pos: dict[str, list[dict]] = defaultdict(list)

    with _open_read(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sp = str(r.get("split", ""))
            co = str(r.get("cohort", ""))
            ss = str(r.get("score_source", ""))
            sc = float(r.get("score", 0.0))
            if co == "user_confirmed":
                probe_rows.append(r)
            if sp != "test":
                continue
            if ss == "hydrogen_xhigh_direct":
                test_gold.append(r)
                if r.get("dsh"):
                    test_dsh_gold.append(r)
            if co == "bili" and len(test_all) < max_test_all:
                test_all.append(r)
            if sc <= 25.0:
                if r.get("era_bucket") == "pre_llm" or co == "pre_llm":
                    if len(pre_llm_hu) < 2000:
                        pre_llm_hu.append(r)
                elif co == "bili" and len(modern_hu) < 2000:
                    modern_hu.append(r)
            elif sc >= 70.0:
                fam = _infer_family_tag(r)
                if len(by_family_pos[fam]) < 400:
                    by_family_pos[fam].append(r)

    res_gold = _eval_subset(engine, test_gold, thresh=t_deploy, is_v09=is_v09)
    res_dsh = _eval_subset(engine, test_dsh_gold, thresh=t_deploy, is_v09=is_v09)
    res_all = _eval_subset(engine, test_all, thresh=t_deploy, is_v09=is_v09)
    res_prb = _eval_subset(engine, probe_rows, thresh=t_deploy, is_v09=is_v09)

    # 1. Temporal Holdout Audit: |FPR(modern_2026) - FPR(pre_llm)| at deployment threshold!
    temporal_audit: dict[str, Any] = {"n_pre_llm": len(pre_llm_hu), "n_modern": len(modern_hu)}
    if pre_llm_hu and modern_hu:
        p_pre = np.array([x["prob"] for x in engine.predict_batch([str(r["text"]) for r in pre_llm_hu])], dtype=np.float32)
        p_mod = np.array([x["prob"] for x in engine.predict_batch([str(r["text"]) for r in modern_hu])], dtype=np.float32)
        fpr_pre = float(np.mean(p_pre >= t_deploy))
        fpr_mod = float(np.mean(p_mod >= t_deploy))
        temporal_audit.update(
            {
                "threshold_used": round(t_deploy, 4),
                "fpr_pre_llm": round(fpr_pre, 5),
                "fpr_modern": round(fpr_mod, 5),
                "abs_fpr_gap": round(abs(fpr_mod - fpr_pre), 5),
                "pass_le_0005": abs(fpr_mod - fpr_pre) <= 0.005,
                "mean_prob_pre_llm": round(float(np.mean(p_pre)), 5),
                "mean_prob_modern": round(float(np.mean(p_mod)), 5),
            }
        )

    # 2. Per-Family / Cohort Recall & AUROC against Shared Clean Modern Humans (LOFO proxy)
    lofo_slices: dict[str, Any] = {}
    if modern_hu and by_family_pos:
        from sklearn.metrics import roc_auc_score

        p_neg = np.array([x["prob"] for x in engine.predict_batch([str(r["text"]) for r in modern_hu[:500]])], dtype=np.float32)
        for fam, pos_rows in sorted(by_family_pos.items()):
            if len(pos_rows) < 10:
                continue
            p_pos = np.array([x["prob"] for x in engine.predict_batch([str(r["text"]) for r in pos_rows])], dtype=np.float32)
            yb = np.concatenate([np.zeros(len(p_neg)), np.ones(len(p_pos))])
            yp = np.concatenate([p_neg, p_pos])
            lofo_slices[fam] = {
                "n_pos": len(pos_rows),
                "auroc_vs_modern_hu": round(float(roc_auc_score(yb, yp)), 4),
                "recall_at_deploy": round(float(np.mean(p_pos >= t_deploy)), 4),
                "mean_prob": round(float(np.mean(p_pos)), 4),
            }

    # 3. 9-Cell Mondrian Subgroup Audit on `test_all` (verifies FPR_g <= alpha per cell on unseen test set!)
    mondrian_test_audit: dict[str, Any] = {}
    if test_all and "_probs" in res_all:
        raws_all = [str(r["text"]) for r in test_all]
        cleans_all, st_all, _ = extract_style_features_40d_batch(raws_all)
        probs_all = res_all["_probs"]
        scores_all = res_all["_scores"]
        cells_all = [assign_mondrian_cell_9(cleans_all[i], st_all[i]) for i in range(len(test_all))]
        for cid in sorted(set(cells_all)):
            cm = np.array([c == cid for c in cells_all], dtype=bool)
            hu_m = cm & (scores_all <= 30.0)
            ai_m = cm & (scores_all >= 70.0)
            t_cell = float((m9_cells.get(cid) or {}).get("t_ai", t_deploy))
            mondrian_test_audit[cid] = {
                "n_hu": int(hu_m.sum()),
                "n_ai": int(ai_m.sum()),
                "t_ai": round(t_cell, 5),
                "test_fpr": round(float(np.mean(probs_all[hu_m] >= t_cell)), 4) if hu_m.any() else None,
                "test_recall": round(float(np.mean(probs_all[ai_m] >= t_cell)), 4) if ai_m.any() else None,
            }

    for d in (res_gold, res_dsh, res_all, res_prb):
        d.pop("_probs", None)
        d.pop("_scores", None)

    report = {
        "bundle_dir": str(bundle_dir),
        "version": manifest.get("version"),
        "schema_version": manifest.get("schema_version"),
        "test_bili_gold": res_gold,
        "test_dsh_gold": res_dsh,
        "test_all_bili_distilled": res_all,
        "probe_u001_u017": res_prb,
        "temporal_era_audit": temporal_audit,
        "lofo_family_slices": lofo_slices,
        "mondrian_9cell_test_audit": mondrian_test_audit,
        "traps": _eval_traps_v07(engine),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, required=True, help="Exported v0.7.0 / v0.8.0 Bundle v4 directory")
    ap.add_argument("--data", type=Path, required=True, help="Corpus jsonl.gz containing frozen test splits")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-test-all", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = run_lofo_and_temporal_audit(args.bundle_dir, args.data, threads=args.threads, max_test_all=args.max_test_all, out_json=args.out)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
