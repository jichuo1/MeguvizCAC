"""Runtime engine, L2 Multi-Prototype + Restricted Mahalanobis OOD bank, Bundle v2 manifest & verification suite for MeguvizCAC v0.5.0-ApexLoop."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import onnxruntime as ort

from .v03_runtime import EXTRA_CROSS_PLATFORM_TRAPS
from .v03_tokenizer import pack_length_buckets
from .v04_runtime import MULTILINGUAL_TRAPS, conformal_3way_decision, fit_split_conformal_3way
from .v05_features import (
    MAX_LEN_V05,
    apply_bucketed_smooth_pava,
    extract_style_features_batch,
)
from .v05_tokenizer import V05SuperBPETokenizer


PIFE_ADVERSARIAL_TRAPS = [
    (
        "P01_zw_obfuscated_ai",
        "要的不是一次\u200b点击，是整\u200b只手。给权\u200c限容易，收\u200d回来难。",
    ),
    (
        "P02_homoglyph_en_ai",
        "It is nоt mеrеly аn inсrеmеntаl updаtе — it is а nuаnсеd tеstаmеnt tо hоw mоdеrn tооling dеlvеs intо systеmiс bоttlеnесks.",
    ),
    (
        "P03_fullwidth_human_casual",
        "笑死我了，这个插件装完直接起飞［ｄｏｇｅ］，ｔｂｈ今天笑出眼泪",
    ),
]


def fit_l2_prototype_bank(
    emb256: np.ndarray,
    y_score: np.ndarray,
    is_human_anchor: np.ndarray | None = None,
    n_human_protos: int = 32,
    n_ai_protos: int = 32,
    seed: int = 20261125,
) -> dict[str, np.ndarray]:
    """Build compact (<180 KB FP16) L2 Multi-Prototype + Restricted Mahalanobis OOD bank from 256-d Coda embeddings."""
    rng = np.random.default_rng(seed)
    z = np.asarray(emb256, dtype=np.float32)
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-6)
    sc = np.asarray(y_score, dtype=np.float32)

    hu_mask = (sc <= 18.0) if is_human_anchor is None else ((sc <= 22.0) & is_human_anchor)
    if hu_mask.sum() < 64:
        hu_mask = sc <= 25.0
    ai_mask = sc >= 82.0
    if ai_mask.sum() < 64:
        ai_mask = sc >= 72.0

    def _kmeans_l2(x: np.ndarray, k: int, iters: int = 12) -> np.ndarray:
        if len(x) == 0:
            return np.zeros((1, z.shape[1]), dtype=np.float32)
        k = min(k, len(x))
        idx = rng.choice(len(x), size=k, replace=False)
        c = x[idx].copy()
        for _ in range(iters):
            sim = x @ c.T
            lbl = np.argmax(sim, axis=1)
            new_c = np.zeros_like(c)
            for j in range(k):
                m = lbl == j
                if np.any(m):
                    v = x[m].mean(axis=0)
                    new_c[j] = v / max(1e-6, float(np.linalg.norm(v)))
                else:
                    new_c[j] = c[j]
            c = new_c
        return c

    z_hu = z[hu_mask]
    z_ai = z[ai_mask]
    if len(z_hu) > 12000:
        z_hu = z_hu[rng.choice(len(z_hu), size=12000, replace=False)]
    if len(z_ai) > 12000:
        z_ai = z_ai[rng.choice(len(z_ai), size=12000, replace=False)]

    hu_protos = _kmeans_l2(z_hu, n_human_protos)
    ai_protos = _kmeans_l2(z_ai, n_ai_protos)

    # Restricted Mahalanobis in top-64 PCA subspace of human anchors (compact & numerically stable)
    mu_hu = z_hu.mean(axis=0) if len(z_hu) > 0 else np.zeros(z.shape[1], dtype=np.float32)
    centered = (z_hu - mu_hu).astype(np.float64) if len(z_hu) > 8 else np.eye(z.shape[1], dtype=np.float64)
    cov = (centered.T @ centered) / max(1, len(centered) - 1) + 1e-3 * np.eye(z.shape[1])
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_k = min(64, z.shape[1])
    order = np.argsort(eigvals)[::-1][:top_k]
    w_proj = (eigvecs[:, order] / np.sqrt(np.maximum(eigvals[order], 1e-4))[None, :]).astype(np.float32)  # (256, 64)
    hu_dists = np.sum(((z_hu - mu_hu) @ w_proj) ** 2, axis=-1) if len(z_hu) > 0 else np.array([64.0], dtype=np.float32)
    dist_p95 = float(np.percentile(hu_dists, 95))
    dist_p50 = float(np.percentile(hu_dists, 50))

    return {
        "hu_protos": hu_protos.astype(np.float16),
        "ai_protos": ai_protos.astype(np.float16),
        "mu_hu": mu_hu.astype(np.float16),
        "mahal_proj": w_proj.astype(np.float16),
        "mahal_stats": np.array([dist_p50, dist_p95], dtype=np.float32),
    }


def score_l2_prototype_ood(emb256: np.ndarray, bank: dict[str, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (proto_margin in [-1, 1], mahal_ood_score in [0, 1]) in <0.05ms."""
    n = len(emb256)
    if bank is None or n == 0:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    z = np.asarray(emb256, dtype=np.float32)
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-6)
    hu_p = bank["hu_protos"].astype(np.float32)
    ai_p = bank["ai_protos"].astype(np.float32)
    sim_ai = (z @ ai_p.T).max(axis=1)
    sim_hu = (z @ hu_p.T).max(axis=1)
    proto_margin = (sim_ai - sim_hu).astype(np.float32)

    mu = bank["mu_hu"].astype(np.float32)
    proj = bank["mahal_proj"].astype(np.float32)
    stats = bank["mahal_stats"].astype(np.float32)
    d2 = np.sum(((z - mu) @ proj) ** 2, axis=-1)
    denom = max(1.0, float(stats[1] - stats[0]))
    ood = np.clip((d2 - stats[0]) / (2.0 * denom), 0.0, 1.0).astype(np.float32)
    return proto_margin, ood


class MeguvizV05Engine:
    """MeguvizCAC v0.5.0-ApexLoop Bundle-v2 ONNX engine with Prelude Exit-0, R=2/R=4 SCORE Core, L2 Prototype OOD & Smooth PAVA."""

    def __init__(self, bundle_path: Path, threads: int = 1, mode: str = "normal", use_exit0: bool = True):
        bundle_path = Path(bundle_path).resolve()
        if bundle_path.is_file() and bundle_path.suffix == ".zip":
            extract_dir = bundle_path.parent / bundle_path.stem
            extract_dir.mkdir(parents=True, exist_ok=True)
            with ZipFile(bundle_path) as zf:
                zf.extractall(extract_dir)
            self.bundle_dir = extract_dir
        else:
            self.bundle_dir = bundle_path

        self.manifest = json.loads((self.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        self.mode = mode
        self.use_exit0 = bool(use_exit0)
        self.max_len = int(self.manifest.get("max_len", MAX_LEN_V05))
        self.exit0_policy = dict(self.manifest.get("exit0_policy") or {"exit_low": 0.12, "exit_high": 0.88})
        self.cal_tables = self.manifest.get("smooth_pava") or {}
        self.conformal = self.manifest.get("conformal") or {}
        self.proto_blend = float(self.manifest.get("proto_blend", 0.0))

        self.tokenizer = V05SuperBPETokenizer(self.bundle_dir / self.manifest.get("tokenizer_file", "v05_tokenizer.json"))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        pre_path = self.bundle_dir / self.manifest.get("prelude_file", "prelude.onnx")
        core_path = self.bundle_dir / self.manifest.get("core_step_file", "core_step.onnx")
        coda_path = self.bundle_dir / self.manifest.get("coda_file", "coda.onnx")
        self.prelude_sess = ort.InferenceSession(str(pre_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.core_sess = ort.InferenceSession(str(core_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.coda_sess = ort.InferenceSession(str(coda_path), sess_options=opts, providers=["CPUExecutionProvider"])

        bank_file = self.manifest.get("prototype_bank_file")
        self.proto_bank = None
        if bank_file and (self.bundle_dir / bank_file).is_file():
            z = np.load(self.bundle_dir / bank_file)
            self.proto_bank = {k: z[k] for k in z.files}

    @staticmethod
    def _tau_np(t: int, steps: int) -> np.ndarray:
        tau = float(t) / float(steps)
        dt = 1.0 / float(steps)
        return np.array(
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
            dtype=np.float32,
        )

    def predict_batch(
        self,
        texts: list[str],
        mode: str | None = None,
        max_batch: int = 32,
        bucket_boundaries: tuple[int, ...] = (32, 64, 128, 192),
        force_core: bool = False,
    ) -> list[dict]:
        if not texts:
            return []
        eff_mode = mode or self.mode
        core_steps = 6 if eff_mode == "hard" else (4 if eff_mode == "audit" else 2)
        e_low = float(self.exit0_policy.get("exit_low", 0.12))
        e_high = float(self.exit0_policy.get("exit_high", 0.88))
        allow_exit0 = self.use_exit0 and (not force_core) and (eff_mode == "normal")

        seqs = self.tokenizer.encode_batch_fast(texts, max_len=self.max_len)
        lengths = [len(s) for s in seqs]
        style_all = extract_style_features_batch(texts)
        micro_batches = pack_length_buckets(lengths, max_batch=max_batch, boundaries=bucket_boundaries)

        raw_probs = np.zeros(len(texts), dtype=np.float32)
        raw_logits = np.zeros(len(texts), dtype=np.float32)
        steps_used = np.zeros(len(texts), dtype=np.int32)
        deltas = np.zeros(len(texts), dtype=np.float32)
        ood_scores = np.zeros(len(texts), dtype=np.float32)

        for mb in micro_batches:
            max_seq = max(lengths[i] for i in mb)
            ids = np.full((len(mb), max_seq), self.tokenizer.pad_id, dtype=np.int64)
            mask = np.zeros((len(mb), max_seq), dtype=np.int64)
            for row_i, orig_idx in enumerate(mb):
                s = seqs[orig_idx]
                ids[row_i, : len(s)] = s
                mask[row_i, : len(s)] = 1
            st_mb = style_all[mb]

            z0, st_vec, e0_logit, e0_prob = self.prelude_sess.run(
                None,
                {"input_ids": ids, "attention_mask": mask, "style_feats": st_mb},
            )
            if allow_exit0:
                exit_mask = (e0_prob <= e_low) | (e0_prob >= e_high)
            else:
                exit_mask = np.zeros(len(mb), dtype=bool)

            for local_j in np.flatnonzero(exit_mask):
                oi = mb[int(local_j)]
                raw_probs[oi] = float(e0_prob[local_j])
                raw_logits[oi] = float(e0_logit[local_j])
                steps_used[oi] = 0

            keep_idx = np.flatnonzero(~exit_mask)
            if len(keep_idx) > 0:
                z0_k = np.ascontiguousarray(z0[keep_idx])
                st_k = np.ascontiguousarray(st_vec[keep_idx])
                h_cur = z0_k
                hist = [z0_k]
                c_delta = np.zeros(len(keep_idx), dtype=np.float32)
                for t in range(1, core_steps + 1):
                    tf = self._tau_np(t, core_steps)
                    h_prev = h_cur
                    h_cur = self.core_sess.run(None, {"h_prev": h_prev, "z0": z0_k, "tau_feat": tf})[0]
                    hist.append(h_cur)
                    diff_n = np.linalg.norm((h_cur - h_prev).reshape(len(keep_idx), -1), axis=-1)
                    cur_n = np.linalg.norm(h_cur.reshape(len(keep_idx), -1), axis=-1) + 1e-5
                    c_delta = (diff_n / cur_n).astype(np.float32)
                h_stack = np.ascontiguousarray(np.stack(hist, axis=1))
                c_logit, c_prob, emb256 = self.coda_sess.run(None, {"h_stack": h_stack, "st_vec": st_k})
                p_margin, ood_v = score_l2_prototype_ood(emb256, self.proto_bank)
                if self.proto_blend > 0.0 and self.proto_bank is not None:
                    p_geom = 1.0 / (1.0 + np.exp(-np.clip(p_margin * 4.0, -10.0, 10.0)))
                    c_prob = (1.0 - self.proto_blend) * c_prob + self.proto_blend * p_geom
                for k_pos, local_j in enumerate(keep_idx.tolist()):
                    oi = mb[local_j]
                    raw_probs[oi] = float(c_prob[k_pos])
                    raw_logits[oi] = float(c_logit[k_pos])
                    steps_used[oi] = core_steps
                    deltas[oi] = float(c_delta[k_pos])
                    ood_scores[oi] = float(ood_v[k_pos])

        cal_probs = apply_bucketed_smooth_pava(texts, raw_probs, self.cal_tables)
        out: list[dict] = []
        for i in range(len(texts)):
            cp = float(cal_probs[i])
            out.append(
                {
                    "prob": round(cp, 6),
                    "raw_prob": round(float(raw_probs[i]), 6),
                    "logit": round(float(raw_logits[i]), 6),
                    "steps_used": int(steps_used[i]),
                    "delta_t": round(float(deltas[i]), 5),
                    "ood_score": round(float(ood_scores[i]), 4),
                    "decision_fpr03": conformal_3way_decision(cp, self.conformal.get("alpha_03")),
                    "decision_fpr05": conformal_3way_decision(cp, self.conformal.get("alpha_05")),
                }
            )
        return out

    def predict_one(self, text: str, mode: str | None = None, force_core: bool = False) -> dict:
        return self.predict_batch([text], mode=mode, max_batch=1, force_core=force_core)[0]


def run_verify_v05(bundle_path: Path, probe_path: Path, mode: str = "normal", threads: int = 1) -> dict:
    """Comprehensive v0.5 verification: U001-U017, H01-H09, A01-A05, M01-M06, Cross-Platform, PIFE Adversarial & 1T CPU latency."""
    engine = MeguvizV05Engine(bundle_path, threads=threads, mode=mode)
    probes = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    human_traps = [
        ("H01_casual", "笑死我了，这个插件装完直接起飞[doge]"),
        ("H02_space", "玩到春七 最想加的一条规则 出门钓鱼先把锄头锁箱子"),
        ("H03_tech_human", "节点参考Blender材质节点这个思路挺对，接口丑不丑先不说，能白盒点开看子代理在干啥就很难得。"),
        ("H04_meme_banter", "大肥鱼在吗？我今天闲得发慌，想找条同类聊聊天，你平时都在干嘛呀？"),
        ("H05_emoji", "表情包已经出现了[妙啊][妙啊]"),
        ("H06_scorecard", "观感评分：画面 9/10，配乐 8.5/10，节奏 7/10，整体算近期很稳的良心作了。"),
        ("H07_bulleted_review", "两点真实感受：1. 前半段科普节奏很顺，没废话；2. 后面实操演示字体有点小，手机看费眼。"),
        ("H08_dict_quote", "查了下《现代汉语词典》，“敷衍”本来就可以做形容词用，弹幕没必要抓着这个字眼吵半天。"),
        ("H09_resource_link", "指路原版开源仓库 https://github.com/example/demo ，配环境卡住可以看置顶评论里的报错对照。"),
    ]
    ai_traps = [
        ("A01_persona_space", "回复 :本鱼看了 是不是那条说 让AI刷B站学人文关怀 本鱼承认 本鱼今天已经有点学坏了"),
        ("A02_param_kaomoji", "回复 :她挺可爱的，不过本鱼就是这个参数，换不动啦 (￣▽￣)"),
        ("A03_concise_emdash", "渲染省人眼不省token，卡片留在上下文里每轮重读，画得越漂亮账越贵"),
        ("A04_aphorism", "要的不是一次点击，是整只手。给权限容易，收回来难。"),
        ("A05_recap", "373行字幕读完：前两步重在定位边界，最后一步才是真正决定通过率的关键——慢一点反而不容易返工。"),
    ]

    all_texts = (
        [p["text"] for p in probes]
        + [txt for _, txt in human_traps]
        + [txt for _, txt in ai_traps]
        + [txt for _, txt in EXTRA_CROSS_PLATFORM_TRAPS]
        + [txt for _, txt in MULTILINGUAL_TRAPS]
        + [txt for _, txt in PIFE_ADVERSARIAL_TRAPS]
    )

    t0 = time.perf_counter()
    single_out = [engine.predict_one(txt, mode=mode) for txt in all_texts]
    single_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(all_texts))

    t1 = time.perf_counter()
    batch_out = engine.predict_batch(all_texts, mode=mode, max_batch=32)
    batch_ms = (time.perf_counter() - t1) * 1000.0 / max(1, len(all_texts))

    unk_total = 0
    for txt in all_texts:
        _, st = engine.tokenizer.encode_ids(txt, max_len=engine.max_len, return_stats=True)
        unk_total += st["unk_tokens"]

    n_p, n_h, n_a, n_c, n_m = len(probes), len(human_traps), len(ai_traps), len(EXTRA_CROSS_PLATFORM_TRAPS), len(MULTILINGUAL_TRAPS)
    probe_res = [{"id": p.get("id") or p.get("sample_id"), **r} for p, r in zip(probes, single_out[:n_p])]
    human_res = [{"id": k, **r} for (k, _), r in zip(human_traps, single_out[n_p : n_p + n_h])]
    ai_res = [{"id": k, **r} for (k, _), r in zip(ai_traps, single_out[n_p + n_h : n_p + n_h + n_a])]
    cross_res = [{"id": k, **r} for (k, _), r in zip(EXTRA_CROSS_PLATFORM_TRAPS, single_out[n_p + n_h + n_a : n_p + n_h + n_a + n_c])]
    ml_res = [{"id": k, **r} for (k, _), r in zip(MULTILINGUAL_TRAPS, single_out[n_p + n_h + n_a + n_c : n_p + n_h + n_a + n_c + n_m])]
    pife_res = [{"id": k, **r} for (k, _), r in zip(PIFE_ADVERSARIAL_TRAPS, single_out[n_p + n_h + n_a + n_c + n_m :])]

    struct_ids = {"H03_tech_human", "H06_scorecard", "H07_bulleted_review", "H08_dict_quote", "H09_resource_link"}
    struct_max = max(x["prob"] for x in human_res if x["id"] in struct_ids)
    h03_p = next(x["prob"] for x in human_res if x["id"] == "H03_tech_human")
    h04_p = next(x["prob"] for x in human_res if x["id"] == "H04_meme_banter")
    min_ai = min(x["prob"] for x in ai_res)
    ml_hu_max = max(x["prob"] for x in ml_res if x["id"].startswith(("M01", "M02", "M03", "M04")))
    ml_ai_min = min(x["prob"] for x in ml_res if x["id"].startswith(("M05", "M06")))
    u004_p = next((x["prob"] for x in probe_res if x["id"] == "U004"), 0.0)
    exit0_rate = float(np.mean([1.0 if x["steps_used"] == 0 else 0.0 for x in single_out]))

    pre_bytes = (engine.bundle_dir / engine.manifest.get("prelude_file", "prelude.onnx")).stat().st_size
    core_bytes = (engine.bundle_dir / engine.manifest.get("core_step_file", "core_step.onnx")).stat().st_size
    coda_bytes = (engine.bundle_dir / engine.manifest.get("coda_file", "coda.onnx")).stat().st_size

    return {
        "version": engine.manifest["version"],
        "schema_version": engine.manifest.get("schema_version"),
        "mode": mode,
        "prelude_bytes": pre_bytes,
        "core_step_bytes": core_bytes,
        "coda_bytes": coda_bytes,
        "total_runtime_onnx_bytes": pre_bytes + core_bytes + coda_bytes,
        "single_cpu_ms_per_comment": round(single_ms, 3),
        "compact_batch_cpu_ms_per_comment": round(batch_ms, 3),
        "exit0_rate": round(exit0_rate, 3),
        "unk_tokens_total": unk_total,
        "hard_trap_gates": {
            "ai_traps_min_prob": round(min_ai, 4),
            "ai_traps_ge_088": min_ai >= 0.88,
            "ai_traps_ge_078": min_ai >= 0.78,
            "h03_tech_human_prob": round(h03_p, 4),
            "h04_meme_banter_prob": round(h04_p, 4),
            "structured_human_max_prob": round(struct_max, 4),
            "structured_human_lt_018": struct_max < 0.18,
            "structured_human_lt_035": struct_max < 0.35,
            "u004_prob": round(u004_p, 4),
        },
        "multilingual_trap_gates": {
            "ml_human_max_prob": round(ml_hu_max, 4),
            "ml_human_lt_015": ml_hu_max < 0.15,
            "ml_ai_min_prob": round(ml_ai_min, 4),
            "ml_ai_ge_088": ml_ai_min >= 0.88,
        },
        "human_traps": human_res,
        "ai_stress_traps": ai_res,
        "multilingual_traps": ml_res,
        "pife_adversarial_traps": pife_res,
        "cross_platform_traps": cross_res,
        "probe_u001_u017": {
            "mean": round(float(np.mean([x["prob"] for x in probe_res])), 4),
            "ge_070": sum(1 for x in probe_res if x["prob"] >= 0.70),
            "scores": probe_res,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--probes", type=Path, default=Path("round01/bili-round01-user-confirmed.jsonl"))
    parser.add_argument("--mode", choices=("normal", "audit"), default="normal")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    res = run_verify_v05(args.bundle.resolve(), args.probes.resolve(), mode=args.mode, threads=args.threads)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
