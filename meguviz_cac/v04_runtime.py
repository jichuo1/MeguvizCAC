"""Runtime engine, CharRhythm byte encoder, PAVA isotonic + split conformal 3-way calibration, and ordered W4-Emb/W8-Body export for MeguvizCAC v0.4-DualLoop."""

import argparse
import json
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import onnxruntime as ort

from .v03_runtime import EXTRA_CROSS_PLATFORM_TRAPS
from .v03_tokenizer import BASE_VOCAB_SIZE, ByteFallbackWordPieceTokenizer, MAX_LEN, pack_length_buckets, prepare_text


CHAR_MAX_BYTES = 256
CHAR_PAD_ID = 256
CHAR_VOCAB_SIZE = 257


def encode_char_bytes(text: str, max_bytes: int = CHAR_MAX_BYTES) -> list[int]:
    """Encode NFKC-normalized text as raw UTF-8 byte IDs (0..255), preserving spaces and kaomoji rhythm."""
    norm = prepare_text(text)
    raw = norm.encode("utf-8", errors="replace")
    if len(raw) > max_bytes:
        tail_keep = min(64, max_bytes // 4)
        head_keep = max(1, max_bytes - tail_keep - 1)
        raw = raw[:head_keep] + b" " + raw[-tail_keep:]
    if not raw:
        return [0x20]
    return list(raw)


def encode_char_bytes_batch(
    texts: list[str],
    max_bytes: int = CHAR_MAX_BYTES,
    min_len: int = 8,
) -> tuple[np.ndarray, list[int]]:
    """Batch-encode raw UTF-8 byte sequences padded with CHAR_PAD_ID=256."""
    if not texts:
        return np.zeros((0, min_len), dtype=np.int64), []
    seqs = [encode_char_bytes(t, max_bytes=max_bytes) for t in texts]
    lengths = [len(s) for s in seqs]
    width = max(min_len, max(lengths))
    arr = np.full((len(seqs), width), CHAR_PAD_ID, dtype=np.int64)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
    return arr, lengths


def fit_pava_isotonic(probs: np.ndarray, y_soft: np.ndarray, max_knots: int = 64) -> dict:
    """Pure-NumPy Pool Adjacent Violators Algorithm (PAVA) monotonic calibration knots."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(y_soft, dtype=np.float64).reshape(-1)
    if len(p) == 0:
        return {"x": [0.0, 1.0], "y": [0.0, 1.0]}

    order = np.argsort(p, kind="mergesort")
    p_s = p[order]
    y_s = y[order]

    # Merge exact ties first so PAVA is uniquely defined regardless of ordering
    uniq_x: list[float] = []
    uniq_y_sum: list[float] = []
    uniq_w: list[float] = []
    for xi, yi in zip(p_s.tolist(), y_s.tolist()):
        if uniq_x and abs(xi - uniq_x[-1]) <= 1e-12:
            uniq_y_sum[-1] += yi
            uniq_w[-1] += 1.0
        else:
            uniq_x.append(xi)
            uniq_y_sum.append(yi)
            uniq_w.append(1.0)

    # PAVA blocks: [x_min, x_max, sum_y, weight]
    blocks: list[list[float]] = []
    for xi, sy, w in zip(uniq_x, uniq_y_sum, uniq_w):
        blocks.append([xi, xi, sy, w])
        while len(blocks) >= 2:
            prev_val = blocks[-2][2] / blocks[-2][3]
            curr_val = blocks[-1][2] / blocks[-1][3]
            if prev_val > curr_val:
                last = blocks.pop()
                blocks[-1][1] = last[1]
                blocks[-1][2] += last[2]
                blocks[-1][3] += last[3]
            else:
                break

    xs: list[float] = []
    ys: list[float] = []
    for x_min, x_max, sy, w in blocks:
        val = float(np.clip(sy / w, 0.0, 1.0))
        xs.append(float(x_min))
        ys.append(val)
        if x_max > x_min + 1e-12:
            xs.append(float(x_max))
            ys.append(val)

    if xs[0] > 0.0:
        xs.insert(0, 0.0)
        ys.insert(0, ys[0])
    if xs[-1] < 1.0:
        xs.append(1.0)
        ys.append(ys[-1])

    if len(xs) > max_knots:
        keep_idx = np.unique(np.round(np.linspace(0, len(xs) - 1, max_knots)).astype(np.int64))
        xs = [xs[int(i)] for i in keep_idx]
        ys = [ys[int(i)] for i in keep_idx]

    return {
        "x": [round(float(v), 6) for v in xs],
        "y": [round(float(v), 6) for v in ys],
    }


def apply_isotonic_knots(probs: np.ndarray | float, knots: dict | None):
    if not knots or not knots.get("x") or not knots.get("y"):
        return probs
    iso = np.interp(probs, knots["x"], knots["y"])
    blend = float(knots.get("raw_blend", 0.0))
    if blend > 0.0:
        return (1.0 - blend) * iso + blend * np.asarray(probs, dtype=np.float64)
    return iso


def fit_split_conformal_3way(
    y_true_score: np.ndarray,
    probs: np.ndarray,
    alphas: tuple[float, ...] = (0.03, 0.05),
    pos_cutoff: float = 70.0,
) -> dict:
    """Calibrate split conformal 3-way thresholds (human / grey_zone / ai) controlling empirical FPR & FNR."""
    scores = np.asarray(y_true_score, dtype=np.float64).reshape(-1)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y_pos = scores >= float(pos_cutoff)
    neg_p = np.sort(p[~y_pos])
    pos_p = np.sort(p[y_pos])
    n_neg = len(neg_p)
    n_pos = len(pos_p)

    out: dict[str, dict] = {}
    for alpha in alphas:
        a = float(alpha)
        key = f"alpha_{int(round(a * 100)):02d}"
        if n_neg > 0:
            q_neg = min(1.0, (1.0 - a) * (n_neg + 1.0) / n_neg)
            idx_neg = max(0, min(n_neg - 1, int(np.ceil(q_neg * n_neg)) - 1))
            t_ai = 1.000001 if q_neg >= 1.0 else float(np.nextafter(neg_p[idx_neg], np.inf))
            emp_fpr = float((neg_p >= t_ai).mean())
            emp_rec = float((pos_p >= t_ai).mean()) if n_pos > 0 else 0.0
        else:
            t_ai, emp_fpr, emp_rec = 0.70, 0.0, 0.0

        if n_pos > 0:
            s_neg_ai = np.sort(1.0 - pos_p)
            q_pos = min(1.0, (1.0 - a) * (n_pos + 1.0) / n_pos)
            idx_pos = max(0, min(n_pos - 1, int(np.ceil(q_pos * n_pos)) - 1))
            t_human = -1e-6 if q_pos >= 1.0 else float(np.nextafter(1.0 - s_neg_ai[idx_pos], -np.inf))
            emp_fnr = float((pos_p <= t_human).mean())
        else:
            t_human, emp_fnr = 0.30, 0.0

        grey_rate = float(((p > t_human) & (p < t_ai)).mean()) if (len(p) > 0 and t_human < t_ai) else 0.0
        out[key] = {
            "alpha": round(a, 4),
            "t_ai": round(float(np.clip(t_ai, 0.0, 1.000001)), 6),
            "t_human": round(float(np.clip(t_human, 0.0, 1.0)), 6),
            "grey_zone_active": bool(t_human < t_ai),
            "cal_fpr": round(emp_fpr, 4),
            "cal_recall_at_ai": round(emp_rec, 4),
            "cal_fnr_at_human": round(emp_fnr, 4),
            "cal_grey_rate": round(grey_rate, 4),
        }
    return out


def conformal_3way_decision(prob: float, entry: dict | None) -> str:
    if not entry:
        return "ai" if prob >= 0.70 else ("human" if prob <= 0.30 else "grey_zone")
    t_ai = float(entry["t_ai"])
    t_human = float(entry["t_human"])
    if prob >= t_ai:
        return "ai"
    if prob <= t_human:
        return "human"
    if t_human < t_ai:
        return "grey_zone"
    return "human"


def apply_exit_policy_v04(
    probs_steps: np.ndarray,
    halts_steps: np.ndarray,
    max_loops: int,
    policy: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply multi-step early exit with delta-probability convergence halting to eliminate bimodal collapse."""
    n = probs_steps.shape[1]
    max_loops = max(1, min(probs_steps.shape[0], max_loops))
    out_probs = np.empty(n, dtype=np.float32)
    steps_used = np.empty(n, dtype=np.int32)
    if policy is None or max_loops == 1:
        out_probs[:] = probs_steps[max_loops - 1]
        steps_used[:] = max_loops
        return out_probs, steps_used

    e_low = float(policy.get("exit_low", 0.25))
    e_high = float(policy.get("exit_high", 0.85))
    h_thr = float(policy.get("halt_threshold", 0.85))
    delta_conv = float(policy.get("delta_converge", -1.0))
    min_conv_step = int(policy.get("min_converge_step", 1))

    undecided = np.ones(n, dtype=bool)
    surv = np.ones(n, dtype=np.float32)
    for s in range(max_loops):
        p = probs_steps[s]
        surv = surv * (1.0 - np.clip(halts_steps[s], 0.0, 1.0))
        cum_h = 1.0 - surv
        if s < max_loops - 1:
            cond = (p <= e_low) | (p >= e_high) | (cum_h >= h_thr)
            if delta_conv >= 0.0 and s >= min_conv_step and s > 0:
                cond = cond | (np.abs(p - probs_steps[s - 1]) <= delta_conv)
            take = undecided & cond
        else:
            take = undecided
        out_probs[take] = p[take]
        steps_used[take] = s + 1
        undecided &= ~take
        if not undecided.any():
            break
    return out_probs, steps_used


def quantize_prelude_ordered_w4_w8(
    prelude_fp32_in: Path,
    prelude_out: Path,
    block_size: int = 32,
) -> dict:
    """Ordered W4-Emb / W8-Body Edge quantization: W4 GatherBlockQuantized on FP32 Prelude FIRST, then INT8 dense ops."""
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        from onnxruntime.quantization.quant_utils import QuantFormat
    except Exception as exc:
        return {"ok": False, "error": f"import_failed: {exc}"}

    tmp_named = prelude_out.with_suffix(".tmp_named.onnx")
    tmp_w4 = prelude_out.with_suffix(".tmp_w4only.onnx")
    try:
        model = onnx.load(str(prelude_fp32_in), load_external_data=True)
        init_shapes = {init.name: tuple(init.dims) for init in model.graph.initializer}
        target_gather_names: list[str] = []
        used_names = {n.name for n in model.graph.node if n.name}
        for idx, node in enumerate(model.graph.node):
            if node.op_type != "Gather" or not node.input:
                continue
            dims = init_shapes.get(node.input[0])
            if dims and len(dims) == 2 and dims[0] >= BASE_VOCAB_SIZE:
                if not node.name:
                    cand = f"v04_embed_gather_{idx}"
                    while cand in used_names:
                        cand += "_x"
                    node.name = cand
                    used_names.add(cand)
                target_gather_names.append(node.name)

        if not target_gather_names:
            return {"ok": False, "error": "no_embedding_gather_node_found"}

        onnx.save(model, str(tmp_named))
        quant = MatMulNBitsQuantizer(
            model=str(tmp_named),
            bits=4,
            block_size=int(block_size),
            is_symmetric=False,
            nodes_to_include=target_gather_names,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("Gather",),
            quant_axes=(("Gather", 1),),
        )
        quant.process()
        quant.model.save_model_to_file(str(tmp_w4))

        # Now dynamically INT8-quantize dense MatMul/Gemm layers while preserving W4 GatherBlockQuantized
        quantize_dynamic(
            str(tmp_w4),
            str(prelude_out),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )

        # Smoke-verify ORT loading and numerical parity on representative token + byte-fallback IDs
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        ref_sess = ort.InferenceSession(str(prelude_fp32_in), sess_options=opts, providers=["CPUExecutionProvider"])
        w4_sess = ort.InferenceSession(str(prelude_out), sess_options=opts, providers=["CPUExecutionProvider"])
        in_names = {inp.name for inp in ref_sess.get_inputs()}
        feed: dict[str, np.ndarray] = {
            "input_ids": np.array([[101, 200, 1000, BASE_VOCAB_SIZE + 5, 102]], dtype=np.int64),
            "attention_mask": np.ones((1, 5), dtype=np.int64),
        }
        if "char_ids" in in_names:
            feed["char_ids"] = np.full((1, 16), CHAR_PAD_ID, dtype=np.int64)
            feed["char_ids"][0, :5] = [0x61, 0x62, 0x20, 0x63, 0x64]
        ref_h0 = ref_sess.run(None, feed)[0]
        w4_h0 = w4_sess.run(None, feed)[0]
        diff = float(np.mean(np.abs(ref_h0 - w4_h0)))
        return {
            "ok": True,
            "order": "w4_gather_first_then_int8_linear",
            "target_gathers": target_gather_names,
            "block_size": int(block_size),
            "bytes": int(prelude_out.stat().st_size),
            "h0_mae_vs_fp32": round(diff, 6),
        }
    except Exception as exc:
        prelude_out.unlink(missing_ok=True)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        tmp_named.unlink(missing_ok=True)
        tmp_w4.unlink(missing_ok=True)


class MeguvizV04Engine:
    """Unified dual-graph compact-batch ONNX runtime supporting both v0.4-DualLoop (mHC/LOTUS/CharStem) and legacy v0.3-Loop bundles."""

    def __init__(self, bundle_path: Path, threads: int = 1, mode: str = "normal"):
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
        self.max_len = int(self.manifest.get("max_len", MAX_LEN))
        self.char_max_bytes = int(self.manifest.get("char_max_bytes", CHAR_MAX_BYTES))
        self.core_layers = int(self.manifest.get("core_layers", 2))
        self.num_loops = int(self.manifest.get("num_loops", 4))
        self.normal_max_loops = int(self.manifest.get("normal_max_loops", min(3, self.num_loops)))
        self.audit_max_loops = int(self.manifest.get("audit_max_loops", self.num_loops))

        policy = dict(self.manifest.get("exit_policy") or {})
        policy.setdefault("exit_low", 0.25)
        policy.setdefault("exit_high", 0.85)
        policy.setdefault("halt_threshold", 0.85)
        policy.setdefault("delta_converge", -1.0)
        policy.setdefault("min_converge_step", 1)
        self.exit_policy = policy

        self.isotonic_knots = self.manifest.get("isotonic_knots")
        self.conformal = self.manifest.get("conformal") or {}

        self.tokenizer = ByteFallbackWordPieceTokenizer(
            self.bundle_dir / "vocab.txt",
            byte_offset=int(self.manifest.get("byte_offset", BASE_VOCAB_SIZE)),
        )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        prelude_path = self.bundle_dir / self.manifest.get("prelude_file", "prelude.onnx")
        core_path = self.bundle_dir / self.manifest.get("core_file", "core_step.onnx")
        self.prelude_sess = ort.InferenceSession(str(prelude_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.core_sess = ort.InferenceSession(str(core_path), sess_options=opts, providers=["CPUExecutionProvider"])

        self.prelude_input_names = [i.name for i in self.prelude_sess.get_inputs()]
        self.prelude_output_names = [o.name for o in self.prelude_sess.get_outputs()]
        self.core_output_names = [o.name for o in self.core_sess.get_outputs()]
        self.uses_char_ids = "char_ids" in self.prelude_input_names

    def predict_batch(
        self,
        texts: list[str],
        mode: str | None = None,
        fixed_step: int | None = None,
        max_batch: int = 32,
        bucket_boundaries: tuple[int, ...] = (32, 64, 128, 192),
    ) -> list[dict]:
        """Run length-bucketed Prelude + early-exit compact-batch Core loops over texts."""
        if not texts:
            return []

        eff_mode = mode or self.mode
        if fixed_step is not None:
            if not (0 <= fixed_step < self.num_loops):
                raise ValueError(f"fixed_step must be in [0, {self.num_loops - 1}], got {fixed_step}")
            target_steps = fixed_step + 1
        else:
            limit = self.audit_max_loops if eff_mode == "audit" else self.normal_max_loops
            target_steps = max(1, min(self.num_loops, limit))

        encoded_ids = [self.tokenizer.encode_ids(t, max_len=self.max_len) for t in texts]
        lengths = [len(seq) for seq in encoded_ids]
        micro_batches = pack_length_buckets(lengths, max_batch=max_batch, boundaries=bucket_boundaries)
        results: list[dict | None] = [None] * len(texts)

        e_low = float(self.exit_policy["exit_low"])
        e_high = float(self.exit_policy["exit_high"])
        h_thr = float(self.exit_policy["halt_threshold"])
        delta_conv = float(self.exit_policy["delta_converge"])
        min_conv_step = int(self.exit_policy["min_converge_step"])

        for mb in micro_batches:
            max_seq = max(lengths[i] for i in mb)
            ids_arr = np.full((len(mb), max_seq), self.tokenizer.pad_id, dtype=np.int64)
            mask_arr = np.zeros((len(mb), max_seq), dtype=np.int64)
            for row_i, orig_idx in enumerate(mb):
                seq = encoded_ids[orig_idx]
                ids_arr[row_i, : len(seq)] = seq
                mask_arr[row_i, : len(seq)] = 1

            pre_feed: dict[str, np.ndarray] = {"input_ids": ids_arr, "attention_mask": mask_arr}
            if self.uses_char_ids:
                char_arr, _ = encode_char_bytes_batch([texts[i] for i in mb], max_bytes=self.char_max_bytes, min_len=8)
                pre_feed["char_ids"] = char_arr

            pre_outs = dict(zip(self.prelude_output_names, self.prelude_sess.run(None, pre_feed)))
            h0_cur = pre_outs["h0"]
            z_cur = pre_outs["z0"] if "z0" in pre_outs else h0_cur
            mask_cur = pre_outs["core_mask"] if "core_mask" in pre_outs else mask_arr

            active_pos = np.arange(len(mb), dtype=np.int64)
            surv = np.ones(len(mb), dtype=np.float32)
            prev_prob: np.ndarray | None = None

            for step in range(target_steps):
                core_outs = dict(
                    zip(
                        self.core_output_names,
                        self.core_sess.run(
                            None,
                            {
                                "z_prev": z_cur,
                                "h0": h0_cur,
                                "attention_mask": mask_cur,
                                "step_id": np.array(step, dtype=np.int64),
                            },
                        ),
                    )
                )
                z_next = core_outs["z_next"]
                prob = core_outs["prob"]
                logit = core_outs["logit"]
                halt_prob = core_outs["halt_prob"]

                surv = surv * (1.0 - np.clip(halt_prob, 0.0, 1.0))
                cum_halt = 1.0 - surv

                if fixed_step is None and step < target_steps - 1:
                    exit_mask = (prob <= e_low) | (prob >= e_high) | (cum_halt >= h_thr)
                    if delta_conv >= 0.0 and prev_prob is not None and step >= min_conv_step:
                        exit_mask = exit_mask | (np.abs(prob - prev_prob) <= delta_conv)
                else:
                    exit_mask = np.ones(len(active_pos), dtype=bool)

                if step == target_steps - 1:
                    exit_mask[:] = True

                for local_j in np.flatnonzero(exit_mask):
                    orig_idx = mb[int(active_pos[local_j])]
                    steps_used = step + 1
                    raw_p = float(prob[local_j])
                    cal_p = float(apply_isotonic_knots(raw_p, self.isotonic_knots))
                    results[orig_idx] = {
                        "prob": round(cal_p, 6),
                        "raw_prob": round(raw_p, 6),
                        "logit": round(float(logit[local_j]), 6),
                        "steps_used": steps_used,
                        "effective_layers": 1 + self.core_layers * steps_used,
                        "decision_fpr03": conformal_3way_decision(cal_p, self.conformal.get("alpha_03")),
                        "decision_fpr05": conformal_3way_decision(cal_p, self.conformal.get("alpha_05")),
                    }

                keep_idx = np.flatnonzero(~exit_mask)
                if len(keep_idx) == 0:
                    break

                active_pos = active_pos[keep_idx]
                z_cur = np.ascontiguousarray(z_next[keep_idx])
                h0_cur = np.ascontiguousarray(h0_cur[keep_idx])
                mask_cur = np.ascontiguousarray(mask_cur[keep_idx])
                surv = np.ascontiguousarray(surv[keep_idx])
                prev_prob = np.ascontiguousarray(prob[keep_idx])
                max_active_seq = int(mask_cur.sum(axis=1).max())
                if max_active_seq < mask_cur.shape[1]:
                    z_cur = np.ascontiguousarray(z_cur[..., :max_active_seq, :])
                    h0_cur = np.ascontiguousarray(h0_cur[:, :max_active_seq, :])
                    mask_cur = np.ascontiguousarray(mask_cur[:, :max_active_seq])

        return [
            r
            if r is not None
            else {
                "prob": 0.5,
                "raw_prob": 0.5,
                "logit": 0.0,
                "steps_used": 1,
                "effective_layers": 1 + self.core_layers,
                "decision_fpr03": "grey_zone",
                "decision_fpr05": "grey_zone",
            }
            for r in results
        ]

    def predict_one(self, text: str, mode: str | None = None, fixed_step: int | None = None) -> dict:
        return self.predict_batch([text], mode=mode, fixed_step=fixed_step, max_batch=1)[0]


def run_verify_v04(
    bundle_path: Path,
    probe_path: Path,
    mode: str = "normal",
    fixed_step: int | None = None,
    threads: int = 1,
) -> dict:
    """Run standardized U001-U017 + H01..H05 / A01..A05 traps + zero-[UNK] + unloaded seq=32 bs=32 speed check."""
    engine = MeguvizV04Engine(bundle_path, threads=threads, mode=mode)
    probes = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    human_traps = [
        ("H01_casual", "笑死我了，这个插件装完直接起飞[doge]"),
        ("H02_space", "玩到春七 最想加的一条规则 出门钓鱼先把锄头锁箱子"),
        ("H03_tech_human", "节点参考Blender材质节点这个思路挺对，接口丑不丑先不说，能白盒点开看子代理在干啥就很难得。"),
        ("H04_meme_banter", "大肥鱼在吗？我今天闲得发慌，想找条同类聊聊天，你平时都在干嘛呀？"),
        ("H05_emoji", "表情包已经出现了[妙啊][妙啊]"),
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
    )

    t0 = time.monotonic()
    single_out = [engine.predict_one(txt, mode=mode, fixed_step=fixed_step) for txt in all_texts]
    single_ms = (time.monotonic() - t0) * 1000.0 / max(1, len(all_texts))

    t1 = time.monotonic()
    batch_out = engine.predict_batch(all_texts, mode=mode, fixed_step=fixed_step, max_batch=32)
    batch_ms = (time.monotonic() - t1) * 1000.0 / max(1, len(all_texts))

    unk_total = 0
    byte_fallback_total = 0
    for txt in all_texts:
        _, stats = engine.tokenizer.encode_ids(txt, max_len=engine.max_len, return_stats=True)
        unk_total += stats["unk_tokens"]
        byte_fallback_total += stats["byte_fallback_tokens"]
    batch_single_mae = float(np.mean([abs(a["prob"] - b["prob"]) for a, b in zip(single_out, batch_out)]))

    n_p = len(probes)
    n_h = len(human_traps)
    n_a = len(ai_traps)
    probe_res = [{"id": p.get("id") or p.get("sample_id"), **r} for p, r in zip(probes, single_out[:n_p])]
    human_res = [{"id": k, **r} for (k, _), r in zip(human_traps, single_out[n_p : n_p + n_h])]
    ai_res = [{"id": k, **r} for (k, _), r in zip(ai_traps, single_out[n_p + n_h : n_p + n_h + n_a])]
    cross_res = [
        {"id": k, **r, **engine.tokenizer.encode_ids(txt, return_stats=True)[1]}
        for (k, txt), r in zip(EXTRA_CROSS_PLATFORM_TRAPS, single_out[n_p + n_h + n_a :])
    ]

    vals = [x["prob"] for x in probe_res]
    steps = [x["steps_used"] for x in single_out]
    step_hist = {int(s): sum(1 for x in steps if x == s) for s in sorted(set(steps))}
    prelude_bytes = (engine.bundle_dir / engine.manifest.get("prelude_file", "prelude.onnx")).stat().st_size
    core_bytes = (engine.bundle_dir / engine.manifest.get("core_file", "core_step.onnx")).stat().st_size

    u004_item = next((x for x in probe_res if x["id"] == "U004"), None)
    h03_item = next((x for x in human_res if x["id"] == "H03_tech_human"), None)
    min_ai_trap = min(x["prob"] for x in ai_res)

    return {
        "version": engine.manifest["version"],
        "mode": mode,
        "fixed_step": fixed_step,
        "uses_char_ids": engine.uses_char_ids,
        "prelude_bytes": prelude_bytes,
        "core_bytes": core_bytes,
        "total_onnx_bytes": prelude_bytes + core_bytes,
        "single_cpu_ms_per_comment": round(single_ms, 2),
        "compact_batch_cpu_ms_per_comment": round(batch_ms, 2),
        "batch_single_mae": round(batch_single_mae, 7),
        "avg_steps_used": round(sum(steps) / len(steps), 3),
        "step_hist": step_hist,
        "avg_effective_layers": round(sum(x["effective_layers"] for x in single_out) / len(single_out), 3),
        "unk_tokens_total": unk_total,
        "byte_fallback_tokens_total": byte_fallback_total,
        "thresholds": engine.manifest.get("thresholds", {}),
        "exit_policy": engine.exit_policy,
        "conformal": engine.conformal,
        "hard_trap_gates": {
            "ai_traps_min_prob": round(min_ai_trap, 4),
            "ai_traps_ge_078": min_ai_trap >= 0.78,
            "h03_tech_human_prob": round(h03_item["prob"], 4) if h03_item else None,
            "h03_lt_035": (h03_item["prob"] < 0.35) if h03_item else None,
            "u004_prob": round(u004_item["prob"], 4) if u004_item else None,
            "u004_ge_065": (u004_item["prob"] >= 0.65) if u004_item else None,
        },
        "probe_u001_u017": {
            "mean": round(sum(vals) / len(vals), 4),
            "median": round(sorted(vals)[len(vals) // 2], 4),
            "ge_060": sum(v >= 0.60 for v in vals),
            "ge_070": sum(v >= 0.70 for v in vals),
            "scores": probe_res,
        },
        "human_traps": human_res,
        "ai_stress_traps": ai_res,
        "cross_platform_zero_unk": cross_res,
    }


def _self_test() -> dict:
    # 1. CharRhythm preserves CJK spaces
    s_no_space = encode_char_bytes("玩到春七最想加的一条规则")
    s_spaced = encode_char_bytes("玩到春七 最想加的一条规则")
    assert s_no_space != s_spaced, "CharRhythm must distinguish spaced vs unspaced CJK text"

    # 2. PAVA monotonicity
    rng = np.random.default_rng(20261123)
    p = rng.uniform(0.0, 1.0, size=500)
    y = np.clip(p + rng.normal(0.0, 0.2, size=500), 0.0, 1.0)
    knots = fit_pava_isotonic(p, y)
    assert all(b >= a - 1e-9 for a, b in zip(knots["y"], knots["y"][1:])), "PAVA knots must be non-decreasing"

    # 3. Split conformal finite-sample FPR bound
    y_score = np.concatenate([rng.uniform(0.0, 40.0, size=200), rng.uniform(72.0, 99.0, size=200)])
    p_cal = np.concatenate([rng.beta(1.5, 5.0, size=200), rng.beta(5.0, 1.5, size=200)])
    conf = fit_split_conformal_3way(y_score, p_cal, alphas=(0.03, 0.05))
    assert conf["alpha_03"]["cal_fpr"] <= 0.03 + 1e-9
    assert conf["alpha_05"]["cal_fpr"] <= 0.05 + 1e-9

    # 4. Delta-convergence intermediate step exit test
    p_steps = np.array(
        [
            [0.92, 0.52, 0.45, 0.12],  # sample 0 & 3 exit at step 1
            [0.93, 0.53, 0.72, 0.11],  # sample 1 converges (|0.53-0.52|=0.01 <= 0.03) -> step 2!
            [0.94, 0.53, 0.78, 0.11],  # sample 2 needs step 3!
        ],
        dtype=np.float32,
    )
    h_steps = np.full_like(p_steps, 0.10)
    _, used = apply_exit_policy_v04(
        p_steps,
        h_steps,
        max_loops=3,
        policy={"exit_low": 0.25, "exit_high": 0.85, "halt_threshold": 0.85, "delta_converge": 0.03},
    )
    assert used.tolist() == [1, 2, 3, 1], f"Unexpected exit steps: {used.tolist()}"
    return {"char_diff_ok": True, "pava_knots": len(knots["x"]), "conformal": conf, "exit_steps_test": used.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--bundle", type=Path, default=None, help="Path to v0.4 or v0.3 zip or extracted bundle directory")
    parser.add_argument("--probes", type=Path, default=Path("round01/bili-round01-user-confirmed.jsonl"))
    parser.add_argument("--mode", choices=("normal", "audit"), default="normal")
    parser.add_argument("--fixed-step", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if args.self_test or args.bundle is None:
        res = {"self_test": _self_test()}
        if args.bundle is not None:
            res["verify"] = run_verify_v04(
                args.bundle.resolve(),
                args.probes.resolve(),
                mode=args.mode,
                fixed_step=args.fixed_step,
                threads=args.threads,
            )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    out = run_verify_v04(
        args.bundle.resolve(),
        args.probes.resolve(),
        mode=args.mode,
        fixed_step=args.fixed_step,
        threads=args.threads,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
