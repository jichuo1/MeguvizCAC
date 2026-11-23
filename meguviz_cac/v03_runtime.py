"""Zero-duplication dual-graph ONNX compact-batch runtime & W4-Emb Edge quantizer for MeguvizCAC v0.3-Loop."""

import argparse
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import onnxruntime as ort

from .v03_tokenizer import BASE_VOCAB_SIZE, ByteFallbackWordPieceTokenizer, MAX_LEN, pack_length_buckets


EXTRA_CROSS_PLATFORM_TRAPS = [
    ("U_kaomoji", "这个配音简直绝了(˘ω˘)ｽﾔｧ 还带着一点٩(◕‿◕｡)۶的微妙吐槽味"),
    ("U_rare_cjk_emoji", "𠮷野家旁边新开的店确实🫠有点离谱，不过味道还算稳"),
    ("U_code_mix", "这里用 asyncio.gather(*tasks, return_exceptions=True) 比手写 ThreadPool 优雅多啦"),
    ("U_symbol_mix", "§①→②★ 你说的『逻辑闭环』在 ℝⁿ 空间里好像根本不成立啊"),
]


def quantize_embedding_w4_opset21(onnx_in: Path, onnx_out: Path, block_size: int = 32) -> dict:
    """Block-quantize the 21384xE word embedding Gather weight into opset-21 com.microsoft::GatherBlockQuantized W4."""
    try:
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        from onnxruntime.quantization.quant_utils import QuantFormat
    except Exception as exc:
        return {"ok": False, "error": f"import_failed: {exc}"}

    tmp_in = onnx_out.with_suffix(".tmp_pre_w4.onnx")
    try:
        model = onnx.load(str(onnx_in), load_external_data=True)
        init_shapes = {init.name: tuple(init.dims) for init in model.graph.initializer}
        target_gather_names: list[str] = []
        used_names = {n.name for n in model.graph.node if n.name}
        for idx, node in enumerate(model.graph.node):
            if node.op_type != "Gather" or not node.input:
                continue
            w_name = node.input[0]
            dims = init_shapes.get(w_name)
            if dims and len(dims) == 2 and dims[0] >= BASE_VOCAB_SIZE:
                if not node.name:
                    cand = f"v03_embed_gather_{idx}"
                    while cand in used_names:
                        cand += "_x"
                    node.name = cand
                    used_names.add(cand)
                target_gather_names.append(node.name)

        if not target_gather_names:
            return {"ok": False, "error": "no_embedding_gather_node_found"}

        onnx.save(model, str(tmp_in))
        quant = MatMulNBitsQuantizer(
            model=str(tmp_in),
            bits=4,
            block_size=int(block_size),
            is_symmetric=False,
            nodes_to_include=target_gather_names,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("Gather",),
            quant_axes=(("Gather", 1),),
        )
        quant.process()
        quant.model.save_model_to_file(str(onnx_out))

        # Smoke-verify ORT loading and numerical parity on representative token IDs
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        ref_sess = ort.InferenceSession(str(onnx_in), sess_options=opts, providers=["CPUExecutionProvider"])
        w4_sess = ort.InferenceSession(str(onnx_out), sess_options=opts, providers=["CPUExecutionProvider"])
        test_ids = np.array([[101, 200, 1000, BASE_VOCAB_SIZE + 5, 102]], dtype=np.int64)
        test_mask = np.ones_like(test_ids, dtype=np.int64)
        ref_h0 = ref_sess.run(None, {"input_ids": test_ids, "attention_mask": test_mask})[0]
        w4_h0 = w4_sess.run(None, {"input_ids": test_ids, "attention_mask": test_mask})[0]
        diff = float(np.mean(np.abs(ref_h0 - w4_h0)))
        return {
            "ok": True,
            "target_gathers": target_gather_names,
            "block_size": int(block_size),
            "bytes": int(onnx_out.stat().st_size),
            "h0_mae_vs_prelude_int8": round(diff, 6),
        }
    except Exception as exc:
        onnx_out.unlink(missing_ok=True)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        tmp_in.unlink(missing_ok=True)


class MeguvizV03Engine:
    """Compact-batch length-bucketed dual-graph inference engine for MeguvizCAC v0.3-Loop."""

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
        self.core_layers = int(self.manifest.get("core_layers", 2))
        self.num_loops = int(self.manifest.get("num_loops", 4))
        self.normal_max_loops = int(self.manifest.get("normal_max_loops", min(3, self.num_loops)))
        self.audit_max_loops = int(self.manifest.get("audit_max_loops", self.num_loops))

        policy = self.manifest.get("exit_policy") or {}
        self.exit_low = float(policy.get("exit_low", 0.25))
        self.exit_high = float(policy.get("exit_high", 0.85))
        self.halt_threshold = float(policy.get("halt_threshold", 0.85))

        vocab_path = self.bundle_dir / "vocab.txt"
        self.tokenizer = ByteFallbackWordPieceTokenizer(
            vocab_path,
            byte_offset=int(self.manifest.get("byte_offset", BASE_VOCAB_SIZE)),
        )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        prelude_path = self.bundle_dir / self.manifest.get("prelude_file", "prelude.onnx")
        core_path = self.bundle_dir / self.manifest.get("core_file", "core_step.onnx")
        self.prelude_sess = ort.InferenceSession(str(prelude_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.core_sess = ort.InferenceSession(str(core_path), sess_options=opts, providers=["CPUExecutionProvider"])

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

        for mb in micro_batches:
            max_seq = max(lengths[i] for i in mb)
            ids_arr = np.full((len(mb), max_seq), self.tokenizer.pad_id, dtype=np.int64)
            mask_arr = np.zeros((len(mb), max_seq), dtype=np.int64)
            for row_i, orig_idx in enumerate(mb):
                seq = encoded_ids[orig_idx]
                ids_arr[row_i, : len(seq)] = seq
                mask_arr[row_i, : len(seq)] = 1

            h0 = self.prelude_sess.run(None, {"input_ids": ids_arr, "attention_mask": mask_arr})[0]
            active_pos = np.arange(len(mb), dtype=np.int64)
            z_cur = h0
            h0_cur = h0
            mask_cur = mask_arr
            surv = np.ones(len(mb), dtype=np.float32)

            for step in range(target_steps):
                z_next, prob, logit, halt_prob = self.core_sess.run(
                    None,
                    {
                        "z_prev": z_cur,
                        "h0": h0_cur,
                        "attention_mask": mask_cur,
                        "step_id": np.array(step, dtype=np.int64),
                    },
                )
                surv = surv * (1.0 - np.clip(halt_prob, 0.0, 1.0))
                cum_halt = 1.0 - surv

                if step == target_steps - 1:
                    exit_mask = np.ones(len(active_pos), dtype=bool)
                elif fixed_step is None:
                    exit_mask = (
                        (prob <= self.exit_low)
                        | (prob >= self.exit_high)
                        | (cum_halt >= self.halt_threshold)
                    )
                else:
                    exit_mask = np.zeros(len(active_pos), dtype=bool)

                for local_j in np.flatnonzero(exit_mask):
                    orig_idx = mb[int(active_pos[local_j])]
                    steps_used = step + 1
                    results[orig_idx] = {
                        "prob": round(float(prob[local_j]), 6),
                        "logit": round(float(logit[local_j]), 6),
                        "steps_used": steps_used,
                        "effective_layers": 1 + self.core_layers * steps_used,
                    }

                keep_idx = np.flatnonzero(~exit_mask)
                if len(keep_idx) == 0:
                    break

                active_pos = active_pos[keep_idx]
                z_cur = np.ascontiguousarray(z_next[keep_idx])
                h0_cur = np.ascontiguousarray(h0_cur[keep_idx])
                mask_cur = np.ascontiguousarray(mask_cur[keep_idx])
                surv = np.ascontiguousarray(surv[keep_idx])
                max_active_seq = int(mask_cur.sum(axis=1).max())
                if max_active_seq < mask_cur.shape[1]:
                    z_cur = np.ascontiguousarray(z_cur[:, :max_active_seq, :])
                    h0_cur = np.ascontiguousarray(h0_cur[:, :max_active_seq, :])
                    mask_cur = np.ascontiguousarray(mask_cur[:, :max_active_seq])

        return [r if r is not None else {"prob": 0.5, "logit": 0.0, "steps_used": 1, "effective_layers": 1 + self.core_layers} for r in results]

    def predict_one(self, text: str, mode: str | None = None, fixed_step: int | None = None) -> dict:
        return self.predict_batch([text], mode=mode, fixed_step=fixed_step, max_batch=1)[0]


def run_verify_v03(
    bundle_path: Path,
    probe_path: Path,
    mode: str = "normal",
    fixed_step: int | None = None,
    threads: int = 1,
) -> dict:
    """Run standardized U001-U017 + adversarial + cross-platform zero-[UNK] benchmark on a v0.3 bundle."""
    engine = MeguvizV03Engine(bundle_path, threads=threads, mode=mode)
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

    # Zero-[UNK] & batch/single parity check
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
    probe_res = [{"id": p["id"], **r} for p, r in zip(probes, single_out[:n_p])]
    human_res = [{"id": k, **r} for (k, _), r in zip(human_traps, single_out[n_p : n_p + n_h])]
    ai_res = [{"id": k, **r} for (k, _), r in zip(ai_traps, single_out[n_p + n_h : n_p + n_h + n_a])]
    cross_res = [
        {"id": k, **r, **engine.tokenizer.encode_ids(txt, return_stats=True)[1]}
        for (k, txt), r in zip(EXTRA_CROSS_PLATFORM_TRAPS, single_out[n_p + n_h + n_a :])
    ]

    vals = [x["prob"] for x in probe_res]
    steps = [x["steps_used"] for x in single_out]
    prelude_bytes = (engine.bundle_dir / engine.manifest.get("prelude_file", "prelude.onnx")).stat().st_size
    core_bytes = (engine.bundle_dir / engine.manifest.get("core_file", "core_step.onnx")).stat().st_size

    return {
        "version": engine.manifest["version"],
        "mode": mode,
        "fixed_step": fixed_step,
        "prelude_bytes": prelude_bytes,
        "core_bytes": core_bytes,
        "total_onnx_bytes": prelude_bytes + core_bytes,
        "single_cpu_ms_per_comment": round(single_ms, 2),
        "compact_batch_cpu_ms_per_comment": round(batch_ms, 2),
        "batch_single_mae": round(batch_single_mae, 7),
        "avg_steps_used": round(sum(steps) / len(steps), 3),
        "avg_effective_layers": round(sum(x["effective_layers"] for x in single_out) / len(single_out), 3),
        "unk_tokens_total": unk_total,
        "byte_fallback_tokens_total": byte_fallback_total,
        "thresholds": engine.manifest.get("thresholds", {}),
        "exit_policy": engine.manifest.get("exit_policy", {}),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Path to v0.3 zip or extracted bundle directory")
    parser.add_argument("--probes", type=Path, default=Path("round01/bili-round01-user-confirmed.jsonl"))
    parser.add_argument("--mode", choices=("normal", "audit"), default="normal")
    parser.add_argument("--fixed-step", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    out = run_verify_v03(
        args.bundle.resolve(),
        args.probes.resolve(),
        mode=args.mode,
        fixed_step=args.fixed_step,
        threads=args.threads,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
