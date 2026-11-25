"""MeguvizCAC v0.5.0-R3 (BGE-4L-pAUC-SAE) Architecture, ONNX INT8/Edge-W4 Exporter & Runtime Engine.

Fuses all empirically verified winners from v0.4.0-R3, v0.5.0-R1/R2, and the RTX 4060 Ti Lab:
- 100% lossless 4-layer BAAI/bge-small-zh-v1.5 backbone (d=512, H=8, FFN=2048; zero dropped layers,
  zero 16-Slot compression bottleneck, zero mHC-4/CharStem overhead)
- Zero-[UNK] WordPiece + 256 UTF-8 byte-fallback tokenizer (21,384 vocab, E=192 weighted SVD)
- Lightweight multi-scale DWConv(k=3,5,7) local phrasing cadence residual (<0.03 MB, near-zero gated)
- 32-d Zero-PII PIFE stylometrics + optional 4-d Zero-PII context echo
- Parallel Residual TopK-SAE Concept Head (K=8/128, empirically proven ultra-short regularizer & M04 killer)
- 256-d L2 Prototype + Restricted Mahalanobis OOD bank + Smooth PAVA + Split Conformal 3-Way
- Clean single-graph INT8 & Ordered W4-Emb/W8-Body ONNX export (zero split-session overhead!)
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v03_tokenizer import ByteFallbackWordPieceTokenizer, MAX_LEN
from .v04_runtime import EXTRA_CROSS_PLATFORM_TRAPS, MULTILINGUAL_TRAPS, conformal_3way_decision
from .v05_features import STYLE_FEAT_DIM, apply_bucketed_smooth_pava, extract_style_features_batch
from .v05_model import QATEmbedding, QATLinear
from .v05_runtime import PIFE_ADVERSARIAL_TRAPS, score_l2_prototype_ood


class QATBertLayer(nn.Module):
    """Exact BAAI/bge-small-zh-v1.5 BERT encoder layer with QATLinear + optional DWConv cadence residual."""

    def __init__(self, hidden_size: int = 512, num_heads: int = 8, intermediate_size: int = 2048, drop_p: float = 0.10, use_dwconv: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_dwconv = bool(use_dwconv)
        self.use_sdpa = False

        self.q_proj = QATLinear(hidden_size, hidden_size, bias=True)
        self.k_proj = QATLinear(hidden_size, hidden_size, bias=True)
        self.v_proj = QATLinear(hidden_size, hidden_size, bias=True)
        self.o_proj = QATLinear(hidden_size, hidden_size, bias=True)
        self.attn_ln = nn.LayerNorm(hidden_size, eps=1e-12)

        self.ff_in = QATLinear(hidden_size, intermediate_size, bias=True)
        self.ff_out = QATLinear(intermediate_size, hidden_size, bias=True)
        self.ff_ln = nn.LayerNorm(hidden_size, eps=1e-12)

        if self.use_dwconv:
            self.dw3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size, bias=False)
            self.dw5 = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size, bias=False)
            self.dw7 = nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3, groups=hidden_size, bias=False)
            self.dw_gate = nn.Parameter(torch.full((1, 1, hidden_size), -4.0))

        self.drop = nn.Dropout(drop_p)

    def load_from_bert_layer(self, hf_layer) -> None:
        with torch.no_grad():
            sa = hf_layer.attention.self
            ao = hf_layer.attention.output
            self.q_proj.weight.copy_(sa.query.weight)
            self.q_proj.bias.copy_(sa.query.bias)
            self.k_proj.weight.copy_(sa.key.weight)
            self.k_proj.bias.copy_(sa.key.bias)
            self.v_proj.weight.copy_(sa.value.weight)
            self.v_proj.bias.copy_(sa.value.bias)
            self.o_proj.weight.copy_(ao.dense.weight)
            self.o_proj.bias.copy_(ao.dense.bias)
            self.attn_ln.weight.copy_(ao.LayerNorm.weight)
            self.attn_ln.bias.copy_(ao.LayerNorm.bias)

            self.ff_in.weight.copy_(hf_layer.intermediate.dense.weight)
            self.ff_in.bias.copy_(hf_layer.intermediate.dense.bias)
            self.ff_out.weight.copy_(hf_layer.output.dense.weight)
            self.ff_out.bias.copy_(hf_layer.output.dense.bias)
            self.ff_ln.weight.copy_(hf_layer.output.LayerNorm.weight)
            self.ff_ln.bias.copy_(hf_layer.output.LayerNorm.bias)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        q = self.q_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.to(q.dtype), dropout_p=self.drop.p if self.training else 0.0)
        else:
            sc = torch.matmul(q, k.transpose(-1, -2)) * self.scale + attn_bias
            att = self.drop(torch.softmax(sc, dim=-1))
            ctx = torch.matmul(att, v)
        ctx = ctx.transpose(1, 2).contiguous().view(b, l, d)

        if self.use_dwconv:
            ht = (x * mask_f).transpose(1, 2)
            dw = ((self.dw3(ht) + self.dw5(ht) + self.dw7(ht)) / 3.0).transpose(1, 2)
            h = self.attn_ln(x + self.drop(self.o_proj(ctx)) + torch.sigmoid(self.dw_gate).to(x.dtype) * dw)
        else:
            h = self.attn_ln(x + self.drop(self.o_proj(ctx)))

        ff = self.ff_out(F.gelu(self.ff_in(h)))
        return self.ff_ln(h + self.drop(ff))


class ParallelTopKSAEHead(nn.Module):
    """RTX 4060 Ti E3b-validated Parallel Residual Top-K Sparse Concept Bottleneck (K=8/128, ~0.06 MB)."""

    def __init__(self, in_dim: int = 256, num_concepts: int = 128, top_k: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.num_concepts = num_concepts
        self.top_k = top_k
        self.enc = QATLinear(in_dim, num_concepts, bias=True)
        self.dec = nn.Linear(num_concepts, in_dim, bias=True)  # training-only reconstruction
        self.concept_head = QATLinear(num_concepts, 1, bias=False)
        self.res_gate = nn.Parameter(torch.tensor(-1.5))  # gentle sigmoid(~0.18) init so dense head stays smooth
        nn.init.zeros_(self.concept_head.weight)

    def forward(self, emb256_norm: torch.Tensor, return_recon: bool = False):
        pre = F.relu(self.enc(emb256_norm))
        vals, idxs = torch.topk(pre, k=self.top_k, dim=-1)
        sparse = torch.zeros_like(pre).scatter_(-1, idxs, vals)
        sae_logit = self.concept_head(sparse).squeeze(-1)
        gate = 0.5 * torch.sigmoid(self.res_gate).to(sae_logit.dtype)
        if not return_recon:
            return gate * sae_logit, sparse
        recon = self.dec(sparse)
        recon_loss = F.mse_loss(recon.float(), emb256_norm.detach().float())
        # AuxK gentle revival on top-16 unused concepts to prevent dead features
        dead_mask = (sparse <= 0).to(pre.dtype)
        aux_vals, aux_idxs = torch.topk(pre * dead_mask, k=min(16, self.num_concepts), dim=-1)
        aux_sparse = torch.zeros_like(pre).scatter_(-1, aux_idxs, aux_vals)
        aux_loss = F.mse_loss(self.dec(aux_sparse).float(), (emb256_norm.detach() - recon.detach()).float())
        return gate * sae_logit, sparse, recon_loss + 0.125 * aux_loss


class MeguvizV05R3BGEModel(nn.Module):
    """Full 4-layer BAAI/bge-small-zh-v1.5 + DWConv + 32-d PIFE + Parallel Residual TopK-SAE + 256-d L2 Coda."""

    def __init__(
        self,
        vocab_size: int = 21384,
        embed_dim: int = 192,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        max_len: int = MAX_LEN,
        style_dim: int = STYLE_FEAT_DIM,
        intent_dim: int = 1024,
        num_codes: int = 8,
        num_prag: int = 8,
        sae_concepts: int = 128,
        sae_topk: int = 8,
        drop_p: float = 0.10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len

        self.word_emb_A = QATEmbedding(vocab_size, embed_dim)
        self.word_emb_P = QATLinear(embed_dim, hidden_size, bias=False)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.emb_ln = nn.LayerNorm(hidden_size, eps=1e-12)
        self.emb_drop = nn.Dropout(drop_p)

        self.layers = nn.ModuleList(
            [
                QATBertLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    drop_p=drop_p,
                    use_dwconv=(i in (0, num_layers - 1)),
                )
                for i in range(num_layers)
            ]
        )

        # Learned attention pooling over full sequence (zero 16-Slot bottleneck loss!)
        self.attn_pool = QATLinear(hidden_size, 1, bias=True)

        # 32-d Zero-PII PIFE Stylometrics + optional 4-d Zero-PII Context Echo
        self.style_net = nn.Sequential(
            QATLinear(style_dim, 64, bias=True),
            nn.SiLU(),
            QATLinear(64, 128, bias=True),
            nn.LayerNorm(128),
        )
        self.ctx_echo_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 128),
        )
        nn.init.zeros_(self.ctx_echo_net[-1].weight)
        nn.init.zeros_(self.ctx_echo_net[-1].bias)

        # Coda 256-d L2 Embedding + Dense Head + Parallel Residual TopK-SAE Concept Head
        self.coda_proj = QATLinear(2 * hidden_size + 128, 256, bias=True)
        self.coda_ln = nn.LayerNorm(256)
        self.ai_head = QATLinear(256, 1, bias=True)
        self.sae_head = ParallelTopKSAEHead(256, num_concepts=sae_concepts, top_k=sae_topk)
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

        # Training-only auxiliary heads (0 inference cost!)
        self.aux_code_head = nn.Linear(256, num_codes)
        self.aux_prag_head = nn.Linear(256, num_prag)
        self.aux_intent_head = nn.Linear(256, intent_dim)
        self.aux_mid_head = nn.Linear(hidden_size, 1)  # Layer-2 deep supervision

    def load_from_bge(self, hf_model, a_mat: np.ndarray, p_mat: np.ndarray) -> None:
        with torch.no_grad():
            self.word_emb_A.weight.copy_(torch.from_numpy(a_mat))
            self.word_emb_P.weight.copy_(torch.from_numpy(p_mat.T))
            pe = hf_model.embeddings.position_embeddings.weight[: self.max_len]
            tt0 = hf_model.embeddings.token_type_embeddings.weight[0:1]
            self.pos_emb.weight.copy_(pe + tt0)
            self.emb_ln.weight.copy_(hf_model.embeddings.LayerNorm.weight)
            self.emb_ln.bias.copy_(hf_model.embeddings.LayerNorm.bias)
            for dst_l, src_l in zip(self.layers, hf_model.encoder.layer[: self.num_layers]):
                dst_l.load_from_bert_layer(src_l)

    def set_sdpa(self, enabled: bool) -> None:
        for lyr in self.layers:
            lyr.use_sdpa = bool(enabled)

    def set_qat_enabled(self, enabled: bool) -> None:
        for m in self.modules():
            if isinstance(m, (QATEmbedding, QATLinear)):
                m.qat_enabled = bool(enabled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        ctx_echo: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        b, l = input_ids.shape
        pos_ids = torch.arange(l, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        x = self.word_emb_P(self.word_emb_A(input_ids)) + self.pos_emb(pos_ids)
        x = self.emb_drop(self.emb_ln(x))

        attn_bias = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * (-10000.0)
        mask_f = attention_mask.to(x.dtype).unsqueeze(-1)

        h_mid = None
        for idx, lyr in enumerate(self.layers):
            x = lyr(x, attn_bias, mask_f)
            if idx == 1:
                h_mid = x

        cls_h = x[:, 0, :]
        pool_sc = self.attn_pool(x).squeeze(-1) + (1.0 - attention_mask.to(x.dtype)) * (-10000.0)
        pool_w = torch.softmax(pool_sc, dim=-1).unsqueeze(-1)
        att_h = (x * pool_w).sum(dim=1)

        st_vec = self.style_net(style_feats.to(x.dtype))
        if ctx_echo is not None:
            st_vec = st_vec + self.ctx_echo_net(ctx_echo.to(x.dtype))

        emb256_raw = F.silu(self.coda_ln(self.coda_proj(torch.cat([cls_h, att_h, st_vec], dim=-1))))
        emb256_norm = F.normalize(emb256_raw.float(), p=2, dim=-1).to(emb256_raw.dtype)
        dense_logit = self.ai_head(emb256_raw).squeeze(-1)

        if return_aux:
            sae_delta, sae_sparse, sae_loss = self.sae_head(emb256_norm, return_recon=True)
            logit = (dense_logit + sae_delta) / self.temperature.to(dense_logit.dtype).clamp(min=0.5)
            mid_logit = self.aux_mid_head(h_mid[:, 0, :]).squeeze(-1) if h_mid is not None else dense_logit
            return {
                "logit": logit,
                "dense_logit": dense_logit,
                "mid_logit": mid_logit,
                "emb256": emb256_norm,
                "sae_sparse": sae_sparse,
                "sae_loss": sae_loss,
                "code_logit": self.aux_code_head(emb256_raw),
                "prag_logit": self.aux_prag_head(emb256_raw),
                "intent_pred": self.aux_intent_head(emb256_raw),
            }

        sae_delta, _ = self.sae_head(emb256_norm, return_recon=False)
        logit = (dense_logit + sae_delta) / self.temperature.to(dense_logit.dtype).clamp(min=0.5)
        return logit, torch.sigmoid(logit), emb256_norm


class V05R3ExportWrapper(nn.Module):
    def __init__(self, model: MeguvizV05R3BGEModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, style_feats: torch.Tensor):
        return self.model(input_ids, attention_mask, style_feats, ctx_echo=None, return_aux=False)


def export_v05_r3_onnx(model_cpu: MeguvizV05R3BGEModel, out_dir: Path, vocab_size: int = 21384) -> dict:
    """Export single-graph INT8 & Ordered W4-Emb/W8-Body ONNX models with FP32 parity verification."""
    import onnx
    import onnxruntime as ort
    from onnx import version_converter
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from onnxruntime.quantization.quant_utils import QuantFormat

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cpu = model_cpu.cpu().eval()
    model_cpu.set_sdpa(False)
    model_cpu.set_qat_enabled(False)

    fp32_p = out_dir / "model.fp32.tmp.onnx"
    int8_p = out_dir / "model_int8.onnx"
    edge_p = out_dir / "model_edge_w4.onnx"

    d_ids = torch.ones((2, 32), dtype=torch.long)
    d_mask = torch.ones((2, 32), dtype=torch.long)
    d_st = torch.zeros((2, STYLE_FEAT_DIM), dtype=torch.float32)

    kw = dict(
        input_names=["input_ids", "attention_mask", "style_feats"],
        output_names=["logit", "prob", "emb256"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "style_feats": {0: "batch"},
            "logit": {0: "batch"},
            "prob": {0: "batch"},
            "emb256": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False

    torch.onnx.export(V05R3ExportWrapper(model_cpu).eval(), (d_ids, d_mask, d_st), str(fp32_p), **kw)
    quantize_dynamic(str(fp32_p), str(int8_p), weight_type=QuantType.QInt8)

    # Ordered W4-Emb (Gather opset21) + W8-Body Edge ONNX
    tmp_named = edge_p.with_suffix(".tmp_named.onnx")
    tmp_w4 = edge_p.with_suffix(".tmp_w4only.onnx")
    edge_res: dict = {"ok": False}
    try:
        m = onnx.load(str(fp32_p), load_external_data=True)
        opset = max((op.version for op in m.opset_import if op.domain in ("", "ai.onnx")), default=0)
        if opset < 21:
            m = version_converter.convert_version(m, 21)
            m.ir_version = max(m.ir_version, 10)
        init_shapes = {init.name: tuple(init.dims) for init in m.graph.initializer}
        target_gathers: list[str] = []
        used = {n.name for n in m.graph.node if n.name}
        for idx, node in enumerate(m.graph.node):
            if node.op_type != "Gather" or not node.input:
                continue
            dims = init_shapes.get(node.input[0])
            if dims and len(dims) == 2 and dims[0] == int(vocab_size):
                if not node.name:
                    cand = f"v05r3_emb_gather_{idx}"
                    while cand in used:
                        cand += "_x"
                    node.name = cand
                    used.add(cand)
                target_gathers.append(node.name)
        if not target_gathers:
            edge_res = {"ok": False, "error": "no_embedding_gather_found"}
        else:
            onnx.save(m, str(tmp_named))
            quant = MatMulNBitsQuantizer(
                model=str(tmp_named),
                bits=4,
                block_size=32,
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
                str(edge_p),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul", "Gemm"],
                extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
            )
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            ref_s = ort.InferenceSession(str(fp32_p), sess_options=opts, providers=["CPUExecutionProvider"])
            w4_s = ort.InferenceSession(str(edge_p), sess_options=opts, providers=["CPUExecutionProvider"])
            feed = {
                "input_ids": np.array([[101, 200, 300, 400, 102]], dtype=np.int64),
                "attention_mask": np.ones((1, 5), dtype=np.int64),
                "style_feats": np.zeros((1, STYLE_FEAT_DIM), dtype=np.float32),
            }
            diff = float(np.mean(np.abs(ref_s.run(None, feed)[1] - w4_s.run(None, feed)[1])))
            edge_res = {"ok": True, "bytes": int(edge_p.stat().st_size), "prob_mae_vs_fp32": round(diff, 6), "target_gathers": target_gathers}
    except Exception as exc:
        edge_p.unlink(missing_ok=True)
        edge_res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        fp32_p.unlink(missing_ok=True)
        tmp_named.unlink(missing_ok=True)
        tmp_w4.unlink(missing_ok=True)

    return {
        "model_int8": int8_p.name,
        "model_int8_bytes": int(int8_p.stat().st_size),
        "model_edge_w4": edge_p.name if edge_res.get("ok") else None,
        "edge_w4_w8": edge_res,
    }


class MeguvizV05R3Engine:
    """Zero-dependency ONNX Runtime CPU engine for MeguvizCAC v0.5.0-R3 (single-graph BGE-4L + PIFE + SAE)."""

    def __init__(self, bundle_path: Path, threads: int = 1, use_edge_w4: bool = False):
        import onnxruntime as ort
        import tempfile

        self.bundle_path = Path(bundle_path)
        self._tmpdir = None
        if self.bundle_path.is_file() and self.bundle_path.suffix == ".zip":
            self._tmpdir = tempfile.TemporaryDirectory(prefix="meguviz_v05r3_")
            with ZipFile(self.bundle_path, "r") as zf:
                zf.extractall(self._tmpdir.name)
            self.bundle_dir = Path(self._tmpdir.name)
        else:
            self.bundle_dir = self.bundle_path

        self.manifest = json.loads((self.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        self.tokenizer = ByteFallbackWordPieceTokenizer(self.bundle_dir / self.manifest.get("vocab_file", "vocab.txt"))
        self.max_len = int(self.manifest.get("max_len", MAX_LEN))
        self.cal_tables = self.manifest.get("smooth_pava") or {}
        self.conformal = self.manifest.get("conformal") or {}

        proto_fn = self.manifest.get("prototype_bank_file")
        self.proto_bank = None
        if proto_fn and (self.bundle_dir / proto_fn).is_file():
            z = np.load(self.bundle_dir / proto_fn)
            self.proto_bank = {k: z[k] for k in z.files}

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        m_fn = self.manifest.get("model_edge_w4") if (use_edge_w4 and self.manifest.get("model_edge_w4")) else self.manifest["model_file"]
        self.sess = ort.InferenceSession(str(self.bundle_dir / m_fn), sess_options=opts, providers=["CPUExecutionProvider"])

    def predict_batch(self, texts: list[str], max_batch: int = 32) -> list[dict]:
        if not texts:
            return []
        seqs = [self.tokenizer.encode_ids(t, max_len=self.max_len) for t in texts]
        lengths = [len(s) for s in seqs]
        style_all = extract_style_features_batch(texts)

        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        raw_probs = np.zeros(len(texts), dtype=np.float32)
        raw_logits = np.zeros(len(texts), dtype=np.float32)
        ood_scores = np.zeros(len(texts), dtype=np.float32)

        for st in range(0, len(order), max_batch):
            mb = order[st : st + max_batch]
            max_seq = max(lengths[i] for i in mb)
            ids = np.full((len(mb), max_seq), self.tokenizer.pad_id, dtype=np.int64)
            mask = np.zeros((len(mb), max_seq), dtype=np.int64)
            for row_i, oi in enumerate(mb):
                s = seqs[oi]
                ids[row_i, : len(s)] = s
                mask[row_i, : len(s)] = 1
            st_mb = style_all[mb]
            lg, pr, emb = self.sess.run(None, {"input_ids": ids, "attention_mask": mask, "style_feats": st_mb})
            _, ood_v = score_l2_prototype_ood(emb, self.proto_bank)
            for row_i, oi in enumerate(mb):
                raw_logits[oi] = float(lg[row_i])
                raw_probs[oi] = float(pr[row_i])
                ood_scores[oi] = float(ood_v[row_i])

        cal_probs = apply_bucketed_smooth_pava(texts, raw_probs, self.cal_tables)
        out: list[dict] = []
        for i in range(len(texts)):
            cp = float(cal_probs[i])
            out.append(
                {
                    "prob": round(cp, 6),
                    "raw_prob": round(float(raw_probs[i]), 6),
                    "logit": round(float(raw_logits[i]), 6),
                    "ood_score": round(float(ood_scores[i]), 4),
                    "decision_fpr03": conformal_3way_decision(cp, self.conformal.get("alpha_03")),
                    "decision_fpr05": conformal_3way_decision(cp, self.conformal.get("alpha_05")),
                }
            )
        return out

    def predict_one(self, text: str) -> dict:
        return self.predict_batch([text], max_batch=1)[0]


def run_verify_v05_r3(bundle_path: Path, probe_path: Path, threads: int = 1, use_edge_w4: bool = False) -> dict:
    engine = MeguvizV05R3Engine(bundle_path, threads=threads, use_edge_w4=use_edge_w4)
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
    single_out = [engine.predict_one(txt) for txt in all_texts]
    single_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(all_texts))
    t1 = time.perf_counter()
    _ = engine.predict_batch(all_texts, max_batch=32)
    batch_ms = (time.perf_counter() - t1) * 1000.0 / max(1, len(all_texts))

    unk_total = sum(engine.tokenizer.encode_ids(txt, max_len=engine.max_len, return_stats=True)[1]["unk_tokens"] for txt in all_texts)
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

    return {
        "version": engine.manifest["version"],
        "schema_version": engine.manifest.get("schema_version"),
        "single_cpu_ms_per_comment": round(single_ms, 3),
        "compact_batch_cpu_ms_per_comment": round(batch_ms, 3),
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
        "probe_u001_u017": probe_res,
        "human_traps": human_res,
        "ai_stress_traps": ai_res,
        "cross_platform_traps": cross_res,
        "multilingual_traps": ml_res,
        "pife_adversarial_traps": pife_res,
    }
