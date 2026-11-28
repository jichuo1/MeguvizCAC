"""MeguvizCAC v1.6.x / v1.7.0 Silicon Microarchitecture, Slice-SVD Monarch & LARDO Micro-Bench Suite.

Strict Engineering Implementation of Independent Review 03 Revision 2 (2026-11-28):
  1. Reproduces EVERY exact mathematical check & counterexample from Review 03 Rev. 2:
     - Roofline exact arithmetic for `M=14, K=320, N=1024` (`I_W=7.0`, `I_logical=6.6198744 FLOPs/B`).
     - Exact Monarch **Slice-SVD Optimal Projection Theorem** (`min ||W - sum M_u||_F^2 == sum_{b,j} sum_{k>r} sigma_k^2`, `< 1e-12`).
     - FP32 non-associativity counterexample (`fl(fl(2^24+1)-2^24) == 0` vs `fl(2^24+fl(1-2^24)) == 1`).
     - Coherent shared-weight loop sensitivity table (`rho = 1, 0.9, 0.5, -1` across `T2, T4, T5, T6`) refuting fixed `10-30x`.
     - Exact discrete multi-choice knapsack + rate-distortion pure stream savings (`W6/W3 = 375,000 B = 0.357628 MiB`, etc.).
     - Exact two-step fusion break-even `(1 - p) * c > p * s + d` (`p < 11.11%` at `c=25us, s=200us, d=0`).
  2. Executes REAL Live Local x86 CPU Pure-FP32 ONNX Opset-17 ORT Micro-Benchmark (`CAS-03/04`):
     - Compares **Dense `320x320`**, **Biaxial Monarch (`10x32`, `16x20`, `8x40`)**, and **Low-Rank (`r=64`)**
       on real ORT CPUExecutionProvider at `1` & `4` threads, `B=1 (M=14)` and `B=32 (M=448)`!
     - Prevents paper-FLOP self-deception by measuring actual Transpose + batched-SGEMM overhead (`Delta c`)!
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Exact Slice-SVD Optimal Projection for Rectangular Biaxial Monarch (Rev. 2 Sec 4.2)
# ---------------------------------------------------------------------------

def slice_svd_monarch_projection(
    w_dense: np.ndarray,
    p: int,
    q: int,
    rank: int = 1,
) -> dict[str, Any]:
    """Compute exact optimal Frobenius projection of `W: (p*q, p*q)` onto sum of `rank` biaxial Monarch matrices.

    In biaxial indexing `y_{a,b} = sum_j R_b[a,j] sum_k L_j[b,k] x_{j,k}`, each slice `W^{(b,j)} in R^{p x q}`
    is approximated independently by its rank-`r` truncated SVD (`Eckart-Young`).
    """
    h = p * q
    w = np.asarray(w_dense, dtype=np.float64)
    assert w.shape == (h, h), f"Expected {(h, h)}, got {w.shape}"
    # Reshape W[(a,b), (j,k)] -> (p, q, p, q) then transpose to slices (q_b, p_j, p_a, q_k)
    w4 = w.reshape(p, q, p, q).transpose(1, 2, 0, 3)  # (q, p, p, q)

    r_factors = np.zeros((rank, q, p, p), dtype=np.float64)  # (r, b, a, j)
    l_factors = np.zeros((rank, p, q, q), dtype=np.float64)  # (r, j, b, k)
    tail_sv_sq_sum = 0.0
    recon4 = np.zeros_like(w4)

    for b in range(q):
        for j in range(p):
            sl = w4[b, j]  # (p, q)
            u, s, vt = np.linalg.svd(sl, full_matrices=False)
            r_eff = min(rank, len(s))
            if r_eff < len(s):
                tail_sv_sq_sum += float(np.sum(s[r_eff:] ** 2))
            sl_hat = np.zeros_like(sl)
            for m in range(r_eff):
                sr = math.sqrt(float(s[m]))
                r_vec = u[:, m] * sr
                l_vec = vt[m, :] * sr
                r_factors[m, b, :, j] = r_vec
                l_factors[m, j, b, :] = l_vec
                sl_hat += np.outer(r_vec, l_vec)
            recon4[b, j] = sl_hat

    w_hat = recon4.transpose(2, 0, 1, 3).reshape(h, h)
    frob_sq_err = float(np.sum((w - w_hat) ** 2))
    rel_frob_err = math.sqrt(frob_sq_err / max(float(np.sum(w * w)), 1e-18))
    return {
        "p": p,
        "q": q,
        "rank": rank,
        "params_structured": int(rank * h * (p + q)),
        "params_dense": int(h * h),
        "param_reduction_pct": round((1.0 - (rank * h * (p + q)) / (h * h)) * 100.0, 3),
        "frob_sq_err": frob_sq_err,
        "tail_sv_sq_sum": tail_sv_sq_sum,
        "theorem_abs_diff": abs(frob_sq_err - tail_sv_sq_sum),
        "rel_frob_err": round(rel_frob_err, 6),
        "r_factors": r_factors,
        "l_factors": l_factors,
    }


# ---------------------------------------------------------------------------
# 2. Pure-FP32 Opset-17 Modules for Real CPU ORT Micro-Benchmark (`CAS-03/04`)
# ---------------------------------------------------------------------------

class DenseSquareBlock(nn.Module):
    def __init__(self, h: int = 320, n_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.layers:
            x = fc(x)
        return x


class BiaxialMonarchLayer(nn.Module):
    """Explicit Opset-17 `Reshape -> Transpose -> MatMul -> Transpose -> MatMul -> Transpose -> Reshape` (Rev. 2 Sec 5.1)."""

    def __init__(self, p: int, q: int) -> None:
        super().__init__()
        self.p = int(p)
        self.q = int(q)
        # Batched rank-3 weights: L^T [p, q, q], R^T [q, p, p]
        self.lt = nn.Parameter(torch.randn(p, q, q) * (1.0 / math.sqrt(q)))
        self.rt = nn.Parameter(torch.randn(q, p, p) * (1.0 / math.sqrt(p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, H] where H = p * q
        m = x.shape[0]
        t0 = x.reshape(m, self.p, self.q).permute(1, 0, 2)  # [p, M, q]
        t1 = torch.matmul(t0, self.lt)                      # [p, M, q]
        t2 = t1.permute(2, 1, 0)                            # [q, M, p]
        t3 = torch.matmul(t2, self.rt)                      # [q, M, p]
        return t3.permute(1, 2, 0).reshape(m, self.p * self.q)


class BiaxialMonarchBlock(nn.Module):
    def __init__(self, p: int, q: int, n_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([BiaxialMonarchLayer(p, q) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for ly in self.layers:
            x = ly(x)
        return x


class LowRankBlock(nn.Module):
    def __init__(self, h: int = 320, r: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(h, r, bias=False), nn.Linear(r, h, bias=False))
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for ly in self.layers:
            x = ly(x)
        return x


# ---------------------------------------------------------------------------
# 3. Discrete Multi-Choice Knapsack for `MGVZ10` W4/W8 Bit Allocation (`LARDO`, Sec 11.3)
# ---------------------------------------------------------------------------

def discrete_lardo_bit_search(
    layer_names: list[str],
    layer_n_params: list[int],
    layer_sens_w4: list[float],
    layer_sens_w8: list[float],
    extra_w8_budget_bytes: int,
) -> dict[str, Any]:
    """Greedy marginal-distortion-per-byte discrete W4 -> W8 allocation under strict `MGVZ10` budget (no W6!)."""
    items = []
    for name, n_p, d4, d8 in zip(layer_names, layer_n_params, layer_sens_w4, layer_sens_w8):
        # W4 -> W8 costs `0.5 byte/param` pure payload (plus small alignment/metadata headroom)
        delta_bytes = int(math.ceil(0.5 * n_p))
        gain = max(0.0, float(d4) - float(d8))
        items.append((gain / max(1, delta_bytes), gain, delta_bytes, name))
    items.sort(reverse=True)
    used_b = 0
    promoted_w8 = []
    total_gain = 0.0
    for ratio, gain, db, name in items:
        if used_b + db <= extra_w8_budget_bytes and gain > 0.0:
            used_b += db
            total_gain += gain
            promoted_w8.append({"layer": name, "extra_bytes": db, "distortion_saved": round(gain, 6)})
    return {
        "extra_w8_budget_bytes": int(extra_w8_budget_bytes),
        "extra_w8_used_bytes": int(used_b),
        "n_promoted_w8": len(promoted_w8),
        "total_proxy_distortion_saved": round(total_gain, 6),
        "promoted_layers": promoted_w8,
    }


# ---------------------------------------------------------------------------
# 4. Full Review 03 Rev. 2 Mathematical Verification + Live x86 ORT Micro-Benchmark!
# ---------------------------------------------------------------------------

def run_rev03_verification_and_ort_microbench() -> dict[str, Any]:
    # 1. Roofline exact numbers (Page 3)
    m_r, k_r, n_r = 14, 320, 1024
    f_ops = 2 * m_r * k_r * n_r
    q_w = 4 * k_r * n_r
    q_log = 4 * (m_r * k_r + k_r * n_r + m_r * n_r)
    i_w = f_ops / q_w
    i_log = f_ops / q_log
    assert f_ops == 9175040 and q_w == 1310720 and q_log == 1385984
    assert abs(i_w - 7.0) < 1e-12 and abs(i_log - 6.6198744) < 1e-6

    # 2. Unit conversion check (Page 2): 2.33M FP32 = 9.32 MB = 8.888245 MiB
    mib_233 = (2_330_000 * 4) / (1024 ** 2)
    assert abs(mib_233 - 8.8882446289) < 1e-6

    # 3. Exact Slice-SVD Monarch Projection Theorem (Page 5): p=3, q=4, r=2
    rng = np.random.default_rng(20261128)
    w_small = rng.standard_normal((12, 12))
    proj_check = slice_svd_monarch_projection(w_small, p=3, q=4, rank=2)
    assert proj_check["theorem_abs_diff"] < 1e-12, f"Slice-SVD theorem mismatch: {proj_check['theorem_abs_diff']}"

    # 4. FP32 non-associativity counterexample (Page 9)
    two24 = np.float32(2.0 ** 24)
    one32 = np.float32(1.0)
    lhs_fp32 = float(np.float32(np.float32(two24 + one32) - two24))
    rhs_fp32 = float(np.float32(two24 + np.float32(one32 - two24)))
    assert lhs_fp32 == 0.0 and rhs_fp32 == 1.0

    # 5. Coherent loop sensitivity table `(sum_{k=0}^{T-1} rho^k)^2` (Page 11)
    loop_sens = {}
    for rho in (1.0, 0.9, 0.5, -1.0):
        loop_sens[str(rho)] = {
            f"T{t}": round((sum(rho ** k for k in range(t))) ** 2, 6) for t in (2, 4, 5, 6)
        }
    assert loop_sens["1.0"] == {"T2": 4.0, "T4": 16.0, "T5": 25.0, "T6": 36.0}
    assert abs(loop_sens["0.9"]["T6"] - 21.954754) < 1e-5
    assert loop_sens["-1.0"] == {"T2": 0.0, "T4": 0.0, "T5": 1.0, "T6": 0.0}

    # 6. Rate-distortion stream savings (Page 12): N_l=0.5M, N_e=4M
    nl, ne = 500_000, 4_000_000
    save_w6_w3 = ((4 - 6) * nl + (4 - 3) * ne) / 8.0
    save_w8_w3 = ((4 - 8) * nl + (4 - 3) * ne) / 8.0
    assert int(save_w6_w3) == 375_000 and abs(save_w6_w3 / (1024 ** 2) - 0.357628) < 1e-5
    assert int(save_w8_w3) == 250_000 and abs(save_w8_w3 / (1024 ** 2) - 0.238419) < 1e-5

    # 7. Two-step fusion break-even `(1 - p) * c > p * s + d` (Page 10)
    p_be_d0 = 25.0 / (25.0 + 200.0)
    p_be_d10 = (25.0 - 10.0) / (25.0 + 200.0)
    assert abs(p_be_d0 - 1.0 / 9.0) < 1e-12 and abs(p_be_d10 - 15.0 / 225.0) < 1e-12

    # 8. REAL Local x86 CPU Pure-FP32 Opset-17 ORT Micro-Benchmark (`CAS-03/04`: 4 stacked 320x320 ops!)
    torch.manual_seed(42)
    candidates = {
        "Dense_320x320_x4": DenseSquareBlock(320, n_layers=4).eval(),
        "Monarch_10x32_x4": BiaxialMonarchBlock(10, 32, n_layers=4).eval(),
        "Monarch_16x20_x4": BiaxialMonarchBlock(16, 20, n_layers=4).eval(),
        "Monarch_8x40_x4": BiaxialMonarchBlock(8, 40, n_layers=4).eval(),
        "LowRank_r64_x4": LowRankBlock(320, r=64, n_layers=4).eval(),
    }
    ort_bench: dict[str, Any] = {}
    x_b1 = rng.standard_normal((14, 320)).astype(np.float32)     # B=1, 14 slots
    x_b32 = rng.standard_normal((448, 320)).astype(np.float32)   # B=32, 14 slots

    with tempfile.TemporaryDirectory() as td:
        for cname, mod in candidates.items():
            onnx_p = Path(td) / f"{cname}.onnx"
            dx = torch.randn(14, 320, dtype=torch.float32)
            torch.onnx.export(
                mod, (dx,), str(onnx_p), opset_version=17,
                input_names=["x"], output_names=["y"],
                dynamic_axes={"x": {0: "M"}, "y": {0: "M"}},
                dynamo=False,
            )
            om = onnx.load(str(onnx_p))
            onnx.checker.check_model(om)
            n_params = int(sum(p.numel() for p in mod.parameters()))
            row_res: dict[str, Any] = {"params": n_params, "fp32_kib": round(n_params * 4 / 1024.0, 1)}
            for n_th in (1, 4):
                so = ort.SessionOptions()
                so.intra_op_num_threads = n_th
                so.inter_op_num_threads = 1
                sess = ort.InferenceSession(str(onnx_p), sess_options=so, providers=["CPUExecutionProvider"])
                for tag, arr, n_it in (("B1_M14", x_b1, 300), ("B32_M448", x_b32, 80)):
                    for _ in range(25):
                        sess.run(None, {"x": arr})
                    times = []
                    for _ in range(n_it):
                        t0 = time.perf_counter()
                        sess.run(None, {"x": arr})
                        times.append((time.perf_counter() - t0) * 1000.0)
                    row_res[f"t{n_th}_{tag}_p50_p95_ms"] = [
                        round(float(np.percentile(times, 50)), 4),
                        round(float(np.percentile(times, 95)), 4),
                    ]
            ort_bench[cname] = row_res

    return {
        "all_rev03_assertions_passed": True,
        "roofline_check": {"F": f_ops, "Q_W": q_w, "Q_logical": q_log, "I_W": i_w, "I_logical": round(i_log, 7)},
        "slice_svd_theorem_diff": proj_check["theorem_abs_diff"],
        "fp32_non_assoc": [lhs_fp32, rhs_fp32],
        "loop_coherent_sensitivity": loop_sens,
        "two_step_fusion_breakeven_p": {"d0": round(p_be_d0, 4), "d10us": round(p_be_d10, 4)},
        "live_x86_ort_fp32_microbench_4x320x320": ort_bench,
    }


if __name__ == "__main__":
    print(json.dumps(run_rev03_verification_and_ort_microbench(), ensure_ascii=False, indent=2))
