"""Round-two Bili/DSH synthetic ablation on the frozen first-round splits."""

import argparse
import gzip
import hashlib
import html
import json
import math
import time
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import sklearn
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import bili_round01_train as first


SPLIT_SHA = "293265436d2e4ab2dc5e4966ffadfb8fafd46b73a5675c4c84cee28c4a2276ce"
CORPUS_SHA = "e64762a497309612d88a299e12b58a89c6406429de379b8811a0866ae438197c"
SYNTHETIC_DELIVERY_SHA = "99e92310a7b31846f2e117117752567b91b5cfe4001377472e3d9eb01ad293e0"
PROBE_SHA = "49739d460fc48c8fd22d85c732f2d81069e780ea90657f8e2f7a4b826e96f372"
COHORT_SIZES = {"deepseek_v41": 100_000, "other_models": 50_000}
SEED = 20261123
C = 2.0
ARMS = (
    ("bili_only", "纯 B 站", 0, 0),
    ("synth_02", "2%", 3_000, 3_000),
    ("synth_10", "10%", 15_000, 15_000),
    ("synth_25", "25%", 37_500, 37_500),
    ("synth_50", "50%", 75_000, 75_000),
    ("synth_100", "100%", 150_000, 150_000),
    ("synth_100_capped", "全量低权重", 150_000, 3_000),
)
MODEL_FILE = "bili-round02-models.joblib"
RESULT_FILE = "bili-round02-metrics.json"
PREDICTION_FILE = "bili-round02-eval-predictions.jsonl.gz"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f"SHA256 不匹配：{path.name}")


def split_fingerprint(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(("\t".join((row[0], row[1], row[3], row[4])) + "\n").encode("utf-8"))
    return digest.hexdigest()


def load_synthetic(root, real_keys, probe_keys):
    delivery_path = root / "delivery.json"
    verify(delivery_path, SYNTHETIC_DELIVERY_SHA)
    exports = json.loads(delivery_path.read_text(encoding="utf-8"))["exports"]
    pools = {}
    seen = set()
    for family, expected_count in COHORT_SIZES.items():
        export = exports[family]
        path = root / export["file"]
        verify(path, export["sha256"])
        expected_source = "zcode_deepseek" if family == "deepseek_v41" else "zcode_other"
        pool = []
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                text = row["text"]
                key = first.body_key(text)
                material_id = "S" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
                if (row["label"], row["split"], row["family"], row["source"]) != (
                    "ai", "train", family, expected_source
                ):
                    raise ValueError("合成样本的来源或训练集合标记无效。")
                if (not key or row["sample_id"] != material_id or key in real_keys
                        or key in probe_keys or key in seen):
                    raise ValueError("发现重复合成正文、样本编号异常或与原语料重合。")
                seen.add(key)
                pool.append((material_id, text))
        if len(pool) != export["rows"] or len(pool) != expected_count:
            raise ValueError(f"合成样本数不匹配：{family}")
        pools[family] = pool

    rng = np.random.default_rng(SEED)
    deepseek = pools["deepseek_v41"]
    other = pools["other_models"]
    deepseek_order = rng.permutation(len(deepseek))
    other_order = rng.permutation(len(other))
    ordering = []
    for offset, index in enumerate(other_order):
        ordering.extend((deepseek[deepseek_order[2 * offset]], deepseek[deepseek_order[2 * offset + 1]], other[index]))
    observed = {family: {"rows": size, "sha256": exports[family]["sha256"]}
                for family, size in COHORT_SIZES.items()}
    return ordering, observed


def fit_cpu(x, y, sample_weights):
    started = time.monotonic()
    model = LogisticRegression(C=C, max_iter=300, solver="liblinear", random_state=20261122)
    model.fit(x, y, sample_weight=sample_weights)
    return {"weights": model.coef_.reshape(-1), "intercept": float(model.intercept_[0])}, {
        "backend": "sklearn/liblinear CPU", "seconds": round(time.monotonic() - started, 3),
        "iterations": int(model.n_iter_[0]),
    }


def fit_cuda(torch, x, y, sample_weights):
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    def device_csr(matrix):
        matrix.sum_duplicates()
        matrix.sort_indices()
        return torch.sparse_csr_tensor(
            torch.as_tensor(matrix.indptr, device=device),
            torch.as_tensor(matrix.indices, device=device),
            torch.as_tensor(matrix.data, dtype=torch.float32, device=device),
            size=matrix.shape,
        )

    matrix = device_csr(x)
    transpose = device_csr(x.T.tocsr())
    signed = torch.as_tensor(y.astype(np.float32) * 2 - 1, device=device)
    weights = torch.as_tensor(sample_weights.astype(np.float32), device=device)
    denominator = weights.sum()
    regularization = 1 / (C * denominator)
    coefficients = torch.zeros(x.shape[1] + 1, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [coefficients], max_iter=280, max_eval=375, history_size=30,
        tolerance_grad=3e-8, tolerance_change=1e-11, line_search_fn="strong_wolfe",
    )

    @torch.no_grad()
    def objective_and_gradient():
        logits = torch.sparse.mm(matrix, coefficients[:-1, None]).reshape(-1) + coefficients[-1]
        negative_margin = -signed * logits
        residual = -signed * torch.sigmoid(negative_margin) * weights / denominator
        gradient = torch.sparse.mm(transpose, residual[:, None]).reshape(-1)
        coefficients.grad = torch.cat((gradient, residual.sum().reshape(1))) + regularization * coefficients
        loss = (torch.nn.functional.softplus(negative_margin) * weights).sum() / denominator
        return loss + regularization * (coefficients @ coefficients) / 2

    optimizer.step(objective_and_gradient)
    loss = float(objective_and_gradient())
    maximum_gradient = float(coefficients.grad.abs().max())
    array = coefficients.detach().cpu().numpy().astype(np.float64)
    state = optimizer.state[coefficients]
    if not np.isfinite(array).all() or not math.isfinite(loss) or maximum_gradient > 2e-5:
        raise RuntimeError(f"CUDA 拟合未通过收敛核验，梯度上限为 {maximum_gradient:.8f}")
    return {"weights": array[:-1], "intercept": float(array[-1])}, {
        "backend": "PyTorch CUDA L-BFGS",
        "seconds": round(time.monotonic() - started, 3),
        "iterations": int(state.get("n_iter", 0)), "evaluations": int(state.get("func_evals", 0)),
        "objective": loss, "maximum_gradient": maximum_gradient,
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
    }


def score(matrix, head):
    return expit(np.asarray(matrix @ head["weights"] + head["intercept"], dtype=np.float64))


def predict(bundle, texts, variant=None):
    if not texts:
        return np.array([], dtype=np.float64)
    return score(first.build_matrix(bundle, texts), bundle["heads"][variant or bundle["preferred_variant"]])


def matched_threshold(scores, negative_mask, false_positive_budget):
    negatives = np.sort(scores[negative_mask])[::-1]
    if not 0 <= false_positive_budget < len(negatives):
        raise ValueError("误判预算必须小于 DSH 验证参考负例数。")
    cutoff = float(np.nextafter(negatives[false_positive_budget], np.inf))
    if int((negatives >= cutoff).sum()) > false_positive_budget:
        raise ValueError("阈值实测超出验证误判预算。")
    return cutoff


def paired_video_intervals(videos, labels, base_alert, variant_alert):
    _, group = np.unique(videos, return_inverse=True)
    group_count = int(group.max()) + 1
    changed = variant_alert.astype(np.int64) - base_alert.astype(np.int64)
    positive = np.bincount(group[labels], minlength=group_count)
    negative = np.bincount(group[~labels], minlength=group_count)
    delta_tp = np.bincount(group[labels], weights=changed[labels], minlength=group_count)
    delta_fp = np.bincount(group[~labels], weights=changed[~labels], minlength=group_count)
    draws = np.random.default_rng(5332).multinomial(group_count, [1 / group_count] * group_count, size=2000)
    report = {}
    for name, difference, total in (
        ("recall", delta_tp, positive), ("false_positive_rate", delta_fp, negative)
    ):
        denominators = draws @ total
        valid = denominators > 0
        estimates = (draws @ difference)[valid] / denominators[valid]
        report[name] = np.quantile(estimates, [0.025, 0.975]).tolist()
    return report


def render_report(report):
    rows = []
    for model in report["models"].values():
        dsh = model["test"]["dsh"]
        fixed = dsh["points"]["dsh_matched"]
        f05 = dsh["points"]["f05"]
        cells = (model["label"], f"{model['synthetic_rows']:,}", f"{f05['recall']:.1%}",
                 f"{f05['false_positive_rate']:.1%}", f"{fixed['recall']:.1%}",
                 f"{fixed['false_positive_rate']:.1%}")
        rows.append("<tr>" + "".join(f"<td>{html.escape(val)}</td>" for val in cells) + "</tr>")
    preferred = report["models"][report["preferred_by_validation"]]["label"]
    payload = html.escape(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    device = html.escape(report["hardware"]["training_backend"])
    sample_count = report["models"]["bili_only"]["test"]["dsh"]["rows"]
    seconds = report["elapsed_seconds"]
    return f"""<div style="font-family:system-ui;padding:11px;border:1px solid #94a3b8;border-radius:9px;max-width:940px">
<h2 style="margin:0 0 5px">第二轮合成比例对照已完成</h2>
<div>DSH 留出 {sample_count:,} 条 · 7 组 · {device} · {seconds:.0f} 秒</div>
<style>.ratio2{{font-size:13px;border-collapse:collapse;width:100%;margin:8px 0}}.ratio2 td,.ratio2 th{{text-align:left;padding:3px 4px;border-bottom:1px solid #9995}}.ratio2 td:not(:first-child){{white-space:nowrap}}</style>
<table class="ratio2"><thead><tr><th>用量</th><th>合成条数</th><th>召回 F0.5</th><th>误判 F0.5</th><th>同预算召回</th><th>同预算误判</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div>验证集优先配置：<b>{html.escape(preferred)}</b> · 按 Gemini 与规则参考标签计分</div>
<details><summary>完整指标和数据校验</summary><pre id="round02-report-json">{payload}</pre></details></div>"""


def run(root, output, use_cuda=True):
    started = time.monotonic()
    split_path = root / "bili-round01-split.json"
    probe_path = root / "bili-round01-user-confirmed.jsonl"
    verify(split_path, SPLIT_SHA)
    verify(probe_path, PROBE_SHA)
    split, sets, real_keys = first.load_inputs(root)
    if split["corpus_sha256"] != CORPUS_SHA:
        raise ValueError("输入与首轮冻结语料不一致。")
    training, validation, test = (sets[part] for part in ("train", "validation", "test"))
    groups = split["video_assignments"]
    held = {part: split_fingerprint(sets[part]) for part in ("train", "validation", "test")}
    probes = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines()]
    probe_keys = {first.body_key(row["text"]) for row in probes}
    ordering, sources = load_synthetic(root, real_keys, probe_keys)
    del real_keys
    print(f"校验通过：训练 {len(training):,} / 验证 {len(validation):,} / 测试 {len(test):,}。"
          f"待复核 {split['review_rows']:,} 条，合成库 {len(ordering):,} 条。", flush=True)

    hardware = {"accelerator": None, "torch": None, "training_backend": "sklearn/liblinear CPU"}
    torch = None
    if use_cuda:
        try:
            import torch as pytorch
            hardware["torch"] = pytorch.__version__
            if pytorch.cuda.is_available():
                torch = pytorch
                hardware["accelerator"] = torch.cuda.get_device_name(0)
        except (ImportError, RuntimeError, OSError) as error:
            hardware["cuda_fallback_reason"] = str(error)
    print(f"已识别加速器：{hardware['accelerator'] or 'CPU'}。开始构建同一套正文特征。", flush=True)

    bundle = {
        "vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=3,
                                     max_features=110_000, lowercase=False, sublinear_tf=True,
                                     dtype=np.float32),
        "scaler": StandardScaler(), "heads": {}, "thresholds": {}, "sklearn": sklearn.__version__,
    }
    features_started = time.monotonic()
    train_matrix = first.build_matrix(bundle, [row[2] for row in training], fit=True)
    synthetic_matrix = first.build_matrix(bundle, [row[1] for row in ordering])
    full_matrix = sparse.vstack((train_matrix, synthetic_matrix), format="csr")
    del synthetic_matrix
    val_matrix = first.build_matrix(bundle, [row[2] for row in validation])
    feature_seconds = round(time.monotonic() - features_started, 3)
    y_real = np.asarray([row[3] == "ai" for row in training], dtype=bool)
    y_full = np.concatenate((y_real, np.ones(len(ordering), dtype=bool)))
    y_val = np.asarray([row[3] == "ai" for row in validation], dtype=bool)
    dsh_val = np.asarray([groups[row[1]]["dsh"] for row in validation], dtype=bool)
    negative_val = dsh_val & ~y_val
    negative_count = int(negative_val.sum())
    imbalance = min(6.0, np.sqrt((~y_real).sum() / y_real.sum()))
    reliability = np.asarray([0.55 if row[4].startswith("nickname_rule:")
                              else 0.65 if row[4].startswith("style_rule:") else 1.0
                              for row in training])
    weights_real = np.where(y_real, imbalance, 1) * reliability
    n_real = len(training)

    print("开始复跑纯 B 站 CPU 对照，用于校验 CUDA 求解。", flush=True)
    cpu_anchor = fit_cpu(train_matrix, y_real, weights_real)
    hardware["cpu_anchor_fit"] = cpu_anchor[1]

    def fit_arms(accelerated, anchor):
        heads = {}
        models = {}
        val_scores = {}
        for name, label, count, effective in ARMS:
            relative_weight = 0.0 if not count else first.SYNTHETIC_WEIGHT * effective / count
            weights = np.concatenate((weights_real, np.full(count, imbalance * relative_weight)))
            end = n_real + count
            print(f"训练 {label} · 合成 {count:,} 条。", flush=True)
            if not count:
                head, fit_record = anchor
            elif accelerated:
                head, fit_record = fit_cuda(torch, full_matrix[:end], y_full[:end], weights)
            else:
                head, fit_record = fit_cpu(full_matrix[:end], y_full[:end], weights)
            prefix = hashlib.sha256()
            for sample_id, _ in ordering[:count]:
                prefix.update((sample_id + "\n").encode())
            models[name] = dict(label=label, synthetic_rows=count, deepseek_rows=count * 2 // 3,
                                other_rows=count // 3, train_rows=end,
                                train_synthetic_share=count / end, relative_synthetic_weight=relative_weight,
                                synthetic_weight_equivalent_rows=effective, sampled_ids_sha256=prefix.hexdigest(),
                                fit=fit_record)
            heads[name] = head
            val_scores[name] = score(val_matrix, head)
            print(f"完成 {label} · {fit_record['seconds']:.2f} 秒。", flush=True)
        return heads, models, val_scores

    if torch is not None:
        try:
            cuda_anchor = fit_cuda(torch, train_matrix, y_real, weights_real)
            difference = np.abs(score(val_matrix, cuda_anchor[0]) - score(val_matrix, cpu_anchor[0]))
            hardware["cuda_anchor_fit"] = cuda_anchor[1]
            hardware["cpu_cuda_validation_score_parity"] = {
                "mae": float(difference.mean()), "p99_abs": float(np.quantile(difference, 0.99)),
                "max_abs": float(difference.max()),
            }
            if hardware["cpu_cuda_validation_score_parity"]["p99_abs"] > 0.012:
                raise RuntimeError("CUDA 与 CPU 基线的验证集分数差异超出预设容差。")
            hardware["training_backend"] = f"{hardware['accelerator']} CUDA"
            heads, models, val_scores = fit_arms(True, cuda_anchor)
        except (RuntimeError, ValueError, NotImplementedError) as error:
            hardware["cuda_fallback_reason"] = str(error)
            hardware["training_backend"] = "sklearn/liblinear CPU"
            print(f"CUDA 对照核验未通过，统一用 CPU 复跑七组：{error}", flush=True)
            heads, models, val_scores = fit_arms(False, cpu_anchor)
    else:
        heads, models, val_scores = fit_arms(False, cpu_anchor)

    baseline_scores = val_scores["bili_only"]
    fixed = first.thresholds(y_val, baseline_scores)["f05"]
    baseline_budget = int((negative_val & (baseline_scores >= fixed)).sum())
    strict_budget = math.floor(negative_count * 0.05)
    model_thresholds = {}
    for name, values in val_scores.items():
        limits = first.thresholds(y_val, values)
        limits["dsh_matched"] = matched_threshold(values, negative_val, baseline_budget)
        limits["dsh_fpr05"] = matched_threshold(values, negative_val, strict_budget)
        models[name]["thresholds"] = limits
        models[name]["validation"] = first.evaluate(validation, groups, values, limits)
        model_thresholds[name] = limits
    preferred = max(models, key=lambda name: (
        models[name]["validation"]["dsh"]["points"]["dsh_matched"]["recall"],
        models[name]["validation"]["dsh_gemini_only"]["average_precision"],
        models[name]["validation"]["overall"]["average_precision"],
    ))
    print(f"阈值与验证集优先配置已锁定：{models[preferred]['label']}。开始留出测试。", flush=True)

    test_matrix = first.build_matrix(bundle, [row[2] for row in test])
    probe_matrix = first.build_matrix(bundle, [row["text"] for row in probes])
    y_test = np.asarray([row[3] == "ai" for row in test], dtype=bool)
    dsh_test = np.asarray([groups[row[1]]["dsh"] for row in test], dtype=bool)
    test_videos = np.asarray([row[1] for row in test])
    test_scores = {name: score(test_matrix, head) for name, head in heads.items()}
    for name, model in models.items():
        cutoffs = model_thresholds[name]
        ranked = test_scores[name]
        model["test"] = first.evaluate(test, groups, ranked, cutoffs)
        probe_scores = score(probe_matrix, heads[name])
        model["user_confirmed_probe"] = {
            cut: first.point(np.ones(len(probes), dtype=bool), probe_scores, cutoffs[cut])
            for cut in ("f05", "dsh_matched", "dsh_fpr05")
        }
        model["dsh_paired_delta_95ci"] = {}
        for cut in ("f05", "dsh_matched", "dsh_fpr05"):
            anchor_alert = test_scores["bili_only"] >= model_thresholds["bili_only"][cut]
            variant_alert = ranked >= cutoffs[cut]
            model["dsh_paired_delta_95ci"][cut] = paired_video_intervals(
                test_videos[dsh_test], y_test[dsh_test], anchor_alert[dsh_test], variant_alert[dsh_test]
            )

    output.mkdir(parents=True, exist_ok=True)
    bundle.update(heads=heads, thresholds=model_thresholds, preferred_variant=preferred)
    joblib.dump(bundle, output / MODEL_FILE, compress=3)
    loaded = joblib.load(output / MODEL_FILE)
    np.testing.assert_allclose(predict(loaded, [row[2] for row in test[:24]]),
                               test_scores[preferred][:24], rtol=2e-6, atol=2e-7)
    with gzip.open(output / PREDICTION_FILE, "wt", encoding="utf-8") as stream:
        for part, data, predictions in (
            ("validation", validation, val_scores), ("test", test, test_scores)
        ):
            for index, row in enumerate(data):
                sample = dict(split=part, sample_id=row[0], video_id=row[1], reference=row[3],
                              source=row[4], level=row[5], dsh=groups[row[1]]["dsh"],
                              scores={name: float(values[index]) for name, values in predictions.items()})
                stream.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "raw_rows": split["raw_rows"], "split_counts": split["counts"], "review_rows": split["review_rows"],
        "split_sha256": SPLIT_SHA, "corpus_sha256": CORPUS_SHA, "eval_fingerprints": held,
        "synthetic_inputs": sources, "sample_seed": SEED,
        "validation_dsh_negatives": negative_count, "validation_dsh_matched_fp_budget": baseline_budget,
        "validation_dsh_5pct_fp_budget": strict_budget,
        "preferred_by_validation": preferred, "models": models, "hardware": hardware,
        "feature_count": full_matrix.shape[1], "feature_seconds": feature_seconds,
        "elapsed_seconds": round(time.monotonic() - started, 3), "sklearn": sklearn.__version__,
    }
    (output / RESULT_FILE).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with ZipFile(output / "bili-round02-artifacts.zip", "w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
        for name in (RESULT_FILE, MODEL_FILE, PREDICTION_FILE):
            archive.write(output / name, name)
        archive.write(split_path, split_path.name)
        archive.write(root / "delivery.json", "synthetic-delivery.json")
        archive.write(Path(first.__file__), "bili_round01_train.py")
        archive.write(Path(__file__), "bili_round02_train.py")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    config = parser.parse_args()
    report = run(config.data, config.output, use_cuda=not config.cpu)
    print(json.dumps({"preferred": report["preferred_by_validation"], "seconds": report["elapsed_seconds"],
                      "backend": report["hardware"]["training_backend"]}, ensure_ascii=False))
