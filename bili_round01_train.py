"""Train and evaluate a text-only Bilibili comment classifier in Colab."""

import argparse
import gzip
import hashlib
import json
import re
import time
import unicodedata
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
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler


TIMESTAMP = re.compile(r"\d{1,2}:\d{2}")
FACES = re.compile(r"\[[^\s\[\]]{1,15}\]")
BREAKS = re.compile(r"[,，。.!！?？;；\n]+")
MODEL_FILE = "bili-round01-models.joblib"
RESULT_FILE = "bili-round01-metrics.json"
PREDICTION_FILE = "bili-round01-test-predictions.jsonl.gz"
SYNTHETIC_WEIGHT = 0.35


def prepared(text):
    text = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    return text[:160] + " " + text[-96:] if len(text) > 256 else text


def body_key(text):
    normalized = "".join(unicodedata.normalize("NFKC", text).casefold().split())
    letters = re.sub(r"[^\w]+|_", "", normalized)
    return letters if len(letters) >= 10 else normalized


def style_vector(text):
    length = max(1, len(text))
    parts = [part for part in BREAKS.split(text) if part.strip()]
    return (np.log1p(length), np.log1p(len(parts)), np.log1p(max(map(len, parts), default=0)),
            sum(char.isdigit() for char in text) / length,
            sum(char in ",，;；:：" for char in text) / length,
            sum(char in "!！?？~～" for char in text) / length,
            sum(char in "\"'“”‘’「」『』" for char in text) / length,
            sum(char in "—-…" for char in text) / length,
            np.log1p(len(FACES.findall(text))), np.log1p(len(TIMESTAMP.findall(text))))


def build_matrix(bundle, texts, fit=False):
    bodies = [prepared(text) for text in texts]
    styles = np.asarray([style_vector(body) for body in bodies], dtype=np.float32)
    vectorizer = bundle["vectorizer"]
    scaler = bundle["scaler"]
    if fit:
        words = vectorizer.fit_transform(bodies)
        styles = scaler.fit_transform(styles)
    else:
        words = vectorizer.transform(bodies)
        styles = scaler.transform(styles)
    np.clip(styles, -8.0, 8.0, out=styles)
    return sparse.hstack((words, sparse.csr_matrix(styles * 0.12)), format="csr")


def predict(bundle, texts, variant=None):
    if not texts:
        return np.array([], dtype=float)
    classifier = bundle["classifiers"][variant or bundle["preferred_variant"]]
    return expit(classifier.decision_function(build_matrix(bundle, texts)))


def thresholds(y, scores):
    precision, recall, values = precision_recall_curve(y, scores)
    precision = precision[:-1]
    recall = recall[:-1]
    ranked = np.sort(scores)
    alerts = len(scores) - np.searchsorted(ranked, values)
    f05 = 1.25 * precision * recall / np.maximum(0.25 * precision + recall, 1e-12)
    selected = {"f05": float(values[np.flatnonzero(f05 == f05.max())[-1]])}
    for floor in (0.95, 0.99):
        valid = np.flatnonzero((precision >= floor) & (alerts >= 30))
        index = max(valid, key=lambda i: (recall[i], precision[i], values[i])) if len(valid) else None
        selected[f"p{int(floor * 100)}"] = float(values[index]) if index is not None else None
    return selected


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def point(y, scores, threshold):
    if threshold is None:
        return None
    alert = scores >= threshold
    tp = int((y & alert).sum())
    fp = int((~y & alert).sum())
    fn = int((y & ~alert).sum())
    tn = int((~y & ~alert).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                false_positive_rate=ratio(fp, fp + tn), f1=ratio(2 * tp, 2 * tp + fp + fn),
                f05=ratio(1.25 * tp, 1.25 * tp + fp + 0.25 * fn),
                accuracy=ratio(tp + tn, tp + fp + fn + tn))


def confidence_intervals(videos, y, scores, threshold):
    _, group = np.unique(videos, return_inverse=True)
    size = int(group.max()) + 1
    alert = scores >= threshold
    categories = (y & alert, ~y & alert, y & ~alert, ~y & ~alert)
    group_counts = np.asarray([np.bincount(group[mask], minlength=size) for mask in categories]).T
    draws = np.random.default_rng(2291).multinomial(size, np.full(size, 1 / size), size=500)
    tp, fp, fn, tn = (draws @ group_counts).T
    pairs = ((tp, tp + fp), (tp, tp + fn), (fp, fp + tn))
    estimates = {}
    for name, (numerator, denominator) in zip(("precision", "recall", "false_positive_rate"), pairs):
        mask = denominator > 0
        estimates[name] = np.quantile(numerator[mask] / denominator[mask], [0.025, 0.975]).tolist() if mask.any() else None
    return estimates


def evaluate(rows, group_map, scores, selected, cluster_intervals=False):
    y = np.asarray([row[3] == "ai" for row in rows], dtype=bool)
    videos = np.asarray([row[1] for row in rows])
    dsh = np.asarray([group_map[video]["dsh"] for video in videos])
    gemini = np.asarray([row[4].startswith("google/") for row in rows])
    reply = np.asarray([row[5] == "reply" for row in rows])
    hard = np.asarray([len(row[2]) >= 80 or TIMESTAMP.search(row[2]) is not None for row in rows])
    scopes = {
        "overall": np.ones(len(y), dtype=bool), "dsh": dsh, "non_dsh": ~dsh,
        "dsh_gemini_only": dsh & gemini, "dsh_top": dsh & ~reply, "dsh_reply": dsh & reply,
        "dsh_hard_human_like": dsh & ~y & hard,
    }
    results = {}
    for name, mask in scopes.items():
        labels = y[mask]
        scoped_scores = scores[mask]
        positives = int(labels.sum())
        count = int(mask.sum())
        result = dict(rows=count, videos=len(set(videos[mask])), positives=positives,
                      reference_ai_rate=ratio(positives, count),
                      average_precision=float(average_precision_score(labels, scoped_scores))
                      if 0 < positives < count else None,
                      points={key: point(labels, scoped_scores, value) for key, value in selected.items()})
        if cluster_intervals and count and name in {"overall", "dsh", "dsh_gemini_only"}:
            result["f05_video_bootstrap_95ci"] = confidence_intervals(videos[mask], labels, scoped_scores, selected["f05"])
        results[name] = result
    return results


def load_inputs(root):
    manifest = json.loads((root / "bili-round01-split.json").read_text(encoding="utf-8"))
    dataset = root / "bili-round01-corpus.jsonl.gz"
    with dataset.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != manifest["corpus_sha256"]:
            raise ValueError("Colab upload hash does not match the frozen corpus.")
    group_map = manifest["video_assignments"]
    selected = {"train": [], "validation": [], "test": []}
    keys = set()
    all_keys = set()
    ids = set()
    counts = Counter()
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sample_id, video, text, label, _, _, _, reason = row
            if sample_id in ids:
                raise ValueError("Duplicated sample ID.")
            ids.add(sample_id)
            key = body_key(text)
            all_keys.add(key)
            split = group_map[video]["split"]
            if reason:
                counts[reason] += 1
                continue
            if label not in {"ai", "human"} or key in keys:
                raise ValueError("Reference label or cross-split duplicate check failed.")
            keys.add(key)
            selected[split].append(row)
    assert len(ids) == manifest["raw_rows"] == 200_370
    for split, rows in selected.items():
        label_counts = Counter(row[3] for row in rows)
        assert label_counts == {key: manifest["counts"][split][key] for key in ("ai", "human")}
    assert sum(map(len, selected.values())) + sum(counts.values()) == len(ids)
    return manifest, selected, all_keys


def load_synthetic(path, real_keys):
    if path is None:
        return [], {"admitted": 0, "excluded_duplicate": 0, "model_counts": {}, "format_counts": {}}
    seen = set()
    admitted = []
    excluded = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["label"] != "ai" or row["split"] != "train" or row["source"] != "zcode_deepseek":
                raise ValueError("Synthetic sample provenance or split is invalid.")
            key = body_key(row["text"])
            if key in seen or key in real_keys:
                excluded += 1
                continue
            seen.add(key)
            admitted.append(row)
    if not admitted:
        raise ValueError("No unique DeepSeek positives were admitted.")
    return admitted, dict(admitted=len(admitted), excluded_duplicate=excluded,
                          model_counts=dict(Counter(row["model_id"] for row in admitted)),
                          format_counts=dict(Counter(row["format"] for row in admitted)))


def train(root, output, synthetic_path):
    started = time.monotonic()
    manifest, selected, real_keys = load_inputs(root)
    group_map = manifest["video_assignments"]
    probe_path = root / "bili-round01-user-confirmed.jsonl"
    probes = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines()]
    if len(probes) != 17 or any(row["label"] != "ai" for row in probes):
        raise ValueError("The separate user-confirmed probe is invalid.")
    train_keys = {body_key(row[2]) for row in selected["train"]}
    probe_keys = {body_key(row["text"]) for row in probes}
    if probe_keys & train_keys:
        raise ValueError("A user-confirmed probe comment appears verbatim in training.")
    synthetic, synthetic_data = load_synthetic(synthetic_path, real_keys | probe_keys)
    training = selected["train"]
    validation = selected["validation"]
    test = selected["test"]
    print(f"已核对 {manifest['raw_rows']:,} 条输入。训练 / 验证 / 测试可用条数："
          f"{len(training):,} / {len(validation):,} / {len(test):,}。", flush=True)
    bundle = {"vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=3,
                                               max_features=110_000, lowercase=False, sublinear_tf=True,
                                               dtype=np.float32), "scaler": StandardScaler(),
              "classifiers": {}, "thresholds": {}, "sklearn": sklearn.__version__}
    x_train = build_matrix(bundle, [row[2] for row in training], fit=True)
    x_val = build_matrix(bundle, [row[2] for row in validation])
    x_test = build_matrix(bundle, [row[2] for row in test])
    x_probe = build_matrix(bundle, [row["text"] for row in probes])
    y = np.asarray([row[3] == "ai" for row in training], dtype=bool)
    y_val = np.asarray([row[3] == "ai" for row in validation], dtype=bool)
    imbalance = min(6.0, np.sqrt((~y).sum() / y.sum()))
    reliability = np.asarray([0.55 if row[4].startswith("nickname_rule:") else
                              0.65 if row[4].startswith("style_rule:") else 1.0 for row in training])
    weight = np.where(y, imbalance, 1) * reliability
    variants = {"bili_only": (x_train, y, weight)}
    if synthetic:
        x_synthetic = build_matrix(bundle, [row["text"] for row in synthetic])
        variants["bili_plus_deepseek"] = (
            sparse.vstack((x_train, x_synthetic), format="csr"),
            np.concatenate((y, np.ones(len(synthetic), dtype=bool))),
            np.concatenate((weight, np.full(len(synthetic), imbalance * SYNTHETIC_WEIGHT))),
        )

    models = {}
    scores = {}
    for name, (x_fit, y_fit, weights) in variants.items():
        fit_started = time.monotonic()
        print(f"开始训练 {name}，样本 {len(y_fit):,} 条。", flush=True)
        classifier = LogisticRegression(C=2.0, max_iter=300, solver="liblinear", random_state=20261122)
        classifier.fit(x_fit, y_fit, sample_weight=weights)
        val_scores = expit(classifier.decision_function(x_val))
        selected_thresholds = thresholds(y_val, val_scores)
        bundle["classifiers"][name] = classifier
        bundle["thresholds"][name] = selected_thresholds
        models[name] = dict(train_rows=len(y_fit), train_positives=int(y_fit.sum()),
                            validation=evaluate(validation, group_map, val_scores, selected_thresholds),
                            thresholds=selected_thresholds, fit_seconds=round(time.monotonic() - fit_started, 2))

    preferred = max(models, key=lambda name: (
        models[name]["validation"]["dsh_gemini_only"]["average_precision"],
        models[name]["validation"]["overall"]["average_precision"]))
    bundle["preferred_variant"] = preferred
    for name, classifier in bundle["classifiers"].items():
        selected_thresholds = bundle["thresholds"][name]
        test_scores = expit(classifier.decision_function(x_test))
        probe_scores = expit(classifier.decision_function(x_probe))
        scores[name] = test_scores
        models[name]["test"] = evaluate(test, group_map, test_scores, selected_thresholds, cluster_intervals=True)
        models[name]["user_confirmed_probe"] = dict(rows=len(probes), positives=len(probes),
            points={key: point(np.ones(len(probes), dtype=bool), probe_scores, threshold)
                    for key, threshold in selected_thresholds.items()},
            scores={row["id"]: float(score) for row, score in zip(probes, probe_scores)})

    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / MODEL_FILE, compress=3)
    reloaded = joblib.load(output / MODEL_FILE)
    np.testing.assert_allclose(predict(reloaded, [row[2] for row in test[:16]]), scores[preferred][:16])
    report = dict(raw_rows=manifest["raw_rows"], split_counts=manifest["counts"],
                  split_seed=manifest["seed"], corpus_sha256=manifest["corpus_sha256"],
                  review_rows=manifest["review_rows"], synthetic=synthetic_data,
                  synthetic_weight=SYNTHETIC_WEIGHT, sklearn=sklearn.__version__,
                  feature_count=x_train.shape[1], preferred_by_validation_dsh_gemini_ap=preferred,
                  models=models, elapsed_seconds=round(time.monotonic() - started, 2))
    (output / RESULT_FILE).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with gzip.open(output / PREDICTION_FILE, "wt", encoding="utf-8") as stream:
        for i, row in enumerate(test):
            prediction = dict(sample_id=row[0], video_id=row[1], reference=row[3], source=row[4],
                              level=row[5], dsh=group_map[row[1]]["dsh"],
                              scores={name: float(values[i]) for name, values in scores.items()})
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    with ZipFile(output / "bili-round01-artifacts.zip", "w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
        for name in (RESULT_FILE, MODEL_FILE, PREDICTION_FILE):
            archive.write(output / name, name)
        archive.write(root / "bili-round01-split.json", "bili-round01-split.json")
        archive.write(Path(__file__), "bili_round01_train.py")
    print(f"Colab 已完成训练和留出测试 · {report['elapsed_seconds']} 秒。", flush=True)
    return report


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data", type=Path, required=True)
    args.add_argument("--output", type=Path, required=True)
    args.add_argument("--synthetic", type=Path)
    options = args.parse_args()
    train(options.data, options.output, options.synthetic)
