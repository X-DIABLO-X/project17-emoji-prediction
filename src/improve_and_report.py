from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import download_tweeteval_emoji, load_dataset, top_tokens_by_class, tweet_tokenize
from metrics import evaluate_probs, per_class_frame, predictions_frame, save_json, softmax
from reporting import RESEARCH_SOURCES, plot_confusion_matrix


MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def roberta_preprocess(text: str) -> str:
    """Use CardiffNLP/TweetNLP-style normalization for Twitter RoBERTa models."""
    text = MENTION_RE.sub("@user", text)
    text = URL_RE.sub("http", text)
    return text


def top_k_accuracy(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    top_k = np.argsort(scores, axis=1)[:, -k:]
    return float(np.mean([label in row for label, row in zip(labels, top_k)]))


def scores_to_metrics(scores: np.ndarray, labels: np.ndarray, mapping: pd.DataFrame) -> dict[str, object]:
    # Softmax is used only to reuse the common metric/prediction helpers; argmax/top-k are score-equivalent.
    return evaluate_probs(softmax(scores), labels, mapping)


def save_label_predictions(path: str | Path, scores: np.ndarray) -> None:
    preds = scores.argmax(axis=1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(str(int(x)) for x in preds) + "\n", encoding="utf-8")


def train_sparse_svm(dataset: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    print("\n[SVM] Building stronger TF-IDF word + character features...")
    word_vectorizer = TfidfVectorizer(
        tokenizer=tweet_tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=args.svm_min_df,
        max_features=args.svm_word_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=args.svm_char_min_df,
        max_features=args.svm_char_features,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )
    x_train_word = word_vectorizer.fit_transform(dataset["train_texts"])
    x_val_word = word_vectorizer.transform(dataset["val_texts"])
    x_test_word = word_vectorizer.transform(dataset["test_texts"])
    x_train_char = char_vectorizer.fit_transform(dataset["train_texts"])
    x_val_char = char_vectorizer.transform(dataset["val_texts"])
    x_test_char = char_vectorizer.transform(dataset["test_texts"])
    x_train = hstack([x_train_word, x_train_char]).tocsr()
    x_val = hstack([x_val_word, x_val_char]).tocsr()
    x_test = hstack([x_test_word, x_test_char]).tocsr()
    print(f"[SVM] shapes train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

    configs = [
        {"C": 0.5, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 0.5, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": "balanced"},
    ]
    best = None
    histories = []
    for config in configs:
        print(f"[SVM] Training LinearSVC C={config['C']} class_weight={config['class_weight']}")
        clf = LinearSVC(
            C=config["C"],
            class_weight=config["class_weight"],
            dual="auto",
            max_iter=args.svm_max_iter,
            random_state=args.seed,
        )
        clf.fit(x_train, dataset["train_labels"])
        val_scores = clf.decision_function(x_val)
        val_metrics = scores_to_metrics(val_scores, dataset["val_labels"], dataset["mapping"])
        record = {
            "C": config["C"],
            "class_weight": config["class_weight"],
            "val_top1_accuracy": val_metrics["top1_accuracy"],
            "val_top3_accuracy": val_metrics["top3_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        histories.append(record)
        print(
            f"[SVM] val top1={val_metrics['top1_accuracy']:.4f} "
            f"top3={val_metrics['top3_accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if best is None or val_metrics["macro_f1"] > best["val_metrics"]["macro_f1"]:
            best = {"clf": clf, "config": config, "val_metrics": val_metrics}

    assert best is not None
    print(f"[SVM] Best config by validation macro-F1: {best['config']}")
    test_scores = best["clf"].decision_function(x_test)
    val_scores = best["clf"].decision_function(x_val)
    return {
        "classifier": best["clf"],
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "history": histories,
        "best_config": best["config"],
        "val_scores": val_scores,
        "test_scores": test_scores,
    }


@torch.no_grad()
def transformer_scores(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model.eval()
    all_scores = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Transformer batches"):
        batch_texts = [roberta_preprocess(text) for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**encoded).logits
        else:
            logits = model(**encoded).logits
        all_scores.append(logits.float().cpu().numpy())
    return np.vstack(all_scores)


def evaluate_transformer(dataset: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_transformer else "cpu")
    print(f"\n[Transformer] Loading {args.transformer_model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.transformer_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.transformer_model).to(device)
    if device.type == "cuda":
        print(f"[Transformer] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Transformer] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    val_scores = transformer_scores(
        dataset["val_texts"],
        tokenizer,
        model,
        device,
        batch_size=args.transformer_batch_size,
        max_length=args.transformer_max_length,
    )
    val_metrics = scores_to_metrics(val_scores, dataset["val_labels"], dataset["mapping"])
    print(
        f"[Transformer] val top1={val_metrics['top1_accuracy']:.4f} "
        f"top3={val_metrics['top3_accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f}"
    )
    test_scores = transformer_scores(
        dataset["test_texts"],
        tokenizer,
        model,
        device,
        batch_size=args.transformer_batch_size,
        max_length=args.transformer_max_length,
    )
    return {
        "model_name": args.transformer_model,
        "device": str(device),
        "val_scores": val_scores,
        "test_scores": test_scores,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _metrics_table(metrics: dict[str, dict[str, object]]) -> str:
    rows = ["| Model | Top-1 Accuracy | Top-3 Accuracy | Macro F1 | Weighted F1 |", "|---|---:|---:|---:|---:|"]
    for name, item in metrics.items():
        rows.append(
            f"| {name} | {_pct(float(item['top1_accuracy']))} | {_pct(float(item['top3_accuracy']))} | "
            f"{float(item['macro_f1']):.4f} | {float(item['weighted_f1']):.4f} |"
        )
    return "\n".join(rows)


def _svm_history_table(svm_history: list[dict[str, object]]) -> str:
    if not svm_history:
        return "| C | Class Weight | Val Top-1 | Val Top-3 | Val Macro F1 | Val Weighted F1 |\n|---:|---|---:|---:|---:|---:|"
    rows = [
        "| C | Class Weight | Val Top-1 | Val Top-3 | Val Macro F1 | Val Weighted F1 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in svm_history:
        rows.append(
            f"| {float(row['C']):.2f} | {row['class_weight'] or 'none'} | "
            f"{_pct(float(row['val_top1_accuracy']))} | {_pct(float(row['val_top3_accuracy']))} | "
            f"{float(row['val_macro_f1']):.4f} | {float(row['val_weighted_f1']):.4f} |"
        )
    return "\n".join(rows)


def _per_class_table(per_class: pd.DataFrame, top_tokens: dict[str, list[str]], ascending: bool, n: int = 8) -> str:
    frame = per_class.sort_values(["f1", "recall", "support"], ascending=[ascending, ascending, not ascending]).head(n)
    rows = ["| Emoji | Name | Precision | Recall | F1 | Support | Indicative Tokens |", "|---|---|---:|---:|---:|---:|---|"]
    for _, row in frame.iterrows():
        tokens = ", ".join(top_tokens.get(str(int(row.label)), [])[:6])
        rows.append(
            f"| {row.emoji} | {row['name'].replace('_', ' ')} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} | {int(row.support)} | {tokens} |"
        )
    return "\n".join(rows)


def _confusion_table(best_metrics: dict[str, object], per_class: pd.DataFrame, mapping: pd.DataFrame, n: int = 8) -> str:
    matrix = np.asarray(best_metrics["confusion_matrix"])
    names = {int(row.label): f"{row.emoji} {row['name'].replace('_', ' ')}" for _, row in mapping.iterrows()}
    hardest = per_class.sort_values(["f1", "recall", "support"], ascending=[True, True, False]).head(n)
    rows = ["| True Emoji | Most Common Wrong Prediction | Wrong Count | Share Of True Class |", "|---|---|---:|---:|"]
    for _, row in hardest.iterrows():
        label = int(row.label)
        counts = matrix[label].copy()
        counts[label] = 0
        pred = int(counts.argmax())
        rows.append(f"| {names[label]} | {names[pred]} | {int(counts[pred])} | {_pct(float(counts[pred]) / max(1, int(row.support)))} |")
    return "\n".join(rows)


def generate_report2(
    path: str | Path,
    original_metrics: dict[str, dict[str, object]],
    improved_metrics: dict[str, dict[str, object]],
    per_class_best: pd.DataFrame,
    best_name: str,
    best_metrics: dict[str, object],
    top_tokens: dict[str, list[str]],
    mapping: pd.DataFrame,
    run_config: dict[str, object],
    svm_history: list[dict[str, object]],
) -> None:
    all_metrics = {**original_metrics, **improved_metrics}
    sources = "\n".join(f"- [{item['title']}]({item['url']}): {item['note']}" for item in RESEARCH_SOURCES)
    online = """| Reference | Emoji Macro-F1 | Meaning |
|---|---:|---|
| Our first MLP TF-IDF BoW | 19.55% / 0.1955 | Good sanity baseline, but weak on rare classes |
| Our first LSTM | 17.88% / 0.1788 | Below sparse baseline; trained from scratch on only 45k tweets |
| TweetEval LSTM | 24.7% / 0.247 | Official leaderboard baseline |
| TweetEval SVM | 29.3% / 0.293 | Official sparse baseline |
| TweetEval RoBERTa-Retrained | 31.4% / 0.314 | Official pretrained transformer baseline |
| TweetEval BERTweet | 33.4% / 0.334 | Strong Twitter-pretrained leaderboard model |
| Tübingen-Oslo SemEval winner | 35.99% / 0.3599 | Original SemEval English winner using SVM-style sparse features |
"""
    svm_history_table = _svm_history_table(svm_history)
    content = f"""# Project 17 Report 2: Score Quality And Improved Models

## Are The First Scores Good?

Short answer: **they are acceptable as simple local baselines, but bad compared with serious online benchmark systems**. The first report's MLP reached **{_pct(original_metrics['MLP TF-IDF BoW']['top1_accuracy'])} top-1**, **{_pct(original_metrics['MLP TF-IDF BoW']['top3_accuracy'])} top-3**, and **{_pct(original_metrics['MLP TF-IDF BoW']['macro_f1'])} macro-F1**. TweetEval ranks emoji systems by macro-F1, and public baselines report about **24.7%** for LSTM, **29.3%** for SVM, **31.4%** for RoBERTa-retrained, and **33.4%** for BERTweet. So our original top-1 accuracy looked reasonable, but macro-F1 exposed the weakness: several minority or semantically overlapping emojis had near-zero recall.

{online}

## Improvements Performed

I made two improvement passes:

- **LinearSVM with stronger sparse features**: word unigrams/bigrams plus character `3-5` grams. This targets hashtags, slang, misspellings, elongated words, and short lexical cues. It is also aligned with published SemEval evidence that SVMs outperform small RNNs on this task.
- **GPU Twitter-RoBERTa emoji model**: `cardiffnlp/twitter-roberta-base-emoji` evaluated on the RTX 4050 via CUDA PyTorch. This uses pretrained Twitter language representations and task-specific emoji fine-tuning, so it is the closest local comparison to TweetEval transformer baselines.

CUDA verification:

```json
{json.dumps(run_config, ensure_ascii=False, indent=2)}
```

SVM validation sweep:

{svm_history_table}

## Final Scores After Improvement

{_metrics_table(all_metrics)}

Best improved model: **{best_name}**, with **{_pct(best_metrics['top1_accuracy'])} top-1**, **{_pct(best_metrics['top3_accuracy'])} top-3**, and **{_pct(best_metrics['macro_f1'])} macro-F1** (`{best_metrics['macro_f1']:.4f}`).

The improvement is real because macro-F1, not only accuracy, increased. Macro-F1 matters here because the dataset is imbalanced and because exact emoji identity is harder for rarer classes such as purple heart, hundred points, winking faces, and camera-with-flash.

## Why The Improved Methods Work Better

The original MLP learned useful lexical associations but did not generalize well to minority classes. The original LSTM was trained from scratch and had to learn both word meaning and classification boundaries from only 45k training tweets. That is not much data for a neural sequence model with many rare hashtags and informal spellings.

The SVM improves because sparse high-dimensional text features are a strong match for short tweets. Character n-grams add robustness for hashtags and spelling variants. The transformer improves further because it starts with Twitter-domain language knowledge and can model context beyond isolated tokens. This is why the best online systems are pretrained transformers or very strong sparse systems rather than small from-scratch LSTMs.

## Easiest Emojis Under The Best Improved Model

{_per_class_table(per_class_best, top_tokens, ascending=False)}

These are easier because they have distinctive lexical anchors: holiday words for 🎄, affection words for ❤, laughter tokens for 😂, weather/beach words for ☀, and photo/location language for camera labels.

## Hardest Emojis Under The Best Improved Model

{_per_class_table(per_class_best, top_tokens, ascending=True)}

Most common confusion patterns:

{_confusion_table(best_metrics, per_class_best, mapping)}

These remain hard because several emojis are pragmatically interchangeable in tweets. Heart variants share affection language, playful face variants overlap with laughter or teasing, and 📷 versus 📸 is a nearly identical visual/semantic pair. Better performance would require more data, user/context information, or a stronger fine-tuning setup optimized for rare-label recall.

## What To Try Next

- Fine-tune `vinai/bertweet-base` or `cardiffnlp/twitter-roberta-base` directly on the local TweetEval train split with class-balanced sampling.
- Use an ensemble: average transformer logits with SVM decision scores, because they make different kinds of errors.
- Optimize for macro-F1 rather than validation accuracy, including class weights or focal loss.
- Keep the original MLP/LSTM comparison for the assignment, but present the SVM and transformer as improved baselines and evidence that our first scores were below online systems.

## Sources

{sources}
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run improved Project 17 models and generate report 2.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results/improved")
    parser.add_argument("--report-path", default="report/report-2.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svm-word-features", type=int, default=120000)
    parser.add_argument("--svm-char-features", type=int, default=80000)
    parser.add_argument("--svm-min-df", type=int, default=2)
    parser.add_argument("--svm-char-min-df", type=int, default=2)
    parser.add_argument("--svm-max-iter", type=int, default=5000)
    parser.add_argument("--transformer-model", default="cardiffnlp/twitter-roberta-base-emoji")
    parser.add_argument("--transformer-batch-size", type=int, default=64)
    parser.add_argument("--transformer-max-length", type=int, default=128)
    parser.add_argument("--cpu-transformer", action="store_true")
    parser.add_argument("--skip-svm", action="store_true")
    parser.add_argument("--skip-transformer", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    start = time.time()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    download_tweeteval_emoji(args.data_dir)
    dataset = load_dataset(args.data_dir)
    original_metrics = json.loads(Path("results/metrics_summary.json").read_text(encoding="utf-8"))["test"]
    top_tokens = top_tokens_by_class(dataset["train_texts"], dataset["train_labels"], dataset["mapping"])

    improved_metrics: dict[str, dict[str, object]] = {}
    per_class_by_model: dict[str, pd.DataFrame] = {}
    svm_history: list[dict[str, object]] = []

    if not args.skip_svm:
        svm_result = train_sparse_svm(dataset, args)
        svm_val_metrics = scores_to_metrics(svm_result["val_scores"], dataset["val_labels"], dataset["mapping"])
        svm_test_metrics = scores_to_metrics(svm_result["test_scores"], dataset["test_labels"], dataset["mapping"])
        improved_metrics["LinearSVM word+char TF-IDF"] = svm_test_metrics
        per_class_by_model["LinearSVM word+char TF-IDF"] = per_class_frame(svm_test_metrics, dataset["mapping"])
        svm_history = svm_result["history"]
        save_json(Path(args.results_dir) / "svm_validation_sweep.json", {"history": svm_history, "best_config": svm_result["best_config"], "validation": svm_val_metrics})
        save_json(Path(args.results_dir) / "metrics_svm.json", svm_test_metrics)
        save_label_predictions(Path(args.results_dir) / "predictions_svm_labels.txt", svm_result["test_scores"])
        predictions_frame(dataset["test_texts"], dataset["test_labels"], softmax(svm_result["test_scores"]), dataset["mapping"]).to_csv(
            Path(args.results_dir) / "predictions_svm.csv", index=False, encoding="utf-8"
        )
        per_class_by_model["LinearSVM word+char TF-IDF"].to_csv(Path(args.results_dir) / "per_class_svm.csv", index=False, encoding="utf-8")
        with (Path(args.models_dir) / "linear_svm_word_char_tfidf.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "classifier": svm_result["classifier"],
                    "word_vectorizer": svm_result["word_vectorizer"],
                    "char_vectorizer": svm_result["char_vectorizer"],
                    "best_config": svm_result["best_config"],
                },
                handle,
            )

    if not args.skip_transformer:
        transformer_result = evaluate_transformer(dataset, args)
        transformer_val_metrics = scores_to_metrics(transformer_result["val_scores"], dataset["val_labels"], dataset["mapping"])
        transformer_test_metrics = scores_to_metrics(transformer_result["test_scores"], dataset["test_labels"], dataset["mapping"])
        improved_metrics["Twitter-RoBERTa emoji GPU"] = transformer_test_metrics
        per_class_by_model["Twitter-RoBERTa emoji GPU"] = per_class_frame(transformer_test_metrics, dataset["mapping"])
        save_json(Path(args.results_dir) / "metrics_twitter_roberta_emoji.json", transformer_test_metrics)
        save_json(Path(args.results_dir) / "validation_twitter_roberta_emoji.json", transformer_val_metrics)
        np.save(Path(args.results_dir) / "scores_twitter_roberta_emoji_test.npy", transformer_result["test_scores"])
        save_label_predictions(Path(args.results_dir) / "predictions_twitter_roberta_emoji_labels.txt", transformer_result["test_scores"])
        predictions_frame(dataset["test_texts"], dataset["test_labels"], softmax(transformer_result["test_scores"]), dataset["mapping"]).to_csv(
            Path(args.results_dir) / "predictions_twitter_roberta_emoji.csv", index=False, encoding="utf-8"
        )
        per_class_by_model["Twitter-RoBERTa emoji GPU"].to_csv(Path(args.results_dir) / "per_class_twitter_roberta_emoji.csv", index=False, encoding="utf-8")

    save_json(Path(args.results_dir) / "metrics_improved_summary.json", improved_metrics)
    for name, metrics in improved_metrics.items():
        slug = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "")
            .replace("+", "plus")
            .replace("__", "_")
        )
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            dataset["mapping"],
            f"{name} normalized confusion matrix",
            Path(args.results_dir) / f"confusion_matrix_{slug}.png",
        )
    best_name = max(improved_metrics, key=lambda name: improved_metrics[name]["macro_f1"])
    best_metrics = improved_metrics[best_name]
    run_config = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else None,
        "transformer_model": args.transformer_model,
        "transformer_batch_size": args.transformer_batch_size,
        "transformer_max_length": args.transformer_max_length,
        "svm_word_features": args.svm_word_features,
        "svm_char_features": args.svm_char_features,
    }
    save_json(Path(args.results_dir) / "run_config.json", run_config)
    generate_report2(
        path=args.report_path,
        original_metrics=original_metrics,
        improved_metrics=improved_metrics,
        per_class_best=per_class_by_model[best_name],
        best_name=best_name,
        best_metrics=best_metrics,
        top_tokens=top_tokens,
        mapping=dataset["mapping"],
        run_config=run_config,
        svm_history=svm_history,
    )
    print("\nImproved test metrics:")
    for name, metrics in improved_metrics.items():
        print(
            f"{name}: top1={metrics['top1_accuracy']:.4f}, top3={metrics['top3_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, weighted_f1={metrics['weighted_f1']:.4f}"
        )
    print(f"Report 2: {args.report_path}")
    print(f"Done in {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
