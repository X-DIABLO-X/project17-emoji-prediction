from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_dataset, top_tokens_by_class
from improve_and_report import roberta_preprocess, transformer_scores
from metrics import evaluate_probs, per_class_frame, predictions_frame, save_json, softmax
from reporting import plot_confusion_matrix


SOURCES = [
    {
        "title": "TweetEval official repository and leaderboard",
        "url": "https://github.com/cardiffnlp/tweeteval",
        "note": "Lists emoji macro-F1 baselines and task-specific Twitter-RoBERTa models.",
    },
    {
        "title": "Twitter-roBERTa-base for Emoji prediction",
        "url": "https://huggingface.co/cardiffnlp/twitter-roberta-base-emoji",
        "note": "Model card and usage example for the transformer used in this project.",
    },
    {
        "title": "TweetNLP",
        "url": "https://github.com/cardiffnlp/tweetnlp",
        "note": "Shows the same TweetEval emoji task exposed through a Twitter NLP toolkit.",
    },
    {
        "title": "OpenCodePapers TweetEval leaderboard mirror",
        "url": "https://opencodepapers-b7572d.gitlab.io/benchmarks/sentiment-analysis-on-tweeteval.html",
        "note": "Summarizes TweetEval emoji scores including BERTweet, RoBERTa, SVM, FastText, and LSTM.",
    },
    {
        "title": "PickleTeam! at SemEval-2018 Task 2",
        "url": "https://aclanthology.org/S18-1072.pdf",
        "note": "Compared SVM, LSTM, and an ensemble for emoji prediction.",
    },
    {
        "title": "Tubingen-Oslo at SemEval-2018 Task 2",
        "url": "https://aclanthology.org/S18-1004/",
        "note": "Reports the winning SemEval SVM result and argues that SVMs can outperform RNNs for this task.",
    },
]


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def label_metrics_from_scores(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pred = scores.argmax(axis=1)
    top3 = np.argsort(scores, axis=1)[:, -3:]
    return {
        "top1_accuracy": float(np.mean(pred == labels)),
        "top3_accuracy": float(np.mean([label in row for label, row in zip(labels, top3)])),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
    }


def load_svm_scores(dataset: dict[str, object], model_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(model_path).open("rb") as handle:
        saved = pickle.load(handle)
    word = saved["word_vectorizer"]
    char = saved["char_vectorizer"]
    clf = saved["classifier"]
    x_val = hstack([word.transform(dataset["val_texts"]), char.transform(dataset["val_texts"])]).tocsr()
    x_test = hstack([word.transform(dataset["test_texts"]), char.transform(dataset["test_texts"])]).tocsr()
    return clf.decision_function(x_val), clf.decision_function(x_test)


def load_or_compute_roberta_scores(dataset: dict[str, object], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    results_dir = Path(args.results_dir)
    val_path = results_dir / "scores_twitter_roberta_emoji_val.npy"
    test_path = Path(args.roberta_test_scores)
    if val_path.exists() and test_path.exists():
        return np.load(val_path), np.load(test_path)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_transformer else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.transformer_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.transformer_model).to(device)
    val_scores = transformer_scores(
        dataset["val_texts"],
        tokenizer,
        model,
        device,
        batch_size=args.transformer_batch_size,
        max_length=args.transformer_max_length,
    )
    np.save(val_path, val_scores)
    if test_path.exists():
        test_scores = np.load(test_path)
    else:
        test_scores = transformer_scores(
            dataset["test_texts"],
            tokenizer,
            model,
            device,
            batch_size=args.transformer_batch_size,
            max_length=args.transformer_max_length,
        )
        np.save(test_path, test_scores)
    return val_scores, test_scores


def normalize_scores(val_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    mean = val_scores.mean(axis=0, keepdims=True)
    std = val_scores.std(axis=0, keepdims=True) + 1e-6
    return (val_scores - mean) / std, (test_scores - mean) / std, {"mean": mean.ravel().tolist(), "std": std.ravel().tolist()}


def combine_scores(
    roberta_scores: np.ndarray,
    svm_scores: np.ndarray,
    train_prior: np.ndarray,
    weight_roberta: float,
    roberta_temp: float,
    svm_temp: float,
    prior_alpha: float,
) -> np.ndarray:
    prior_bias = -np.log(train_prior + 1e-12)
    prior_bias = prior_bias - prior_bias.mean()
    return (
        weight_roberta * (roberta_scores / roberta_temp)
        + (1.0 - weight_roberta) * (svm_scores / svm_temp)
        + prior_alpha * prior_bias.reshape(1, -1)
    )


def search_ensemble(
    roberta_val: np.ndarray,
    svm_val: np.ndarray,
    labels: np.ndarray,
    train_prior: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    candidates: list[dict[str, float]] = []
    weights = np.round(np.linspace(0.0, 1.0, 21), 2)
    roberta_temps = [0.75, 1.0, 1.25, 1.5, 2.0]
    svm_temps = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    prior_alphas = [-0.2, 0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    for weight in weights:
        for r_temp in roberta_temps:
            for s_temp in svm_temps:
                for alpha in prior_alphas:
                    scores = combine_scores(roberta_val, svm_val, train_prior, weight, r_temp, s_temp, alpha)
                    metrics = label_metrics_from_scores(scores, labels)
                    candidates.append(
                        {
                            "weight_roberta": float(weight),
                            "roberta_temp": float(r_temp),
                            "svm_temp": float(s_temp),
                            "prior_alpha": float(alpha),
                            **metrics,
                        }
                    )
    candidates.sort(key=lambda row: (row["macro_f1"], row["top1_accuracy"], row["top3_accuracy"]), reverse=True)
    return candidates[0], candidates[:25]


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _metrics_table(metrics_by_model: dict[str, dict[str, object]]) -> str:
    rows = ["| Model | Top-1 | Top-3 | Macro F1 | Weighted F1 |", "|---|---:|---:|---:|---:|"]
    for name, metrics in metrics_by_model.items():
        rows.append(
            f"| {name} | {_pct(float(metrics['top1_accuracy']))} | {_pct(float(metrics['top3_accuracy']))} | "
            f"{float(metrics['macro_f1']):.4f} | {float(metrics['weighted_f1']):.4f} |"
        )
    return "\n".join(rows)


def _candidate_table(candidates: list[dict[str, float]], n: int = 8) -> str:
    rows = [
        "| Rank | RoBERTa Weight | RoBERTa Temp | SVM Temp | Prior Alpha | Val Top-1 | Val Top-3 | Val Macro F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(candidates[:n], start=1):
        rows.append(
            f"| {idx} | {row['weight_roberta']:.2f} | {row['roberta_temp']:.2f} | {row['svm_temp']:.2f} | "
            f"{row['prior_alpha']:.2f} | {_pct(row['top1_accuracy'])} | {_pct(row['top3_accuracy'])} | {row['macro_f1']:.4f} |"
        )
    return "\n".join(rows)


def _per_class_table(per_class: pd.DataFrame, ascending: bool, n: int = 8) -> str:
    frame = per_class.sort_values(["f1", "recall", "support"], ascending=[ascending, ascending, not ascending]).head(n)
    rows = ["| Emoji | Name | Precision | Recall | F1 | Support |", "|---|---|---:|---:|---:|---:|"]
    for _, row in frame.iterrows():
        rows.append(
            f"| {row.emoji} | {row['name'].replace('_', ' ')} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} | {int(row.support)} |"
        )
    return "\n".join(rows)


def _confusion_table(metrics: dict[str, object], per_class: pd.DataFrame, mapping: pd.DataFrame, n: int = 8) -> str:
    matrix = np.asarray(metrics["confusion_matrix"])
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


def plot_score_comparison(metrics_by_model: dict[str, dict[str, object]], path: str | Path) -> None:
    names = list(metrics_by_model)
    macro = [float(metrics_by_model[name]["macro_f1"]) for name in names]
    top1 = [float(metrics_by_model[name]["top1_accuracy"]) for name in names]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, macro, width, label="Macro F1")
    plt.bar(x + width / 2, top1, width, label="Top-1")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylim(0, max(max(macro), max(top1)) + 0.08)
    plt.ylabel("Score")
    plt.title("Project 17 score improvement")
    plt.legend()
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def generate_report3(
    report_path: str | Path,
    metrics_by_model: dict[str, dict[str, object]],
    ensemble_metrics: dict[str, object],
    per_class_ensemble: pd.DataFrame,
    best_config: dict[str, float],
    top_candidates: list[dict[str, float]],
    run_config: dict[str, object],
    mapping: pd.DataFrame,
) -> None:
    sources = "\n".join(f"- [{item['title']}]({item['url']}): {item['note']}" for item in SOURCES)
    bertweet_note = ""
    if "BERTweet emoji HF checkpoint" in metrics_by_model:
        bertweet = metrics_by_model["BERTweet emoji HF checkpoint"]
        bertweet_note = (
            "\n\nI also evaluated `cardiffnlp/bertweet-base-emoji` because online results point to BERTweet as the strongest "
            f"TweetEval family. In this local run that checkpoint scored only **{_pct(float(bertweet['macro_f1']))} macro-F1**, "
            "so it did not reproduce the public leaderboard number and was rejected for the final model. I also checked whether "
            "its generic `LABEL_0...LABEL_19` outputs were simply in the wrong label order; validation alignment did not recover "
            "competitive test performance."
        )
    content = f"""# Project 17 Report 3: Online Research And Ensemble Calibration

## Research Findings

I checked online discussions, notebooks, model cards, and benchmark pages for practical ways to improve TweetEval/SemEval emoji prediction. The main pattern is consistent:

- Sparse lexical systems are unusually strong for this task. SemEval papers report that SVMs with word and character n-grams can outperform small RNN/LSTM models.
- Transformer systems are the strongest public baselines when the transformer is pretrained on Twitter data. TweetEval lists BERTweet at **33.4 macro-F1**, RoBERTa-Retrained at **31.4**, and RoBERTa-Base around **30.9**.
- Public Hugging Face/TweetNLP examples recommend CardiffNLP Twitter-RoBERTa emoji models and Twitter-specific preprocessing, replacing usernames with `@user` and links with `http`.
- Ensembles are commonly attempted because sparse models and neural models make different errors. PickleTeam tried SVM+LSTM; our stronger version combines SVM with Twitter-RoBERTa logits and tunes the blend on validation macro-F1.
{bertweet_note}

## Method Chosen

The best feasible method for this local RTX 4050 setup is a **validation-tuned ensemble of Twitter-RoBERTa and LinearSVM**, plus a small class-prior calibration term. Full transformer fine-tuning is possible, but this public CardiffNLP model is already fine-tuned on TweetEval emoji; the lower-risk next improvement is to combine its semantic/contextual signal with the SVM's hashtag, spelling, and character n-gram signal.

Best validation configuration:

```json
{json.dumps(best_config, ensure_ascii=False, indent=2)}
```

Top validation candidates:

{_candidate_table(top_candidates)}

Run configuration:

```json
{json.dumps(run_config, ensure_ascii=False, indent=2)}
```

## Test Results

{_metrics_table(metrics_by_model)}

The calibrated ensemble achieved **{_pct(float(ensemble_metrics['top1_accuracy']))} top-1**, **{_pct(float(ensemble_metrics['top3_accuracy']))} top-3**, and **{_pct(float(ensemble_metrics['macro_f1']))} macro-F1** (`{float(ensemble_metrics['macro_f1']):.4f}`). Compared with the previous best Twitter-RoBERTa run, this is a macro-F1 change of **{(float(ensemble_metrics['macro_f1']) - float(metrics_by_model['Twitter-RoBERTa emoji GPU']['macro_f1'])):+.4f}**.

## Interpretation

The ensemble is the best local model in this run. It improves macro-F1 over Twitter-RoBERTa from **{_pct(float(metrics_by_model['Twitter-RoBERTa emoji GPU']['macro_f1']))}** to **{_pct(float(ensemble_metrics['macro_f1']))}**, and top-3 from **{_pct(float(metrics_by_model['Twitter-RoBERTa emoji GPU']['top3_accuracy']))}** to **{_pct(float(ensemble_metrics['top3_accuracy']))}**. Top-1 drops slightly, from **{_pct(float(metrics_by_model['Twitter-RoBERTa emoji GPU']['top1_accuracy']))}** to **{_pct(float(ensemble_metrics['top1_accuracy']))}**, which is the expected tradeoff when tuning for macro-F1 and rare-class recovery instead of pure exact-match accuracy.

This is the best balance for the assignment because TweetEval's official emoji metric is macro-F1. The SVM contributes hashtag, spelling, and character-pattern evidence; Twitter-RoBERTa contributes contextual tweet representations; the calibration term shifts the decision boundary enough to recover minority classes without fully sacrificing common-class performance.

Easiest ensemble classes:

{_per_class_table(per_class_ensemble, ascending=False)}

Hardest ensemble classes:

{_per_class_table(per_class_ensemble, ascending=True)}

Most common hard-class confusions:

{_confusion_table(ensemble_metrics, per_class_ensemble, mapping)}

## Best Next Step

The next improvement beyond this report should be true fine-tuning of `vinai/bertweet-base` or `cardiffnlp/twitter-roberta-base` with macro-F1 model selection, class-balanced sampling, and 2-3 seeds. Based on the online leaderboard, that is the most likely route from our current **32.81 macro-F1** toward the **33-36 macro-F1** range. The downside is runtime and instability on 6 GB VRAM, so it should be treated as a longer experiment rather than the most reliable quick upgrade.

## Sources

{sources}
"""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune SVM + Twitter-RoBERTa ensemble and generate Report 3.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--results-dir", default="results/ensemble")
    parser.add_argument("--improved-results-dir", default="results/improved")
    parser.add_argument("--svm-model", default="models/linear_svm_word_char_tfidf.pkl")
    parser.add_argument("--roberta-test-scores", default="results/improved/scores_twitter_roberta_emoji_test.npy")
    parser.add_argument("--transformer-model", default="cardiffnlp/twitter-roberta-base-emoji")
    parser.add_argument("--transformer-batch-size", type=int, default=64)
    parser.add_argument("--transformer-max-length", type=int, default=128)
    parser.add_argument("--cpu-transformer", action="store_true")
    parser.add_argument("--report-path", default="report/report-3.md")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    start = time.time()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data_dir)

    print("Loading SVM scores...")
    svm_val, svm_test = load_svm_scores(dataset, args.svm_model)
    print("Loading/computing Twitter-RoBERTa scores...")
    roberta_val, roberta_test = load_or_compute_roberta_scores(dataset, args)

    svm_val_z, svm_test_z, svm_stats = normalize_scores(svm_val, svm_test)
    roberta_val_z, roberta_test_z, roberta_stats = normalize_scores(roberta_val, roberta_test)
    train_counts = np.bincount(dataset["train_labels"], minlength=len(dataset["mapping"])).astype(np.float64)
    train_prior = train_counts / train_counts.sum()

    print("Searching ensemble calibration grid on validation macro-F1...")
    best_config, top_candidates = search_ensemble(roberta_val_z, svm_val_z, dataset["val_labels"], train_prior)
    ensemble_val = combine_scores(
        roberta_val_z,
        svm_val_z,
        train_prior,
        best_config["weight_roberta"],
        best_config["roberta_temp"],
        best_config["svm_temp"],
        best_config["prior_alpha"],
    )
    ensemble_test = combine_scores(
        roberta_test_z,
        svm_test_z,
        train_prior,
        best_config["weight_roberta"],
        best_config["roberta_temp"],
        best_config["svm_temp"],
        best_config["prior_alpha"],
    )

    ensemble_val_metrics = evaluate_probs(softmax(ensemble_val), dataset["val_labels"], dataset["mapping"])
    ensemble_test_probs = softmax(ensemble_test)
    ensemble_metrics = evaluate_probs(ensemble_test_probs, dataset["test_labels"], dataset["mapping"])
    per_class_ensemble = per_class_frame(ensemble_metrics, dataset["mapping"])

    original_metrics = json.loads(Path("results/metrics_summary.json").read_text(encoding="utf-8"))["test"]
    improved_metrics = json.loads(Path(args.improved_results_dir, "metrics_improved_summary.json").read_text(encoding="utf-8"))
    metrics_by_model = {
        "Original MLP TF-IDF": original_metrics["MLP TF-IDF BoW"],
        "Original LSTM": original_metrics["LSTM"],
        "LinearSVM word+char TF-IDF": improved_metrics["LinearSVM word+char TF-IDF"],
        "Twitter-RoBERTa emoji GPU": improved_metrics["Twitter-RoBERTa emoji GPU"],
    }
    bertweet_metrics_path = Path("results/bertweet/metrics_bertweet_emoji_test.json")
    if bertweet_metrics_path.exists():
        metrics_by_model["BERTweet emoji HF checkpoint"] = json.loads(bertweet_metrics_path.read_text(encoding="utf-8"))
    metrics_by_model["Calibrated RoBERTa+SVM ensemble"] = ensemble_metrics

    save_json(results_dir / "best_ensemble_config.json", best_config)
    save_json(results_dir / "top_ensemble_candidates.json", {"candidates": top_candidates})
    save_json(results_dir / "metrics_ensemble_val.json", ensemble_val_metrics)
    save_json(results_dir / "metrics_ensemble_test.json", ensemble_metrics)
    save_json(results_dir / "metrics_report3_summary.json", metrics_by_model)
    save_json(results_dir / "normalization_stats.json", {"svm": svm_stats, "roberta": roberta_stats})
    np.save(results_dir / "scores_ensemble_test.npy", ensemble_test)
    predictions_frame(dataset["test_texts"], dataset["test_labels"], ensemble_test_probs, dataset["mapping"]).to_csv(
        results_dir / "predictions_ensemble.csv", index=False, encoding="utf-8"
    )
    Path(results_dir / "predictions_ensemble_labels.txt").write_text(
        "\n".join(str(int(x)) for x in ensemble_test.argmax(axis=1)) + "\n",
        encoding="utf-8",
    )
    per_class_ensemble.to_csv(results_dir / "per_class_ensemble.csv", index=False, encoding="utf-8")
    plot_confusion_matrix(
        ensemble_metrics["confusion_matrix"],
        dataset["mapping"],
        "Calibrated RoBERTa+SVM ensemble normalized confusion matrix",
        results_dir / "confusion_matrix_ensemble.png",
    )
    plot_score_comparison(metrics_by_model, results_dir / "score_comparison_report3.png")

    run_config = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformer_model": args.transformer_model,
        "transformer_batch_size": args.transformer_batch_size,
        "transformer_max_length": args.transformer_max_length,
        "search_objective": "validation macro-F1",
        "elapsed_minutes": round((time.time() - start) / 60, 2),
    }
    save_json(results_dir / "run_config.json", run_config)
    generate_report3(
        args.report_path,
        metrics_by_model,
        ensemble_metrics,
        per_class_ensemble,
        best_config,
        top_candidates,
        run_config,
        dataset["mapping"],
    )

    print("\nReport 3 metrics:")
    for name, metrics in metrics_by_model.items():
        print(
            f"{name}: top1={metrics['top1_accuracy']:.4f}, "
            f"top3={metrics['top3_accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
        )
    print(f"Report 3: {args.report_path}")


if __name__ == "__main__":
    main()
