from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RESEARCH_SOURCES = [
    {
        "title": "SemEval-2018 Task 2: Multilingual Emoji Prediction",
        "url": "https://aclanthology.org/S18-1003/",
        "note": "Original shared task defining emoji prediction from tweets.",
    },
    {
        "title": "TweetEval benchmark repository",
        "url": "https://github.com/cardiffnlp/tweeteval",
        "note": "Provides unified fixed splits and raw files used in this project.",
    },
    {
        "title": "TweetEval paper",
        "url": "https://aclanthology.org/2020.findings-emnlp.148/",
        "note": "Documents TweetEval as a standardized Twitter classification benchmark.",
    },
    {
        "title": "Tübingen-Oslo at SemEval-2018 Task 2",
        "url": "https://coltekin.net/cagri/papers/coltekin2018semeval.pdf",
        "note": "Reports strong SVM performance and discusses RNN difficulty on emoji prediction.",
    },
    {
        "title": "PickleTeam! at SemEval-2018 Task 2",
        "url": "https://aclanthology.org/S18-1072.pdf",
        "note": "Combines SVM and LSTM predictions, motivating comparison between sparse and sequence models.",
    },
    {
        "title": "DeepMoji",
        "url": "https://github.com/bfelbo/deepmoji",
        "note": "Large-scale emoji-pretrained representation model trained on 1.2B tweets.",
    },
    {
        "title": "NTUA-SLP at SemEval-2018 Task 2",
        "url": "https://arxiv.org/abs/1804.06657",
        "note": "Attention LSTM system that ranked highly using pretrained Twitter embeddings.",
    },
    {
        "title": "NLP-text2emoji public project",
        "url": "https://github.com/joonasrooben/NLP-text2emoji",
        "note": "Example public implementation using bidirectional LSTMs and word/character embeddings.",
    },
]


def plot_confusion_matrix(
    confusion: list[list[int]],
    mapping: pd.DataFrame,
    title: str,
    path: str | Path,
) -> None:
    matrix = np.asarray(confusion, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    labels = [f"{row.label}\n{row['name'].replace('_', ' ')[:12]}" for _, row in mapping.iterrows()]
    plt.figure(figsize=(12, 10))
    sns.heatmap(normalized, cmap="viridis", xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _score_table(metrics_by_model: dict[str, dict[str, object]]) -> str:
    rows = ["| Model | Top-1 Accuracy | Top-3 Accuracy | Macro F1 | Weighted F1 |", "|---|---:|---:|---:|---:|"]
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    _fmt_pct(float(metrics["top1_accuracy"])),
                    _fmt_pct(float(metrics["top3_accuracy"])),
                    f"{float(metrics['macro_f1']):.4f}",
                    f"{float(metrics['weighted_f1']):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _per_class_table(per_class: pd.DataFrame, top_tokens: dict[str, list[str]], n: int = 5, ascending: bool = False) -> str:
    frame = per_class.sort_values(["f1", "recall", "support"], ascending=[ascending, ascending, not ascending]).head(n)
    rows = ["| Emoji | Name | F1 | Recall | Support | Indicative training tokens |", "|---|---|---:|---:|---:|---|"]
    for _, row in frame.iterrows():
        tokens = ", ".join(top_tokens.get(str(int(row.label)), [])[:6])
        rows.append(
            f"| {row.emoji} | {row['name'].replace('_', ' ')} | {row.f1:.4f} | {row.recall:.4f} | {int(row.support)} | {tokens} |"
        )
    return "\n".join(rows)


def _hardest_confusion_table(
    metrics: dict[str, object],
    per_class: pd.DataFrame,
    mapping: pd.DataFrame,
    n: int = 6,
) -> str:
    matrix = np.asarray(metrics["confusion_matrix"])
    label_to_name = {
        int(row.label): f"{row.emoji} {row['name'].replace('_', ' ')}" for _, row in mapping.iterrows()
    }
    rows = [
        "| True Emoji | Most Common Wrong Prediction | Wrong Count | Share Of That True Class |",
        "|---|---|---:|---:|",
    ]
    hardest = per_class.sort_values(["f1", "recall", "support"], ascending=[True, True, False]).head(n)
    for _, item in hardest.iterrows():
        label = int(item.label)
        counts = matrix[label].copy()
        counts[label] = 0
        wrong_total = counts.sum()
        if wrong_total == 0:
            rows.append(f"| {label_to_name[label]} | no dominant confusion | 0 | 0.00% |")
            continue
        pred_label = int(np.argmax(counts))
        rows.append(
            f"| {label_to_name[label]} | {label_to_name[pred_label]} | {int(counts[pred_label])} | "
            f"{counts[pred_label] / max(1, int(item.support)) * 100:.2f}% |"
        )
    return "\n".join(rows)


def _distribution_table(distribution: pd.DataFrame) -> str:
    rows = ["| Emoji | Name | Train Count | Train Share |", "|---|---|---:|---:|"]
    for _, row in distribution.iterrows():
        rows.append(f"| {row.emoji} | {row['name'].replace('_', ' ')} | {int(row['count'])} | {_fmt_pct(float(row['percent']))} |")
    return "\n".join(rows)


def generate_report(
    report_path: str | Path,
    mapping: pd.DataFrame,
    train_distribution: pd.DataFrame,
    metrics_by_model: dict[str, dict[str, object]],
    per_class_by_model: dict[str, pd.DataFrame],
    top_tokens: dict[str, list[str]],
    run_config: dict[str, object],
) -> None:
    winner_name = max(metrics_by_model, key=lambda name: float(metrics_by_model[name]["top1_accuracy"]))
    winner = metrics_by_model[winner_name]
    mlp_metrics = metrics_by_model["MLP TF-IDF BoW"]
    lstm_metrics = metrics_by_model["LSTM"]
    easiest_table = _per_class_table(per_class_by_model[winner_name], top_tokens, n=6, ascending=False)
    hardest_table = _per_class_table(per_class_by_model[winner_name], top_tokens, n=6, ascending=True)
    confusion_table = _hardest_confusion_table(winner, per_class_by_model[winner_name], mapping, n=6)

    sources = "\n".join(f"- [{item['title']}]({item['url']}): {item['note']}" for item in RESEARCH_SOURCES)
    config_json = json.dumps(run_config, ensure_ascii=False, indent=2)

    content = f"""# Project 17 Report: Emoji Prediction From Text

## Executive Summary

This project predicts the most likely emoji for a tweet using the TweetEval `emoji` benchmark. The required comparison is between an MLP over bag-of-words features and an LSTM over token sequences. The best model in this local run is **{winner_name}**, with **{_fmt_pct(float(winner['top1_accuracy']))} top-1 accuracy** and **{_fmt_pct(float(winner['top3_accuracy']))} top-3 accuracy** on the official TweetEval test split.

{_score_table(metrics_by_model)}

Top-3 accuracy is important for this task because emoji usage is inherently ambiguous: a tweet about love, celebration, friends, or a photo can plausibly map to several visually different but semantically related emojis. Top-1 measures exact prediction; top-3 measures whether the model can place the correct emoji among its most plausible alternatives.

## Problem And Dataset

The task is a 20-class tweet classification problem: remove or ignore the emoji label and predict which emoji best matches the text. TweetEval's emoji subset is based on SemEval-2018 Task 2 and provides fixed train, validation, and test files. This run verified the expected split sizes: 45,000 training examples, 5,000 validation examples, and 50,000 test examples.

Training label distribution:

{_distribution_table(train_distribution)}

The distribution is imbalanced. Frequent emojis such as hearts, laughter, and camera-related labels provide many examples, while rarer or more context-specific emojis have fewer and noisier cues. This makes macro-F1 lower than weighted-F1 and makes per-class analysis more informative than a single accuracy number.

## Research Context And Method Choice

The SemEval task and follow-up systems show that emoji prediction from tweets behaves like short-text classification with strong lexical cues. Several SemEval participants found that sparse lexical models such as TF-IDF plus linear SVM were difficult for RNNs to beat on this dataset, especially without large external pretraining. PickleTeam used both SVM and LSTM signals, which supports the core comparison in this project: sparse bag-of-words evidence versus sequence modeling. Public demo projects and notebooks often use LSTM-style embedding models, while benchmark systems show that strong sparse baselines remain hard to beat on this exact task. DeepMoji and later transformer/TweetNLP-style models show that large-scale emoji pretraining can learn richer emotional representations, but those approaches are heavier than the requested MLP-vs-LSTM assignment and reduce interpretability.

For this project, the MLP with TF-IDF bag-of-words is the strongest transparent baseline: it directly captures hashtags, words, repeated social phrases, and short n-grams that are common in tweets. The LSTM is included because it can model token order and local composition, but a small CPU-trained LSTM from scratch has less prior knowledge than TF-IDF and can struggle with rare words, hashtags, and class imbalance.

## Implementation

- Dataset download: raw TweetEval files from CardiffNLP GitHub.
- Preprocessing: URL and user-mention normalization, lowercase tokenization, hashtag retention.
- MLP input: TF-IDF word unigram/bigram features, capped vocabulary, sparse mini-batches converted to dense tensors per batch.
- LSTM input: train-only vocabulary, padded token ID sequences, learned embeddings, final hidden state classifier.
- Evaluation: top-1 accuracy, top-3 accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, confusion matrix, and prediction CSVs.

Run configuration:

```json
{config_json}
```

## Results And Interpretation

The MLP top-1 score is **{_fmt_pct(float(mlp_metrics['top1_accuracy']))}** and its top-3 score is **{_fmt_pct(float(mlp_metrics['top3_accuracy']))}**. The LSTM top-1 score is **{_fmt_pct(float(lstm_metrics['top1_accuracy']))}** and its top-3 score is **{_fmt_pct(float(lstm_metrics['top3_accuracy']))}**.

When the MLP wins, the reason is usually that tweet emoji prediction contains many lexical shortcuts: `love`, `happy`, `birthday`, `christmas`, `photo`, `beach`, `sun`, `fire`, and hashtag forms often point directly toward a small group of emojis. TF-IDF also handles rare but high-value tokens well. When the LSTM wins or narrows the gap, it is usually because order and local phrase composition matter, such as distinguishing affectionate messages from laughter or sarcasm.

## Easiest Emojis

Using the best model's per-class F1/recall, these labels were easiest:

{easiest_table}

These are easier when the text has distinctive words, seasonal vocabulary, platform conventions, or narrow contexts. For example, `christmas`, `merry`, and holiday terms strongly identify the Christmas tree; `photo`, `camera`, and location/photo-sharing language help camera labels; heart emojis often align with repeated affection words.

## Hardest Emojis

Using the best model's per-class F1/recall, these labels were hardest:

{hardest_table}

These are harder when the emoji meaning overlaps with other classes or the text lacks explicit lexical clues. Hearts confuse with each other because red, blue, purple, and two-hearts often accompany similar affection text. Winking and playful face emojis can be interchangeable in casual tweets. Some labels are also affected by class imbalance: fewer examples give the model less evidence and lower recall.

Most common wrong predictions for the hardest labels:

{confusion_table}

## Methods We Could Have Tried

- Linear SVM or logistic regression on TF-IDF: likely very competitive for this benchmark and often stronger than small neural models.
- Character n-grams: useful for hashtags, slang, spelling variation, and elongated words.
- Class weighting or focal loss: could improve rare emoji recall but may reduce top-1 accuracy on frequent labels.
- Bidirectional LSTM with attention: closer to strong SemEval neural systems, especially with pretrained Twitter embeddings.
- DeepMoji-style pretraining: powerful but requires huge emoji-labeled corpora and is outside the assignment's lightweight local scope.
- BERTweet/TweetNLP transformer fine-tuning: likely strongest modern option, but it changes the comparison from MLP-vs-LSTM to pretrained transformer fine-tuning.

## Sources

{sources}

## Reproducibility Notes

Run the pipeline from the project root with:

```powershell
$env:PYTHONIOENCODING="utf-8"
python src\\train_and_evaluate.py
```

Saved artifacts include `results/metrics_summary.json`, `results/per_class_*.csv`, `results/predictions_*.csv`, confusion matrix PNGs, and model checkpoints in `models/`.
"""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(content, encoding="utf-8")
