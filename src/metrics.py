from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def top_k_accuracy_from_probs(probs: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    top_k = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([label in row for label, row in zip(labels, top_k)]))


def evaluate_probs(probs: np.ndarray, labels: np.ndarray, mapping: pd.DataFrame) -> dict[str, object]:
    pred = probs.argmax(axis=1)
    target_names = [f"{row.emoji} {row['name']}" for _, row in mapping.iterrows()]
    report = classification_report(
        labels,
        pred,
        labels=mapping["label"].tolist(),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "top1_accuracy": top_k_accuracy_from_probs(probs, labels, k=1),
        "top3_accuracy": top_k_accuracy_from_probs(probs, labels, k=3),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(labels, pred, labels=mapping["label"].tolist()).tolist(),
    }


def per_class_frame(metrics: dict[str, object], mapping: pd.DataFrame) -> pd.DataFrame:
    report = metrics["classification_report"]
    rows = []
    for _, item in mapping.iterrows():
        key = f"{item.emoji} {item['name']}"
        values = report.get(key, {})
        rows.append(
            {
                "label": int(item.label),
                "emoji": item.emoji,
                "name": item["name"],
                "precision": float(values.get("precision", 0.0)),
                "recall": float(values.get("recall", 0.0)),
                "f1": float(values.get("f1-score", 0.0)),
                "support": int(values.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


def predictions_frame(
    texts: list[str],
    labels: np.ndarray,
    probs: np.ndarray,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    label_to_emoji = dict(zip(mapping["label"], mapping["emoji"]))
    label_to_name = dict(zip(mapping["label"], mapping["name"]))
    pred = probs.argmax(axis=1)
    top3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    rows = []
    for text, true_label, pred_label, top_labels in zip(texts, labels, pred, top3):
        rows.append(
            {
                "text": text,
                "true_label": int(true_label),
                "true_emoji": label_to_emoji[int(true_label)],
                "true_name": label_to_name[int(true_label)],
                "pred_label": int(pred_label),
                "pred_emoji": label_to_emoji[int(pred_label)],
                "pred_name": label_to_name[int(pred_label)],
                "top3_labels": " ".join(str(int(x)) for x in top_labels),
                "top3_emojis": " ".join(label_to_emoji[int(x)] for x in top_labels),
                "correct_top1": bool(int(true_label) == int(pred_label)),
                "correct_top3": bool(int(true_label) in set(int(x) for x in top_labels)),
            }
        )
    return pd.DataFrame(rows)


def save_json(path: str | Path, payload: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

