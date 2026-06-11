from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_dataset
from metrics import evaluate_probs, per_class_frame, predictions_frame, save_json, softmax
from reporting import plot_confusion_matrix


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


@torch.no_grad()
def predict_scores(texts: list[str], tokenizer, model, device: torch.device, batch_size: int, max_length: int) -> np.ndarray:
    model.eval()
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="BERTweet batches"):
        batch_texts = [tokenizer.normalizeTweet(text) if hasattr(tokenizer, "normalizeTweet") else text for text in texts[start : start + batch_size]]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**encoded).logits
        else:
            logits = model(**encoded).logits
        chunks.append(logits.float().cpu().numpy())
    return np.vstack(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CardiffNLP BERTweet emoji model on TweetEval emoji.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--results-dir", default="results/bertweet")
    parser.add_argument("--model", default="cardiffnlp/bertweet-base-emoji")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    start = time.time()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    val_scores_path = results_dir / "scores_bertweet_emoji_val.npy"
    test_scores_path = results_dir / "scores_bertweet_emoji_test.npy"
    if val_scores_path.exists():
        val_scores = np.load(val_scores_path)
    else:
        val_scores = predict_scores(dataset["val_texts"], tokenizer, model, device, args.batch_size, args.max_length)
        np.save(val_scores_path, val_scores)
    if test_scores_path.exists():
        test_scores = np.load(test_scores_path)
    else:
        test_scores = predict_scores(dataset["test_texts"], tokenizer, model, device, args.batch_size, args.max_length)
        np.save(test_scores_path, test_scores)

    val_metrics = evaluate_probs(softmax(val_scores), dataset["val_labels"], dataset["mapping"])
    test_probs = softmax(test_scores)
    test_metrics = evaluate_probs(test_probs, dataset["test_labels"], dataset["mapping"])
    per_class = per_class_frame(test_metrics, dataset["mapping"])
    save_json(results_dir / "metrics_bertweet_emoji_val.json", val_metrics)
    save_json(results_dir / "metrics_bertweet_emoji_test.json", test_metrics)
    save_json(
        results_dir / "run_config.json",
        {
            "model": args.model,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "elapsed_minutes": round((time.time() - start) / 60, 2),
        },
    )
    predictions_frame(dataset["test_texts"], dataset["test_labels"], test_probs, dataset["mapping"]).to_csv(
        results_dir / "predictions_bertweet_emoji.csv", index=False, encoding="utf-8"
    )
    Path(results_dir / "predictions_bertweet_emoji_labels.txt").write_text(
        "\n".join(str(int(x)) for x in test_scores.argmax(axis=1)) + "\n",
        encoding="utf-8",
    )
    per_class.to_csv(results_dir / "per_class_bertweet_emoji.csv", index=False, encoding="utf-8")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        dataset["mapping"],
        "BERTweet emoji normalized confusion matrix",
        results_dir / "confusion_matrix_bertweet_emoji.png",
    )
    print(
        f"BERTweet emoji: top1={test_metrics['top1_accuracy']:.4f}, "
        f"top3={test_metrics['top3_accuracy']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}, "
        f"weighted_f1={test_metrics['weighted_f1']:.4f}"
    )


if __name__ == "__main__":
    main()

