from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import (
    build_vocab,
    class_distribution,
    download_tweeteval_emoji,
    encode_texts,
    load_dataset,
    top_tokens_by_class,
    tweet_tokenize,
)
from metrics import evaluate_probs, per_class_frame, predictions_frame, save_json, softmax
from models import BowMLP, TweetLSTM
from reporting import generate_report, plot_confusion_matrix


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def iter_sparse_batches(x_matrix, y: np.ndarray, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    indices = np.arange(x_matrix.shape[0])
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        x_batch = x_matrix[batch_idx].toarray().astype(np.float32, copy=False)
        y_batch = y[batch_idx]
        yield x_batch, y_batch


@torch.no_grad()
def predict_mlp_logits(model: nn.Module, x_matrix, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    logits = []
    for start in range(0, x_matrix.shape[0], batch_size):
        x_batch = x_matrix[start : start + batch_size].toarray().astype(np.float32, copy=False)
        xb = torch.from_numpy(x_batch).to(device)
        logits.append(model(xb).cpu().numpy())
    return np.vstack(logits)


@torch.no_grad()
def predict_lstm_logits(model: nn.Module, encoded: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    logits = []
    for start in range(0, len(encoded), batch_size):
        xb = torch.from_numpy(encoded[start : start + batch_size]).long().to(device)
        logits.append(model(xb).cpu().numpy())
    return np.vstack(logits)


def train_mlp(
    train_texts: list[str],
    train_labels: np.ndarray,
    val_texts: list[str],
    val_labels: np.ndarray,
    test_texts: list[str],
    args: argparse.Namespace,
    device: torch.device,
    num_classes: int,
) -> dict[str, object]:
    print("\n[MLP] Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        tokenizer=tweet_tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)
    print(f"[MLP] TF-IDF shape: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

    model = BowMLP(
        input_dim=x_train.shape[1],
        num_classes=num_classes,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    best_state = copy.deepcopy(model.state_dict())
    best_val_top1 = -1.0
    history = []
    for epoch in range(1, args.epochs_mlp + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for x_batch, y_batch in iter_sparse_batches(x_train, train_labels, args.batch_size_mlp, rng, shuffle=True):
            xb = torch.from_numpy(x_batch).to(device)
            yb = torch.from_numpy(y_batch).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y_batch)
            total_count += len(y_batch)

        val_probs = softmax(predict_mlp_logits(model, x_val, args.eval_batch_size, device))
        val_metrics = evaluate_probs(val_probs, val_labels, args.mapping)
        avg_loss = total_loss / max(1, total_count)
        history.append({"epoch": epoch, "train_loss": avg_loss, **{k: val_metrics[k] for k in ["top1_accuracy", "top3_accuracy", "macro_f1"]}})
        print(
            f"[MLP] epoch {epoch}/{args.epochs_mlp} loss={avg_loss:.4f} "
            f"val_top1={val_metrics['top1_accuracy']:.4f} val_top3={val_metrics['top3_accuracy']:.4f}"
        )
        if float(val_metrics["top1_accuracy"]) > best_val_top1:
            best_val_top1 = float(val_metrics["top1_accuracy"])
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    val_probs = softmax(predict_mlp_logits(model, x_val, args.eval_batch_size, device))
    test_probs = softmax(predict_mlp_logits(model, x_test, args.eval_batch_size, device))
    return {
        "model": model,
        "vectorizer": vectorizer,
        "history": history,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "input_dim": x_train.shape[1],
    }


def train_lstm(
    train_texts: list[str],
    train_labels: np.ndarray,
    val_texts: list[str],
    val_labels: np.ndarray,
    test_texts: list[str],
    args: argparse.Namespace,
    device: torch.device,
    num_classes: int,
) -> dict[str, object]:
    print("\n[LSTM] Building vocabulary and encoded sequences...")
    vocab = build_vocab(train_texts, max_vocab=args.lstm_vocab_size, min_freq=args.min_df)
    x_train = encode_texts(train_texts, vocab, max_len=args.max_len)
    x_val = encode_texts(val_texts, vocab, max_len=args.max_len)
    x_test = encode_texts(test_texts, vocab, max_len=args.max_len)
    print(f"[LSTM] vocab={len(vocab)} max_len={args.max_len}")

    model = TweetLSTM(
        vocab_size=len(vocab),
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.lstm_hidden_dim,
        num_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_dataset = TensorDataset(torch.from_numpy(x_train).long(), torch.from_numpy(train_labels).long())
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_lstm,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_top1 = -1.0
    history = []
    for epoch in range(1, args.epochs_lstm + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total_count += len(yb)

        val_probs = softmax(predict_lstm_logits(model, x_val, args.eval_batch_size, device))
        val_metrics = evaluate_probs(val_probs, val_labels, args.mapping)
        avg_loss = total_loss / max(1, total_count)
        history.append({"epoch": epoch, "train_loss": avg_loss, **{k: val_metrics[k] for k in ["top1_accuracy", "top3_accuracy", "macro_f1"]}})
        print(
            f"[LSTM] epoch {epoch}/{args.epochs_lstm} loss={avg_loss:.4f} "
            f"val_top1={val_metrics['top1_accuracy']:.4f} val_top3={val_metrics['top3_accuracy']:.4f}"
        )
        if float(val_metrics["top1_accuracy"]) > best_val_top1:
            best_val_top1 = float(val_metrics["top1_accuracy"])
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    val_probs = softmax(predict_lstm_logits(model, x_val, args.eval_batch_size, device))
    test_probs = softmax(predict_lstm_logits(model, x_test, args.eval_batch_size, device))
    return {
        "model": model,
        "vocab": vocab,
        "history": history,
        "val_probs": val_probs,
        "test_probs": test_probs,
    }


def maybe_subset(dataset: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    if not args.quick:
        return dataset
    rng = np.random.default_rng(args.seed)
    subset = dict(dataset)
    for split, size in [("train", 8000), ("val", 1000), ("test", 5000)]:
        texts = dataset[f"{split}_texts"]
        labels = dataset[f"{split}_labels"]
        idx = rng.choice(len(labels), size=min(size, len(labels)), replace=False)
        subset[f"{split}_texts"] = [texts[i] for i in idx]
        subset[f"{split}_labels"] = labels[idx]
    print("[quick] Using subset: train=8000, val=1000, test=5000")
    return subset


def save_artifacts(
    args: argparse.Namespace,
    dataset: dict[str, object],
    mlp_result: dict[str, object],
    lstm_result: dict[str, object],
    metrics_by_model: dict[str, dict[str, object]],
    per_class_by_model: dict[str, pd.DataFrame],
    top_tokens: dict[str, list[str]],
    train_distribution: pd.DataFrame,
    run_config: dict[str, object],
) -> None:
    results_dir = Path(args.results_dir)
    models_dir = Path(args.models_dir)
    report_dir = Path(args.report_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": {
            "name": "TweetEval emoji",
            "train_size": len(dataset["train_labels"]),
            "val_size": len(dataset["val_labels"]),
            "test_size": len(dataset["test_labels"]),
            "num_classes": len(dataset["mapping"]),
        },
        "run_config": run_config,
        "validation": {
            "MLP TF-IDF BoW": evaluate_probs(mlp_result["val_probs"], dataset["val_labels"], dataset["mapping"]),
            "LSTM": evaluate_probs(lstm_result["val_probs"], dataset["val_labels"], dataset["mapping"]),
        },
        "test": metrics_by_model,
    }
    save_json(results_dir / "metrics_summary.json", summary)
    save_json(results_dir / "top_tokens_by_class.json", top_tokens)

    pd.DataFrame(mlp_result["history"]).to_csv(results_dir / "history_mlp.csv", index=False, encoding="utf-8")
    pd.DataFrame(lstm_result["history"]).to_csv(results_dir / "history_lstm.csv", index=False, encoding="utf-8")
    train_distribution.to_csv(results_dir / "train_label_distribution.csv", index=False, encoding="utf-8")

    for model_name, per_class in per_class_by_model.items():
        slug = model_name.lower().replace(" ", "_").replace("-", "").replace("__", "_")
        per_class.to_csv(results_dir / f"per_class_{slug}.csv", index=False, encoding="utf-8")
        plot_confusion_matrix(
            metrics_by_model[model_name]["confusion_matrix"],
            dataset["mapping"],
            f"{model_name} normalized confusion matrix",
            results_dir / f"confusion_matrix_{slug}.png",
        )

    predictions_frame(
        dataset["test_texts"],
        dataset["test_labels"],
        mlp_result["test_probs"],
        dataset["mapping"],
    ).to_csv(results_dir / "predictions_mlp_tfidf_bow.csv", index=False, encoding="utf-8")
    predictions_frame(
        dataset["test_texts"],
        dataset["test_labels"],
        lstm_result["test_probs"],
        dataset["mapping"],
    ).to_csv(results_dir / "predictions_lstm.csv", index=False, encoding="utf-8")

    with (models_dir / "tfidf_vectorizer.pkl").open("wb") as handle:
        pickle.dump(mlp_result["vectorizer"], handle)
    torch.save(
        {
            "model_state_dict": mlp_result["model"].state_dict(),
            "input_dim": mlp_result["input_dim"],
            "num_classes": len(dataset["mapping"]),
            "config": run_config,
        },
        models_dir / "mlp_tfidf_bow.pt",
    )
    (models_dir / "lstm_vocab.json").write_text(json.dumps(lstm_result["vocab"], ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": lstm_result["model"].state_dict(),
            "vocab_size": len(lstm_result["vocab"]),
            "num_classes": len(dataset["mapping"]),
            "config": run_config,
        },
        models_dir / "lstm.pt",
    )

    generate_report(
        report_path=report_dir / "report-1.md",
        mapping=dataset["mapping"],
        train_distribution=train_distribution,
        metrics_by_model=metrics_by_model,
        per_class_by_model=per_class_by_model,
        top_tokens=top_tokens,
        run_config=run_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP BoW and LSTM models for TweetEval emoji prediction.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--report-dir", default="report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or CUDA device such as 'cuda'")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test subset.")
    parser.add_argument("--epochs-mlp", type=int, default=4)
    parser.add_argument("--epochs-lstm", type=int, default=4)
    parser.add_argument("--batch-size-mlp", type=int, default=256)
    parser.add_argument("--batch-size-lstm", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--lstm-vocab-size", type=int, default=20000)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    set_seed(args.seed)
    start = time.time()
    project_root = Path.cwd()
    for path in [args.data_dir, args.models_dir, args.results_dir, args.report_dir]:
        Path(path).mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print("Downloading/verifying TweetEval emoji files...")
    download_tweeteval_emoji(args.data_dir)
    dataset = load_dataset(args.data_dir)
    dataset = maybe_subset(dataset, args)
    args.mapping = dataset["mapping"]
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(
        f"Dataset sizes: train={len(dataset['train_labels'])}, "
        f"val={len(dataset['val_labels'])}, test={len(dataset['test_labels'])}, classes={len(dataset['mapping'])}"
    )

    run_config = {
        "seed": args.seed,
        "device": str(device),
        "quick": args.quick,
        "epochs_mlp": args.epochs_mlp,
        "epochs_lstm": args.epochs_lstm,
        "batch_size_mlp": args.batch_size_mlp,
        "batch_size_lstm": args.batch_size_lstm,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "min_df": args.min_df,
        "max_features": args.max_features,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "lstm_vocab_size": args.lstm_vocab_size,
        "max_len": args.max_len,
        "embedding_dim": args.embedding_dim,
        "lstm_hidden_dim": args.lstm_hidden_dim,
        "lstm_layers": args.lstm_layers,
    }

    train_distribution = class_distribution(dataset["train_labels"], dataset["mapping"])
    top_tokens = top_tokens_by_class(dataset["train_texts"], dataset["train_labels"], dataset["mapping"])

    mlp_result = train_mlp(
        dataset["train_texts"],
        dataset["train_labels"],
        dataset["val_texts"],
        dataset["val_labels"],
        dataset["test_texts"],
        args,
        device,
        len(dataset["mapping"]),
    )
    lstm_result = train_lstm(
        dataset["train_texts"],
        dataset["train_labels"],
        dataset["val_texts"],
        dataset["val_labels"],
        dataset["test_texts"],
        args,
        device,
        len(dataset["mapping"]),
    )

    metrics_by_model = {
        "MLP TF-IDF BoW": evaluate_probs(mlp_result["test_probs"], dataset["test_labels"], dataset["mapping"]),
        "LSTM": evaluate_probs(lstm_result["test_probs"], dataset["test_labels"], dataset["mapping"]),
    }
    per_class_by_model = {
        name: per_class_frame(metrics, dataset["mapping"]) for name, metrics in metrics_by_model.items()
    }

    save_artifacts(
        args,
        dataset,
        mlp_result,
        lstm_result,
        metrics_by_model,
        per_class_by_model,
        top_tokens,
        train_distribution,
        run_config,
    )

    print("\nFinal test metrics:")
    for name, metrics in metrics_by_model.items():
        print(
            f"{name}: top1={metrics['top1_accuracy']:.4f}, "
            f"top3={metrics['top3_accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )
    print(f"\nDone in {(time.time() - start) / 60:.2f} minutes.")
    print(f"Report: {Path(args.report_dir) / 'report-1.md'}")


if __name__ == "__main__":
    main()
