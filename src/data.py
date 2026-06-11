from __future__ import annotations

import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emoji"
FILES = [
    "train_text.txt",
    "train_labels.txt",
    "val_text.txt",
    "val_labels.txt",
    "test_text.txt",
    "test_labels.txt",
    "mapping.txt",
]
EXPECTED_SPLIT_SIZES = {"train": 45000, "val": 5000, "test": 50000}

TOKEN_RE = re.compile(
    r"https?://\S+|www\.\S+|@\w+|#\w+|[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+|[^\w\s]",
    flags=re.UNICODE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
USER_RE = re.compile(r"@\w+")


def download_tweeteval_emoji(data_dir: str | Path) -> None:
    """Download TweetEval emoji raw files if they are missing."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        target = data_path / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"{BASE_URL}/{filename}"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        target.write_bytes(payload)


def normalize_text(text: str) -> str:
    """Normalize only high-variance tweet artifacts while preserving lexical signal."""
    text = URL_RE.sub(" URL ", text)
    text = USER_RE.sub(" USER ", text)
    return re.sub(r"\s+", " ", text).strip()


def tweet_tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text).lower())


def read_lines(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def load_mapping(data_dir: str | Path) -> pd.DataFrame:
    rows = []
    for line in read_lines(Path(data_dir) / "mapping.txt"):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rows.append(
            {
                "label": int(parts[0]),
                "emoji": parts[1],
                "name": parts[2].strip("_"),
                "display": f"{parts[1]} {parts[2].strip('_').replace('_', ' ')}",
            }
        )
    mapping = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    if len(mapping) != 20:
        raise ValueError(f"Expected 20 emoji labels, found {len(mapping)}")
    return mapping


def load_split(data_dir: str | Path, split: str) -> tuple[list[str], np.ndarray]:
    if split not in EXPECTED_SPLIT_SIZES:
        raise ValueError(f"Unknown split: {split}")
    texts = read_lines(Path(data_dir) / f"{split}_text.txt")
    labels = np.asarray([int(x) for x in read_lines(Path(data_dir) / f"{split}_labels.txt")], dtype=np.int64)
    expected = EXPECTED_SPLIT_SIZES[split]
    if len(texts) != expected or len(labels) != expected:
        raise ValueError(
            f"{split} size mismatch: expected {expected}, got {len(texts)} texts and {len(labels)} labels"
        )
    return texts, labels


def load_dataset(data_dir: str | Path) -> dict[str, object]:
    mapping = load_mapping(data_dir)
    train_texts, train_labels = load_split(data_dir, "train")
    val_texts, val_labels = load_split(data_dir, "val")
    test_texts, test_labels = load_split(data_dir, "test")
    return {
        "mapping": mapping,
        "train_texts": train_texts,
        "train_labels": train_labels,
        "val_texts": val_texts,
        "val_labels": val_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
    }


def build_vocab(texts: Iterable[str], max_vocab: int = 20000, min_freq: int = 2) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tweet_tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in counter.most_common():
        if count < min_freq:
            continue
        if token in vocab:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= max_vocab:
            break
    return vocab


def encode_texts(texts: Iterable[str], vocab: dict[str, int], max_len: int = 48) -> np.ndarray:
    if not isinstance(texts, list):
        texts = list(texts)
    encoded = np.zeros((len(texts), max_len), dtype=np.int64)
    unk = vocab["<unk>"]
    for row, text in enumerate(texts):
        ids = [vocab.get(token, unk) for token in tweet_tokenize(text)[:max_len]]
        if ids:
            encoded[row, : len(ids)] = ids
    return encoded


def class_distribution(labels: np.ndarray, mapping: pd.DataFrame) -> pd.DataFrame:
    counts = pd.Series(labels).value_counts().rename_axis("label").reset_index(name="count")
    out = mapping.merge(counts, on="label", how="left").fillna({"count": 0})
    out["count"] = out["count"].astype(int)
    out["percent"] = out["count"] / max(1, len(labels))
    return out


def top_tokens_by_class(
    texts: list[str],
    labels: np.ndarray,
    mapping: pd.DataFrame,
    min_class_count: int = 4,
    top_n: int = 10,
) -> dict[str, list[str]]:
    global_counts: Counter[str] = Counter()
    per_class: dict[int, Counter[str]] = {int(label): Counter() for label in mapping["label"]}
    for text, label in zip(texts, labels):
        tokens = [token for token in tweet_tokenize(text) if len(token) > 1 or token in {"!", "?", "$"}]
        unique_tokens = set(tokens)
        global_counts.update(unique_tokens)
        per_class[int(label)].update(unique_tokens)

    result: dict[str, list[str]] = {}
    for label, counts in per_class.items():
        scored = []
        for token, count in counts.items():
            if count < min_class_count:
                continue
            score = count / (global_counts[token] ** 0.55)
            scored.append((score, count, token))
        scored.sort(reverse=True)
        result[str(label)] = [token for _, _, token in scored[:top_n]]
    return result
