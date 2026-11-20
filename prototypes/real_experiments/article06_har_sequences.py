"""Compare GRU and Transformer on the official subject-held-out UCI HAR sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F


SIGNALS = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)


class GRUClassifier(nn.Module):
    def __init__(self, channels: int = 9, hidden: int = 32, classes: int = 6) -> None:
        super().__init__()
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.classifier(output[:, -1])


class TransformerClassifier(nn.Module):
    def __init__(self, channels: int = 9, hidden: int = 32, classes: int = 6, sequence_length: int = 128) -> None:
        super().__init__()
        self.input_projection = nn.Linear(channels, hidden)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x) + self.position[:, : x.shape[1]]
        hidden = self.encoder(hidden)
        return self.classifier(hidden.mean(dim=1))


def estimate_macs(sequence_length: int, channels: int = 9, hidden: int = 32) -> dict[str, int]:
    if min(sequence_length, channels, hidden) < 1:
        raise ValueError("sequence_length, channels, and hidden must be positive")
    return {
        "gru_macs": int(3 * sequence_length * hidden * (channels + hidden)),
        "transformer_macs": int(2 * sequence_length * sequence_length * hidden),
    }


def _read_partition(data_dir: Path, partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    signals_dir = data_dir / partition / "Inertial Signals"
    tensors = []
    hashes = {}
    for name in SIGNALS:
        path = signals_dir / f"{name}_{partition}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path.relative_to(data_dir))] = content_hash
        tensors.append(np.loadtxt(path, dtype=np.float32))
    X = np.stack(tensors, axis=-1)
    y = np.loadtxt(data_dir / partition / f"y_{partition}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(data_dir / partition / f"subject_{partition}.txt", dtype=np.int64)
    if X.ndim != 3 or X.shape[1:] != (128, 9) or len(X) != len(y) or len(y) != len(subjects):
        raise ValueError(f"Unexpected UCI HAR {partition} array shapes: X={X.shape}, y={y.shape}, subjects={subjects.shape}")
    return X, y, subjects, hashes


def _train_for_budget(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    budget_seconds: float,
    seed: int,
    batch_size: int = 64,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    rng = np.random.default_rng(seed)
    steps = 0
    losses = []
    started = time.perf_counter()
    order = rng.permutation(len(y))
    cursor = 0
    model.train()
    while time.perf_counter() - started < budget_seconds or steps == 0:
        if cursor >= len(order):
            order = rng.permutation(len(y))
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += len(indices)
        logits = model(X[indices])
        loss = F.cross_entropy(logits, y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps += 1
        if steps == 1 or steps % 20 == 0:
            losses.append({"step": steps, "batch_cross_entropy": float(loss.detach())})
    return {
        "training_seconds": time.perf_counter() - started,
        "steps": steps,
        "training_rows_processed": steps * batch_size,
        "loss_history": losses,
    }


def _evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 128) -> dict:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            predictions.append(model(X[start : start + batch_size]).argmax(dim=1).cpu().numpy())
    predicted = np.concatenate(predictions)
    truth = y.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
    }


def _inference_time(model: nn.Module, X: torch.Tensor, repeats: int = 10, batch_size: int = 128) -> float:
    batch = X[:batch_size]
    with torch.inference_mode():
        model(batch)
        started = time.perf_counter()
        for _ in range(repeats):
            model(batch)
    return (time.perf_counter() - started) / repeats


def run_real_experiment(
    data_dir: Path,
    seed: int = 6401,
    training_budget_seconds: float = 15.0,
    train_rows: int = 2048,
) -> dict:
    if training_budget_seconds <= 0 or train_rows < 12:
        raise ValueError("training_budget_seconds must be positive and train_rows >= 12")
    torch.set_num_threads(1)
    data_dir = Path(data_dir)
    X_train, y_train, subject_train, train_hashes = _read_partition(data_dir, "train")
    X_test, y_test, subject_test, test_hashes = _read_partition(data_dir, "test")
    if set(np.unique(subject_train)) & set(np.unique(subject_test)):
        raise AssertionError("Official UCI HAR subjects overlap between train and test")
    all_indices = np.arange(len(y_train))
    if train_rows < len(y_train):
        selected, _ = train_test_split(
            all_indices,
            train_size=train_rows,
            stratify=y_train,
            random_state=seed,
        )
    else:
        selected = all_indices
    X_train_t = torch.as_tensor(X_train[selected], dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train[selected], dtype=torch.long)
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32)
    y_test_t = torch.as_tensor(y_test, dtype=torch.long)

    results = []
    for name, model_type in (("gru", GRUClassifier), ("transformer", TransformerClassifier)):
        torch.manual_seed(seed)
        model = model_type().cpu()
        fit = _train_for_budget(model, X_train_t, y_train_t, training_budget_seconds, seed)
        metrics = _evaluate(model, X_test_t, y_test_t)
        results.append(
            {
                "model": name,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "estimated_macs_per_window": estimate_macs(128),
                **fit,
                **metrics,
                "inference_seconds_per_batch_128": _inference_time(model, X_test_t),
            }
        )
    return {
        "article_id": "06",
        "experiment": "GRU versus Transformer on UCI HAR raw inertial sequences",
        "evidence_type": "measured_public_dataset",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "UCI Human Activity Recognition Using Smartphones",
            "source_id": "UCI-240-UCI-HAR",
            "source_url": "https://archive.ics.uci.edu/dataset/240/human-activity-recognition-using-smartphones",
            "license": "CC BY 4.0",
            "train_rows_total": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "training_rows_used": int(len(selected)),
            "sequence_shape": [128, 9],
            "official_train_subjects": sorted(int(x) for x in np.unique(subject_train)),
            "official_test_subjects": sorted(int(x) for x in np.unique(subject_test)),
            "subject_overlap_count": 0,
            "input_sha256": {**train_hashes, **test_hashes},
        },
        "method": {
            "training_budget_seconds_per_model": training_budget_seconds,
            "training_subset_rule": "stratified subset of official training partition only",
            "test_selection": "official UCI test partition, participants disjoint from train",
            "seed": seed,
            "batch_size": 64,
            "optimizer": "AdamW(lr=0.002)",
            "gru": "one-layer GRU, hidden=32, last timestep classifier",
            "transformer": "one-layer encoder, d_model=32, 4 heads, FF=64, mean pooling",
            "no_test_tuning": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpus": os.cpu_count(),
            "physical_memory_bytes": psutil.virtual_memory().total,
            "torch": version("torch"),
            "numpy": version("numpy"),
            "torch_threads": torch.get_num_threads(),
            "device": "CPU",
        },
        "runs": results,
        "limitations": [
            "One seed, one sensor classification task, and a 2,048-window training subset make this exploratory rather than a broad architecture ranking.",
            "Equal wall-clock budgets are hardware-specific and do not imply equal FLOPs or equal training data passes.",
            "The dataset contains overlapping time windows and its official participant split is used to avoid cross-subject leakage.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("evidence/08_real/data/UCI HAR Dataset"))
    parser.add_argument("--out", type=Path, default=Path("evidence/06_real"))
    parser.add_argument("--budget-seconds", type=float, default=15.0)
    parser.add_argument("--train-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=6401)
    args = parser.parse_args(argv)
    result = run_real_experiment(args.data_dir, args.seed, args.budget_seconds, args.train_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = args.out / f"article-06-uci-har-sequences-{stamp}.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    print(path)
    for row in result["runs"]:
        print(row["model"], "accuracy", round(row["accuracy"], 4), "macro_f1", round(row["macro_f1"], 4), "steps", row["steps"], "train_seconds", round(row["training_seconds"], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
