#!/usr/bin/env python
"""Train and evaluate a single method at a single label fraction / seed.

Usage:
    python train.py --config configs/default.yaml --method protonet \
        --label-fraction 0.1 --seed 0

Prints a JSON line with accuracy and f1_macro to stdout. Intended to be
called directly for debugging a single configuration, or invoked by
run_all_experiments.py to sweep the full grid.
"""
import argparse
import json

import numpy as np

from src.data.splits import (
    cross_load_split,
    stratified_label_fraction_split,
    train_test_split_stratified,
)
from src.evaluation.metrics import compute_metrics
from src.features.handcrafted import extract_features_batch
from src.models.classical import build_random_forest, build_svm
from src.models.cnn1d import CNN1DClassifier
from src.models.episodic import evaluate_protonet, train_protonet
from src.models.lstm import ShallowLSTMClassifier
from src.models.prototypical import PrototypicalNetwork
from src.models.train_utils import predict, train_classifier
from src.self_supervised.simclr import SimCLRModel, nt_xent_loss
from src.self_supervised.augmentations import augment_pair
from src.utils.config import load_config
from src.utils.seeding import set_seed

import torch


def load_windows(processed_path: str):
    data = np.load(processed_path, allow_pickle=True)
    return data["X"], data["y"], data["load_hp"], list(data["class_names"])


def run_classical(method, X_train, y_train, X_test, y_test, seed, fs):
    feat_train = extract_features_batch(X_train, fs=fs)
    feat_test = extract_features_batch(X_test, fs=fs)
    model = build_svm(seed) if method == "svm" else build_random_forest(seed)
    model.fit(feat_train, y_train)
    y_pred = model.predict(feat_test)
    return compute_metrics(y_test, y_pred)


def run_cnn1d(X_train, y_train, X_test, y_test, n_classes, cfg, seed):
    model = CNN1DClassifier(n_classes=n_classes, embedding_dim=cfg["training"]["embedding_dim"])
    model = train_classifier(
        model, X_train, y_train,
        epochs=cfg["training"]["epochs"], batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"],
        device=cfg["training"]["device"],
    )
    y_pred = predict(model, X_test, device=cfg["training"]["device"])
    return compute_metrics(y_test, y_pred)


def run_lstm(X_train, y_train, X_test, y_test, n_classes, cfg, seed):
    model = ShallowLSTMClassifier(n_classes=n_classes)
    model = train_classifier(
        model, X_train, y_train,
        epochs=cfg["training"]["epochs"], batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"],
        device=cfg["training"]["device"],
    )
    y_pred = predict(model, X_test, device=cfg["training"]["device"])
    return compute_metrics(y_test, y_pred)


def run_protonet(X_train, y_train, X_test, y_test, cfg, seed):
    pcfg = cfg["protonet"]
    n_support = min(pcfg["n_support"], min(np.bincount(y_train)) - 1) if len(np.unique(y_train)) > 0 else pcfg["n_support"]
    n_support = max(1, n_support)
    n_query = min(pcfg["n_query"], max(1, min(np.bincount(y_train)) - n_support))
    model = PrototypicalNetwork(embedding_dim=cfg["training"]["embedding_dim"])
    model = train_protonet(
        model, X_train, y_train, n_support=n_support, n_query=n_query,
        episodes_per_epoch=pcfg["episodes_per_epoch"], epochs=pcfg["epochs"],
        lr=cfg["training"]["learning_rate"], seed=seed, device=cfg["training"]["device"],
    )
    y_pred = evaluate_protonet(model, X_train, y_train, X_test, device=cfg["training"]["device"])
    return compute_metrics(y_test, y_pred)


def run_simclr_linear(X_unlabeled, X_train, y_train, X_test, y_test, n_classes, cfg, seed):
    scfg = cfg["simclr"]
    device = cfg["training"]["device"]
    model = SimCLRModel(embedding_dim=cfg["training"]["embedding_dim"], projection_dim=scfg["projection_dim"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    X_unlabeled_t = torch.from_numpy(X_unlabeled.astype(np.float32))
    n = len(X_unlabeled_t)
    batch_size = scfg["batch_size"]
    for _ in range(scfg["epochs"]):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 2:
                continue
            xb = X_unlabeled_t[idx].to(device)
            v1, v2 = augment_pair(xb)
            z1, z2 = model(v1), model(v2)
            loss = nt_xent_loss(z1, z2, temperature=scfg["temperature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Linear probe on frozen encoder using the (scarce) labeled data.
    for p in model.encoder.parameters():
        p.requires_grad = False
    linear = torch.nn.Linear(cfg["training"]["embedding_dim"], n_classes).to(device)
    lin_opt = torch.optim.Adam(linear.parameters(), lr=cfg["training"]["learning_rate"])
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    for _ in range(scfg["linear_probe_epochs"]):
        model.encoder.eval()
        with torch.no_grad():
            emb = model.encoder(X_train_t.to(device))
        logits = linear(emb)
        loss = torch.nn.functional.cross_entropy(logits, y_train_t.to(device))
        lin_opt.zero_grad()
        loss.backward()
        lin_opt.step()

    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
        emb_test = model.encoder(X_test_t)
        y_pred = linear(emb_test).argmax(dim=1).cpu().numpy()
    return compute_metrics(y_test, y_pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", required=True)
    parser.add_argument("--label-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    X, y, load_hp, class_names = load_windows(cfg["data"]["processed_path"])
    n_classes = len(class_names)

    if cfg.get("cross_load", {}).get("enabled", False):
        train_idx, test_idx = cross_load_split(
            load_hp, cfg["cross_load"]["train_loads"], cfg["cross_load"]["test_loads"]
        )
        X_train_full, y_train_full = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
    else:
        X_train_full, X_test, y_train_full, y_test = train_test_split_stratified(
            X, y, test_size=cfg["experiment"]["test_size"], seed=args.seed
        )

    sub_idx = stratified_label_fraction_split(y_train_full, args.label_fraction, args.seed)
    X_train, y_train = X_train_full[sub_idx], y_train_full[sub_idx]

    fs = cfg["data"]["sampling_rate_hz"]
    method = args.method

    if method in ("svm", "random_forest"):
        metrics = run_classical(method, X_train, y_train, X_test, y_test, args.seed, fs)
    elif method == "cnn1d":
        metrics = run_cnn1d(X_train, y_train, X_test, y_test, n_classes, cfg, args.seed)
    elif method == "lstm":
        metrics = run_lstm(X_train, y_train, X_test, y_test, n_classes, cfg, args.seed)
    elif method == "protonet":
        metrics = run_protonet(X_train, y_train, X_test, y_test, cfg, args.seed)
    elif method == "simclr_linear":
        metrics = run_simclr_linear(X_train_full, X_train, y_train, X_test, y_test, n_classes, cfg, args.seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    result = {
        "method": method,
        "label_fraction": args.label_fraction,
        "seed": args.seed,
        **metrics,
    }
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
