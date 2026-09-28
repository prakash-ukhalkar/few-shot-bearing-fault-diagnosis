"""Shared training loop helpers for torch-based models on CPU."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_classifier(model, X_train, y_train, epochs, batch_size, lr, weight_decay=1e-4, device="cpu"):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict(model, X, device="cpu", batch_size=256):
    model.eval()
    model.to(device)
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    preds = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i : i + batch_size].to(device)
        logits = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)
