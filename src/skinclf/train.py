"""Fine-tune EfficientNetB0 on one split and save validation and test logits.

The recipe matches the original notebook (class-weighted cross entropy, AdamW at
1e-3, cosine schedule, flips, rotation and colour jitter, last three blocks
unfrozen). Two things differ on purpose: the best epoch is chosen by validation
balanced accuracy instead of plain accuracy, and the split can be by lesion.

    python -m skinclf.train --split lesion --data data --out runs/lesion
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

from . import CLASSES
from .data import build_cache, download, load_metadata
from .model import build_model
from .splits import SPLITTERS, images_with_seen_lesion, shared_lesions

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CachedImages(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, rows: np.ndarray, train: bool):
        self.images, self.labels, self.rows = images, labels, rows
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
               T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)] if train else []
        self.tf = T.Compose([T.ToImage(), *aug, T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        return self.tf(np.array(self.images[r])), int(self.labels[r])


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Same weighting as the notebook: inverse frequency, scaled to sum to the class count."""
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    w = 1.0 / counts
    w = w / w.sum() * len(CLASSES)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    return np.concatenate([model(x).numpy() for x, _ in loader])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SPLITTERS), required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    download(args.data)
    meta = load_metadata(args.data / "HAM10000_metadata.csv")
    images = build_cache(meta, args.data / "images", args.data / "cache224.npy")
    labels = meta["label"].to_numpy()
    split = SPLITTERS[args.split](meta, seed=args.seed)

    split_info = {
        "split": args.split,
        "seed": args.seed,
        "images": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "lesions": {k: int(meta["lesion_id"].iloc[v].nunique()) for k, v in
                    {"train": split.train, "val": split.val, "test": split.test}.items()},
        "lesions_shared_train_test": shared_lesions(meta, split.train, split.test),
        "test_images_with_lesion_in_train": images_with_seen_lesion(meta, split.train, split.test),
    }
    print(json.dumps(split_info, indent=2), flush=True)

    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(CachedImages(images, labels, split.train, True), args.batch_size,
                          shuffle=True, generator=g, num_workers=2, persistent_workers=True)
    val_dl = DataLoader(CachedImages(images, labels, split.val, False), 64, num_workers=2)
    test_dl = DataLoader(CachedImages(images, labels, split.test, False), 64, num_workers=2)

    model = build_model(pretrained=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights(labels[split.train]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    history, best = [], -1.0
    y_val = labels[split.val]
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        val_logits = predict(model, val_dl)
        val_bacc = balanced_accuracy_score(y_val, val_logits.argmax(1))
        val_acc = float((val_logits.argmax(1) == y_val).mean())
        row = {"epoch": epoch, "train_loss": total / n, "val_acc": val_acc,
               "val_balanced_acc": float(val_bacc), "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_bacc > best:
            best = val_bacc
            torch.save(model.state_dict(), args.out / "model.pth")

    model.load_state_dict(torch.load(args.out / "model.pth"))
    np.savez_compressed(
        args.out / "predictions.npz",
        val_logits=predict(model, val_dl), val_rows=split.val,
        test_logits=predict(model, test_dl), test_rows=split.test,
        train_rows=split.train,
    )
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    (args.out / "split.json").write_text(json.dumps(split_info, indent=2))


if __name__ == "__main__":
    main()
