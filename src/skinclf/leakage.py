"""Measure how much an image-level split inflates results, across many seeds.

Fine-tuning a network twice per seed is too slow on CPU, so this experiment
freezes the ImageNet EfficientNetB0, extracts one 1280-dimension embedding per
image, and fits a class-weighted logistic regression on each split. It answers
one narrow question: with everything else fixed, how much does letting photos of
the same lesion appear in train and test change the score?

    python -m skinclf.leakage --data data --out runs/leakage --seeds 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import build_cache, download, load_metadata
from .metrics import summary
from .model import build_model
from .splits import SPLITTERS, images_with_seen_lesion
from .train import CachedImages


@torch.no_grad()
def embed(images: np.ndarray, threads: int = 4) -> np.ndarray:
    torch.set_num_threads(threads)
    net = build_model(pretrained=True, trainable_blocks=0).eval()
    rows = np.arange(len(images))
    dl = torch.utils.data.DataLoader(CachedImages(images, np.zeros(len(images), int), rows, False),
                                     batch_size=64, num_workers=2)
    out = []
    for i, (x, _) in enumerate(dl):
        out.append(torch.flatten(net.avgpool(net.features(x)), 1).numpy())
        if i % 20 == 0:
            print(f"embedded {i * 64}/{len(images)}", flush=True)
    return np.concatenate(out)


def nearest_same_lesion(feats: np.ndarray, lesions: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    """Share of test images whose nearest training image (cosine) is the same lesion."""
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    hits = 0
    for start in range(0, len(test), 512):
        chunk = test[start:start + 512]
        nn_idx = train[(f[chunk] @ f[train].T).argmax(axis=1)]
        hits += int((lesions[nn_idx] == lesions[chunk]).sum())
    return hits / len(test)


def run(feats: np.ndarray, meta, seeds: int) -> dict:
    y = meta["label"].to_numpy()
    lesions = meta["lesion_id"].to_numpy()
    rows = []
    for seed in range(seeds):
        for name, splitter in SPLITTERS.items():
            s = splitter(meta, seed=seed)
            scaler = StandardScaler().fit(feats[s.train])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
            clf.fit(scaler.transform(feats[s.train]), y[s.train])
            probs = clf.predict_proba(scaler.transform(feats[s.test]))
            row = {"seed": seed, "split": name, **summary(probs, y[s.test]),
                   "test_images_with_lesion_in_train": images_with_seen_lesion(meta, s.train, s.test),
                   "test_images": len(s.test),
                   "nearest_train_neighbour_same_lesion": nearest_same_lesion(feats, lesions, s.train, s.test)}
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    return {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.data / "cache224.npy").exists():
        download(args.data)
    meta = load_metadata(args.data / "HAM10000_metadata.csv")
    images = build_cache(meta, args.data / "images", args.data / "cache224.npy")
    feats_path = args.out / "embeddings.npy"
    feats = np.load(feats_path) if feats_path.exists() else embed(images)
    np.save(feats_path, feats)
    (args.out / "leakage.json").write_text(json.dumps(run(feats, meta, args.seeds), indent=2))


if __name__ == "__main__":
    main()
