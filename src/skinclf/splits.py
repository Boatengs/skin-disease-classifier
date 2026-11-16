"""Train, validation and test splits.

`image_split` reproduces the original notebook: a stratified random split over
images. HAM10000 often holds several photos of the same lesion, so that split can
put one photo of a lesion in training and another photo of the same lesion in test.

`lesion_split` keeps every photo of a lesion in one partition, stratified by
diagnosis, so the test set measures performance on lesions the model has not seen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def frame(self, meta: pd.DataFrame) -> pd.Series:
        part = pd.Series("", index=meta.index, dtype=object)
        part.iloc[self.train] = "train"
        part.iloc[self.val] = "val"
        part.iloc[self.test] = "test"
        return part


def image_split(meta: pd.DataFrame, seed: int = 42) -> Split:
    """Original notebook split: 80/10/10 over images, stratified by label."""
    idx = np.arange(len(meta))
    y = meta["dx"].to_numpy()
    train, temp = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed, stratify=y[temp])
    return Split(np.sort(train), np.sort(val), np.sort(test))


def lesion_split(meta: pd.DataFrame, seed: int = 42) -> Split:
    """Roughly 80/10/10 with every lesion confined to one partition."""
    idx = np.arange(len(meta))
    y = meta["dx"].to_numpy()
    groups = meta["lesion_id"].to_numpy()

    outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    rest, test = next(outer.split(idx, y, groups))

    inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=seed + 1)
    tr, va = next(inner.split(rest, y[rest], groups[rest]))
    return Split(np.sort(rest[tr]), np.sort(rest[va]), np.sort(test))


def shared_lesions(meta: pd.DataFrame, a: np.ndarray, b: np.ndarray) -> int:
    """Number of lesions with at least one image in both index sets."""
    la = set(meta["lesion_id"].iloc[a])
    lb = set(meta["lesion_id"].iloc[b])
    return len(la & lb)


def images_with_seen_lesion(meta: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> int:
    """Test images whose lesion also has an image in training."""
    seen = set(meta["lesion_id"].iloc[train])
    return int(meta["lesion_id"].iloc[test].isin(seen).sum())


SPLITTERS = {"image": image_split, "lesion": lesion_split}
