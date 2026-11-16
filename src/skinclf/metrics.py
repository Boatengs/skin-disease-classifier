"""Evaluation metrics, calibration and lesion-level bootstrap intervals."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from . import CLASSES, MALIGNANT, MEL

MALIGNANT_IDX = [CLASSES.index(c) for c in MALIGNANT]


def per_class(y: np.ndarray, pred: np.ndarray) -> dict[str, dict[str, float]]:
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    out = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        support = cm[i].sum()
        predicted = cm[:, i].sum()
        out[c] = {
            "support": int(support),
            "recall": float(tp / support) if support else float("nan"),
            "precision": float(tp / predicted) if predicted else float("nan"),
        }
    return out


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    lp = log_softmax(logits / temperature, axis=1)
    return float(-lp[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single temperature minimising validation negative log likelihood."""
    res = minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def threshold_for_sensitivity(score: np.ndarray, positive: np.ndarray, target: float) -> float:
    """Largest threshold whose sensitivity on these cases is at least `target`."""
    pos = np.sort(score[positive.astype(bool)])
    if len(pos) == 0:
        raise ValueError("no positive cases to set a threshold")
    k = int(np.floor((1.0 - target) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def sens_spec(score: np.ndarray, positive: np.ndarray, threshold: float) -> tuple[float, float]:
    positive = positive.astype(bool)
    flagged = score >= threshold
    sens = flagged[positive].mean() if positive.any() else float("nan")
    spec = (~flagged[~positive]).mean() if (~positive).any() else float("nan")
    return float(sens), float(spec)


def summary(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Headline numbers for one set of predictions."""
    pred = probs.argmax(axis=1)
    mel = y == MEL
    malignant = np.isin(y, MALIGNANT_IDX)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=range(len(CLASSES)), zero_division=0),
        "melanoma_recall": float((pred[mel] == MEL).mean()) if mel.any() else float("nan"),
        "melanoma_auroc": roc_auc_score(mel, probs[:, MEL]) if 0 < mel.sum() < len(y) else float("nan"),
        "malignant_auroc": (
            roc_auc_score(malignant, probs[:, MALIGNANT_IDX].sum(axis=1))
            if 0 < malignant.sum() < len(y)
            else float("nan")
        ),
        "ece": expected_calibration_error(probs, y),
    }
    return {k: float(v) for k, v in out.items()}


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval resampling whole lesions, not single images.

    `stat` receives an array of row indices and returns a number.
    """
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([members[g] for g in pick])
        v = stat(rows)
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return softmax(logits / temperature, axis=1)
