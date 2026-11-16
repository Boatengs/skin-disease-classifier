"""Turn saved predictions into results/summary.json, results/RESULTS.md and figures.

    python -m skinclf.report --data data --runs runs --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASS_NAMES, CLASSES, MEL
from .data import load_metadata
from .metrics import (
    cluster_bootstrap,
    expected_calibration_error,
    fit_temperature,
    nll,
    per_class,
    probs_from_logits,
    sens_spec,
    summary,
    threshold_for_sensitivity,
)

TRIAGE_TARGET = 0.90
MAIN_SEED = 42
CI_KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "melanoma_recall", "melanoma_auroc"]


def evaluate_run(run: Path, meta: pd.DataFrame, n_boot: int) -> dict:
    p = np.load(run / "predictions.npz")
    y = meta["label"].to_numpy()
    lesions = meta["lesion_id"].to_numpy()
    val_rows, test_rows = p["val_rows"], p["test_rows"]
    yv, yt = y[val_rows], y[test_rows]

    temp = fit_temperature(p["val_logits"], yv)
    raw = probs_from_logits(p["test_logits"])
    cal = probs_from_logits(p["test_logits"], temp)
    point = summary(cal, yt)

    groups = lesions[test_rows]
    cis = {k: cluster_bootstrap(lambda r, k=k: summary(cal[r], yt[r])[k], groups, n_boot=n_boot) for k in CI_KEYS}

    # Melanoma triage: pick the threshold on validation, then apply it unchanged to test.
    thr = threshold_for_sensitivity(probs_from_logits(p["val_logits"], temp)[:, MEL], yv == MEL, TRIAGE_TARGET)
    sens, spec = sens_spec(cal[:, MEL], yt == MEL, thr)
    sens_ci = cluster_bootstrap(lambda r: sens_spec(cal[r, MEL], yt[r] == MEL, thr)[0], groups, n_boot=n_boot)
    spec_ci = cluster_bootstrap(lambda r: sens_spec(cal[r, MEL], yt[r] == MEL, thr)[1], groups, n_boot=n_boot)

    out = {
        "split": json.loads((run / "split.json").read_text()),
        "history": json.loads((run / "history.json").read_text()),
        "temperature": temp,
        "test": point,
        "test_ci95": cis,
        "test_uncalibrated": {"ece": expected_calibration_error(raw, yt), "nll": nll(p["test_logits"], yt)},
        "test_calibrated_nll": nll(p["test_logits"], yt, temp),
        "per_class": per_class(yt, cal.argmax(1)),
        "melanoma_triage": {
            "target_sensitivity_on_validation": TRIAGE_TARGET,
            "threshold": thr,
            "test_sensitivity": sens, "test_sensitivity_ci95": sens_ci,
            "test_specificity": spec, "test_specificity_ci95": spec_ci,
        },
    }

    return out


def leakage_table(path: Path) -> dict:
    rows = pd.DataFrame(json.loads(path.read_text())["rows"])
    cols = ["accuracy", "balanced_accuracy", "melanoma_recall", "nearest_train_neighbour_same_lesion",
            "test_images_with_lesion_in_train"]
    agg = rows.groupby("split")[cols].agg(["mean", "std", "min", "max"])
    wide = rows.pivot(index="seed", columns="split", values="balanced_accuracy")
    gap = wide["image"] - wide["lesion"]
    return {
        "seeds": int(rows["seed"].nunique()),
        "by_split": {s: {c: {m: float(agg.loc[s, (c, m)]) for m in ["mean", "std", "min", "max"]}
                         for c in cols} for s in agg.index},
        "balanced_accuracy_gap": {"mean": float(gap.mean()), "min": float(gap.min()), "max": float(gap.max()),
                                  "seeds_image_split_higher": int((gap > 0).sum())},
    }


def figures(results: dict, meta: pd.DataFrame, runs: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    from .metrics import probs_from_logits as pfl

    run = results["runs"][f"lesion-s{MAIN_SEED}"]
    p = np.load(runs / f"lesion-s{MAIN_SEED}" / "predictions.npz")
    y = meta["label"].to_numpy()[p["test_rows"]]
    cal = pfl(p["test_logits"], run["temperature"])
    raw = pfl(p["test_logits"])

    cm = confusion_matrix(y, cal.argmax(1), labels=range(len(CLASSES)))
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                    color="white" if pct[i, j] > 55 else "#1b2028", fontsize=9)
    ticks = [c.upper() for c in CLASSES]
    ax.set_xticks(range(len(CLASSES)), ticks)
    ax.set_yticks(range(len(CLASSES)), ticks)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Lesion split test set, row %")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    bins = np.linspace(0, 1, 11)
    for probs, label in [(raw, "Before temperature scaling"), (cal, "After temperature scaling")]:
        conf, correct = probs.max(1), probs.argmax(1) == y
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 10:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        ax.plot(xs, ys, marker="o", label=label)
    ax.plot([0, 1], [0, 1], ls="--", color="#999", label="Perfect calibration")
    ax.set_xlabel("Model confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability, lesion split test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150)
    plt.close(fig)

    if "leakage" in results:
        lk = json.loads((runs / "leakage" / "leakage.json").read_text())["rows"]
        df = pd.DataFrame(lk)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k, (name, color) in enumerate([("image", "#c8553d"), ("lesion", "#2f6fe4")]):
            vals = df[df.split == name].sort_values("seed")["balanced_accuracy"].to_numpy()
            ax.scatter(np.full(len(vals), k) + np.linspace(-0.12, 0.12, len(vals)), vals * 100, color=color)
            ax.hlines(vals.mean() * 100, k - 0.25, k + 0.25, color=color, lw=2)
        ax.set_xticks([0, 1], ["Image split\n(original method)", "Lesion split"])
        ax.set_ylabel("Balanced accuracy, %")
        ax.set_title(f"Frozen features, {df.seed.nunique()} random seeds")
        fig.tight_layout()
        fig.savefig(out / "leakage.png", dpi=150)
        plt.close(fig)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def ci(pair) -> str:
    return f"{pair[0] * 100:.1f} to {pair[1] * 100:.1f}"


def markdown(results: dict) -> str:
    L = ["# Results", "", "Generated by `python -m skinclf.report` from the saved predictions in this run. "
         "Intervals are 95% percentile bootstrap intervals that resample whole lesions.", ""]
    L += [f"## Fine-tuned EfficientNetB0, same recipe, two splits (seed {MAIN_SEED})", "",
          "| Metric | Image split (original method) | Lesion split |", "|---|---|---|"]
    im, le = results["runs"][f"image-s{MAIN_SEED}"], results["runs"][f"lesion-s{MAIN_SEED}"]
    for k, label in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
                     ("macro_f1", "Macro F1"), ("melanoma_recall", "Melanoma recall"),
                     ("melanoma_auroc", "Melanoma AUROC")]:
        if k == "melanoma_auroc":
            L.append(f"| {label} | {im['test'][k]:.3f} ({im['test_ci95'][k][0]:.3f} to {im['test_ci95'][k][1]:.3f}) | "
                     f"{le['test'][k]:.3f} ({le['test_ci95'][k][0]:.3f} to {le['test_ci95'][k][1]:.3f}) |")
            continue
        L.append(f"| {label} | {pct(im['test'][k])} ({ci(im['test_ci95'][k])}) | "
                 f"{pct(le['test'][k])} ({ci(le['test_ci95'][k])}) |")
    L.append(f"| Test images whose lesion is also in training | "
             f"{im['split']['test_images_with_lesion_in_train']} of {im['split']['images']['test']} | "
             f"{le['split']['test_images_with_lesion_in_train']} of {le['split']['images']['test']} |")
    L.append("")
    seeds = sorted({n.split("-s")[1] for n in results["runs"]}, key=int)
    pairs = [(results["runs"][f"image-s{sd}"], results["runs"][f"lesion-s{sd}"]) for sd in seeds
             if f"image-s{sd}" in results["runs"] and f"lesion-s{sd}" in results["runs"]]
    if len(pairs) > 1:
        getters = [
            ("Accuracy", lambda r: r["test"]["accuracy"], True),
            ("Balanced accuracy", lambda r: r["test"]["balanced_accuracy"], True),
            ("Melanoma AUROC", lambda r: r["test"]["melanoma_auroc"], False),
            ("Malignant vs benign AUROC", lambda r: r["test"]["malignant_auroc"], False),
            ("Melanoma triage sensitivity", lambda r: r["melanoma_triage"]["test_sensitivity"], True),
            ("Melanoma triage specificity", lambda r: r["melanoma_triage"]["test_specificity"], True),
        ]
        L += [f"## All {len(pairs)} fine-tuning seeds", "",
              "Each seed changes the split, the initial head weights and the batch order. "
              "Triage uses a melanoma threshold set on validation to catch 90% of melanomas.", "",
              "| Metric | Image split, per seed | Lesion split, per seed | Mean gap (image minus lesion) | "
              "Seeds where image split is higher |", "|---|---|---|---|---|"]
        for label, get, as_pct in getters:
            a = [get(x) for x, _ in pairs]
            b = [get(y) for _, y in pairs]
            fmt = (lambda v: f"{v * 100:.1f}") if as_pct else (lambda v: f"{v:.3f}")
            gap = float(np.mean(np.subtract(a, b)))
            gap_s = f"{gap * 100:+.1f} points" if as_pct else f"{gap:+.3f}"
            higher = sum(x > y for x, y in zip(a, b, strict=True))
            L.append(f"| {label} | {', '.join(fmt(v) for v in a)} | {', '.join(fmt(v) for v in b)} | "
                     f"{gap_s} | {higher} of {len(pairs)} |")
        L += ["", f"Seeds in order: {', '.join(seeds)}. Triage sensitivity and specificity trade off against each "
              "other, so the two triage rows should be read together; AUROC is the threshold-free comparison.", ""]
    if "leakage" in results:
        lk = results["leakage"]
        b = lk["by_split"]
        L += ["## Frozen-feature check across seeds", "",
              f"Frozen ImageNet EfficientNetB0 features with a logistic regression, {lk['seeds']} seeds per split.", "",
              "| | Image split | Lesion split |", "|---|---|---|",
              f"| Balanced accuracy, mean (min to max) | {pct(b['image']['balanced_accuracy']['mean'])} "
              f"({pct(b['image']['balanced_accuracy']['min'])} to {pct(b['image']['balanced_accuracy']['max'])}) | "
              f"{pct(b['lesion']['balanced_accuracy']['mean'])} "
              f"({pct(b['lesion']['balanced_accuracy']['min'])} to {pct(b['lesion']['balanced_accuracy']['max'])}) |",
              f"| Test photos whose nearest training photo is the same lesion | "
              f"{pct(b['image']['nearest_train_neighbour_same_lesion']['mean'])} | "
              f"{pct(b['lesion']['nearest_train_neighbour_same_lesion']['mean'])} |", "",
              f"The image split scored higher on {lk['balanced_accuracy_gap']['seeds_image_split_higher']} of "
              f"{lk['seeds']} seeds, by {lk['balanced_accuracy_gap']['mean'] * 100:.1f} percentage points on average.", ""]
    L += ["## Lesion split model in detail", "", "| Class | Test images | Recall | Precision |", "|---|---|---|---|"]
    for c in CLASSES:
        r = le["per_class"][c]
        L.append(f"| {CLASS_NAMES[c]} | {r['support']} | {pct(r['recall'])} | {pct(r['precision'])} |")
    t = le["melanoma_triage"]
    L += ["", "### Melanoma triage operating point", "",
          f"Threshold on melanoma probability chosen on validation to catch {pct(t['target_sensitivity_on_validation'])} "
          f"of melanomas, then applied unchanged to test.", "",
          f"- Test sensitivity: {pct(t['test_sensitivity'])} ({ci(t['test_sensitivity_ci95'])})",
          f"- Test specificity: {pct(t['test_specificity'])} ({ci(t['test_specificity_ci95'])})", "",
          "### Calibration", "",
          f"- Temperature fitted on validation: {le['temperature']:.2f}",
          f"- Expected calibration error before: {pct(le['test_uncalibrated']['ece'])}, after: {pct(le['test']['ece'])}",
          f"- Test negative log likelihood before: {le['test_uncalibrated']['nll']:.3f}, "
          f"after: {le['test_calibrated_nll']:.3f}", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(args.data / "HAM10000_metadata.csv")
    names = sorted(d.name for d in args.runs.iterdir() if (d / "predictions.npz").exists())
    results = {"main_seed": MAIN_SEED,
               "runs": {n: evaluate_run(args.runs / n, meta, args.boot if n.endswith(f"s{MAIN_SEED}") else 200)
                        for n in names}}
    if (args.runs / "leakage" / "leakage.json").exists():
        results["leakage"] = leakage_table(args.runs / "leakage" / "leakage.json")
    (args.out / "summary.json").write_text(json.dumps(results, indent=2))
    (args.out / "RESULTS.md").write_text(markdown(results))
    figures(results, meta, args.runs, args.out)
    print(markdown(results))


if __name__ == "__main__":
    main()
