import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from skinclf import MEL
from skinclf.metrics import (
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


def test_ece_zero_for_perfectly_calibrated_bins():
    probs = np.array([[0.8, 0.2]] * 10)
    y = np.array([0] * 8 + [1] * 2)
    assert expected_calibration_error(probs, y) == pytest.approx(0.0)


def test_ece_known_value():
    probs = np.array([[0.9, 0.1]] * 10)
    y = np.array([0] * 5 + [1] * 5)
    assert expected_calibration_error(probs, y) == pytest.approx(0.4)


def test_temperature_recovers_known_scale():
    rng = np.random.default_rng(1)
    true_logits = rng.normal(size=(4000, 7)) * 2
    y = np.array([rng.choice(7, p=p) for p in probs_from_logits(true_logits)])
    overconfident = true_logits * 3
    t = fit_temperature(overconfident, y)
    assert t == pytest.approx(3.0, rel=0.15)
    assert nll(overconfident, y, t) < nll(overconfident, y)


def test_threshold_hits_target_sensitivity():
    rng = np.random.default_rng(2)
    score = rng.random(1000)
    pos = rng.random(1000) < 0.2
    thr = threshold_for_sensitivity(score, pos, 0.9)
    sens, _ = sens_spec(score, pos, thr)
    assert sens >= 0.9
    assert sens < 0.92


def test_summary_matches_sklearn():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 7, 500)
    probs = rng.dirichlet(np.ones(7), 500)
    s = summary(probs, y)
    assert s["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, probs.argmax(1)))
    mel = y == MEL
    assert s["melanoma_recall"] == pytest.approx((probs.argmax(1)[mel] == MEL).mean())


def test_per_class_counts():
    y = np.array([0, 0, 1, 1, 1])
    pred = np.array([0, 1, 1, 1, 0])
    pc = per_class(y, pred)
    assert pc["mel"]["support"] == 2 and pc["mel"]["recall"] == pytest.approx(0.5)
    assert pc["nv"]["precision"] == pytest.approx(2 / 3)


def test_cluster_bootstrap_resamples_groups_and_is_wider_than_naive():
    # Ten groups of 50 identical outcomes: lesions behave as single units.
    groups = np.repeat(np.arange(10), 50)
    values = np.repeat(np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0], float), 50)
    lo, hi = cluster_bootstrap(lambda r: values[r].mean(), groups, n_boot=2000, seed=0)
    assert lo < 0.4 and hi > 0.8
    rng = np.random.default_rng(0)
    naive = [values[rng.integers(0, 500, 500)].mean() for _ in range(2000)]
    assert (hi - lo) > (np.quantile(naive, 0.975) - np.quantile(naive, 0.025)) * 2
