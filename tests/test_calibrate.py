"""Tests for the calibration maths. No model, no network, runs in milliseconds."""

from __future__ import annotations

import math
import random

import pytest

from nanojev.calibrate import (
    apply_temperature,
    brier_score,
    evaluate,
    expected_calibration_error,
    fit_temperature,
    nll_score,
)


def test_brier_is_zero_for_perfect_confident_predictions():
    probs = [[1.0, 0.0], [0.0, 1.0]]
    assert brier_score(probs, [0, 1]) == pytest.approx(0.0, abs=1e-9)


def test_brier_is_maximal_for_confidently_wrong_predictions():
    probs = [[1.0, 0.0]]
    # Squared error of 1 on each of the two classes.
    assert brier_score(probs, [1]) == pytest.approx(2.0, abs=1e-6)


def test_uniform_guessing_scores_between_the_extremes():
    probs = [[0.5, 0.5]]
    assert brier_score(probs, [0]) == pytest.approx(0.5, abs=1e-9)
    assert nll_score(probs, [0]) == pytest.approx(math.log(2), abs=1e-9)


def test_temperature_one_is_the_identity():
    probs = [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]
    for got, want in zip(apply_temperature(probs, 1.0), probs):
        assert got == pytest.approx(want, abs=1e-9)


def test_high_temperature_flattens_toward_uniform():
    """Raising the temperature must move monotonically toward uniform.

    Asserting the direction rather than a fixed tolerance: the limit is uniform
    but the approach is gradual, so any single cutoff would be arbitrary.
    """
    peaked = [[0.98, 0.01, 0.01]]
    uniform = 1 / 3

    def distance_from_uniform(temperature: float) -> float:
        return max(abs(p - uniform) for p in apply_temperature(peaked, temperature)[0])

    distances = [distance_from_uniform(t) for t in (1.0, 2.0, 10.0, 100.0, 1000.0)]
    assert distances == sorted(distances, reverse=True)
    assert distances[-1] < 1e-2


def test_low_temperature_sharpens_toward_one_hot():
    sharpened = apply_temperature([[0.6, 0.3, 0.1]], 0.01)[0]
    assert sharpened[0] == pytest.approx(1.0, abs=1e-6)


def test_temperature_scaling_never_changes_the_winner():
    """The property the whole approach rests on: it rescales, it cannot reorder."""
    rng = random.Random(0)
    probs = []
    for _ in range(200):
        raw = [rng.random() for _ in range(5)]
        total = sum(raw)
        probs.append([v / total for v in raw])

    before = [max(range(5), key=row.__getitem__) for row in probs]
    for temperature in (0.2, 0.5, 2.0, 7.0):
        after = [
            max(range(5), key=row.__getitem__)
            for row in apply_temperature(probs, temperature)
        ]
        assert after == before


def test_ece_is_zero_when_confidence_matches_accuracy():
    # Claim 80% on ten cases and get exactly eight of them right.
    probs = [[0.8, 0.2]] * 10
    labels = [0] * 8 + [1] * 2
    ece, _ = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_ece_catches_overconfidence():
    # Claim 99% and be right only half the time: a gap of roughly 0.49.
    probs = [[0.99, 0.01]] * 10
    labels = [0] * 5 + [1] * 5
    ece, _ = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.49, abs=1e-6)


def test_fit_temperature_recovers_a_known_distortion():
    """Sharpen an honest set of predictions, then check the fit undoes it."""
    rng = random.Random(7)
    honest, labels = [], []
    for _ in range(600):
        p = rng.uniform(0.35, 0.95)
        honest.append([p, 1 - p])
        labels.append(0 if rng.random() < p else 1)

    overconfident = apply_temperature(honest, 0.35)
    recovered = fit_temperature(overconfident, labels, objective="brier")
    # Undoing a 0.35 sharpening means a temperature near 1/0.35 ~= 2.86.
    assert recovered == pytest.approx(1 / 0.35, rel=0.35)


def test_fitting_improves_calibration_without_touching_accuracy():
    rng = random.Random(11)
    honest, labels = [], []
    for _ in range(600):
        p = rng.uniform(0.4, 0.9)
        honest.append([p, 1 - p])
        labels.append(0 if rng.random() < p else 1)
    overconfident = apply_temperature(honest, 0.3)

    temperature = fit_temperature(overconfident, labels, objective="brier")
    before = evaluate(overconfident, labels, temperature=1.0)
    after = evaluate(overconfident, labels, temperature=temperature)

    assert after.ece < before.ece
    assert after.brier <= before.brier
    assert after.accuracy == pytest.approx(before.accuracy)


def test_probabilities_are_renormalised_defensively():
    # Rows that do not sum to 1 should be handled, not silently mis-scored.
    assert brier_score([[2.0, 2.0]], [0]) == pytest.approx(0.5, abs=1e-9)
