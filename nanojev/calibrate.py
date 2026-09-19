"""Calibration: making the probabilities mean what they say.

This is the part that matters. Any model can emit a number between 0 and 1; the
claim worth testing is that when it says 0.7 it is right about 70% of the time.
Our smoke test opened with `is_urgent = 1.000`, which is a model asserting
certainty it has not earned, and that is precisely the failure that makes people
put a human back in the loop.

The commercial approach is to train for this with RL against a proper scoring
rule. We are not going to run RL on a laptop, and we do not need to: post-hoc
temperature scaling (Guo et al. 2017, arXiv:1706.04599) recovers most of the
available calibration from a few hundred labelled examples and a single scalar.
It cannot make the model smarter, only more honest about what it knows, which
happens to be the property we actually want to gate on.

Scoring rules used here:
  Brier  - mean squared error against the one-hot truth. Proper: uniquely
           minimised by reporting your true beliefs.
  NLL    - log score. Also proper, punishes confident errors far harder.
  ECE    - not a scoring rule but the readable one: average gap between stated
           confidence and observed accuracy, bucketed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationReport:
    temperature: float
    accuracy: float
    brier: float
    nll: float
    ece: float
    bins: list[tuple[float, float, int]]  # (mean confidence, accuracy, count)

    def render(self, title: str = "") -> str:
        head = f"{title}\n" if title else ""
        lines = [
            f"{head}  temperature {self.temperature:.3f}",
            f"  accuracy    {self.accuracy:.4f}",
            f"  brier       {self.brier:.4f}   (lower is better)",
            f"  nll         {self.nll:.4f}   (lower is better)",
            f"  ece         {self.ece:.4f}   (lower is better)",
            "",
            f"  {'confidence':>12} {'accuracy':>9} {'n':>6}   reliability",
        ]
        for conf, acc, n in self.bins:
            if n == 0:
                continue
            gap = acc - conf
            bar = "#" * int(round(acc * 30))
            lines.append(f"  {conf:>12.3f} {acc:>9.3f} {n:>6}   {bar:<30} {gap:+.3f}")
        return "\n".join(lines)


def _as_arrays(probs: list[list[float]], labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    return p, np.asarray(labels, dtype=np.int64)


def brier_score(probs, labels) -> float:
    """Multiclass Brier: mean over samples of the squared error vs one-hot."""
    p, y = _as_arrays(probs, labels)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def nll_score(probs, labels) -> float:
    p, y = _as_arrays(probs, labels)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def expected_calibration_error(probs, labels, n_bins: int = 10):
    """ECE over the top-1 confidence, plus the per-bin reliability data."""
    p, y = _as_arrays(probs, labels)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: list[tuple[float, float, int]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            bins.append((float((lo + hi) / 2), 0.0, 0))
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (n / len(y)) * abs(bin_acc - bin_conf)
        bins.append((bin_conf, bin_acc, n))
    return float(ece), bins


def apply_temperature(probs, temperature: float) -> list[list[float]]:
    """Re-temper a distribution. T>1 softens, T<1 sharpens, T=1 is identity."""
    p, _ = _as_arrays(probs, [0] * len(probs))
    logits = np.log(p) / max(temperature, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=1, keepdims=True)).tolist()


def fit_temperature(
    probs, labels, objective: str = "brier", lo: float = 0.05, hi: float = 20.0
) -> float:
    """Find the single scalar that best calibrates these predictions.

    Golden-section search over log-temperature. One parameter fit on held-out
    data, which is deliberately the least powerful thing that could work: it
    cannot change which option wins, only how sure the model claims to be.
    """
    score = brier_score if objective == "brier" else nll_score

    def f(log_t: float) -> float:
        return score(apply_temperature(probs, math.exp(log_t)), labels)

    a, b = math.log(lo), math.log(hi)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
        if abs(b - a) < 1e-6:
            break
    return float(math.exp((a + b) / 2))


def evaluate(probs, labels, temperature: float = 1.0, n_bins: int = 10) -> CalibrationReport:
    scaled = apply_temperature(probs, temperature) if temperature != 1.0 else probs
    p, y = _as_arrays(scaled, labels)
    ece, bins = expected_calibration_error(scaled, labels, n_bins=n_bins)
    return CalibrationReport(
        temperature=temperature,
        accuracy=float((p.argmax(axis=1) == y).mean()),
        brier=brier_score(scaled, labels),
        nll=nll_score(scaled, labels),
        ece=ece,
        bins=bins,
    )
