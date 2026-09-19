"""Measure, then fix, how honest the probabilities are.

Protocol: run the model over a real labelled dataset, split the predictions into
a fit half and a test half, fit one temperature on the fit half only, and report
test-set metrics before and after. Fitting and reporting on the same data would
make any method look good, so we do not.

banking77 is a deliberate choice: it is literally a routing task (customer
message -> intent), which is the workload System One models are pitched at, and
its classes are close enough together to make confident errors likely.
"""

from __future__ import annotations

import argparse
import random
import time

from datasets import load_dataset

from nanojev import Choice, EncoderSystemOne, SystemOne
from nanojev.calibrate import evaluate, fit_temperature


def load_banking(n_classes: int, n_samples: int, seed: int):
    # The canonical PolyAI/banking77 repo is script-based, which recent
    # `datasets` refuses to execute; the legacy mirror ships plain parquet.
    ds = load_dataset("legacy-datasets/banking77", split="test")
    names = ds.features["label"].names

    rng = random.Random(seed)
    keep = sorted(rng.sample(range(len(names)), n_classes))
    keep_set = set(keep)
    label_names = [names[i].replace("_", " ") for i in keep]
    remap = {orig: new for new, orig in enumerate(keep)}

    rows = [r for r in ds if r["label"] in keep_set]
    rng.shuffle(rows)
    rows = rows[:n_samples]
    return label_names, [(r["text"], remap[r["label"]]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=20)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--rotations", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=("decoder", "encoder"), default="decoder")
    args = ap.parse_args()

    label_names, rows = load_banking(args.classes, args.samples, args.seed)
    print(f"banking77: {len(label_names)} intents, {len(rows)} messages")
    print(f"intents: {', '.join(label_names)}\n")

    client = (
        SystemOne(rotations=args.rotations)
        if args.backend == "decoder"
        else EncoderSystemOne()
    )
    print(f"backend: {args.backend}")
    question = {
        "intent": Choice(
            instructions="Which banking support intent does this customer message express",
            criteria=label_names,
        )
    }

    probs: list[list[float]] = []
    labels: list[int] = []
    t0 = time.perf_counter()
    for i, (text, label) in enumerate(rows):
        resp = client.decide(text, question)
        answer = resp.answers["intent"]
        probs.append([answer.probabilities[name] for name in label_names])
        labels.append(label)
        if (i + 1) % 50 == 0:
            rate = (time.perf_counter() - t0) / (i + 1) * 1000
            print(f"  {i + 1}/{len(rows)}  {rate:.0f} ms/decision")
    total = time.perf_counter() - t0
    print(f"\n{len(rows)} decisions in {total:.1f}s = {total / len(rows) * 1000:.0f} ms each\n")

    half = len(probs) // 2
    fit_p, fit_y = probs[:half], labels[:half]
    test_p, test_y = probs[half:], labels[half:]

    before = evaluate(test_p, test_y, temperature=1.0)
    print(before.render("=== BEFORE calibration (test half) ==="))

    t_brier = fit_temperature(fit_p, fit_y, objective="brier")
    t_nll = fit_temperature(fit_p, fit_y, objective="nll")
    print(f"\nfitted on the other half: T(brier)={t_brier:.3f}  T(nll)={t_nll:.3f}")

    after = evaluate(test_p, test_y, temperature=t_brier)
    print("\n" + after.render("=== AFTER calibration, T fitted on held-out half ==="))

    print("\n=== summary ===")
    print(f"{'metric':>12} {'before':>10} {'after':>10} {'change':>10}")
    for name, b, a in (
        ("accuracy", before.accuracy, after.accuracy),
        ("brier", before.brier, after.brier),
        ("nll", before.nll, after.nll),
        ("ece", before.ece, after.ece),
    ):
        print(f"{name:>12} {b:>10.4f} {a:>10.4f} {a - b:>+10.4f}")
    print("\nAccuracy is unchanged by construction: temperature scaling cannot")
    print("reorder the options, only restate how sure the model is.")

    escalation_table(test_p, test_y, t_brier)


def escalation_table(test_p, test_y, temperature: float) -> None:
    """What calibration actually buys you: a threshold you can trust.

    The point of honest probabilities is confidence-gated routing. Auto-handle
    the cases the model is sure about, escalate the rest to something slower and
    more expensive. That trade is only safe if "sure" means what it says, so
    here is the trade priced out at several thresholds.
    """
    import numpy as np

    from nanojev.calibrate import apply_temperature

    cal = np.asarray(apply_temperature(test_p, temperature))
    raw = np.asarray(test_p)
    y = np.asarray(test_y)

    print("\n=== confidence-gated routing on the test half ===")
    print("auto-handle when confidence >= threshold, escalate the rest\n")
    print(f"{'threshold':>10} {'auto %':>8} {'auto acc':>9} {'esc %':>7} {'promised':>9} {'gap':>7}")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        conf = cal.max(axis=1)
        pred = cal.argmax(axis=1)
        mask = conf >= thr
        n = int(mask.sum())
        if n == 0:
            print(f"{thr:>10.2f} {0.0:>7.1f}% {'-':>9} {100.0:>6.1f}% {'-':>9} {'-':>7}")
            continue
        auto_acc = float((pred[mask] == y[mask]).mean())
        promised = float(conf[mask].mean())
        print(
            f"{thr:>10.2f} {100 * n / len(y):>7.1f}% {auto_acc:>9.3f} "
            f"{100 * (1 - n / len(y)):>6.1f}% {promised:>9.3f} {auto_acc - promised:>+7.3f}"
        )

    print("\n'promised' is the mean confidence the model claimed on the cases it")
    print("auto-handled; 'gap' is how far reality fell short. Near zero is the")
    print("whole game: it means the threshold you pick is the accuracy you get.")

    print("\nSame table, uncalibrated, for contrast:")
    print(f"{'threshold':>10} {'auto %':>8} {'auto acc':>9} {'promised':>9} {'gap':>7}")
    for thr in (0.5, 0.7, 0.9, 0.95):
        conf = raw.max(axis=1)
        pred = raw.argmax(axis=1)
        mask = conf >= thr
        n = int(mask.sum())
        if n == 0:
            continue
        auto_acc = float((pred[mask] == y[mask]).mean())
        promised = float(conf[mask].mean())
        print(
            f"{thr:>10.2f} {100 * n / len(y):>7.1f}% {auto_acc:>9.3f} "
            f"{promised:>9.3f} {auto_acc - promised:>+7.3f}"
        )


if __name__ == "__main__":
    main()
