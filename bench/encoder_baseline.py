"""Is a decoder reading its own logits actually better than an encoder classifier?

The common hypothesis about decision models is that they are encoder-only models
with classification heads - the GLiNER / GLiClass / deberta-zeroshot family -
rather than a new class of model. That is a testable claim, so this tests the
local version of it: same dataset, same labels, same metrics, one decoder reading
label logits versus one off-the-shelf zero-shot encoder.

Whatever wins, the comparison is the honest one to make before calling either
approach a new category.
"""

from __future__ import annotations

import argparse
import time

from transformers import pipeline

from bench.calibration_eval import load_banking
from nanojev.calibrate import evaluate, fit_temperature

ENCODER = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=20)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=ENCODER)
    args = ap.parse_args()

    label_names, rows = load_banking(args.classes, args.samples, args.seed)
    print(f"encoder baseline: {args.model}")
    print(f"banking77: {len(label_names)} intents, {len(rows)} messages\n")

    clf = pipeline("zero-shot-classification", model=args.model, device="mps")

    probs: list[list[float]] = []
    labels: list[int] = []
    t0 = time.perf_counter()
    for i, (text, label) in enumerate(rows):
        out = clf(text, candidate_labels=label_names, multi_label=False)
        by_label = dict(zip(out["labels"], out["scores"]))
        probs.append([by_label[name] for name in label_names])
        labels.append(label)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}  {(time.perf_counter() - t0) / (i + 1) * 1000:.0f} ms/decision")
    total = time.perf_counter() - t0
    print(f"\n{len(rows)} decisions in {total:.1f}s = {total / len(rows) * 1000:.0f} ms each\n")

    half = len(probs) // 2
    before = evaluate(probs[half:], labels[half:], temperature=1.0)
    t = fit_temperature(probs[:half], labels[:half], objective="brier")
    after = evaluate(probs[half:], labels[half:], temperature=t)

    print(before.render("=== encoder BEFORE calibration (test half) ==="))
    print(f"\nfitted T = {t:.3f}")
    print("\n" + after.render("=== encoder AFTER calibration ==="))

    print("\n=== summary (encoder) ===")
    print(f"{'metric':>12} {'before':>10} {'after':>10}")
    for name, b, a in (
        ("accuracy", before.accuracy, after.accuracy),
        ("brier", before.brier, after.brier),
        ("nll", before.nll, after.nll),
        ("ece", before.ece, after.ece),
    ):
        print(f"{name:>12} {b:>10.4f} {a:>10.4f}")
    print(f"\nms/decision: {total / len(rows) * 1000:.0f}")


if __name__ == "__main__":
    main()
