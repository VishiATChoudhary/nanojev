"""Regenerate every figure quoted in the README, in one consistent run.

After the label-token fix, every decoder number had to be recomputed. Doing it
from one script means the README cannot drift into quoting results from three
different versions of the code.
"""

from __future__ import annotations

import statistics
import time

import torch

from nanojev import Choice, EncoderSystemOne, Noul, Score, SystemOne

CANONICAL = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP."
)
PARAPHRASE = "Stripe has been failing for 3 days and I am losing sales. Help ASAP."

TRIAGE = {
    "department": Choice(
        instructions="Which team should handle this",
        criteria={
            "billing": "Payment or subscription issues",
            "technical": "Bugs or integration problems",
            "sales": "Pricing or account questions",
        },
    ),
    "frustration": Score(
        instructions="How frustrated the customer appears",
        criteria=["Calm, just stating facts", "Frustrated but civil", "Very angry"],
    ),
    "is_urgent": Noul(instructions="The message conveys urgency"),
}


def timeit(fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        s.append((time.perf_counter() - t0) * 1000)
    return statistics.median(s)


def main() -> None:
    dec, enc = SystemOne(), EncoderSystemOne()

    print("### 1. Jev quickstart parity")
    for name, c in (("decoder", dec), ("encoder", enc)):
        a = c.decide(CANONICAL, TRIAGE).answers
        print(
            f"  {name:>8}: department={a['department'].choice:<10} "
            f"frustration={a['frustration'].score:.3f}  is_urgent={a['is_urgent'].noul:.3f}"
        )
    print("  Jev ref : department=technical  frustration=1.035   is_urgent=0.999")

    print("\n### 2. Latency, three questions, one state")
    for name, c in (("decoder", dec), ("encoder", enc)):
        print(f"  {name:>8}: {timeit(lambda c=c: c.decide(CANONICAL, TRIAGE)):.1f} ms")

    print("\n### 3. Opaque vs described options")
    opaque = {"sev": Choice(instructions="Incident severity", criteria=["sev0", "sev1", "sev2", "sev3"])}
    described = {
        "sev": Choice(
            instructions="Incident severity",
            criteria={
                "sev0": "a total outage, customers cannot use the product at all",
                "sev1": "a major feature is broken for many customers",
                "sev2": "a minor feature is degraded",
                "sev3": "cosmetic or internal only, no customer impact",
            },
        )
    }
    state = "The server is on fire and customers cannot check out."
    for name, c in (("decoder", dec), ("encoder", enc)):
        for tag, qs in (("opaque", opaque), ("described", described)):
            a = c.decide(state, qs).answers["sev"]
            print(f"  {name:>8} {tag:>10}: {a.choice:<6} confidence={a.confidence:.3f}")

    print("\n### 4. Paraphrase robustness")
    for name, c in (("decoder", dec), ("encoder", enc)):
        for tag, st in (("canonical", CANONICAL), ("paraphrase", PARAPHRASE)):
            a = c.decide(st, TRIAGE).answers["department"]
            print(
                f"  {name:>8} {tag:>10}: {a.choice:<10} "
                f"p(technical)={a.probabilities['technical']:.3f} "
                f"p(billing)={a.probabilities['billing']:.3f}"
            )

    print("\n### 5. Literal vs inferential yes/no")
    # The claim must be paired with a state it actually restates, otherwise the
    # comparison tests nothing. "a payment problem" restates a declined card;
    # it does not restate a 500 error.
    st = "checkout is returning 500s for everyone"
    claims = [
        ("restatement", "checkout is failing"),
        ("restatement", "there are server errors"),
        ("inference", "customers cannot use the product right now"),
        ("inference", "this is an outage"),
        ("control", "the weather is nice today"),
    ]
    for name, c in (("decoder", dec), ("encoder", enc)):
        for kind, claim in claims:
            v = c.decide(st, {"q": Noul(instructions=claim)}).answers["q"].noul
            print(f"  {name:>8} {kind:>11}: {v:.3f}  '{claim[:46]}'")

    print("\n### 6. Two-stage path, 120 options")
    opts = [f"intent_{i}" for i in range(120)]
    opts[73] = "card_payment_declined"
    r = dec.decide(
        "My card was declined when I tried to pay.",
        {"intent": Choice(instructions="Which intent", criteria=opts)},
    )
    a = r.answers["intent"]
    print(
        f"  choice={a.choice} confidence={a.confidence:.3f} "
        f"mass={sum(a.probabilities.values()):.6f} sequences={r.sequences} "
        f"latency={r.latency_ms:.0f}ms"
    )


if __name__ == "__main__":
    main()
