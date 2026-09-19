"""Head-to-head, both backends, same questions, same batching discipline."""

from __future__ import annotations

import statistics
import time

import torch

from nanojev import Choice, EncoderSystemOne, Noul, Score, SystemOne

TICKET = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP."
)

QUESTIONS = {
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


def timeit(fn, warmup: int = 3, runs: int = 10) -> float:
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


for name, client in (("decoder (Qwen3-0.6B)", SystemOne()), ("encoder (deberta-v3-base)", EncoderSystemOne())):
    r = client.decide(TICKET, QUESTIONS)
    med = timeit(lambda c=client: c.decide(TICKET, QUESTIONS))
    d, f, u = r.answers["department"], r.answers["frustration"], r.answers["is_urgent"]
    print(f"\n=== {name} ===")
    print(f"  department  {d.choice!r} conf={d.confidence:.3f}")
    print(f"  frustration {f.score:.3f}")
    print(f"  is_urgent   {u.noul:.3f}")
    print(f"  median latency for all 3 questions: {med:.1f} ms")
