"""Does the central System One claim hold on a laptop?

TypeSafe says questions are evaluated in parallel so "adding questions barely
changes the response time", and that skipping generation buys 20-200x. Both are
measurable. This reports what actually happens, including where the claim is
weaker than the marketing.
"""

from __future__ import annotations

import statistics
import time

import torch

from nanojev import Choice, Noul, Score, SystemOne

SHORT_STATE = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP."
)

# A realistic "state": a support thread with logs. This is the regime System One
# models are actually pitched at, and the regime where sharing the prefix pays.
LONG_STATE = SHORT_STATE + "\n\n" + "\n".join(
    f"[2026-09-{10 + i % 5:02d} 14:{i:02d}:11] webhook stripe.account.updated "
    f"retry={i} status=502 upstream=connect-api latency_ms={120 + i * 7} "
    f"payload_bytes={2048 + i * 13} trace_id=7f{i:03x}c2a1"
    for i in range(60)
)

POOL = {
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
    "is_churn_risk": Noul(instructions="The customer may cancel their subscription"),
    "priority": Choice(instructions="Triage priority", criteria=["p0", "p1", "p2", "p3"]),
    "is_bug": Noul(instructions="The message reports a product defect"),
    "needs_human": Noul(instructions="This requires a human agent, not automation"),
    "sentiment": Score(
        instructions="Overall sentiment", criteria=["negative", "neutral", "positive"]
    ),
}


def timeit(fn, warmup: int = 2, runs: int = 8) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    client = SystemOne(rotations=1)
    keys = list(POOL)
    print(f"device={client.device}\n")

    for name, state in (("short state (~30 tok)", SHORT_STATE), ("long state (~1.5k tok)", LONG_STATE)):
        probe = client.decide(state, {"x": POOL["is_bug"]}, use_prefix_cache=False)
        print(f"=== latency vs question count, {name}, {probe.input_tokens} tok/seq ===")
        print(f"{'questions':>10} {'flat ms':>9} {'cached ms':>10} {'cached/q':>9} {'speedup':>8}")
        first = None
        for n in (1, 2, 4, 8):
            qs = {k: POOL[k] for k in keys[:n]}
            flat = timeit(lambda qs=qs: client.decide(state, qs, use_prefix_cache=False))
            cached = timeit(lambda qs=qs: client.decide(state, qs, use_prefix_cache=True))
            if first is None:
                first = cached
            print(f"{n:>10} {flat:>9.1f} {cached:>10.1f} {cached / n:>9.1f} {flat / cached:>7.2f}x")
        print(f"  -> 8 questions cost {cached / first:.2f}x one question "
              f"(linear=8.00x, flat=1.00x)\n")

    print("=== decision-by-logit-read vs generating the answer as text ===")
    print("(same model, same prompt; generation forced to run, not stopped at EOS)")
    qs = {k: POOL[k] for k in keys[:3]}
    dec = timeit(lambda: client.decide(SHORT_STATE, qs))

    tok, model = client.tokenizer, client.model
    prompt = client._render(SHORT_STATE, POOL["department"], [0, 1, 2])

    def generate(n_tok: int):
        def run():
            ids = tok(prompt, return_tensors="pt").to(client.device)
            with torch.inference_mode():
                model.generate(
                    **ids,
                    min_new_tokens=n_tok,
                    max_new_tokens=n_tok,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
        return run

    print(f"\n{'mode':>38} {'median ms':>10} {'vs nanojev':>11}")
    print(f"{'nanojev: 3 questions, 0 decode steps':>38} {dec:>10.1f} {'1.00x':>11}")
    for n_tok in (1, 16, 64, 256):
        g = timeit(generate(n_tok), warmup=1, runs=4)
        label = f"generate 1 answer, {n_tok} token{'s' if n_tok > 1 else ''}"
        print(f"{label:>38} {g:>10.1f} {g / dec:>10.2f}x")


if __name__ == "__main__":
    main()
