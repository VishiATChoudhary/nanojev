"""The prefix-cache path must compute the same thing as the plain path.

A faster wrong answer is not a speedup, so this asserts the two code paths agree
before any timing number is allowed to mean anything.
"""

from nanojev import Choice, Noul, Score, SystemOne

client = SystemOne(rotations=2)

STATES = [
    "Stripe connection has failed for 3 days and I am losing sales. Help ASAP.",
    "Just wanted to say the new dashboard looks great, no issues here.",
    "How much does the enterprise plan cost for 200 seats?",
]

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
        criteria=["Calm", "Frustrated but civil", "Very angry"],
    ),
    "is_urgent": Noul(instructions="The message conveys urgency"),
}

worst = 0.0
for state in STATES:
    a = client.decide(state, QUESTIONS, use_prefix_cache=False)
    b = client.decide(state, QUESTIONS, use_prefix_cache=True)
    for key in QUESTIONS:
        pa, pb = a.answers[key], b.answers[key]
        if hasattr(pa, "probabilities"):
            for opt in pa.probabilities:
                worst = max(worst, abs(pa.probabilities[opt] - pb.probabilities[opt]))
        else:
            worst = max(worst, abs(pa.noul - pb.noul))
    print(f"prefix_tokens={b.meta['prefix_tokens']:>4}  state={state[:44]!r}")

print(f"\nmax probability difference across all questions: {worst:.2e}")
assert worst < 2e-2, f"paths disagree by {worst}"
print("PASS: prefix cache is equivalent to the plain path")
