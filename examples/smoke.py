"""The canonical support-triage example, run against a local model.

The same ticket and questions a commercial decision-model quickstart uses, so the
output here can be compared against a hosted one directly.
"""

from nanojev import Choice, Noul, Score, SystemOne

client = SystemOne(rotations=3)
print(f"device={client.device} labels={client.max_options}")

ticket = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP."
)

resp = client.decide(
    state=ticket,
    questions={
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
            criteria=[
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
        ),
        "is_urgent": Noul(
            instructions="The message conveys urgency or time-sensitivity",
        ),
    },
)

d = resp.answers["department"]
f = resp.answers["frustration"]
u = resp.answers["is_urgent"]
print(f"department  = {d.choice!r}  conf={d.confidence:.3f}  {({k: round(v,3) for k,v in d.probabilities.items()})}")
print(f"frustration = {f.score:.3f}  conf={f.confidence:.3f}")
print(f"is_urgent   = {u.noul:.3f}  conf={u.confidence:.3f}")
print(f"\nlatency={resp.latency_ms:.0f}ms  input_tokens={resp.input_tokens}  sequences={resp.sequences}")
print("hosted reference:  department='technical'  frustration=1.035  is_urgent=0.999")
