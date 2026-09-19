"""Try nanojev from the command line.

    python -m nanojev "my card was declined at the shop"
    python -m nanojev --backend decoder "the server is down"
    python -m nanojev --choice urgency=low,medium,high "the building is on fire"

With no --choice/--noul/--score flags you get a default triage set, so there is
something to look at immediately.
"""

from __future__ import annotations

import argparse
import sys

from ._quiet import quiet

quiet()

from . import Choice, EncoderSystemOne, Noul, Score, SystemOne

# The same question set used in the README and in bench/, so the documented
# output and the output you get are the same thing.
DEFAULTS = {
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


def parse_questions(args: argparse.Namespace) -> dict:
    questions: dict = {}
    for spec in args.choice or []:
        name, _, opts = spec.partition("=")
        questions[name] = Choice(
            instructions=name.replace("_", " "), criteria=[o.strip() for o in opts.split(",")]
        )
    for spec in args.score or []:
        name, _, levels = spec.partition("=")
        questions[name] = Score(
            instructions=name.replace("_", " "), criteria=[l.strip() for l in levels.split(",")]
        )
    for spec in args.noul or []:
        name, _, claim = spec.partition("=")
        questions[name] = Noul(instructions=claim or name.replace("_", " "))
    return questions or DEFAULTS


DEMO = [
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP.",
    "just wanted to say the new dashboard is lovely, no issues at all",
    "what would the enterprise plan cost for 200 seats?",
]


def render(resp, state: str, backend: str) -> None:
    shown = state if len(state) <= 68 else state[:65] + "..."
    print(f'\n  \033[2m"\033[0m{shown}\033[2m"\033[0m')
    print(f"  \033[2m{backend} · {resp.latency_ms:.0f} ms\033[0m\n")
    for key, answer in resp.answers.items():
        if hasattr(answer, "choice"):
            ranked = sorted(answer.probabilities.items(), key=lambda kv: -kv[1])
            bars = "  ".join(f"\033[2m{k} {v:.0%}\033[0m" for k, v in ranked[:4])
            print(f"  {key:<13} \033[1m{answer.choice:<12}\033[0m {bars}")
        elif hasattr(answer, "score"):
            top = len(answer.probabilities) - 1
            filled = int(round(answer.score / top * 20)) if top else 0
            meter = "\u2588" * filled + "\u2591" * (20 - filled)
            print(f"  {key:<13} \033[1m{answer.score:.2f}\033[0m / {top}       {meter}")
        else:
            filled = int(round(answer.noul * 20))
            meter = "\u2588" * filled + "\u2591" * (20 - filled)
            print(f"  {key:<13} \033[1m{answer.noul:>6.1%}\033[0m        {meter}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(prog="nanojev")
    ap.add_argument("state", nargs="*", help="the text to make decisions about")
    ap.add_argument("--demo", action="store_true", help="run a few worked examples and exit")
    # Decoder by default: it is the robust one. The encoder is more accurate on
    # Choice questions with well-described options, but it judges yes/no claims
    # by literal entailment, so anything needing an inference step fails closed.
    ap.add_argument("--backend", choices=("decoder", "encoder"), default="decoder")
    ap.add_argument("--choice", action="append", metavar="name=a,b,c")
    ap.add_argument("--score", action="append", metavar="name=low,mid,high")
    ap.add_argument("--noul", action="append", metavar="name=the claim to judge")
    args = ap.parse_args()

    questions = parse_questions(args)
    client = EncoderSystemOne() if args.backend == "encoder" else SystemOne()

    if args.demo:
        for example in DEMO:
            render(client.decide(example, questions), example, args.backend)
        return

    state = " ".join(args.state) or (sys.stdin.read().strip() if not sys.stdin.isatty() else "")
    if not state:
        ap.error("give me some text, as an argument or on stdin (or try --demo)")

    resp = client.decide(state, questions)

    render(resp, state, args.backend)


if __name__ == "__main__":
    main()
