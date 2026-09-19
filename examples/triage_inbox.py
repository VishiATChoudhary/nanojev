"""Confidence-gated inbox triage: the thing calibration is actually for.

A stream of support messages arrives. Each one gets routed. The ones the model
is genuinely sure about are handled automatically; the rest are escalated to a
human. Then we check the promise against reality on the auto-handled set.

That last step is the point. Anyone can route tickets. The question is whether
the confidence score is honest enough that you can safely stop looking at the
ones above your threshold.

It also shows why there are two backends. Routing is a Choice question, where
the encoder is more accurate. "Is this customer angry?" is an inferential claim,
where the encoder is useless (it scores ~0.00 on every message, because tone is
not literally entailed by the text) and the decoder works. Each question goes to
the backend that is good at it, behind one identical API.

    python examples/triage_inbox.py
    python examples/triage_inbox.py --threshold 0.9 --backend decoder
"""

from __future__ import annotations

import argparse
import time

from nanojev._quiet import quiet

quiet()

from nanojev import Choice, EncoderSystemOne, Noul, Score, SystemOne

DIM, BOLD, GREEN, YELLOW, RED, CYAN, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m"
)

QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this message",
        criteria={
            "billing": "a charge, refund, invoice, subscription or payment problem",
            "technical": "a bug, error, outage or broken integration",
            "sales": "a question about pricing, plans, quotes or buying more",
            "account": "login, password, profile or permissions",
        },
    ),
    "urgency": Score(
        instructions="How urgently this needs a reply",
        criteria=["whenever, no rush", "today would be good", "right now, revenue is affected"],
    ),
    "angry": Noul(instructions="the customer is angry or frustrated"),
}

# Hand-labelled so the promise can be checked. Small and curated on purpose:
# this demonstrates the mechanism, it is not a benchmark. For that see bench/.
INBOX: list[tuple[str, str]] = [
    ("I was charged twice for March, please refund one of them", "billing"),
    ("the export button throws a 500 every single time", "technical"),
    ("what does the enterprise tier cost for 200 seats?", "sales"),
    ("I cannot log in, it says my password is wrong after the reset", "account"),
    ("third invoice in a row with the wrong VAT number. sort it out.", "billing"),
    ("your webhooks have been down for six hours and checkout is dead", "technical"),
    ("can I add 5 more seats to my current plan mid-cycle?", "sales"),
    ("please remove admin rights from my former colleague immediately", "account"),
    ("the thing is broken again", "technical"),
    ("hi, quick question about my account", "account"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument(
        "--backend",
        choices=("hybrid", "encoder", "decoder"),
        default="hybrid",
        help="hybrid sends each question to the backend suited to it",
    )
    args = ap.parse_args()

    if args.backend == "hybrid":
        routing, judging = EncoderSystemOne(), SystemOne()
        label = "encoder routes, decoder judges tone"
    else:
        routing = judging = EncoderSystemOne() if args.backend == "encoder" else SystemOne()
        label = args.backend

    routed = {k: v for k, v in QUESTIONS.items() if k != "angry"}
    tone = {"angry": QUESTIONS["angry"]}

    print(f"\n{BOLD}  inbox triage{RESET}{DIM} · {label} · auto-handle above "
          f"{args.threshold:.0%} confidence{RESET}\n")

    auto, escalated, correct, promised, elapsed = [], [], 0, 0.0, 0.0
    for message, true_department in INBOX:
        t0 = time.perf_counter()
        answers = routing.decide(message, routed).answers
        answers |= judging.decide(message, tone).answers
        elapsed += (time.perf_counter() - t0) * 1000

        dept = answers["department"]
        urgency = answers["urgency"].score
        angry = answers["angry"].noul

        confident = dept.confidence >= args.threshold
        # The tone flag trips at 0.8, not 0.5, because this model's raw Noul
        # output is compressed into the top of the range: it orders messages
        # correctly but calls a neutral one 0.63. Fitting a temperature on
        # labelled examples is the principled fix (see bench/calibration_eval.py);
        # a higher cutoff is the honest workaround until you have those labels.
        flags = "".join((
            f"{RED}!{RESET}" if urgency > 1.5 else " ",
            f"{YELLOW}~{RESET}" if angry > 0.8 else " ",
        ))

        if confident:
            auto.append(message)
            correct += dept.choice == true_department
            promised += dept.confidence
            verdict = f"{GREEN}auto{RESET}    "
        else:
            escalated.append(message)
            verdict = f"{YELLOW}escalate{RESET}"

        hit = " " if dept.choice == true_department else f"{RED}x{RESET}"
        print(f"  {verdict} {flags} {hit} {dept.choice:<10}{DIM}{dept.confidence:>5.0%}{RESET}  "
              f"{DIM}{message[:52]}{RESET}")

    n = len(auto)
    print(f"\n{DIM}  {'─' * 74}{RESET}")
    print(f"  {len(INBOX)} messages in {elapsed:.0f} ms "
          f"{DIM}({elapsed / len(INBOX):.0f} ms each){RESET}")
    print(f"  {GREEN}{n} auto-handled{RESET}, {YELLOW}{len(escalated)} escalated{RESET}"
          f"{DIM} — {n / len(INBOX):.0%} of the inbox handled without a human{RESET}")

    if n:
        actual = correct / n
        claimed = promised / n
        gap = actual - claimed
        colour = GREEN if gap >= -0.05 else RED
        print(f"\n  on the {n} it auto-handled:")
        print(f"    promised {BOLD}{claimed:.1%}{RESET} accuracy")
        print(f"    delivered {BOLD}{actual:.1%}{RESET}  {colour}({gap:+.1%}){RESET}")
        print(f"\n{DIM}  The threshold you pick is the accuracy you get. That is the"
              f" whole point:{RESET}")
        print(f"{DIM}  a confidence score you can safely stop looking behind.{RESET}\n")
    print(f"{DIM}  ! urgent   ~ angry   x misrouted{RESET}\n")


if __name__ == "__main__":
    main()
