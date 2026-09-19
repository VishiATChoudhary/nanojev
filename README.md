<div align="center">

# nanojev

**Typed, calibrated decisions from a local model. No token generation.**

[![ci](https://github.com/VishiATChoudhary/nanojev/actions/workflows/ci.yml/badge.svg)](https://github.com/VishiATChoudhary/nanojev/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

*A small local take on the "decision model" idea, built to find out which of the claims hold up.*

</div>

---

Ask a language model to classify something and it *writes you an answer*, one
token at a time, which your code then parses, validates and retries. But the
answer was one of three options you already knew.

nanojev never lets it write. It lays the options out, runs the model forward
**once**, and reads the scores at the position where the answer would have
started. You get a probability for every option, and an off-menu answer is not
filtered out — it is unrepresentable.

```console
$ uv run nanojev --demo

  "Hi, I've been trying to connect my Stripe account for 3 days and ..."
  decoder · 153 ms

  department    technical    technical 78%  sales 20%  billing 2%
  frustration   1.19 / 2       ████████████░░░░░░░░
  is_urgent      97.8%        ████████████████████

  "what would the enterprise plan cost for 200 seats?"
  decoder · 151 ms

  department    sales        sales 62%  billing 37%  technical 1%
  frustration   0.74 / 2       ███████░░░░░░░░░░░░░
  is_urgent      68.9%        ██████████████░░░░░░
```

## Install

```bash
git clone https://github.com/VishiATChoudhary/nanojev && cd nanojev
uv venv && uv pip install -e .

uv run nanojev --demo                      # worked examples
uv run nanojev "my card was declined"      # your own text
```

Apple Silicon, CUDA or CPU. No API key, no network at inference time.

## Use it

```bash
uv run nanojev \
  --choice "severity=total outage,major feature broken,minor glitch" \
  --noul   "page_oncall=customers cannot use the product right now" \
  "checkout is returning 500s for everyone"
```

```python
from nanojev import Choice, Noul, Score, SystemOne

client = SystemOne()
resp = client.decide(
    state="Stripe has been failing for 3 days and I'm losing sales. Help ASAP.",
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
            criteria=["Calm", "Frustrated but civil", "Very angry"],
        ),
        "is_urgent": Noul(instructions="The message conveys urgency"),
    },
)

resp.answers["department"].choice          # 'technical'
resp.answers["department"].probabilities   # {'billing': .02, 'technical': .78, 'sales': .20}
resp.answers["frustration"].score          # 1.19 — a value between levels, not an index
resp.answers["is_urgent"].noul             # 0.978 — a probability; you pick the threshold
```

Three question types. They map onto control flow: a **Choice** is a `match`, a
**Score** is a sort key, a **Noul** is an `if` (short for *bernoulli*).

## How it works

```
   [STATE]  Stripe keeps failing...            one forward pass
   [OPTIONS]                                   ────────────────
     A. billing                                      ▼
     B. technical        ───────────────────▶ ┌─────────────┐
     C. sales                                 │    model    │
   Answer:                                    └──────┬──────┘
                                                     │
                                                     ▼
                              read the scores for " A" " B" " C" only
                                                     │
                                                     ▼
                           billing 2%   technical 78%   sales 1%
                                 no text was ever generated
```

The model is stopped at the instant before it would speak, holding a score for
every token it might emit. We read three of them and discard the rest. Prefill
only, zero decode steps. Full detail in [docs/how-it-works.md](docs/how-it-works.md).

## Two backends, one API

banking77, 20 intents, 400 messages, temperature fitted on one half and reported
on the other:

| | decoder (Qwen3-0.6B) | encoder (deberta-v3-base, 184M) |
|---|---|---|
| method | read label-token logits | NLI entailment per option |
| accuracy | 0.595 | **0.700** |
| ECE, raw | 0.134 | **0.070** |
| fitted temperature | 1.759 | 1.086 |
| three questions, one state | 165 ms | **37 ms** |

```python
from nanojev import EncoderSystemOne
client = EncoderSystemOne()   # identical decide() contract
```

**The encoder wins, and that is the interesting result.** The common guess about
models in this category is that they are encoders with classification heads —
the GLiNER / GLiClass / deberta-zeroshot family — rather than a new class of
model. Here a 184M public checkpoint, which I did not train or modify, beats the
0.6B decoder and arrives **already calibrated**: a fitted temperature of 1.086
means there was almost nothing to correct.

That proves nothing about any particular commercial model. It does suggest
"decision-only model" needs no new architecture to work well.

**Which one to use** depends on your inputs, and the tradeoff is sharp — the
encoder is more accurate but more literal, and flips on paraphrases the decoder
shrugs off. Numbers in [docs/backends.md](docs/backends.md).

## Calibration is the point

Any model can emit a number between 0 and 1. The claim worth testing is whether
0.7 means right-70%-of-the-time.

| metric | before | after | |
|---|---|---|---|
| accuracy | 0.595 | 0.595 | unchanged, by construction |
| ECE | 0.134 | **0.101** | |
| Brier | 0.576 | 0.541 | |
| NLL | 1.798 | 1.486 | |

Temperature scaling cannot reorder the options, only rescale how sure the model
claims to be — which is the part that was lying.

> [!WARNING]
> A model that always outputs the base rate is *perfectly* calibrated and
> useless. Reliability is not resolution, so accuracy is reported beside every
> calibration figure here.

### What it buys: a threshold you can trust

Uncalibrated, the confidence score oversells itself:

```
 threshold   auto %  auto acc  promised     gap
      0.90    32.0%     0.891     0.976   -0.085
      0.95    26.0%     0.923     0.987   -0.064
```

Calibrated, the promise holds, and the gap turns conservative:

```
 threshold   auto %  auto acc  promised     gap
      0.70    30.0%     0.933     0.864   +0.069
      0.80    23.0%     0.957     0.898   +0.058
      0.90    11.0%     1.000     0.941   +0.059
```

The encoder, being better calibrated to begin with, gates more traffic at a
similar accuracy: at 0.80 it auto-handles **45.5%** of messages at **93.4%**
accuracy, having promised 90.1%.

That is the whole product. Auto-handle what the model is genuinely sure about,
escalate the rest, and have the threshold mean what it says.

## Honest limitations

- **Accuracy is modest.** 0.595 / 0.700 on 20-way banking77 (chance is 5%).
  Useful *because of* confidence gating, not despite it.
- **"Cannot hallucinate" means cannot go off-schema.** A schema-valid but wrong
  answer is exactly what the accuracy numbers measure.
- **"Adding questions is nearly free" is overstated.** Measured here, eight
  questions cost ~3.7x one question. Sublinear, not flat.
- **Post-hoc calibration is not trained-in calibration.** One scalar cannot fix a
  *mis-ranked* distribution, and may not survive distribution shift.
- **One dataset.** banking77 only. Treat these as a working baseline, not a
  benchmark result.

## Development

```bash
uv pip install -e ".[dev,bench]"
uv run pytest                                    # 18 tests, no model download
uv run python bench/equivalence.py               # cached path == plain path
uv run python bench/calibration_eval.py --backend encoder
uv run python bench/latency.py
```

| | |
|---|---|
| [docs/how-it-works.md](docs/how-it-works.md) | the mechanism in detail |
| [docs/backends.md](docs/backends.md) | decoder vs encoder, with numbers |
| [docs/glossary.md](docs/glossary.md) | every term and abbreviation |
| [docs/findings.md](docs/findings.md) | what building it revealed, including the bugs |

## Prior art

This studies an idea rather than reimplementing any particular product, and the
idea is older than its current branding. Zero-shot classification via natural
language inference dates to ~2019. **GLiNER2**
([arXiv:2507.18546](https://arxiv.org/abs/2507.18546)) already does
runtime-defined schemas in a single pass. Temperature scaling is
[Guo et al. 2017](https://arxiv.org/abs/1706.04599). Proper scoring rules are
Gneiting & Raftery 2007; the reliability-versus-resolution decomposition is
Murphy 1973.

MIT licensed.
