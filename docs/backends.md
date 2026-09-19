# Choosing a backend

Both backends implement the same `decide()` contract. They differ in how they
score options, and the tradeoff is sharper than the headline accuracy suggests.

```python
from nanojev import SystemOne, EncoderSystemOne

SystemOne()          # decoder: Qwen3-0.6B, reads label-token logits
EncoderSystemOne()   # encoder: deberta-v3-base-zeroshot, NLI entailment
```

## The aggregate

banking77, 20 intents, 400 messages, temperature fitted on a held-out half:

| | decoder | encoder |
|---|---|---|
| accuracy | 0.595 | **0.700** |
| ECE, raw | 0.134 | **0.070** |
| ECE, calibrated | 0.101 | 0.068 |
| fitted temperature | 1.759 | 1.086 |
| Brier, raw | 0.576 | **0.432** |
| three questions on one state | 165 ms | **37 ms** |

On this dataset the encoder wins on every axis. It is also already calibrated:
a fitted temperature of 1.086 means barely any correction was available.

## Where the encoder is fragile

### It needs option strings that mean something

It matches your option text against the state semantically, so opaque codes give
it nothing to work with:

| options | decoder | encoder |
|---|---|---|
| `sev0 sev1 sev2 sev3` | `sev2`, confidence 0.187 | `sev0`, confidence **0.005** |
| each option described | `sev0`, confidence 0.721 | `sev0`, confidence **0.842** |

Confidence 0.005 is near-uniform: the encoder is correctly reporting that it has
no idea. Describe your options and it becomes the stronger model.

```python
Choice(instructions="Incident severity", criteria={
    "sev0": "a total outage, customers cannot use the product at all",
    "sev1": "a major feature is broken for many customers",
    "sev2": "a minor feature is degraded",
    "sev3": "cosmetic or internal only, no customer impact",
})
```

### It leans on specific wording

Same ticket, one paraphrase, nothing else changed. Dropping "connect my
**account**" removes the only cue that this is an integration problem rather
than a payments one:

| | "trying to **connect my Stripe account**… keeps failing" | "**Stripe has been failing**… losing sales" |
|---|---|---|
| decoder | technical, p=0.782 | technical, p=0.670 |
| encoder | technical, p=0.717 | **billing, p=0.853** |

The encoder flips. The decoder degrades gracefully.

### It is literal about yes/no claims

Entailment is narrower than reasoning. Given *"checkout is returning 500s for
everyone"*:

| claim | kind | decoder | encoder |
|---|---|---|---|
| "checkout is failing" | restatement | 0.771 | **0.980** |
| "there are server errors" | restatement | 0.887 | **0.997** |
| "customers cannot use the product right now" | inference | **0.402** | 0.195 |
| "this is an outage" | inference | **0.832** | 0.316 |
| "the weather is nice today" | control | 0.212 | **0.000** |

The encoder is sharper on claims that restate the state and cleaner on the
control, but a claim needing one inferential step scores far lower than it
should. 500 errors do not *literally* entail an outage.

> [!NOTE]
> The encoder's yes/no path routes the claim through the checkpoint's own
> hypothesis template. Passing the bare claim is off-distribution and collapses
> to ~0.005 even for direct restatements — see [findings.md](findings.md).

## Rule of thumb

**Use the encoder** when your option strings carry meaning or can be described,
your inputs are reasonably consistent in phrasing, and your yes/no claims
restate rather than infer.

**Use the decoder** when options are opaque identifiers, inputs are
paraphrase-diverse free text, or yes/no claims need a reasoning step. It is the
CLI default for this reason.

If your data is real user text rather than templated records, measure both on
**your own** data. The 10-point aggregate gap is smaller than the swing either
backend shows between its best and worst case here.

## Option-count limits

The decoder can use as many options as the tokenizer has usable single-token
labels — 52 for Qwen3. Beyond that it splits into two stages automatically
(score within groups, then choose between winners). Verified on 120 options and
on all 77 banking77 intents. The encoder has no such limit.
