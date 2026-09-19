# Glossary

Every abbreviation and term used in this project, with the actual numbers from
our runs as examples.

---

## Part 1: the calibration terms

These are the ones that matter most, because calibration is the whole point.

### Calibration

Whether a stated confidence matches reality. A model is **calibrated** if, across
all the times it says "70% sure", it turns out to be right about 70% of the time.

It has nothing to do with being *right*. A weather forecaster who says "30% rain"
on exactly the days it rains 30% of the time is perfectly calibrated, even though
they're never certain. Being accurate and being honest about your uncertainty are
two separate skills.

**Why it matters here:** if you want software to auto-handle the confident cases
and escalate the rest, the confidence number has to mean something. Ours did not,
until it was fixed. Raw, the model claimed 98.7% confidence and was right 70.8% of
the time.

### Confidence

How sure the model claims to be, from 0 to 1. For a Choice question we compute it
from how *concentrated* the probabilities are: if all options are equally likely,
confidence is 0; if one option has all the mass, it's 1.

### ECE — Expected Calibration Error

**The readable calibration number.** The average gap between claimed confidence
and actual accuracy.

How it's computed: sort every prediction into buckets by confidence (0–10%,
10–20%, …). In each bucket, compare the average claimed confidence against the
actual fraction that were correct. Average those gaps, weighted by bucket size.

- **0.0** = perfect. Every claim matches reality.
- **0.37** = our decoder before fixing. Claims were off by 37 percentage points on average.
- **0.11** = after fixing.

⚠️ ECE is a *biased* estimator and depends on how you choose the buckets — which
means a vendor reporting their own ECE also chose the bins. Worth knowing when
reading anyone's marketing.

### Brier score

**A scoring rule for probabilistic predictions.** Squared error against the truth.

For each prediction, take the probability assigned to every option, subtract 1
from whichever option was actually correct, square it all, add it up. Lower is
better. Ours went 0.850 → 0.684.

The useful property: say 0.9 and be right, small penalty. Say 0.99 and be wrong,
big penalty. Say 0.6 when you're genuinely 60% sure, and you beat both. It
rewards honesty specifically.

### NLL — Negative Log Likelihood (also "log score", "cross-entropy")

Another scoring rule. Take the probability the model gave to the correct answer,
take its logarithm, negate it. Lower is better. Ours went 2.966 → 1.908.

Difference from Brier: NLL punishes confident mistakes **far** harder. Assign 0
probability to something that then happens and the penalty is infinite. Use Brier
when you want a stable number, NLL when confident errors are genuinely costly.

### Proper scoring rule

A scoring rule is **proper** if you score best by reporting what you actually
believe — you can't game it by exaggerating or hedging. Brier and NLL are proper.
"Percentage of times the top answer was right" is not, because it ignores the
probabilities entirely.

This is the mathematical foundation for training a model to be honest, and the
reason RLCD is a coherent idea. Reference: Gneiting & Raftery, *JASA* 2007.

### Reliability vs Resolution — and why calibration alone is a trap

Brier decomposes into three parts (Murphy, 1973):

- **Reliability** — are your stated probabilities honest? (calibration)
- **Resolution** — do you actually distinguish cases from one another?
- **Uncertainty** — how hard is the problem?

**The trap:** a model that ignores the input and always outputs the base rate is
*perfectly* calibrated and completely useless. Great reliability, zero resolution.

So a calibration number with no accuracy number beside it cannot be interpreted.
This is why every ECE in this project is reported with an accuracy next to it.

### Discrimination

The ability to tell cases apart — resolution, from another angle. Accuracy is a
crude measure of it. "Discrimination and calibration" together are what you need;
neither alone is sufficient.

### Temperature / temperature scaling

**The one-dial fix.** Divide the model's raw scores by a single number `T` before
turning them into percentages.

- `T = 1` — unchanged
- `T > 1` — softens overconfident predictions toward the middle
- `T < 1` — sharpens

You find the best `T` by testing values against held-out labelled data. Ours came
out at 2.668 for the decoder (very overconfident, needed heavy softening) and
1.086 for the encoder (already nearly honest).

**Key property:** it cannot change *which* option wins, only how sure the model
claims to be. That's why our accuracy is identical before and after. It fixes
dishonesty, not wrongness. Reference: Guo et al., arXiv:1706.04599, 2017.

### Reliability diagram

The table in our output with `confidence`, `accuracy`, `n` columns. Each row is
one confidence bucket. If calibration is good, the two numbers match on every row
and the `gap` column is near zero.

### Held-out / fit-half vs test-half

You must fit the temperature on *different* data than you report results on.
Otherwise you're grading your own homework. We split 400 examples in half: fit on
one, report on the other.

### Distribution shift

When live data differs from what you tuned on. This is where a post-hoc dial is
expected to be weaker than trained-in calibration, and the main honest argument
for why RLCD might beat temperature scaling.

### Base rate

How often something happens in general, ignoring the specific case. "5% of tickets
are urgent." A model that always outputs the base rate is the degenerate
perfectly-calibrated model above.

---

## Part 2: the abbreviations

| Term | Stands for | What it means here |
|---|---|---|
| **LLM** | Large Language Model | The normal kind. Writes text one token at a time. |
| **RLHF** | Reinforcement Learning from Human Feedback | Standard tuning method: reward answers humans prefer. Known to *damage* calibration, because humans prefer confident answers. |
| **RLCD** | Reinforcement Learning for Calibrated Decisions | Training that rewards honest probabilities rather than human preference. Named in a commercial launch; no paper published. |
| **RLCR** | Reinforcement Learning with Calibration Rewards | The published version of essentially that idea (arXiv:2507.16806, 2025). Reward = correctness + Brier. |
| **ECE** | Expected Calibration Error | See above. |
| **NLL** | Negative Log Likelihood | See above. |
| **NLI** | Natural Language Inference | "Does sentence A imply sentence B?" How our encoder backend scores options. Its limitation: it's *literal*. |
| **DX** | Developer Experience | How pleasant something is to build against. |
| **KV cache** | Key/Value cache | Stored intermediate results so the model doesn't re-read the same text repeatedly. |
| **SDPA** | Scaled Dot-Product Attention | A memory-efficient attention implementation. Without it, long inputs allocate enormous matrices and crash. |
| **MPS** | Metal Performance Shaders | Apple's GPU backend. What we run on. |
| **bf16** | bfloat16 | A 16-bit number format. Half the memory of standard floats, fine for inference. |
| **HN** | Hacker News | Where the launch thread and the CEO's admissions live. |
| **arXiv** | — | Open preprint server. `arXiv:2203.02155` is a paper ID. |

---

## Part 3: the machinery

### Token

A chunk of text, roughly a short word or word-piece. Models read and write in
tokens, and APIs bill per token. "Input tokens" = what you send; "output tokens" =
what the model writes. A decision model's output is free because it writes nothing.

### Logits

The model's **raw scores** for every possible next token, before being converted
to percentages. Typically ~150,000 numbers, one per token in its vocabulary.

Reading logits directly instead of letting the model write is the core trick: you
look at the scores for exactly `A`, `B`, `C` and ignore the rest.

### Softmax

The function that turns raw scores into percentages that sum to 1. Bigger scores
get exponentially more of the mass.

### Autoregressive

Generating one token at a time, each one conditioned on all the previous ones. The
reason normal LLMs are slow: 100 tokens means 100 sequential passes. Decision
models are pitched on being *non*-autoregressive.

### Prefill vs decode

**Prefill** = reading your input (fast, fully parallel). **Decode** = writing the
answer (slow, strictly sequential). nanojev does prefill only, zero decode steps.
That's the entire speed story.

### Encoder vs decoder

**Decoder** — built to generate text (Qwen3-0.6B here). **Encoder** — built to
understand text and output a score, never generating (deberta here). Classifiers
are traditionally encoders, which is why decision models are widely assumed
to be encoders underneath.

### Zero-shot

Handling categories it was never trained on, supplied at request time. You can
invent new options today and it works, with no retraining. Both backends are
zero-shot.

### argmax

"Whichever option has the highest score." The winner.

### Expected value

A probability-weighted average. How Score works: with levels 0/1/2 and
probabilities 20%/60%/20%, the answer is `0(0.2) + 1(0.6) + 2(0.2)` = **1.0**. This
is why scores land *between* levels instead of snapping to one.

### Entropy

A measure of how spread out a distribution is. Maximum when all options are
equally likely, zero when one is certain. We turn it into the `confidence` number.

### One-hot

Representing the true answer as all zeros with a single 1. What Brier compares
your probabilities against.

---

## Part 4: named things

- **System One model** — the commercial framing for this class of model, after Kahneman's *Thinking, Fast and Slow*: System 1 is fast and intuitive, System 2 slow and deliberate. The argument is that reasoning LLMs are System 2 and the machine-facing System 1 was missing.
- **Noul** — short for **Bernoulli**, the coin-flip distribution. The three question types map onto code: Choice→`match`, Score→sort key, Noul→`if`.
- **GLiNER2 / GLiClass** — existing open encoder models taking a runtime-defined schema and returning structured output in one pass. The closest published prior art ([arXiv:2507.18546](https://arxiv.org/abs/2507.18546)).
- **banking77** — public dataset of 13,083 bank-support messages across 77 intents. The test set used throughout.
