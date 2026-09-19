# How it works

## The waste

Ask a language model "is this email billing, technical, or sales?" and it
answers by *writing*: one token at a time, each conditioned on the last. Your
code then parses the string, validates it, and retries when it is malformed.

But the answer is one of three options you already knew. You have used a machine
built to write essays to produce about 1.5 bits, and the slow part — the
sequential writing — did no useful work.

## The mechanism

Lay the options out with single-token labels:

```
[STATE]
Hi, I've been trying to connect my Stripe account for 3 days...

[QUESTION]
Which team should handle this

[OPTIONS]
A. billing - Payment or subscription issues
B. technical - Bugs or integration problems
C. sales - Pricing or account questions

Reply with exactly one label character.
Answer:
```

Run the model forward **once** and stop at the position after `Answer:`. At that
instant the model holds a score for every token it might emit next — roughly
150,000 of them. Take the three for `" A"`, `" B"`, `" C"`, normalise them
against each other, and discard everything else.

```
billing 3%    technical 93%    sales 4%
```

No token was generated. In transformer terms: **prefill only, zero decode
steps**.

An off-menu answer is not rejected by a validator; there was never a slot for it
to occupy. That is a stronger guarantee than validate-and-retry, and it is why
this can be claimed without a benchmark.

### Read the token the model would actually emit

The anchor ends without a trailing space, and the labels are read in their
space-prefixed form (`" A"`, not `"A"`). This is load-bearing rather than
fussy: with the wrong form the label tokens held 0.3% of the probability mass
and the answers were noise. See [findings.md](findings.md).

## The three question types

| type | returns | maps to |
|---|---|---|
| `Choice` | the winner, a probability per option, a confidence | a `match` statement |
| `Score` | a probability-weighted value over ordered levels | a sort key |
| `Noul` | a probability that a claim holds | an `if` |

`Score` returns an expected value, not an index, so it can land *between*
levels: given levels `calm / annoyed / angry` with probabilities 20/60/20, the
answer is `0(0.2) + 1(0.6) + 2(0.2) = 1.0`. A model torn between two levels says
so, rather than being forced to pick one.

`Noul` is short for *bernoulli*. It returns a probability rather than a boolean
because the threshold belongs to the caller.

## Cancelling position bias

Small models care where an option appears in the list, not just what it says.
`nanojev` renders each question several times with the options cyclically
rotated, then averages the probabilities back in canonical order. Every option
spends equal time in every slot, so the positional preference cancels.

The rotations go into the *same batch*, so this costs extra rows in a matrix
multiply rather than extra round trips. Measured on banking77 (20-way), it is
worth 12.5 accuracy points.

## Many questions, one state

All questions about a state go into a single batched forward pass. The shared
prefix — system prompt plus the state itself — is encoded once and its keys and
values are reused across every question, so cost scales with the questions you
add rather than with the state you would otherwise re-read.

The split point is the longest common **token** prefix across the prompts, so
the cached path computes exactly what the uncached path would. `bench/
equivalence.py` asserts this before any timing is trusted.

Measured: eight questions cost about 3.7x one question. Sublinear, not flat.
The cache earns more as the state grows — on a ~3.9k-token state, sharing the
prefix was 2.7x faster than not.

## More options than labels

The decoder can only use as many options as the tokenizer has usable
single-token labels (52 for Qwen3). Beyond that, questions are split in two
stages: score within groups, then choose between the group winners, combining as
`P(option) = P(group) × P(option | group)`.

This mirrors the two-stage approach described for commercial models of this kind
beyond their own option limit. The result
stays a proper normalised distribution, so calibration applies unchanged through
it. Verified on 120 synthetic options and on all 77 banking77 intents.

## Then make the numbers honest

Everything above produces probabilities. It does not make them *true*. That is
a separate problem, and the more important one — see the calibration section of
the [README](../README.md).
