# What building it revealed

The point of building a rough version was to find out which claims survive
contact with a measurement. These are the things that only showed up because
something was being measured.

## The bug that invalidated every decoder number

The model lays out options as `A. billing / B. technical / C. sales` and the
prompt ends with an anchor, `Answer:`. We then read the scores for the label
tokens.

The first version anchored with `"Answer: "` — trailing space — and read the
token for the bare character `"C"`. But after that anchor the model's preferred
token is the **space-prefixed** `" C"`. Measured on Qwen3-0.6B:

```
UNRESTRICTED top-8 next tokens:
    0.6959  ' C'      <- what the model actually wants to say
    0.0647  '2'
    0.0504  '8'
    ...
mass on the label tokens we were reading:
   A: 0.000001
   B: 0.000006
   C: 0.003223      <- 0.3% of the distribution
```

Every label probability was being computed from the **tail** of the
distribution. The ratios between three numbers that small are mostly noise,
which produced a convincing-looking phantom: option `A` almost never won,
whatever was in it, including correct answers.

Fixed by anchoring with `"Answer:"` and reading `" A"`, `" B"`, `" C"`. Label
mass went from **0.003 to 0.9999**.

The lesson generalises past this project: when you read logits for specific
tokens, check what fraction of the probability mass you are actually looking at.
`nanojev` now warns at runtime if the label tokens hold under 5% of the mass.

## I nearly removed the feature that mattered most

Averaging over cyclic rotations of the option order is meant to cancel position
bias. On three hand-picked examples it looked actively harmful, and I was ready
to delete it.

The real evaluation said otherwise:

| rotations | accuracy | ECE (raw) |
|---|---|---|
| 1 | 0.470 | 0.355 |
| 3 | **0.595** | **0.134** |

Worth 12.5 accuracy points and more than half the miscalibration. Three examples
are an anecdote; 400 are a measurement. The default is now 3.

## Two silent crashes, no traceback

1. `out.logits` computes a vocabulary-sized vector at **every** position. A
   batch of long states allocated roughly 7 GB to produce one row we read.
   Fixed with `logits_to_keep=1`.
2. Eager attention materialises a full `batch × heads × tokens × tokens` matrix.
   Fixed with SDPA, plus a token budget that splits oversized batches.

Both killed the process with no error message, which is its own lesson: a
benchmark that dies silently looks identical to one that is still running.

## A bug disguised as calibrated uncertainty

The encoder's yes/no path returned exactly `0.500` on every question. That looks
precisely like a model being honestly unsure — the most plausible possible
output in a project about honest uncertainty.

The checkpoint is a 2-way head (`entailment` / `not_entailment`). Looking up
`contradiction` fell back to index 0, which is *also* entailment, so the code
was comparing a number against itself.

## The same checkpoint, off its own training template

Even after that fix, the encoder scored `0.005` on a claim that plainly restated
the state. These checkpoints are trained with a specific hypothesis template
(`"This example is {}."`), and a bare claim is off-distribution:

| hypothesis form | "checkout is failing" vs a state about failing checkouts |
|---|---|
| bare claim | 0.005 |
| templated | **0.980** |

## A benchmark that measured nothing

The first "how fast is generating instead of reading?" comparison reported
identical times for 16, 64 and 256 tokens. The model emitted one label and hit
end-of-sequence, so all three measured a single token. `min_new_tokens` forces
the comparison to be real.

## An unfair baseline, one hour after criticising one

The first encoder measurement said 229 ms/decision, four times slower than the
decoder. That number was Hugging Face's convenience pipeline running twenty
unbatched passes, not the method. Batched properly through the same API, the
encoder is roughly **twice as fast**.

This is the classic benchmarking error — an uncharitable baseline flattering the
thing you built — and I had just finished reading criticism of exactly it
elsewhere. It took an hour to repeat, and was only caught by re-measuring.

## A test case that tested nothing

While regenerating figures, a "restatement" check paired the claim *"the
customer had a payment problem"* with a state about HTTP 500 errors. That is not
a restatement, it is an inference, so the comparison was meaningless. The
original version of the test had paired it with a declined-card state.

Worth stating because the failure mode is subtle: the code was correct, the
numbers were real, and the experiment still measured the wrong thing.
