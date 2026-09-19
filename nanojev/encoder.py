"""An encoder backend behind the same API, because the evidence pointed here.

The common guess about hosted decision models is that they are encoders with
classification heads rather than a new class of model (the GLiNER / GLiClass /
deberta-zeroshot family). Measuring it locally supports that guess: an
off-the-shelf 184M zero-shot encoder beats our 0.6B decoder logit-read on
banking77 and arrives already calibrated.

So this exposes that path through the identical Choice / Score / Noul API. The
entire premise of a typed decision layer is that callers should not have to care
which machine answered, only that the answer is in-schema and the probability is
honest. Swapping the backend is a constructor argument.

Mechanically: each option becomes an NLI hypothesis, the state is the premise,
and we softmax the entailment logits across options. Every (state, option) pair
for every question goes into one batch, so the comparison against the decoder is
a fair one rather than an artefact of an unbatched pipeline.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .engine import Response, _pick_device
from .primitives import Answer, Question, _entropy_confidence

DEFAULT_ENCODER = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


class EncoderSystemOne:
    """Same contract as SystemOne, different machine underneath."""

    def __init__(
        self,
        model_id: str = DEFAULT_ENCODER,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        temperature: float = 1.0,
        hypothesis_template: str = "This example is {}.",
        max_batch_rows: int = 64,
    ):
        self.device = device or _pick_device()
        self.temperature = temperature
        self.hypothesis_template = hypothesis_template
        self.max_batch_rows = max_batch_rows

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

        # These checkpoints come in two shapes: 3-way NLI (entailment / neutral
        # / contradiction) and 2-way (entailment / not_entailment). For a Noul we
        # need the opposing class, and picking it by name is not optional: an
        # earlier version defaulted the opposite to index 0, which on a 2-way
        # head is *also* entailment, so every yes/no question returned exactly
        # 0.500 and looked like honest uncertainty rather than a bug.
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail_id = labels.get("entailment", self.model.config.num_labels - 1)
        opposite = labels.get("contradiction", labels.get("not_entailment"))
        if opposite is None or opposite == self.entail_id:
            candidates = [i for i in range(self.model.config.num_labels) if i != self.entail_id]
            if not candidates:
                raise ValueError(f"cannot find a non-entailment class in {self.model.config.id2label}")
            opposite = candidates[-1]
        self.contra_id = opposite

    @property
    def max_options(self) -> int:
        return 10_000  # no label-token budget here, unlike the decoder path

    @torch.inference_mode()
    def decide(self, state: str, questions: dict[str, Question], **_: object) -> Response:
        t0 = time.perf_counter()

        from .primitives import Noul

        premises: list[str] = []
        hypotheses: list[str] = []
        for key, q in questions.items():
            if isinstance(q, Noul):
                # The claim itself is the hypothesis, but it must go through the
                # same template the checkpoint was trained on. Passing the bare
                # claim is off-distribution and collapses: measured on
                # deberta-v3-base-zeroshot, "checkout is failing" against a state
                # about failing checkouts scored 0.005 bare and 0.980 templated.
                premises.append(state)
                hypotheses.append(self.hypothesis_template.format(q.instructions.rstrip(".")))
                continue
            for opt, desc in zip(q.options, q.descriptions()):
                # A description, when present, is the more informative hypothesis.
                target = f"{opt}: {desc}" if desc else opt
                premises.append(state)
                hypotheses.append(self.hypothesis_template.format(target))

        rows: list[list[float]] = []
        for start in range(0, len(premises), self.max_batch_rows):
            stop = start + self.max_batch_rows
            batch = self.tokenizer(
                premises[start:stop],
                hypotheses[start:stop],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            rows.extend(self.model(**batch).logits.float().tolist())

        answers: dict[str, Answer] = {}
        cursor = 0
        for key, q in questions.items():
            if isinstance(q, Noul):
                # Entailment vs contradiction on the single claim, neutral left
                # out: "the state does not say" is not evidence against.
                logits = rows[cursor]
                cursor += 1
                pair = torch.tensor([logits[self.contra_id], logits[self.entail_id]])
                probs = torch.softmax(pair / self.temperature, dim=-1).tolist()
            else:
                n = len(q.options)
                raw = torch.tensor([r[self.entail_id] for r in rows[cursor : cursor + n]])
                cursor += n
                probs = torch.softmax(raw / self.temperature, dim=-1).tolist()
            answers[key] = self._to_answer(q, probs)

        return Response(
            answers=answers,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            input_tokens=0,
            sequences=len(premises),
            meta={"backend": "encoder"},
        )

    def _to_answer(self, q: Question, probs: list[float]) -> Answer:
        from .primitives import ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer

        dist = {opt: p for opt, p in zip(q.options, probs)}
        confidence = _entropy_confidence(probs)
        if isinstance(q, Noul):
            return NoulAnswer(noul=dist["true"], confidence=abs(dist["true"] - 0.5) * 2.0)
        if isinstance(q, Score):
            return ScoreAnswer(
                score=sum(i * p for i, p in enumerate(probs)),
                probabilities=dist,
                confidence=confidence,
            )
        return ChoiceAnswer(
            choice=max(dist, key=dist.get), probabilities=dist, confidence=confidence
        )
