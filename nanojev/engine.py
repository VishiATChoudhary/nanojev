"""The decision engine: one batched forward pass, no autoregressive decoding.

How this works, and why it is fast:

A normal LLM answers a classification question by generating tokens one at a
time, then you parse the text and hope it matches your schema. Almost all of
that work is wasted. The answer to "which of these five teams handles this
ticket" is at most log2(5) bits of information, and we know the five options
before the model runs.

So instead: lay out the options with single-token labels, run the model forward
exactly once, and read the logits at the position where the answer would have
started. Restrict those logits to the label tokens and softmax. The result is a
full probability distribution over the legal answers, obtained for the price of
a prefill and zero decode steps. Off-schema answers are not filtered out, they
are unrepresentable.
"""

from __future__ import annotations

import string
import time
import warnings
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .primitives import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    _entropy_confidence,
)

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

SYSTEM_PROMPT = (
    "You are a decision engine. You read a state and answer a question by "
    "choosing exactly one labelled option. You never explain and never write "
    "anything except the single label character."
)

ANSWER_ANCHOR = "Answer:"


@dataclass
class Response:
    answers: dict[str, Answer]
    latency_ms: float
    input_tokens: int
    sequences: int
    meta: dict = field(default_factory=dict, repr=False)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SystemOne:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        temperature: float = 1.0,
        rotations: int = 3,
        max_batch_tokens: int = 8192,
    ):
        """
        temperature: calibration temperature applied to label logits. Fit it with
            nanojev.calibrate rather than guessing; 1.0 means uncalibrated.
        rotations: how many cyclic rotations of the option order to average over.
            Small models have a real position bias, and averaging over rotations
            cancels most of it. Costs an extra sequence in the batch per
            rotation, not an extra round trip. Measured on banking77 (20-way):
            rotations=1 scores 0.470 accuracy at ECE 0.355, rotations=3 scores
            0.595 at ECE 0.134. Worth the extra sequences; the default is 3.
        """
        self.device = device or _pick_device()
        self.temperature = temperature
        self.rotations = max(1, rotations)
        # Token budget per forward pass. Questions beyond it are split across
        # several passes, so a long state with many questions degrades into
        # more passes rather than into an out-of-memory kill.
        self.max_batch_tokens = max_batch_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # SDPA keeps attention memory linear in sequence length. With eager
        # attention a batch of long states materialises a full
        # batch x heads x tokens x tokens matrix and the process dies.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation="sdpa"
        )
        self.model.to(self.device)
        self.model.eval()

        self.label_chars = self._build_label_alphabet()
        self.label_ids = {c: self._single_token_id(c) for c in self.label_chars}
        self._warned_low_mass = False

    # -- label alphabet -------------------------------------------------

    def _single_token_id(self, ch: str) -> int:
        """Token id for a label as it appears after the answer anchor.

        The anchor ends without a space, so the model's next token is the
        SPACE-PREFIXED form (" C"), not the bare character ("C"). Reading the
        bare character instead puts you in the tail of the distribution: on
        Qwen3-0.6B the bare "C" held 0.003 of the mass while " C" held 0.696,
        so the label probabilities were essentially noise. Always read the form
        the model would actually emit.
        """
        ids = self.tokenizer.encode(" " + ch, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"label {ch!r} is not a single token: {ids}")
        return ids[0]

    def _build_label_alphabet(self) -> list[str]:
        """Keep only characters this tokenizer encodes as exactly one token.

        Reading a single logit position only works if each label is one token,
        so we verify rather than assume. Uppercase first (models are most used to
        multiple-choice in that form), then lowercase, then digits.
        """
        candidates = list(string.ascii_uppercase) + list(string.ascii_lowercase)
        good = []
        for ch in candidates:
            try:
                self._single_token_id(ch)
            except ValueError:
                continue
            good.append(ch)
        if len(good) < 2:
            raise RuntimeError("tokenizer has fewer than 2 usable single-token labels")
        return good

    @property
    def max_options(self) -> int:
        return len(self.label_chars)

    # -- prompt construction --------------------------------------------

    def _render(self, state: str, question: Question, order: list[int]) -> str:
        """Render one question with its options in the given order.

        `order` indexes into the question's canonical option list, so a rotated
        prompt and its un-rotated twin differ only in the order the options are
        printed. Everything else is byte-identical.
        """
        options = question.options
        descriptions = question.descriptions()

        lines = []
        for slot, opt_idx in enumerate(order):
            label = self.label_chars[slot]
            desc = descriptions[opt_idx]
            if desc:
                lines.append(f"{label}. {options[opt_idx]} - {desc}")
            else:
                lines.append(f"{label}. {options[opt_idx]}")
        rendered_options = "\n".join(lines)

        if isinstance(question, Noul):
            # Presented as a claim to be judged, not a question to be answered.
            # The generic framing below made this model agree with everything.
            user = (
                f"[STATE]\n{state}\n\n"
                f'[CLAIM]\n"{question.instructions}"\n\n'
                f"[OPTIONS]\n{rendered_options}\n\n"
                f"Is that claim true of the state? Reply with one label character."
            )
        else:
            preamble = (
                "Rate the state against these ordered levels, lowest first. "
                "Choose the level that fits best."
                if isinstance(question, Score)
                else "Choose the option that fits the state best."
            )
            user = (
                f"[STATE]\n{state}\n\n"
                f"[QUESTION]\n{question.instructions}\n\n"
                f"[TASK]\n{preamble}\n\n"
                f"[OPTIONS]\n{rendered_options}\n\n"
                f"Reply with exactly one label character."
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Qwen-style templates may open a reasoning block; we want the very next
        # token to be the answer label, so close it and anchor the answer.
        if text.rstrip().endswith("<think>"):
            text = text.rstrip()[: -len("<think>")]
        return text + ANSWER_ANCHOR

    def _rotations_for(self, n_options: int) -> list[list[int]]:
        r = min(self.rotations, n_options)
        step = max(1, n_options // r)
        return [[(i + k * step) % n_options for i in range(n_options)] for k in range(r)]

    # -- inference -------------------------------------------------------

    @staticmethod
    def _common_prefix_len(seqs: list[list[int]]) -> int:
        """Longest token prefix shared by every prompt in the batch.

        Splitting on tokens rather than characters means the cached prefix and
        the per-question suffixes concatenate back to exactly the token ids the
        uncached path would have used. The two code paths are therefore the same
        computation, which is what makes the speedup a real one rather than an
        accounting trick.
        """
        if not seqs:
            return 0
        shortest = min(len(s) for s in seqs)
        first = seqs[0]
        i = 0
        while i < shortest and all(s[i] == first[i] for s in seqs):
            i += 1
        return i

    def _build_prompts(
        self, state: str, questions: dict[str, Question]
    ) -> tuple[list[str], list[tuple[str, list[int]]]]:
        prompts: list[str] = []
        index: list[tuple[str, list[int]]] = []
        for key, q in questions.items():
            n = len(q.options)
            if n > self.max_options:
                raise ValueError(
                    f"question {key!r} has {n} options, max is {self.max_options}"
                )
            if n < 2:
                raise ValueError(f"question {key!r} needs at least 2 options")
            for order in self._rotations_for(n):
                prompts.append(self._render(state, q, order))
                index.append((key, order))
        return prompts, index

    def decide(
        self,
        state: str,
        questions: dict[str, Question],
        use_prefix_cache: bool = True,
    ) -> Response:
        """Answer every question, splitting any that exceed the label budget.

        Questions with more options than we have single-token labels are handled
        in two stages, which is the same shape TypeSafe describes for Choice
        beyond 255: "we do a 2 stage-system of scoring independently then making
        an explicit choice".
        """
        oversized = {
            k: q for k, q in questions.items() if len(q.options) > self.max_options
        }
        if not oversized:
            return self._decide_flat(state, questions, use_prefix_cache)

        normal = {k: q for k, q in questions.items() if k not in oversized}
        response = (
            self._decide_flat(state, normal, use_prefix_cache)
            if normal
            else Response(answers={}, latency_ms=0.0, input_tokens=0, sequences=0)
        )
        for key, q in oversized.items():
            answer, extra = self._decide_hierarchical(state, q, use_prefix_cache)
            response.answers[key] = answer
            response.input_tokens += extra["input_tokens"]
            response.sequences += extra["sequences"]
            response.latency_ms += extra["latency_ms"]
        # Restore the caller's original key order.
        response.answers = {k: response.answers[k] for k in questions}
        return response

    def _decide_hierarchical(
        self, state: str, question: Question, use_prefix_cache: bool
    ) -> tuple[Answer, dict]:
        """Two stages: score within groups, then choose between group winners.

        P(option) = P(its group wins) * P(option within its group), which stays a
        real normalised distribution rather than a heuristic score, so the
        calibration machinery downstream still applies unchanged.
        """
        from .primitives import Choice, ChoiceAnswer

        options = question.options
        descriptions = question.descriptions()
        size = self.max_options
        groups = [list(range(i, min(i + size, len(options)))) for i in range(0, len(options), size)]

        cost = {"input_tokens": 0, "sequences": 0, "latency_ms": 0.0}
        within: list[dict[str, float]] = []
        stage1 = {}
        for gi, g in enumerate(groups):
            stage1[f"g{gi}"] = Choice(
                instructions=question.instructions,
                criteria={options[i]: descriptions[i] for i in g}
                if any(descriptions[i] for i in g)
                else [options[i] for i in g],
            )
        r1 = self._decide_flat(state, stage1, use_prefix_cache)
        for k in ("input_tokens", "sequences", "latency_ms"):
            cost[k] += getattr(r1, k)
        for gi in range(len(groups)):
            within.append(r1.answers[f"g{gi}"].probabilities)

        winners = [max(d, key=d.get) for d in within]
        if len(winners) == 1:
            group_probs = [1.0]
        else:
            # Duplicate winners would collapse into one option and lose mass.
            unique = list(dict.fromkeys(winners))
            r2 = self._decide_flat(
                state,
                {"final": Choice(instructions=question.instructions, criteria=unique)},
                use_prefix_cache,
            )
            for k in ("input_tokens", "sequences", "latency_ms"):
                cost[k] += getattr(r2, k)
            final = r2.answers["final"].probabilities
            group_probs = [final[w] / winners.count(w) for w in winners]

        dist: dict[str, float] = {}
        for gi, g in enumerate(groups):
            for i in g:
                dist[options[i]] = group_probs[gi] * within[gi][options[i]]
        total = sum(dist.values()) or 1.0
        dist = {k: v / total for k, v in dist.items()}

        return (
            ChoiceAnswer(
                choice=max(dist, key=dist.get),
                probabilities=dist,
                confidence=_entropy_confidence(list(dist.values())),
            ),
            cost,
        )

    @torch.inference_mode()
    def _decide_flat(
        self,
        state: str,
        questions: dict[str, Question],
        use_prefix_cache: bool = True,
    ) -> Response:
        """Answer every question about `state`, reading logits instead of decoding.

        With `use_prefix_cache`, the state is encoded exactly once and the cached
        keys and values are shared across every question. Cost then scales with
        the questions you add, not with the state you keep re-reading, which is
        what makes many questions about one state nearly free.
        """
        t0 = time.perf_counter()
        prompts, index = self._build_prompts(state, questions)
        encoded = [self.tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]

        if use_prefix_cache and len(encoded) > 1:
            logits, n_tokens, prefix_len = self._forward_shared_prefix(encoded)
        else:
            logits, n_tokens = self._forward_flat(encoded)
            prefix_len = 0

        self._check_label_mass(logits, index)

        acc: dict[str, torch.Tensor] = {}
        for row, (key, order) in enumerate(index):
            n = len(order)
            slot_ids = [self.label_ids[self.label_chars[s]] for s in range(n)]
            slot_probs = torch.softmax(logits[row, slot_ids] / self.temperature, dim=-1)
            canonical = torch.empty(n, dtype=slot_probs.dtype)
            for slot, opt_idx in enumerate(order):
                canonical[opt_idx] = slot_probs[slot]
            acc[key] = acc.get(key, torch.zeros(n)) + canonical

        answers: dict[str, Answer] = {}
        for key, q in questions.items():
            probs = (acc[key] / acc[key].sum()).tolist()
            answers[key] = self._to_answer(q, probs)

        return Response(
            answers=answers,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            input_tokens=n_tokens,
            sequences=len(prompts),
            meta={"prefix_tokens": prefix_len, "prefix_cache": bool(prefix_len)},
        )

    def _chunk(self, seqs: list[list[int]], overhead: int = 0) -> list[list[int]]:
        """Split row indices into groups that fit the per-pass token budget."""
        width = max(len(s) for s in seqs)
        per_row = max(1, width + overhead)
        rows = max(1, self.max_batch_tokens // per_row)
        return [list(range(i, min(i + rows, len(seqs)))) for i in range(0, len(seqs), rows)]

    def _forward_flat(self, encoded: list[list[int]]) -> tuple[torch.Tensor, int]:
        """Baseline: every prompt in full, left padded, batched within budget."""
        groups = self._chunk(encoded)
        if len(groups) > 1:
            outs, total = [], 0
            for g in groups:
                logits, n = self._forward_flat_one([encoded[i] for i in g])
                outs.append(logits)
                total += n
            return torch.cat(outs, dim=0), total
        return self._forward_flat_one(encoded)

    def _forward_flat_one(self, encoded: list[list[int]]) -> tuple[torch.Tensor, int]:
        width = max(len(e) for e in encoded)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.tensor(
            [[pad] * (width - len(e)) + e for e in encoded], device=self.device
        )
        attention_mask = torch.tensor(
            [[0] * (width - len(e)) + [1] * len(e) for e in encoded], device=self.device
        )
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

        # logits_to_keep=1 is not an optimisation detail, it is load bearing:
        # without it the LM head runs over every position, and a batch of long
        # states allocates a vocab-sized tensor per token (gigabytes) to compute
        # one row we actually read.
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            logits_to_keep=1,
        )
        # Left padding puts every sequence's final real token in the last column.
        return out.logits[:, -1, :].float(), int(attention_mask.sum().item())

    def _forward_shared_prefix(
        self, encoded: list[list[int]]
    ) -> tuple[torch.Tensor, int, int]:
        """Encode the shared prefix once, then run only the differing suffixes."""
        prefix_len = self._common_prefix_len(encoded)
        suffixes = [e[prefix_len:] for e in encoded]
        # Nothing shared worth caching: fall back rather than pay for the setup.
        if prefix_len == 0 or all(len(s) == 0 for s in suffixes):
            logits, n = self._forward_flat(encoded)
            return logits, n, 0

        batch = len(encoded)
        prefix_ids = torch.tensor([encoded[0][:prefix_len]], device=self.device)
        cache = DynamicCache()
        self.model(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            position_ids=torch.arange(prefix_len, device=self.device).unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache.batch_repeat_interleave(batch)

        # The suffix block is left padded, so every row's final real token lands
        # in the same last column and a single logit slice serves the whole batch.
        # Padding sits between the cached prefix and the real suffix tokens, which
        # is harmless: it is masked out, and causal attention means it cannot
        # influence anything to its left either.
        width = max(len(s) for s in suffixes)
        pad = self.tokenizer.pad_token_id
        suffix_ids = torch.tensor(
            [[pad] * (width - len(s)) + s for s in suffixes], device=self.device
        )
        suffix_mask = torch.tensor(
            [[0] * (width - len(s)) + [1] * len(s) for s in suffixes], device=self.device
        )
        attention_mask = torch.cat(
            [torch.ones(batch, prefix_len, dtype=suffix_mask.dtype, device=self.device),
             suffix_mask],
            dim=1,
        )
        # Real suffix tokens continue the prefix's positions; pads are masked so
        # their position value is irrelevant.
        offsets = torch.tensor([width - len(s) for s in suffixes], device=self.device)
        cols = torch.arange(width, device=self.device).unsqueeze(0)
        position_ids = (prefix_len + cols - offsets.unsqueeze(1)).clamp(min=0)

        out = self.model(
            input_ids=suffix_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = out.logits[:, -1, :].float()
        n_tokens = prefix_len + sum(len(s) for s in suffixes)
        return logits.float(), n_tokens, prefix_len

    def _check_label_mass(self, logits: torch.Tensor, index: list) -> None:
        """Warn if the label tokens hold almost none of the probability mass.

        If the model wants to say something other than a label, we are reading
        the tail of the distribution and the ratios between labels are noise.
        This happened during development: the anchor ended in a space, so the
        model's preferred token was " C" while we were reading "C", leaving
        0.3% of the mass under inspection and producing answers that looked
        plausible but were arbitrary. Fail loudly rather than silently.
        """
        if self._warned_low_mass:
            return
        row_probs = torch.softmax(logits[0], dim=-1)
        n = len(index[0][1])
        captured = sum(
            row_probs[self.label_ids[self.label_chars[s]]].item() for s in range(n)
        )
        if captured < 0.05:
            self._warned_low_mass = True
            top = torch.topk(row_probs, 3)
            wanted = [self.tokenizer.decode([i]) for i in top.indices.tolist()]
            warnings.warn(
                f"Label tokens hold only {captured:.1%} of the probability mass; "
                f"the model would rather emit {wanted!r}. Answers are being read "
                f"from the tail of the distribution and are probably meaningless. "
                f"Check ANSWER_ANCHOR and the label tokenisation for this model.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _to_answer(self, q: Question, probs: list[float]) -> Answer:
        options = q.options
        dist = {opt: p for opt, p in zip(options, probs)}
        confidence = _entropy_confidence(probs)

        if isinstance(q, Noul):
            return NoulAnswer(noul=dist["true"], confidence=abs(dist["true"] - 0.5) * 2.0)
        if isinstance(q, Score):
            expected = sum(i * p for i, p in enumerate(probs))
            return ScoreAnswer(score=expected, probabilities=dist, confidence=confidence)
        best = max(dist, key=dist.get)
        return ChoiceAnswer(choice=best, probabilities=dist, confidence=confidence)
