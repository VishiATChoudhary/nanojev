"""Question types: Choice, Score and Noul.

Every question type must be able to enumerate its complete answer space before
the model runs. That is the whole trick: if the set of legal answers is known up
front, the model never generates text, it only distributes probability mass over
a fixed set. Off-schema output is then impossible by construction rather than by
validation-and-retry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence


def _entropy_confidence(probs: Sequence[float]) -> float:
    """Confidence as normalised certainty: 1 - H(p)/H_max, in [0, 1].

    A uniform distribution scores 0, a one-hot distribution scores 1. This is
    reported separately from the probabilities so callers can gate on "how sure"
    without having to re-derive it from the distribution every time.
    """
    n = len(probs)
    if n < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0.0)
    return max(0.0, min(1.0, 1.0 - h / math.log(n)))


@dataclass
class Choice:
    """Select exactly one option from a defined set.

    `criteria` may be a plain sequence of option names, or a mapping from option
    name to a description of when that option applies. Descriptions cost a few
    input tokens and are usually worth it when options are easily confused.
    """

    instructions: str
    criteria: Sequence[str] | Mapping[str, str]

    @property
    def options(self) -> list[str]:
        return list(self.criteria)

    def descriptions(self) -> list[str | None]:
        if isinstance(self.criteria, Mapping):
            return [self.criteria[k] for k in self.criteria]
        return [None] * len(self.criteria)


@dataclass
class Score:
    """Rate against ordered, descriptive levels.

    The answer is the expected value over the level distribution, not the
    argmax, which is why scores come back fractional (1.035 rather than 1). A
    model that is torn between "calm" and "frustrated" should say so by landing
    between them, not by picking a side.

    Needs at least two levels. Jev caps these at ten, and the reasoning applies
    here too: levels only help while you can still describe each one distinctly.
    """

    instructions: str
    criteria: Sequence[str]

    @property
    def options(self) -> list[str]:
        return list(self.criteria)

    def descriptions(self) -> list[str | None]:
        return [None] * len(self.criteria)


@dataclass
class Noul:
    """A yes/no question, answered with a probability rather than a boolean.

    "Noul" is short for bernoulli. The three question types map onto control
    flow: a Choice is a match statement, a Score is a sort key, a Noul is an if.
    Collapsing to a bool is the caller's job, and the threshold is theirs.
    """

    instructions: str

    @property
    def options(self) -> list[str]:
        # "false"/"true" rather than "no"/"yes" is not cosmetic. Measured on
        # Qwen3-0.6B, the no/yes framing saturated at ~1.0 for every input,
        # including plainly non-urgent ones; false/true separates them.
        return ["false", "true"]

    def descriptions(self) -> list[str | None]:
        return [None, None]


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    raw: dict[str, float] = field(default_factory=dict, repr=False)


@dataclass
class ScoreAnswer:
    score: float
    probabilities: dict[str, float]
    confidence: float
    raw: dict[str, float] = field(default_factory=dict, repr=False)


@dataclass
class NoulAnswer:
    noul: float
    confidence: float
    raw: dict[str, float] = field(default_factory=dict, repr=False)


Question = Choice | Score | Noul
Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer
