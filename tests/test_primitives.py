"""Tests for the question types. No model needed."""

import pytest

from nanojev.primitives import Choice, Noul, Score, _entropy_confidence


def test_confidence_is_zero_for_a_uniform_distribution():
    assert _entropy_confidence([0.25] * 4) == pytest.approx(0.0, abs=1e-9)


def test_confidence_is_one_for_a_certain_distribution():
    assert _entropy_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-9)


def test_confidence_rises_as_mass_concentrates():
    spread = _entropy_confidence([0.4, 0.3, 0.3])
    peaked = _entropy_confidence([0.9, 0.05, 0.05])
    assert 0.0 < spread < peaked < 1.0


def test_choice_accepts_a_bare_list_or_a_described_mapping():
    bare = Choice(instructions="pick", criteria=["a", "b"])
    assert bare.options == ["a", "b"]
    assert bare.descriptions() == [None, None]

    described = Choice(instructions="pick", criteria={"a": "the first", "b": "the second"})
    assert described.options == ["a", "b"]
    assert described.descriptions() == ["the first", "the second"]


def test_noul_is_a_two_option_question_ordered_false_then_true():
    # Ordering matters: the answer reads index 1 as the probability of "true".
    assert Noul(instructions="is it urgent").options == ["false", "true"]


def test_score_exposes_its_levels_in_order():
    score = Score(instructions="how angry", criteria=["calm", "annoyed", "furious"])
    assert score.options == ["calm", "annoyed", "furious"]
