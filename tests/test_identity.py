"""Resolving the identity instruction, including the failure that matters.

Silently guessing a ``REC`` token for a trained checkpoint would produce a
meaningless identity prediction, and the symptom -- a bad ``alpha = 0`` frame --
looks like a modelling problem rather than a configuration one. So the
guess is refused.
"""

from __future__ import annotations

import pytest

from adaor.identity import (
    PROMPT_ONLY_INSTRUCTIONS,
    REC_INSTRUCTIONS,
    find_rare_tokens,
    resolve_rec_instruction,
)


def test_explicit_override_wins_over_every_default() -> None:
    assert (
        resolve_rec_instruction(
            "lucy-edit", rec_instruction="<my-token>", trained_rec_token=True
        )
        == "<my-token>"
    )


def test_prompt_only_is_the_default_for_an_off_the_shelf_checkpoint() -> None:
    resolved = resolve_rec_instruction("qwen-image-edit")
    assert resolved == PROMPT_ONLY_INSTRUCTIONS["qwen-image-edit"]


def test_unknown_backbone_falls_back_to_the_generic_prompt() -> None:
    assert (
        resolve_rec_instruction("some-new-model")
        == PROMPT_ONLY_INSTRUCTIONS["default"]
    )


def test_trained_token_is_looked_up_when_known() -> None:
    assert (
        resolve_rec_instruction("lucy-edit", trained_rec_token=True)
        == REC_INSTRUCTIONS["lucy-edit"]
    )


def test_trained_token_refuses_to_guess() -> None:
    with pytest.raises(ValueError, match="no default REC token is known"):
        resolve_rec_instruction("qwen-image-edit", trained_rec_token=True)


class FakeTokenizer:
    """Minimal stand-in exposing only what find_rare_tokens uses."""

    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def test_find_rare_tokens_keeps_only_rare_single_characters() -> None:
    tokenizer = FakeTokenizer(
        {
            "hello": 1,  # multi-character
            "a": 2,  # single but common
            "א": 3,  # rare single character
            "中": 4,  # rare single character
            "בג": 5,  # multi-character
        }
    )
    assert find_rare_tokens(tokenizer) == [("א", 3), ("中", 4)]


def test_find_rare_tokens_is_ordered_by_id_and_respects_limit() -> None:
    tokenizer = FakeTokenizer({chr(0x4E00 + i): 100 - i for i in range(10)})
    found = find_rare_tokens(tokenizer, limit=3)
    assert [token_id for _, token_id in found] == [91, 92, 93]
