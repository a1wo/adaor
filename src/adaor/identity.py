"""Resolving the identity (``REC``) instruction.

AdaOr needs a text condition that means "apply the identity transformation".
There are two ways to obtain one, and they trade training cost against fidelity:

Learned token
    Add a token to the text encoder's vocabulary and train the editing model on
    identity pairs -- identical source and target images -- paired with that
    token. This is what the paper does: roughly 10% of training samples are
    replaced by such pairs, teaching the token to mean faithful reconstruction.
    It gives the cleanest ``alpha = 0`` behaviour, because the model has been
    supervised to reproduce the input exactly.

Prompt only
    Condition on a plain natural-language instruction such as "keep the image
    exactly the same". No training, works with any off-the-shelf editing model,
    and useful for checking the mechanism before committing to a training run.
    The identity prediction is only approximately the identity, so ``alpha = 0``
    reconstructs the input less exactly.

A learned token should be a string that the tokenizer maps to a single id and
that will never appear in a real instruction. Sentinel tokens the vocabulary
already reserves (``<extra_id_0>`` and friends) are the safest choice; rare
single-character tokens also work, and :func:`find_rare_tokens` helps locate
them.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

__all__ = [
    "PROMPT_ONLY_INSTRUCTIONS",
    "REC_INSTRUCTIONS",
    "find_rare_tokens",
    "resolve_rec_instruction",
]

#: Identity instructions for backbones trained with a dedicated ``REC`` token.
#:
#: These must match the token used at training time -- a checkpoint trained with
#: one sentinel will not respond to another. Entries are only defaults; pass an
#: explicit instruction whenever you are unsure.
REC_INSTRUCTIONS: Dict[str, str] = {
    # Lucy-Edit, the paper's backbone: the T5 sentinel token.
    "lucy-edit": "<extra_id_0>",
}

#: Training-free fallbacks, per backbone family. Plain instructions, so any
#: off-the-shelf editing model understands them.
PROMPT_ONLY_INSTRUCTIONS: Dict[str, str] = {
    "default": "keep the image exactly the same",
    "qwen-image-edit": "keep the image exactly the same, do not change anything",
    "flux-klein": "keep the image exactly the same, do not change anything",
}


def resolve_rec_instruction(
    backbone: str,
    *,
    rec_instruction: str | None = None,
    trained_rec_token: bool = False,
) -> str:
    """Pick the identity instruction to condition on.

    Args:
        backbone: backbone key, e.g. ``"qwen-image-edit"``.
        rec_instruction: explicit override; returned unchanged when given.
        trained_rec_token: whether the checkpoint was trained with a dedicated
            ``REC`` token. When ``True`` and no override is given, the token is
            looked up in :data:`REC_INSTRUCTIONS`.

    Raises:
        ValueError: if ``trained_rec_token`` is set for a backbone with no known
            token. Pass ``rec_instruction`` explicitly in that case -- guessing
            would silently produce a meaningless identity prediction, and the
            failure would look like a bad ``alpha = 0`` result rather than a
            configuration error.
    """
    if rec_instruction is not None:
        return rec_instruction

    if trained_rec_token:
        try:
            return REC_INSTRUCTIONS[backbone]
        except KeyError:
            raise ValueError(
                f"no default REC token is known for backbone {backbone!r}. "
                "Pass rec_instruction explicitly, using the same token the "
                "checkpoint was trained with."
            ) from None

    return PROMPT_ONLY_INSTRUCTIONS.get(
        backbone, PROMPT_ONLY_INSTRUCTIONS["default"]
    )


def find_rare_tokens(
    tokenizer,
    *,
    limit: int = 20,
    min_codepoint: int = 1000,
) -> List[Tuple[str, int]]:
    """List single-character tokens unlikely to collide with real instructions.

    Useful when adding a ``REC`` token to a vocabulary that has no free sentinel
    tokens: a token above the Latin range will not appear in an English edit
    instruction, so training it to mean identity cannot degrade normal edits.

    Args:
        tokenizer: any tokenizer exposing ``get_vocab()``.
        limit: maximum number of candidates to return.
        min_codepoint: only consider characters above this codepoint.

    Returns:
        Up to ``limit`` ``(token, token_id)`` pairs, ordered by token id so the
        result is stable across runs.
    """
    vocab = tokenizer.get_vocab()
    candidates = [
        (token, token_id)
        for token, token_id in vocab.items()
        if len(token) == 1 and ord(token) > min_codepoint
    ]
    candidates.sort(key=lambda item: item[1])
    return candidates[:limit]
