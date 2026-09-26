"""Continuous Control of Editing Models via Adaptive-Origin Guidance.

Reference implementation of AdaOr. See the README for the method in one
paragraph, and :mod:`adaor.guidance` for the equations.

The quickest path to a continuous sweep:

.. code-block:: python

    from adaor import sweep_edit

    images = sweep_edit(
        image="input.png",
        instruction="make the hair blonde",
        alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
    )

``adaor.guidance`` is importable on its own and needs neither ``torch`` nor
``diffusers``, so the combination rule can be inspected and tested in isolation.
"""

from __future__ import annotations

from .guidance import (
    GuidanceCoefficients,
    adaptive_origin_guidance,
    guidance_coefficients,
    required_predictions,
)
from .identity import (
    PROMPT_ONLY_INSTRUCTIONS,
    REC_INSTRUCTIONS,
    find_rare_tokens,
    resolve_rec_instruction,
)
from .schedules import SCHEDULES, resolve_schedule

__version__ = "0.1.0"

__all__ = [
    "GuidanceCoefficients",
    "PROMPT_ONLY_INSTRUCTIONS",
    "REC_INSTRUCTIONS",
    "SCHEDULES",
    "__version__",
    "adaptive_origin_guidance",
    "edit",
    "find_rare_tokens",
    "guidance_coefficients",
    "required_predictions",
    "resolve_rec_instruction",
    "resolve_schedule",
    "sweep_edit",
]


def __getattr__(name: str):
    # `edit` and `sweep_edit` reach into diffusers. Resolving them lazily keeps
    # `import adaor` free of heavy dependencies.
    if name in ("edit", "sweep_edit"):
        from . import sweep

        return getattr(sweep, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
