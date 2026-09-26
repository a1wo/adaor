"""Edit-strength schedules used by Adaptive-Origin Guidance.

A schedule maps the user-facing edit strength ``alpha`` in ``[0, 1]`` onto a
coefficient in ``[0, 1]``. Two schedules are needed by
:func:`adaor.guidance.adaptive_origin_guidance`:

``origin_schedule``
    ``s(alpha)``, the mixing weight between the identity (REC) prediction and
    the standard null prediction that together form the adaptive origin.

``steer_schedule``
    the coefficient applied to the CFG steering term.

Every schedule here satisfies ``f(0) == 0`` and ``f(1) == 1``, which is what
gives AdaOr its two boundary conditions: at ``alpha = 0`` the prediction is
exactly the identity prediction, and at ``alpha = 1`` it is exactly standard
CFG. See :mod:`adaor.guidance` for the derivation.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Union

Schedule = Callable[[float], float]
ScheduleLike = Union[str, Schedule]

__all__ = ["SCHEDULES", "Schedule", "ScheduleLike", "resolve_schedule"]


def _linear(alpha: float) -> float:
    return alpha


def _sqrt(alpha: float) -> float:
    return math.sqrt(alpha)


def _square(alpha: float) -> float:
    return alpha * alpha


def _cosine(alpha: float) -> float:
    """Smooth ease-in/ease-out, ``0`` and ``1`` approached with zero slope."""
    return 0.5 * (1.0 - math.cos(math.pi * alpha))


#: Named schedules available to the CLI and to :func:`resolve_schedule`.
SCHEDULES: Dict[str, Schedule] = {
    "linear": _linear,
    "sqrt": _sqrt,
    "square": _square,
    "cosine": _cosine,
}


def resolve_schedule(schedule: ScheduleLike) -> Schedule:
    """Return a callable schedule from a name or a callable.

    Raises:
        KeyError: if ``schedule`` is a string that is not in :data:`SCHEDULES`.
        TypeError: if ``schedule`` is neither a string nor callable.
    """
    if callable(schedule):
        return schedule
    if isinstance(schedule, str):
        try:
            return SCHEDULES[schedule]
        except KeyError:
            raise KeyError(
                f"unknown schedule {schedule!r}; "
                f"available: {sorted(SCHEDULES)}"
            ) from None
    raise TypeError(
        f"schedule must be a name or a callable, got {type(schedule).__name__}"
    )
