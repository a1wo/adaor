"""Adaptive-Origin Guidance (AdaOr).

Standard classifier-free guidance on an instruction-based editing model reads

.. code-block:: text

    eps^w = eps(z_t; c_I, null) + w * (eps(z_t; c_I, c_T) - eps(z_t; c_I, null))

The first term is the *guidance origin*. In an editing model the null
instruction behaves as an "any edit" command rather than a "no edit" command,
so as ``w -> 0`` the prediction collapses onto a generic manifold of edited
images instead of onto the input. That is why sweeping ``w`` does not give a
smooth transition from the input to the edited result.

AdaOr replaces the fixed origin with an origin that depends on the edit
strength ``alpha`` in ``[0, 1]``:

.. code-block:: text

    O(alpha) = s(alpha) * eps(z_t; c_I, null) + (1 - s(alpha)) * eps(z_t; c_I, REC)

    eps^{w,alpha} = O(alpha) + g(alpha) * w * (eps(z_t; c_I, c_T) - eps(z_t; c_I, null))

where ``REC`` is an instruction meaning the identity transformation, ``s`` is
the origin schedule and ``g`` the steering schedule. Both are monotonically
increasing with ``f(0) = 0`` and ``f(1) = 1``, which gives the two boundary
conditions that make the sweep continuous:

``alpha = 0``
    the prediction is exactly ``eps(z_t; c_I, REC)`` -- the input is preserved.

``alpha = 1``
    the prediction is exactly standard CFG -- the model's default editing
    behaviour is recovered.

Schedules
---------
The paper uses ``s(alpha) = sqrt(alpha)`` for the origin and ``g(alpha) =
alpha`` for the steering term; those are the defaults here. A variant that
circulated in correspondence about the method mixes the origin linearly and
takes the square root in the steering term instead; reproduce it with
``origin_schedule="linear", steer_schedule="sqrt"``. Both satisfy the boundary
conditions but trace different paths in between, so a sweep is not identical
between them.

Backends
--------
The functions below use only arithmetic that is shared by ``torch.Tensor`` and
``numpy.ndarray``, so they work with either and have no hard dependency on a
deep-learning framework. Scalar coefficients stay Python floats, which lets
``torch`` handle dtype and device promotion on its own -- no ``.to()`` calls and
no ``numpy`` scalars leaking into a CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

from .schedules import ScheduleLike, resolve_schedule

__all__ = [
    "GuidanceCoefficients",
    "adaptive_origin_guidance",
    "guidance_coefficients",
    "required_predictions",
]


@dataclass(frozen=True)
class GuidanceCoefficients:
    """Scalar weights applied to each noise prediction for one ``alpha``.

    The guided prediction is

    ``uncond * uncond_weight + identity * identity_weight + text * text_weight``

    Attributes:
        origin_mix: ``s(alpha)``, the origin mixing weight.
        steer: ``g(alpha) * guidance_scale``, the full steering coefficient.
        uncond_weight: weight on the null prediction.
        identity_weight: weight on the identity (REC) prediction.
        text_weight: weight on the instruction-conditioned prediction.
    """

    origin_mix: float
    steer: float
    uncond_weight: float
    identity_weight: float
    text_weight: float


def guidance_coefficients(
    alpha: float,
    guidance_scale: float,
    *,
    origin_schedule: ScheduleLike = "sqrt",
    steer_schedule: ScheduleLike = "linear",
    steer_from_identity: bool = False,
) -> GuidanceCoefficients:
    """Compute the per-prediction weights for a given edit strength.

    Exposed separately from :func:`adaptive_origin_guidance` so that a pipeline
    can ask which forward passes it actually needs (see
    :func:`required_predictions`) before paying for them, and so that the
    algebra can be unit-tested without allocating tensors.

    Args:
        alpha: edit strength in ``[0, 1]``.
        guidance_scale: the backbone's usual CFG scale ``w``.
        origin_schedule: ``s(alpha)``, by name or callable.
        steer_schedule: ``g(alpha)``, by name or callable.
        steer_from_identity: if ``True``, subtract the identity prediction
            rather than the null prediction in the steering term. This is the
            unstable ablation discussed in the paper, kept so it can be
            reproduced; it is not the method.

    Raises:
        ValueError: if ``alpha`` is outside ``[0, 1]``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    origin_mix = float(resolve_schedule(origin_schedule)(alpha))
    steer = float(resolve_schedule(steer_schedule)(alpha)) * float(guidance_scale)

    # Origin: s * uncond + (1 - s) * identity.
    uncond_weight = origin_mix
    identity_weight = 1.0 - origin_mix

    # Steering: steer * (text - baseline), where the baseline is normally the
    # null prediction.
    text_weight = steer
    if steer_from_identity:
        identity_weight -= steer
    else:
        uncond_weight -= steer

    return GuidanceCoefficients(
        origin_mix=origin_mix,
        steer=steer,
        uncond_weight=uncond_weight,
        identity_weight=identity_weight,
        text_weight=text_weight,
    )


def required_predictions(
    alpha: float,
    guidance_scale: float,
    *,
    origin_schedule: ScheduleLike = "sqrt",
    steer_schedule: ScheduleLike = "linear",
    steer_from_identity: bool = False,
) -> FrozenSet[str]:
    """Return which of ``{"uncond", "identity", "text"}`` affect the result.

    A prediction whose weight is exactly zero can be skipped, which saves a
    transformer forward pass. With the default schedules this drops the cost
    from three passes to two at both ends of the sweep: ``alpha = 0`` needs only
    the identity prediction, and ``alpha = 1`` only the null and instruction
    predictions.

    Skipping is decided from the computed weights rather than from ``alpha``, so
    a custom schedule that does not vanish at the endpoints stays correct.
    """
    coefficients = guidance_coefficients(
        alpha,
        guidance_scale,
        origin_schedule=origin_schedule,
        steer_schedule=steer_schedule,
        steer_from_identity=steer_from_identity,
    )
    needed = {
        "uncond": coefficients.uncond_weight,
        "identity": coefficients.identity_weight,
        "text": coefficients.text_weight,
    }
    return frozenset(name for name, weight in needed.items() if weight != 0.0)


def adaptive_origin_guidance(
    *,
    noise_pred_uncond=None,
    noise_pred_text=None,
    noise_pred_identity=None,
    guidance_scale: float,
    alpha: float,
    origin_schedule: ScheduleLike = "sqrt",
    steer_schedule: ScheduleLike = "linear",
    steer_from_identity: bool = False,
):
    """Combine three noise predictions into one adaptive-origin prediction.

    Args:
        noise_pred_uncond: ``eps(z_t; c_I, null)``.
        noise_pred_text: ``eps(z_t; c_I, c_T)``, the instruction prediction.
        noise_pred_identity: ``eps(z_t; c_I, REC)``, the identity prediction.
        guidance_scale: the backbone's usual CFG scale ``w``.
        alpha: edit strength in ``[0, 1]``.
        origin_schedule: ``s(alpha)``; defaults to the paper's ``sqrt``.
        steer_schedule: ``g(alpha)``; defaults to the paper's ``linear``.
        steer_from_identity: reproduce the unstable ablation (see
            :func:`guidance_coefficients`).

    A prediction may be ``None`` exactly when :func:`required_predictions` says
    it is not needed for this ``alpha``, so callers can skip the corresponding
    forward pass.

    Returns:
        The guided prediction, same type and shape as the inputs.

    Raises:
        ValueError: if ``alpha`` is outside ``[0, 1]``, or if a prediction that
            carries a nonzero weight was not supplied.
    """
    coefficients = guidance_coefficients(
        alpha,
        guidance_scale,
        origin_schedule=origin_schedule,
        steer_schedule=steer_schedule,
        steer_from_identity=steer_from_identity,
    )

    terms = (
        ("noise_pred_uncond", noise_pred_uncond, coefficients.uncond_weight),
        ("noise_pred_identity", noise_pred_identity, coefficients.identity_weight),
        ("noise_pred_text", noise_pred_text, coefficients.text_weight),
    )

    guided = None
    for name, prediction, weight in terms:
        if weight == 0.0:
            continue
        if prediction is None:
            raise ValueError(
                f"{name} is required at alpha={alpha} (weight {weight:.6g}) "
                "but was not provided"
            )
        contribution = weight * prediction
        guided = contribution if guided is None else guided + contribution

    if guided is None:
        # Every weight vanished. Only reachable via a custom schedule; fall back
        # to whichever prediction the caller did supply so we return the right
        # shape rather than None.
        for name, prediction, _ in terms:
            if prediction is not None:
                return 0.0 * prediction
        raise ValueError("no noise predictions were provided")

    return guided
