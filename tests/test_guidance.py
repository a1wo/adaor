"""The two boundary conditions are the method's contract; everything else follows.

At ``alpha = 0`` AdaOr must reduce exactly to the identity prediction, and at
``alpha = 1`` exactly to standard CFG. If either drifts, the sweep stops being
continuous at its endpoints -- the failure the method exists to fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaor.guidance import (
    adaptive_origin_guidance,
    guidance_coefficients,
    required_predictions,
)

UNCOND = np.array([1.0, 2.0, 3.0])
TEXT = np.array([4.0, -1.0, 0.5])
IDENTITY = np.array([-2.0, 0.0, 7.0])
SCALE = 5.0


def guide(alpha: float, **kwargs):
    return adaptive_origin_guidance(
        noise_pred_uncond=UNCOND,
        noise_pred_text=TEXT,
        noise_pred_identity=IDENTITY,
        guidance_scale=SCALE,
        alpha=alpha,
        **kwargs,
    )


# -- boundary conditions ---------------------------------------------------


def test_alpha_zero_is_exactly_the_identity_prediction() -> None:
    np.testing.assert_allclose(guide(0.0), IDENTITY)


def test_alpha_one_is_exactly_standard_cfg() -> None:
    expected = UNCOND + SCALE * (TEXT - UNCOND)
    np.testing.assert_allclose(guide(1.0), expected)


@pytest.mark.parametrize(
    ("origin", "steer"),
    [
        ("sqrt", "linear"),  # the paper
        ("linear", "sqrt"),  # the variant from correspondence
        ("cosine", "cosine"),
        ("square", "square"),
    ],
)
def test_boundaries_hold_for_every_schedule_pairing(origin: str, steer: str) -> None:
    kwargs = {"origin_schedule": origin, "steer_schedule": steer}
    np.testing.assert_allclose(guide(0.0, **kwargs), IDENTITY)
    np.testing.assert_allclose(
        guide(1.0, **kwargs), UNCOND + SCALE * (TEXT - UNCOND)
    )


def test_the_two_schedule_pairings_differ_in_between() -> None:
    # Same endpoints, different path -- so a sweep is not interchangeable
    # between the paper's schedules and the variant.
    paper = guide(0.5, origin_schedule="sqrt", steer_schedule="linear")
    variant = guide(0.5, origin_schedule="linear", steer_schedule="sqrt")
    assert not np.allclose(paper, variant)


# -- algebra ---------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_weights_form_an_affine_combination(alpha: float) -> None:
    # The origin weights sum to 1 and the steering term contributes +steer and
    # -steer, so the three weights must always sum to 1. A drift here would
    # scale the prediction's magnitude and change the noise level.
    c = guidance_coefficients(alpha, SCALE)
    total = c.uncond_weight + c.identity_weight + c.text_weight
    assert total == pytest.approx(1.0)


def test_matches_the_written_formula() -> None:
    alpha = 0.36
    c = guidance_coefficients(alpha, SCALE)
    s = np.sqrt(alpha)
    expected = (
        s * UNCOND + (1.0 - s) * IDENTITY + alpha * SCALE * (TEXT - UNCOND)
    )
    assert c.origin_mix == pytest.approx(s)
    np.testing.assert_allclose(guide(alpha), expected)


def test_zero_guidance_scale_leaves_the_bare_origin() -> None:
    result = adaptive_origin_guidance(
        noise_pred_uncond=UNCOND,
        noise_pred_text=TEXT,
        noise_pred_identity=IDENTITY,
        guidance_scale=0.0,
        alpha=0.25,
    )
    s = np.sqrt(0.25)
    np.testing.assert_allclose(result, s * UNCOND + (1.0 - s) * IDENTITY)


def test_steer_from_identity_ablation_subtracts_the_identity_prediction() -> None:
    alpha = 0.5
    result = guide(alpha, steer_from_identity=True)
    s = np.sqrt(alpha)
    expected = (
        s * UNCOND + (1.0 - s) * IDENTITY + alpha * SCALE * (TEXT - IDENTITY)
    )
    np.testing.assert_allclose(result, expected)


def test_continuity_across_the_sweep() -> None:
    # Adjacent alphas must not jump: a discontinuity in the combination rule
    # would show up as a hard cut between frames of the sweep.
    alphas = [i / 64.0 for i in range(65)]
    outputs = [guide(a) for a in alphas]
    steps = [np.abs(b - a).max() for a, b in zip(outputs, outputs[1:])]
    assert max(steps) < 1.0


# -- which forward passes are needed --------------------------------------


def test_endpoints_need_only_two_forward_passes() -> None:
    assert required_predictions(0.0, SCALE) == frozenset({"identity"})
    assert required_predictions(1.0, SCALE) == frozenset({"uncond", "text"})


def test_interior_needs_all_three() -> None:
    assert required_predictions(0.5, SCALE) == frozenset(
        {"uncond", "identity", "text"}
    )


def test_skipped_predictions_may_be_omitted() -> None:
    # What lets a pipeline skip a transformer call at the endpoints.
    at_zero = adaptive_origin_guidance(
        noise_pred_identity=IDENTITY, guidance_scale=SCALE, alpha=0.0
    )
    np.testing.assert_allclose(at_zero, IDENTITY)

    at_one = adaptive_origin_guidance(
        noise_pred_uncond=UNCOND,
        noise_pred_text=TEXT,
        guidance_scale=SCALE,
        alpha=1.0,
    )
    np.testing.assert_allclose(at_one, UNCOND + SCALE * (TEXT - UNCOND))


def test_omitting_a_needed_prediction_is_an_error() -> None:
    with pytest.raises(ValueError, match="noise_pred_identity is required"):
        adaptive_origin_guidance(
            noise_pred_uncond=UNCOND,
            noise_pred_text=TEXT,
            guidance_scale=SCALE,
            alpha=0.5,
        )


# -- input validation ------------------------------------------------------


@pytest.mark.parametrize("alpha", [-0.01, 1.01, 2.0, -1.0])
def test_alpha_outside_the_unit_interval_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
        guidance_coefficients(alpha, SCALE)


# -- backend independence --------------------------------------------------


def test_shape_and_dtype_are_preserved() -> None:
    a = np.zeros((2, 4, 8), dtype=np.float32)
    result = adaptive_origin_guidance(
        noise_pred_uncond=a,
        noise_pred_text=a,
        noise_pred_identity=a,
        guidance_scale=SCALE,
        alpha=0.5,
    )
    assert result.shape == a.shape
    assert result.dtype == a.dtype


def test_plain_floats_work_so_the_rule_can_be_checked_by_hand() -> None:
    # Coefficients stay Python floats and the maths is pure arithmetic, which is
    # also what keeps torch free to handle dtype and device promotion.
    result = adaptive_origin_guidance(
        noise_pred_uncond=1.0,
        noise_pred_text=2.0,
        noise_pred_identity=3.0,
        guidance_scale=2.0,
        alpha=0.25,
    )
    assert result == pytest.approx(0.5 * 1.0 + 0.5 * 3.0 + 0.25 * 2.0 * (2.0 - 1.0))
