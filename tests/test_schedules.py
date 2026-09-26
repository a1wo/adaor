"""Every schedule must pin both endpoints and rise monotonically in between.

These are the properties AdaOr's boundary conditions rest on, so they are
checked for all registered schedules rather than only the defaults.
"""

from __future__ import annotations

import pytest

from adaor.schedules import SCHEDULES, resolve_schedule


@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_endpoints_are_pinned(name: str) -> None:
    schedule = SCHEDULES[name]
    assert schedule(0.0) == pytest.approx(0.0)
    assert schedule(1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_monotonically_increasing(name: str) -> None:
    schedule = SCHEDULES[name]
    values = [schedule(i / 32.0) for i in range(33)]
    assert all(b >= a for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_stays_in_unit_interval(name: str) -> None:
    schedule = SCHEDULES[name]
    assert all(0.0 <= schedule(i / 32.0) <= 1.0 for i in range(33))


def test_sqrt_is_the_paper_default_and_front_loads_the_origin() -> None:
    # s(alpha) = sqrt(alpha) moves the origin off the identity prediction faster
    # than linear, which is what keeps low-alpha edits from stalling.
    assert SCHEDULES["sqrt"](0.25) == pytest.approx(0.5)
    assert SCHEDULES["sqrt"](0.25) > SCHEDULES["linear"](0.25)


def test_resolve_accepts_names_and_callables() -> None:
    assert resolve_schedule("linear")(0.3) == pytest.approx(0.3)
    assert resolve_schedule(lambda a: a**3)(0.5) == pytest.approx(0.125)


def test_resolve_rejects_unknown_name_and_wrong_type() -> None:
    with pytest.raises(KeyError, match="unknown schedule"):
        resolve_schedule("nope")
    with pytest.raises(TypeError):
        resolve_schedule(3)
