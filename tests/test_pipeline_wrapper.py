"""The transformer proxy, exercised with a fake pipeline.

This is the part with the most room for silent breakage: if the proxy stops
overriding the conditioning keyword arguments, or stops restoring the original
transformer, the pipeline keeps running and produces plausible-looking but wrong
images. So the tests assert on what the transformer was actually called with.

No ``diffusers`` import happens here -- every call uses ``return_dict=False``,
the form real pipelines use.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaor.pipelines.base import AdaOrTransformer, adaor_guidance


class FakeTransformer:
    """Returns a value keyed by the conditioning it was given."""

    def __init__(self) -> None:
        self.calls = []
        self.config = {"marker": "original"}

    def __call__(self, latents, *, encoder_hidden_states, return_dict=True, **kwargs):
        self.calls.append(
            {"encoder_hidden_states": encoder_hidden_states, **kwargs}
        )
        prediction = np.full_like(latents, float(encoder_hidden_states))
        return (prediction,) if not return_dict else prediction


class FakePipeline:
    def __init__(self) -> None:
        self.transformer = FakeTransformer()
        self._execution_device = "cpu"


class FakeAdapter:
    """Encodes a prompt as the float embedded in its name, e.g. "p:3" -> 3.0."""

    backbone = "fake"

    def __init__(self) -> None:
        self.disable_calls = 0

    def condition_kwargs(self, pipe, prompt):
        return {"encoder_hidden_states": float(prompt.split(":")[1])}

    def guidance_scale(self, call_kwargs):
        return float(call_kwargs.get("true_cfg_scale", 1.0))

    def disable_native_cfg(self, call_kwargs):
        self.disable_calls += 1
        call_kwargs["true_cfg_scale"] = 1.0
        return call_kwargs


CONDITIONS = {
    "uncond": {"encoder_hidden_states": 1.0},
    "identity": {"encoder_hidden_states": 2.0},
    "text": {"encoder_hidden_states": 4.0},
}


def make_wrapper(alpha: float, scale: float = 5.0, **kwargs) -> AdaOrTransformer:
    return AdaOrTransformer(
        FakeTransformer(),
        CONDITIONS,
        guidance_scale=scale,
        alpha=alpha,
        **kwargs,
    )


# -- combination -----------------------------------------------------------


def test_combines_the_three_predictions() -> None:
    wrapper = make_wrapper(0.25)
    latents = np.zeros((1, 3))
    (result,) = wrapper(latents, encoder_hidden_states=999.0, return_dict=False)

    s = np.sqrt(0.25)
    expected = s * 1.0 + (1 - s) * 2.0 + 0.25 * 5.0 * (4.0 - 1.0)
    np.testing.assert_allclose(result, np.full((1, 3), expected))


def test_pipeline_conditioning_is_overridden_not_merged() -> None:
    # The pipeline passes its own prompt embeddings; every one of them must be
    # replaced, or one of the three predictions would silently use the wrong
    # condition.
    wrapper = make_wrapper(0.5)
    wrapper(np.zeros((1, 2)), encoder_hidden_states=999.0, return_dict=False)

    used = [call["encoder_hidden_states"] for call in wrapper.wrapped.calls]
    assert 999.0 not in used
    assert sorted(used) == [1.0, 2.0, 4.0]


def test_unrelated_pipeline_kwargs_are_forwarded() -> None:
    wrapper = make_wrapper(0.5)
    wrapper(
        np.zeros((1, 2)),
        encoder_hidden_states=999.0,
        timestep=7,
        img_ids="ids",
        return_dict=False,
    )
    for call in wrapper.wrapped.calls:
        assert call["timestep"] == 7
        assert call["img_ids"] == "ids"


# -- forward-pass count ----------------------------------------------------


@pytest.mark.parametrize(
    ("alpha", "expected"), [(0.0, 1), (1.0, 2), (0.5, 3), (0.99, 3)]
)
def test_only_the_needed_forward_passes_run(alpha: float, expected: int) -> None:
    wrapper = make_wrapper(alpha)
    assert wrapper.forward_passes_per_step == expected
    wrapper(np.zeros((1, 2)), encoder_hidden_states=999.0, return_dict=False)
    assert len(wrapper.wrapped.calls) == expected


def test_alpha_zero_runs_the_identity_condition_only() -> None:
    wrapper = make_wrapper(0.0)
    wrapper(np.zeros((1, 2)), encoder_hidden_states=999.0, return_dict=False)
    assert [c["encoder_hidden_states"] for c in wrapper.wrapped.calls] == [2.0]


# -- proxying --------------------------------------------------------------


def test_delegates_unknown_attributes_to_the_wrapped_transformer() -> None:
    wrapper = make_wrapper(0.5)
    assert wrapper.config == {"marker": "original"}


def test_missing_conditions_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="missing conditions"):
        AdaOrTransformer(
            FakeTransformer(),
            {"uncond": {}, "text": {}},
            guidance_scale=5.0,
            alpha=0.5,
        )


# -- context manager -------------------------------------------------------


def test_transformer_is_swapped_in_and_restored() -> None:
    pipe = FakePipeline()
    original = pipe.transformer

    with adaor_guidance(
        pipe,
        FakeAdapter(),
        instruction="p:4",
        rec_instruction="p:2",
        negative_instruction="p:1",
        alpha=0.5,
        guidance_scale=5.0,
    ):
        assert isinstance(pipe.transformer, AdaOrTransformer)
        assert pipe.transformer.wrapped is original

    assert pipe.transformer is original


def test_transformer_is_restored_even_when_the_pipeline_raises() -> None:
    # Without this a failed run would leave the pipeline permanently wrapped,
    # and the next sweep would silently double-wrap it.
    pipe = FakePipeline()
    original = pipe.transformer

    with pytest.raises(RuntimeError, match="boom"):
        with adaor_guidance(
            pipe,
            FakeAdapter(),
            instruction="p:4",
            rec_instruction="p:2",
            negative_instruction="p:1",
            alpha=0.5,
            guidance_scale=5.0,
        ):
            raise RuntimeError("boom")

    assert pipe.transformer is original


def test_native_cfg_is_disabled_for_the_yielded_kwargs() -> None:
    pipe = FakePipeline()
    adapter = FakeAdapter()

    with adaor_guidance(
        pipe,
        adapter,
        instruction="p:4",
        rec_instruction="p:2",
        negative_instruction="p:1",
        alpha=0.5,
        guidance_scale=5.0,
        call_kwargs={"num_inference_steps": 8, "true_cfg_scale": 5.0},
    ) as call_kwargs:
        assert call_kwargs["true_cfg_scale"] == 1.0
        assert call_kwargs["num_inference_steps"] == 8

    assert adapter.disable_calls == 1
