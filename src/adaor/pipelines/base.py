"""Attaching AdaOr to a diffusers pipeline without forking it.

The problem
-----------
A diffusers editing pipeline computes CFG inside its own denoising loop,
roughly::

    noise_pred = transformer(latents, encoder_hidden_states=prompt_embeds, ...)
    noise_uncond = transformer(latents, encoder_hidden_states=negative_embeds, ...)
    noise_pred = noise_uncond + scale * (noise_pred - noise_uncond)

AdaOr needs a third prediction and a different combination. Subclassing the
pipeline means copying its whole ``__call__`` -- hundreds of lines that then rot
against upstream. Patching the loop in place is worse.

The approach
------------
Wrap the *transformer* instead, and neutralise the pipeline's own CFG by setting
its guidance scale to 1. The pipeline then makes a single transformer call per
step and uses the result directly as the noise prediction. Our wrapper turns
that one call into up to three -- null, identity and instruction -- and returns
the AdaOr combination.

The pipeline is untouched, so it keeps its scheduler, its latent packing and its
image preprocessing, and the integration surface shrinks to one thing: which
keyword arguments carry the text conditioning. That is what an
:class:`AdaOrAdapter` supplies.

The wrapper is installed by :func:`adaor_guidance`, a context manager that
restores the original transformer on exit, including when the pipeline raises.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Mapping, Optional, Protocol

from ..guidance import adaptive_origin_guidance, required_predictions
from ..schedules import ScheduleLike

__all__ = ["AdaOrAdapter", "AdaOrTransformer", "adaor_guidance"]

#: Names of the three conditions, in the order predictions are computed.
CONDITIONS = ("uncond", "identity", "text")


class AdaOrAdapter(Protocol):
    """Backbone-specific glue.

    An adapter knows two things about its pipeline: how to encode a prompt into
    the keyword arguments the transformer expects, and how to disable the
    pipeline's built-in CFG.
    """

    #: Registry key, also used to look up default REC instructions.
    backbone: str

    def condition_kwargs(self, pipe: Any, prompt: str) -> Dict[str, Any]:
        """Encode ``prompt`` into transformer keyword arguments."""
        ...

    def disable_native_cfg(self, call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``call_kwargs`` with the pipeline's own CFG neutralised."""
        ...

    def guidance_scale(self, call_kwargs: Mapping[str, Any]) -> float:
        """Read the CFG scale ``w`` the caller asked for."""
        ...


class AdaOrTransformer:
    """Transformer proxy that returns an adaptive-origin noise prediction.

    Delegates every attribute except the forward call to the wrapped
    transformer, so the pipeline cannot tell the difference: it still sees
    ``.config``, ``.dtype``, ``.device`` and anything else it reaches for.

    Args:
        transformer: the pipeline's transformer.
        conditions: ``{condition_name: transformer_kwargs}`` for each of
            ``uncond``, ``identity`` and ``text``.
        guidance_scale: the CFG scale ``w``.
        alpha: edit strength in ``[0, 1]``.
        origin_schedule: ``s(alpha)``.
        steer_schedule: ``g(alpha)``.
        steer_from_identity: reproduce the paper's unstable ablation.
    """

    def __init__(
        self,
        transformer: Any,
        conditions: Mapping[str, Mapping[str, Any]],
        *,
        guidance_scale: float,
        alpha: float,
        origin_schedule: ScheduleLike = "sqrt",
        steer_schedule: ScheduleLike = "linear",
        steer_from_identity: bool = False,
    ) -> None:
        missing = set(CONDITIONS) - set(conditions)
        if missing:
            raise ValueError(f"missing conditions: {sorted(missing)}")

        self._transformer = transformer
        self._conditions = {name: dict(conditions[name]) for name in CONDITIONS}
        self._guidance_scale = float(guidance_scale)
        self._alpha = float(alpha)
        self._origin_schedule = origin_schedule
        self._steer_schedule = steer_schedule
        self._steer_from_identity = steer_from_identity

        # Which forward passes actually change the result. With the default
        # schedules this is two passes instead of three at alpha = 0 and 1.
        self._needed = required_predictions(
            self._alpha,
            self._guidance_scale,
            origin_schedule=origin_schedule,
            steer_schedule=steer_schedule,
            steer_from_identity=steer_from_identity,
        )

    # -- introspection -----------------------------------------------------

    @property
    def wrapped(self) -> Any:
        """The transformer this proxy stands in for."""
        return self._transformer

    @property
    def forward_passes_per_step(self) -> int:
        """How many transformer calls each denoising step costs."""
        return len(self._needed)

    def __getattr__(self, name: str) -> Any:
        # Reached only for attributes not found on the proxy itself, so the
        # private attributes set in __init__ never recurse through here.
        return getattr(self._transformer, name)

    # -- forward -----------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return_dict = kwargs.pop("return_dict", True)

        predictions = {}
        for name in CONDITIONS:
            if name not in self._needed:
                continue
            condition_kwargs = {**kwargs, **self._conditions[name]}
            output = self._transformer(*args, return_dict=False, **condition_kwargs)
            predictions[name] = output[0] if isinstance(output, tuple) else output

        guided = adaptive_origin_guidance(
            noise_pred_uncond=predictions.get("uncond"),
            noise_pred_text=predictions.get("text"),
            noise_pred_identity=predictions.get("identity"),
            guidance_scale=self._guidance_scale,
            alpha=self._alpha,
            origin_schedule=self._origin_schedule,
            steer_schedule=self._steer_schedule,
            steer_from_identity=self._steer_from_identity,
        )

        if not return_dict:
            return (guided,)

        # Mirror whatever output container the wrapped transformer uses, so a
        # pipeline that reads `.sample` keeps working.
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample=guided)


@contextlib.contextmanager
def adaor_guidance(
    pipe: Any,
    adapter: AdaOrAdapter,
    *,
    instruction: str,
    rec_instruction: str,
    alpha: float,
    guidance_scale: float,
    negative_instruction: str = "",
    origin_schedule: ScheduleLike = "sqrt",
    steer_schedule: ScheduleLike = "linear",
    steer_from_identity: bool = False,
    call_kwargs: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Run a pipeline with adaptive-origin guidance.

    Yields the keyword arguments to call ``pipe`` with -- the pipeline's own CFG
    already neutralised, since the wrapper performs guidance instead. Prompt
    arguments are left out deliberately: the adapter has encoded them, and
    passing a prompt as well would make the pipeline encode it a second time.

    .. code-block:: python

        with adaor_guidance(pipe, adapter, instruction="make the hair blonde",
                            rec_instruction="keep the image exactly the same",
                            alpha=0.4, guidance_scale=5.0) as kwargs:
            image = pipe(image=source, **kwargs).images[0]

    The transformer is restored on exit, including on error, so the same
    pipeline object can be reused across a sweep without reloading weights.
    """
    conditions = {
        "text": adapter.condition_kwargs(pipe, instruction),
        "uncond": adapter.condition_kwargs(pipe, negative_instruction),
        "identity": adapter.condition_kwargs(pipe, rec_instruction),
    }

    wrapper = AdaOrTransformer(
        pipe.transformer,
        conditions,
        guidance_scale=guidance_scale,
        alpha=alpha,
        origin_schedule=origin_schedule,
        steer_schedule=steer_schedule,
        steer_from_identity=steer_from_identity,
    )

    original = pipe.transformer
    pipe.transformer = wrapper
    try:
        yield adapter.disable_native_cfg(dict(call_kwargs or {}))
    finally:
        pipe.transformer = original
