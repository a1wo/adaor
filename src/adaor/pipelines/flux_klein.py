"""AdaOr adapter for FLUX.2 Klein image editing.

Two things about Klein shape this adapter.

It is guidance-distilled. Klein takes a ``guidance`` embedding inside the
transformer rather than running two forward passes, so out of the box there is
no classifier-free guidance for AdaOr to build on. The adaptive origin still
applies -- ``alpha = 0`` conditions on the identity instruction, ``alpha = 1`` on
the edit instruction, and the origin interpolates between the two predictions --
but the steering term needs real CFG with ``true_cfg_scale > 1``. A distilled
model run with real CFG is off-distribution, so expect to tune the scale down
from what an undistilled backbone wants. Set ``true_cfg_scale=1`` to get origin
interpolation alone, which is stable and already gives a continuous sweep.

Its conditioning signature is not stable across diffusers releases. FLUX
variants differ in how many tensors ``encode_prompt`` returns and what the
transformer calls them, so rather than hard-coding one spelling the adapter
inspects what it gets back and maps it. :meth:`FluxKleinAdapter.describe` prints
the mapping it inferred, which is the first thing to check if results look
wrong.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

__all__ = ["FluxKleinAdapter", "load_pipeline"]

#: Transformer keyword names for each tensor ``encode_prompt`` may return, in
#: the order FLUX pipelines return them.
_EMBED_KWARGS: Sequence[str] = (
    "encoder_hidden_states",
    "pooled_projections",
    "txt_ids",
)


class FluxKleinAdapter:
    """Glue between AdaOr and a FLUX.2 Klein editing pipeline.

    Args:
        image: the source image, kept for adapters that need it during prompt
            encoding. Unused when ``encode_prompt`` takes text only.
        num_images_per_prompt: must match the value passed to the pipeline.
    """

    backbone = "flux-klein"

    def __init__(self, image: Any = None, *, num_images_per_prompt: int = 1) -> None:
        self.image = image
        self.num_images_per_prompt = num_images_per_prompt
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._mapping: Dict[str, str] | None = None

    def condition_kwargs(self, pipe: Any, prompt: str) -> Dict[str, Any]:
        """Encode ``prompt``, inferring which transformer kwargs to fill."""
        if prompt in self._cache:
            return self._cache[prompt]

        outputs = pipe.encode_prompt(
            prompt=[prompt],
            prompt_2=[prompt],
            device=pipe._execution_device,
            num_images_per_prompt=self.num_images_per_prompt,
        )
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Drop trailing None entries so an optional pooled projection that this
        # variant does not produce is not passed as an explicit None.
        tensors = [tensor for tensor in outputs if tensor is not None]
        if len(tensors) > len(_EMBED_KWARGS):
            raise RuntimeError(
                f"{type(pipe).__name__}.encode_prompt returned {len(tensors)} "
                f"tensors; this adapter knows how to place at most "
                f"{len(_EMBED_KWARGS)} ({', '.join(_EMBED_KWARGS)}). The "
                "pipeline's conditioning signature has changed -- map the "
                "extra tensors by subclassing FluxKleinAdapter."
            )

        kwargs = dict(zip(_EMBED_KWARGS, tensors))
        self._mapping = {
            name: tuple(tensor.shape) for name, tensor in kwargs.items()
        }
        self._cache[prompt] = kwargs
        return kwargs

    def describe(self) -> Dict[str, str]:
        """The inferred kwarg-to-shape mapping, for sanity-checking a new release.

        Empty until :meth:`condition_kwargs` has run at least once.
        """
        return dict(self._mapping or {})

    def guidance_scale(self, call_kwargs: Mapping[str, Any]) -> float:
        """Read the real CFG scale; ``1.0`` means origin interpolation only."""
        return float(call_kwargs.get("true_cfg_scale", 1.0))

    def disable_native_cfg(self, call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Neutralise the pipeline's own CFG, keeping distilled guidance intact.

        ``guidance_scale`` is left alone: on a distilled model it is an input
        embedding, not a CFG weight, and zeroing it would change the model's
        operating point rather than disabling guidance.
        """
        call_kwargs["true_cfg_scale"] = 1.0
        call_kwargs.setdefault("prompt", "")
        call_kwargs.pop("negative_prompt", None)
        return call_kwargs


def load_pipeline(
    model_id: str = "black-forest-labs/FLUX.2-Klein",
    *,
    torch_dtype: Any = None,
    device: str | None = None,
) -> Any:
    """Load a FLUX.2 Klein editing pipeline.

    Resolved through :func:`diffusers.AutoPipelineForImageToImage` so the
    concrete class name does not have to be pinned here; ``model_id`` decides
    which FLUX.2 checkpoint is loaded.
    """
    import torch
    from diffusers import AutoPipelineForImageToImage

    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = AutoPipelineForImageToImage.from_pretrained(
        model_id, torch_dtype=torch_dtype
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe
