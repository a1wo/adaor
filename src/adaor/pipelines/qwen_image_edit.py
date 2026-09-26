"""AdaOr adapter for ``QwenImageEditPipeline``.

Qwen-Image-Edit conditions its text encoder on the source image, so prompt
encoding needs that image -- which is why the adapter is constructed per edit
rather than being a module-level singleton.

The pipeline exposes two separate scales. ``guidance_scale`` feeds a distilled
guidance embedding inside the transformer and is not classifier-free guidance;
``true_cfg_scale`` is the real thing, and it is the ``w`` AdaOr builds on. The
adapter pins ``true_cfg_scale`` to 1 so the pipeline stops doing its own CFG,
and hands the requested value to the wrapper instead.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

__all__ = ["QwenImageEditAdapter", "load_pipeline"]


class QwenImageEditAdapter:
    """Glue between AdaOr and ``QwenImageEditPipeline``.

    Args:
        image: the source image being edited. Passed to ``encode_prompt``
            because Qwen's text encoder is image-conditioned.
        num_images_per_prompt: must match the value passed to the pipeline, or
            the conditioning batch will not line up with the latents.
    """

    backbone = "qwen-image-edit"

    def __init__(self, image: Any, *, num_images_per_prompt: int = 1) -> None:
        self.image = image
        self.num_images_per_prompt = num_images_per_prompt
        self._cache: Dict[str, Dict[str, Any]] = {}

    def condition_kwargs(self, pipe: Any, prompt: str) -> Dict[str, Any]:
        """Encode ``prompt`` into transformer keyword arguments.

        Results are cached per prompt string: the three conditions are encoded
        once and reused across every denoising step.
        """
        if prompt in self._cache:
            return self._cache[prompt]

        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            image=self.image,
            prompt=[prompt],
            device=pipe._execution_device,
            num_images_per_prompt=self.num_images_per_prompt,
        )
        kwargs = {
            "encoder_hidden_states": prompt_embeds,
            "encoder_hidden_states_mask": prompt_embeds_mask,
        }
        self._cache[prompt] = kwargs
        return kwargs

    def guidance_scale(self, call_kwargs: Mapping[str, Any]) -> float:
        """Read ``true_cfg_scale``, the real CFG scale."""
        return float(call_kwargs.get("true_cfg_scale", 4.0))

    def disable_native_cfg(self, call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Neutralise the pipeline's own CFG and its prompt encoding.

        ``true_cfg_scale=1`` makes the pipeline take the transformer output as
        the noise prediction directly. ``prompt=""`` still has to be present
        because the pipeline validates it, but the wrapper overrides the
        conditioning keyword arguments on every call, so nothing the pipeline
        encodes from it reaches the transformer.
        """
        call_kwargs["true_cfg_scale"] = 1.0
        call_kwargs.setdefault("prompt", "")
        call_kwargs.pop("negative_prompt", None)
        return call_kwargs


def load_pipeline(
    model_id: str = "Qwen/Qwen-Image-Edit",
    *,
    lora_weights: str | None = None,
    torch_dtype: Any = None,
    device: str | None = None,
) -> Any:
    """Load ``QwenImageEditPipeline``, optionally with REC LoRA weights.

    Args:
        model_id: Hub id or local path.
        lora_weights: path to a LoRA checkpoint trained with a ``REC`` token.
            Without it, use a prompt-only identity instruction -- see
            :mod:`adaor.identity`.
        torch_dtype: defaults to ``bfloat16``.
        device: defaults to ``cuda`` when available, otherwise ``cpu``.
    """
    import torch
    from diffusers import QwenImageEditPipeline

    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = QwenImageEditPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    if lora_weights is not None:
        pipe.load_lora_weights(lora_weights)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe
