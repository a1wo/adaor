"""High-level API: one edit, or a sweep over edit strengths.

The sweep is the point of the method, so it gets the ergonomic entry point.
Because a sweep reuses one loaded pipeline and one set of encoded conditions
across every ``alpha``, running the sweep is much cheaper than calling
:func:`edit` in a loop.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .identity import resolve_rec_instruction
from .pipelines import adaor_guidance, get_adapter_class, load_pipeline
from .schedules import ScheduleLike

__all__ = ["DEFAULT_ALPHAS", "edit", "sweep_edit"]

#: Edit strengths used in the paper's figures.
DEFAULT_ALPHAS: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _load_image(image: Any) -> Any:
    """Accept a path, a PIL image, or anything PIL can open."""
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert("RGB")
    return image


def sweep_edit(
    image: Any,
    instruction: str,
    *,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
    backbone: str = "qwen-image-edit",
    pipe: Any = None,
    model_id: str | None = None,
    lora_weights: str | None = None,
    rec_instruction: str | None = None,
    trained_rec_token: bool = False,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 50,
    seed: int = 655,
    origin_schedule: ScheduleLike = "sqrt",
    steer_schedule: ScheduleLike = "linear",
    steer_from_identity: bool = False,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Edit one image at several strengths, returning one image per ``alpha``.

    Args:
        image: source image, or a path to one.
        instruction: the edit instruction, e.g. ``"make the hair blonde"``.
        alphas: edit strengths in ``[0, 1]``. ``0`` reconstructs the input, ``1``
            reproduces the backbone's default edit.
        backbone: ``"qwen-image-edit"`` or ``"flux-klein"``.
        pipe: an already-loaded pipeline; loaded from ``backbone`` if omitted.
        model_id: checkpoint to load when ``pipe`` is not given.
        lora_weights: REC LoRA, for backbones that support one.
        rec_instruction: identity instruction. Defaults to a plain "keep the
            image the same" prompt, or to the backbone's trained token when
            ``trained_rec_token`` is set.
        trained_rec_token: the checkpoint was trained with a dedicated ``REC``
            token. Leave ``False`` for an off-the-shelf checkpoint.
        guidance_scale: the CFG scale ``w``.
        num_inference_steps: denoising steps.
        seed: every ``alpha`` is generated from this same seed, which is what
            makes the sweep comparable -- a varying seed would change content
            between frames and hide whether the transition is smooth.
        origin_schedule: ``s(alpha)``; the paper's ``"sqrt"`` by default.
        steer_schedule: ``g(alpha)``; the paper's ``"linear"`` by default.
        steer_from_identity: reproduce the paper's unstable ablation.
        pipeline_kwargs: extra keyword arguments forwarded to the pipeline.

    Returns:
        One image per requested ``alpha``, in the order given.
    """
    import torch

    source = _load_image(image)

    if pipe is None:
        load_kwargs: Dict[str, Any] = {}
        if model_id is not None:
            load_kwargs["model_id"] = model_id
        if lora_weights is not None:
            load_kwargs["lora_weights"] = lora_weights
        pipe = load_pipeline(backbone, **load_kwargs)

    resolved_rec = resolve_rec_instruction(
        backbone,
        rec_instruction=rec_instruction,
        trained_rec_token=trained_rec_token,
    )

    adapter_class = get_adapter_class(backbone)
    adapter = adapter_class(image=source)

    base_kwargs: Dict[str, Any] = {
        "image": source,
        "num_inference_steps": num_inference_steps,
        **(pipeline_kwargs or {}),
    }

    results: List[Any] = []
    for alpha in alphas:
        # A fresh generator per alpha, seeded identically, so the initial noise
        # is shared across the sweep.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with adaor_guidance(
            pipe,
            adapter,
            instruction=instruction,
            rec_instruction=resolved_rec,
            alpha=float(alpha),
            guidance_scale=guidance_scale,
            origin_schedule=origin_schedule,
            steer_schedule=steer_schedule,
            steer_from_identity=steer_from_identity,
            call_kwargs=dict(base_kwargs),
        ) as call_kwargs:
            output = pipe(generator=generator, **call_kwargs)
        results.append(output.images[0])

    return results


def edit(image: Any, instruction: str, *, alpha: float = 1.0, **kwargs: Any) -> Any:
    """Edit one image at a single strength.

    Thin wrapper over :func:`sweep_edit`; see it for the full argument list. To
    produce several strengths, call :func:`sweep_edit` directly rather than this
    in a loop -- it reuses the loaded pipeline and the encoded conditions.
    """
    return sweep_edit(image, instruction, alphas=[alpha], **kwargs)[0]
