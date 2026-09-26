"""Backbone adapters.

Adapter modules import ``torch`` and ``diffusers`` lazily, so
``import adaor`` works -- and the guidance maths stays testable -- in an
environment where neither is installed.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import AdaOrAdapter, AdaOrTransformer, adaor_guidance

__all__ = [
    "ADAPTERS",
    "AdaOrAdapter",
    "AdaOrTransformer",
    "adaor_guidance",
    "get_adapter_class",
    "load_pipeline",
]

#: Backbone key -> ``(module, adapter class name)``.
ADAPTERS: Dict[str, tuple] = {
    "qwen-image-edit": (".qwen_image_edit", "QwenImageEditAdapter"),
    "flux-klein": (".flux_klein", "FluxKleinAdapter"),
}


def _module(backbone: str):
    from importlib import import_module

    try:
        module_name, class_name = ADAPTERS[backbone]
    except KeyError:
        raise KeyError(
            f"unknown backbone {backbone!r}; available: {sorted(ADAPTERS)}"
        ) from None
    return import_module(module_name, __package__), class_name


def get_adapter_class(backbone: str) -> Type[Any]:
    """Return the adapter class for ``backbone``."""
    module, class_name = _module(backbone)
    return getattr(module, class_name)


def load_pipeline(backbone: str, **kwargs: Any) -> Any:
    """Load the pipeline for ``backbone``, forwarding keyword arguments."""
    module, _ = _module(backbone)
    return module.load_pipeline(**kwargs)
