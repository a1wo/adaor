"""``adaor-sweep`` -- run an edit-strength sweep from the command line.

.. code-block:: console

    adaor-sweep input.png "make the hair blonde" -o outputs/blonde

Writes one image per ``alpha`` plus a ``strip.png`` contact sheet, which is the
quickest way to see whether the transition is actually continuous.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

from .pipelines import ADAPTERS
from .schedules import SCHEDULES
from .sweep import DEFAULT_ALPHAS


def _parse_alphas(text: str) -> List[float]:
    """Parse ``"0,0.5,1"`` or ``"0 0.5 1"`` into a list of strengths."""
    values = [chunk for chunk in text.replace(",", " ").split() if chunk]
    try:
        alphas = [float(value) for value in values]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"could not parse alpha values from {text!r}"
        ) from None
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise argparse.ArgumentTypeError(
                f"alpha values must be in [0, 1], got {alpha}"
            )
    if not alphas:
        raise argparse.ArgumentTypeError("no alpha values given")
    return alphas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaor-sweep",
        description=(
            "Edit an image at several strengths using Adaptive-Origin Guidance."
        ),
    )
    parser.add_argument("image", type=Path, help="source image")
    parser.add_argument("instruction", help='edit instruction, e.g. "make it snow"')
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="where to write the sweep (default: outputs)",
    )
    parser.add_argument(
        "-a",
        "--alphas",
        type=_parse_alphas,
        default=list(DEFAULT_ALPHAS),
        help="edit strengths, comma- or space-separated "
        f"(default: {','.join(str(a) for a in DEFAULT_ALPHAS)})",
    )
    parser.add_argument(
        "-b",
        "--backbone",
        choices=sorted(ADAPTERS),
        default="qwen-image-edit",
        help="editing backbone (default: qwen-image-edit)",
    )
    parser.add_argument("--model-id", help="checkpoint id or path")
    parser.add_argument(
        "--lora-weights", help="REC LoRA checkpoint, if you trained one"
    )
    parser.add_argument(
        "--rec-instruction",
        help="identity instruction; defaults to a plain "
        '"keep the image the same" prompt',
    )
    parser.add_argument(
        "--trained-rec-token",
        action="store_true",
        help="the checkpoint was trained with a dedicated REC token",
    )
    parser.add_argument(
        "-w",
        "--guidance-scale",
        type=float,
        default=4.0,
        help="CFG scale w (default: 4.0)",
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=50, help="denoising steps (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=655,
        help="shared across the sweep so frames stay comparable (default: 655)",
    )
    parser.add_argument(
        "--origin-schedule",
        choices=sorted(SCHEDULES),
        default="sqrt",
        help="s(alpha); the paper's choice is sqrt (default: sqrt)",
    )
    parser.add_argument(
        "--steer-schedule",
        choices=sorted(SCHEDULES),
        default="linear",
        help="g(alpha); the paper's choice is linear (default: linear)",
    )
    parser.add_argument(
        "--steer-from-identity",
        action="store_true",
        help="ablation: steer away from the identity prediction instead of the "
        "null prediction. Unstable by design -- see the paper.",
    )
    parser.add_argument(
        "--no-strip",
        action="store_true",
        help="skip writing the strip.png contact sheet",
    )
    return parser


def save_strip(images: Sequence, path: Path, labels: Sequence[str] = ()) -> None:
    """Write the sweep side by side in one image, left to right."""
    from PIL import Image

    if not images:
        return

    height = max(image.height for image in images)
    widths = [round(image.width * height / image.height) for image in images]
    strip = Image.new("RGB", (sum(widths), height), "white")

    offset = 0
    for image, width in zip(images, widths):
        strip.paste(image.resize((width, height), Image.LANCZOS), (offset, 0))
        offset += width

    if labels:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(strip)
        offset = 0
        for label, width in zip(labels, widths):
            draw.text((offset + 8, 8), label, fill="white")
            draw.text((offset + 7, 7), label, fill="black")
            offset += width

    strip.save(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.image.is_file():
        print(f"error: no such image: {args.image}", file=sys.stderr)
        return 2

    # Imported here so `--help` works without torch or diffusers installed.
    from .sweep import sweep_edit

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"image:       {args.image}")
    print(f"instruction: {args.instruction}")
    print(f"backbone:    {args.backbone}")
    print(f"alphas:      {', '.join(str(a) for a in args.alphas)}")
    print(f"schedules:   origin={args.origin_schedule} steer={args.steer_schedule}")
    print(f"output:      {args.output_dir}")

    images = sweep_edit(
        args.image,
        args.instruction,
        alphas=args.alphas,
        backbone=args.backbone,
        model_id=args.model_id,
        lora_weights=args.lora_weights,
        rec_instruction=args.rec_instruction,
        trained_rec_token=args.trained_rec_token,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        seed=args.seed,
        origin_schedule=args.origin_schedule,
        steer_schedule=args.steer_schedule,
        steer_from_identity=args.steer_from_identity,
    )

    labels = [f"a={alpha:g}" for alpha in args.alphas]
    for image, alpha in zip(images, args.alphas):
        path = args.output_dir / f"alpha_{alpha:.2f}.png"
        image.save(path)
        print(f"wrote {path}")

    if not args.no_strip:
        strip_path = args.output_dir / "strip.png"
        save_strip(images, strip_path, labels)
        print(f"wrote {strip_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
