"""A single continuous sweep, end to end.

Run on a GPU with the `inference` extra installed:

    pip install -e ".[inference]"
    python examples/basic_edit.py path/to/photo.jpg "make the hair blonde"

Without a trained REC token this uses a plain "keep the image the same"
instruction as the identity condition, so it works with the off-the-shelf
checkpoint -- see README for what a trained token buys you.
"""

from __future__ import annotations

import sys
from pathlib import Path

from adaor import sweep_edit
from adaor.cli import save_strip

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)

    image_path, instruction = sys.argv[1], sys.argv[2]
    output_dir = Path("outputs/basic_edit")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sweep_edit(
        image_path,
        instruction,
        alphas=ALPHAS,
        guidance_scale=4.0,
        num_inference_steps=50,
        seed=655,
    )

    for image, alpha in zip(images, ALPHAS):
        image.save(output_dir / f"alpha_{alpha:.2f}.png")

    save_strip(images, output_dir / "strip.png", [f"a={a:g}" for a in ALPHAS])
    print(f"wrote {len(images)} frames and strip.png to {output_dir}")


if __name__ == "__main__":
    main()
