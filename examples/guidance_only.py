"""Inspect the combination rule without loading a model.

    python examples/guidance_only.py

Prints the weight each noise prediction receives across the sweep, and how many
transformer forward passes each step costs. Useful for checking a schedule
before spending GPU time on it.
"""

from __future__ import annotations

from adaor import guidance_coefficients, required_predictions

SCALE = 4.0
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def report(origin_schedule: str, steer_schedule: str) -> None:
    print(f"\norigin=s({origin_schedule})  steer=g({steer_schedule})  w={SCALE}")
    print(f"{'alpha':>6} {'null':>9} {'identity':>9} {'text':>9} {'passes':>7}")
    for alpha in ALPHAS:
        c = guidance_coefficients(
            alpha,
            SCALE,
            origin_schedule=origin_schedule,
            steer_schedule=steer_schedule,
        )
        passes = len(
            required_predictions(
                alpha,
                SCALE,
                origin_schedule=origin_schedule,
                steer_schedule=steer_schedule,
            )
        )
        print(
            f"{alpha:>6.2f} {c.uncond_weight:>9.3f} "
            f"{c.identity_weight:>9.3f} {c.text_weight:>9.3f} {passes:>7}"
        )


if __name__ == "__main__":
    report("sqrt", "linear")  # the paper
    report("linear", "sqrt")  # the variant described in correspondence
    print("\nWeights sum to 1 at every alpha, so the prediction's scale is preserved.")
