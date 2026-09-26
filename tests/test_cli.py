"""Argument parsing, checked without torch or diffusers installed.

``--help`` and argument validation must work on a laptop, so that a typo is
caught before renting a GPU. That only holds while the heavy imports stay inside
:func:`adaor.cli.main`.
"""

from __future__ import annotations

import argparse

import pytest

from adaor.cli import _parse_alphas, build_parser, main


def parse(*argv):
    return build_parser().parse_args(argv)


def test_defaults_match_the_paper() -> None:
    args = parse("in.png", "make it snow")
    assert args.alphas == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert args.origin_schedule == "sqrt"
    assert args.steer_schedule == "linear"
    assert args.backbone == "qwen-image-edit"
    assert args.steer_from_identity is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0,0.5,1", [0.0, 0.5, 1.0]),
        ("0 0.5 1", [0.0, 0.5, 1.0]),
        ("0, 0.25 , 1", [0.0, 0.25, 1.0]),
        ("0.5", [0.5]),
    ],
)
def test_alphas_accept_commas_and_spaces(text: str, expected: list) -> None:
    assert _parse_alphas(text) == expected


@pytest.mark.parametrize("text", ["-0.1", "1.5", "abc", "", "0,2"])
def test_invalid_alphas_are_rejected(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_alphas(text)


def test_unknown_backbone_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse("in.png", "edit", "-b", "not-a-backbone")


def test_missing_image_exits_before_loading_a_model(tmp_path, capsys) -> None:
    # Returns an error code rather than raising ImportError, which is what makes
    # this checkable without torch installed.
    code = main([str(tmp_path / "missing.png"), "make it snow"])
    assert code == 2
    assert "no such image" in capsys.readouterr().err
