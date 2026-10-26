#!/usr/bin/env python3
"""Shared producer-mirror control for the Qwen4Exp DP-2 pair harnesses.

The dense-wide F16 pair only stages an activation when nothing else already
serves the second consumer. The GR read publishes an F16 mirror of the mixed
activation by default and the pair declines when it finds one, so a harness that
compares the staged route against the parent without rolling that mirror back
measures the parent route on both arms. Every live A/B arm therefore runs with
``HIPENGINE_QWEN4_EXP_GR_F16_MIRROR=0``; the harness records the rollback and
puts back whatever the caller had set.
"""
from __future__ import annotations

import contextlib
import os
from typing import Iterator

GR_F16_MIRROR_ENV = "HIPENGINE_QWEN4_EXP_GR_F16_MIRROR"


def rollback_producer_mirror() -> str | None:
    """Roll the producer mirror back and return the value it replaced."""

    prior = os.environ.get(GR_F16_MIRROR_ENV)
    os.environ[GR_F16_MIRROR_ENV] = "0"
    return prior


def restore_producer_mirror(prior: str | None) -> None:
    """Put the caller's value back, or unset the variable when it was unset."""

    if prior is None:
        os.environ.pop(GR_F16_MIRROR_ENV, None)
    else:
        os.environ[GR_F16_MIRROR_ENV] = prior


@contextlib.contextmanager
def pair_staging_without_producer_mirror() -> Iterator[str | None]:
    """Run the enclosed arms with the producer mirror rolled back."""

    prior = rollback_producer_mirror()
    try:
        yield prior
    finally:
        restore_producer_mirror(prior)


def producer_mirror_record(prior: str | None) -> dict[str, object]:
    """Report fragment naming the rollback and what it replaced."""

    return dict(
        env=GR_F16_MIRROR_ENV,
        value="0",
        prior=prior,
        reason="the GR read's F16 mirror serves both consumers, so the pair "
               "would stage no conversion and both arms would measure the "
               "parent route; the mirror is rolled back for every arm",
    )
