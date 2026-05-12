"""Quant plugin protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class QuantPlugin(Protocol):
    name: str
    weight_storage: str
    activation_preprocess: str
    compute_dtype: str
    scale_granularity: str
    calibration_artifact: str
    kernel_family: str
