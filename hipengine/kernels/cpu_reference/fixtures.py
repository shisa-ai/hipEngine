"""JSON layer-fixture format for CPU-reference checks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.kernels.registry import resolve

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Tolerances:
    atol: float = 1e-6
    rtol: float = 1e-6


@dataclass(frozen=True)
class LayerFixture:
    name: str
    layer: str
    quant: str
    inputs: Mapping[str, Any]
    expected: np.ndarray
    backend: str = "cpu_reference"
    tolerances: Tolerances = field(default_factory=Tolerances)
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerCheckResult:
    name: str
    layer: str
    quant: str
    passed: bool
    max_abs: float
    max_rel: float


def load_fixture(path: str | Path) -> LayerFixture:
    data = json.loads(Path(path).read_text())
    schema = data.get("schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported fixture schema {schema!r}; expected {SCHEMA_VERSION}")
    tolerances = Tolerances(**data.get("tolerances", {}))
    inputs = {name: _decode_value(value) for name, value in data["inputs"].items()}
    expected = _decode_array(data["expected"])
    return LayerFixture(
        name=data["name"],
        layer=data["layer"],
        quant=data.get("quant", "fp16"),
        backend=data.get("backend", "cpu_reference"),
        inputs=inputs,
        expected=expected,
        tolerances=tolerances,
        metadata=data.get("metadata", {}),
    )


def save_fixture(path: str | Path, fixture: LayerFixture) -> None:
    payload = {
        "schema": SCHEMA_VERSION,
        "name": fixture.name,
        "backend": fixture.backend,
        "layer": fixture.layer,
        "quant": fixture.quant,
        "inputs": {name: _encode_value(value) for name, value in fixture.inputs.items()},
        "expected": _encode_array(fixture.expected),
        "tolerances": {"atol": fixture.tolerances.atol, "rtol": fixture.tolerances.rtol},
        "metadata": dict(fixture.metadata),
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_fixture(fixture: LayerFixture) -> LayerCheckResult:
    kernel = resolve(
        backend=fixture.backend,
        layer=fixture.layer,
        quant=fixture.quant,
    )
    actual = np.asarray(kernel(**fixture.inputs), dtype=np.float32)
    expected = np.asarray(fixture.expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"fixture {fixture.name!r} shape mismatch: "
            f"actual {actual.shape}, expected {expected.shape}"
        )
    diff = np.abs(actual - expected)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    denom = np.maximum(np.abs(expected), fixture.tolerances.atol)
    max_rel = float(np.max(diff / denom)) if diff.size else 0.0
    passed = bool(
        max_abs <= fixture.tolerances.atol
        or np.allclose(actual, expected, atol=fixture.tolerances.atol, rtol=fixture.tolerances.rtol)
    )
    return LayerCheckResult(
        name=fixture.name,
        layer=fixture.layer,
        quant=fixture.quant,
        passed=passed,
        max_abs=max_abs,
        max_rel=max_rel,
    )


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict) and "data" in value:
        return _decode_array(value)
    return value


def _decode_array(value: Mapping[str, Any]) -> np.ndarray:
    dtype = value.get("dtype", "float32")
    return np.asarray(value["data"], dtype=np.dtype(dtype))


def _encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _encode_array(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _encode_array(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value)
    return {"dtype": str(arr.dtype), "data": arr.tolist()}
