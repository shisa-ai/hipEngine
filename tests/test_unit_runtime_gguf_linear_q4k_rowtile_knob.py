"""The Q4_K rowtile decode route is default-on with a scoped comparison control.

Ambient legacy values must not disable the route. Dispatch coverage also keeps
explicit and scoped comparison precedence separate from the coarse resolver.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
from tests._gguf_device_weight_fixtures import q4_k_t16_weight as _sibling_weight

REPO = pathlib.Path(__file__).resolve().parents[1]
LINEAR = "hipengine.runtime.gguf_linear"


def _rowtile_resolver():
    module = sys.modules.get(LINEAR) or __import__(LINEAR, fromlist=["*"])
    return module


@pytest.mark.parametrize("value", [None, "", "1", "0", "false", "off", "invalid"])
def test_retired_env_does_not_disable_rowtile(monkeypatch, value):
    module = _rowtile_resolver()
    if value is None:
        monkeypatch.delenv("HIPENGINE_GGUF_Q4K_ROWTILE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", value)
    with module.q4k_rowtile_session(None):
        assert module._resolve_use_q4k_rowtile(None) is True


@pytest.mark.parametrize("rows", [1, 2, 5, 8, 9])
def test_retired_env_preserves_actual_raw_rowtile_launch(monkeypatch, rows):
    from hipengine.kernels.registry import KernelKey
    from tests.test_unit_gguf_gemv_decode_dispatch import _capture_launch

    module = _rowtile_resolver()
    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", "0")
    rowtile = KernelKey(
        "hip_gfx1100", "linear", "gguf_q4_k", "rowtile_bf16_bf16_out",
    )
    parents = tuple(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", variant)
        for variant in ("gemv_bf16_bf16_out", "prefill_bf16_bf16_out")
    )
    with module.q4k_rowtile_session(None), module.wmma_prefill_session(False):
        key, _args, _kwargs = _capture_launch(
            rows=rows, quant_key="gguf_q4_k", use_gemv_decode=False,
            extra_keys=(rowtile, *parents),
        )

    assert key is not None
    assert (key == rowtile) is (2 <= rows <= 8)


def test_dense_rows_above_one_resolver_is_independent_of_retired_env(monkeypatch):
    """Coarse T16 resolution does not establish the rowtile launch choice."""
    module = _rowtile_resolver()
    weight = _sibling_weight(0x2000)

    monkeypatch.delenv("HIPENGINE_GGUF_Q4K_ROWTILE", raising=False)
    on = module.resolve_gguf_linear_dispatch(weight, rows=2)
    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", "0")
    off = module.resolve_gguf_linear_dispatch(weight, rows=2)
    assert on.key == off.key
    assert on.key.variant == "t16_wmma_prefill_bf16_bf16_out"


def test_scoped_control_is_consumed_by_the_launch_sites_not_the_resolver():
    """Pin the three launch sites that consume the scoped comparison control."""
    module = _rowtile_resolver()
    source = pathlib.Path(module.__file__).read_text()
    sites = [
        line.strip()
        for line in source.splitlines()
        if "_resolve_use_q4k_rowtile(" in line and not line.lstrip().startswith("def ")
    ]
    assert len(sites) == 3, f"expected 3 consumers, found {sites}"
    assert sum("use_rowtile=" in s for s in sites) == 1
    assert sum("f_rowtile" in s for s in sites) == 1


def test_rows_1_route_is_outside_the_rowtile_domain(monkeypatch):
    # rows==1 has its own single-row decode family; the rowtile chunker cannot serve it at all.
    module = _rowtile_resolver()
    weight = _sibling_weight(0x2001)

    monkeypatch.delenv("HIPENGINE_GGUF_Q4K_ROWTILE", raising=False)
    on = module.resolve_gguf_linear_dispatch(weight, rows=1)
    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", "0")
    off = module.resolve_gguf_linear_dispatch(weight, rows=1)
    assert on.key == off.key
    assert "rowtile" not in on.key.variant


def test_coarse_resolver_above_the_rowtile_cap(monkeypatch):
    module = _rowtile_resolver()
    weight = _sibling_weight(0x2002)

    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", "1")
    above = module.resolve_gguf_linear_dispatch(weight, rows=module._ROWTILE_MAX_ROWS + 1)
    assert "rowtile" not in above.key.variant
    assert module._ROWTILE_MIN_ROWS == 2 and module._ROWTILE_MAX_ROWS == 8


def test_scoped_comparison_restores_default_after_exception(monkeypatch):
    module = _rowtile_resolver()
    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", "0")
    with module.q4k_rowtile_session(None):
        assert module._resolve_use_q4k_rowtile(None) is True
        with pytest.raises(RuntimeError, match="comparison failure"):
            with module.q4k_rowtile_session(False):
                assert module._resolve_use_q4k_rowtile(None) is False
                with module.q4k_rowtile_session(True):
                    assert module._resolve_use_q4k_rowtile(None) is True
                assert module._resolve_use_q4k_rowtile(None) is False
                raise RuntimeError("comparison failure")
        assert module._resolve_use_q4k_rowtile(None) is True


def test_explicit_kwarg_beats_the_session(monkeypatch):
    module = _rowtile_resolver()
    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_ROWTILE", "1")
    with module.q4k_rowtile_session(True):
        assert module._resolve_use_q4k_rowtile(False) is False
