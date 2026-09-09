"""AOTriton admission for the slot-local transient-BF16-oracle prefill route.

The ``int8_direct`` KV route forces slot-local full-attention prefill and, since
the compact-serial-c4 qualification, hard-disabled AOTriton for every layer that
owns a transient BF16 oracle. That left the route on the native split-K paged
kernel even though the declared strict arithmetic for the route is the oracle
pair read *through AOTriton* (see
``scripts/execution_profile_gguf_int8_direct_prefill_gate.py``).

These tests pin the admission decision itself. It is pure host policy, so no HIP
runtime or model artifact is required.
"""

from __future__ import annotations

import pytest

from hipengine.runtime.qwen35_gguf_runner import (
    _GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON_ENV,
    _gguf_int8_prefill_slot_local_aotriton_enabled,
    _gguf_slot_local_prefill_allow_aotriton,
)


def test_layers_without_a_transient_oracle_always_admit_aotriton(monkeypatch):
    """BF16 and mirrored-INT8 layers keep the engine-wide AOTriton default."""

    monkeypatch.delenv(_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON_ENV, raising=False)
    assert _gguf_slot_local_prefill_allow_aotriton(transient_direct_oracle=False) is True


def test_transient_oracle_layers_default_to_the_native_kernel(monkeypatch):
    """Default stays on the retained pre-flag behavior so rollback is the default."""

    monkeypatch.delenv(_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON_ENV, raising=False)
    assert _gguf_int8_prefill_slot_local_aotriton_enabled() is False
    assert _gguf_slot_local_prefill_allow_aotriton(transient_direct_oracle=True) is False


@pytest.mark.parametrize("raw", ["1", "true", "on", "yes"])
def test_opt_in_admits_aotriton_on_transient_oracle_layers(monkeypatch, raw):
    monkeypatch.setenv(_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON_ENV, raw)
    assert _gguf_int8_prefill_slot_local_aotriton_enabled() is True
    assert _gguf_slot_local_prefill_allow_aotriton(transient_direct_oracle=True) is True


@pytest.mark.parametrize("raw", ["0", "false", "off", "no"])
def test_explicit_rollback_keeps_the_native_kernel(monkeypatch, raw):
    monkeypatch.setenv(_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON_ENV, raw)
    assert _gguf_slot_local_prefill_allow_aotriton(transient_direct_oracle=True) is False


def test_opt_in_never_suppresses_a_non_oracle_layer(monkeypatch):
    """The flag only ever widens admission; it must not gate the default path."""

    monkeypatch.setenv(_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON_ENV, "0")
    assert _gguf_slot_local_prefill_allow_aotriton(transient_direct_oracle=False) is True
