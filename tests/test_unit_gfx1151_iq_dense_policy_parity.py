"""Dense raw-IQ owner-policy parity between the gfx1100 and gfx1151 packages.

``register_gfx1151_kernels`` aliases the whole gfx1100 kernel key space into
``hip_gfx1151``, so both backends resolve the *same* dense-IQ owner objects -
the local32 decode GEMV, its verifier sibling, and the W4A16 prefill owner.
The routing tables are not aliased: each backend declares its own policy
constants, and gfx1151 declared only ``GGUF_IQ_DENSE_PREFILL_POLICY``.

That left the decode and verifier row regimes undeclared, so every dense raw-IQ
projection on gfx1151 kept the strict per-row GEMV, and it left both per-slot
quality pin tables absent, so the one slot gfx1100 pins back to strict prefill
for UD-Q4_K_M ran the route that breached the calibrated ceiling there.

Because the owner objects are literally the same kernels, the pins are a
property of that arithmetic and transfer by construction. These tests hold the
two packages together: the value tables must agree, every variant the policy
names must be registered on gfx1151, and the pins must actually keep the strict
owner at dispatch time.
"""

from __future__ import annotations

import pathlib

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime import gguf_linear as gl

# Both packages must be imported at module scope, not inside a test.
# ``tests/conftest.py`` snapshots the kernel registry in
# ``pytest_collection_finish`` and restores it at every test teardown, so a
# backend first imported *during* a test has its aliases discarded before the
# next test runs. Importing here puts the gfx1151 aliases in the baseline.
import hipengine.kernels.hip_gfx1100  # noqa: F401,E402
import hipengine.kernels.hip_gfx1151  # noqa: F401,E402

SOURCE = "hip_gfx1100"
TARGET = "hip_gfx1151"
PINNED_ARTIFACT = ("MOSTLY_Q4_K_M", "gguf_ud_q4_k_m")
UD_Q4_K_M = pathlib.Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf")

_DENSE_IQ_QUANTS = (
    "gguf_iq4_xs", "gguf_iq4_nl", "gguf_iq3_s", "gguf_iq3_xxs",
    "gguf_iq2_s", "gguf_iq2_xs",
)


def _cap(backend: str, name: str):
    return backend_package_capability(backend, name, None)


# --------------------------------------------------------------------------
# Value parity
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy_name",
    ["GGUF_IQ_DENSE_DECODE_POLICY", "GGUF_IQ_DENSE_VERIFY_POLICY"],
)
def test_gfx1151_declares_the_same_dense_iq_row_regime_policy(policy_name):
    """Both row regimes must route the same quants to the same owner.

    The owner kernels are aliases of one another, so a backend-specific
    difference here cannot be justified by a kernel difference - only a
    deliberate measurement would justify it, and none is recorded.
    """

    source = _cap(SOURCE, policy_name)
    target = _cap(TARGET, policy_name)
    assert source, f"{SOURCE} declares no {policy_name}"
    assert target == source, (
        f"{TARGET}.{policy_name} must match {SOURCE}'s: "
        f"missing={sorted(set(source) - set(target or {}))} "
        f"extra={sorted(set(target or {}) - set(source))}"
    )
    assert set(target) == set(_DENSE_IQ_QUANTS)


@pytest.mark.parametrize(
    "table_name",
    ["GGUF_IQ_DENSE_PREFILL_STRICT_SLOTS", "GGUF_IQ_DENSE_DECODE_STRICT_SLOTS"],
)
def test_gfx1151_declares_the_same_dense_iq_strict_slot_pins(table_name):
    """The per-artifact strict pins are a property of the shared arithmetic.

    gfx1100 pins UD-Q4_K_M's ``layers.0.ffn_up`` out of the Q3_K W4A16
    prefill route and its IQ3_S decode slots out of the local32 route because
    the unpinned routes compounded past the calibrated max-row ceiling. A
    backend that resolves the same kernels and omits the pin runs the
    configuration the pin exists to prevent.
    """

    source = _cap(SOURCE, table_name)
    target = _cap(TARGET, table_name)
    assert source, f"{SOURCE} declares no {table_name}"
    assert target == source, f"{TARGET}.{table_name} must match {SOURCE}'s"
    assert PINNED_ARTIFACT in target


def test_gfx1151_prefill_policy_is_deliberately_not_the_gfx1100_prefill_policy():
    """Guard the one place the two packages are *meant* to differ.

    gfx1151 routes all seven dense-IQ quants through the one-wave W4A16 owner
    while gfx1100 refines four of them to the cooperative owners. That is a
    recorded gfx1151 choice, so the parity tests above must not be generalised
    into "every dense-IQ policy matches".
    """

    source = _cap(SOURCE, "GGUF_IQ_DENSE_PREFILL_POLICY")
    target = _cap(TARGET, "GGUF_IQ_DENSE_PREFILL_POLICY")
    assert set(source) == set(target)
    variants = {str(entry["variant"]) for entry in target.values()}
    assert variants == {"dense_wmma_w4a16_prefill_bf16_bf16_out"}


# --------------------------------------------------------------------------
# Reachability: a declared route must name a kernel the backend owns
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy_name",
    ["GGUF_IQ_DENSE_DECODE_POLICY", "GGUF_IQ_DENSE_VERIFY_POLICY",
     "GGUF_IQ_DENSE_PREFILL_POLICY"],
)
def test_gfx1151_declared_dense_iq_variants_are_registered(policy_name):
    """A policy entry naming an unregistered variant is silently inert.

    ``_iq_dense_decode_dispatch`` falls back to the strict owner when
    ``is_registered`` fails, so a typo or a missing alias would read as
    "routed" in the table and behave as "not routed" at runtime. This is the
    check that keeps the win on the path production actually selects.
    """

    policy = _cap(TARGET, policy_name)
    assert policy
    for quant, entry in policy.items():
        variant = str(entry["variant"])
        key = KernelKey(TARGET, "linear", quant, variant)
        assert is_registered(key), f"{TARGET} declares but does not register {key}"


# --------------------------------------------------------------------------
# Behaviour: the pins must actually reach the dispatch
# --------------------------------------------------------------------------


def _strict_decode(variant: str = "gemv_bf16_bf16_out"):
    return gl.GGUFLinearDispatch(
        KernelKey(TARGET, "linear", "gguf_iq4_xs", variant), "raw"
    )


def _strict_prefill():
    return gl.GGUFLinearDispatch(
        KernelKey(TARGET, "linear", "gguf_q3_k", "prefill_bf16_bf16_out"), "raw"
    )


def test_gfx1151_decode_takes_the_local32_owner_for_an_unpinned_slot():
    """The decode policy must reach dispatch, not just sit in the package."""

    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_source_mmq_prefill import (
        iq_dense_mmq_session,
    )

    with iq_dense_mmq_session(True):
        routed = gl._iq_dense_decode_dispatch(
            _strict_decode(), rows=1, out_features=5120, slot_path="layers.3.ffn_up"
        )
    assert routed.key.variant == "local32_gemv_bf16_bf16_out"
    assert routed.key.backend == TARGET


def test_gfx1151_decode_keeps_the_strict_owner_for_a_pinned_slot():
    """The pin must survive into the decode dispatch for the pinned artifact."""

    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_source_mmq_prefill import (
        iq_dense_mmq_session,
    )

    pinned = tuple(_cap(TARGET, "GGUF_IQ_DENSE_DECODE_STRICT_SLOTS")[PINNED_ARTIFACT])
    assert pinned
    for slot in pinned:
        with iq_dense_mmq_session(True, decode_strict_slots=pinned):
            held = gl._iq_dense_decode_dispatch(
                _strict_decode(), rows=1, out_features=17408, slot_path=slot
            )
        assert held.key.variant == "gemv_bf16_bf16_out", slot


def test_gfx1151_decode_keeps_the_strict_owner_without_a_session():
    """No execution owner means no approximate route, pinned or not."""

    held = gl._iq_dense_decode_dispatch(
        _strict_decode(), rows=1, out_features=5120, slot_path="layers.3.ffn_up"
    )
    assert held.key.variant == "gemv_bf16_bf16_out"


def test_gfx1151_prefill_keeps_the_strict_owner_for_the_pinned_slot():
    """``layers.0.ffn_up`` is the slot the gfx1100 gate pinned. Same kernels,
    same arithmetic, so gfx1151 must pin it too."""

    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_source_mmq_prefill import (
        iq_dense_mmq_session,
    )

    pinned = tuple(_cap(TARGET, "GGUF_IQ_DENSE_PREFILL_STRICT_SLOTS")[PINNED_ARTIFACT])
    assert pinned
    with iq_dense_mmq_session(True, strict_slots=pinned):
        held = gl._iq_dense_prefill_dispatch(
            _strict_prefill(), rows=512, in_features=5120, out_features=17408,
            slot_path=pinned[0],
        )
        routed = gl._iq_dense_prefill_dispatch(
            _strict_prefill(), rows=512, in_features=5120, out_features=17408,
            slot_path="layers.14.ffn_up",
        )
    assert held.key.variant == "prefill_bf16_bf16_out"
    assert routed.key.variant == "dense_wmma_w4a16_prefill_bf16_bf16_out"


# --------------------------------------------------------------------------
# The pins must name slots the pinned artifact actually has
# --------------------------------------------------------------------------


@pytest.mark.skipif(not UD_Q4_K_M.exists(), reason="UD-Q4_K_M artifact not present")
def test_pins_name_real_ud_q4_k_m_slots():
    """A pin naming a slot the artifact does not have is inert.

    Slot paths are ``layers.<block>.<role>`` against ``blk.<block>.<role>``
    tensor names. This is the check that catches a transposed block index or a
    role rename, either of which would silently un-pin the artifact.
    """

    from hipengine.loading.gguf import GGUFReader

    reader = GGUFReader(UD_Q4_K_M)
    present = set()
    for tensor in reader.info.tensors:
        parts = tensor.name.split(".")
        if len(parts) >= 3 and parts[0] == "blk":
            present.add(f"layers.{parts[1]}.{parts[2]}")

    for table_name in ("GGUF_IQ_DENSE_PREFILL_STRICT_SLOTS",
                       "GGUF_IQ_DENSE_DECODE_STRICT_SLOTS"):
        for slot in _cap(TARGET, table_name)[PINNED_ARTIFACT]:
            assert slot in present, f"{table_name} pins absent slot {slot}"
