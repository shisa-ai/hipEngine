"""Concrete GGUF consumer-surface parity (UD-U1 F3).

The cold-path admission coverage names concrete four-axis consumer keys and a
mirrored linear dispatch surface.  Both are metadata (the admission preflight
must not import GPU backend packages), so their truth is bound by explicit
parity against the production registration surface here:

- the mirrored ``GGUF_LINEAR_DISPATCH_SURFACE`` equals the production
  ``runtime/gguf_linear._DISPATCH_TABLE`` exactly (every layout/activation/
  output row, layer, quant token, variant, and ABI), including the
  rows>1 variant mapping the dispatcher applies;
- every concrete consumer key named by the certified coverage records is
  actually registered (or, for the one direct-wrapper consumer, the module
  symbol exists) on every backend that declares GGUF consumer layers;
- every declared GGUF consumer layer has at least one registered ``gguf_*``
  consumer on each declaring backend (declarations are never stale);
- ``cuda_sm120a`` declares no GGUF consumer layers and admission therefore
  refuses it (the scaffold inventory), while both HIP backends declare the
  full certified surface.

Importing the runtime dispatcher modules and the gfx1151 alias registrar is
CPU-safe (registration only; no HIP library load, no kernel build).
"""

from __future__ import annotations

import importlib

import pytest

from hipengine.loading.qwen35_gguf_admission import (
    CERTIFIED_F32_INPUT_OPERATION_COVERAGE,
    CERTIFIED_OPERATION_COVERAGE,
)
from hipengine.loading.qwen35_gguf_consumer_surface import (
    GGUF_CONSUMER_LAYER_DECLARATION_NAME,
    backend_gguf_consumer_layers,
    read_backend_gguf_consumer_layers,
)
from hipengine.kernels.backends import CUDA_BACKEND_TARGET_ARCH, HIP_BACKEND_TARGET_ARCH

GGUF_CONSUMER_BACKENDS = ("hip_gfx1100", "hip_gfx1151")


@pytest.fixture
def production_registry():
    """Import the production registration surface, then (re-)run the exact
    production registrars the runtime dispatcher's import graph triggers.

    Function-scoped on purpose: the pytest conftest restores the
    collection-time registry baseline after every test, which wipes the
    import-time self-registrations of the kernel modules.  Re-running the
    production registrars with ``replace=False`` (the established pattern in
    ``test_coverage_families_are_registered_consumers_not_just_valid_keys``)
    restores the exact registration state after every wipe; the gfx1151
    alias registrar then fills the peer backend exactly as production
    ``load_backend_kernel_package`` does.
    """

    import hipengine.runtime.gguf_linear  # noqa: F401  (import-time registration)
    import hipengine.runtime.gguf_embedding  # noqa: F401  (import-time registration)
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import register_gguf_ops
    from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
        register_dense_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
        register_qwen35_linear_attn_conv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
        register_qwen35_linear_attn_gdn_kernels,
    )
    from hipengine.kernels.hip_gfx1100.moe.router import (
        register_qwen35_router_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        register_gguf_k_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
        register_gguf_k_t16_selected_prefill_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q3_k_gemv import (
        register_gguf_q3_k_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        register_gguf_q4_k_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import (
        register_gguf_q6_k_embedding_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        register_gguf_q6_k_t16_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv import (
        register_gguf_q8_0_t16_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        register_gguf_t16_selected_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_gemv import (
        register_gguf_iq_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_x8_selected_gemv import (
        register_gguf_x8_selected_gemv_kernels,
    )
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import (
        DuplicateKernelError,
        is_registered,
        registered_keys,
    )

    for registrar in (
        register_gguf_k_gemv_kernels,
        register_gguf_q4_k_gemv_kernels,
        register_gguf_q3_k_gemv_kernels,
        register_gguf_iq_gemv_kernels,
        register_gguf_x8_selected_gemv_kernels,
        register_gguf_q8_0_t16_gemv_kernels,
        register_gguf_t16_selected_gemv_kernels,
        register_gguf_q6_k_t16_gemv_kernels,
        register_gguf_k_t16_selected_prefill_kernels,
        register_gguf_q6_k_embedding_kernels,
        register_dense_gemv_kernels,
        register_gguf_ops,
        register_qwen35_linear_attn_conv_kernels,
        register_qwen35_linear_attn_gdn_kernels,
        register_qwen35_router_kernels,
    ):
        try:
            registrar(replace=False)
        except DuplicateKernelError:
            pass
    for backend in GGUF_CONSUMER_BACKENDS:
        load_backend_kernel_package(backend)
    return is_registered, registered_keys


def test_dispatch_surface_mirror_equals_production_table():
    import hipengine.runtime.gguf_linear as production
    from hipengine.runtime.gguf_linear import _variant_for_rows

    from hipengine.loading.qwen35_gguf_consumer_surface import (
        GGUF_LINEAR_DISPATCH_SURFACE,
    )

    surface_keys = {
        (row.layout, row.activation, row.output): row for row in GGUF_LINEAR_DISPATCH_SURFACE
    }
    table_keys = dict(production._DISPATCH_TABLE)
    assert set(surface_keys) == set(table_keys), (
        "mirrored dispatch surface drifted from the production table: "
        f"mirror-only={sorted(set(surface_keys) - set(table_keys))} "
        f"table-only={sorted(set(table_keys) - set(surface_keys))}"
    )
    for key, dispatch in table_keys.items():
        row = surface_keys[key]
        assert row.layer == dispatch.key.layer, (key, row, dispatch.key)
        assert row.quant == dispatch.key.quant, (key, row, dispatch.key)
        assert row.variant == dispatch.key.variant, (key, row, dispatch.key)
        assert row.abi == dispatch.abi, (key, row, dispatch.abi)
        # The rows>1 variant must be exactly what the production dispatcher
        # resolves for multirow launches (including the Q4 T16 rewrite).
        for rows in (1, 2, 8):
            expected = _variant_for_rows(dispatch.key.variant, rows=rows)
            if key[0] in {"gguf_q4_k_t16_v1", "gguf_q4_k_qmicro_t16_v1"} and rows > 1:
                expected = "t16_wmma_prefill_bf16_bf16_out"
            assert row.variant_for_rows(rows) == expected, (key, rows)


def test_backend_declarations_are_literally_readable():
    declarations = read_backend_gguf_consumer_layers()
    for backend in (*HIP_BACKEND_TARGET_ARCH, *CUDA_BACKEND_TARGET_ARCH):
        assert backend in declarations
        layers = declarations[backend]
        assert isinstance(layers, frozenset)
        # The declaration name is a real module constant in each backend
        # package (the same convention the audit's capability reader uses).
        module = importlib.import_module(f"hipengine.kernels.{backend}")
        declared = frozenset(
            getattr(module, GGUF_CONSUMER_LAYER_DECLARATION_NAME, frozenset())
        )
        assert declared == layers


def test_cuda_scaffold_declares_no_gguf_consumers(production_registry):
    """Inventory check (not an AMD blanket assumption): the cuda_sm120a
    package registers real kernels (moonshine/maple/PARO families), but none
    under GGUF consumer contracts — no gguf-quant key participates in the
    GGUF linear/embedding/router/GDN dispatch.  Its declaration is therefore
    empty and GGUF admission must refuse it."""

    _is_registered, registered_keys = production_registry
    for backend in CUDA_BACKEND_TARGET_ARCH:
        assert backend_gguf_consumer_layers(backend) == frozenset()
        gguf_keys = [
            key
            for key in registered_keys()
            if key.backend == backend
            and (
                key.quant.startswith("gguf")
                or key.layer
                in {
                    "embedding",
                    "router_logits",
                    "gdn_recurrent_rmsnorm_gate",
                    "linear_attn_conv_decode",
                    "linear_attn_conv_prefill",
                }
            )
        ]
        assert gguf_keys == [], gguf_keys[:8]


@pytest.mark.parametrize("backend", GGUF_CONSUMER_BACKENDS)
def test_declared_layers_have_registered_gguf_consumers(production_registry, backend):
    """Every declared GGUF consumer layer is backed by at least one real
    registration on that backend (declarations never go stale silently)."""

    _is_registered, registered_keys = production_registry
    declared = backend_gguf_consumer_layers(backend)
    assert declared, "HIP GGUF backends must declare their consumer layers"
    registered_layers = {
        key.layer for key in registered_keys() if key.backend == backend
    }
    for layer in declared:
        assert layer in registered_layers, (backend, layer)


_ALL_RECORDS = (*CERTIFIED_OPERATION_COVERAGE, *CERTIFIED_F32_INPUT_OPERATION_COVERAGE)


def test_every_coverage_record_names_a_concrete_consumer():
    for record in _ALL_RECORDS:
        assert record.kernel_layer, record
        assert record.kernel_variant, record
        if record.consumer_module is None:
            assert record.kernel_quant, record
        else:
            assert record.consumer_symbol, record
        # Row-mode contract: when rows>1 resolves a different variant, the
        # record names both concrete variants.
        if record.kernel_variant_rows_many is not None:
            assert record.kernel_variant_rows_many != record.kernel_variant


@pytest.mark.parametrize("backend", GGUF_CONSUMER_BACKENDS)
def test_every_certified_consumer_key_is_registered(production_registry, backend):
    """No skips: every concrete consumer named by every certified record
    (default and F32-input override sets) must exist on every backend that
    declares GGUF consumers — registry keys exactly registered, direct
    wrapper consumers present as module symbols."""

    is_registered, _registered_keys = production_registry
    from hipengine.kernels.registry import KernelKey

    declared = backend_gguf_consumer_layers(backend)
    for record in _ALL_RECORDS:
        assert record.kernel_layer in declared, (backend, record)
        variants = [record.kernel_variant]
        if record.kernel_variant_rows_many is not None:
            variants.append(record.kernel_variant_rows_many)
        if record.consumer_module is not None:
            module = importlib.import_module(record.consumer_module)
            assert hasattr(module, str(record.consumer_symbol)), (backend, record)
            continue
        for variant in variants:
            key = KernelKey(backend, record.kernel_layer, record.kernel_quant, variant)
            assert is_registered(key), (
                f"coverage record names an unregistered consumer: {key} "
                f"(operation={record.operation} role={record.role_class})"
            )


def test_backend_declares_exactly_the_certified_consumer_layers():
    certified_layers = {record.kernel_layer for record in _ALL_RECORDS}
    for backend in GGUF_CONSUMER_BACKENDS:
        assert backend_gguf_consumer_layers(backend) == certified_layers, backend


def test_linear_records_bind_the_mirrored_dispatch_surface():
    """Every linear-family certified record's dtype/variant contract is the
    concrete mirrored dispatch row for its (layout, activation, output) —
    coverage can never name a variant the dispatcher would not resolve."""

    from hipengine.loading.qwen35_gguf_consumer_surface import (
        GGUF_LINEAR_DISPATCH_SURFACE,
        gguf_linear_dispatch_row,
        source_linear_dispatch_row,
    )

    surface = {
        (row.layout, row.activation, row.output): row for row in GGUF_LINEAR_DISPATCH_SURFACE
    }
    linear_records = [
        record
        for record in _ALL_RECORDS
        if record.kernel_layer in {"linear", "dense_gemv"}
        # Selected-expert consumers bypass the layout dispatch table (direct
        # selected-GEMV launch with expert IDs); they are covered by the
        # registration parity test instead.
        and record.role_class != "moe_experts"
    ]
    assert linear_records
    for record in linear_records:
        row = surface.get((record.resident_layout, record.input_dtype, record.output_dtype))
        assert row is not None, record
        assert record.kernel_layer == row.layer, record
        source_type = next(iter(sorted(record.source_ggml_types)))
        concrete = source_linear_dispatch_row(
            source_type,
            record.resident_layout,
            record.input_dtype,
            record.output_dtype,
        )
        assert concrete is not None, record
        assert record.kernel_quant == concrete.quant, record
        assert record.kernel_variant == concrete.variant, record
        assert gguf_linear_dispatch_row(
            record.resident_layout,
            record.input_dtype,
            record.output_dtype,
        ) is not None
