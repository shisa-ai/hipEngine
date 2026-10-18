"""Structured provenance for the Qwen4Exp AR batch decode path.

The c1 execution-profile manifest describes prefill and single-request decode
scopes. It is not evidence for the N-row batch mixers, the request-indexed GDN
and PLE kernels, the mixed QSA rows attention, or the row-dependent projection
dispatch. This module defines a separate batch variant manifest and validates it
against the live callables and registry keys instead of matching substrings in
the c1 manifest.

The manifest records three different kinds of binding and never conflates them:

``direct``
    The batch module imports and calls the function itself. Its module-qualified
    identity, source hash, function hash, and build identity are recorded. A
    matching registry key may be listed as provenance, but the entry stays
    direct and a registry key on a direct entry is rejected.

``registry``
    The batch module resolves the key through the four-axis registry. The key,
    the resolved function, and the registered strict fallback are recorded.

``registry_dispatch``
    The projection launcher resolves a row-dependent key per call. The manifest
    records the keys observed during the measured batch steps, including the
    resolved function identity for each key.

A manifest is not trusted because it says so. :func:`validate_batch_variant_manifest`
recomputes every direct identity and every registry resolution from the live
process, checks the canonical hash, binds the profile manifest hash and source
revision, and requires the batch-specific roles. Roles the batch path does not
bind are reported in ``unbound`` or ``not_used`` instead of being implied.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import inspect
import json
from pathlib import Path
import re
from typing import Any

from hipengine.execution_profiles import manifest_sha256
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    qwen35_write_paged_kv_f32_spans,
)
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
    qwen4_exp_qsa_scatter_index_keys_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_f32,
    qwen4_exp_qsa_split_norm_mrope_f32,
    qwen4_exp_qsa_split_norm_rope_f32,
    qwen4_exp_qsa_topk_expand_f32_i64,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_ple import (
    qwen4_exp_ple_dilated_depthwise_conv_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_conv_decode_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
    qwen4_exp_gdn_decode_f32,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.runtime.gguf_linear import launch_gguf_linear
from hipengine.runtime.qwen4_exp_batch_gdn import batch_gdn_call_table
from hipengine.runtime.qwen4_exp_batch_ple import (
    batch_ple_call_table,
    batch_ple_indexed_conv_fallback_key,
    batch_ple_indexed_conv_key,
)
from hipengine.runtime.qwen4_exp_batch_qsa import batch_qsa_call_table
from hipengine.runtime.qwen4_exp_batch_qsa_index import batch_qsa_index_call_table
from hipengine.runtime.qwen4_exp_batch_runner import (
    Qwen4ExpGGUFBatchRunner,
    batch_runner_call_table,
)
from hipengine.runtime.qwen4_exp_runner import (
    run_qwen4_exp_dense_qsa_token_mixer,
    run_qwen4_exp_ple,
)


BATCH_VARIANT_MANIFEST_KIND = "hipengine_batch_variant_manifest"
BATCH_VARIANT_MANIFEST_SCHEMA_VERSION = 1
BATCH_PREFILL_SOURCE = "c1_qwen4_exp_gguf_resident_model_runner"
BATCH_DECODE_SCOPE = "teacher_forced_batch_rows"

#: Roles the batch-specific decode claim always requires. PLE roles are added
#: when the runtime config has PLE layers.
BATCH_CORE_REQUIRED_ROLES = (
    "gdn_conv_indexed",
    "gdn_recurrence_norm_gate",
    "qsa_mixed_mixer",
    "qsa_split_norm_rope",
    "qsa_paged_kv_write",
    "qsa_rows_attention",
    "qsa_gate_context",
    "qsa_index_append",
    "qsa_index_pool_norm_rope",
    "qsa_index_score",
    "qsa_index_select",
    "projection_dispatch",
)
BATCH_PLE_REQUIRED_ROLES = ("ple_stage", "ple_indexed_conv")

_SHARED_ROLES = frozenset(
    {
        "gr_read",
        "gr_write_f32",
        "gr_write_bf16",
        "moe",
        "repeat_branches",
        "embedding",
    }
)
_FREE_DECODE_ROLES = frozenset({"argmax"})

#: Direct calls whose matching registry key is useful provenance. The builder
#: verifies each key resolves to the same function; the entry remains direct.
_DIRECT_REGISTERED_KEYS: dict[str, tuple[str, str, str]] = {
    "gdn_conv_indexed": (
        "linear_attn_conv_decode",
        "f32_state",
        "qwen4exp_conv_indexed_decode",
    ),
    "gdn_recurrence_norm_gate": (
        "gdn_recurrence_norm_gate",
        "f32_state",
        "qwen4exp_gdn_indexed_decode",
    ),
    "qsa_split_norm_rope": ("qsa_split_norm_rope", "f32", "strict_rows"),
    "qsa_rows_attention": ("qsa_sparse_attention", "bf16_kv", "strict_rows_spans"),
    "qsa_index_append": ("qsa_index_append", "f32", "strict_device_position_c1"),
    "qsa_index_pool_norm_rope": ("qsa_pool_norm_rope", "f32", "strict"),
    "qsa_index_score": ("qsa_index_score", "f32", "strict"),
    "qsa_index_select": (
        "qsa_select_blocks",
        "f32_i64",
        "strict_device_expand_rows",
    ),
}

#: The strict single-request or single-row owner for each direct batch call.
_ORACLE_CALLS: dict[str, Callable[..., Any]] = {
    "gdn_conv_indexed": qwen35_linear_attn_conv_decode_f32,
    "gdn_recurrence_norm_gate": qwen4_exp_gdn_decode_f32,
    "qsa_mixed_mixer": run_qwen4_exp_dense_qsa_token_mixer,
    "qsa_split_norm_rope": qwen4_exp_qsa_split_norm_rope_f32,
    "qsa_split_norm_mrope": qwen4_exp_qsa_split_norm_mrope_f32,
    "qsa_paged_kv_write": qwen35_write_paged_kv_f32_spans,
    "qsa_rows_attention": qwen4_exp_qsa_sparse_attention_paged_bf16_f32,
    "qsa_index_append": qwen4_exp_qsa_scatter_index_keys_f32,
    "qsa_index_select": qwen4_exp_qsa_topk_expand_f32_i64,
    "ple_stage": run_qwen4_exp_ple,
    "ple_indexed_conv": qwen4_exp_ple_dilated_depthwise_conv_f32,
}

_KERNEL_SYMBOL_RE = re.compile(r'"(hipengine_[A-Za-z0-9_]+)"')
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROOT = Path(__file__).resolve().parents[2]


# -- identity helpers -------------------------------------------------------


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def function_identity(fn: Callable[..., Any] | type[Any]) -> str:
    """Return the stable module-qualified identity of a callable."""

    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname:
        raise ValueError(f"callable has no module-qualified identity: {fn!r}")
    return f"{module}.{qualname}"


def source_path(fn: Callable[..., Any] | type[Any]) -> str:
    """Return the source file path, relative to the repository when possible."""

    path = inspect.getsourcefile(fn) or inspect.getfile(fn)
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(_ROOT))
    except ValueError:
        return str(resolved)


def source_file_sha256(fn: Callable[..., Any] | type[Any]) -> str:
    """Hash the whole source file that defines ``fn``."""

    path = inspect.getsourcefile(fn) or inspect.getfile(fn)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_source_sha256(fn: Callable[..., Any] | type[Any]) -> str:
    """Hash the exact source text of ``fn``."""

    return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()


def kernel_symbol(fn: Callable[..., Any]) -> str | None:
    """Return the one exported HIP symbol named in ``fn``'s source, if unique.

    A wrapper that names its symbol through a module constant, or names more
    than one symbol, returns ``None`` rather than guessing.
    """

    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    symbols = sorted(set(_KERNEL_SYMBOL_RE.findall(source)))
    return symbols[0] if len(symbols) == 1 else None


def function_build_identity(
    fn: Callable[..., Any] | type[Any],
    *,
    kernel_symbol: str | None = None,
) -> str:
    """Stable source/build identity for a Python wrapper.

    This identifies the wrapper and its build entry point. It is not a
    cryptographic attestation of the compiled ``.so``; the compiled artifact is
    named by the wrapper's kernel symbol and source revision where available.
    """

    return _sha256_json(
        {
            "identity": function_identity(fn),
            "source_sha256": source_file_sha256(fn),
            "function_sha256": function_source_sha256(fn),
            "kernel_symbol": kernel_symbol,
        }
    )


def _key_payload(key: KernelKey) -> dict[str, str]:
    return {
        "backend": key.backend,
        "layer": key.layer,
        "quant": key.quant,
        "variant": key.variant,
    }


def _key_from_payload(value: Any) -> KernelKey:
    if not isinstance(value, Mapping):
        raise ValueError("registry key must be a mapping")
    try:
        return KernelKey(
            backend=str(value["backend"]),
            layer=str(value["layer"]),
            quant=str(value["quant"]),
            variant=str(value["variant"]),
        )
    except KeyError as exc:
        raise ValueError(f"registry key missing field: {exc}") from None


def _resolve_key(key: KernelKey) -> Callable[..., Any]:
    load_backend_kernel_package(key.backend)
    if not is_registered(key):
        raise ValueError(f"registry key is not registered: {key.display()}")
    return resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )


def _direct_reference(fn: Callable[..., Any]) -> dict[str, Any]:
    symbol = kernel_symbol(fn)
    return {
        "binding": "direct",
        "identity": function_identity(fn),
        "source": source_path(fn),
        "source_sha256": source_file_sha256(fn),
        "function_sha256": function_source_sha256(fn),
        "build_identity": function_build_identity(fn, kernel_symbol=symbol),
        "kernel_symbol": symbol,
        "registry_key": None,
    }


def _registry_reference(key: KernelKey) -> dict[str, Any]:
    fn = _resolve_key(key)
    symbol = kernel_symbol(fn)
    return {
        "binding": "registry",
        "identity": function_identity(fn),
        "source": source_path(fn),
        "source_sha256": source_file_sha256(fn),
        "function_sha256": function_source_sha256(fn),
        "build_identity": function_build_identity(fn, kernel_symbol=symbol),
        "kernel_symbol": symbol,
        "registry_key": _key_payload(key),
    }


def _direct_entry(
    role: str,
    fn: Callable[..., Any],
    *,
    backend: str,
    scope: str,
    oracle: Callable[..., Any] | None = None,
    registered_key: tuple[str, str, str] | None = None,
) -> dict[str, Any]:
    symbol = kernel_symbol(fn)
    entry = {
        "role": role,
        "binding": "direct",
        "scope": scope,
        "identity": function_identity(fn),
        "source": source_path(fn),
        "source_sha256": source_file_sha256(fn),
        "function_sha256": function_source_sha256(fn),
        "build_identity": function_build_identity(fn, kernel_symbol=symbol),
        "kernel_symbol": symbol,
        "registry_key": None,
        "registered_keys": [],
        "strict_fallback": None,
        "oracle": None,
    }
    if registered_key is not None:
        layer, quant, variant = registered_key
        key = KernelKey(backend, layer, quant, variant)
        registered_fn = _resolve_key(key)
        if registered_fn is not fn:
            raise ValueError(
                f"direct role {role!r} claims registry key {key.display()} but it "
                "resolves to a different function"
            )
        entry["registered_keys"] = [_key_payload(key)]
    if oracle is not None:
        reference = _direct_reference(oracle)
        entry["strict_fallback"] = reference
        entry["oracle"] = dict(reference)
    return entry


def _registry_entry(
    role: str,
    key: KernelKey,
    *,
    scope: str,
    fallback_key: KernelKey | None = None,
) -> dict[str, Any]:
    fn = _resolve_key(key)
    symbol = kernel_symbol(fn)
    entry = {
        "role": role,
        "binding": "registry",
        "scope": scope,
        "identity": function_identity(fn),
        "source": source_path(fn),
        "source_sha256": source_file_sha256(fn),
        "function_sha256": function_source_sha256(fn),
        "build_identity": function_build_identity(fn, kernel_symbol=symbol),
        "kernel_symbol": symbol,
        "registry_key": _key_payload(key),
        "registered_keys": [_key_payload(key)],
        "strict_fallback": None,
        "oracle": None,
    }
    if fallback_key is not None:
        fallback = _registry_reference(fallback_key)
        entry["strict_fallback"] = fallback
        entry["oracle"] = dict(fallback)
    return entry


def _projection_entry(
    records: Sequence[Mapping[str, Any]],
    *,
    required_rows: Sequence[int],
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in records:
        key = _key_from_payload(raw.get("key"))
        rows = int(raw["rows"])
        in_features = int(raw["in_features"])
        out_features = int(raw["out_features"])
        activation_dtype = str(raw["activation_dtype"])
        output_dtype = str(raw["output_dtype"])
        abi = str(raw["abi"])
        if rows <= 0 or in_features <= 0 or out_features <= 0:
            raise ValueError("projection dispatch geometry must be positive")
        if activation_dtype != "f32" or output_dtype != "f32":
            raise ValueError(
                "batch projection dispatch must use F32 activation and output "
                f"dtypes, got {activation_dtype!r}/{output_dtype!r}"
            )
        group_key = (
            rows,
            key,
            in_features,
            out_features,
            activation_dtype,
            output_dtype,
            abi,
        )
        group = grouped.get(group_key)
        if group is None:
            grouped[group_key] = {
                "key": key,
                "rows": rows,
                "in_features": in_features,
                "out_features": out_features,
                "activation_dtype": activation_dtype,
                "output_dtype": output_dtype,
                "abi": abi,
                "calls": 1,
            }
        else:
            group["calls"] += 1
    observed: list[dict[str, Any]] = []
    for group_key in sorted(grouped, key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5], item[6])):
        group = grouped[group_key]
        fn = _resolve_key(group["key"])
        symbol = kernel_symbol(fn)
        observed.append(
            {
                "key": _key_payload(group["key"]),
                "identity": function_identity(fn),
                "source": source_path(fn),
                "source_sha256": source_file_sha256(fn),
                "function_sha256": function_source_sha256(fn),
                "build_identity": function_build_identity(fn, kernel_symbol=symbol),
                "kernel_symbol": symbol,
                "rows": group["rows"],
                "in_features": group["in_features"],
                "out_features": group["out_features"],
                "activation_dtype": group["activation_dtype"],
                "output_dtype": group["output_dtype"],
                "abi": group["abi"],
                "calls": group["calls"],
            }
        )
    observed_rows = sorted({row["rows"] for row in observed})
    missing_rows = sorted(set(int(row) for row in required_rows) - set(observed_rows))
    if missing_rows:
        raise ValueError(
            "projection dispatch has no record for step width(s): "
            f"{missing_rows}"
        )
    resolver = launch_gguf_linear
    return {
        "role": "projection_dispatch",
        "binding": "registry_dispatch",
        "scope": "decode_rows_1_4",
        "identity": function_identity(resolver),
        "source": source_path(resolver),
        "source_sha256": source_file_sha256(resolver),
        "function_sha256": function_source_sha256(resolver),
        "build_identity": function_build_identity(resolver),
        "kernel_symbol": None,
        "registry_key": None,
        "registered_keys": [],
        "strict_fallback": None,
        "oracle": None,
        "resolver": function_identity(resolver),
        "observed": observed,
        "rows": observed_rows,
        "required_rows": sorted(set(int(row) for row in required_rows)),
    }


def _runtime_identity(runtime_class: type[Any]) -> dict[str, Any]:
    symbol = None
    return {
        "module": runtime_class.__module__,
        "class": runtime_class.__qualname__,
        "identity": function_identity(runtime_class),
        "source": source_path(runtime_class),
        "source_sha256": source_file_sha256(runtime_class),
        "function_sha256": function_source_sha256(runtime_class),
        "build_identity": function_build_identity(runtime_class, kernel_symbol=symbol),
    }


def _config_summary(config: Any) -> dict[str, Any]:
    layer_types = [str(kind) for kind in getattr(config, "layer_types", ()) or ()]
    ple_layers = sorted(int(layer) for layer in getattr(config, "ple_layers", ()) or ())
    return {
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "context_length": int(getattr(config, "context_length", 0)),
        "qsa_compression_ratio": int(getattr(config, "qsa_compression_ratio", 0)),
        "layer_types": layer_types,
        "gdn_layer_count": layer_types.count("gdn"),
        "qsa_layer_count": layer_types.count("qsa"),
        "ple_layers": ple_layers,
    }


def _embedding_quant(runner: Any, explicit: str | None) -> str | None:
    if explicit:
        return str(explicit)
    try:
        return str(runner.resident.weight("root.token_embedding").spec.quant_key)
    except (AttributeError, KeyError, TypeError):
        return None


def live_batch_direct_calls() -> dict[str, Callable[..., Any]]:
    """Return the exact callables the batch modules import and call directly."""

    calls: dict[str, Callable[..., Any]] = {}
    calls.update(batch_gdn_call_table())
    calls.update(batch_qsa_call_table())
    calls.update(batch_qsa_index_call_table())
    calls.update(batch_ple_call_table())
    calls.update(batch_runner_call_table())
    return calls


def live_batch_oracle_calls() -> dict[str, Callable[..., Any]]:
    """Return the strict oracle/fallback callables for the batch-specific roles."""

    return dict(_ORACLE_CALLS)


# -- builder ----------------------------------------------------------------


def build_qwen4_exp_batch_variant_manifest(
    *,
    runner: Any,
    profile_manifest: Mapping[str, Any],
    source_revision: str,
    dispatch_records: Sequence[Mapping[str, Any]],
    step_widths: Mapping[int, int],
    runtime_class: type[Any] | None = None,
    embedding_quant: str | None = None,
) -> dict[str, Any]:
    """Build and validate the batch manifest from the live runtime.

    ``dispatch_records`` must come from
    :func:`hipengine.runtime.gguf_linear.record_gguf_linear_dispatch` around the
    measured batch steps. The builder resolves every recorded key again and
    rejects a record that does not match the live registry.
    """

    backend = str(getattr(runner, "backend", ""))
    if not backend:
        raise ValueError("batch runner has no backend identity")
    profile_hash = manifest_sha256(profile_manifest)
    if str(profile_manifest.get("backend")) != backend:
        raise ValueError(
            "batch runner backend does not match the profile manifest backend"
        )
    revision = str(source_revision or "").strip()
    if not revision:
        raise ValueError("batch variant manifest needs a source revision")

    config = _config_summary(getattr(runner, "config", None))
    required_roles = list(BATCH_CORE_REQUIRED_ROLES)
    if config["ple_layers"]:
        required_roles.extend(BATCH_PLE_REQUIRED_ROLES)

    entries: list[dict[str, Any]] = []
    unbound: list[dict[str, str]] = [
        {
            "role": "prefill",
            "reason": (
                "prefill is the committed c1 resident path and is excluded from "
                "the batch claim"
            ),
        },
        {
            "role": "ple_hash_rows",
            "reason": "host CPU reference n-gram hash, not a registered device kernel",
        },
        {
            "role": "stage_qwen4_exp_ple_rows",
            "reason": "host staging helper, not a registered device kernel",
        },
    ]
    not_used: list[dict[str, str]] = [
        {
            "role": "qsa_split_norm_mrope",
            "reason": "batch runner passes no MRoPE position pointer",
        },
        {
            "role": "dense_paged_decode",
            "reason": (
                "mixed QSA attends through the variable-selection rows kernel; "
                "the dense paged decode is not called"
            ),
        },
    ]

    direct_calls = live_batch_direct_calls()
    for role, fn in sorted(direct_calls.items()):
        if role == "qsa_split_norm_mrope":
            continue
        scope = (
            "shared_c1"
            if role in _SHARED_ROLES
            else "free_decode_only"
            if role in _FREE_DECODE_ROLES
            else "batch_decode"
        )
        entries.append(
            _direct_entry(
                role,
                fn,
                backend=backend,
                scope=scope,
                oracle=_ORACLE_CALLS.get(role),
                registered_key=_DIRECT_REGISTERED_KEYS.get(role),
            )
        )

    if config["ple_layers"]:
        entries.append(
            _registry_entry(
                "ple_indexed_conv",
                batch_ple_indexed_conv_key(backend),
                scope="batch_decode",
                fallback_key=batch_ple_indexed_conv_fallback_key(backend),
            )
        )

    quant = _embedding_quant(runner, embedding_quant)
    if quant is None:
        unbound.append(
            {
                "role": "embedding",
                "reason": "embedding quant identity is unavailable on this runner",
            }
        )
    else:
        try:
            entries.append(
                _registry_entry(
                    "embedding",
                    KernelKey(backend, "embedding", quant, "lookup_bf16_out"),
                    scope="shared_c1",
                )
            )
        except ValueError as exc:
            unbound.append({"role": "embedding", "reason": str(exc)})

    widths = sorted({int(width) for width in step_widths})
    if not widths:
        raise ValueError("batch variant manifest needs at least one step width")
    entries.append(_projection_entry(dispatch_records, required_rows=widths))

    selected_runtime = runtime_class or type(runner)

    body = {
        "kind": BATCH_VARIANT_MANIFEST_KIND,
        "schema_version": BATCH_VARIANT_MANIFEST_SCHEMA_VERSION,
        "backend": backend,
        "model": str(profile_manifest["model"]),
        "quant": str(profile_manifest["quant"]),
        "execution_profile": str(profile_manifest["execution_profile"]),
        "profile_manifest_sha256": profile_hash,
        "source_revision": revision,
        "prefill_source": BATCH_PREFILL_SOURCE,
        "decode_scope": BATCH_DECODE_SCOPE,
        "runtime": _runtime_identity(selected_runtime),
        "runtime_config": config,
        "required_roles": required_roles,
        "entries": entries,
        "unbound": unbound,
        "not_used": not_used,
        "observed_step_widths": {
            str(width): int(step_widths[width]) for width in widths
        },
    }
    normalized = normalize_batch_variant_manifest(body)
    normalized["manifest_sha256"] = batch_variant_manifest_sha256(normalized)
    validation = validate_batch_variant_manifest(
        normalized,
        expected_profile_manifest_sha256=profile_hash,
        expected_source_revision=revision,
        expected_backend=backend,
        expected_model=str(profile_manifest["model"]),
        expected_quant=str(profile_manifest["quant"]),
    )
    if not validation["valid"]:
        raise ValueError(
            "built batch variant manifest did not validate: "
            + "; ".join(validation["reasons"])
        )
    return normalized


# -- normalization ----------------------------------------------------------


def _require_nonempty(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"batch manifest {field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, *, field: str) -> str:
    text = _require_nonempty(value, field=field)
    if not _SHA256_RE.match(text):
        raise ValueError(f"batch manifest {field} must be a SHA-256 hex digest")
    return text


def _normalize_key(value: Any) -> dict[str, str]:
    key = _key_from_payload(value)
    return _key_payload(key)


def _normalize_reference(value: Any, *, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"batch manifest {field} must be a mapping or null")
    binding = _require_nonempty(value.get("binding"), field=f"{field}.binding")
    if binding not in {"direct", "registry"}:
        raise ValueError(f"batch manifest {field}.binding is unsupported: {binding!r}")
    result: dict[str, Any] = {
        "binding": binding,
        "identity": _require_nonempty(value.get("identity"), field=f"{field}.identity"),
        "source": _require_nonempty(value.get("source"), field=f"{field}.source"),
        "source_sha256": _require_sha256(
            value.get("source_sha256"), field=f"{field}.source_sha256"
        ),
        "function_sha256": _require_sha256(
            value.get("function_sha256"), field=f"{field}.function_sha256"
        ),
        "build_identity": _require_sha256(
            value.get("build_identity"), field=f"{field}.build_identity"
        ),
        "kernel_symbol": value.get("kernel_symbol"),
        "registry_key": None,
    }
    if result["kernel_symbol"] is not None:
        result["kernel_symbol"] = _require_nonempty(
            result["kernel_symbol"], field=f"{field}.kernel_symbol"
        )
    raw_key = value.get("registry_key")
    if binding == "registry":
        if raw_key is None:
            raise ValueError(f"batch manifest {field}.registry_key is required")
        result["registry_key"] = _normalize_key(raw_key)
    elif raw_key is not None:
        raise ValueError(
            f"batch manifest {field} is direct and must not claim a registry key"
        )
    unknown = set(value) - set(result)
    if unknown:
        raise ValueError(f"batch manifest {field} has unknown fields: {sorted(unknown)}")
    return result


def _normalize_entry(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("batch manifest entry must be a mapping")
    role = _require_nonempty(value.get("role"), field="entry.role")
    binding = _require_nonempty(value.get("binding"), field=f"entry[{role}].binding")
    if binding not in {"direct", "registry", "registry_dispatch"}:
        raise ValueError(f"batch manifest entry[{role}] has unsupported binding {binding!r}")
    result: dict[str, Any] = {
        "role": role,
        "binding": binding,
        "scope": _require_nonempty(value.get("scope"), field=f"entry[{role}].scope"),
        "identity": _require_nonempty(
            value.get("identity"), field=f"entry[{role}].identity"
        ),
        "source": _require_nonempty(value.get("source"), field=f"entry[{role}].source"),
        "source_sha256": _require_sha256(
            value.get("source_sha256"), field=f"entry[{role}].source_sha256"
        ),
        "function_sha256": _require_sha256(
            value.get("function_sha256"), field=f"entry[{role}].function_sha256"
        ),
        "build_identity": _require_sha256(
            value.get("build_identity"), field=f"entry[{role}].build_identity"
        ),
        "kernel_symbol": value.get("kernel_symbol"),
        "registry_key": None,
        "registered_keys": [],
        "strict_fallback": None,
        "oracle": None,
    }
    if result["kernel_symbol"] is not None:
        result["kernel_symbol"] = _require_nonempty(
            result["kernel_symbol"], field=f"entry[{role}].kernel_symbol"
        )
    raw_key = value.get("registry_key")
    if binding == "registry":
        if raw_key is None:
            raise ValueError(f"batch manifest entry[{role}] needs a registry key")
        result["registry_key"] = _normalize_key(raw_key)
    elif binding == "registry_dispatch":
        if raw_key is not None:
            raise ValueError(
                f"batch manifest dispatch entry[{role}] must not claim one "
                "registry key; its observed records carry the keys"
            )
    elif raw_key is not None:
        raise ValueError(
            f"batch manifest direct entry[{role}] must not claim a registry key"
        )
    raw_registered = value.get("registered_keys", [])
    if not isinstance(raw_registered, Sequence) or isinstance(
        raw_registered, (str, bytes)
    ):
        raise ValueError(f"batch manifest entry[{role}].registered_keys must be a list")
    result["registered_keys"] = [_normalize_key(key) for key in raw_registered]
    result["strict_fallback"] = _normalize_reference(
        value.get("strict_fallback"), field=f"entry[{role}].strict_fallback"
    )
    result["oracle"] = _normalize_reference(
        value.get("oracle"), field=f"entry[{role}].oracle"
    )
    if binding == "registry_dispatch":
        result["resolver"] = _require_nonempty(
            value.get("resolver"), field=f"entry[{role}].resolver"
        )
        raw_observed = value.get("observed")
        if not isinstance(raw_observed, Sequence) or isinstance(
            raw_observed, (str, bytes)
        ):
            raise ValueError(f"batch manifest entry[{role}].observed must be a list")
        observed: list[dict[str, Any]] = []
        for index, record in enumerate(raw_observed):
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"batch manifest entry[{role}].observed[{index}] must be a mapping"
                )
            observed.append(
                {
                    "key": _normalize_key(record.get("key")),
                    "identity": _require_nonempty(
                        record.get("identity"),
                        field=f"entry[{role}].observed[{index}].identity",
                    ),
                    "source": _require_nonempty(
                        record.get("source"),
                        field=f"entry[{role}].observed[{index}].source",
                    ),
                    "source_sha256": _require_sha256(
                        record.get("source_sha256"),
                        field=f"entry[{role}].observed[{index}].source_sha256",
                    ),
                    "function_sha256": _require_sha256(
                        record.get("function_sha256"),
                        field=f"entry[{role}].observed[{index}].function_sha256",
                    ),
                    "build_identity": _require_sha256(
                        record.get("build_identity"),
                        field=f"entry[{role}].observed[{index}].build_identity",
                    ),
                    "kernel_symbol": record.get("kernel_symbol"),
                    "rows": int(record.get("rows")),
                    "in_features": int(record.get("in_features")),
                    "out_features": int(record.get("out_features")),
                    "activation_dtype": _require_nonempty(
                        record.get("activation_dtype"),
                        field=f"entry[{role}].observed[{index}].activation_dtype",
                    ),
                    "output_dtype": _require_nonempty(
                        record.get("output_dtype"),
                        field=f"entry[{role}].observed[{index}].output_dtype",
                    ),
                    "abi": _require_nonempty(
                        record.get("abi"),
                        field=f"entry[{role}].observed[{index}].abi",
                    ),
                    "calls": int(record.get("calls")),
                }
            )
            record_result = observed[-1]
            if record_result["kernel_symbol"] is not None:
                record_result["kernel_symbol"] = _require_nonempty(
                    record_result["kernel_symbol"],
                    field=f"entry[{role}].observed[{index}].kernel_symbol",
                )
            if (
                record_result["rows"] <= 0
                or record_result["in_features"] <= 0
                or record_result["out_features"] <= 0
                or record_result["calls"] <= 0
            ):
                raise ValueError(
                    f"batch manifest entry[{role}].observed[{index}] geometry and "
                    "call count must be positive"
                )
        result["observed"] = observed
        raw_rows = value.get("rows")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise ValueError(f"batch manifest entry[{role}].rows must be a list")
        result["rows"] = sorted({int(row) for row in raw_rows})
        raw_required = value.get("required_rows")
        if not isinstance(raw_required, Sequence) or isinstance(
            raw_required, (str, bytes)
        ):
            raise ValueError(
                f"batch manifest entry[{role}].required_rows must be a list"
            )
        result["required_rows"] = sorted({int(row) for row in raw_required})
    else:
        if "observed" in value or "rows" in value or "required_rows" in value:
            raise ValueError(
                f"batch manifest entry[{role}] is not a registry dispatch but "
                "carries dispatch records"
            )
    unknown = set(value) - set(result)
    if unknown:
        raise ValueError(f"batch manifest entry[{role}] has unknown fields: {sorted(unknown)}")
    return result


def normalize_batch_variant_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate structure and return a canonical manifest without its hash."""

    if not isinstance(payload, Mapping):
        raise ValueError("batch variant manifest must be a mapping")
    required = {
        "kind",
        "schema_version",
        "backend",
        "model",
        "quant",
        "execution_profile",
        "profile_manifest_sha256",
        "source_revision",
        "prefill_source",
        "decode_scope",
        "runtime",
        "runtime_config",
        "required_roles",
        "entries",
        "unbound",
        "not_used",
        "observed_step_widths",
    }
    allowed = required | {"manifest_sha256"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"batch variant manifest missing fields: {sorted(missing)}")
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"batch variant manifest has unknown fields: {sorted(unknown)}")
    if payload.get("kind") != BATCH_VARIANT_MANIFEST_KIND:
        raise ValueError(
            f"batch variant manifest kind must be {BATCH_VARIANT_MANIFEST_KIND!r}"
        )
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != BATCH_VARIANT_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            "batch variant manifest schema_version must be "
            f"{BATCH_VARIANT_MANIFEST_SCHEMA_VERSION}"
        )
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("batch variant manifest runtime must be a mapping")
    runtime_result = {
        "module": _require_nonempty(runtime.get("module"), field="runtime.module"),
        "class": _require_nonempty(runtime.get("class"), field="runtime.class"),
        "identity": _require_nonempty(
            runtime.get("identity"), field="runtime.identity"
        ),
        "source": _require_nonempty(runtime.get("source"), field="runtime.source"),
        "source_sha256": _require_sha256(
            runtime.get("source_sha256"), field="runtime.source_sha256"
        ),
        "function_sha256": _require_sha256(
            runtime.get("function_sha256"), field="runtime.function_sha256"
        ),
        "build_identity": _require_sha256(
            runtime.get("build_identity"), field="runtime.build_identity"
        ),
    }
    runtime_unknown = set(runtime) - set(runtime_result)
    if runtime_unknown:
        raise ValueError(
            f"batch variant manifest runtime has unknown fields: {sorted(runtime_unknown)}"
        )
    raw_config = payload.get("runtime_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("batch variant manifest runtime_config must be a mapping")
    config_result = {
        "hidden_size": int(raw_config.get("hidden_size", 0)),
        "context_length": int(raw_config.get("context_length", 0)),
        "qsa_compression_ratio": int(raw_config.get("qsa_compression_ratio", 0)),
        "layer_types": [str(kind) for kind in raw_config.get("layer_types", [])],
        "gdn_layer_count": int(raw_config.get("gdn_layer_count", 0)),
        "qsa_layer_count": int(raw_config.get("qsa_layer_count", 0)),
        "ple_layers": sorted(int(layer) for layer in raw_config.get("ple_layers", [])),
    }
    config_unknown = set(raw_config) - set(config_result)
    if config_unknown:
        raise ValueError(
            "batch variant manifest runtime_config has unknown fields: "
            f"{sorted(config_unknown)}"
        )
    raw_roles = payload.get("required_roles")
    if not isinstance(raw_roles, Sequence) or isinstance(raw_roles, (str, bytes)):
        raise ValueError("batch variant manifest required_roles must be a list")
    required_roles = [_require_nonempty(role, field="required_roles") for role in raw_roles]
    if len(set(required_roles)) != len(required_roles):
        raise ValueError("batch variant manifest required_roles has duplicates")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise ValueError("batch variant manifest entries must be a list")
    entries = [_normalize_entry(entry) for entry in raw_entries]
    if not entries:
        raise ValueError("batch variant manifest entries must not be empty")
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        key = (entry["role"], entry["binding"], entry["identity"])
        if key in seen:
            raise ValueError(f"batch variant manifest has duplicate entry: {key!r}")
        seen.add(key)
    entries.sort(key=lambda entry: (entry["role"], entry["binding"], entry["identity"]))

    def _normalize_role_list(value: Any, *, field: str) -> list[dict[str, str]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"batch variant manifest {field} must be a list")
        result = []
        for item in value:
            if not isinstance(item, Mapping):
                raise ValueError(f"batch variant manifest {field} item must be a mapping")
            result.append(
                {
                    "role": _require_nonempty(item.get("role"), field=f"{field}.role"),
                    "reason": _require_nonempty(
                        item.get("reason"), field=f"{field}.reason"
                    ),
                }
            )
        return result

    unbound = _normalize_role_list(payload.get("unbound"), field="unbound")
    not_used = _normalize_role_list(payload.get("not_used"), field="not_used")
    raw_widths = payload.get("observed_step_widths")
    if not isinstance(raw_widths, Mapping):
        raise ValueError("batch variant manifest observed_step_widths must be a mapping")
    widths: dict[str, int] = {}
    for raw_width, raw_count in raw_widths.items():
        width = int(raw_width)
        count = int(raw_count)
        if width <= 0 or count <= 0:
            raise ValueError("batch variant manifest step widths and counts must be positive")
        widths[str(width)] = count
    normalized = {
        "kind": BATCH_VARIANT_MANIFEST_KIND,
        "schema_version": BATCH_VARIANT_MANIFEST_SCHEMA_VERSION,
        "backend": _require_nonempty(payload.get("backend"), field="backend"),
        "model": _require_nonempty(payload.get("model"), field="model"),
        "quant": _require_nonempty(payload.get("quant"), field="quant"),
        "execution_profile": _require_nonempty(
            payload.get("execution_profile"), field="execution_profile"
        ),
        "profile_manifest_sha256": _require_sha256(
            payload.get("profile_manifest_sha256"), field="profile_manifest_sha256"
        ),
        "source_revision": _require_nonempty(
            payload.get("source_revision"), field="source_revision"
        ),
        "prefill_source": _require_nonempty(
            payload.get("prefill_source"), field="prefill_source"
        ),
        "decode_scope": _require_nonempty(
            payload.get("decode_scope"), field="decode_scope"
        ),
        "runtime": runtime_result,
        "runtime_config": config_result,
        "required_roles": required_roles,
        "entries": entries,
        "unbound": unbound,
        "not_used": not_used,
        "observed_step_widths": {key: widths[key] for key in sorted(widths, key=int)},
    }
    return normalized


def batch_variant_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 over the canonical manifest body."""

    return _sha256_json(normalize_batch_variant_manifest(payload))


# -- validation -------------------------------------------------------------


def _check_direct_reference(
    reference: Mapping[str, Any] | None,
    *,
    field: str,
    known: Mapping[str, Callable[..., Any]],
    reasons: list[str],
) -> None:
    if reference is None:
        return
    fn = known.get(reference["identity"])
    if fn is None:
        reasons.append(f"{field}_identity_not_live:{reference['identity']}")
        return
    if reference["source_sha256"] != source_file_sha256(fn):
        reasons.append(f"{field}_source_hash_mismatch:{reference['identity']}")
    if reference["function_sha256"] != function_source_sha256(fn):
        reasons.append(f"{field}_function_hash_mismatch:{reference['identity']}")
    expected = function_build_identity(fn, kernel_symbol=kernel_symbol(fn))
    if reference["build_identity"] != expected:
        reasons.append(f"{field}_build_identity_mismatch:{reference['identity']}")


def _check_registry_reference(
    reference: Mapping[str, Any] | None,
    *,
    field: str,
    reasons: list[str],
) -> None:
    if reference is None:
        return
    if reference["binding"] != "registry" or reference["registry_key"] is None:
        reasons.append(f"{field}_must_be_registry_reference")
        return
    key = _key_from_payload(reference["registry_key"])
    try:
        fn = _resolve_key(key)
    except ValueError as exc:
        reasons.append(f"{field}_unresolved:{exc}")
        return
    if reference["identity"] != function_identity(fn):
        reasons.append(f"{field}_identity_mismatch:{key.display()}")
    if reference["source_sha256"] != source_file_sha256(fn):
        reasons.append(f"{field}_source_hash_mismatch:{key.display()}")
    if reference["function_sha256"] != function_source_sha256(fn):
        reasons.append(f"{field}_function_hash_mismatch:{key.display()}")
    expected = function_build_identity(fn, kernel_symbol=kernel_symbol(fn))
    if reference["build_identity"] != expected:
        reasons.append(f"{field}_build_identity_mismatch:{key.display()}")


def validate_batch_variant_manifest(
    payload: Mapping[str, Any],
    *,
    expected_profile_manifest_sha256: str | None = None,
    expected_source_revision: str | None = None,
    expected_backend: str | None = None,
    expected_model: str | None = None,
    expected_quant: str | None = None,
) -> dict[str, Any]:
    """Validate a batch manifest and return a structured binding result.

    Every direct identity, registry resolution, runtime class, and projection
    dispatch key is recomputed from the current process. A manifest that cannot
    be reproduced from the live runtime is not evidence.
    """

    reasons: list[str] = []
    missing: list[dict[str, Any]] = []
    try:
        normalized = normalize_batch_variant_manifest(payload)
    except ValueError as exc:
        return {
            "valid": False,
            "bound": False,
            "reasons": [str(exc)],
            "missing": [],
            "entries": [],
            "unbound": [],
            "not_used": [],
            "manifest_sha256": None,
            "profile_manifest_sha256": None,
            "source_revision": None,
            "projection_dispatch": None,
        }
    expected_hash = batch_variant_manifest_sha256(normalized)
    actual_hash = payload.get("manifest_sha256")
    if actual_hash != expected_hash:
        reasons.append("manifest_sha256_mismatch")
    if (
        expected_profile_manifest_sha256 is not None
        and normalized["profile_manifest_sha256"] != expected_profile_manifest_sha256
    ):
        reasons.append("profile_manifest_sha256_mismatch")
    if (
        expected_source_revision is not None
        and normalized["source_revision"] != expected_source_revision
    ):
        reasons.append("source_revision_mismatch")
    if expected_backend is not None and normalized["backend"] != expected_backend:
        reasons.append("backend_mismatch")
    if expected_model is not None and normalized["model"] != expected_model:
        reasons.append("model_mismatch")
    if expected_quant is not None and normalized["quant"] != expected_quant:
        reasons.append("quant_mismatch")

    entries_by_role: dict[str, list[dict[str, Any]]] = {}
    for entry in normalized["entries"]:
        if entry["role"] in entries_by_role:
            reasons.append(f"duplicate_role_entry:{entry['role']}")
        entries_by_role.setdefault(entry["role"], []).append(entry)
    unbound_roles = {row["role"] for row in normalized["unbound"]}
    for role in normalized["required_roles"]:
        if role not in entries_by_role:
            missing.append(
                {
                    "role": role,
                    "identity": None,
                    "reason": "required_role_missing",
                }
            )
            reasons.append(f"required_role_missing:{role}")
        elif role in unbound_roles:
            missing.append(
                {
                    "role": role,
                    "identity": None,
                    "reason": "required_role_unbound",
                }
            )
            reasons.append(f"required_role_unbound:{role}")

    live = live_batch_direct_calls()
    oracles = live_batch_oracle_calls()
    known: dict[str, Callable[..., Any]] = {
        function_identity(fn): fn for fn in (*live.values(), *oracles.values())
    }
    for entry in normalized["entries"]:
        role = entry["role"]
        binding = entry["binding"]
        if binding == "direct":
            fn = live.get(role)
            if fn is None:
                reasons.append(f"direct_role_not_live:{role}")
                continue
            if entry["identity"] != function_identity(fn):
                reasons.append(f"direct_identity_mismatch:{role}")
            if entry["source_sha256"] != source_file_sha256(fn):
                reasons.append(f"direct_source_hash_mismatch:{role}")
            if entry["function_sha256"] != function_source_sha256(fn):
                reasons.append(f"direct_function_hash_mismatch:{role}")
            expected = function_build_identity(fn, kernel_symbol=kernel_symbol(fn))
            if entry["build_identity"] != expected:
                reasons.append(f"direct_build_identity_mismatch:{role}")
            if entry["registry_key"] is not None:
                reasons.append(f"direct_entry_claims_registry_key:{role}")
            for raw_registered_key in entry.get("registered_keys", []):
                registered_key = _key_from_payload(raw_registered_key)
                try:
                    registered_fn = _resolve_key(registered_key)
                except ValueError as exc:
                    reasons.append(
                        f"direct_registered_key_unresolved:{role}:{exc}"
                    )
                    continue
                if registered_fn is not fn:
                    reasons.append(
                        "direct_registered_key_identity_mismatch:"
                        f"{role}:{registered_key.display()}"
                    )
            for field in ("strict_fallback", "oracle"):
                _check_direct_reference(
                    entry.get(field), field=f"{role}.{field}", known=known, reasons=reasons
                )
        elif binding == "registry":
            if role in live:
                reasons.append(f"registry_entry_for_direct_role:{role}")
            if entry["registry_key"] is None:
                reasons.append(f"registry_entry_missing_key:{role}")
                continue
            key = _key_from_payload(entry["registry_key"])
            try:
                fn = _resolve_key(key)
            except ValueError as exc:
                reasons.append(f"registry_key_unresolved:{role}:{exc}")
                continue
            if entry["identity"] != function_identity(fn):
                reasons.append(f"registry_identity_mismatch:{role}")
            if entry["source_sha256"] != source_file_sha256(fn):
                reasons.append(f"registry_source_hash_mismatch:{role}")
            if entry["function_sha256"] != function_source_sha256(fn):
                reasons.append(f"registry_function_hash_mismatch:{role}")
            expected = function_build_identity(fn, kernel_symbol=kernel_symbol(fn))
            if entry["build_identity"] != expected:
                reasons.append(f"registry_build_identity_mismatch:{role}")
            if entry["strict_fallback"] is None and role in normalized["required_roles"]:
                reasons.append(f"registry_entry_missing_strict_fallback:{role}")
            _check_registry_reference(
                entry.get("strict_fallback"),
                field=f"{role}.strict_fallback",
                reasons=reasons,
            )
        elif binding == "registry_dispatch":
            resolver = function_identity(launch_gguf_linear)
            if entry.get("resolver") != resolver:
                reasons.append(f"projection_resolver_mismatch:{role}")
            if entry["identity"] != resolver:
                reasons.append(f"projection_identity_mismatch:{role}")
            if entry["source_sha256"] != source_file_sha256(launch_gguf_linear):
                reasons.append("projection_resolver_source_hash_mismatch")
            if entry["function_sha256"] != function_source_sha256(launch_gguf_linear):
                reasons.append("projection_resolver_function_hash_mismatch")
            observed = entry.get("observed", [])
            if not observed:
                reasons.append("projection_dispatch_has_no_records")
            for index, record in enumerate(observed):
                key = _key_from_payload(record["key"])
                try:
                    fn = _resolve_key(key)
                except ValueError as exc:
                    reasons.append(
                        f"projection_key_unresolved:{index}:{exc}"
                    )
                    continue
                if record["identity"] != function_identity(fn):
                    reasons.append(f"projection_identity_mismatch:{index}")
                if record["source_sha256"] != source_file_sha256(fn):
                    reasons.append(f"projection_source_hash_mismatch:{index}")
                if record["function_sha256"] != function_source_sha256(fn):
                    reasons.append(f"projection_function_hash_mismatch:{index}")
                expected = function_build_identity(fn, kernel_symbol=kernel_symbol(fn))
                if record["build_identity"] != expected:
                    reasons.append(f"projection_build_identity_mismatch:{index}")
                if (
                    record["activation_dtype"] != "f32"
                    or record["output_dtype"] != "f32"
                ):
                    reasons.append(f"projection_dtype_not_f32:{index}")
            observed_rows = {int(row["rows"]) for row in observed}
            required_rows = {int(row) for row in entry.get("required_rows", [])}
            if required_rows - observed_rows:
                reasons.append(
                    "projection_dispatch_missing_rows:"
                    f"{sorted(required_rows - observed_rows)}"
                )
            if {int(row) for row in entry.get("rows", [])} != observed_rows:
                reasons.append("projection_dispatch_rows_do_not_match_records")
            step_rows = {int(width) for width in normalized["observed_step_widths"]}
            if step_rows - observed_rows:
                reasons.append(
                    "projection_dispatch_missing_step_widths:"
                    f"{sorted(step_rows - observed_rows)}"
                )

    # The manifest may not silently omit a core role from its own requirement
    # list, and PLE roles are required whenever the runtime config has PLE layers.
    for role in BATCH_CORE_REQUIRED_ROLES:
        if role not in normalized["required_roles"]:
            reasons.append(f"required_roles_omits_core:{role}")
    if normalized["runtime_config"]["ple_layers"]:
        for role in BATCH_PLE_REQUIRED_ROLES:
            if role not in normalized["required_roles"]:
                reasons.append(f"required_roles_omits_ple:{role}")

    expected_runtime = _runtime_identity(Qwen4ExpGGUFBatchRunner)
    if normalized["runtime"]["identity"] != expected_runtime["identity"]:
        reasons.append("runtime_identity_mismatch")
    if normalized["runtime"]["source_sha256"] != expected_runtime["source_sha256"]:
        reasons.append("runtime_source_hash_mismatch")

    projection_entry = next(
        (entry for entry in normalized["entries"] if entry["role"] == "projection_dispatch"),
        None,
    )
    result = {
        "valid": not reasons,
        "bound": not reasons,
        "reasons": reasons,
        "missing": missing,
        "entries": normalized["entries"],
        "unbound": normalized["unbound"],
        "not_used": normalized["not_used"],
        "manifest_sha256": expected_hash,
        "profile_manifest_sha256": normalized["profile_manifest_sha256"],
        "source_revision": normalized["source_revision"],
        "projection_dispatch": projection_entry,
        "required_roles": normalized["required_roles"],
        "observed_step_widths": normalized["observed_step_widths"],
    }
    return result


__all__ = [
    "BATCH_CORE_REQUIRED_ROLES",
    "BATCH_DECODE_SCOPE",
    "BATCH_PLE_REQUIRED_ROLES",
    "BATCH_PREFILL_SOURCE",
    "BATCH_VARIANT_MANIFEST_KIND",
    "BATCH_VARIANT_MANIFEST_SCHEMA_VERSION",
    "batch_variant_manifest_sha256",
    "build_qwen4_exp_batch_variant_manifest",
    "function_build_identity",
    "function_identity",
    "function_source_sha256",
    "kernel_symbol",
    "live_batch_direct_calls",
    "live_batch_oracle_calls",
    "normalize_batch_variant_manifest",
    "source_file_sha256",
    "source_path",
    "validate_batch_variant_manifest",
]
