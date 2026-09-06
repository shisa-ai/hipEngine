"""Pure GGUF dense-weight policy shared by the runtime loader and the audit.

Cold-path only: no device, no HIP runtime, no kernel-package import, no
``backend == ...``/``quant == ...`` dispatch. Runtime callers pass
``backend_package_capability`` as the capability reader (which may import a
backend package inside the loader process); metadata audits pass a
source-reading reader so no backend package is ever imported. Both callers get
identical flag resolution from :func:`resolve_gguf_dense_flags`, which is the
single home of the backend-capability and environment-override semantics that
``materialize_qwen35_gguf_weights`` uses.

The raw-IQ predicate :func:`gguf_ar_raw_iq_contract` is the production
``contract_q3_f32_linear`` predicate from ``plan_qwen35_gguf_materialization``:
it both vetoes decode repack for raw-IQ AR layers and contracts those files'
F32 alpha/beta/router linear slots to BF16. Sharing it here keeps the quant-route
audit from growing a regex/string mirror of production policy
(``docs/REFACTOR.md`` 2026-09-06 entry).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any

from hipengine.quant.gguf import GGMLQuantizationType

# A capability reader resolves one backend-package constant by name. Runtime:
# hipengine.kernels.backends.backend_package_capability (imports the package).
# Metadata audit: a source-reading reader that never imports it.
CapabilityReader = Callable[[str, str, Any], Any]

# Every backend-package constant the dense GGUF weight policy consumes, plus
# the two environment overrides. The audit reports per-capability resolution
# status for exactly these names.
GGUF_DENSE_CAPABILITY_NAMES = (
    "GGUF_DENSE_Q4_T16",
    "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP",
    "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES",
    "GGUF_DENSE_Q4_T16_ATTN_Q_08B",
    "GGUF_DENSE_Q5_T16_SSM_OUT",
    "GGUF_C8_Q5_RAW_MMQ_SSM_OUT",
    "GGUF_DENSE_Q5_T16_SSM_OUT_08B",
    "GGUF_DENSE_Q5_T16_QKV",
    "GGUF_DENSE_Q5_T16_H5120",
    "GGUF_DENSE_Q6_T16_QMICRO_PLANAR",
    "GGUF_DENSE_Q6_T16_QMICRO_PLANAR_EXCLUDED_SLOTS",
    "GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES",
)

HIPENGINE_GGUF_C8_Q5_RAW_MMQ_ENV = "HIPENGINE_GGUF_C8_Q5_RAW_MMQ"
HIPENGINE_C8_Q5_PLANAR_DP4A_ENV = "HIPENGINE_C8_Q5_PLANAR_DP4A"

# GGML type ids whose raw storage triggers the AR raw-IQ contract (decode-repack
# veto plus model-wide F32 linear contraction) in plan_qwen35_gguf_materialization.
_AR_RAW_IQ_GGML_TYPE_IDS = frozenset(
    int(value)
    for value in (
        GGMLQuantizationType.IQ2_XS,
        GGMLQuantizationType.IQ3_XXS,
        GGMLQuantizationType.IQ4_XS,
    )
)

_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def gguf_ar_raw_iq_contract(ggml_type_ids: Iterable[int]) -> bool:
    """Return the production AR raw-IQ contract predicate for AR layer types.

    True when any AR layer tensor stores one of the raw-IQ types. The caller
    passes ``tensor.ggml_type`` for the AR layers only (root and draft tensors
    never trigger this contract in the production planner).
    """

    return any(int(ggml_type) in _AR_RAW_IQ_GGML_TYPE_IDS for ggml_type in ggml_type_ids)


def gguf_ar_decode_repack_veto(ggml_type_ids: Iterable[int]) -> bool:
    """Return the per-tensor decode-repack veto for one AR type scope.

    UD-U1 policy knob, separated from the F32 linear contraction so a future
    per-tensor repack-eligibility change (UD-U3 layout selection) cannot
    silently move the model-wide F32 contraction, and vice versa. Today both
    knobs derive from the same raw-IQ predicate, so every unchanged manifest
    plans identically.
    """

    return gguf_ar_raw_iq_contract(ggml_type_ids)


def gguf_ar_f32_linear_contraction(ggml_type_ids: Iterable[int]) -> bool:
    """Return the model-wide F32 alpha/beta/router linear contraction gate.

    UD-U1 policy knob, separated from :func:`gguf_ar_decode_repack_veto` for
    the same reason; the defaults remain identical for every current manifest.
    """

    return gguf_ar_raw_iq_contract(ggml_type_ids)


def _env_enabled(environ: Any, name: str, default: str) -> bool:
    raw = environ.get(name)
    if raw is None:
        raw = default
    return raw.strip().lower() in _TRUE_ENV_VALUES


def _file_type_set(value: Any) -> frozenset[str]:
    if isinstance(value, (tuple, list, set, frozenset)):
        return frozenset(str(item) for item in value)
    return frozenset()


def resolve_gguf_dense_flags(
    backend: str,
    file_type_name: str | None,
    *,
    capability_reader: CapabilityReader,
    environ: Any = None,
) -> dict[str, Any]:
    """Resolve the dense-weight planner flags for one backend and file type.

    Returns exactly the keyword arguments (minus ``decode_repack``) accepted by
    ``plan_qwen35_gguf_materialization`` and ``plan_qwen35_gguf_weight_spec``.
    Missing capabilities resolve to the same defaults the runtime reader uses
    (bools False, containers empty). Environment overrides:

    - ``HIPENGINE_GGUF_C8_Q5_RAW_MMQ`` (default on) gates the raw-MMQ Q5
      sidecar on the ``GGUF_C8_Q5_RAW_MMQ_SSM_OUT`` capability;
    - ``HIPENGINE_C8_Q5_PLANAR_DP4A`` (default off) gates the optional planar
      Q5 sidecar on the same capability.

    ``GGUF_DENSE_Q4_QMICRO_T16_GATE_UP`` is additionally gated on the file-type
    stamp via ``GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES`` (membership is
    case-sensitive, matching the runtime loader).
    """

    env = os.environ if environ is None else environ
    q5_raw_mmq_capable = bool(capability_reader(backend, "GGUF_C8_Q5_RAW_MMQ_SSM_OUT", False))
    qmicro_types = _file_type_set(
        capability_reader(backend, "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES", ())
    )
    q6_excluded_raw = capability_reader(
        backend, "GGUF_DENSE_Q6_T16_QMICRO_PLANAR_EXCLUDED_SLOTS", ()
    )
    q6_excluded = (
        tuple(str(slot) for slot in q6_excluded_raw)
        if isinstance(q6_excluded_raw, (tuple, list, set, frozenset))
        else ()
    )
    return {
        "dense_q4_t16": bool(capability_reader(backend, "GGUF_DENSE_Q4_T16", False)),
        "dense_q4_qmicro_t16_gate_up": (
            bool(capability_reader(backend, "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP", False))
            and file_type_name in qmicro_types
        ),
        "dense_q4_t16_attn_q_08b": bool(
            capability_reader(backend, "GGUF_DENSE_Q4_T16_ATTN_Q_08B", False)
        ),
        "dense_q5_t16_ssm_out": bool(capability_reader(backend, "GGUF_DENSE_Q5_T16_SSM_OUT", False)),
        "dense_q5_raw_mmq_ssm_out": (
            _env_enabled(env, HIPENGINE_GGUF_C8_Q5_RAW_MMQ_ENV, "1") and q5_raw_mmq_capable
        ),
        "dense_q5_qmicro_planar_ssm_out": (
            _env_enabled(env, HIPENGINE_C8_Q5_PLANAR_DP4A_ENV, "0") and q5_raw_mmq_capable
        ),
        "dense_q5_t16_ssm_out_08b": bool(
            capability_reader(backend, "GGUF_DENSE_Q5_T16_SSM_OUT_08B", False)
        ),
        "dense_q5_t16_qkv": bool(capability_reader(backend, "GGUF_DENSE_Q5_T16_QKV", False)),
        "dense_q5_t16_h5120": bool(capability_reader(backend, "GGUF_DENSE_Q5_T16_H5120", False)),
        "dense_q6_qmicro_planar": bool(
            capability_reader(backend, "GGUF_DENSE_Q6_T16_QMICRO_PLANAR", False)
        ),
        "dense_q6_qmicro_planar_excluded_slots": q6_excluded,
    }


def gguf_fp16_recurrent_state_default(
    backend: str | None,
    file_type_name: str | None,
    *,
    capability_reader: CapabilityReader,
) -> bool:
    """Mirror the runner's file-type-stamp default for FP16 recurrent state.

    An absent capability means the backend declares no such default (False).
    Comparison normalizes case/whitespace exactly like the runtime runner.
    Runner-level environment overrides live in the runner and are not part of
    this backend default.
    """

    if backend is None or file_type_name is None:
        return False
    defaults = capability_reader(backend, "GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES", ())
    normalized = {
        str(value).strip().lower() for value in defaults if str(value).strip()
    }
    return str(file_type_name).strip().lower() in normalized
