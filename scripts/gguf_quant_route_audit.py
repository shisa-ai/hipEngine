#!/usr/bin/env python3
"""Report which GGUF quant types a Qwen dense GGUF file needs and how they load.

Answers one question per file: after hipEngine validates the tensor map and runs
each tensor through the production weight planner, which tensors get a real
quantized kernel, which get expanded to dense BF16 at load, and which make the
loader refuse the file. Reads metadata and the tensor-info table only; no weights
are read, so a partially downloaded file can still be inspected. Nothing here runs
a model, allocates device memory, or touches a GPU.

Route tables come from the actual production tensor maps: the AR map built by
``hipengine.loading.qwen35_gguf.build_qwen35_gguf_tensor_map`` (root slots such
as ``root.token_embedding`` plus per-layer ``layers.<id>.<slot>`` consumers) and
the separate runtime NextN map built by
``hipengine.loading.qwen35_gguf_nextn.build_qwen35_gguf_nextn_tensor_map``
(``draft.layer.<slot>`` / ``draft.nextn.<slot>`` consumers, with fallback slots
resolved to AR roots). AR-excluded trailing MTP block tensors are never counted
in AR routes. Reports distinguish logical consumer slots from unique physical
source tensors, dedup shared sources by source identity, and list explicit
aliases; source deduplication is not resident-allocation deduplication (one
source can plan different layouts for different consumers). When the model map
cannot be built the audit stays parser-only: no route table is produced from
guessed slot names, and no output ever claims the file is loadable or
consumer-qualified -- map availability and route verdicts are diagnostics, not
consumer qualification.

Parser validation is two-tier. Complete files are admitted by production
``hipengine.loading.gguf.scan_gguf`` (supported version from
GGUF_SUPPORTED_VERSIONS, unique metadata keys and tensor names, non-zero
power-of-two alignment, tensor byte ranges within the file); the report marks
them format-valid. Format-valid is format-layer admission only, not a claim
that the model loads: the plugin tensor map and per-backend planner verdicts
are reported separately and may still refuse. Files production refuses fall
into an explicit partial diagnostic mode: a bounded mirror of the same checks
records the defect and stops -- an unsupported or unreadable version is not
further interpreted as a known layout -- while known-version layouts keep
parsing after each recoverable failure so one pass reports unsupported
versions, duplicate metadata keys or tensor names, invalid alignment, and
incomplete descriptor table versus incomplete payload as distinct statuses.
Hostile metadata values (a string or array ``general.alignment``, an unknown
metadata value-type id, a tensor shape violating its block layout) refuse
production through plain ValueError/TypeError and are diagnosed the same way.
Refused files are never marked format-valid or loadable.

Backend capabilities are read from kernel package source with a bounded AST
literal reader (never an import, never expression evaluation); missing or
nonliteral constants are reported per capability and resolve to the same
defaults the runtime reader returns. Flag resolution itself goes through the
shared pure policy API ``hipengine.loading.qwen35_gguf_policy`` -- the same
function the runtime loader calls with ``backend_package_capability`` -- so the
audit cannot drift from runtime policy. Per-scope allocation sections report
requested weight bytes from the production allocation formula
(``planned_qwen35_gguf_weight_allocation_nbytes``), sidecar allocations with
reasons, and both hypothetical refusal treatments (compressed-source lower
bound and BF16 expansion scenario); refusals are never silently omitted from
the totals. Report ``schema_version`` 2 adds header identity (SHA256 over the
[0, data_start) header region), the allocation sections, per-capability
resolution status, and ``f32_contracted_slots``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from math import prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipengine.loading.gguf import (  # noqa: E402
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUF_SUPPORTED_VERSIONS,
    GGUFFormatError,
    GGUFModelInfo,
    GGUFTensorInfo,
    _align_up,
    _read_exact,
    _read_scalar,
    _read_string,
    _read_value,
    scan_gguf,
)
from hipengine.loading.qwen35_gguf import (  # noqa: E402
    Qwen35GGUFModelMap,
    build_qwen35_gguf_tensor_map,
)
from hipengine.loading.qwen35_gguf_nextn import (  # noqa: E402
    Qwen35GGUFNextNMap,
    build_qwen35_gguf_nextn_tensor_map,
)
from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q4_K_QMICRO_T16,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q4_K_X8,
    LAYOUT_GGUF_Q5_K_QMICRO_T16,
    LAYOUT_GGUF_Q5_K_T16,
    LAYOUT_GGUF_Q5_K_X8,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
    LAYOUT_GGUF_Q6_K_X8,
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    gguf_decode_repack_enabled,
    plan_qwen35_gguf_materialization,
    plan_qwen35_gguf_weight_spec,
    planned_qwen35_gguf_weight_allocation_nbytes,
)
from hipengine.loading.qwen35_gguf_policy import (  # noqa: E402
    GGUF_DENSE_CAPABILITY_NAMES,
    gguf_ar_raw_iq_contract,
    gguf_fp16_recurrent_state_default,
    resolve_gguf_dense_flags,
)
from hipengine.quant.gguf import (  # noqa: E402
    GGMLQuantizationType,
    GGUFValueType,
    ggml_type,
    ggml_type_name,
    llama_file_type_name,
    nbytes_for_shape,
    quant_shape_to_byte_shape,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# hipengine.kernels.<backend> holds these as plain module constants. Importing
# the package runs kernel registration, which needs the HIP runtime, so the
# values are read from source with a bounded AST literal reader. A constant
# that is absent or defined by a nonliteral expression is never guessed: it is
# reported with its resolution status and resolves to the same default the
# runtime reader returns for a missing attribute. Flag resolution itself goes
# through the shared pure policy API, so the audit carries no policy mirror.
_CAPABILITY_BACKENDS = ("hip_gfx1100", "hip_gfx1151")
_LITERAL_CALL_WRAPPERS = {
    "frozenset": frozenset,
    "set": set,
    "tuple": tuple,
    "list": list,
}


def _literal_value(node: ast.expr) -> tuple[object, str]:
    """Resolve a module-level assignment expression, bounded to literals.

    Accepts ``ast.literal_eval`` forms plus the container-wrapper calls the
    backend packages actually use (``frozenset({...})``, ``set(...)``,
    ``tuple(...)``, ``list(...)``) around a single literal argument. Anything
    else is "nonliteral": the value is never guessed and never evaluated.
    """

    try:
        return ast.literal_eval(node), "literal"
    except (ValueError, TypeError):
        pass
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _LITERAL_CALL_WRAPPERS
        and len(node.args) == 1
        and not node.keywords
    ):
        try:
            inner = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            return None, "nonliteral"
        return _LITERAL_CALL_WRAPPERS[node.func.id](inner), "literal"
    return None, "nonliteral"


def backend_source_assignments(root: Path, backend: str) -> dict[str, tuple[object, str]]:
    """Map module-level assignments of one backend ``__init__.py`` to values.

    Returns ``{constant name: (value, status)}`` where status is ``"literal"``
    when the assigned expression resolved through :func:`_literal_value`, and
    ``"nonliteral"`` when it did not (the value is then unusable, never
    guessed). Absent names are simply not in the mapping. The reader never
    imports the package and never evaluates expressions.
    """

    source = (root / "hipengine" / "kernels" / backend / "__init__.py").read_text()
    assignments: dict[str, tuple[object, str]] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        value, status = _literal_value(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = (value, status)
    return assignments


def source_capability_reader(root: Path | None = None):
    """Capability reader for the shared policy API without backend imports.

    Missing or nonliteral constants return the caller's default -- exactly what
    the runtime reader returns for a missing attribute -- so an unreadable or
    refactored source degrades like an absent capability instead of inventing
    policy. ``capability_status`` reports what actually happened.
    """

    root = REPO_ROOT if root is None else root
    assignments = {
        backend: backend_source_assignments(root, backend) for backend in _CAPABILITY_BACKENDS
    }

    def reader(backend: str, name: str, default):
        entry = assignments.get(backend, {}).get(name)
        if entry is None or entry[1] != "literal":
            return default
        return entry[0]

    return reader


def capability_status(root: Path | None = None) -> dict[str, dict[str, str]]:
    """Per-capability resolution status for the shared policy capability names."""

    root = REPO_ROOT if root is None else root
    statuses: dict[str, dict[str, str]] = {}
    for backend in _CAPABILITY_BACKENDS:
        assignments = backend_source_assignments(root, backend)
        statuses[backend] = {
            name: assignments.get(name, (None, "missing"))[1]
            for name in GGUF_DENSE_CAPABILITY_NAMES
        }
    return statuses


LAYOUT_MEANING = {
    "dense_bf16": "bf16-expand",
    "dense_f32": "f32-resident",
    "raw_gguf": "raw-gguf-kernel",
}
GIB = 2**30

# Canonical order for collapsing a file's diagnostics into one status string.
_DIAGNOSTIC_CHECK_ORDER = (
    "unopenable",
    "gguf_magic",
    "unsupported_version",
    "unparseable",
    "duplicate_metadata",
    "duplicate_tensor",
    "invalid_alignment",
    "incomplete_table",
    "incomplete_data",
    "production_scan_rejected",
)

# Expected plugin-qualification failures when a file's model metadata or tensor
# set is incomplete or unrelated to the dense-Qwen plugin: missing qwen35 config
# keys, unexpected architecture, missing required tensors, unreadable backend
# capability sources. These sections are reported as diagnostic-only errors
# instead of aborting the audit. Anything outside this tuple -- programming
# errors -- must propagate.
_PLUGIN_QUALIFICATION_ERRORS = (KeyError, ValueError, TypeError, OSError)

# Read/decode failures possible while consuming GGUF structures: short reads
# (production _read_exact raises EOFError), bad UTF-8 in strings, invalid
# value-type ids and struct unpacking problems. Anything else is a programming
# error and propagates.
_READ_ERRORS = (EOFError, UnicodeDecodeError, ValueError, struct.error)

# Type/shape resolution failures for an already-parsed descriptor: unknown GGML
# type id (KeyError) and block-layout violations (ValueError).
_TYPE_SHAPE_ERRORS = (KeyError, ValueError)

# Refusal surfaces of production scan_gguf this audit must survive and
# diagnose. ValueError and TypeError are deliberate, not sloppy catch-alls:
# production refuses actual malformed inputs through them -- int() coercion of
# a string or array general.alignment, GGUFValueType() on an unknown metadata
# value-type id, and block-layout-violating tensor shapes in
# quant_shape_to_byte_shape. ZeroDivisionError is included deliberately:
# production currently raises it from _align_up for a zero alignment instead of
# a format error, and the diagnostic mode must still report that file.
_SCAN_REFUSAL_ERRORS = (
    GGUFFormatError,
    EOFError,
    KeyError,
    ValueError,
    TypeError,
    UnicodeDecodeError,
    OSError,
    ZeroDivisionError,
)


@dataclass(frozen=True)
class ParserValidation:
    """Format-layer admission verdict for one GGUF file.

    ``format_valid`` is true only when production ``scan_gguf`` accepts the
    file's header, metadata, tensor table, and declared byte ranges. It is a
    format-layer statement only: it does not mean the model loads. Consumer
    admission -- the plugin tensor map and per-backend weight planner -- is a
    separate question answered by the route sections, which are reported
    regardless and may still refuse tensors or require a complete model map.
    A file with an incomplete descriptor table or incomplete payload can still
    be inspected through ``diagnose_header``, but it is never format-valid.
    ``data_start`` is None when the payload boundary could not be established;
    ``table_complete`` and ``data_complete`` are True only when verified.
    """

    path: Path
    production_scan: str  # "accepted" | "rejected"
    production_scan_error: str | None
    version: int | None
    version_supported: bool
    metadata: dict
    tensors: list[GGUFTensorInfo]
    data_start: int | None
    declared_tensor_count: int
    table_complete: bool
    data_complete: bool
    format_valid: bool
    status: str
    diagnostics: tuple[dict, ...]

    def to_json(self) -> dict:
        return {
            "production_scan": self.production_scan,
            "production_scan_error": self.production_scan_error,
            "gguf_version": self.version,
            "version_supported": self.version_supported,
            "declared_tensor_count": self.declared_tensor_count,
            "parsed_tensor_count": len(self.tensors),
            "table_complete": self.table_complete,
            "data_complete": self.data_complete,
            "data_start_bytes": self.data_start,
            "format_valid": self.format_valid,
            "status": self.status,
            "diagnostics": list(self.diagnostics),
        }


def _status_from_diagnostics(diagnostics: list[dict]) -> str:
    if not diagnostics:
        return "complete"
    named = [d["check"] for d in diagnostics]
    ordered = [check for check in _DIAGNOSTIC_CHECK_ORDER if check in named]
    extras = sorted(set(named) - set(ordered))
    return "+".join(ordered + extras)


def diagnose_header(path: Path, production_scan_error: str | None = None) -> ParserValidation:
    """Parse a GGUF header fail-closed, recording diagnostics instead of raising.

    Mirrors the validation production ``scan_gguf`` performs for known-version
    layouts. Bounded by design: parsing stops at the first loss of structural
    boundary -- unreadable magic or version, an unsupported version whose body
    layout is unknown, truncated counts or metadata, a truncated descriptor
    table -- and anything that could not be established honestly is reported as
    unknown (``data_start`` None, completeness flags False, no tensor records)
    instead of invented valid-looking values. Within an intact boundary,
    duplicate metadata keys and tensor names keep the first occurrence and
    record a diagnostic. An invalid ``general.alignment`` leaves the payload
    boundary unknown: no data offsets, no tensor records, and no completeness
    verdict are reported for that file. ``table_complete`` and
    ``data_complete`` are True only when verified; a descriptor whose size
    cannot be resolved (unknown type id, block-layout-violating shape) keeps
    ``data_complete`` False even when the rest of the table parsed. The
    ``incomplete_table`` / ``incomplete_data`` diagnostics mark definitively
    short regions. Every diagnostic is fatal for format validity.
    """

    diagnostics: list[dict] = []

    def fail(check: str, detail: str) -> None:
        diagnostics.append({"check": check, "severity": "error", "detail": detail})

    metadata: dict = {}
    raw: list[tuple[str, tuple[int, ...], int, int]] = []
    tensors: list[GGUFTensorInfo] = []
    version: int | None = None
    version_supported = False
    declared = 0
    table_complete = False
    data_complete = False
    data_start: int | None = None
    try:
        file_size = path.stat().st_size
    except OSError as error:
        file_size = 0
        fail("unopenable", f"stat failed: {type(error).__name__}: {error}")

    try:
        handle = path.open("rb")
    except OSError as error:
        # A missing path already reported unopenable above; report a second
        # failure only for a path that stats but cannot be opened.
        if not any(d["check"] == "unopenable" for d in diagnostics):
            fail("unopenable", f"open failed: {type(error).__name__}: {error}")
    else:
        with handle:
            try:
                magic_ok = _read_exact(handle, 4) == GGUF_MAGIC
            except EOFError:
                magic_ok = False
                fail("gguf_magic", f"file too small for GGUF magic: {path}")
            if not magic_ok:
                if not any(d["check"] == "gguf_magic" for d in diagnostics):
                    fail("gguf_magic", f"not a GGUF file: {path}")
            else:
                try:
                    version = int(_read_scalar(handle, GGUFValueType.UINT32))
                except _READ_ERRORS as error:
                    fail("unparseable", f"could not read GGUF version: {type(error).__name__}: {error}")
                else:
                    version_supported = version in GGUF_SUPPORTED_VERSIONS
                    if not version_supported:
                        # An unsupported version's body layout is unknown; it is
                        # never parsed as if it were a known one.
                        supported = ", ".join(str(v) for v in GGUF_SUPPORTED_VERSIONS)
                        fail(
                            "unsupported_version",
                            f"GGUF version {version} is not supported; expected one of: {supported}",
                        )
                    else:
                        boundary_lost = False
                        try:
                            declared = int(_read_scalar(handle, GGUFValueType.UINT64))
                            metadata_count = int(_read_scalar(handle, GGUFValueType.UINT64))
                        except _READ_ERRORS as error:
                            boundary_lost = True
                            fail("unparseable", f"could not read GGUF counts: {type(error).__name__}: {error}")
                        if not boundary_lost:
                            for _ in range(metadata_count):
                                try:
                                    key = _read_string(handle)
                                    value_type = GGUFValueType(_read_scalar(handle, GGUFValueType.UINT32))
                                    value = _read_value(handle, value_type)
                                except _READ_ERRORS as error:
                                    boundary_lost = True
                                    fail(
                                        "unparseable",
                                        f"metadata region truncated after {len(metadata)} entries; "
                                        f"descriptor table boundary lost: {type(error).__name__}: {error}",
                                    )
                                    break
                                if key in metadata:
                                    fail(
                                        "duplicate_metadata",
                                        f"metadata key {key!r} appears more than once; kept first value "
                                        f"{metadata[key]!r}, ignored later value {value!r}",
                                    )
                                else:
                                    metadata[key] = value
                        if not boundary_lost:
                            seen: set[str] = set()
                            parsed_count = 0
                            for _ in range(declared):
                                try:
                                    name = _read_string(handle)
                                    n_dims = int(_read_scalar(handle, GGUFValueType.UINT32))
                                    ggml_shape = tuple(
                                        int(_read_scalar(handle, GGUFValueType.UINT64)) for _ in range(n_dims)
                                    )
                                    qtype_id = int(_read_scalar(handle, GGUFValueType.UINT32))
                                    offset = int(_read_scalar(handle, GGUFValueType.UINT64))
                                except _READ_ERRORS as error:
                                    fail(
                                        "incomplete_table",
                                        f"tensor-info table truncated after {parsed_count} of {declared} "
                                        f"descriptors; payload boundary lost: {type(error).__name__}: {error}",
                                    )
                                    break
                                parsed_count += 1
                                if name in seen:
                                    fail(
                                        "duplicate_tensor",
                                        f"tensor name {name!r} appears more than once; kept first "
                                        "descriptor, ignored later ones",
                                    )
                                    continue
                                seen.add(name)
                                raw.append((name, ggml_shape, qtype_id, offset))
                            table_complete = parsed_count == declared
                        if table_complete:
                            alignment_raw = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
                            try:
                                alignment = int(alignment_raw)
                            except (TypeError, ValueError):
                                alignment = 0
                            if alignment <= 0 or alignment & (alignment - 1):
                                # Fail-closed: without a valid alignment the payload
                                # boundary is unknown. No default-alignment substitute,
                                # no data offsets, no tensor records, no completeness.
                                fail(
                                    "invalid_alignment",
                                    f"general.alignment {alignment_raw!r} is not a non-zero power of "
                                    "two; the tensor payload boundary is unknown: no data offset, "
                                    "no tensor records, and no completeness verdict are reported",
                                )
                            else:
                                data_start = _align_up(handle.tell(), alignment)
                                unresolved = 0
                                for name, ggml_shape, qtype_id, offset in raw:
                                    shape = tuple(reversed(ggml_shape))
                                    try:
                                        qtype = ggml_type(qtype_id)
                                        nbytes = nbytes_for_shape(shape, qtype)
                                        byte_shape = quant_shape_to_byte_shape(shape, qtype)
                                    except _TYPE_SHAPE_ERRORS as error:
                                        unresolved += 1
                                        fail(
                                            "unparseable",
                                            f"tensor {name!r} has unusable type or shape (type id "
                                            f"{qtype_id}, shape {shape}): {type(error).__name__}: {error}; "
                                            "its payload size is unresolved, so data completeness "
                                            "cannot be verified",
                                        )
                                        continue
                                    tensors.append(
                                        GGUFTensorInfo(
                                            name=name,
                                            shape=shape,
                                            ggml_shape=ggml_shape,
                                            ggml_type=int(qtype),
                                            ggml_type_name=ggml_type_name(qtype),
                                            n_elements=int(prod(shape)),
                                            nbytes=nbytes,
                                            offset=offset,
                                            data_offset=data_start + offset,
                                            byte_shape=byte_shape,
                                        )
                                    )
                                # A dropped descriptor makes completeness unverifiable:
                                # an empty range loop must not report data_complete.
                                # The range loop still runs over the resolved tensors
                                # even when other descriptors are unresolved: an
                                # out-of-range known tensor is its own defect.
                                data_complete = unresolved == 0
                                incomplete: list[str] = []
                                for tensor in tensors:
                                    if (
                                        tensor.data_offset < data_start
                                        or tensor.data_offset + tensor.nbytes > file_size
                                    ):
                                        missing = max(0, tensor.data_offset + tensor.nbytes - file_size)
                                        incomplete.append(
                                            f"{tensor.name!r} needs [{tensor.data_offset}, "
                                            f"{tensor.data_offset + tensor.nbytes}) of {file_size} bytes "
                                            f"on disk ({missing} missing)"
                                        )
                                if incomplete:
                                    data_complete = False
                                    preview = "; ".join(incomplete[:4])
                                    more = "" if len(incomplete) <= 4 else f" (+{len(incomplete) - 4} more)"
                                    fail(
                                        "incomplete_data",
                                        f"tensor payload extends past end of file for {len(incomplete)} "
                                        f"tensor(s): {preview}{more}",
                                    )

    return ParserValidation(
        path=path,
        production_scan="rejected",
        production_scan_error=production_scan_error,
        version=version,
        version_supported=version_supported,
        metadata=metadata,
        tensors=tensors,
        data_start=data_start,
        declared_tensor_count=declared,
        table_complete=table_complete,
        data_complete=data_complete,
        format_valid=False,
        status=_status_from_diagnostics(diagnostics),
        diagnostics=tuple(diagnostics),
    )


def validate_parser(path: Path) -> ParserValidation:
    """Production scan first; structured diagnostics for the files it refuses."""

    try:
        info = scan_gguf(path)
    except _SCAN_REFUSAL_ERRORS as error:
        detail = f"{type(error).__name__}: {error}"
        parsed = diagnose_header(path, production_scan_error=detail)
        if not parsed.diagnostics:
            # The diagnostic mirror found no defect but production refused the
            # file (mirror drift). Production is the authority: stay refused.
            gap = {
                "check": "production_scan_rejected",
                "severity": "error",
                "detail": f"production scan_gguf refused the file but the diagnostic "
                f"mirror found no defect; report this mirror gap: {detail}",
            }
            parsed = replace(parsed, diagnostics=(gap,), status="production_scan_rejected")
        return parsed
    return ParserValidation(
        path=info.path,
        production_scan="accepted",
        production_scan_error=None,
        version=info.version,
        version_supported=info.version in GGUF_SUPPORTED_VERSIONS,
        metadata=dict(info.metadata),
        tensors=list(info.tensors),
        data_start=info.tensor_data_offset,
        declared_tensor_count=len(info.tensors),
        table_complete=True,
        data_complete=True,
        format_valid=True,
        status="complete",
        diagnostics=(),
    )


def read_header(path: Path) -> tuple[dict, list[GGUFTensorInfo], int | None, int]:
    """Backward-compatible view of the diagnostic header parse.

    Returns ``(metadata, tensors, data_start, declared_tensor_count)``.
    ``data_start`` is None when the payload boundary could not be established.
    Format defects no longer raise here (except a non-GGUF magic, as before);
    read them from ``validate_parser`` / ``diagnose_header`` instead.
    """

    parsed = diagnose_header(path)
    if any(d["check"] == "gguf_magic" for d in parsed.diagnostics):
        raise ValueError(f"not a GGUF file: {path}")
    return parsed.metadata, parsed.tensors, parsed.data_start, parsed.declared_tensor_count


@dataclass(frozen=True)
class MappedTensorMaps:
    """Actual production AR and NextN maps for one parsed GGUF file.

    ``model_map`` is always built (non-strict); ``nextn_map`` is built only when
    the file has AR-excluded trailing MTP blocks and its construction succeeds.
    ``section`` is the JSON-serializable map/source summary. Map availability
    and validation state are diagnostics: they never qualify the file as
    loadable or consumer-qualified.
    """

    info: GGUFModelInfo
    model_map: Qwen35GGUFModelMap
    nextn_map: Qwen35GGUFNextNMap | None
    section: dict


def _alias_records(slot_sources: dict[str, str]) -> list[dict]:
    """Group consumer slots that share one physical source tensor.

    Keys are scope-prefixed slot paths (``ar:``, ``nextn:``,
    ``nextn_fallback:``) so identical slot names in different maps stay
    distinct consumers. Sources with a single consumer are not aliases.
    """

    by_source: dict[str, list[str]] = defaultdict(list)
    for slot in sorted(slot_sources):
        by_source[slot_sources[slot]].append(slot)
    return [
        {
            "source": source,
            "consumer_slots": slots,
            "consumer_count": len(slots),
        }
        for source, slots in sorted(by_source.items())
        if len(slots) > 1
    ]


def _unique_source_names(slot_sources: dict[str, str]) -> list[str]:
    return sorted(set(slot_sources.values()))


def _nextn_tensor_count(ignored_names: tuple[str, ...], block_ids: tuple[int, ...]) -> int:
    prefixes = tuple(f"blk.{block_id}.nextn." for block_id in block_ids)
    return sum(1 for name in ignored_names if name.startswith(prefixes))


def build_tensor_maps(
    path: Path,
    metadata: dict,
    tensors: list[GGUFTensorInfo],
    version: int | None,
) -> MappedTensorMaps:
    """Build the actual production AR map and separate NextN map.

    The AR map is built non-strict so a partial file still reports which slots
    map; its validation state stays visible in the section and marks every
    route table diagnostic. The NextN map is attempted only for files with
    AR-excluded trailing MTP blocks; its failures are captured inside the
    section and can never discard the AR report. Qualification-class failures
    of the AR map itself (missing qwen35 metadata, unreadable config) raise and
    are reported as ``tensor_map`` diagnostic errors by the caller -- there is
    deliberately no guessed slot fallback.
    """

    info = GGUFModelInfo(
        path=path.resolve(),
        # 0 marks an unknown/unparseable version in diagnostic mode; the map is
        # derived from metadata, never from this field.
        version=version if version is not None else 0,
        alignment=int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)),
        metadata=metadata,
        tensors=tuple(tensors),
        tensor_data_offset=0,
    )
    model_map = build_qwen35_gguf_tensor_map(info, strict=False)
    validation = model_map.validation
    config = validation.config

    ar_slot_sources: dict[str, str] = {}
    root_slot_sources: dict[str, str] = {}
    for slot, tensor in model_map.root_tensors.items():
        ar_slot_sources[f"ar:root.{slot}"] = tensor.name
        root_slot_sources[f"root.{slot}"] = tensor.name
    for layer in model_map.layers:
        for slot, tensor in layer.tensors.items():
            ar_slot_sources[f"ar:layers.{layer.layer_id}.{slot}"] = tensor.name
    ar_unique = _unique_source_names(ar_slot_sources)

    ignored_names = tuple(validation.ignored)
    nextn_section: dict = {
        "blocks": 0,
        "note": "no AR-excluded trailing MTP block in this file",
    }
    nextn_map: Qwen35GGUFNextNMap | None = None
    if config.ignored_block_ids:
        try:
            nextn_map = build_qwen35_gguf_nextn_tensor_map(info, strict=False)
        except _PLUGIN_QUALIFICATION_ERRORS as error:
            nextn_section = {
                "blocks": len(config.ignored_block_ids),
                "diagnostic_only": True,
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            nv = nextn_map.validation
            ar_source_set = set(ar_unique)
            own_slot_sources: dict[str, str] = {}
            for slot, tensor in nextn_map.layer_tensors.items():
                own_slot_sources[f"nextn:draft.layer.{slot}"] = tensor.name
            for slot, tensor in nextn_map.nextn_tensors.items():
                own_slot_sources[f"nextn:draft.nextn.{slot}"] = tensor.name
            fallback_slot_sources: dict[str, str] = {}
            for slot, tensor in nextn_map.fallback_tensors.items():
                fallback_slot_sources[f"nextn_fallback:root.{slot}"] = tensor.name
            # AR borrowing is decided by actual source identity, not by the
            # fallback slot's root-shaped name: a present block-local tensor
            # (e.g. shared_head_norm) is a within-NextN alias, not a borrow.
            fallback_records = [
                {
                    "slot_path": slot.removeprefix("nextn_fallback:"),
                    "source": fallback_slot_sources[slot],
                    "borrows_ar_root": fallback_slot_sources[slot] in ar_source_set,
                }
                for slot in sorted(fallback_slot_sources)
            ]
            own_unique = _unique_source_names(own_slot_sources)
            nextn_section = {
                "blocks": len(config.ignored_block_ids),
                "block_id": nv.block_id,
                "validation_passed": nv.passed,
                "diagnostic": not nv.passed,
                "missing": list(nv.missing),
                "unexpected": list(nv.unexpected),
                "dtype_errors": list(nv.dtype_errors),
                "shape_errors": list(nv.shape_errors),
                "own_consumer_slots": len(own_slot_sources),
                "own_unique_sources": len(own_unique),
                "fallback_consumer_slots": len(fallback_slot_sources),
                "fallback_slots": fallback_records,
                "ar_borrowed_fallback_slots": [
                    record["slot_path"] for record in fallback_records if record["borrows_ar_root"]
                ],
                "aliases": _alias_records({**own_slot_sources, **fallback_slot_sources}),
            }

    nextn_unique: list[str] = []
    nextn_all_slot_sources: dict[str, str] = {}
    if nextn_map is not None:
        for slot, tensor in nextn_map.layer_tensors.items():
            nextn_all_slot_sources[f"nextn:draft.layer.{slot}"] = tensor.name
        for slot, tensor in nextn_map.nextn_tensors.items():
            nextn_all_slot_sources[f"nextn:draft.nextn.{slot}"] = tensor.name
        for slot, tensor in nextn_map.fallback_tensors.items():
            nextn_all_slot_sources[f"nextn_fallback:root.{slot}"] = tensor.name
        nextn_unique = _unique_source_names(nextn_all_slot_sources)
    combined_slot_sources = {**ar_slot_sources, **nextn_all_slot_sources}
    combined_unique = sorted(set(ar_unique) | set(nextn_unique))

    layer_types = Counter(config.layer_types)
    section = {
        "available": True,
        "architecture": config.architecture,
        "validation_passed": validation.passed,
        "diagnostic": not validation.passed,
        "tensors_on_disk": len(tensors),
        "ar": {
            "layers": len(model_map.layers),
            "layer_types": {
                str(name): count for name, count in sorted(layer_types.items())
            },
            "ignored_block_ids": list(config.ignored_block_ids),
            "consumer_slots": len(ar_slot_sources),
            "unique_sources": len(ar_unique),
            "root_slots": dict(sorted(root_slot_sources.items())),
            "aliases": _alias_records(ar_slot_sources),
        },
        "ignored": {
            "tensor_count": len(ignored_names),
            "block_ids": list(config.ignored_block_ids),
            "nextn_tensor_count": _nextn_tensor_count(ignored_names, config.ignored_block_ids),
        },
        "nextn": nextn_section,
        "combined": {
            "consumer_slots": len(combined_slot_sources),
            "unique_sources": len(combined_unique),
            "aliases": _alias_records(combined_slot_sources),
            "accounting": (
                "logical consumer slots vs unique physical source tensors; "
                "shared sources counted once and reported as aliases; "
                "source-identity dedup is not resident-allocation dedup "
                "(one source may plan different layouts per consumer)"
            ),
        },
        "validation": {
            "passed": validation.passed,
            "missing": list(validation.missing),
            "unexpected": list(validation.unexpected),
            "shape_errors": list(validation.shape_errors),
            "ignored": list(ignored_names),
        },
    }
    return MappedTensorMaps(
        info=info, model_map=model_map, nextn_map=nextn_map, section=section
    )


# The production AR planner's raw-IQ predicate (decode-repack veto plus
# model-wide F32 linear contraction) is the shared pure policy function
# ``gguf_ar_raw_iq_contract``; the audit calls it instead of keeping a local
# mirror of the production set.


def _ar_slot_tensor_pairs(model_map: Qwen35GGUFModelMap) -> list[tuple[str, GGUFTensorInfo]]:
    """Production AR consumer slots in plan order: roots, then layers."""

    return [
        *((f"root.{slot}", tensor) for slot, tensor in model_map.root_tensors.items()),
        *(
            (f"layers.{layer.layer_id}.{slot}", tensor)
            for layer in model_map.layers
            for slot, tensor in layer.tensors.items()
        ),
    ]


def _route_account(planned: list[tuple[str, GGUFTensorInfo, object]]) -> tuple[dict, float, float]:
    """Count routes per GGML type for already-planned (slot, tensor, spec) triples."""

    routes: dict[str, Counter] = defaultdict(Counter)
    expand_stored = expand_resident = 0.0
    for _slot_path, tensor, spec in planned:
        kind = LAYOUT_MEANING.get(spec.layout, f"kernel:{spec.quant_key}")
        routes[tensor.ggml_type_name][kind] += 1
        if spec.layout == "dense_bf16":
            expand_stored += tensor.nbytes / GIB
            expand_resident += tensor.n_elements * 2 / GIB
    return {k: dict(v) for k, v in sorted(routes.items())}, expand_stored, expand_resident


# Integral allocations of each resident layout; anything else in a spec's
# allocation_names is a sidecar record on top of the primary resident.
_TILE_BASE_ALLOCATIONS = frozenset({"tiles"})
_RAW_BASE_ALLOCATIONS = frozenset({"raw"})
_LAYOUT_BASE_ALLOCATIONS = {
    LAYOUT_Q4_K_PACK8: frozenset({"qweight", "scales", "mins"}),
    LAYOUT_DENSE_F32: _RAW_BASE_ALLOCATIONS,
    LAYOUT_DENSE_BF16: _RAW_BASE_ALLOCATIONS,
    LAYOUT_RAW_GGUF: _RAW_BASE_ALLOCATIONS,
    LAYOUT_GGUF_Q4_K_T16: _TILE_BASE_ALLOCATIONS,
    LAYOUT_GGUF_Q4_K_QMICRO_T16: _TILE_BASE_ALLOCATIONS,
    LAYOUT_GGUF_Q4_K_X8: _TILE_BASE_ALLOCATIONS,
    LAYOUT_GGUF_Q5_K_T16: _TILE_BASE_ALLOCATIONS,
    LAYOUT_GGUF_Q5_K_QMICRO_T16: _TILE_BASE_ALLOCATIONS,
    LAYOUT_GGUF_Q5_K_X8: _TILE_BASE_ALLOCATIONS,
    LAYOUT_GGUF_Q6_K_T16: _TILE_BASE_ALLOCATIONS,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR: _TILE_BASE_ALLOCATIONS,
    LAYOUT_GGUF_Q6_K_X8: _TILE_BASE_ALLOCATIONS,
    LAYOUT_GGUF_Q8_0_T16: _TILE_BASE_ALLOCATIONS,
}
_ALLOCATION_REASONS = {
    "raw": "raw GGUF sidecar (raw-MMQ/verification consumers)",
    "qmicro_planar": "planar qmicro sidecar (HIPENGINE_C8_Q5_PLANAR_DP4A=1)",
    "x8": "X8 top-1 sidecar (HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR)",
    "decode_tiles": "Q4_K T16 decode sidecar retained on the pack8 resident",
    "decode_tiles_r3plus": "Q4_K T16 r3+ decode sidecar retained on the pack8 resident",
}
_ALLOCATION_ACCOUNTING_NOTE = (
    "requested planned weight-allocation bytes from"
    " planned_qwen35_gguf_weight_allocation_nbytes over unique (source, layout)"
    " residents, including sidecars; excludes allocator alignment beyond the"
    " formula, runtime scratch, KV, recurrent state, graph pools and measured"
    " residency. native_refusal_lower_bound keeps refused tensors at their"
    " compressed source size; bf16_refusal_scenario expands them to BF16;"
    " refusals are never omitted from these totals and neither treatment is a"
    " working load."
)


def _allocation_account(
    planned: list[tuple[str, GGUFTensorInfo, object]],
    refused_pairs: list[tuple[str, GGUFTensorInfo]],
) -> dict:
    """Requested allocation-formula bytes for one planner scope.

    Counts unique ``(source name, layout)`` residents exactly like the
    production materializer's ownership dedup, sums
    ``planned_qwen35_gguf_weight_allocation_nbytes`` records (primary resident
    plus sidecars), and aggregates sidecar counts/bytes with reasons. Refused
    slots enter the two hypothetical totals -- compressed-source lower bound
    and BF16 expansion scenario -- deduplicated by physical source so a
    refused source with two consumers is never double counted.
    """

    seen: set[tuple[str, str]] = set()
    residents: Counter = Counter()
    layout_bytes: dict[str, int] = defaultdict(int)
    sidecars: dict[str, dict] = {}
    accepted = 0
    expert_sidecar_residents = 0
    formula_unavailable: list[dict] = []
    for _slot, tensor, spec in planned:
        key = (tensor.name, spec.layout)
        if key in seen:
            continue
        seen.add(key)
        try:
            records = planned_qwen35_gguf_weight_allocation_nbytes(spec)
        except ValueError as error:
            # The allocation formula requires tile-aligned shapes; synthetic or
            # pathological metadata can plan a layout it cannot size. Report
            # the gap explicitly instead of crashing or inventing bytes.
            formula_unavailable.append(
                {"source": tensor.name, "layout": spec.layout, "reason": str(error)}
            )
            continue
        base = _LAYOUT_BASE_ALLOCATIONS.get(spec.layout, frozenset())
        resident_bytes = sum(int(n) for _, n in records)
        accepted += resident_bytes
        layout_bytes[spec.layout] += resident_bytes
        residents[spec.layout] += 1
        if spec.sidecar_layouts:
            expert_sidecar_residents += 1
        for allocation_name, nbytes in records:
            if allocation_name in base:
                continue
            record = sidecars.setdefault(
                allocation_name,
                {
                    "count": 0,
                    "planned_bytes": 0,
                    "reason": _ALLOCATION_REASONS.get(
                        allocation_name, "allocation beyond the primary resident"
                    ),
                },
            )
            record["count"] += 1
            record["planned_bytes"] += int(nbytes)
    refused_sources: dict[str, GGUFTensorInfo] = {}
    for _slot, tensor in refused_pairs:
        refused_sources.setdefault(tensor.name, tensor)
    refused_source_bytes = sum(int(t.nbytes) for t in refused_sources.values())
    refused_bf16_bytes = sum(2 * int(t.n_elements) for t in refused_sources.values())
    return {
        "unique_residents": len(seen),
        "accepted_planned_bytes": accepted,
        "accepted_planned_gib": round(accepted / GIB, 6),
        "accepted_planned_bytes_complete": not formula_unavailable,
        "formula_unavailable_residents": formula_unavailable,
        "residents_by_layout": {
            layout: {"count": residents[layout], "planned_bytes": layout_bytes[layout]}
            for layout in sorted(residents)
        },
        "sidecars": {name: sidecars[name] for name in sorted(sidecars)},
        "expert_pack8_sidecar_residents": expert_sidecar_residents,
        "refused_slot_count": len(refused_pairs),
        "refused_unique_sources": len(refused_sources),
        "refused_source_bytes": refused_source_bytes,
        "refused_source_gib": round(refused_source_bytes / GIB, 6),
        "refused_bf16_bytes": refused_bf16_bytes,
        "refused_bf16_gib": round(refused_bf16_bytes / GIB, 6),
        "native_refusal_lower_bound_bytes": accepted + refused_source_bytes,
        "native_refusal_lower_bound_gib": round((accepted + refused_source_bytes) / GIB, 6),
        "bf16_refusal_scenario_bytes": accepted + refused_bf16_bytes,
        "bf16_refusal_scenario_gib": round((accepted + refused_bf16_bytes) / GIB, 6),
        "accounting": _ALLOCATION_ACCOUNTING_NOTE,
    }


def _plan_slots_per_slot(
    pairs: list[tuple[str, GGUFTensorInfo]],
    *,
    repack: bool,
    flags: dict,
    contract_f32_linear: bool = False,
) -> tuple[list[tuple[str, GGUFTensorInfo, object]], dict[str, list[str]], list[tuple[str, GGUFTensorInfo]]]:
    """Plan each consumer slot independently, collecting every refusal.

    Mirrors the production per-slot planner calls (plan_qwen35_gguf_nextn_materialization
    is exactly this loop without try/except) so one refused slot never discards the
    report for the other slots. ``contract_f32_linear`` carries the model-wide
    F32 contraction the production AR planner derives from its raw-IQ predicate,
    so per-slot fallback planning matches production planner semantics; the
    NextN planner has no such contraction and is called with the default.
    Returns planned triples, rejection strings per GGML type, and the refused
    (slot, tensor) pairs for byte accounting.
    """

    planned: list[tuple[str, GGUFTensorInfo, object]] = []
    rejections: dict[str, list[str]] = defaultdict(list)
    refused_pairs: list[tuple[str, GGUFTensorInfo]] = []
    for slot_path, tensor in pairs:
        try:
            spec = plan_qwen35_gguf_weight_spec(
                slot_path,
                tensor,
                decode_repack=repack,
                contract_f32_linear=contract_f32_linear,
                **flags,
            )
        except ValueError:
            rejections[tensor.ggml_type_name].append(
                f"{slot_path} ({'x'.join(map(str, tensor.shape))})"
            )
            refused_pairs.append((slot_path, tensor))
            continue
        planned.append((slot_path, tensor, spec))
    return planned, {k: v for k, v in sorted(rejections.items())}, refused_pairs


def plan(backend: str, metadata: dict, maps: MappedTensorMaps, *, environ=None) -> dict:
    """Route actual AR and NextN map slots through the production weight planner.

    AR routes come from the production AR planner over the actual map (per-slot
    fallback with rejection collection only when the whole-AR plan refuses a
    slot; the fallback passes the same model-wide F32 contraction the
    production planner derives from the shared raw-IQ predicate). NextN routes
    are reported separately as ``nextn_routes`` with the draft's own slots
    (``own``) and its root-shaped fallback slots (``fallback``); NextN failures
    never discard the AR report, and the NextN planner receives only the four
    dense flags the production NextN planner passes -- AR policies are not
    applied to the draft indiscriminately. Route tables are per consumer slot
    through ``plan_qwen35_gguf_weight_spec`` semantics: one physical source may
    plan different layouts for different consumers, so source reuse never
    collapses routes (the tied lm_head of a Q4_K embedding plans pack8 while
    the embedding itself plans raw). Dense flags resolve through the shared
    pure policy API with a source-reading capability reader, so no backend
    package is imported and the audit cannot drift from runtime policy.
    """

    file_type = llama_file_type_name(metadata.get("general.file_type"))
    flags = resolve_gguf_dense_flags(
        backend, file_type, capability_reader=source_capability_reader(), environ=environ
    )
    requested_repack = gguf_decode_repack_enabled(None)
    # The production AR planner vetoes decode repack itself for raw-IQ AR layers;
    # pre-applying the same veto keeps the per-slot fallback path identical.
    raw_iq = gguf_ar_raw_iq_contract(
        tensor.ggml_type
        for layer in maps.model_map.layers
        for tensor in layer.tensors.values()
    )
    repack = requested_repack and not raw_iq

    ar_pairs = _ar_slot_tensor_pairs(maps.model_map)
    try:
        production = plan_qwen35_gguf_materialization(
            maps.model_map, decode_repack=repack, **flags
        )
        planned = [
            *[(spec.slot_path, spec.source, spec) for spec in production.root_specs.values()],
            *[
                (spec.slot_path, spec.source, spec)
                for layer in production.layer_specs
                for spec in layer.values()
            ],
        ]
        rejections: dict[str, list[str]] = {}
        refused_pairs: list[tuple[str, GGUFTensorInfo]] = []
        planner_mode = "production_planner"
    except ValueError:
        # Whole-AR planning refused at least one slot: re-plan per slot so every
        # refusal is collected instead of discarding the other slots' routes.
        pairs = ar_pairs
        planned, rejections, refused_pairs = _plan_slots_per_slot(
            pairs, repack=repack, flags=flags, contract_f32_linear=raw_iq
        )
        planner_mode = "per_slot_fallback"
    routes, expand_stored, expand_resident = _route_account(planned)
    allocation = _allocation_account(planned, refused_pairs)

    nextn_routes: dict | None = None
    if maps.nextn_map is not None:
        nextn_map = maps.nextn_map
        # plan_qwen35_gguf_nextn_materialization passes only these four dense
        # flags and materialize_qwen35_gguf_nextn_weights defaults decode_repack
        # to True (the runtime draft path does not read the repack env var).
        nextn_flags = {
            name: flags[name]
            for name in (
                "dense_q4_t16",
                "dense_q5_t16_ssm_out",
                "dense_q5_t16_h5120",
                "dense_q6_qmicro_planar",
            )
        }
        nextn_repack = True
        own_pairs = [
            *((f"draft.layer.{slot}", tensor) for slot, tensor in nextn_map.layer_tensors.items()),
            *((f"draft.nextn.{slot}", tensor) for slot, tensor in nextn_map.nextn_tensors.items()),
        ]
        fallback_pairs = [
            (f"root.{slot}", tensor) for slot, tensor in nextn_map.fallback_tensors.items()
        ]

        def scope_report(pairs: list[tuple[str, GGUFTensorInfo]]) -> dict:
            planned_scope, scope_rejections, scope_refused = _plan_slots_per_slot(
                pairs, repack=nextn_repack, flags=nextn_flags
            )
            scope_routes, scope_stored, scope_resident = _route_account(planned_scope)
            return {
                "consumer_slots": len(pairs),
                "unique_sources": len({tensor.name for _, tensor in pairs}),
                "routes": scope_routes,
                "rejections": scope_rejections,
                "rejected_tensors": sum(len(v) for v in scope_rejections.values()),
                "bf16_expand_stored_gib": round(scope_stored, 3),
                "bf16_expand_resident_gib": round(scope_resident, 3),
                "allocation": _allocation_account(planned_scope, scope_refused),
            }

        nextn_routes = {
            "own": scope_report(own_pairs),
            "fallback": scope_report(fallback_pairs),
        }

    f32_contracted = sum(
        1
        for _slot, tensor, spec in planned
        if spec.layout == LAYOUT_DENSE_BF16
        and GGMLQuantizationType(tensor.ggml_type) == GGMLQuantizationType.F32
    )
    statuses = capability_status().get(backend, {})
    # UD-U1 F5: bind the artifact qualification (pinned plain control /
    # UD preset / unknown-manifest sentinel) so the reported default can never
    # claim a plain-certified stamp default for an unqualified manifest.
    from hipengine.loading.qwen35_gguf_admission import (
        qwen35_gguf_artifact_preset_key,
    )

    artifact_preset_key = qwen35_gguf_artifact_preset_key(
        maps.model_map,
        nextn_map=maps.nextn_map,
        file_type_stamp=file_type,
    )
    return {
        "backend": backend,
        "file_type_name": file_type,
        "artifact_preset_key": artifact_preset_key,
        "scope": "ar_map_plus_nextn_map",
        "map_validation_passed": maps.model_map.validation.passed,
        "planner_mode": planner_mode,
        "f32_contracted_slots": f32_contracted,
        "decode_repack_requested": requested_repack,
        "decode_repack_enabled": repack,
        "package_flags": flags,
        "capability_status": statuses,
        "fp16_recurrent_state_default_on": gguf_fp16_recurrent_state_default(
            backend,
            file_type,
            capability_reader=source_capability_reader(),
            artifact_preset_key=artifact_preset_key,
        ),
        "ar_consumer_slots": len(ar_pairs),
        "ar_unique_sources": len({tensor.name for _, tensor in ar_pairs}),
        "routes": routes,
        "rejections": rejections,
        "rejected_tensors": sum(len(v) for v in rejections.values()),
        "bf16_expand_stored_gib": round(expand_stored, 3),
        "bf16_expand_resident_gib": round(expand_resident, 3),
        "allocation": allocation,
        "nextn_routes": nextn_routes,
    }


def _print_tensor_map_summary(tensor_map: dict) -> None:
    """Print the scoped AR vs NextN map availability/validation summary.

    The AR map and the NextN map validate independently: an AR pass never
    stands in for a NextN pass, so the verdicts are scoped explicitly and a
    failed or unbuilt NextN map is visible in the default text (missing,
    unexpected, dtype, and shape details or the construction exception), not
    only in the JSON. A NextN map absent by design (no AR-excluded trailing MTP
    block) is reported as not applicable, never as a failure. NextN route
    tables printed later are diagnostics whenever the NextN map failed here;
    zero per-slot planner refusals are not map admission and not consumer
    qualification.
    """

    ar = tensor_map["ar"]
    combined = tensor_map["combined"]
    nextn = tensor_map["nextn"]
    ar_word = "passed" if tensor_map["validation_passed"] else "FAILED (diagnostic)"
    print(
        f"    tensor map: architecture={tensor_map['architecture']} ar_validation={ar_word}"
        f" ar_layers={ar['layers']} ar_slots={ar['consumer_slots']} ar_sources={ar['unique_sources']}"
        f" ignored={tensor_map['ignored']['tensor_count']}"
        f" nextn_blocks={nextn['blocks']}"
        f" combined_slots={combined['consumer_slots']} combined_sources={combined['unique_sources']}"
    )
    for alias in combined["aliases"]:
        print(
            f"      alias source {alias['source']}: {alias['consumer_count']} consumer slots"
            f" ({', '.join(alias['consumer_slots'])})"
        )
    if "error" in nextn:
        # The NextN map could not be constructed; it has no routes to report.
        # The AR verdict above is unaffected and stays printed.
        print(
            f"    nextn map: NOT BUILT (diagnostic only; nextn routes withheld): {nextn['error']}"
        )
        return
    if not nextn["blocks"]:
        print(
            f"    nextn map: not applicable ({nextn.get('note', 'no AR-excluded trailing MTP block')})"
        )
        return
    nextn_word = "passed" if nextn.get("validation_passed") else "FAILED (diagnostic)"
    print(
        f"    nextn map: validation={nextn_word} block_id={nextn['block_id']}"
        f" own_slots={nextn['own_consumer_slots']}/{nextn['own_unique_sources']}"
        f" fallback_slots={nextn['fallback_consumer_slots']}"
    )
    for label, key in (
        ("missing", "missing"),
        ("unexpected", "unexpected"),
        ("dtype", "dtype_errors"),
        ("shape", "shape_errors"),
    ):
        items = nextn.get(key) or []
        if items:
            preview = "; ".join(str(item) for item in items[:4])
            more = "" if len(items) <= 4 else f" (+{len(items) - 4} more)"
            print(f"      nextn {label}: {preview}{more}")


def _header_identity(path: Path, data_start: int | None) -> dict:
    """Header identity for one parsed file, with its exact boundary.

    The identity is SHA256 over the raw file bytes ``[0, data_start)`` where
    ``data_start`` is production ``scan_gguf``'s ``tensor_data_offset`` (the
    tensor-info table end aligned up to ``general.alignment``). The region
    covers magic, version, metadata, the descriptor table and the alignment
    padding after it, and excludes every tensor payload byte; raw bytes, no
    canonicalization beyond that boundary. It identifies the inspected
    metadata, never the complete model bytes. When the payload boundary could
    not be established (format-invalid file) no identity is claimed.
    """

    boundary = (
        "sha256 over raw file bytes [0, data_start_bytes): GGUF magic, version,"
        " metadata, tensor-info table and the alignment padding after it;"
        " excludes every tensor payload byte (metadata identity, not model bytes)"
    )
    if data_start is None:
        return {
            "sha256": None,
            "bytes": None,
            "note": "payload boundary unknown (format-invalid); no header identity claimed",
            "boundary": boundary,
        }
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            remaining = data_start
            while remaining:
                chunk = handle.read(min(1 << 22, remaining))
                if not chunk:
                    raise OSError("file shorter than its header boundary")
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError as error:
        return {
            "sha256": None,
            "bytes": data_start,
            "note": f"header bytes unreadable: {type(error).__name__}: {error}",
            "boundary": boundary,
        }
    return {"sha256": digest.hexdigest(), "bytes": data_start, "boundary": boundary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="GGUF files to inspect")
    parser.add_argument("--json", type=Path, help="write the full result here")
    parser.add_argument("--check-constants", action="store_true", help="fail if a package capability constant vanished")
    args = parser.parse_args(argv)

    # A capability that only one backend qualifies is legitimately absent on the
    # other: the source reader returns the caller's default there. A rename
    # removes the name from every backend, so that is what this gate catches.
    if args.check_constants:
        statuses = capability_status()
        gone = [
            name
            for name in GGUF_DENSE_CAPABILITY_NAMES
            if all(statuses[backend][name] == "missing" for backend in statuses)
        ]
        if gone:
            print(f"capability constants not defined by any backend: {gone}", file=sys.stderr)
            return 2

    report: dict = {
        "schema_version": 2,
        "schema_notes": (
            "v2: header identity over [0, data_start); per-scope allocation"
            " accounting with sidecar reasons and both hypothetical refusal"
            " treatments; shared-policy capability resolution with"
            " per-capability status; f32_contracted_slots replaces"
            " f32_contraction_applied. v1 was the implicit schema of"
            " docs/UD-QUANTS-REVIEW.json."
        ),
        "files": [],
    }
    for path in args.paths:
        parsed = validate_parser(path)
        metadata, tensors = parsed.metadata, parsed.tensors
        try:
            file_size_on_disk: int | None = path.stat().st_size
        except OSError:
            file_size_on_disk = None  # size unknown; never claim zero bytes
        types = Counter(t.ggml_type_name for t in tensors)
        entry = {
            "file": str(path),
            "file_size_on_disk_bytes": file_size_on_disk,
            "header_identity": _header_identity(path, parsed.data_start),
            "tensors_parsed": len(tensors),
            "tensors_declared_in_header": parsed.declared_tensor_count,
            "tensor_table_complete": parsed.table_complete,
            "file_type": metadata.get("general.file_type"),
            "file_type_name": llama_file_type_name(metadata.get("general.file_type")),
            "dtype_histogram": dict(types.most_common()),
            "stored_weight_gib": round(sum(t.nbytes for t in tensors) / GIB, 3),
            "backends": [],
            "plugin_tensor_map": {},
            "parser_validation": parsed.to_json(),
        }
        # The actual production tensor maps (AR + separate NextN) are built
        # first: route planning is only meaningful when the real map exists, so
        # a map failure keeps the audit parser-only with no guessed route table.
        try:
            maps = build_tensor_maps(path, metadata, tensors, parsed.version)
        except _PLUGIN_QUALIFICATION_ERRORS as error:
            maps = None
            entry["tensor_map"] = {
                "available": False,
                "diagnostic_only": True,
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            entry["tensor_map"] = maps.section
        for backend in ("hip_gfx1100", "hip_gfx1151"):
            if maps is None:
                entry["backends"].append(
                    {
                        "backend": backend,
                        "diagnostic_only": True,
                        "error": "tensor map unavailable: " + entry["tensor_map"]["error"],
                    }
                )
                continue
            try:
                entry["backends"].append(plan(backend, metadata, maps))
            except _PLUGIN_QUALIFICATION_ERRORS as error:
                entry["backends"].append(
                    {
                        "backend": backend,
                        "diagnostic_only": True,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        report["files"].append(entry)
        print(f"\n=== {path.name}: file_type {entry['file_type']} = {entry['file_type_name']}")
        print(f"    dtype histogram: {entry['dtype_histogram']}")
        pv = entry["parser_validation"]
        if pv["format_valid"]:
            verdict = "format-valid (format-layer admission only; tensor-map/planner verdicts below)"
        else:
            verdict = "NOT loadable (format-invalid; partial diagnostic mode, not consumer qualification)"
        print(
            f"    parser: GGUF v{pv['gguf_version']} status={pv['status']}"
            f" production_scan={pv['production_scan']} -> {verdict}"
        )
        for diagnostic in pv["diagnostics"]:
            print(f"      {diagnostic['check']}: {diagnostic['detail']}")
        tensor_map = entry["tensor_map"]
        if not tensor_map.get("available"):
            print(f"    tensor map: unavailable (diagnostic only): {tensor_map.get('error')}")
            for backend in entry["backends"]:
                print(
                    f"    {backend['backend']}: planner unavailable (diagnostic only): {backend.get('error')}"
                )
        else:
            _print_tensor_map_summary(tensor_map)
        # When the NextN map's own validation failed, its route lines below are
        # diagnostics: rejected=0 there means no per-slot planner refusals, not
        # that the NextN map was admitted.
        nextn_diagnostic = bool(tensor_map.get("nextn", {}).get("diagnostic"))
        for backend in entry["backends"]:
            if "error" in backend and "routes" not in backend:
                if tensor_map.get("available"):
                    print(f"    {backend['backend']}: planner unavailable (diagnostic only): {backend['error']}")
                continue
            print(
                f"    {backend['backend']}: scope={backend['scope']}"
                f" repack={'on' if backend['decode_repack_enabled'] else 'OFF'}"
                f" fp16_recurrent_state={'on' if backend['fp16_recurrent_state_default_on'] else 'off'}"
                f" ar_slots={backend['ar_consumer_slots']}/{backend['ar_unique_sources']} sources"
                f" rejected={backend['rejected_tensors']}"
                f" bf16_expand={backend['bf16_expand_stored_gib']} -> {backend['bf16_expand_resident_gib']} GiB"
            )
            for qtype, kinds in sorted(backend["routes"].items(), key=lambda kv: -sum(kv[1].values())):
                print(f"      {qtype:<8} x{sum(kinds.values()):<4} {kinds}")
                for example in backend["rejections"].get(qtype, [])[:2]:
                    print(f"               rejected: {example}")
            nextn_routes = backend.get("nextn_routes")
            if nextn_routes:
                nextn_scope_note = (
                    " [map validation FAILED; diagnostic only]" if nextn_diagnostic else ""
                )
                for scope_name in ("own", "fallback"):
                    scope = nextn_routes.get(scope_name)
                    if not scope:
                        continue
                    print(
                        f"      nextn {scope_name}{nextn_scope_note}: slots={scope['consumer_slots']}"
                        f"/{scope['unique_sources']} sources rejected={scope['rejected_tensors']}"
                        f" bf16_expand={scope['bf16_expand_stored_gib']}"
                        f" -> {scope['bf16_expand_resident_gib']} GiB"
                    )
                    for qtype, kinds in sorted(scope["routes"].items(), key=lambda kv: -sum(kv[1].values())):
                        print(f"        {qtype:<8} x{sum(kinds.values()):<4} {kinds}")
                        for example in scope["rejections"].get(qtype, [])[:2]:
                            print(f"                 rejected: {example}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
