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
"""

from __future__ import annotations

import argparse
import json
import re
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
    gguf_decode_repack_enabled,
    plan_qwen35_gguf_materialization,
    plan_qwen35_gguf_weight_spec,
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

# hipengine.kernels.<backend> holds these as plain module constants. Importing the
# package runs kernel registration, which needs the HIP runtime, so the values are
# read from source instead. A renamed constant therefore shows up as False here,
# not as a silent pass: --check-constants fails if a name disappears.
CAPABILITY_NAMES = (
    "GGUF_DENSE_Q4_T16",
    "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP",
    "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES",
    "GGUF_DENSE_Q4_T16_ATTN_Q_08B",
    "GGUF_DENSE_Q5_T16_SSM_OUT",
    "GGUF_DENSE_Q5_T16_SSM_OUT_08B",
    "GGUF_DENSE_Q5_T16_QKV",
    "GGUF_DENSE_Q5_T16_H5120",
    "GGUF_DENSE_Q6_T16_QMICRO_PLANAR",
    "GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES",
)
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


def quoted_members(source_value: str) -> set[str]:
    """Members of a container constant written as a source expression.

    ``frozenset({"a", "b"})``, ``("a",)``, and ``set()`` all reduce to the quoted
    strings inside them. Used to read backend capability constants without
    importing the kernel package.
    """

    return {value.strip().lower() for value in re.findall(r"['\"]([^'\"]+)['\"]", source_value)}


def fp16_recurrent_state_default(backend: str, file_type_name: str) -> bool:
    """Mirror the runner's check: compare normalized file-type names.

    Absent capability means the backend has no such default, which is False.
    """

    raw = backend_capabilities(backend)["GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES"]
    if raw == "<missing>":
        return False
    return str(file_type_name).strip().lower() in quoted_members(raw)


def backend_capabilities(backend: str) -> dict[str, str]:
    source = (REPO_ROOT / "hipengine" / "kernels" / backend / "__init__.py").read_text()
    caps: dict[str, str] = {}
    for name in CAPABILITY_NAMES:
        match = re.search(rf"^{name} = (.+)$", source, re.M)
        caps[name] = match.group(1).strip() if match else "<missing>"
    return caps


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


# The production AR planner vetoes decode repack when AR layers carry raw-IQ
# weights (plan_qwen35_gguf_materialization's contract_q3_f32_linear predicate,
# which also reassociates those files' F32 alpha/beta/router slots to BF16).
# This mirror exists only to REPORT the effective repack veto; the actual
# planning always goes through the production planner itself. Replace it with a
# pure shared policy API per docs/UD-QUANTS.md U0 (see docs/REFACTOR.md).
_AR_CONTRACT_IQ_TYPES = frozenset(
    {
        GGMLQuantizationType.IQ2_XS,
        GGMLQuantizationType.IQ3_XXS,
        GGMLQuantizationType.IQ4_XS,
    }
)


def _ar_iq_contract(model_map: Qwen35GGUFModelMap) -> bool:
    """Mirror the AR planner's raw-IQ predicate over AR layer tensors only."""

    return any(
        GGMLQuantizationType(tensor.ggml_type) in _AR_CONTRACT_IQ_TYPES
        for layer in model_map.layers
        for tensor in layer.tensors.values()
    )


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


def _plan_slots_per_slot(
    pairs: list[tuple[str, GGUFTensorInfo]],
    *,
    repack: bool,
    flags: dict,
) -> tuple[list[tuple[str, GGUFTensorInfo, object]], dict[str, list[str]]]:
    """Plan each consumer slot independently, collecting every refusal.

    Mirrors the production per-slot planner calls (plan_qwen35_gguf_nextn_materialization
    is exactly this loop without try/except) so one refused slot never discards the
    report for the other slots. Known limitation versus the production AR planner:
    the model-wide F32 contraction is NOT applied here (plan_qwen35_gguf_weight_spec
    does not expose it), so F32 alpha/beta/router slots report f32-resident in this
    mode; the entry's ``f32_contraction_applied`` flag marks that.
    """

    planned: list[tuple[str, GGUFTensorInfo, object]] = []
    rejections: dict[str, list[str]] = defaultdict(list)
    for slot_path, tensor in pairs:
        try:
            spec = plan_qwen35_gguf_weight_spec(
                slot_path, tensor, decode_repack=repack, **flags
            )
        except ValueError:
            rejections[tensor.ggml_type_name].append(
                f"{slot_path} ({'x'.join(map(str, tensor.shape))})"
            )
            continue
        planned.append((slot_path, tensor, spec))
    return planned, {k: v for k, v in sorted(rejections.items())}


def plan(backend: str, metadata: dict, maps: MappedTensorMaps) -> dict:
    """Route actual AR and NextN map slots through the production weight planner.

    AR routes come from the production AR planner over the actual map (per-slot
    fallback with rejection collection only when the whole-AR plan refuses a
    slot). NextN routes are reported separately as ``nextn_routes`` with the
    draft's own slots (``own``) and its root-shaped fallback slots
    (``fallback``); NextN failures never discard the AR report. Route tables
    are per consumer slot through ``plan_qwen35_gguf_weight_spec`` semantics:
    one physical source may plan different layouts for different consumers, so
    source reuse never collapses routes (the tied lm_head of a Q4_K embedding
    plans pack8 while the embedding itself plans raw).
    """

    caps = backend_capabilities(backend)
    file_type = llama_file_type_name(metadata.get("general.file_type"))
    qmicro_types = quoted_members(caps["GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES"])
    flags = dict(
        dense_q4_t16=caps["GGUF_DENSE_Q4_T16"] == "True",
        dense_q4_qmicro_t16_gate_up=(
            caps["GGUF_DENSE_Q4_QMICRO_T16_GATE_UP"] == "True"
            and str(file_type).lower() in qmicro_types
        ),
        dense_q4_t16_attn_q_08b=caps["GGUF_DENSE_Q4_T16_ATTN_Q_08B"] == "True",
        dense_q5_t16_ssm_out=caps["GGUF_DENSE_Q5_T16_SSM_OUT"] == "True",
        dense_q5_t16_ssm_out_08b=caps["GGUF_DENSE_Q5_T16_SSM_OUT_08B"] == "True",
        dense_q5_t16_qkv=caps["GGUF_DENSE_Q5_T16_QKV"] == "True",
        dense_q5_t16_h5120=caps["GGUF_DENSE_Q5_T16_H5120"] == "True",
        dense_q6_qmicro_planar=caps["GGUF_DENSE_Q6_T16_QMICRO_PLANAR"] == "True",
    )
    requested_repack = gguf_decode_repack_enabled(None)
    # The production AR planner vetoes decode repack itself for raw-IQ AR layers;
    # pre-applying the same veto keeps the per-slot fallback path identical.
    repack = requested_repack and not _ar_iq_contract(maps.model_map)

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
        planner_mode = "production_planner"
        contraction_applied = True
    except ValueError:
        # Whole-AR planning refused at least one slot: re-plan per slot so every
        # refusal is collected instead of discarding the other slots' routes.
        pairs = ar_pairs
        planned, rejections = _plan_slots_per_slot(pairs, repack=repack, flags=flags)
        planner_mode = "per_slot_fallback"
        contraction_applied = False
    routes, expand_stored, expand_resident = _route_account(planned)

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
            planned_scope, scope_rejections = _plan_slots_per_slot(
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
            }

        nextn_routes = {
            "own": scope_report(own_pairs),
            "fallback": scope_report(fallback_pairs),
        }

    return {
        "backend": backend,
        "file_type_name": file_type,
        "scope": "ar_map_plus_nextn_map",
        "map_validation_passed": maps.model_map.validation.passed,
        "planner_mode": planner_mode,
        "f32_contraction_applied": contraction_applied,
        "decode_repack_requested": requested_repack,
        "decode_repack_enabled": repack,
        "package_flags": flags,
        "fp16_recurrent_state_default_on": fp16_recurrent_state_default(backend, file_type),
        "ar_consumer_slots": len(ar_pairs),
        "ar_unique_sources": len({tensor.name for _, tensor in ar_pairs}),
        "routes": routes,
        "rejections": rejections,
        "rejected_tensors": sum(len(v) for v in rejections.values()),
        "bf16_expand_stored_gib": round(expand_stored, 3),
        "bf16_expand_resident_gib": round(expand_resident, 3),
        "nextn_routes": nextn_routes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="GGUF files to inspect")
    parser.add_argument("--json", type=Path, help="write the full result here")
    parser.add_argument("--check-constants", action="store_true", help="fail if a package capability constant vanished")
    args = parser.parse_args(argv)

    # A capability that only one backend qualifies is legitimately absent on the
    # other: backend_package_capability() returns the caller's default. A rename
    # removes the name from every backend, so that is what this gate catches.
    if args.check_constants:
        caps = {b: backend_capabilities(b) for b in ("hip_gfx1100", "hip_gfx1151")}
        gone = [n for n in CAPABILITY_NAMES if all(caps[b][n] == "<missing>" for b in caps)]
        if gone:
            print(f"capability constants not defined by any backend: {gone}", file=sys.stderr)
            return 2

    report: dict = {"files": []}
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
            ar = tensor_map["ar"]
            combined = tensor_map["combined"]
            validation_word = "passed" if tensor_map["validation_passed"] else "FAILED (diagnostic)"
            print(
                f"    tensor map: architecture={tensor_map['architecture']} validation={validation_word}"
                f" ar_layers={ar['layers']} ar_slots={ar['consumer_slots']} ar_sources={ar['unique_sources']}"
                f" ignored={tensor_map['ignored']['tensor_count']}"
                f" nextn_blocks={tensor_map['nextn']['blocks']}"
                f" combined_slots={combined['consumer_slots']} combined_sources={combined['unique_sources']}"
            )
            for alias in combined["aliases"]:
                print(
                    f"      alias source {alias['source']}: {alias['consumer_count']} consumer slots"
                    f" ({', '.join(alias['consumer_slots'])})"
                )
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
                for scope_name in ("own", "fallback"):
                    scope = nextn_routes.get(scope_name)
                    if not scope:
                        continue
                    print(
                        f"      nextn {scope_name}: slots={scope['consumer_slots']}"
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
