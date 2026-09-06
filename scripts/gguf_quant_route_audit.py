#!/usr/bin/env python3
"""Report which GGUF quant types a Qwen dense GGUF file needs and how they load.

Answers one question per file: after hipEngine validates the tensor map and runs
each tensor through the production weight planner, which tensors get a real
quantized kernel, which get expanded to dense BF16 at load, and which make the
loader refuse the file. Reads metadata and the tensor-info table only; no weights
are read, so a partially downloaded file can still be inspected. Nothing here runs
a model, allocates device memory, or touches a GPU.

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
    build_qwen35_gguf_tensor_map,
)
from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    gguf_decode_repack_enabled,
    plan_qwen35_gguf_weight_spec,
)
from hipengine.quant.gguf import (  # noqa: E402
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


def slot_path(name: str) -> str:
    if name == "token_embd.weight":
        return "root.token_embd"
    if name == "output.weight":
        return "root.lm_head"
    if name == "output_norm.weight":
        return "root.output_norm"
    match = re.match(r"^blk\.(\d+)\.(.+?)\.weight$", name) or re.match(r"^blk\.(\d+)\.(.+)$", name)
    return f"layers.{match.group(1)}.{match.group(2)}" if match else name


def plan(backend: str, metadata: dict, tensors: list[GGUFTensorInfo]) -> dict:
    caps = backend_capabilities(backend)
    file_type = llama_file_type_name(metadata.get("general.file_type"))
    # plan_qwen35_gguf_materialization disables decode repack when the file carries
    # raw-IQ weights, because those residents are consumed as compressed rank-3
    # blocks. Copy that rule; otherwise this table would claim repack for a file
    # that never gets it.
    raw_iq = any(t.ggml_type_name in ("IQ2_XS", "IQ3_XXS", "IQ4_XS") for t in tensors)
    repack = gguf_decode_repack_enabled(None) and not raw_iq
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
    routes: dict[str, Counter] = defaultdict(Counter)
    rejections: dict[str, list[str]] = defaultdict(list)
    expand_stored = expand_resident = 0.0
    for tensor in tensors:
        try:
            spec = plan_qwen35_gguf_weight_spec(slot_path(tensor.name), tensor, decode_repack=repack, **flags)
        except ValueError as error:
            routes[tensor.ggml_type_name]["rejected"] += 1
            rejections[tensor.ggml_type_name].append(f"{tensor.name} ({'x'.join(map(str, tensor.shape))})")
            continue
        kind = LAYOUT_MEANING.get(spec.layout, f"kernel:{spec.quant_key}")
        routes[tensor.ggml_type_name][kind] += 1
        if spec.layout == "dense_bf16":
            expand_stored += tensor.nbytes / GIB
            expand_resident += tensor.n_elements * 2 / GIB
    return {
        "backend": backend,
        "file_type_name": file_type,
        "decode_repack_enabled": repack,
        "package_flags": flags,
        "fp16_recurrent_state_default_on": fp16_recurrent_state_default(backend, file_type),
        "routes": {k: dict(v) for k, v in routes.items()},
        "rejections": {k: v for k, v in rejections.items()},
        "rejected_tensors": sum(c.get("rejected", 0) for c in routes.values()),
        "bf16_expand_stored_gib": round(expand_stored, 3),
        "bf16_expand_resident_gib": round(expand_resident, 3),
    }


def mapping_result(path: Path, metadata: dict, tensors: list[GGUFTensorInfo], version: int | None) -> dict:
    """Validate the file against the dense-Qwen plugin's expected tensor map."""

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
    validation = build_qwen35_gguf_tensor_map(info, strict=False).validation
    return {
        "present": len(validation.present),
        "missing": list(validation.missing),
        "unexpected": list(validation.unexpected),
        "shape_errors": list(validation.shape_errors),
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
        # Route planning and the plugin map are consumer qualification: keep them
        # best-effort so a refused file still yields its parser verdict. A section
        # that cannot run reports an error instead of aborting the audit.
        for backend in ("hip_gfx1100", "hip_gfx1151"):
            try:
                entry["backends"].append(plan(backend, metadata, tensors))
            except _PLUGIN_QUALIFICATION_ERRORS as error:
                entry["backends"].append(
                    {
                        "backend": backend,
                        "diagnostic_only": True,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        try:
            entry["plugin_tensor_map"] = mapping_result(path, metadata, tensors, parsed.version)
        except _PLUGIN_QUALIFICATION_ERRORS as error:
            entry["plugin_tensor_map"] = {
                "diagnostic_only": True,
                "error": f"{type(error).__name__}: {error}",
            }
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
        for backend in entry["backends"]:
            if "error" in backend:
                print(f"    {backend['backend']}: planner unavailable (diagnostic only): {backend['error']}")
                continue
            print(
                f"    {backend['backend']}: repack={'on' if backend['decode_repack_enabled'] else 'OFF'}"
                f" fp16_recurrent_state={'on' if backend['fp16_recurrent_state_default_on'] else 'off'}"
                f" rejected={backend['rejected_tensors']}"
                f" bf16_expand={backend['bf16_expand_stored_gib']} -> {backend['bf16_expand_resident_gib']} GiB"
            )
            for qtype, kinds in sorted(backend["routes"].items(), key=lambda kv: -sum(kv[1].values())):
                print(f"      {qtype:<8} x{sum(kinds.values()):<4} {kinds}")
                for example in backend["rejections"].get(qtype, [])[:2]:
                    print(f"               rejected: {example}")
        map_result = entry["plugin_tensor_map"]
        if "error" in map_result:
            print(f"    plugin tensor map: unavailable (diagnostic only): {map_result['error']}")
        else:
            print(
                f"    plugin tensor map: present={map_result['present']} missing={len(map_result['missing'])}"
                f" unexpected={len(map_result['unexpected'])} shape_errors={len(map_result['shape_errors'])}"
            )

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
