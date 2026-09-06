"""Guards for the GGUF quant-route audit used by docs/UD-QUANTS.md.

The script reads backend capability constants out of kernel package source instead
of importing the package, because importing it registers kernels and needs the HIP
runtime. Two bugs found while writing it are pinned here: reading
``frozenset({"mostly_q4_k_s"})`` as a single token, which silently reported the
FP16-recurrent-state default as off on the backend that turns it on for Q4_K_S
files; and treating a capability defined by only one backend as "renamed", which
made the constant check fail on the backend that legitimately has no such default.

The UD-U0a section pins the parser-validation contract: complete files are
admitted by production ``scan_gguf``; files production refuses fall into an
explicit partial diagnostic mode that stops at the first loss of structural
boundary, reports unknown values as unknown, and never marks the file
format-valid or loadable. Production also refuses hostile metadata through
plain ValueError/TypeError (a string or array ``general.alignment``, an unknown
metadata value-type id, a tensor shape violating its block layout); the audit
diagnoses those too and the CLI still writes its JSON verdict.

The UD-U0b section pins the tensor-map contract: route tables come from the
actual production AR map (``build_qwen35_gguf_tensor_map``) and the actual
runtime NextN map (``build_qwen35_gguf_nextn_tensor_map``), never from guessed
slot names over all disk tensors. Slot paths are the production ones
(``root.token_embedding``, ``layers.<id>.<slot>``, ``draft.layer.<slot>``,
``draft.nextn.<slot>``). Reports distinguish logical consumer slots from unique
physical source tensors, dedup shared sources by source identity (a NextN
fallback slot counts as AR borrowing only when its actual source is an AR root
-- a present block-local ``shared_head_norm`` is a within-NextN alias, not an
AR borrow), and keep NextN failures from discarding a valid AR report. When the
model map cannot be built, the audit stays parser-only: no route table is
produced from guessed slots and the file is never called loadable or
consumer-qualified.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from struct import pack

import pytest

from hipengine.loading.gguf import GGUF_SUPPORTED_VERSIONS, scan_gguf
from hipengine.quant.gguf import GGMLQuantizationType, GGUFValueType

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gguf_quant_route_audit.py"

spec = importlib.util.spec_from_file_location("gguf_quant_route_audit", SCRIPT)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
# Register before exec: the module defines dataclasses, and dataclass processing
# resolves its own module through sys.modules.
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def test_quoted_members_reads_container_forms():
    assert audit.quoted_members('frozenset({"mostly_q4_k_s"})') == {"mostly_q4_k_s"}
    assert audit.quoted_members('frozenset({"Q4_K_S", "Q4_K_M"})') == {"q4_k_s", "q4_k_m"}
    assert audit.quoted_members('("MOSTLY_Q4_K_S",)') == {"mostly_q4_k_s"}
    assert audit.quoted_members("frozenset()") == set()


def test_fp16_recurrent_state_default_follows_the_file_type_stamp():
    # gfx1151 defaults FP16 recurrent state on for files stamped Q4_K_S.
    assert audit.fp16_recurrent_state_default("hip_gfx1151", "MOSTLY_Q4_K_S") is True
    assert audit.fp16_recurrent_state_default("hip_gfx1151", "MOSTLY_Q4_K_M") is False
    # gfx1100 declares no such default, which must read as off rather than crash.
    assert audit.fp16_recurrent_state_default("hip_gfx1100", "MOSTLY_Q4_K_S") is False


def test_every_capability_name_is_defined_by_some_backend():
    # A capability may legitimately exist on one backend only; a rename removes it
    # from both, which is what this catches.
    caps = {b: audit.backend_capabilities(b) for b in ("hip_gfx1100", "hip_gfx1151")}
    missing_everywhere = [
        name for name in audit.CAPABILITY_NAMES if all(caps[b][name] == "<missing>" for b in caps)
    ]
    assert missing_everywhere == []


def test_slot_path_guessing_is_removed_from_the_audit():
    # UD-U0b: the regex slot-name guesser is replaced by the actual production
    # tensor maps. Its presence would let a guessed route table masquerade as AR.
    assert not hasattr(audit, "slot_path")


def test_header_reader_reports_incomplete_tensor_data(tmp_path: pathlib.Path):
    """A truncated file still yields the full tensor table, which is the point."""

    def entry(key: str, value_type: GGUFValueType, raw: bytes) -> bytes:
        encoded = key.encode()
        return len(encoded).to_bytes(8, "little") + encoded + int(value_type).to_bytes(4, "little") + raw

    def string(value: str) -> bytes:
        encoded = value.encode()
        return len(encoded).to_bytes(8, "little") + encoded

    metadata = (
        entry("general.architecture", GGUFValueType.STRING, string("qwen35"))
        + entry("general.file_type", GGUFValueType.UINT32, pack("<I", 15))
        + entry("general.alignment", GGUFValueType.UINT32, pack("<I", 32))
    )
    tensor_name = b"blk.0.attn_gate.weight"
    tensor = (
        len(tensor_name).to_bytes(8, "little")
        + tensor_name
        + (2).to_bytes(4, "little")  # two dimensions
        + (32).to_bytes(8, "little")
        + (32).to_bytes(8, "little")  # GGML stores input-by-output order
        + int(GGMLQuantizationType.F32).to_bytes(4, "little")
        + (0).to_bytes(8, "little")  # tensor-data offset
    )
    # The 4 KiB of F32 data that the table describes are never written.
    header = b"GGUF" + (3).to_bytes(4, "little") + (1).to_bytes(8, "little") + (3).to_bytes(8, "little")
    path = tmp_path / "truncated.gguf"
    path.write_bytes(header + metadata + tensor)

    parsed_metadata, tensors, data_start, declared = audit.read_header(path)

    assert declared == 1 and len(tensors) == 1
    assert parsed_metadata["general.architecture"] == "qwen35"
    assert parsed_metadata["general.file_type"] == 15
    assert tensors[0].name == "blk.0.attn_gate.weight"
    assert tensors[0].shape == (32, 32)  # reversed into this repository's out-by-input order
    assert tensors[0].ggml_type_name == "F32"
    assert tensors[0].nbytes == 32 * 32 * 4
    assert data_start > 0


# --- UD-U0a parser validation -------------------------------------------------
#
# Format-layer admission is separate from consumer qualification: complete files
# are validated by production scan_gguf; refused files get bounded, fail-closed
# diagnostics and are never labeled format-valid or loadable. Version support
# comes from GGUF_SUPPORTED_VERSIONS, never a hardcoded 3.


def _kv(key: str, value_type: GGUFValueType, raw: bytes) -> bytes:
    encoded = key.encode()
    return (
        len(encoded).to_bytes(8, "little")
        + encoded
        + int(value_type).to_bytes(4, "little")
        + raw
    )


def _string(value: str) -> bytes:
    encoded = value.encode()
    return len(encoded).to_bytes(8, "little") + encoded


def _header(version: int, tensor_count: int, metadata_count: int) -> bytes:
    return (
        b"GGUF"
        + version.to_bytes(4, "little")
        + tensor_count.to_bytes(8, "little")
        + metadata_count.to_bytes(8, "little")
    )


def _descriptor(
    name: str, qtype: int | GGMLQuantizationType = GGMLQuantizationType.F32, offset: int = 0
) -> bytes:
    encoded = name.encode()
    return (
        len(encoded).to_bytes(8, "little")
        + encoded
        + (2).to_bytes(4, "little")  # two dimensions
        + (32).to_bytes(8, "little")
        + (32).to_bytes(8, "little")  # GGML stores input-by-output order
        + int(qtype).to_bytes(4, "little")
        + offset.to_bytes(8, "little")
    )


def _standard_metadata(alignment: int | str | list[int] = 32, extra: bytes = b"") -> bytes:
    # A string alignment refuses production through int()'s ValueError; an array
    # through TypeError. Both classes must be diagnosed, never crash the audit.
    if isinstance(alignment, str):
        alignment_kv = _kv("general.alignment", GGUFValueType.STRING, _string(alignment))
    elif isinstance(alignment, list):
        alignment_kv = _kv(
            "general.alignment",
            GGUFValueType.ARRAY,
            int(GGUFValueType.UINT32).to_bytes(4, "little")
            + len(alignment).to_bytes(8, "little")
            + pack(f"<{len(alignment)}I", *alignment),
        )
    else:
        alignment_kv = _kv("general.alignment", GGUFValueType.UINT32, pack("<I", alignment))
    return (
        _kv("general.architecture", GGUFValueType.STRING, _string("qwen35"))
        + _kv("general.file_type", GGUFValueType.UINT32, pack("<I", 15))
        + alignment_kv
        + extra
    )


def _aligned(prefix: bytes, alignment: int = 32) -> bytes:
    """Pad the header region so a following payload starts exactly at data_start."""
    return prefix + bytes((-len(prefix)) % alignment)


_TENSOR_NAME = "blk.0.attn_gate.weight"
_TENSOR_BYTES = 32 * 32 * 4


def _complete_file(tmp_path: pathlib.Path, version: int = 3, name: str = "ok.gguf") -> pathlib.Path:
    path = tmp_path / name
    prefix = _header(version, 1, 3) + _standard_metadata() + _descriptor(_TENSOR_NAME)
    path.write_bytes(_aligned(prefix) + bytes(_TENSOR_BYTES))
    return path


def _checks(parsed) -> list[str]:
    return [diagnostic["check"] for diagnostic in parsed.diagnostics]


def test_complete_file_is_validated_by_production_scan(tmp_path, capsys):
    path = _complete_file(tmp_path)
    out = tmp_path / "report.json"

    assert audit.main([str(path), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    entry = json.loads(out.read_text())["files"][0]

    parsed = entry["parser_validation"]
    assert parsed["production_scan"] == "accepted"
    assert parsed["production_scan_error"] is None
    assert parsed["gguf_version"] == 3
    assert parsed["version_supported"] is True
    assert parsed["table_complete"] is True and parsed["data_complete"] is True
    assert parsed["format_valid"] is True
    assert parsed["status"] == "complete"
    assert parsed["diagnostics"] == []
    # Existing report fields stay coherent for a complete file.
    assert entry["tensors_parsed"] == 1
    assert entry["tensors_declared_in_header"] == 1
    assert entry["tensor_table_complete"] is True
    # Explicit CLI labeling: format-valid is qualified as format-layer only.
    assert "format-valid" in printed
    # The tiny fixture has no qwen35 model metadata, so the production tensor
    # map cannot be built: the map section must be unavailable and diagnostic
    # only, and no route table may be guessed from disk tensor names.
    assert entry["tensor_map"]["available"] is False
    assert entry["tensor_map"]["diagnostic_only"] is True
    assert "error" in entry["tensor_map"]
    for backend in entry["backends"]:
        assert backend["diagnostic_only"] is True
        assert "routes" not in backend


def test_version_2_complete_file_is_supported_without_hardcoding_three(tmp_path):
    path = _complete_file(tmp_path, version=2, name="v2.gguf")

    parsed = audit.validate_parser(path)

    assert parsed.production_scan == "accepted"
    assert parsed.version == 2
    assert parsed.version_supported is True
    assert parsed.format_valid is True
    assert parsed.status == "complete"


def test_supported_version_list_is_not_redefined_by_the_audit():
    # The diagnostic mirror and production must agree on what is supported.
    assert audit.GGUF_SUPPORTED_VERSIONS == GGUF_SUPPORTED_VERSIONS


def test_unsupported_version_is_refused_and_diagnosed_bounded(tmp_path):
    path = _complete_file(tmp_path, version=4, name="v4.gguf")

    with pytest.raises(Exception):
        scan_gguf(path)  # production remains the authority and refuses it
    parsed = audit.validate_parser(path)

    assert parsed.production_scan == "rejected"
    assert parsed.production_scan_error is not None
    assert parsed.version == 4  # actual version is reported, not silently 3
    assert parsed.version_supported is False
    # Bounded: the body of an unsupported version is never parsed as a known layout.
    assert _checks(parsed) == ["unsupported_version"]
    assert parsed.metadata == {} and parsed.tensors == []
    assert parsed.data_start is None
    assert parsed.format_valid is False
    assert parsed.status == "unsupported_version"


def test_duplicate_metadata_key_is_refused_and_keeps_first_value(tmp_path):
    path = tmp_path / "dup_meta.gguf"
    path.write_bytes(
        _aligned(
            _header(3, 1, 4)
            + _standard_metadata(extra=_kv("general.file_type", GGUFValueType.UINT32, pack("<I", 999)))
            + _descriptor(_TENSOR_NAME)
        )
        + bytes(_TENSOR_BYTES)
    )

    parsed = audit.validate_parser(path)

    assert parsed.production_scan == "rejected"
    assert _checks(parsed) == ["duplicate_metadata"]
    assert parsed.format_valid is False
    assert parsed.status == "duplicate_metadata"
    assert parsed.metadata["general.file_type"] == 15  # first occurrence kept


def test_duplicate_tensor_name_is_refused_and_keeps_first_descriptor(tmp_path):
    path = tmp_path / "dup_tensor.gguf"
    prefix = (
        _header(3, 2, 3)
        + _standard_metadata()
        + _descriptor(_TENSOR_NAME, offset=0)
        + _descriptor(_TENSOR_NAME, offset=4096)
    )
    path.write_bytes(_aligned(prefix) + bytes(_TENSOR_BYTES))

    parsed = audit.validate_parser(path)

    assert parsed.production_scan == "rejected"
    assert _checks(parsed) == ["duplicate_tensor"]
    assert parsed.format_valid is False
    assert parsed.status == "duplicate_tensor"
    assert parsed.table_complete is True  # boundary intact; offsets stay computable
    assert len(parsed.tensors) == 1  # first descriptor kept, later duplicates dropped


@pytest.mark.parametrize(
    ("alignment", "label"),
    [(0, "zero"), (3, "three"), ("bad", "string"), ([32, 64], "array")],
    ids=["zero", "three", "string", "array"],
)
def test_invalid_alignment_is_refused_without_invented_fields(tmp_path, alignment, label):
    path = tmp_path / f"align-{label}.gguf"
    prefix = _header(3, 1, 3) + _standard_metadata(alignment=alignment) + _descriptor(_TENSOR_NAME)
    # Payload is physically padded at 32 bytes so only the alignment field is
    # defective in this fixture.
    path.write_bytes(_aligned(prefix) + bytes(_TENSOR_BYTES))

    parsed = audit.validate_parser(path)  # must not raise, for any hostile alignment

    assert parsed.production_scan == "rejected"
    assert parsed.production_scan_error is not None
    assert _checks(parsed) == ["invalid_alignment"]
    assert parsed.format_valid is False
    assert parsed.status == "invalid_alignment"
    # Fail-closed honesty: without a valid alignment the payload boundary is
    # unknown, so no data offset, no tensor records, and no completeness verdict
    # may be reported. The descriptor table itself was read fine.
    assert parsed.data_start is None
    assert parsed.data_complete is False
    assert parsed.tensors == []
    assert parsed.table_complete is True
    assert parsed.declared_tensor_count == 1


def _hostile_fixture(tmp_path: pathlib.Path, kind: str) -> tuple[pathlib.Path, str]:
    """Build one file per documented production-refusal input class.

    Returns the path and the expected production exception-name prefix.
    All classes refuse production ``scan_gguf`` through plain ValueError or
    TypeError rather than GGUFFormatError.
    """

    if kind == "alignment_string":
        prefix = _header(3, 1, 3) + _standard_metadata(alignment="bad") + _descriptor(_TENSOR_NAME)
        expected = "ValueError:"
    elif kind == "alignment_array":
        prefix = _header(3, 1, 3) + _standard_metadata(alignment=[32, 64]) + _descriptor(_TENSOR_NAME)
        expected = "TypeError:"
    elif kind == "unknown_metadata_enum":
        # Fourth declared metadata entry carries value-type id 999, which
        # production GGUFValueType() refuses before reading any value bytes.
        bad_enum = len(b"general.alignment").to_bytes(8, "little") + b"general.alignment" + (999).to_bytes(4, "little")
        prefix = _header(3, 1, 4) + _standard_metadata(extra=bad_enum) + _descriptor(_TENSOR_NAME)
        expected = "ValueError:"
    else:  # malformed_block_shape: valid Q4_K id, row 32 not a multiple of its 256 block
        prefix = (
            _header(3, 1, 3)
            + _standard_metadata()
            + _descriptor("blk.0.short_k.weight", qtype=GGMLQuantizationType.Q4_K)
        )
        expected = "ValueError:"
    path = tmp_path / f"hostile-{kind}.gguf"
    path.write_bytes(_aligned(prefix) + bytes(_TENSOR_BYTES))
    return path, expected


@pytest.mark.parametrize(
    "kind",
    ["alignment_string", "alignment_array", "unknown_metadata_enum", "malformed_block_shape"],
)
def test_hostile_metadata_files_get_cli_json_verdict_not_crash(tmp_path, kind, capsys):
    path, expected_error = _hostile_fixture(tmp_path, kind)

    with pytest.raises((ValueError, TypeError)):
        scan_gguf(path)  # production remains the authority and refuses these inputs

    # The CLI must turn that refusal into a structured verdict, not a traceback.
    out = tmp_path / "report.json"
    assert audit.main([str(path), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    entry = json.loads(out.read_text())["files"][0]

    parsed = entry["parser_validation"]
    assert parsed["production_scan"] == "rejected"
    assert parsed["production_scan_error"].startswith(expected_error)
    assert parsed["format_valid"] is False
    assert "NOT loadable" in printed


@pytest.mark.parametrize(
    ("qtype", "label"),
    [(999, "unknown-type-id"), (GGMLQuantizationType.Q4_K, "malformed-block-shape")],
    ids=["unknown-qtype", "bad-block-shape"],
)
def test_unresolvable_tensor_size_keeps_data_completeness_unknown(tmp_path, qtype, label):
    # Reviewer reproducer: declared table completes, no payload bytes follow, and
    # the single descriptor's size cannot be resolved (unknown type id, or a
    # block-layout-violating shape). Completeness is then unknown, never True.
    path = tmp_path / f"unresolvable-{label}.gguf"
    path.write_bytes(_header(3, 1, 0) + _descriptor("x", qtype=qtype))

    parsed = audit.validate_parser(path)

    assert parsed.production_scan == "rejected"
    assert parsed.table_complete is True
    assert parsed.declared_tensor_count == 1
    assert _checks(parsed) == ["unparseable"]
    assert parsed.tensors == []  # an unusable descriptor is not fabricated
    assert parsed.data_complete is False  # size unresolved: completeness NOT claimed
    assert parsed.data_start is not None  # payload boundary itself is known
    assert parsed.format_valid is False


def test_planner_qualification_error_is_reported_not_raised(tmp_path, capsys, monkeypatch):
    """A map/planner failure is a documented qualification error: error-shaped
    sections in JSON and stdout, exit 0, parser verdict untouched."""

    path = _complete_file(tmp_path)

    def broken_maps(path, metadata, tensors, version):
        raise ValueError("capability source unreadable")

    monkeypatch.setattr(audit, "build_tensor_maps", broken_maps)
    out = tmp_path / "report.json"

    assert audit.main([str(path), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    entry = json.loads(out.read_text())["files"][0]

    assert entry["tensor_map"]["diagnostic_only"] is True
    assert entry["tensor_map"]["available"] is False
    assert entry["tensor_map"]["error"].startswith("ValueError:")
    backend = entry["backends"][0]
    assert backend["diagnostic_only"] is True
    assert "routes" not in backend
    assert "planner unavailable (diagnostic only)" in printed
    assert entry["parser_validation"]["format_valid"] is True


def test_incomplete_descriptor_table_is_distinct_from_incomplete_payload(tmp_path):
    # File A: header declares two tensors, file ends inside the descriptor table.
    truncated_table = tmp_path / "truncated_table.gguf"
    truncated_table.write_bytes(_header(3, 2, 3) + _standard_metadata() + _descriptor(_TENSOR_NAME))
    # File B: descriptor table is complete, but no payload bytes follow it.
    truncated_payload = tmp_path / "truncated_payload.gguf"
    truncated_payload.write_bytes(_header(3, 1, 3) + _standard_metadata() + _descriptor(_TENSOR_NAME))

    table = audit.validate_parser(truncated_table)
    payload = audit.validate_parser(truncated_payload)

    # A: the descriptor table itself is truncated; the payload boundary is
    # unknown, so no data offset or range check is invented.
    assert table.table_complete is False
    assert _checks(table) == ["incomplete_table"]
    assert table.tensors == [] and table.data_start is None
    assert table.data_complete is False
    assert table.format_valid is False
    # B: the table is complete; only the tensor payload is missing.
    assert payload.table_complete is True
    assert payload.data_complete is False
    assert _checks(payload) == ["incomplete_data"]
    assert payload.status == "incomplete_data"
    assert payload.format_valid is False


def test_unknown_descriptor_does_not_skip_range_check_for_known_tensors(tmp_path):
    # One unresolvable descriptor must not suppress the independent payload-range
    # loop over the successfully resolved tensors: the known tensor still has no
    # payload on disk, so the file is both unparseable and incomplete.
    path = tmp_path / "unknown_and_truncated.gguf"
    path.write_bytes(_header(3, 2, 0) + _descriptor("unknown", qtype=999) + _descriptor("known"))

    parsed = audit.validate_parser(path)

    assert parsed.table_complete is True
    assert parsed.data_complete is False
    assert parsed.format_valid is False
    assert _checks(parsed) == ["unparseable", "incomplete_data"]


def test_truncated_metadata_stops_before_the_descriptor_table(tmp_path):
    path = tmp_path / "truncated_meta.gguf"
    path.write_bytes(
        _header(3, 2, 3) + _kv("general.architecture", GGUFValueType.STRING, _string("qwen35"))
    )

    parsed = audit.validate_parser(path)

    # Structural boundary lost inside metadata: no table, offsets, or ranges.
    assert _checks(parsed) == ["unparseable"]
    assert parsed.data_start is None
    assert parsed.tensors == []
    assert parsed.format_valid is False


def test_magic_only_file_is_reported_unparseable_not_loadable(tmp_path):
    path = tmp_path / "magic.gguf"
    path.write_bytes(b"GGUF")

    parsed = audit.validate_parser(path)

    assert parsed.version is None
    assert parsed.format_valid is False
    assert _checks(parsed) == ["unparseable"]


def test_partial_file_report_labels_not_loadable_and_sections_print_safely(tmp_path, capsys):
    path = tmp_path / "partial.gguf"
    path.write_bytes(_header(3, 1, 3) + _standard_metadata() + _descriptor(_TENSOR_NAME))
    out = tmp_path / "report.json"

    assert audit.main([str(path), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    entry = json.loads(out.read_text())["files"][0]

    parsed = entry["parser_validation"]
    assert parsed["format_valid"] is False
    assert parsed["status"] == "incomplete_data"
    # The printed verdict labels the file explicitly as not loadable.
    assert "NOT loadable" in printed
    assert "incomplete_data" in printed
    # Diagnostic route sections are still reported for inspection, and the
    # tensor-map section reports its failure as diagnostic-only rather than
    # crashing before the JSON is written (tiny fixture has no model metadata).
    assert "backends" in entry and "tensor_map" in entry
    assert entry["tensor_map"]["diagnostic_only"] is True
    assert entry["tensor_map"]["available"] is False
    for backend in entry["backends"]:
        assert "routes" not in backend


def test_missing_file_reports_structured_verdict_without_crashing(tmp_path, capsys):
    path = tmp_path / "missing.gguf"
    out = tmp_path / "report.json"

    assert audit.main([str(path), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    entry = json.loads(out.read_text())["files"][0]

    parsed = entry["parser_validation"]
    assert parsed["format_valid"] is False
    assert "unopenable" in parsed["status"]
    # The size is unknown for a missing path; it is never claimed to be zero.
    assert entry["file_size_on_disk_bytes"] is None
    assert "NOT loadable" in printed


# --- UD-U0b production tensor maps ---------------------------------------------
#
# Route tables come from the actual production AR map and the actual runtime
# NextN map, not from guessed slot names over all disk tensors. Fixtures follow
# the synthetic GGUFModelInfo pattern of tests/test_qwen35_gguf_mtp_mapping.py:
# tiny deterministic shapes, no local model dependency, no weights read.

from math import prod  # noqa: E402

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo  # noqa: E402

# 64 AR layers (8 full-attention at interval 8, 56 linear) + trailing NextN
# block 64. Dense untied head is intentionally absent: the head is tied, so
# root.lm_head and root.token_embedding share one physical source.
_QWEN35_METADATA = {
    "general.architecture": "qwen35",
    "general.file_type": 15,
    "qwen35.block_count": 65,
    "qwen35.embedding_length": 8,
    "qwen35.feed_forward_length": 5,
    "qwen35.context_length": 128,
    "qwen35.attention.head_count": 2,
    "qwen35.attention.head_count_kv": 1,
    "qwen35.attention.key_length": 4,
    "qwen35.attention.value_length": 4,
    "qwen35.full_attention_interval": 8,
    "qwen35.rope.dimension_count": 4,
    "qwen35.rope.dimension_sections": (),
    "qwen35.ssm.inner_size": 16,
    "qwen35.ssm.group_count": 2,
    "qwen35.ssm.state_size": 3,
    "qwen35.ssm.conv_kernel": 4,
    "qwen35.ssm.time_step_rank": 2,
}

_AR_LAYERS = 64
_FULL_INTERVAL = 8


def _t(
    name: str,
    shape: tuple[int, ...],
    qtype: GGMLQuantizationType = GGMLQuantizationType.F32,
) -> "object":
    n_elements = int(prod(shape))
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(qtype),
        ggml_type_name=qtype.name,
        n_elements=n_elements,
        nbytes=n_elements * (4 if qtype == GGMLQuantizationType.F32 else 2),
        offset=0,
        data_offset=0,
        byte_shape=shape,
    )


def _ar_layer_tensors(layer_id: int) -> list:
    prefix = f"blk.{layer_id}"
    full = (layer_id + 1) % _FULL_INTERVAL == 0
    tensors = [
        _t(f"{prefix}.attn_norm.weight", (8,)),
        _t(f"{prefix}.post_attention_norm.weight", (8,)),
    ]
    if full:
        tensors += [
            _t(f"{prefix}.attn_q.weight", (16, 8)),
            _t(f"{prefix}.attn_k.weight", (4, 8)),
            _t(f"{prefix}.attn_v.weight", (4, 8)),
            _t(f"{prefix}.attn_output.weight", (8, 8)),
            _t(f"{prefix}.attn_q_norm.weight", (4,)),
            _t(f"{prefix}.attn_k_norm.weight", (4,)),
        ]
    else:
        tensors += [
            _t(f"{prefix}.attn_gate.weight", (16, 8)),
            _t(f"{prefix}.attn_qkv.weight", (28, 8)),
            _t(f"{prefix}.ssm_a", (2,)),
            _t(f"{prefix}.ssm_alpha.weight", (2, 8)),
            _t(f"{prefix}.ssm_beta.weight", (2, 8)),
            _t(f"{prefix}.ssm_conv1d.weight", (28, 4)),
            _t(f"{prefix}.ssm_dt.bias", (2,)),
            _t(f"{prefix}.ssm_norm.weight", (3,)),
            _t(f"{prefix}.ssm_out.weight", (8, 16)),
        ]
    return tensors + [
        _t(f"{prefix}.ffn_gate.weight", (5, 8)),
        _t(f"{prefix}.ffn_up.weight", (5, 8)),
        _t(f"{prefix}.ffn_down.weight", (8, 5)),
    ]


def _nextn_block_tensors(
    *,
    with_optionals: bool,
    eh_proj_qtype: GGMLQuantizationType = GGMLQuantizationType.Q8_0,
    drop: set[str] = frozenset(),
) -> list:
    # Dense NextN dtypes follow hipengine.loading.qwen35_gguf_nextn's
    # _EXPECTED_DENSE_QTYPES so the default fixture validates clean.
    tensors = [
        _t("blk.64.attn_norm.weight", (8,)),
        _t("blk.64.post_attention_norm.weight", (8,)),
        _t("blk.64.attn_q.weight", (16, 8), GGMLQuantizationType.Q4_K),
        _t("blk.64.attn_k.weight", (4, 8), GGMLQuantizationType.Q4_K),
        _t("blk.64.attn_v.weight", (4, 8), GGMLQuantizationType.Q6_K),
        _t("blk.64.attn_output.weight", (8, 8), GGMLQuantizationType.Q4_K),
        _t("blk.64.attn_q_norm.weight", (4,)),
        _t("blk.64.attn_k_norm.weight", (4,)),
        _t("blk.64.ffn_gate.weight", (5, 8), GGMLQuantizationType.Q4_K),
        _t("blk.64.ffn_up.weight", (5, 8), GGMLQuantizationType.Q4_K),
        _t("blk.64.ffn_down.weight", (8, 5), GGMLQuantizationType.Q6_K),
        _t("blk.64.nextn.eh_proj.weight", (8, 16), eh_proj_qtype),
        _t("blk.64.nextn.enorm.weight", (8,)),
        _t("blk.64.nextn.hnorm.weight", (8,)),
        _t("blk.64.nextn.shared_head_norm.weight", (8,)),
    ]
    if with_optionals:
        tensors += [
            _t("blk.64.nextn.embed_tokens.weight", (11, 8), GGMLQuantizationType.Q4_K),
            _t("blk.64.nextn.shared_head_head.weight", (11, 8), GGMLQuantizationType.Q6_K),
        ]
    return [tensor for tensor in tensors if tensor.name not in drop]


def _fixture_tensors(
    *,
    with_optionals: bool = False,
    eh_proj_qtype: GGMLQuantizationType = GGMLQuantizationType.Q8_0,
    embedding_qtype: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
    drop: set[str] = frozenset(),
) -> list:
    tensors = [
        _t("token_embd.weight", (11, 8), embedding_qtype),
        _t("output_norm.weight", (8,)),
    ]
    for layer_id in range(_AR_LAYERS):
        tensors += _ar_layer_tensors(layer_id)
    tensors += _nextn_block_tensors(
        with_optionals=with_optionals, eh_proj_qtype=eh_proj_qtype, drop=drop
    )
    return [tensor for tensor in tensors if tensor.name not in drop]


def _fixture_info(tensors: list) -> GGUFModelInfo:
    return GGUFModelInfo(
        path=pathlib.Path("synthetic-qwen35-ud-map.gguf"),
        version=3,
        alignment=32,
        metadata=dict(_QWEN35_METADATA),
        tensors=tuple(tensors),
        tensor_data_offset=0,
    )


@pytest.fixture(autouse=True)
def _clean_repack_env(monkeypatch):
    # Deterministic routing: the decode-repack default is env-controlled. The
    # parser-only tests above do not read this variable.
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_REPACK", raising=False)


def _mapped(tensors: list, drop_from_map: set[str] = frozenset()):
    kept = [t for t in tensors if t.name not in drop_from_map]
    return audit.build_tensor_maps(
        pathlib.Path("synthetic-qwen35-ud-map.gguf"),
        dict(_QWEN35_METADATA),
        kept,
        3,
    )


def test_production_map_reports_64_ar_layers_and_ignores_block64_nextn():
    tensors = _fixture_tensors()
    maps = _mapped(tensors)
    section = maps.section

    assert section["available"] is True
    assert section["architecture"] == "qwen35"
    assert section["validation_passed"] is True
    ar = section["ar"]
    assert ar["layers"] == _AR_LAYERS
    assert ar["layer_types"] == {"full_attention": 8, "linear_attention": 56}
    assert ar["ignored_block_ids"] == [64]
    # Logical consumers vs unique physical sources: 3 root slots + 56x14 linear
    # + 8x11 full-attention layer slots = 875 consumers; the tied head shares
    # the embedding source, so 874 unique sources.
    assert ar["consumer_slots"] == 875
    assert ar["unique_sources"] == 874
    assert len(section["validation"]["ignored"]) == 15
    # Ignored (AR-excluded) block-64 tensors: 11 layer + 4 nextn.* tensors.
    assert section["ignored"] == {
        "tensor_count": 15,
        "block_ids": [64],
        "nextn_tensor_count": 4,
    }
    # Disk accounting: 2 root tensors + 872 layer tensors + 15 block64 tensors.
    assert section["tensors_on_disk"] == 889

    nextn = section["nextn"]
    assert nextn["blocks"] == 1 and nextn["block_id"] == 64
    assert nextn["validation_passed"] is True
    assert nextn["own_consumer_slots"] == 15
    assert nextn["own_unique_sources"] == 15
    assert nextn["fallback_consumer_slots"] == 3
    # Optional embed/head absent: their fallback slots borrow AR root sources.
    assert sorted(nextn["ar_borrowed_fallback_slots"]) == [
        "root.lm_head",
        "root.token_embedding",
    ]
    # The block-local shared_head_norm is present, so that fallback is not an
    # AR borrow; it aliases the block's own tensor.
    fallback_by_slot = {r["slot_path"]: r for r in nextn["fallback_slots"]}
    assert fallback_by_slot["root.output_norm"]["source"] == (
        "blk.64.nextn.shared_head_norm.weight"
    )
    assert fallback_by_slot["root.output_norm"]["borrows_ar_root"] is False
    assert fallback_by_slot["root.token_embedding"]["source"] == "token_embd.weight"
    assert fallback_by_slot["root.token_embedding"]["borrows_ar_root"] is True

    # Combined ownership counts shared sources once: 893 logical consumers but
    # 889 unique physical sources == all disk tensors mapped exactly once.
    combined = section["combined"]
    assert combined["consumer_slots"] == 893
    assert combined["unique_sources"] == 889
    combined_aliases = {a["source"]: a for a in combined["aliases"]}
    assert set(combined_aliases) == {
        "token_embd.weight",
        "blk.64.nextn.shared_head_norm.weight",
    }
    # The tied head: root.token_embedding and root.lm_head, plus the two NextN
    # fallback slots that borrow it when the optional block-local tensors are
    # absent -- four logical consumers of one physical source.
    assert combined_aliases["token_embd.weight"]["consumer_count"] == 4
    ar_aliases = {a["source"]: a for a in ar["aliases"]}
    assert ar_aliases["token_embd.weight"]["consumer_count"] == 2


def test_ar_embedding_routes_raw_q4_at_root_token_embedding_not_legacy_embd():
    tensors = _fixture_tensors()
    maps = _mapped(tensors)
    section = maps.section

    # Exact production root slot paths, not the legacy guessed spelling.
    assert section["ar"]["root_slots"]["root.token_embedding"] == "token_embd.weight"
    assert "root.token_embd" not in json.dumps(section)
    assert not hasattr(audit, "slot_path")

    backend = audit.plan("hip_gfx1100", dict(_QWEN35_METADATA), maps)
    # Raw Q4 embedding under default decode-repack: the legacy guessed slot
    # `root.token_embd` was not a token-embedding slot and reported pack8. The
    # tied lm_head consumer is planned per its own slot path and production
    # materializes it SEPARATELY (pack8): the (source, layout) alias dedup only
    # merges residents when layouts match, so source reuse is not layout
    # identity -- exactly the duplicate-resident hazard this table must surface.
    assert backend["routes"]["Q4_K"] == {"raw-gguf-kernel": 1, "kernel:gguf_q4_k": 1}
    # Every AR F32 layer slot stays f32-resident and nothing is rejected.
    assert backend["routes"]["F32"] == {"f32-resident": 873}
    assert backend["rejected_tensors"] == 0
    assert backend["scope"] == "ar_map_plus_nextn_map"
    assert backend["planner_mode"] == "production_planner"
    assert backend["map_validation_passed"] is True
    assert backend["ar_consumer_slots"] == 875
    assert backend["ar_unique_sources"] == 874
    # MTP-only tensors never enter AR route counts.
    assert all("layers.64." not in slot for slots in backend["rejections"].values() for slot in slots)
    # NextN routes are separate: 15 own slots plan clean; the three fallback
    # slots resolve to the embedding (raw), the tied head (pack8) and the
    # block-local shared_head_norm (f32).
    assert backend["nextn_routes"]["own"]["rejected_tensors"] == 0
    assert backend["nextn_routes"]["fallback"]["routes"]["Q4_K"] == {
        "raw-gguf-kernel": 1,
        "kernel:gguf_q4_k": 1,
    }


def test_mtp_only_tensors_are_excluded_from_ar_planner_counts():
    # An unsupported draft dtype (IQ3_S eh_proj) must surface in the NextN
    # scope only: the AR report keeps zero rejections.
    tensors = _fixture_tensors(eh_proj_qtype=GGMLQuantizationType.IQ3_S)
    maps = _mapped(tensors)
    section = maps.section

    assert section["validation_passed"] is True  # AR map unaffected
    backend = audit.plan("hip_gfx1100", dict(_QWEN35_METADATA), maps)
    assert backend["rejected_tensors"] == 0
    assert backend["routes"]["F32"] == {"f32-resident": 873}

    nextn = backend["nextn_routes"]
    assert nextn["own"]["rejected_tensors"] == 1
    assert list(nextn["own"]["rejections"]) == ["IQ3_S"]
    assert nextn["own"]["rejections"]["IQ3_S"][0].startswith("draft.nextn.eh_proj")
    # The NextN map validation records the dtype refusal diagnostically.
    assert maps.section["nextn"]["validation_passed"] is False
    assert any("eh_proj" in e for e in maps.section["nextn"]["dtype_errors"])


def test_optional_nextn_absent_fallbacks_borrow_ar_roots_deduplicated():
    tensors = _fixture_tensors()
    maps = _mapped(tensors)
    section = maps.section

    # AR roots borrowed by NextN fallbacks are counted once: unique combined
    # sources equal every distinct physical tensor on disk.
    combined = section["combined"]
    assert combined["unique_sources"] == section["tensors_on_disk"]
    borrowed = {r["slot_path"]: r for r in section["nextn"]["fallback_slots"]}
    assert borrowed["root.lm_head"]["borrows_ar_root"] is True
    assert borrowed["root.lm_head"]["source"] == "token_embd.weight"  # tied head
    assert borrowed["root.token_embedding"]["borrows_ar_root"] is True


def test_optional_nextn_present_fallbacks_alias_block_local_sources():
    # Parent edge: when the optional block-local tensors are present, the
    # fallback slots select them; dedup is by actual source identity and NO
    # fallback counts as AR borrowing -- even though the slot is named like an
    # AR root.
    tensors = _fixture_tensors(with_optionals=True)
    maps = _mapped(tensors)
    section = maps.section

    nextn = section["nextn"]
    assert nextn["own_consumer_slots"] == 17  # 15 required + 2 present optionals
    assert nextn["own_unique_sources"] == 17
    assert nextn["ar_borrowed_fallback_slots"] == []
    fallback_by_slot = {r["slot_path"]: r for r in nextn["fallback_slots"]}
    assert fallback_by_slot["root.token_embedding"]["source"] == (
        "blk.64.nextn.embed_tokens.weight"
    )
    assert fallback_by_slot["root.lm_head"]["source"] == (
        "blk.64.nextn.shared_head_head.weight"
    )
    assert fallback_by_slot["root.output_norm"]["source"] == (
        "blk.64.nextn.shared_head_norm.weight"
    )
    # The same block-local norm appears in nextn_tensors and fallback_tensors;
    # the alias record must show both consumers of the one source (scope-keyed:
    # the draft's own slot and the root-shaped fallback slot).
    aliases = {a["source"]: a for a in nextn["aliases"]}
    assert aliases["blk.64.nextn.shared_head_norm.weight"]["consumer_slots"] == [
        "nextn:draft.nextn.shared_head_norm",
        "nextn_fallback:root.output_norm",
    ]
    combined = section["combined"]
    assert combined["consumer_slots"] == 895
    assert combined["unique_sources"] == 891
    assert combined["unique_sources"] == section["tensors_on_disk"]
    # token_embd (2 AR consumers) + the three block-local optionals/norm (own
    # slot + fallback slot each).
    assert len(combined["aliases"]) == 4


def test_nextn_refusal_does_not_discard_valid_ar_report():
    tensors = _fixture_tensors(drop={"blk.64.nextn.eh_proj.weight"})
    maps = _mapped(tensors)
    section = maps.section

    assert section["validation_passed"] is True
    assert section["ar"]["consumer_slots"] == 875
    nextn = section["nextn"]
    assert nextn["validation_passed"] is False
    assert nextn["diagnostic"] is True
    assert "blk.64.nextn.eh_proj.weight" in nextn["missing"]
    assert nextn["own_consumer_slots"] == 14

    backend = audit.plan("hip_gfx1100", dict(_QWEN35_METADATA), maps)
    assert backend["map_validation_passed"] is True
    assert backend["routes"]["F32"] == {"f32-resident": 873}
    assert backend["routes"]["Q4_K"] == {"raw-gguf-kernel": 1, "kernel:gguf_q4_k": 1}
    assert backend["rejected_tensors"] == 0
    # The incomplete draft still routes its present tensors.
    assert backend["nextn_routes"]["own"]["rejected_tensors"] == 0


def test_ar_slot_refusals_are_collected_per_slot_not_all_or_nothing():
    # A Q3_K embedding (the published K_S shape of refusal) makes the whole-AR
    # production plan raise on its first unsupported slot. The audit must fall
    # back to per-slot planning, keep every other AR route, and report BOTH
    # tied-head consumers of the refused source as distinct slot rejections
    # instead of discarding the AR report.
    tensors = _fixture_tensors(embedding_qtype=GGMLQuantizationType.Q3_K)
    maps = _mapped(tensors)
    section = maps.section

    # The map itself is structurally valid; refusal is a planner verdict.
    assert section["validation_passed"] is True
    backend = audit.plan("hip_gfx1100", dict(_QWEN35_METADATA), maps)

    assert backend["planner_mode"] == "per_slot_fallback"
    assert backend["map_validation_passed"] is True
    assert backend["rejected_tensors"] == 2
    assert list(backend["rejections"]) == ["Q3_K"]
    assert all(
        entry.startswith(prefix)
        for entry, prefix in zip(
            sorted(backend["rejections"]["Q3_K"]),
            sorted(["root.lm_head", "root.token_embedding"]),
        )
    )
    # Every other AR slot still routed: 873 F32 residents, nothing else lost.
    assert backend["routes"]["F32"] == {"f32-resident": 873}
    assert backend["ar_consumer_slots"] == 875
    # The refusal is a route-table verdict only; NextN fallback slots that
    # borrow the same source refuse identically and never touch AR counts.
    assert backend["nextn_routes"]["own"]["rejected_tensors"] == 0
    assert backend["nextn_routes"]["fallback"]["rejected_tensors"] == 2


def test_map_unavailable_produces_no_guessed_route_table(tmp_path, capsys):
    # Byte-level CLI fixture: a format-valid file with no qwen35 model metadata.
    # The map cannot be built, so no route table may be produced from guessed
    # slot names, and nothing may call the file consumer-qualified.
    path = _complete_file(tmp_path, name="no-model-metadata.gguf")
    out = tmp_path / "report.json"

    assert audit.main([str(path), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    entry = json.loads(out.read_text())["files"][0]

    assert entry["parser_validation"]["format_valid"] is True
    assert entry["tensor_map"]["available"] is False
    assert entry["tensor_map"]["diagnostic_only"] is True
    assert "qwen35.block_count" in entry["tensor_map"]["error"]
    for backend in entry["backends"]:
        assert backend["diagnostic_only"] is True
        assert "routes" not in backend
        assert "nextn_routes" not in backend
    assert "raw-gguf" not in printed
    assert "f32-resident" not in printed
    assert "consumer-qualified" not in printed
