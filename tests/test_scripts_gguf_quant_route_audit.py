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


def test_slot_path_matches_the_loader_slot_names():
    assert audit.slot_path("token_embd.weight") == "root.token_embd"
    assert audit.slot_path("output.weight") == "root.lm_head"
    assert audit.slot_path("blk.3.ffn_down.weight") == "layers.3.ffn_down"
    assert audit.slot_path("blk.48.nextn.shared_head_head.weight") == "layers.48.nextn.shared_head_head"
    # An unmapped name must not raise: the planner reports it as an unexpected slot.
    assert audit.slot_path("some.other.tensor") == "some.other.tensor"


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
    # The tiny fixture is not a complete model, so the plugin-map section must
    # report its failure as diagnostic-only instead of crashing the CLI.
    assert entry["plugin_tensor_map"]["diagnostic_only"] is True
    assert "error" in entry["plugin_tensor_map"]


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
    """A planner failure is a documented qualification error: error-shaped
    backend entries in JSON and stdout, exit 0, parser verdict untouched."""

    path = _complete_file(tmp_path)

    def broken_planner(backend: str, metadata: dict, tensors: list) -> dict:
        raise ValueError("capability source unreadable")

    monkeypatch.setattr(audit, "plan", broken_planner)
    out = tmp_path / "report.json"

    assert audit.main([str(path), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    entry = json.loads(out.read_text())["files"][0]

    backend = entry["backends"][0]
    assert backend["diagnostic_only"] is True
    assert backend["error"].startswith("ValueError:")
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
    # plugin-map section reports its failure as diagnostic-only rather than
    # crashing before the JSON is written (tiny fixture has no model metadata).
    assert "backends" in entry and "plugin_tensor_map" in entry
    assert entry["plugin_tensor_map"]["diagnostic_only"] is True
    assert "error" in entry["plugin_tensor_map"]


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
