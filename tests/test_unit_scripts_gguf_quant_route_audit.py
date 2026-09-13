"""Guards for the GGUF quant-route audit used by docs/UD-QUANTS.md.

The script reads backend capability constants out of kernel package source instead
of importing the package, because importing it registers kernels and needs the HIP
runtime. Two bugs found while writing it are pinned here: reading
``frozenset({"mostly_q4_k_s"})`` as a single token, which silently reported the
FP16-recurrent-state default as off on the backend that turns it on for Q4_K_S
files; and treating a capability defined by only one backend as "renamed", which
made the constant check fail on the backend that legitimately has no such default.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gguf_quant_route_audit.py"

spec = importlib.util.spec_from_file_location("gguf_quant_route_audit", SCRIPT)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
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

    from struct import pack

    from hipengine.loading.gguf import GGUFValueType
    from hipengine.quant.gguf import GGMLQuantizationType

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
