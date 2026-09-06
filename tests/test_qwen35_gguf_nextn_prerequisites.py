"""Whole-plan NextN prerequisite ordering on real tiny GGUF CPU fixtures."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading import materialize as host
from hipengine.loading import qwen35_gguf_materialize as weights
from hipengine.loading import qwen35_gguf_nextn_materialize as nextn
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.gguf_mtp_hot_vocab import gguf_tokenizer_tokens_sha256
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_nextn import build_qwen35_gguf_nextn_tensor_map
from hipengine.quant.gguf import GGMLQuantizationType as Q
from tests._qwen35_gguf_fixture import (
    default_fixture_tensors, fixture_metadata, write_qwen35_gguf,
)


def _write_nextn(path, vocab=65):
    tensors = default_fixture_tensors(1)
    tensors[0] = ("token_embd.weight", (vocab, 256), Q.Q8_0)
    tensors.append(("output.weight", (vocab, 256), Q.Q6_K))
    # Explicit independent NextN fixture: head_count=4, key_length=64 gives
    # q_width=256, so every draft Q4/Q6 K is block-aligned and every N is legal.
    tensors += [("blk.1." + name, shape, qtype) for name, shape, qtype in (
        ("attn_norm.weight", (256,), Q.F32),
        ("post_attention_norm.weight", (256,), Q.F32),
        ("attn_q.weight", (512, 256), Q.Q4_K),
        ("attn_k.weight", (64, 256), Q.Q4_K),
        ("attn_v.weight", (64, 256), Q.Q6_K),
        ("attn_output.weight", (256, 256), Q.Q4_K),
        ("attn_q_norm.weight", (64,), Q.F32),
        ("attn_k_norm.weight", (64,), Q.F32),
        ("ffn_gate.weight", (512, 256), Q.Q4_K),
        ("ffn_up.weight", (512, 256), Q.Q4_K),
        ("ffn_down.weight", (256, 512), Q.Q6_K),
        ("nextn.eh_proj.weight", (256, 512), Q.Q8_0),
        ("nextn.enorm.weight", (256,), Q.F32),
        ("nextn.hnorm.weight", (256,), Q.F32),
        ("nextn.shared_head_norm.weight", (256,), Q.F32),
    )]
    overrides = {"qwen35.block_count": 2, "qwen35.attention.head_count": 4}
    metadata = [(key, kind, overrides.get(key, value)) for key, kind, value in fixture_metadata(1)]
    metadata.append(("tokenizer.ggml.tokens", 9, (8, [f"t{i}" for i in range(vocab)])))
    write_qwen35_gguf(path, tensors, metadata)
    reader = GGUFReader(path)
    assert build_qwen35_gguf_tensor_map(reader.info).validation.passed
    assert build_qwen35_gguf_nextn_tensor_map(reader.info).validation.passed
    return reader


def _forbid_payload_and_allocation(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        pytest.fail("NextN touched payload/allocation before whole-plan refusal")

    monkeypatch.setattr(nextn, "materialize_qwen35_gguf_weight_spec", forbidden)
    monkeypatch.setattr(GGUFReader, "tensor_data", forbidden)
    monkeypatch.setattr(host, "malloc", forbidden)
    monkeypatch.setattr(weights, "malloc", forbidden)
    monkeypatch.setattr(nextn, "load_host_array_to_device_as_dtype", forbidden)
    return calls


def _borrow_head(reader, layout=weights.LAYOUT_RAW_GGUF):
    source = reader.info.tensor("output.weight")
    spec = weights.plan_qwen35_gguf_weight_spec("root.lm_head", source, decode_repack=False)
    assert spec.layout == weights.LAYOUT_RAW_GGUF
    if layout != weights.LAYOUT_RAW_GGUF:
        spec = replace(spec, layout=layout, quant_key=layout, allocation_names=("tiles",))
    borrowed = SimpleNamespace(spec=spec, backend="hip_gfx1100", frees=[])
    borrowed.free = lambda **kwargs: borrowed.frees.append(True)
    return borrowed


def test_late_owned_nextn_head_refuses_before_any_payload_or_allocation(monkeypatch, tmp_path):
    reader = _write_nextn(tmp_path / "n65.gguf")
    calls = _forbid_payload_and_allocation(monkeypatch)
    with pytest.raises(ValueError, match="root.lm_head.*divisible by 16"):
        nextn.materialize_qwen35_gguf_nextn_weights(reader, decode_repack=True)
    assert calls == []


def test_invalid_actual_borrowed_nextn_head_refuses_before_loading(monkeypatch, tmp_path):
    reader = _write_nextn(tmp_path / "borrowed-invalid.gguf")
    borrowed = _borrow_head(reader, weights.LAYOUT_GGUF_Q6_K_T16)
    calls = _forbid_payload_and_allocation(monkeypatch)
    with pytest.raises(ValueError, match="root.lm_head.*divisible by 16"):
        nextn.materialize_qwen35_gguf_nextn_weights(
            reader, borrowed_fallback_weights={"lm_head": borrowed},
        )
    assert calls == []
    assert borrowed.frees == []


def _cpu_uploads(monkeypatch, *, fail_hot_token_upload=False):
    allocations = []

    def upload(name, array, dtype, **kwargs):
        if fail_hot_token_upload and ".mtp_hot_vocab" in name and name.endswith(".token_ids"):
            raise RuntimeError("injected hot token upload failure")
        allocation = SimpleNamespace(
            name=name, array=np.asarray(array).copy(), frees=0,
            tensor=SimpleNamespace(dtype=dtype, ptr=0x1000 + len(allocations) * 0x100),
        )

        def free(**kwargs):
            allocation.frees += 1

        allocation.free = free
        allocations.append(allocation)
        return allocation

    monkeypatch.setattr(weights, "_load_host_array_to_device_as_dtype", upload)
    monkeypatch.setattr(nextn, "load_host_array_to_device_as_dtype", upload)
    return allocations


@pytest.mark.parametrize("vocab,borrow", [(64, False), (65, True)])
def test_legal_owned_neighbor_and_actual_borrowed_raw_head_keep_ownership(
    monkeypatch, tmp_path, vocab, borrow,
):
    reader = _write_nextn(tmp_path / "legal.gguf", vocab=vocab)
    borrowed = _borrow_head(reader) if borrow else None
    # Prove N65's unborrowed replacement really would be illegal, while the
    # actual selected raw borrowed resident is legal and must not be rejected.
    if borrow:
        plan = nextn.plan_qwen35_gguf_nextn_materialization(
            build_qwen35_gguf_nextn_tensor_map(reader.info), decode_repack=True,
        )
        with pytest.raises(ValueError, match="divisible by 16"):
            weights.validate_qwen35_gguf_resident_prerequisites(plan.fallback_specs["lm_head"])
    allocations = _cpu_uploads(monkeypatch)
    reads = []
    original = GGUFReader.tensor_data

    def read(self, name):
        reads.append(name)
        return original(self, name)

    monkeypatch.setattr(GGUFReader, "tensor_data", read)
    resident = nextn.materialize_qwen35_gguf_nextn_weights(
        reader, borrowed_fallback_weights={"lm_head": borrowed} if borrow else None,
        hot_vocab_path="auto",  # unknown fixture identity must still miss packaged maps
    )
    assert allocations
    assert resident.hot_vocab is None
    assert ("output.weight" in reads) == (not borrow)
    if borrow:
        assert resident.fallback("lm_head") is borrowed
        assert resident.plan.fallback_specs["lm_head"] is borrowed.spec
        assert all(weight is not borrowed for weight in resident.owned_weights)
    else:
        assert resident.fallback("lm_head").spec.layout == weights.LAYOUT_GGUF_Q6_K_T16
    # Shared NextN head norm is still an alias, with one owning allocation.
    assert resident.fallback("output_norm") is resident.nextn("shared_head_norm")
    resident.free()
    assert all(allocation.frees == 1 for allocation in allocations)
    if borrow:
        assert borrowed.frees == []


def _hot_map(path, reader, size=16):
    path.write_text(json.dumps({
        "schema_version": 1, "kind": "hipengine.gguf_mtp_hot_vocab",
        "model": {
            "vocab_size": len(reader.info.metadata["tokenizer.ggml.tokens"]),
            "tokenizer_tokens_sha256": gguf_tokenizer_tokens_sha256(reader.info),
            "architecture": reader.info.metadata["general.architecture"],
            "basename": reader.info.metadata.get("general.basename"),
            "block_count": reader.info.metadata["qwen35.block_count"],
            "file_type": reader.info.file_type_name,
        },
        "token_ids": list(range(size)),
    }))
    return path


@pytest.mark.parametrize("size", [15, 16])
def test_hot_vocab_selection_and_resident_checks_precede_all_loading(monkeypatch, tmp_path, size):
    reader = _write_nextn(tmp_path / "hot.gguf", vocab=64)
    hot_path = _hot_map(tmp_path / "hot.json", reader, size)
    calls = _forbid_payload_and_allocation(monkeypatch)
    # size15 fails the real selection reader; size16 has a valid map but the
    # actual head is standard T16, not the compact head builder's planar input.
    with pytest.raises(ValueError, match="divisible by 16|planar Q6 T16"):
        nextn.materialize_qwen35_gguf_nextn_weights(reader, hot_vocab_path=hot_path)
    assert calls == []


@pytest.mark.parametrize("bad", ["compact_n", "source_vocab", "token_bounds"])
def test_direct_hot_vocab_shape_checks_precede_payload(monkeypatch, tmp_path, bad):
    from hipengine.loading.gguf_mtp_hot_vocab import GGUFHotVocabSelection

    reader = _write_nextn(tmp_path / "direct-hot.gguf", vocab=64)
    borrowed = _borrow_head(reader, weights.LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR)
    selection = GGUFHotVocabSelection(
        tuple(range(15 if bad == "compact_n" else 16)), "fixture", tmp_path / "hot.json", {},
    )
    if bad == "token_bounds":
        selection = replace(selection, token_ids=(*range(15), 64))
    if bad == "source_vocab":
        # A valid resident from a different tokenizer extent is not a valid
        # source for this selection; no payload read should discover that.
        from tests.test_gguf_ud_admission import _tensor
        borrowed.spec = replace(borrowed.spec, source=_tensor("output.weight", (32, 256), Q.Q6_K))
    calls = _forbid_payload_and_allocation(monkeypatch)
    with pytest.raises(ValueError, match="divisible by 16|unexpected shape|outside"):
        nextn._materialize_hot_vocab(
            reader, borrowed.spec, selection, device=None, runtime=None, backend="hip_gfx1100",
        )
    assert calls == []


@pytest.mark.parametrize("upload_failure", [False, True])
def test_hot_vocab_uses_resolved_borrowed_planar_head_and_keeps_cleanup(
    monkeypatch, tmp_path, upload_failure,
):
    reader = _write_nextn(tmp_path / "hot-planar.gguf", vocab=64)
    borrowed = _borrow_head(reader, weights.LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR)
    hot_path = _hot_map(tmp_path / "hot.json", reader)
    allocations = _cpu_uploads(monkeypatch, fail_hot_token_upload=upload_failure)
    kwargs = dict(borrowed_fallback_weights={"lm_head": borrowed}, hot_vocab_path=hot_path)
    if upload_failure:
        with pytest.raises(RuntimeError, match="injected hot token upload failure"):
            nextn.materialize_qwen35_gguf_nextn_weights(reader, **kwargs)
    else:
        resident = nextn.materialize_qwen35_gguf_nextn_weights(reader, **kwargs)
        assert resident.fallback("lm_head") is borrowed
        assert resident.hot_vocab.size == 16
        # CPU converter/real upload adapter output, with only device allocation
        # mocked; metadata source remains the full source (existing provenance).
        from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
        raw = np.asarray(reader.tensor_data("output.weight"))
        expected = repack_gguf_q6_k_tile16_qmicro_planar(raw[:16][None, ...]).tiles
        np.testing.assert_array_equal(resident.hot_vocab.lm_head.allocation("tiles").array, expected)
        np.testing.assert_array_equal(resident.hot_vocab.token_ids.array, np.arange(16, dtype=np.int32))
        resident.free()
    assert allocations
    assert all(allocation.frees == 1 for allocation in allocations)
    assert borrowed.frees == []
