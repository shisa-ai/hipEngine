import pytest

from scripts.qwen4exp_framework_family_refresh import (
    HOST_ID,
    TAXONOMY,
    annotated_sections,
    baseline_commands,
    check_capture_identity,
    normalize_hip_roles,
    summarize_sections,
)


def section(nodes="0x1", op="MUL_MAT_ID q5_1", us=10):
    suffix = f" HE_NODES={nodes}" if nodes else ""
    return f"Vulkan Timings:\n{op}{suffix}: 1 x {us} us = {us} us\nTotal time: {us} us.\n"


def test_metadata_is_bound_to_each_graph_not_last_pointer_value():
    text = "HE_OWNER 0x1 moe MUL_MAT_ID 66666e -\n" + section()
    text += "HE_OWNER 0x1 gdn GATED_DELTA_NET 67646e -\n" + section(op="GATED_DELTA_NET")
    result = annotated_sections(text)
    assert result[0]["rows"][0]["owner"] == "moe"
    assert result[1]["rows"][0]["owner"] == "gdn"
    summary = summarize_sections(result)
    assert summary["matched_gap_eligible"]
    assert summary["owner_ms"]["moe"] == pytest.approx(0.01)
    assert summary["owner_ms"]["gdn"] == pytest.approx(0.01)


def test_cross_family_fusion_and_missing_metadata_are_not_guessed():
    text = "HE_OWNER 0x1 linear MUL_MAT - -\nHE_OWNER 0x2 gr_read UNARY - -\n"
    result = summarize_sections(annotated_sections(text + section("0x1,0x2")))
    assert not result["matched_gap_eligible"]
    assert result["owner_ms"]["mixed_fused"] == pytest.approx(0.01)
    unknown = summarize_sections(annotated_sections(section()))
    assert unknown["owner_ms"]["unclassified"] == pytest.approx(0.01)
    assert not unknown["matched_gap_eligible"]


def test_empty_fusion_members_do_not_create_false_mixed_cost():
    text = "HE_OWNER 0x1 moe MUL_MAT_ID - -\nHE_OWNER 0x2 boundary VIEW - -\n"
    assert summarize_sections(annotated_sections(text + section("0x1,0x2")))["matched_gap_eligible"]


def test_reject_unfinished_duplicate_and_unreconciled_timings():
    with pytest.raises(ValueError, match="unfinished"):
        annotated_sections("Vulkan Timings:\n")
    with pytest.raises(ValueError, match="twice"):
        annotated_sections(section("0x1,0x1"))
    with pytest.raises(ValueError, match="reconcile"):
        annotated_sections(section(us=100).replace("Total time: 100", "Total time: 200"))
    with pytest.raises(ValueError, match="invalid"):
        annotated_sections(section(us=-1))


def test_shared_projection_normalization_is_the_same_on_both_backends():
    weight = b"blk.0.ffn_down_shexp.weight".hex()
    text = f"HE_OWNER 0x1 linear MUL_MAT - {weight}\n" + section()
    assert annotated_sections(text)[0]["rows"][0]["owner"] == "moe"
    hip = normalize_hip_roles(
        {
            "roles": [
                {"name": "linear:layers.*.shared_down", "ms": 2},
                {"name": "moe:layers.*.expert_gate", "ms": 3},
                {"name": "qsa_decode:layers.*.attn_q", "ms": 1},
            ],
            "unattributed_ms": 0,
            "attributed_ms": 6,
            "window_ms": 7,
        }
    )
    assert hip["owner_ms"] == {"moe": 5, "qsa": 1}
    with pytest.raises(ValueError, match="unattributed"):
        normalize_hip_roles({"roles": [], "unattributed_ms": 1})


def test_identity_rejects_cross_host_and_cross_fixture():
    capture = {
        "taxonomy": TAXONOMY,
        "fixture_sha256": "fixture",
        "quant": "q4",
        "kv_dtype": "bf16",
        "status": "captured",
        "host": {"machine_id": HOST_ID},
        "model_identity": {"fingerprint": {"value": "model"}},
    }
    check_capture_identity(capture, capture)
    with pytest.raises(ValueError, match="host"):
        check_capture_identity(capture, capture | {"host": {"machine_id": "zbook"}})
    with pytest.raises(ValueError, match="fixture"):
        check_capture_identity(capture, capture | {"fixture_sha256": "different"})


def test_baseline_commands_preserve_full_suite_and_use_original_binaries(tmp_path):
    queue = {
        "fixture": {"path": "fixture.json"},
        "model": {"root": "/model"},
        "comparator": {
            "source": "/reference",
            "vulkan_binary": "/vk/llama-server",
            "hip_binary": "/hip/llama-server",
            "server_args": ["-ctk", "bf16"],
        },
    }
    commands = baseline_commands(queue, tmp_path)
    assert [name for name, _ in commands] == ["hipengine", "halo-box-vulkan", "halo-box-hip"]
    for _, argv in commands:
        assert "--case-id" not in argv
        assert argv[argv.index("--repetitions") + 1] == "3"
        assert argv[argv.index("--warmups") + 1] == "1"
    assert "--server-arg=bf16" in commands[1][1]
