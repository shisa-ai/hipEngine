import numpy as np
import pytest
import json


def test_replay_identity_rejects_other_host_model_or_fixture():
    from scripts.qwen4exp_routing_capture import validate_replay_identity
    from scripts.qwen4exp_framework_family_refresh import HOST_ID, MODEL_FINGERPRINT
    packet = dict(
        host={"machine_id": HOST_ID}, fixture_sha256="fixture",
        model_identity={"fingerprint": {"value": MODEL_FINGERPRINT}})
    validate_replay_identity(packet, fixture_sha256="fixture")
    for key, value in (
        ("host", {"machine_id": "zbook"}),
        ("fixture_sha256", "other"),
        ("model_identity", {"fingerprint": {"value": "other"}}),
    ):
        with pytest.raises(ValueError):
            validate_replay_identity({**packet, key: value}, fixture_sha256="fixture")


def test_routing_count_validation():
    from scripts.qwen4exp_routing_capture import routing_record
    result = routing_record(np.array([0,0,2,9],np.int64),9,3)
    assert result["counts"] == [0,2,7]
    assert result["weight_passes_by_row_batch"] == {"8":2,"16":2,"32":2}
    for starts in ([1,2,3,9],[0,4,3,9],[0,2,3,8],[0,2,9]):
        with pytest.raises(ValueError):
            routing_record(np.array(starts,np.int64),9,3)


def test_summary_weights_rows_not_expert_medians():
    from scripts.qwen4exp_routing_capture import summarize_routing
    packet = {"status": "captured_exact", "records": [
        {"case_id": "code-p512", "family": "q4", "counts": [0, 1, 9]},
        {"case_id": "code-p512", "family": "q4", "counts": [0, 2, 8]},
    ]}
    row = summarize_routing(packet)[0]
    assert row["boundaries"] == 2
    assert row["active_expert_instances"] == 4
    assert row["compact_rows"] == 20
    assert row["rows_in_experts_gt8_share"] == 9 / 20
    assert row["weight_passes_by_row_batch"] == {"8": 5, "16": 4, "32": 4}


def test_summary_cli_preserves_identity_without_raw_counts(tmp_path, monkeypatch):
    from scripts.qwen4exp_routing_capture import main
    from scripts.qwen4exp_framework_family_refresh import HOST_ID, MODEL_FINGERPRINT
    from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture
    raw, out = tmp_path / "raw.json", tmp_path / "summary.json"
    packet = dict(status="captured_exact", cases=[],
        host={"machine_id": HOST_ID},
        model_identity={"fingerprint": {"value": MODEL_FINGERPRINT}},
        fixture_sha256=load_fixture(DEFAULT_FIXTURE)[1],
        records=[dict(case_id="code-p512", family="q4", counts=[1, 9])])
    raw.write_text(json.dumps(packet))
    monkeypatch.setattr("sys.argv", ["capture", "--summarize", str(raw), "--output", str(out)])
    main()
    result = json.loads(out.read_text())
    assert not result["performance_claim"]
    assert "records" not in result["captures"][0]
    assert result["captures"][0]["host"] == packet["host"]
    assert result["captures"][0]["summary"][0]["compact_rows"] == 10
    assert len(result["captures"][0]["raw_sha256"]) == 64


def test_replay_requires_matching_case_layer_shape():
    from scripts.qwen4exp_routing_capture import select_routing
    record = dict(case_id="code-p512",family="q4",tensor="blk.0.ffn_gate_exps.weight",
                  chunk_index=0,compact_rows=20,num_experts=4,in_features=2560,
                  out_features=640,counts=[0,3,7,10])
    packet = dict(status="captured_exact",records=[record])
    counts = select_routing(packet,case_id="code-p512",layer=0,chunk=0,tokens=2)
    np.testing.assert_array_equal(counts,[0,3,7,10])
    with pytest.raises(ValueError):
        select_routing(packet,case_id="code-p512",layer=1,chunk=0,tokens=2)
    with pytest.raises(ValueError):
        select_routing(packet,case_id="code-p512",layer=0,chunk=0,tokens=512)
    with pytest.raises(ValueError):
        select_routing({**packet, "status": "running"},
                       case_id="code-p512",layer=0,chunk=0,tokens=2)
    with pytest.raises(ValueError):
        select_routing({**packet, "records": [record, record]},
                       case_id="code-p512",layer=0,chunk=0,tokens=2)
    with pytest.raises(ValueError):
        select_routing({**packet, "records": [{**record, "counts": [0.5,3,7,10]}]},
                       case_id="code-p512",layer=0,chunk=0,tokens=2)
