import pytest
from scripts import qwen4exp_halo_box_campaign_ab as ab
from tests.test_unit_qwen4exp_halo_box_campaign_ab import _sample


def test_one_pair_sequences():
    assert ab.measurement_sequence(0,1)==("before","after")
    assert ab.measurement_sequence(1,1)==("after","before")
    assert ab.measurement_sequence(0,3)==ab.arm_sequence(0)
    with pytest.raises(ValueError):
        ab.measurement_sequence(0,2)


def test_one_pair_does_not_claim_repeatability():
    samples=[_sample(mode=m,case_id="code-p512",category="code",
                     prompt_tokens=512,repetition=0,prefill_ms=10,decode_ms=10)
             for m in ("before","after")]
    result=ab.summarize_campaign_ab(samples,repetitions_per_mode=1)
    assert result["correctness"]["within_mode_deterministic"] is None
    assert result["by_case"]["code-p512"]["within_mode_deterministic"] is None
    for mode in ("before","after"):
        case=result[mode]["cases"]["code-p512"]
        assert case["prefill_tok_s"]["coefficient_of_variation"] is None


def test_publication_requires_three_before_loading_model():
    with pytest.raises(SystemExit,match="publication protocol"):
        ab.main(["--model-root","/unused","--repetitions-per-mode","1","--output","/unused"])


def test_retention_rejects_screen_packet(tmp_path,monkeypatch):
    import json
    from scripts import qwen4exp_campaign_ab_retention as retention
    raw=tmp_path/"screen.json"
    raw.write_text(json.dumps(dict(status="completed",diagnostic_subset=False,screen_only=True)))
    monkeypatch.setattr("sys.argv",["retention","--input",str(raw),"--output",str(tmp_path/"out.json")])
    with pytest.raises(ValueError,match="full-suite"):
        retention.main()
