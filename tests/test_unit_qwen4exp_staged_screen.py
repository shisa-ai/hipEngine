import pytest
from scripts import qwen4exp_halo_box_campaign_ab as ab
from tests.test_unit_qwen4exp_halo_box_campaign_ab import _sample


def samples(losses=0):
    return [_sample(mode=mode,case_id=f"case{i}",category="code",
        prompt_tokens=512,repetition=0,
        prefill_ms=110 if i<losses and mode=="after" else 100,
        decode_ms=110 if i<losses and mode=="after" else 100)
        for i in range(12) for mode in ("before","after")]


def test_schedule_keeps_all_samples_in_order():
    for case in (0,1):
        first=ab.staged_slots(case,0)
        rest=ab.staged_slots(case,1)
        assert first+rest==list(enumerate(ab.arm_sequence(case)))
        assert len(first)==2 and len(rest)==4


def test_only_multiple_clear_losses_stop():
    assert not ab.clear_screen_losses(samples(0),.05)
    assert not ab.clear_screen_losses(samples(1),.05)
    assert ab.clear_screen_losses(samples(2),.05)==["case0","case1"]


def test_screen_fails_closed_on_missing_or_wrong_outputs():
    with pytest.raises(ValueError):
        ab.clear_screen_losses(samples()[:-1],.05)
    bad=samples()
    bad[0]["output_token_ids_sha256"]="wrong"
    with pytest.raises(ValueError):
        ab.clear_screen_losses(bad,.05)


def test_extension_preserves_first_pair_and_balanced_repetitions():
    rows=[]
    for stage in (0,1):
        for case in range(12):
            counts={"before":stage,"after":stage}
            for slot,mode in ab.staged_slots(case,stage):
                row=_sample(mode=mode,case_id=f"case{case}",category="code",
                    prompt_tokens=512,repetition=counts[mode],prefill_ms=100,decode_ms=100)
                counts[mode]+=1
                row["sequence_slot"]=slot
                rows.append(row)
        if stage==0:
            first=list(rows)
            assert not ab.clear_screen_losses(first,.05)
    assert rows[:24]==first
    assert len(rows)==72
    assert ab.summarize_campaign_ab(rows,repetitions_per_mode=3)["correctness"]["within_mode_deterministic"]


def test_prefill_only_loss_does_not_stop_request_win():
    rows=samples(2)
    for row in rows:
        if row["mode"]=="after":
            row["decode_ms"]=50
    assert not ab.clear_screen_losses(rows,.05)


def test_staged_rejects_subset_before_model_load():
    with pytest.raises(SystemExit,match="staged-screen"):
        ab.main(["--model-root","/unused","--output","/unused","--staged-screen",
                 "--case-id","code-p512"])
