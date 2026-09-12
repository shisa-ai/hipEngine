import pytest


def test_capacity_cases_cover_all_short_categories():
    from scripts.qwen4exp_context_capacity_ab import select_cases
    cases = [dict(id=f"{cat}-p{n}",prompt_tokens=n) for cat in
             ("code","general_en","general_ja","mixed_ja_en") for n in (512,1024,4096)]
    selected = select_cases(cases,2051,262144,128)
    assert len(selected)==8
    assert {c["prompt_tokens"] for c in selected}=={512,1024}
    with pytest.raises(ValueError):
        select_cases(cases,1024,262144,128)
