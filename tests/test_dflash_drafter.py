from __future__ import annotations

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.loading.dflash import dflash_draft_config_from_hf
from hipengine.speculative import DFlashRootQueryPlan, DFlashRootQueryRequest, TargetVerifyBatch, draft_batch_from_topk


def test_dflash_root_query_plan_prepares_root_and_mask_rows() -> None:
    config = dflash_draft_config_from_hf(_config())
    req = DFlashRootQueryRequest(
        request_id=3,
        root_token=17,
        root_position=12,
        context_length=5,
        target_hidden_rows=_tensor(0x1000, (5, config.target_hidden_concat_size)),
    )

    plan = DFlashRootQueryPlan.from_requests([req], config=config)

    assert plan.request_ids == (3,)
    assert plan.root_tokens == (17,)
    assert plan.root_positions == (12,)
    assert plan.context_lengths == (5,)
    assert plan.noise_token_ids == ((17, 99, 99, 99),)
    assert plan.position_ids == ((5, 6, 7, 8),)
    assert plan.target_hidden_concat_size == 32


def test_dflash_root_query_plan_validates_hidden_tap_shape() -> None:
    config = dflash_draft_config_from_hf(_config())
    req = DFlashRootQueryRequest(
        request_id=3,
        root_token=17,
        root_position=12,
        context_length=5,
        target_hidden_rows=_tensor(0x1000, (5, config.target_hidden_concat_size - 1)),
    )

    with pytest.raises(ValueError, match="target hidden concat size"):
        DFlashRootQueryPlan.from_requests([req], config=config)


def test_dflash_draft_batch_from_topk_remains_candidate_only() -> None:
    config = dflash_draft_config_from_hf(_config())
    requests = [
        DFlashRootQueryRequest(1, 10, 7, 2, _tensor(0x1000, (2, config.target_hidden_concat_size))),
        DFlashRootQueryRequest(2, 20, 11, 3, _tensor(0x2000, (3, config.target_hidden_concat_size))),
    ]
    plan = DFlashRootQueryPlan.from_requests(requests, config=config)

    draft = draft_batch_from_topk(
        plan,
        topk_token_ids=(
            ((101, 201), (102, 202), (103, 203)),
            ((111, 211), (112, 212), (113, 213)),
        ),
        candidate_budget=2,
        topk_rank=0,
    )

    assert draft.request_ids == (1, 2)
    assert draft.candidate_tokens == (101, 102, 111, 112)
    assert draft.parent_positions == (7, 8, 11, 12)
    assert draft.draft_depths == (1, 2, 1, 2)
    assert draft.row_to_request == (1, 1, 2, 2)
    assert draft.active_mask == (True, True, True, True)
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(10, 20), root_positions=(7, 11))
    assert target.root_rows == (0, 1)
    assert target.tokens == (10, 20, 101, 102, 111, 112)


def test_dflash_draft_batch_from_topk_can_select_nonzero_rank() -> None:
    config = dflash_draft_config_from_hf(_config())
    plan = DFlashRootQueryPlan.from_requests(
        [DFlashRootQueryRequest(1, 10, 7, 2, _tensor(0x1000, (2, config.target_hidden_concat_size)))],
        config=config,
    )

    draft = draft_batch_from_topk(plan, topk_token_ids=(((101, 201), (102, 202)),), candidate_budget=2, topk_rank=1)

    assert draft.candidate_tokens == (201, 202)
    with pytest.raises(ValueError, match="topk_rank"):
        draft_batch_from_topk(plan, topk_token_ids=(((101,),),), candidate_budget=2, topk_rank=1)


def _tensor(ptr: int, shape: tuple[int, ...]) -> Tensor:
    return Tensor.from_handle(ptr, shape, "bf16", Device("hip", 0))


def _config() -> dict:
    return {
        "architectures": ["DFlashDraftModel"],
        "block_size": 4,
        "dflash_config": {"mask_token_id": 99, "target_layer_ids": [1, 3]},
        "dtype": "bfloat16",
        "head_dim": 4,
        "hidden_size": 16,
        "intermediate_size": 32,
        "layer_types": ["full_attention", "full_attention"],
        "model_type": "qwen3",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_target_layers": 4,
        "vocab_size": 100,
    }
