"""Bounded feature cycles preserve the greedy sequence and free prompt taps."""
from types import SimpleNamespace
import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("method", ["eagle4", "dspine"])
@pytest.mark.parametrize("accepted", [True, False])
@pytest.mark.parametrize("limit", [0, 1, 7])
def test_incremental_cycles_and_blocking_wrapper_share_tokens(monkeypatch, method, accepted, limit):
    module = importlib.import_module(f"hipengine.speculative.{method}_cycle")
    calls = []
    session = SimpleNamespace(
        runner=SimpleNamespace(hidden_size=1), position=2,
        reset=lambda: calls.append("reset"), last_native_spec_target_submitted=True,
    )
    session._commit_verify_linear_state_row = lambda k, position: setattr(session, "position", position)
    native = SimpleNamespace(ctx_len=2, root=7)
    native.prefill_context = lambda *args: setattr(native, "root", 7)
    native.reset_projected_context = lambda *args: None
    native.draft_greedy_chain_cached = lambda n: ([native.root + 1 if accepted else 30, native.root + 2], None)
    native.forward = lambda *args, anchor_id: setattr(native, "root", anchor_id)
    native.refine = lambda *args: [native.root + 1 if accepted else 30, native.root + 2]
    native.append_committed = lambda aux, ids: setattr(native, "root", ids[-1])
    native.append_projected_rows = lambda *args: None
    nd = SimpleNamespace(aux_layers=[0], config={"dflash_config": {"target_layer_ids": [0]}, "block_size": 3},
                         mask_token_id=0, project_target_hidden=lambda x: x)
    capture_name = "_capture_prompt_aux" if method == "eagle4" else "_capture_prompt_taps"
    monkeypatch.setattr(module, capture_name, lambda *args: (np.zeros((2, 1), np.float32), 7))
    def verify(session, tokens, **kwargs):
        calls.append("verify")
        return SimpleNamespace(token_ids=[tokens[0] + i + 1 for i in range(3)],
                               layer_output_hidden_device=None,
                               layer_output_hidden={0: np.zeros((3, 1), np.float32)})
    monkeypatch.setattr(module, "verify_qwen35_gguf_native_b2_target", verify)
    args = [session, native, nd]
    kwargs = dict(prompt_ids=(1, 2), max_new_tokens=limit, runtime=object(), ignore_eos=True)
    if method == "eagle4":
        kwargs["n_draft"] = 2
    else:
        args += [np.zeros((32, 1), np.float32), object()]
        kwargs["candidate_budget"] = 2
    steps = getattr(module, f"iter_{method}_cycle")(*args, **kwargs)
    assert next(steps) is None
    assert calls == ["reset"]
    result = None
    for update in steps:
        previous = 0 if result is None else result["draft_forwards"]
        assert update["draft_forwards"] - previous == (1 if limit else 0)
        result = update
    assert result["output_ids"] == list(range(7, 7 + limit))
    assert result["finished"]
    blocking = getattr(module, f"run_{method}_cycle")(*args, **kwargs)
    for key in ("output_ids", "draft_forwards", "graph_verify_cycles", "mean_acceptance", "finish_reason"):
        assert blocking[key] == result[key]


@pytest.mark.parametrize("method", ["eagle4", "dspine"])
@pytest.mark.parametrize("failure", ["allocate", "prefill"])
def test_prompt_tap_failure_frees_every_allocated_buffer(monkeypatch, method, failure):
    module = importlib.import_module(f"hipengine.speculative.{method}_cycle")
    allocated, released = [], []
    def malloc(size):
        if failure == "allocate" and allocated:
            raise RuntimeError("allocate")
        ptr = 100 + len(allocated)
        allocated.append(ptr)
        return ptr
    runtime = SimpleNamespace(malloc=malloc)
    monkeypatch.setattr(module, "free", lambda buffer, **kwargs: released.append(buffer.ptr))
    def prefill(*args, **kwargs):
        raise RuntimeError("prefill")
    session = SimpleNamespace(prefill=prefill)
    capture = getattr(module, "_capture_prompt_aux" if method == "eagle4" else "_capture_prompt_taps")
    with pytest.raises(RuntimeError, match=failure):
        capture(session, (1, 2), (0, 1), 1, runtime)
    assert released == allocated
