"""Live DMS + INT8 MTP parity gate against the same DMS + INT8 AR policy.

Both arms are the same retention topology on the same artifact: one
``dms_metadata_path`` and one ``dms_backend_factory``, so the compact INT8 DMS
policy -- its codec, window, target compression ratio, and eviction decisions --
is identical by construction. The gate asserts that speculative cycles on that
policy produce exactly the tokens the autoregressive arm produces, on prompts
from every category in the suite, and that the DMS policy actually evicted
rather than being bypassed.

Scope, stated exactly: this is a live resident-session gate. DMS has no server
or engine surface yet (no ``dms_metadata_path`` reaches the engine or the HTTP
layer), so "public" is not available for a DMS row and is not claimed here.

Two things this gate pins deliberately:

* Drafts are the autoregressive oracle, so every candidate is a token AR would
  itself produce. That isolates the retention/transaction axis under test
  instead of mixing in draft quality; the production draft source (the NextN
  provider) is orthogonal to DMS. One case corrupts a candidate so partial
  acceptance is exercised too.
* The context is longer than the DMS window, so the policy evicts. A gate on a
  context short enough to skip eviction would pass without exercising DMS at
  all.

Skips unless ROCm, the dense GGUF model, and DMS sidecar metadata are all
present.
"""

from __future__ import annotations

import ctypes
import json
import os
from functools import lru_cache
from pathlib import Path

import pytest

_MODEL = Path(
    os.environ.get("HIPENGINE_DMS_MTP_MODEL", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
)
_METADATA = Path(
    os.environ.get(
        "HIPENGINE_DMS_MTP_METADATA",
        str(Path.home() / "dms-artifacts/qwen38-external-v1/sidecar/dms_metadata.json"),
    )
)
_PROMPTS = Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl")

# The DMS window is 256 tokens, so a context this long forces real eviction.
_CONTEXT_TOKENS = 768
_CYCLE_WIDTH = 3
_CYCLES = 3
_DECODE_STEPS = _CYCLES * (_CYCLE_WIDTH + 1)
_EOS_TOKEN_ID = 248046
# The gfx1151 peer builds the binaries this device can execute, and aliases the
# whole gfx1100 key space at import time.
_BACKEND = "hip_gfx1151"

# Complete kernel table captured on the first case of the file; see
# restore_kernel_registrations.
_REFERENCE_KERNELS: dict[object, object] | None = None


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_hip_available() and _MODEL.is_file() and _METADATA.is_file()),
    reason="requires ROCm, the dense GGUF model, and DMS sidecar metadata",
)


def _category_cases() -> list[tuple[str, tuple[int, ...]]]:
    """One prompt per category, first in file order, tokenized once."""

    return list(_category_cases_cached())


@lru_cache(maxsize=1)
def _category_cases_cached() -> tuple[tuple[str, tuple[int, ...]], ...]:
    from hipengine.loading.gguf import GGUFReader
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    rows = [
        json.loads(line)
        for line in _PROMPTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(GGUFReader(_MODEL).info)
    seen: dict[str, tuple[int, ...]] = {}
    for row in rows:
        category = str(row["category"])
        if category in seen:
            continue
        tokens = tuple(
            int(token) for token in tokenizer.encode(row["messages"][0]["content"])
        )
        # Repeat deterministically to clear the DMS window. Eviction is a
        # function of positions, not of semantics, and both arms see this
        # identical context.
        repeated: list[int] = []
        while len(repeated) < _CONTEXT_TOKENS:
            repeated.extend(tokens)
        seen[category] = tuple(repeated[:_CONTEXT_TOKENS])
    return tuple(sorted(seen.items()))


@pytest.fixture(autouse=True)
def restore_kernel_registrations(shared_runner):
    """Keep the backend's full key space alive across the file's cases.

    The shared conftest restores the kernel table to its collection-time
    baseline after every test. This backend's package is imported lazily on
    first use, which happens after that baseline is taken, so the baseline does
    not contain its aliases -- including the ``dflash_accept_chain`` kernel the
    verifier resolves in its constructor. Re-registering those aliases cannot
    help: they are derived from the gfx1100 key space, whose modules are already
    imported, so re-importing them registers nothing. The first case of the file
    therefore snapshots the complete table and later cases restore it.
    """

    del shared_runner
    global _REFERENCE_KERNELS

    from hipengine.kernels import registry
    from hipengine.speculative.native_cycle_graph import register_native_spec_gguf_graphs

    if _REFERENCE_KERNELS is None:
        # The shared session has just imported this backend for the first time,
        # so its keys are present now and will not be again.
        _REFERENCE_KERNELS = dict(registry._KERNELS)
    else:
        registry.restore_registry_for_tests(_REFERENCE_KERNELS)
    register_native_spec_gguf_graphs()


@pytest.fixture(scope="module")
def shared_runner():
    from hipengine.runtime import qwen35_gguf_runner as runner_module
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner_module, "_GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS", 0)
        yield Qwen35GGUFFullStackRunner(_MODEL, backend=_BACKEND)


def _dms_session(shared_runner):
    """A DMS row on the one retention policy both arms must share."""

    from hipengine.kvcache.dms import create_dms_int8_evaluation_backend
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    return Qwen35GGUFResidentSession(
        _MODEL,
        backend=_BACKEND,
        shared_runner=shared_runner,
        max_sequence_length=_CONTEXT_TOKENS + _DECODE_STEPS + 8,
        use_wmma_prefill=True,
        use_gemv_decode=True,
        dms_metadata_path=_METADATA,
        dms_backend_factory=create_dms_int8_evaluation_backend,
        dms_max_new_tokens=_DECODE_STEPS + 8,
    )


def _finish_reason(tokens: tuple[int, ...]) -> str:
    return "stop" if _EOS_TOKEN_ID in tokens else "length"


def _prefill(session, prompt_tokens: tuple[int, ...]) -> int:
    seed = session.prefill(
        prompt_tokens, use_bulk=True, bulk_attention_mode="bulk", return_logits=False
    )
    return int(seed.token_id)


def _dms_evidence(session) -> dict:
    snapshot = session._dms_backend.observability_snapshot()
    return {
        "evicted_tokens": int(snapshot["operations"]["evicted_tokens"]),
        "target_compression_ratio": int(snapshot["capacity"]["target_compression_ratio"]),
        "actual_compression_ratio": float(snapshot["capacity"]["actual_compression_ratio"]),
        "logical_token_rows": int(snapshot["capacity"]["logical_token_rows"]),
        "max_live_count": int(snapshot["capacity"]["max_live_count"]),
        "decode_appends": int(snapshot["operations"]["decode_appends"]),
    }


def _ar_arm(
    shared_runner, prompt_tokens: tuple[int, ...], *, steps: int = _DECODE_STEPS
) -> dict:
    """Greedy autoregressive decode on the DMS row."""

    with _dms_session(shared_runner) as session:
        root = _prefill(session, prompt_tokens)
        generated: list[int] = []
        token = root
        for _ in range(steps):
            token = int(session.step(token, return_logits=False).token_id)
            generated.append(token)
        return {
            "root": root,
            "tokens": tuple(generated),
            "finish_reason": _finish_reason(tuple(generated)),
            "dms": _dms_evidence(session),
        }


def _mtp_arm(
    shared_runner,
    prompt_tokens: tuple[int, ...],
    oracle: tuple[int, ...],
    *,
    corrupt_candidate: int | None = None,
) -> dict:
    """Speculative cycles on the DMS row, drafted from the AR oracle."""

    from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier
    from hipengine.speculative import DraftBatch, TargetCommitPlan, TargetVerifyBatch

    with _dms_session(shared_runner) as session:
        root = _prefill(session, prompt_tokens)
        generated: list[int] = []
        accepted_counts: list[int] = []
        rejections: list[int] = []
        token = root
        for cycle in range(_CYCLES):
            width = _CYCLE_WIDTH
            candidates = list(oracle[len(generated) : len(generated) + width])
            if corrupt_candidate is not None and cycle == corrupt_candidate:
                candidates[1] = (candidates[1] + 1) % shared_runner.vocab_size
            position = int(session.position)
            draft = DraftBatch(
                request_ids=(0,),
                candidate_tokens=tuple(candidates),
                parent_positions=tuple(position + index for index in range(width)),
                draft_depths=tuple(range(1, width + 1)),
                row_to_request=(0,) * width,
                mode="verify_chain",
            )
            batch = TargetVerifyBatch.from_draft(
                draft, root_tokens=(token,), root_positions=(position,)
            )
            with Qwen35GGUFTransactionalVerifier(
                session,
                max_candidate_budget=width,
                quant="gguf_q4_k_m",
                target_verify_mode="serial_exact",
            ) as verifier:
                prepared = verifier.prepare(
                    batch,
                    transaction_id=cycle + 1,
                    graph_bucket=verifier.graph_bucket(f"dms-c{cycle}", batch),
                    remaining_decode=(_DECODE_STEPS,),
                    allow_graph=False,
                )
                assert prepared.gpu_accept_match_cpu
                accepted = int(prepared.summary.accepted_counts[0])
                accepted_counts.append(accepted)
                rejections.append(width - accepted)
                plan = TargetCommitPlan(
                    transaction_id=cycle + 1,
                    request_ids=batch.request_ids,
                    accepted_counts=prepared.summary.accepted_counts,
                    commit_rows=prepared.summary.commit_rows,
                    commit_tokens=prepared.summary.commit_tokens,
                    commit_positions=prepared.summary.commit_positions,
                    next_tokens=prepared.summary.next_tokens,
                    candidate_counts=batch.candidate_counts,
                    draft_depth=batch.draft_depth,
                    tree_shape=batch.tree_shape,
                    mode=batch.mode,
                )
                verifier.commit(prepared, plan)
                verifier.finish(prepared)
            generated.extend(int(t) for t in prepared.summary.accepted_tokens[0])
            token = int(prepared.summary.next_tokens[0])
            generated.append(token)
        return {
            "root": root,
            "tokens": tuple(generated),
            "finish_reason": _finish_reason(tuple(generated)),
            "accepted_counts": tuple(accepted_counts),
            "rejections": tuple(rejections),
            "dms": _dms_evidence(session),
        }


def _assert_same_policy(ar: dict, mtp: dict) -> None:
    """The retention policy must be identical on both arms, and it must evict.

    Compaction happens at prefill, so the eviction evidence is the live count
    falling below the logical token count; ``evicted_tokens`` only counts
    decode-time evictions and is reported for context rather than asserted.
    The live count is a function of the decode position, so the two arms must
    be compared at the same token count: a speculative arm that commits a
    shorter prefix reaches a lower position and legitimately holds fewer live
    rows. At equal positions the policy's outcome -- the live count after
    compaction -- must be identical, which is what makes the verify batch's
    extra rows and any rollback invisible to the policy.
    """

    left, right = ar["dms"], mtp["dms"]
    assert left["target_compression_ratio"] == right["target_compression_ratio"] == 2
    assert left["max_live_count"] == right["max_live_count"], (left, right)
    assert left["actual_compression_ratio"] == right["actual_compression_ratio"]
    for label, arm in (("ar", ar), ("mtp", mtp)):
        dms = arm["dms"]
        # Non-vacuity: each arm fed every token it generated back through the
        # DMS append path. A shorter arm appends proportionally fewer rows.
        assert dms["decode_appends"] >= len(arm["tokens"]), label
        assert dms["logical_token_rows"] > dms["max_live_count"], (
            f"{label} arm kept every token: the DMS policy was not exercised"
        )


@pytest.mark.parametrize("case", _category_cases(), ids=lambda case: case[0])
def test_dms_int8_mtp_matches_dms_int8_ar(shared_runner, case) -> None:
    """Speculative cycles reproduce the AR ids and finish reason exactly."""

    category, prompt_tokens = case
    ar = _ar_arm(shared_runner, prompt_tokens)
    mtp = _mtp_arm(shared_runner, prompt_tokens, ar["tokens"])

    assert mtp["root"] == ar["root"], category
    assert mtp["tokens"] == ar["tokens"], (
        f"{category}: MTP diverged from AR\n  mtp={mtp['tokens']}\n  ar ={ar['tokens']}"
    )
    assert mtp["finish_reason"] == ar["finish_reason"]
    # Speculative cycles really ran and really committed.
    assert sum(mtp["accepted_counts"]) > 0, category
    assert len(mtp["accepted_counts"]) == _CYCLES
    _assert_same_policy(ar, mtp)


def test_dms_int8_mtp_partial_acceptance_keeps_ar_parity(shared_runner) -> None:
    """A rejected candidate commits only the accepted prefix, and stays exact."""

    _, prompt_tokens = _category_cases()[0]
    ar = _ar_arm(shared_runner, prompt_tokens)
    mtp = _mtp_arm(shared_runner, prompt_tokens, ar["tokens"], corrupt_candidate=0)

    assert mtp["rejections"][0] >= 1, (
        "the corrupted candidate was accepted, so partial acceptance was not exercised"
    )
    assert mtp["accepted_counts"][0] == 1
    # A rejected draft must not corrupt the output: the target's own token
    # replaces it, so the committed sequence stays an exact prefix of AR, one
    # token shallower than a fully accepted cycle.
    committed = mtp["tokens"]
    assert committed == ar["tokens"][: len(committed)], (
        f"partial acceptance diverged from AR\n  mtp={committed}\n  ar ={ar['tokens']}"
    )
    assert len(committed) < len(ar["tokens"])
    # A partial cycle commits two fewer tokens than a fully accepted one, so
    # the policy has to be compared against an AR run that stops at the same
    # position rather than against the longer oracle run above.
    matched = _ar_arm(shared_runner, prompt_tokens, steps=len(committed))
    assert matched["tokens"] == committed
    assert matched["finish_reason"] == mtp["finish_reason"]
    _assert_same_policy(matched, mtp)
