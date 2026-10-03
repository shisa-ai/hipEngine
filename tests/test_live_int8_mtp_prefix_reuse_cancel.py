"""Prefix reuse after a cancelled request on the INT8 MTP chain.

Readiness row: "Prefix reuse after a cancelled request is tested with MTP enabled
and disabled." The lifecycle half of that row is covered by
``tests/test_live_dms_int8_mtp_lifecycle.py``, which aborts a cycle on a resident
DMS row and compares the DMS store. What this file adds is the axis that lives
one level up. Prefix reuse is an engine mechanism -- ``HIPENGINE_PREFIX_CACHE``
is read by the engine loop and the server, while ``dms_metadata_path`` is read by
the resident session -- so the engine surface is where a request is cancelled
*after* its prefix was reused, and that is what these arms drive.

Each arm seeds a boundary with one full request, then issues the same prompt
again, which the engine answers from the retained prefix, and cancels that
request mid-generation. The arms assert that the reuse really happened (a
measured ``reused_tokens``, not an inferred one), that the engine reported the
cancellation in the request's own finish details, that the pool and allocator
state return to the seeded baseline and stay there across a second cancelled
request, and that the prefix the cancellation interrupted still produces the
seed's exact ids.

Both cache configurations run, and both MTP configurations run inside each of
them (``generate_speculative_mtp_detailed`` and the ordinary
``generate_detailed``). Cache off is the control: the same cancellation with no
reused prefix, so a clean trace there cannot be mistaken for the reuse path
being exercised.

Two surface details shape the arms. The request is blocking with a timer-driven
cancel rather than a streamed one, because a cancelled speculative *stream* on
this path ends without publishing its terminal chunk, so the engine's own
``FinishDetails(reason="cancelled")`` is not observable there; the blocking
response carries it. And the prompt is text rather than pre-tokenized ids,
because the engine renders and tokenizes it -- the DMS parity gate's cases are
bare content tokens repeated to a fixed length, and those stop on the first
token through this surface.
"""

from __future__ import annotations

import ctypes
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping

import pytest

from hipengine import LLM, SamplingParams
from hipengine.generation.deadline import GenerationCancellationToken

MODEL = Path(
    os.environ.get("HIPENGINE_INT8_MTP_MODEL", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
)
PROMPTS = Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl")
BACKEND = "hip_gfx1151"
CACHE_MODES = ("radix", "off")

# The engine retains a boundary only for a prompt longer than 256 tokens, and
# reuse restores state at a 256-token-aligned boundary, so the prompt has to
# clear 512 tokens for a two-page reuse.
_MIN_PROMPT_CHARS = 3200
_REUSED_TOKENS = 256
_SEED_TOKENS = 8
_CANCEL_TOKENS = 48
_CANCELLATIONS = 2
# Long enough that admission, the prefix restore and the first tokens are all
# done, and short enough to land inside a request that generates far more.
_CANCEL_DELAY_SECONDS = 0.5

_REFERENCE_KERNELS: dict[Any, Any] | None = None


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.exists()),
    reason="requires ROCm + the dense GGUF model",
)


@pytest.fixture(autouse=True)
def restore_kernel_registrations() -> None:
    """Keep the backend's key space alive across this file's cases.

    The shared conftest restores the kernel table to its collection-time
    baseline after every test, and this backend's package is imported lazily on
    first use -- after that baseline was taken -- so from the second case on the
    runtime execution-profile plan cannot resolve its own keys. Importing once
    here and restoring the resulting table before every case fixes the table at
    the point where it is complete.
    """

    global _REFERENCE_KERNELS

    from hipengine.kernels import registry
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.speculative.native_cycle_graph import register_native_spec_gguf_graphs

    if _REFERENCE_KERNELS is None:
        load_backend_kernel_package(BACKEND)
        register_native_spec_gguf_graphs()
        _REFERENCE_KERNELS = dict(registry._KERNELS)
    else:
        registry.restore_registry_for_tests(_REFERENCE_KERNELS)
        register_native_spec_gguf_graphs()


def _prompt_text() -> str:
    """A real prompt, repeated deterministically to clear the reuse boundary."""

    rows = [
        json.loads(line)
        for line in PROMPTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    content = str(rows[0]["messages"][0]["content"])
    assert content.strip(), "the prompt set's first row has no content"
    text = content
    while len(text) < _MIN_PROMPT_CHARS:
        text = f"{text}\n\n{content}"
    return text


def _observability(llm: LLM) -> dict[str, Any]:
    """The resident runner's snapshot, reached from the text generator.

    Only the runner publishes the snapshot, and it sits behind the generator's
    own forwarding links, so this walks the same ``_driver``/``_runner``/
    ``_inner`` chain the benchmark's resident-observability helper walks.
    """

    pending: list[Any] = [llm._get_text_generator()]
    seen: set[int] = set()
    while pending:
        owner = pending.pop(0)
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        snapshot = getattr(owner, "observability_snapshot", None)
        if callable(snapshot):
            payload = snapshot()
            if isinstance(payload, dict):
                return payload
        pending.extend(
            getattr(owner, name, None) for name in ("_driver", "_runner", "_inner")
        )
    raise AssertionError(
        "no observability snapshot is reachable from the text generator, so the "
        "pool and session ownership this row names cannot be read"
    )


def _trace(llm: LLM) -> dict[str, int]:
    """The ownership the row names, read from the live engine."""

    from hipengine.core.memory import memory_stats

    snapshot = _observability(llm)
    pool = snapshot.get("kv_pool") or {}
    runner = snapshot.get("model_runner") or {}
    return {
        "active_allocations": int(memory_stats()["active_allocations"]),
        "current_pages": int(pool.get("current_pages", -1)),
        "refcounted_pages": int(pool.get("refcounted_pages", -1)),
        "pinned_pages": int(pool.get("pinned_pages", -1)),
        "active_requests": int(runner.get("active_requests", -1)),
        "available_sessions": int(runner.get("available_sessions", -1)),
        "packed_workspace_owner_sessions": int(
            runner.get("packed_workspace_owner_sessions", -1)
        ),
    }


def _prefix_cache(output: Any) -> dict[str, Any]:
    """The prefix-cache block the response published."""

    telemetry = getattr(output, "telemetry", None)
    diagnostics = getattr(telemetry, "diagnostics", None)
    assert isinstance(diagnostics, Mapping), (
        "the response published no diagnostics, so the reuse this row is about "
        "could not be confirmed"
    )
    block = diagnostics.get("prefix_cache")
    assert isinstance(block, Mapping), (
        f"the response published no prefix-cache block: {sorted(diagnostics)}"
    )
    return dict(block)


def _finish_reports_cancellation(finish: Any) -> bool:
    """Whether a finish detail is the engine's cancellation signal."""

    if finish is None:
        return False
    if isinstance(finish, Mapping):
        return bool(finish.get("cancelled")) or str(finish.get("reason")) == "cancelled"
    return bool(getattr(finish, "cancelled", False)) or str(
        getattr(finish, "reason", "")
    ) == "cancelled"


def _generate(
    llm: LLM, prompt: str, params: SamplingParams, *, mtp: bool
) -> Any:
    if mtp:
        return llm.generate_speculative_mtp_detailed([prompt], params)[0]
    return llm.generate_detailed([prompt], params)[0]


def _seed(llm: LLM, prompt: str, *, mtp: bool) -> tuple[int, ...]:
    params = SamplingParams(max_tokens=_SEED_TOKENS, temperature=0.0)
    output = _generate(llm, prompt, params, mtp=mtp)
    ids = tuple(int(token) for token in (output.generated_token_ids or ()))
    assert len(ids) == _SEED_TOKENS, (
        f"the seeding request produced {len(ids)} tokens, not {_SEED_TOKENS}"
    )
    return ids


def _cancel_reused_request(
    llm: LLM,
    prompt: str,
    *,
    cache_mode: str,
    mtp: bool,
) -> dict[str, Any]:
    """Reuse the prefix, cancel the request mid-generation, return what it saw."""

    token = GenerationCancellationToken()
    params = SamplingParams(
        max_tokens=_CANCEL_TOKENS,
        temperature=0.0,
        cancellation_token=token,
    )
    timer = threading.Timer(_CANCEL_DELAY_SECONDS, token.cancel)
    timer.start()
    try:
        output = _generate(llm, prompt, params, mtp=mtp)
    finally:
        timer.cancel()

    emitted = len(tuple(output.generated_token_ids or ()))
    finish = getattr(output, "finish_details", None)
    assert _finish_reports_cancellation(finish), (
        "the engine did not report a cancelled finish for a request whose "
        f"cancellation was requested mid-generation: finish={finish!r}"
    )
    assert emitted < _CANCEL_TOKENS, (
        f"the request published {emitted} tokens, reaching the "
        f"{_CANCEL_TOKENS}-token limit, so the cancellation never landed"
    )

    prefix = _prefix_cache(output)
    assert prefix.get("mode") == cache_mode, prefix
    if cache_mode == "radix":
        assert prefix.get("hit") is True, (
            f"the second request did not reuse the seeded prefix: {prefix}"
        )
        reused = int(prefix.get("reused_tokens", 0))
        assert reused >= _REUSED_TOKENS, (
            f"the reuse restored {reused} tokens, short of the {_REUSED_TOKENS} "
            f"a page-level restore covers: {prefix}"
        )
    else:
        assert prefix.get("hit") is not True, (
            f"the cache-off control reported a reuse: {prefix}"
        )
    return {"emitted": emitted, "finish": finish, "prefix": prefix}


def _arm(llm: LLM, prompt: str, *, cache_mode: str, mtp: bool) -> None:
    label = f"cache={cache_mode} mtp={mtp}"
    seeded_ids = _seed(llm, prompt, mtp=mtp)
    seeded = _trace(llm)

    for attempt in range(_CANCELLATIONS):
        observed = _cancel_reused_request(llm, prompt, cache_mode=cache_mode, mtp=mtp)
        after = _trace(llm)
        assert after == seeded, (
            f"{label}: cancellation {attempt + 1} left the engine off its seeded "
            f"baseline ({observed['emitted']} tokens published, "
            f"finish={observed['finish']!r}); seeded={seeded} after={after}"
        )

    # The prefix the cancellation interrupted must still be intact: the same
    # request has to reproduce the seeded ids exactly.
    repeated = _seed(llm, prompt, mtp=mtp)
    assert repeated == seeded_ids, (
        f"{label}: the request after the cancellations produced {repeated}, not "
        f"the seeded {seeded_ids}, so the reused prefix was corrupted"
    )


@pytest.mark.parametrize("cache_mode", CACHE_MODES)
def test_prefix_reuse_after_cancellation_leaves_no_orphaned_ownership(
    cache_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled request that reused its prefix returns the engine to baseline."""

    # The gfx1151 artifact's INT8 no-mirror contract is quality-rejected, so the
    # diagnostic override is what reaches the INT8 cell on this host.
    monkeypatch.setenv("HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED", "1")
    prompt = _prompt_text()
    llm = LLM(
        str(MODEL),
        backend=BACKEND,
        kv_storage="int8_per_token_head",
        prefix_cache=cache_mode,
        max_active_requests=2,
    )
    try:
        llm.prepare(max_sequence_length=1024)
        prompt_tokens = int(llm.count_tokens(prompt))
        assert prompt_tokens > 512, (
            f"the prompt is {prompt_tokens} tokens, which does not clear the "
            "512-token two-page reuse boundary"
        )
        for mtp in (True, False):
            _arm(llm, prompt, cache_mode=cache_mode, mtp=mtp)
    finally:
        llm.close()
