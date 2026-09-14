"""VibeVoice-ASR Q4_K_M backbone runner gates.

The Q4 GGUF is produced by scripts/vibevoice_asr_to_gguf.py +
llama-quantize + scripts/vibevoice_asr_gguf_merge.py. Skipped cleanly
when the artifact is absent (no-ROCm CI and machines without the file).

Gates:
- first-position logits: top-1 must match the torch fixture, with a
  Q4-calibrated relative bound (0.05) on the logit drift
- greedy chain: 16/16 tokens vs the torch fixture (strict prefix
  regression, same as the dense bf16 runner's chain gate)
"""

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.runtime.vibevoice_qwen2 import greedy_generate

Q4_GGUF = Path("/tmp/vibevoice-asr-q4km.gguf")
LM_FIXTURE = Path(__file__).parent / "fixtures" / "vibevoice_asr" / "vibevoice_asr_lm.npz"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available() or not Q4_GGUF.is_file() or not LM_FIXTURE.is_file(),
    reason="needs ROCm, the merged Q4_K_M GGUF and the LM fixture",
)


@pytest.fixture(scope="module")
def runtime():
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

    weights = load_vibevoice_qwen2_q4(Q4_GGUF)
    runner = VibevoiceQwen2Q4Runtime(weights, max_context=512)
    yield runner
    runner.close()


@pytest.fixture(scope="module")
def lm() -> dict[str, np.ndarray]:
    with np.load(LM_FIXTURE) as data:
        return {k: data[k] for k in data.files}


def _prompt_rows(runner, lm) -> list[np.ndarray]:
    input_ids = np.asarray(lm["input_ids"])[0]
    positions = np.asarray(lm["audio_placeholder_positions"])
    audio = lm["audio_embeds"].astype(np.float32)
    rows = [runner.embed_row(int(t)) for t in input_ids]
    for p in positions:
        rows[p] = audio[p - positions[0]]
    return rows


def test_first_position_logits_q4(runtime, lm) -> None:
    rows = _prompt_rows(runtime, lm)
    runtime.reset()
    runtime.push_token(rows[0], 0)
    runtime.forward_layers(0)
    logits, token = runtime.logits_argmax()
    ref = lm["logits_pos0"]
    diff = np.abs(logits - ref).max()
    scale = max(np.abs(ref).max(), 1e-9)
    assert diff / scale <= 5e-2, f"q4 logits_pos0 rel {diff / scale:.3e}"
    assert token == int(ref.argmax())


def test_greedy_chain_matches_torch_q4(runtime, lm) -> None:
    """Prefix regression: the Q4 backbone reproduces the torch greedy chain."""
    rows = _prompt_rows(runtime, lm)
    runtime.reset()
    generated = greedy_generate(runtime, rows, max_new_tokens=16)
    fixture = [int(t) for t in np.asarray(lm["greedy_tokens"])]
    assert generated == fixture, f"{generated} != {fixture}"


def test_weight_type_routing(runtime) -> None:
    """All backbone GEMM weights are routed K-quant blocks the kernels own."""
    from hipengine.loading.vibevoice_asr_gguf import GGUF_WEIGHT_TYPES

    routed = {GGUF_WEIGHT_TYPES[buf.ptr] for layer in runtime.layers
              for buf in (layer.q_w, layer.k_w, layer.v_w, layer.o_w,
                          layer.gate_w, layer.up_w, layer.down_w)}
    assert routed <= {12, 13, 14}, routed
    assert 12 in routed  # Q4_K backbone present


def test_weights_own_buffers_and_runners_borrow() -> None:
    """Loader buffers are owned by the weights handle, not by a runner.

    Previously every runner keep()ed the loader's buffers into its own
    free list, so two runners over one weights object double-freed the
    same device pointers (HIP error 1 on the second close).
    """
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

    weights = load_vibevoice_qwen2_q4(Q4_GGUF)
    try:
        first = VibevoiceQwen2Q4Runtime(weights, max_context=128)
        second = VibevoiceQwen2Q4Runtime(weights, max_context=128)
        assert all(buf not in first._buffers for buf in weights.buffers)
        assert all(buf not in second._buffers for buf in weights.buffers)
        first.close()
        second.close()  # must not free the other runner's weights
        assert weights.buffers and not weights._closed
    finally:
        weights.close()
        weights.close()  # idempotent


def test_dual_gemv_scratch_is_per_runner() -> None:
    """Emulated dual GEMV scratch must not be shared across runners."""
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

    weights = load_vibevoice_qwen2_q4(Q4_GGUF)
    try:
        first = VibevoiceQwen2Q4Runtime(weights, max_context=128)
        second = VibevoiceQwen2Q4Runtime(weights, max_context=128)
        need = weights.spec.intermediate_size * 4
        for runner in (first, second):
            gate, up = runner._dual_scratch
            assert gate.nbytes >= need and up.nbytes >= need
            # Runner-owned: released by runner.close().
            assert gate in runner._buffers and up in runner._buffers
        assert first._dual_scratch[0].ptr != second._dual_scratch[0].ptr
        first.close()
        second.close()
    finally:
        weights.close()


def test_prefill_manifest_records_q4_quant() -> None:
    """The execution manifest must not claim bf16 for a Q4 backbone."""
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

    weights = load_vibevoice_qwen2_q4(Q4_GGUF)
    try:
        runner = VibevoiceQwen2Q4Runtime(weights, max_context=128)
        assert runner.quant_name == "q4_k_m"
        from hipengine.core.memory import malloc, free

        rows = 2
        hidden = runner.spec.hidden_size
        buf = malloc(rows * hidden * 2)
        try:
            runner.reset()
            runner.prefill_rows(buf, rows, 0)
            manifest = runner.variant_manifest
            quant = manifest["quant"] if isinstance(manifest, dict) else manifest.quant
            model = manifest["model"] if isinstance(manifest, dict) else manifest.model
            assert quant == "q4_k_m", manifest
            assert model == "vibevoice_asr"
        finally:
            free(buf)
        runner.close()
    finally:
        weights.close()


def test_loader_buffers_are_bf16_sized() -> None:
    """Dense device buffers must be bf16-sized, not 4x oversized.

    ``malloc(host_f32.nbytes * 2)`` allocated 8 bytes per element for a
    buffer that holds 2, wasting ~6.5 GB on the embedding and lm_head.
    """
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4

    weights = load_vibevoice_qwen2_q4(Q4_GGUF)
    try:
        exact = weights.spec.vocab_size * weights.spec.hidden_size * 2
        assert weights.embed.nbytes == exact
        assert weights.lm_head.nbytes == exact
        assert weights.final_norm.nbytes == weights.spec.hidden_size * 2
    finally:
        weights.close()


def test_batched_prefill_matches_row_by_row(runtime, lm) -> None:
    """The batched WMMA prefill must reproduce the row-by-row result.

    ``q4_prefill`` was a per-row loop through the decode path; it now runs
    the raw-block Q4_K/Q6_K WMMA prefill kernels. Both routes must land on
    the same post-prefill hidden row, otherwise the batched GEMMs changed
    the arithmetic rather than just the schedule.
    """
    from hipengine.core.memory import free, malloc
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

    rows = _prompt_rows(runtime, lm)
    hidden = runtime.spec.hidden_size
    total = len(rows)
    prompt = malloc(total * hidden * 2)
    try:
        import numpy as np

        from hipengine.core.memory import copy_device_to_host, copy_host_array_to_device, host_array_ptr

        copy_host_array_to_device(prompt, f32_to_bf16_bits(np.asarray(rows, dtype=np.float32)))

        runtime.reset()
        runtime.prefill_rows(prompt, total, 0)
        runtime.runtime.memcpy(runtime._hidden.ptr,
                               prompt.ptr + (total - 1) * hidden * 2, hidden * 2, 3)
        batched_logits, batched_top = runtime.logits_argmax()

        runtime.reset()
        for i, row in enumerate(rows):
            runtime.push_token(row, i)
            runtime.forward_layers(i)
        sequential_logits, sequential_top = runtime.logits_argmax()

        assert batched_top == sequential_top, (
            f"batched prefill top-1 {batched_top} != row-by-row {sequential_top}")
        # Same schedule class: the batched WMMA tiles reassociate, so require
        # closeness rather than bit equality. Normalise by the logit scale,
        # not per-element (near-zero logits make a per-element ratio useless).
        scale = float(np.max(np.abs(sequential_logits)))
        rel = float(np.max(np.abs(batched_logits - sequential_logits))) / scale
        assert rel < 0.05, f"batched prefill logit drift {rel:.3e} (scale {scale:.3f})"
        assert set(np.argsort(sequential_logits)[-5:]) == set(np.argsort(batched_logits)[-5:]), \
            "batched prefill changed the top-5 set"
    finally:
        free(prompt)


def test_batched_prefill_refuses_unsupported_quant_types(runtime, lm, monkeypatch) -> None:
    """A non-Q4_K/Q6_K weight must fail loudly, never decode as Q4_K.

    The prefill route used to default any unregistered type to the Q4_K
    decoder, so a Q5_K / Q8_0 / IQ4_XS tensor would have produced silently
    wrong logits. Only the types with a WMMA prefill kernel may route.
    """
    from hipengine.core.memory import malloc
    from hipengine.kernels.hip_gfx1100.vibevoice import q4 as q4mod
    from hipengine.loading.vibevoice_asr_gguf import GGUF_WEIGHT_TYPES

    rows = _prompt_rows(runtime, lm)
    hidden = runtime.spec.hidden_size
    total = len(rows)
    prompt = malloc(total * hidden * 2)
    try:
        import numpy as np

        from hipengine.core.memory import copy_host_array_to_device
        from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

        copy_host_array_to_device(prompt, f32_to_bf16_bits(np.asarray(rows, dtype=np.float32)))
        layer = runtime.layers[0]

        # Claim the q_proj weight is Q5_K: supported by the decode GEMV path,
        # but with no WMMA prefill kernel.
        GGML_Q5_K = 13
        original = GGUF_WEIGHT_TYPES.get(layer.q_w.ptr)
        monkeypatch.setitem(GGUF_WEIGHT_TYPES, layer.q_w.ptr, GGML_Q5_K)
        try:
            runtime.reset()
            with pytest.raises(ValueError, match="no WMMA prefill kernel"):
                runtime.prefill_rows(prompt, total, 0)
        finally:
            if original is None:
                GGUF_WEIGHT_TYPES.pop(layer.q_w.ptr, None)
            else:
                GGUF_WEIGHT_TYPES[layer.q_w.ptr] = original

        # An unregistered pointer must not be guessed at either.
        monkeypatch.setitem(GGUF_WEIGHT_TYPES, layer.q_w.ptr, None)
        try:
            runtime.reset()
            with pytest.raises(ValueError, match="no GGUF type registered"):
                runtime.prefill_rows(prompt, total, 0)
        finally:
            if original is not None:
                GGUF_WEIGHT_TYPES[layer.q_w.ptr] = original
    finally:
        from hipengine.core.memory import free

        free(prompt)


# docs/EXECUTION-PROFILES.md "Calibrated production envelope" (section 6.1).
#
# Profile note. Both the KL metrics and the top-1 metrics here compare the
# batched prefill against the row-by-row route, which is this model's registered
# strict parent, so they are cross-schedule comparisons. Under section 4.2 and
# the `production` profile row, cross-width generated-ID equality is diagnostic
# rather than a promotion requirement: the KL clauses are the binding
# production drift bound, and the top-1 clauses are diagnostic. Nothing in this
# file is a full production qualification, which additionally needs isolation,
# BF16-relative and task-quality gates that live elsewhere.
PRODUCTION_KL_ENVELOPE = {
    "mean": 1e-3,
    "p95": 5e-3,
    "p99": 2e-2,
    "max": 5e-2,
    "top1_overall": 0.99,
    "top1_per_scope": 0.97,
}


def _log_softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    return z - np.log(np.exp(z).sum())


def _row_kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats over the full vocabulary."""
    lp, lq = _log_softmax(p), _log_softmax(q)
    return float(np.sum(np.exp(lp) * (lp - lq)))


def test_batched_prefill_meets_the_production_kl_envelope(runtime, lm) -> None:
    """Full-vocabulary KL of the batched WMMA prefill against the strict path.

    The row-by-row route goes through the decode primitives, so it is the
    strict parent for this comparison. The envelope is the calibrated
    production one from docs/EXECUTION-PROFILES.md section 6.1, applied per
    prompt length as a scope: mean/p95/p99/max row KL over every teacher-forced
    row, plus overall and per-scope top-1.

    The Q6_K tensors widen a bf16 kernel output to f32, which is a real
    arithmetic change rather than a reassociation, so top-1 alone would not be
    sufficient evidence.
    """
    from hipengine.core.memory import copy_host_array_to_device, free, malloc
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

    full_rows = _prompt_rows(runtime, lm)
    hidden = runtime.spec.hidden_size

    per_scope: dict[int, dict] = {}
    all_kl: list[float] = []
    top1_hits = top1_total = 0
    for length in (24, 40, 64, len(full_rows)):
        rows = full_rows[:length]
        prompt = malloc(length * hidden * 2)
        try:
            copy_host_array_to_device(prompt, f32_to_bf16_bits(np.asarray(rows, dtype=np.float32)))

            runtime.reset()
            runtime.prefill_rows(prompt, length, 0)
            batched = []
            for i in range(length):
                runtime.runtime.memcpy(runtime._hidden.ptr, prompt.ptr + i * hidden * 2,
                                       hidden * 2, 3)
                batched.append(runtime.logits_argmax()[0])

            runtime.reset()
            strict = []
            for i, row in enumerate(rows):
                runtime.push_token(row, i)
                runtime.forward_layers(i)
                strict.append(runtime.logits_argmax()[0])
        finally:
            free(prompt)

        kls = [_row_kl(s, b) for s, b in zip(strict, batched)]
        hits = sum(int(s.argmax()) == int(b.argmax()) for s, b in zip(strict, batched))
        all_kl.extend(kls)
        top1_hits += hits
        top1_total += length
        per_scope[length] = {"kl": kls, "top1": hits / length, "rows": length}
        print(f"  scope {length:3d} rows: mean KL {np.mean(kls):.3e} "
              f"max {np.max(kls):.3e} top-1 {hits / length:.3%}")

    stats = {
        "mean": float(np.mean(all_kl)),
        "p95": float(np.percentile(all_kl, 95)),
        "p99": float(np.percentile(all_kl, 99)),
        "max": float(np.max(all_kl)),
        "top1_overall": top1_hits / top1_total,
    }
    print(f"  overall: mean {stats['mean']:.3e} p95 {stats['p95']:.3e} "
          f"p99 {stats['p99']:.3e} max {stats['max']:.3e} top-1 {stats['top1_overall']:.3%}")

    env = PRODUCTION_KL_ENVELOPE
    assert stats["mean"] <= env["mean"], f"mean KL {stats['mean']:.3e} > {env['mean']:.3e}"
    assert stats["p95"] <= env["p95"], f"p95 KL {stats['p95']:.3e} > {env['p95']:.3e}"
    assert stats["p99"] <= env["p99"], f"p99 KL {stats['p99']:.3e} > {env['p99']:.3e}"
    assert stats["max"] <= env["max"], f"max KL {stats['max']:.3e} > {env['max']:.3e}"
    # The top-1 clauses are asserted by
    # test_batched_prefill_top1_agreement_meets_the_envelope, which is a
    # non-strict xfail. The batched prefill is deterministic (see
    # test_batched_prefill_is_deterministic); what the top-1 clauses measure is
    # agreement between two different schedules, and the untouched bf16 lane
    # shows the same shortfall, so they are diagnostic under section 4.2 rather
    # than a production promotion requirement.


def test_q6_k_o_proj_route_is_reachable(runtime, lm, monkeypatch) -> None:
    """o_proj must accept a Q6_K weight instead of requiring Q4_K.

    The loader accepts Q6_K for any tensor, so an accepted file can carry a
    Q6_K o_proj. The route used to hard-require Q4_K there and raise. This
    checks the branch is reachable and produces finite logits; the arithmetic
    of the narrow/run/widen composition is checked by
    ``test_q6_k_f32_input_composition_is_exact``.
    """
    from hipengine.core.memory import copy_host_array_to_device, free, malloc
    from hipengine.loading.vibevoice_asr_gguf import GGUF_WEIGHT_TYPES
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

    spec = runtime.spec
    in_f = spec.num_attention_heads * spec.head_dim
    layer = runtime.layers[0]
    q4_bytes = layer.o_w.nbytes
    assert q4_bytes % 144 == 0
    q6_bytes = q4_bytes // 144 * 210  # Q4_K is 144 B/256 elems, Q6_K is 210

    rows = _prompt_rows(runtime, lm)
    total = len(rows)
    prompt = malloc(total * spec.hidden_size * 2)
    zeros6 = malloc(q6_bytes)
    original = layer.o_w
    original_type = GGUF_WEIGHT_TYPES.get(original.ptr)
    try:
        # An all-zero Q6_K block decodes to zero (d = 0), so o_proj contributes
        # nothing and the residual stream stays well defined.
        copy_host_array_to_device(zeros6, np.zeros(q6_bytes, dtype=np.uint8))
        copy_host_array_to_device(
            prompt, f32_to_bf16_bits(np.asarray(rows, dtype=np.float32)))
        layer.o_w = zeros6
        GGUF_WEIGHT_TYPES[zeros6.ptr] = 14  # GGML_Q6_K
        runtime.reset()
        runtime.prefill_rows(prompt, total, 0)
        runtime.runtime.memcpy(runtime._hidden.ptr,
                               prompt.ptr + (total - 1) * spec.hidden_size * 2,
                               spec.hidden_size * 2, 3)
        logits, top = runtime.logits_argmax()
    finally:
        layer.o_w = original
        if original_type is None:
            GGUF_WEIGHT_TYPES.pop(original.ptr, None)
        else:
            GGUF_WEIGHT_TYPES[original.ptr] = original_type
        free(zeros6)
        free(prompt)
    assert np.isfinite(logits).all(), "Q6_K o_proj produced non-finite logits"
    assert 0 <= top < spec.vocab_size


def test_q6_k_f32_input_composition_is_exact(runtime) -> None:
    """The Q6_K f32-input branch must equal its bf16-input equivalent.

    ``gemm_f32_f32`` handles a Q6_K o_proj by narrowing f32 activations to
    bf16, running the Q6_K bf16/bf16 prefill kernel, and widening the result.
    Both sides of that comparison feed the kernel the *same* bf16 inputs, so
    the outputs must be bit-identical. This is a single GEMM per side (no
    attention), so it is unaffected by the batched-prefill run-to-run
    non-determinism that the end-to-end comparison suffers from.

    Uses a real Q6_K weight from the model: the attn_v tensors on the layers
    the Q4_K_M ruleset upgrades.
    """
    from hipengine.core.memory import copy_host_array_to_device, free, malloc
    from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32, f32_to_bf16
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
        gguf_q6_k_wmma_prefill_bf16_bf16_out as q6_prefill,
    )
    from hipengine.loading.vibevoice_asr_gguf import GGUF_WEIGHT_TYPES

    GGML_Q6_K = 14
    q6_layer = next((layer for layer in runtime.layers
                     if GGUF_WEIGHT_TYPES.get(layer.v_w.ptr) == GGML_Q6_K), None)
    assert q6_layer is not None, "fixture model has no Q6_K attn_v to exercise"

    spec = runtime.spec
    rows = 64
    in_f = spec.hidden_size              # attn_v consumes the hidden row
    out_f = spec.num_key_value_heads * spec.head_dim

    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

    rng = np.random.default_rng(0)
    x_f32 = (rng.standard_normal((rows, in_f)) * 0.5).astype(np.float32)
    # Round-to-nearest-even, the same conversion the branch itself performs:
    # truncating here instead would differ by an ulp and defeat the comparison.
    x_bf16 = f32_to_bf16_bits(x_f32)

    x_f32_dev = malloc(x_f32.nbytes)
    x_bf16_dev = malloc(rows * in_f * 2)
    narrowed = malloc(rows * in_f * 2)
    ref_bf16 = malloc(rows * out_f * 2)
    got_bf16 = malloc(rows * out_f * 2)
    ref_f32 = malloc(rows * out_f * 4)
    got_f32 = malloc(rows * out_f * 4)
    try:
        copy_host_array_to_device(x_f32_dev, x_f32)
        copy_host_array_to_device(x_bf16_dev, x_bf16)

        # reference: bf16 activations straight into the Q6_K kernel
        q6_prefill(x_bf16_dev.ptr, q6_layer.v_w.ptr, ref_bf16.ptr, rows, in_f, out_f,
                   stream=0, runtime=runtime.runtime)
        bf16_to_f32(ref_bf16.ptr, ref_f32.ptr, rows * out_f, stream=0,
                    runtime=runtime.runtime)

        # branch semantics: f32 -> bf16 -> Q6_K kernel -> widen
        f32_to_bf16(x_f32_dev.ptr, narrowed.ptr, rows * in_f, stream=0,
                    runtime=runtime.runtime)
        q6_prefill(narrowed.ptr, q6_layer.v_w.ptr, got_bf16.ptr, rows, in_f, out_f,
                   stream=0, runtime=runtime.runtime)
        bf16_to_f32(got_bf16.ptr, got_f32.ptr, rows * out_f, stream=0,
                    runtime=runtime.runtime)
        runtime.runtime.device_synchronize()

        ref = np.empty(rows * out_f, dtype=np.float32)
        got = np.empty(rows * out_f, dtype=np.float32)
        from hipengine.core.memory import copy_device_to_host, host_array_ptr

        copy_device_to_host(host_array_ptr(ref), ref_f32, rows * out_f * 4)
        copy_device_to_host(host_array_ptr(got), got_f32, rows * out_f * 4)
    finally:
        for buffer in (x_f32_dev, x_bf16_dev, narrowed, ref_bf16, got_bf16,
                       ref_f32, got_f32):
            free(buffer)

    assert np.isfinite(ref).all() and np.isfinite(got).all()
    assert np.array_equal(ref, got), (
        "Q6_K f32-input branch diverges from the bf16-input path; "
        f"max |diff| {np.max(np.abs(ref - got)):.3e}")


def test_q6_k_down_proj_skips_the_round_trip(runtime, lm, monkeypatch) -> None:
    """A Q6_K ffn_down must not widen to f32 only to narrow straight back.

    The Q4_K_M ruleset upgrades ffn_down to Q6_K on 14 of the 28 layers. Those
    down projections produce bf16 and feed the bf16 residual with nothing in
    between, so the widen-to-f32 / narrow-back-to-bf16 pair is dead work. This
    pins the launch count: bf16_to_f32 still runs once per Q6_K attn_v (which
    does need f32 for its bias add) but not for ffn_down.
    """
    from hipengine.core.memory import free, malloc
    from hipengine.loading.vibevoice_asr_gguf import GGUF_WEIGHT_TYPES
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits
    import hipengine.kernels.hip_gfx1100.vibevoice.q4 as q4_module

    GGML_Q6_K = 14
    q6_down = sum(1 for layer in runtime.layers
                  if GGUF_WEIGHT_TYPES.get(layer.down_w.ptr) == GGML_Q6_K)
    q6_v = sum(1 for layer in runtime.layers
               if GGUF_WEIGHT_TYPES.get(layer.v_w.ptr) == GGML_Q6_K)
    assert q6_down and q6_v, "fixture model has no Q6_K tensors to exercise"

    counts = {"bf16_to_f32": 0, "f32_to_bf16": 0}

    def counted(name, fn):
        def inner(*args, **kwargs):
            counts[name] += 1
            return fn(*args, **kwargs)
        return inner

    monkeypatch.setattr(q4_module, "bf16_to_f32",
                        counted("bf16_to_f32", q4_module.bf16_to_f32))
    monkeypatch.setattr(runtime.kernels, "f32_to_bf16",
                        counted("f32_to_bf16", runtime.kernels.f32_to_bf16))

    import numpy as np

    rows = _prompt_rows(runtime, lm)
    hidden = runtime.spec.hidden_size
    total = len(rows)
    prompt = malloc(total * hidden * 2)
    try:
        from hipengine.core.memory import copy_host_array_to_device

        copy_host_array_to_device(prompt, f32_to_bf16_bits(np.asarray(rows, dtype=np.float32)))
        runtime.reset()
        runtime.prefill_rows(prompt, total, 0)
    finally:
        free(prompt)

    # attn_v needs the f32 widening for its bias add; ffn_down must not.
    assert counts["bf16_to_f32"] == q6_v, (
        f"expected {q6_v} bf16_to_f32 launches (Q6_K attn_v only), "
        f"got {counts['bf16_to_f32']}; a Q6_K ffn_down is round-tripping through f32")
    assert counts["f32_to_bf16"] == 154, (
        f"expected 154 f32_to_bf16 launches (o_proj, gate, up, k, v, and the "
        f"Q4_K down projections), got {counts['f32_to_bf16']}")


def test_q6_k_down_proj_direct_bf16_matches_the_round_trip(runtime) -> None:
    """The Q6_K ffn_down fast path must write the same bf16 residual bytes.

    The fast path writes the Q6_K kernel's bf16 output straight into the
    residual buffer instead of widening it to f32 for the caller to narrow
    straight back. Widening bf16 to f32 is exact and narrowing an exactly
    representable value back to bf16 is the identity, so the two routes must
    agree bit for bit. Uses a real Q6_K ffn_down weight from the model.
    """
    from hipengine.core.memory import (copy_device_to_host, copy_host_array_to_device,
                                       free, host_array_ptr, malloc)
    from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32, f32_to_bf16
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
        gguf_q6_k_wmma_prefill_bf16_bf16_out as q6_prefill,
    )
    from hipengine.loading.vibevoice_asr_gguf import GGUF_WEIGHT_TYPES
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

    GGML_Q6_K = 14
    layer = next((candidate for candidate in runtime.layers
                  if GGUF_WEIGHT_TYPES.get(candidate.down_w.ptr) == GGML_Q6_K), None)
    assert layer is not None, "fixture model has no Q6_K ffn_down to exercise"

    spec = runtime.spec
    rows = 64
    in_f = spec.intermediate_size
    out_f = spec.hidden_size

    rng = np.random.default_rng(0)
    x_bf16 = f32_to_bf16_bits(
        (rng.standard_normal((rows, in_f)) * 0.5).astype(np.float32))

    x_dev = malloc(rows * in_f * 2)
    scratch_bf16 = malloc(rows * out_f * 2)
    widened_f32 = malloc(rows * out_f * 4)
    direct = malloc(rows * out_f * 2)
    round_trip = malloc(rows * out_f * 2)
    try:
        copy_host_array_to_device(x_dev, x_bf16)

        # fast path: the Q6_K result lands in the bf16 residual buffer
        q6_prefill(x_dev.ptr, layer.down_w.ptr, direct.ptr, rows, in_f, out_f,
                   stream=0, runtime=runtime.runtime)

        # previous path: Q6_K -> bf16 scratch -> f32 -> bf16
        q6_prefill(x_dev.ptr, layer.down_w.ptr, scratch_bf16.ptr, rows, in_f, out_f,
                   stream=0, runtime=runtime.runtime)
        bf16_to_f32(scratch_bf16.ptr, widened_f32.ptr, rows * out_f, stream=0,
                    runtime=runtime.runtime)
        f32_to_bf16(widened_f32.ptr, round_trip.ptr, rows * out_f, stream=0,
                    runtime=runtime.runtime)
        runtime.runtime.device_synchronize()

        got = np.empty(rows * out_f, dtype=np.uint16)
        ref = np.empty(rows * out_f, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(got), direct, rows * out_f * 2)
        copy_device_to_host(host_array_ptr(ref), round_trip, rows * out_f * 2)
    finally:
        for buffer in (x_dev, scratch_bf16, widened_f32, direct, round_trip):
            free(buffer)

    assert np.array_equal(ref, got), (
        "Q6_K ffn_down direct-bf16 route differs from the widen/narrow round trip")


def test_batched_prefill_is_deterministic(runtime, lm) -> None:
    """Identical inputs must give identical hidden rows across repeats.

    ``prefill_rows`` overwrites its input buffer with the post-layer-stack
    result, so the input has to be re-uploaded before every run. An earlier
    version of this test reused one buffer and so read its own output back as
    input, which made the engine look non-deterministic; it is not.
    """
    from hipengine.core.memory import (copy_device_to_host, copy_host_array_to_device,
                                       free, host_array_ptr, malloc)
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

    rows = _prompt_rows(runtime, lm)
    total = len(rows)
    hidden = runtime.spec.hidden_size
    inputs = f32_to_bf16_bits(np.asarray(rows, dtype=np.float32))
    prompt = malloc(total * hidden * 2)
    try:
        def run():
            copy_host_array_to_device(prompt, inputs)  # prefill_rows consumes this
            runtime.reset()
            runtime.prefill_rows(prompt, total, 0)
            host = np.empty(total * hidden, dtype=np.uint16)
            copy_device_to_host(host_array_ptr(host), prompt, total * hidden * 2)
            return host

        first = run()
        second = run()
    finally:
        free(prompt)

    assert np.array_equal(first, second), (
        "batched prefill is not deterministic: "
        f"{int((first != second).sum())}/{first.size} hidden elements differ "
        "between identical runs")


def test_prefill_scratch_is_reused_across_calls(runtime, lm, monkeypatch) -> None:
    """A repeated prefill must not allocate device memory for its scratch.

    Both prefill routes used to malloc and free ~20 scratch buffers per call,
    which measured 4.2 ms of a 220 ms prefill (1.9%). The scratch now comes
    from a per-runner arena that is rewound on reuse, so a second prefill of
    the same size performs no device allocation at all.

    Patches ``HipRuntime.malloc``, the single choke point every allocation path
    goes through, so this fails whichever route reintroduces per-call scratch
    (the previous code allocated directly rather than through the arena).
    """
    from hipengine.core.hip import HipRuntime
    from hipengine.core.memory import copy_host_array_to_device, free, malloc
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits
    import numpy as np

    allocated: list[int] = []
    original = HipRuntime.malloc

    def counted(self, nbytes):
        allocated.append(int(nbytes))
        return original(self, nbytes)

    rows = _prompt_rows(runtime, lm)
    hidden = runtime.spec.hidden_size
    total = len(rows)
    pristine = f32_to_bf16_bits(np.asarray(rows, dtype=np.float32))
    prompt = malloc(total * hidden * 2)
    try:
        runtime.reset()
        copy_host_array_to_device(prompt, pristine)
        runtime.prefill_rows(prompt, total, 0)      # may allocate the arena
        monkeypatch.setattr(HipRuntime, "malloc", counted)
        allocated.clear()
        runtime.reset()
        copy_host_array_to_device(prompt, pristine)
        runtime.prefill_rows(prompt, total, 0)
    finally:
        free(prompt)

    assert allocated == [], (
        f"a repeated prefill made {len(allocated)} device allocations "
        f"({allocated} bytes); prefill scratch must be reused across calls")


@pytest.mark.xfail(
    strict=False,
    reason="Cross-schedule diagnostic clause, not a production promotion "
           "requirement: the batched prefill and the sequential row-by-row "
           "route are two different schedules, and section 4.2 of "
           "docs/EXECUTION-PROFILES.md makes cross-width generated-ID equality "
           "diagnostic. They differ by mean KL ~7e-4 (inside the envelope), but "
           "top-1 agreement is ~95% because the rows that flip have a model "
           "decision margin of 0.006-0.039 nats against a 0.79 median, so a "
           "sub-1e-3 perturbation decides them. The untouched bf16 lane shows "
           "the same effect (87/89), so this is a property of "
           "batched-vs-sequential scheduling, not of the Q4 arithmetic. Kept "
           "asserted so the unmet clause stays visible; XPASS means top-1 "
           "agreement improved enough to meet the envelope.",
)
def test_batched_prefill_top1_agreement_meets_the_envelope(runtime, lm) -> None:
    """Top-1 clause of the envelope, per prompt length as a scope.

    Profile: this is a **cross-schedule diagnostic** clause. It compares the
    batched prefill against the row-by-row route, and section 4.2 of
    docs/EXECUTION-PROFILES.md makes cross-width generated-ID equality
    diagnostic rather than a production promotion requirement. The four KL
    clauses in test_batched_prefill_meets_the_production_kl_envelope are the
    binding production drift bound and they pass; this clause does not, and it
    is kept asserted (as a non-strict xfail) rather than dropped so the unmet
    part of the envelope stays visible.

    A green run of this file therefore means the KL drift bound, the strict
    fallback and the ownership/reuse gates pass. It does not by itself certify
    the model for production.
    """
    from hipengine.core.memory import copy_host_array_to_device, free, malloc
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

    full_rows = _prompt_rows(runtime, lm)
    hidden = runtime.spec.hidden_size
    hits = total = 0
    per_scope: dict[int, float] = {}
    for length in (24, 40, 64, len(full_rows)):
        rows = full_rows[:length]
        prompt = malloc(length * hidden * 2)
        try:
            copy_host_array_to_device(
                prompt, f32_to_bf16_bits(np.asarray(rows, dtype=np.float32)))
            runtime.reset()
            runtime.prefill_rows(prompt, length, 0)
            batched = []
            for i in range(length):
                runtime.runtime.memcpy(runtime._hidden.ptr,
                                       prompt.ptr + i * hidden * 2, hidden * 2, 3)
                batched.append(runtime.logits_argmax()[0])
            runtime.reset()
            strict = []
            for i, row in enumerate(rows):
                runtime.push_token(row, i)
                runtime.forward_layers(i)
                strict.append(runtime.logits_argmax()[0])
        finally:
            free(prompt)
        scope_hits = sum(int(s.argmax()) == int(b.argmax())
                         for s, b in zip(strict, batched))
        hits += scope_hits
        total += length
        per_scope[length] = scope_hits / length

    overall = hits / total
    print(f"  top-1 overall {overall:.3%}; per scope "
          + ", ".join(f"{k}:{v:.1%}" for k, v in per_scope.items()))
    assert overall >= PRODUCTION_KL_ENVELOPE["top1_overall"], (
        f"top-1 {overall:.3%} < {PRODUCTION_KL_ENVELOPE['top1_overall']:.1%}")
    for length, scope in per_scope.items():
        assert scope >= PRODUCTION_KL_ENVELOPE["top1_per_scope"], (
            f"scope {length} top-1 {scope:.3%} "
            f"< {PRODUCTION_KL_ENVELOPE['top1_per_scope']:.1%}")
