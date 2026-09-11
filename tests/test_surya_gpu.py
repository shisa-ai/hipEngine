"""GPU (hip_gfx1151) Surya OCR pipeline tests: vision tower, end-to-end OCR.

Correctness contract: the HIP vision tower matches the CPU reference
fp32 vision forward within calibrated fp32-GEMM tolerances, and the full
GPU pipeline (GPU vision features + GPU prefill/decode) reproduces the
torch fp32 oracle greedy IDs exactly. Skips without ROCm or the model
checkpoint.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP not available"),
]

FIXTURES = Path("tests/fixtures/surya")
MODEL_ID = "datalab-to/surya-ocr-2"


def _model_dir() -> Path:
    from hipengine.loading.surya import resolve_surya_path

    try:
        return resolve_surya_path(MODEL_ID)
    except FileNotFoundError:
        pytest.skip(f"{MODEL_ID} not in local HF cache")


@pytest.fixture(scope="module")
def runner():
    from hipengine.loading.surya import load_surya_spec, load_surya_weights
    from hipengine.runtime.surya import SuryaGpuRunner

    model_dir = _model_dir()
    spec = load_surya_spec(model_dir)
    weights = load_surya_weights(model_dir)
    r = SuryaGpuRunner(weights, spec)
    yield r, weights, spec
    r.close()


def _page_inputs():
    from hipengine.loading.surya import preprocess_image_surya
    from PIL import Image

    page = FIXTURES / "page_small.png"
    if not page.exists():
        pytest.skip("page_small.png fixture not present")
    return preprocess_image_surya(Image.open(page).convert("RGB"))


def _rect_inputs():
    from hipengine.loading.surya import preprocess_image_surya
    from PIL import Image

    page = FIXTURES / "page_rect.png"
    if not page.exists():
        pytest.skip("page_rect.png fixture not present")
    return preprocess_image_surya(Image.open(page).convert("RGB"))


def _full_page_inputs():
    from hipengine.loading.surya import preprocess_image_surya
    from PIL import Image

    page = FIXTURES / "page_full.png"
    if not page.exists():
        pytest.skip("page_full.png fixture not present")
    return preprocess_image_surya(Image.open(page).convert("RGB"))


def _gpu_ocr_ids(r, spec, tokenizer, pixel_rows, grid, max_tokens: int) -> list[int]:
    from hipengine.loading.surya import compute_mrope_positions, render_chat_prompt

    merged = r.vision_forward(pixel_rows, [grid])
    n_img = (grid[1] // 2) * (grid[2] // 2)
    ids, mm = render_chat_prompt(tokenizer, "Transcribe this page.", n_img)
    pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
    logits = r.prefill(np.asarray(ids, dtype=np.int64), pos, visual_features=merged)
    generated: list[int] = []
    p = int(pos[:, -1].max())
    for step in range(max_tokens):
        nxt = int(np.argmax(logits))
        if nxt == spec.eos_token_id:
            break
        generated.append(nxt)
        if step + 1 >= max_tokens:
            break
        logits = r.decode_step(nxt, p + 1 + step)
    return generated


def _cpu_ocr_ids(weights, spec, tokenizer, pixel_rows, grid, max_tokens: int) -> list[int]:
    from hipengine.kernels.cpu_reference.surya import (
        text_decode_step,
        text_prefill,
        vision_forward,
    )
    from hipengine.loading.surya import compute_mrope_positions, render_chat_prompt

    _, _, merged = vision_forward(weights, spec, pixel_rows, [grid])
    n_img = (grid[1] // 2) * (grid[2] // 2)
    ids, mm = render_chat_prompt(tokenizer, "Transcribe this page.", n_img)
    pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
    hidden, state = text_prefill(
        weights, spec, np.array([ids], dtype=np.int64), pos,
        visual_features=merged[None],
    )
    emb = weights["model.language_model.embed_tokens.weight"]
    logits = hidden[:, -1] @ emb.T
    generated: list[int] = []
    p = int(pos[:, -1].max())
    for step in range(max_tokens):
        nxt = int(np.argmax(logits[0]))
        if nxt == spec.eos_token_id:
            break
        generated.append(nxt)
        if step + 1 >= max_tokens:
            break
        logits = text_decode_step(weights, spec, nxt, state, p + 1 + step)
    return generated


def test_gpu_vision_matches_oracle_rect(runner) -> None:
    """Rectangular page (non-square 14x22 grid) against the torch oracle."""

    r, _weights, _spec = runner

    path = FIXTURES / "oracle_rect.npz"
    if not path.exists():
        pytest.skip("oracle_rect.npz not present; run scripts/surya_oracle_torch.py")
    with np.load(path) as z:
        merged_oracle = z["vision_merged"]

    pixel_rows, grid = _rect_inputs()
    merged_gpu = r.vision_forward(pixel_rows, [grid])
    assert merged_gpu.shape == merged_oracle.shape
    d = np.abs(merged_gpu - merged_oracle)
    assert np.isfinite(merged_gpu).all()
    assert d.max() < 0.5, f"rect vision diverged: max|d|={d.max():.3e}"
    assert d.mean() < 0.05, f"rect vision drift: mean|d|={d.mean():.3e}"


def test_gpu_repeated_request_isolation(runner) -> None:
    """Repeated requests on one runner must not leak state across calls."""

    r, weights, spec = runner
    from hipengine.loading.surya import SuryaTokenizer, resolve_surya_path

    tokenizer = SuryaTokenizer(resolve_surya_path(MODEL_ID))
    small = _page_inputs()
    rect = _rect_inputs()
    max_tokens = 16

    for label, (pixel_rows, grid) in (("small", small), ("rect", rect)):
        ref = _cpu_ocr_ids(weights, spec, tokenizer, pixel_rows, grid, max_tokens)
        assert ref, f"{label}: CPU reference produced no tokens"
        first = _gpu_ocr_ids(r, spec, tokenizer, pixel_rows, grid, max_tokens)
        # a different page in between must not perturb the next request
        _gpu_ocr_ids(r, spec, tokenizer, rect[0], rect[1], max_tokens)
        second = _gpu_ocr_ids(r, spec, tokenizer, pixel_rows, grid, max_tokens)
        assert first == ref, f"{label}: GPU ids diverge from CPU reference"
        assert second == ref, f"{label}: repeated request is not isolated"


def test_gpu_vision_matches_cpu_reference(runner) -> None:
    r, weights, spec = runner
    from hipengine.kernels.cpu_reference.surya import vision_forward

    pixel_rows, grid = _page_inputs()
    _, _, merged_cpu = vision_forward(weights, spec, pixel_rows, [grid])
    merged_gpu = r.vision_forward(pixel_rows, [grid])

    assert merged_gpu.shape == merged_cpu.shape
    d = np.abs(merged_gpu - merged_cpu)
    # fp32 GEMM reassociation through a 12-block tower; the tight gate is
    # end-to-end greedy-ID equality below
    assert d.max() < 0.5, f"vision merger diverged: max|d|={d.max():.3e}"
    assert d.mean() < 0.05, f"vision merger drift: mean|d|={d.mean():.3e}"
    # no NaN/Inf anywhere
    assert np.isfinite(merged_gpu).all()


def test_gpu_vision_matches_oracle_features(runner) -> None:
    """Compare against the torch fp32 oracle's merged features if captured."""

    r, weights, spec = runner
    from hipengine.kernels.cpu_reference.surya import vision_forward

    path = FIXTURES / "oracle_image.npz"
    if not path.exists():
        pytest.skip("oracle_image.npz not present; run scripts/surya_oracle_torch.py")
    with np.load(path) as z:
        if "vision_merged" not in z.files:
            pytest.skip("oracle_image.npz has no vision_merged features")
        merged_oracle = z["vision_merged"]

    pixel_rows, grid = _page_inputs()
    _, _, merged_cpu = vision_forward(weights, spec, pixel_rows, [grid])
    merged_gpu = r.vision_forward(pixel_rows, [grid])
    d_gpu = np.abs(merged_gpu - merged_oracle)
    d_cpu = np.abs(merged_cpu - merged_oracle)
    # the GPU lane must sit in the same noise band as the (oracle-gated)
    # CPU reference, not merely be finite
    assert d_gpu.max() <= d_cpu.max() + 1e-3, (
        f"GPU vision deviates beyond CPU-reference band: "
        f"gpu max={d_gpu.max():.3e} cpu max={d_cpu.max():.3e}"
    )


def test_gpu_vision_first_call_on_fresh_runner(runner) -> None:
    """A brand-new runner's *first* vision forward must match the oracle.

    Regression: the shared-memory softmax reduction in
    ``hipengine_evie_softmax_rows_f32`` read the pass-1 max from ``sm[0]``
    and then overwrote ``sm`` with the partial exponential sums without a
    barrier in between. Whichever thread won that race fed a wrong max into
    the exponentials, so the first call on a fresh runner diverged while
    later calls on the same runner were exact.
    """

    _r, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner

    path = FIXTURES / "oracle_image.npz"
    if not path.exists():
        pytest.skip("oracle_image.npz not present; run scripts/surya_oracle_torch.py")
    with np.load(path) as z:
        if "vision_merged" not in z.files:
            pytest.skip("oracle_image.npz has no vision_merged features")
        merged_oracle = z["vision_merged"]

    pixel_rows, grid = _page_inputs()
    for attempt in range(3):
        r = SuryaGpuRunner(weights, spec)
        try:
            first = r.vision_forward(pixel_rows, [grid])
            second = r.vision_forward(pixel_rows, [grid])
        finally:
            r.close()
        assert np.isfinite(first).all(), f"attempt {attempt}: non-finite output"
        # the race made the first call diverge by ~1e-1; the fixed path sits
        # in the same fp32-GEMM noise band as the CPU reference
        d = np.abs(first - merged_oracle)
        assert d.max() < 5e-3, (
            f"attempt {attempt}: first call on a fresh runner diverged: "
            f"max|d|={d.max():.3e}"
        )
        assert np.array_equal(first, second), (
            f"attempt {attempt}: first and second calls disagree"
        )


def test_gpu_end_to_end_ocr_matches_oracle(runner) -> None:
    """Full GPU pipeline: vision + prefill + decode reproduce oracle IDs."""

    r, weights, spec = runner
    from hipengine.loading.surya import (
        compute_mrope_positions,
        render_chat_prompt,
    )
    from hipengine.loading.surya import SuryaTokenizer, resolve_surya_path

    ref_path = FIXTURES / "oracle_greedy.json"
    if not ref_path.exists():
        pytest.skip("oracle_greedy.json not present; capture with transformers")
    ref = json.loads(ref_path.read_text())

    pixel_rows, grid = _page_inputs()
    merged = r.vision_forward(pixel_rows, [grid])
    n_img = (grid[1] // 2) * (grid[2] // 2)
    tokenizer = SuryaTokenizer(resolve_surya_path(MODEL_ID))
    ids, mm = render_chat_prompt(tokenizer, "Transcribe this page.", n_img)
    pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)

    logits = r.prefill(np.asarray(ids, dtype=np.int64), pos,
                       visual_features=merged)
    generated: list[int] = []
    p = int(pos[:, -1].max())
    for step in range(64):
        nxt = int(np.argmax(logits))
        if nxt == spec.eos_token_id:
            break
        generated.append(nxt)
        logits = r.decode_step(nxt, p + 1 + step)

    assert generated == ref["ids"], (
        "GPU pipeline greedy decode diverged from the torch fp32 reference"
    )


def test_gpu_full_page_layout_matches_oracle(runner) -> None:
    """Full-size realistic page (1024x1024, 1024 merged tokens) vs oracle.

    ``page_small`` (256x256) and ``page_rect`` (320x192) are bar patterns
    whose oracle continuation is a degenerate repeated ``<ul><li>`` run, so
    greedy parity against them is a weak gate. This page renders real words,
    produces a 1x64x64 patch grid, and decodes a long non-degenerate
    layout-JSON sequence — long enough that a small numerical difference
    flips an argmax.
    """

    r, _weights, spec = runner
    from hipengine.loading.surya import (
        compute_mrope_positions,
        render_chat_prompt,
    )
    from hipengine.loading.surya import SuryaTokenizer, resolve_surya_path

    ref_path = FIXTURES / "oracle_fullpage_greedy.json"
    if not ref_path.exists():
        pytest.skip(
            "oracle_fullpage_greedy.json not present; "
            "run scripts/surya_oracle_fullpage.py"
        )
    ref = json.loads(ref_path.read_text())

    pixel_rows, grid = _full_page_inputs()
    assert grid == (1, 64, 64), f"unexpected full-page grid {grid}"
    n_img = (grid[1] // 2) * (grid[2] // 2)
    tokenizer = SuryaTokenizer(resolve_surya_path(MODEL_ID))
    ids, mm = render_chat_prompt(tokenizer, "Transcribe this page.", n_img)
    pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)

    merged = r.vision_forward(pixel_rows, [grid])
    logits = r.prefill(np.asarray(ids, dtype=np.int64), pos,
                       visual_features=merged)
    generated: list[int] = []
    p = int(pos[:, -1].max())
    for step in range(len(ref["ids"]) + 8):
        nxt = int(np.argmax(logits))
        if nxt == spec.eos_token_id:
            break
        generated.append(nxt)
        logits = r.decode_step(nxt, p + 1 + step)

    if generated != ref["ids"]:
        n = min(len(generated), len(ref["ids"]))
        first = next((i for i in range(n) if generated[i] != ref["ids"][i]), n)
        raise AssertionError(
            "GPU full-page greedy decode diverged from the torch fp32 "
            f"reference at index {first} "
            f"(gpu len={len(generated)}, ref len={len(ref['ids'])})"
        )

    # Parity alone only shows the lanes agree. Also check the decoded text is
    # the right answer for the page that was drawn: `_make_synthetic_page_full`
    # puts a 40 pt heading at (60, 50), a horizontal rule at y=105, two prose
    # blocks from y=140, a "Table 1: Regional breakdown" caption at y=458 and
    # four table rows from y=498.
    records = json.loads(tokenizer.decode(generated))
    assert isinstance(records, list) and records, "OCR output is not a JSON list"
    for rec in records:
        assert set(rec) >= {"label", "bbox", "count"}, f"bad record {rec}"
        x1, y1, x2, y2 = (int(v) for v in rec["bbox"].split())
        assert 0 <= x1 < x2 <= 1024 and 0 <= y1 < y2 <= 1024, f"bbox out of range {rec}"

    labels = [rec["label"] for rec in records]
    assert labels == ["Section-Header", "Text", "Text", "Caption", "Table"], (
        f"unexpected layout labels: {labels}"
    )
    header = next(r for r in records if r["label"] == "Section-Header")
    _, hy1, _, hy2 = (int(v) for v in header["bbox"].split())
    assert abs(hy1 - 50) <= 12, f"header top {hy1} does not match the drawn heading"
    assert abs(hy2 - 94) <= 15, f"header bottom {hy2} does not match the drawn heading"
    caption = next(r for r in records if r["label"] == "Caption")
    table = next(r for r in records if r["label"] == "Table")
    _, cy1, _, _ = (int(v) for v in caption["bbox"].split())
    _, ty1, _, _ = (int(v) for v in table["bbox"].split())
    assert 440 <= cy1 <= 480, f"caption top {cy1} does not match the drawn caption"
    assert ty1 >= cy1, f"table top {ty1} should not be above the caption {cy1}"


def test_gpu_full_page_vision_matches_oracle(runner) -> None:
    """Vision tower at full-page scale (4096 patches) vs the torch oracle.

    Localizes failures: if ``test_gpu_full_page_layout_matches_oracle``
    fails, this says whether the vision tower or the decoder is at fault.
    The gate is a fixed bound rather than a CPU-reference band so the test
    does not pay for the ~19 s NumPy vision forward; measured on gfx1151,
    gpu-vs-oracle max|d| is 9.19e-4 and cpu-vs-oracle max|d| is 2.59e-4.
    """

    r, _weights, spec = runner

    path = FIXTURES / "oracle_fullpage.npz"
    if not path.exists():
        pytest.skip(
            "oracle_fullpage.npz not present; "
            "run scripts/surya_oracle_fullpage.py"
        )
    with np.load(path) as z:
        if "vision_merged" not in z.files:
            pytest.skip("oracle_fullpage.npz has no vision_merged features")
        merged_oracle = z["vision_merged"]

    pixel_rows, grid = _full_page_inputs()
    merged_gpu = r.vision_forward(pixel_rows, [grid])
    assert merged_gpu.shape == merged_oracle.shape
    assert np.isfinite(merged_gpu).all()
    d = np.abs(merged_gpu - merged_oracle)
    assert d.max() < 2e-3, f"full-page vision diverged: max|d|={d.max():.3e}"
    assert d.mean() < 1e-5, f"full-page vision drift: mean|d|={d.mean():.3e}"


def test_gpu_text_only_matches_oracle(runner) -> None:
    """Text-only (no image) path: GPU prefill logits and greedy ids.

    The GPU lane exposes text-only generation, but every other GPU test goes
    through the multimodal path. Gate the first-step logits against the torch
    fp32 oracle (``oracle_text.npz``) and the greedy continuation against the
    NumPy CPU reference.
    """

    r, weights, spec = runner
    from hipengine.kernels.cpu_reference.surya import (
        text_decode_step,
        text_prefill,
    )
    from hipengine.loading.surya import (
        compute_mrope_positions,
        render_chat_prompt,
    )
    from hipengine.loading.surya import SuryaTokenizer, resolve_surya_path

    path = FIXTURES / "oracle_text.npz"
    if not path.exists():
        pytest.skip("oracle_text.npz not present; run scripts/surya_oracle_torch.py")
    with np.load(path) as z:
        ref_ids = z["input_ids"][0]
        ref_logits = z["logits_first"]

    tokenizer = SuryaTokenizer(resolve_surya_path(MODEL_ID))
    ids, mm = render_chat_prompt(tokenizer, "Transcribe this page.")
    assert np.array_equal(np.asarray(ids), ref_ids), (
        "text-only prompt drifted from the torch oracle"
    )
    pos = compute_mrope_positions(mm, None)

    logits = r.prefill(np.asarray(ids, dtype=np.int64), pos, visual_features=None)

    def _prob(x):
        x = np.asarray(x, dtype=np.float64)
        e = np.exp(x - x.max())
        return e / e.sum()

    p, q = _prob(logits), _prob(ref_logits)
    kl = float((q * (np.log(q + 1e-300) - np.log(p + 1e-300))).sum())
    assert kl < 1e-4, f"text-only prefill KL vs torch oracle = {kl:.3e}"
    assert int(np.argmax(logits)) == int(np.argmax(ref_logits)), (
        "text-only prefill argmax diverged from the torch oracle"
    )

    emb = weights["model.language_model.embed_tokens.weight"]
    hidden, state = text_prefill(
        weights, spec, np.array([ids], dtype=np.int64), pos, visual_features=None
    )
    cpu_logits = (hidden[:, -1] @ emb.T)[0]
    p = int(pos[:, -1].max())
    cpu_ids: list[int] = []
    for step in range(16):
        c = int(np.argmax(cpu_logits))
        if c == spec.eos_token_id:
            break
        cpu_ids.append(c)
        cpu_logits = text_decode_step(weights, spec, c, state, p + 1 + step)[0]
    assert cpu_ids, "text-only greedy produced no tokens"

    def _gpu_text_only() -> list[int]:
        lg = r.prefill(np.asarray(ids, dtype=np.int64), pos, visual_features=None)
        out: list[int] = []
        for step in range(16):
            nx = int(np.argmax(lg))
            if nx == spec.eos_token_id:
                break
            out.append(nx)
            lg = r.decode_step(nx, p + 1 + step)
        return out

    gpu_ids = _gpu_text_only()
    assert gpu_ids == cpu_ids, (
        f"text-only greedy diverged from the CPU reference: {gpu_ids} vs {cpu_ids}"
    )

    # Regression: a preceding multimodal request must not perturb a following
    # text-only request. decode_step used to advance _seq_len before running
    # the stack, so the KV scatter landed one slot above the next free slot and
    # attention read a slot this request never wrote -- stale KV left by the
    # previous, longer request. A fresh runner masked it because the slot held
    # zeros. The full page is required to expose it: page_small's 256-token
    # request leaves stale KV at the skipped slot that is close enough to the
    # correct value not to flip an argmax.
    pixel_rows, grid = _full_page_inputs()
    merged = r.vision_forward(pixel_rows, [grid])
    n_img = (grid[1] // 2) * (grid[2] // 2)
    m_ids, m_mm = render_chat_prompt(tokenizer, "Transcribe this page.", n_img)
    m_pos = compute_mrope_positions(m_mm, grid, spec.vision_spatial_merge_size)
    m_logits = r.prefill(
        np.asarray(m_ids, dtype=np.int64), m_pos, visual_features=merged
    )
    m_p = int(m_pos[:, -1].max())
    for step in range(8):
        m_nxt = int(np.argmax(m_logits))
        m_logits = r.decode_step(m_nxt, m_p + 1 + step)

    after_ids = _gpu_text_only()
    assert after_ids == cpu_ids, (
        "text-only decode is not isolated from a preceding multimodal "
        f"request: {after_ids} vs {cpu_ids}"
    )


def test_gpu_runner_repeated_create_use_close_releases_memory(runner) -> None:
    """Repeated create/use/close must not leak tracked device allocations.

    Regression guard for the constructor clobbering ``_head_bias_bufs``
    after persistent state/staging allocations populated it, and for the
    untracked cached pointer-array buffers.
    """

    _, weights, spec = runner
    from hipengine.core.memory import memory_stats
    from hipengine.loading.surya import (
        SuryaTokenizer,
        compute_mrope_positions,
        render_chat_prompt,
        resolve_surya_path,
    )
    from hipengine.runtime.surya import SuryaGpuRunner

    pixel_rows, grid = _page_inputs()
    tokenizer = SuryaTokenizer(resolve_surya_path(MODEL_ID))
    n_img = (grid[1] // 2) * (grid[2] // 2)
    ids, mm = render_chat_prompt(tokenizer, "Transcribe this page.", n_img)
    pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
    baseline = memory_stats()["active_allocations"]
    baseline_bytes = memory_stats()["current_allocated_bytes"]

    for _ in range(2):
        r = SuryaGpuRunner(weights, spec)
        try:
            merged = r.vision_forward(pixel_rows, [grid])
            assert np.isfinite(merged).all()
            # text prefill + decode exercise the cached pointer-array buffers
            logits = r.prefill(np.asarray(ids, dtype=np.int64), pos,
                               visual_features=merged)
            nxt = int(np.argmax(logits))
            r.decode_step(nxt, int(pos[:, -1].max()) + 1)
            assert len(r._ptr_array_bufs) > 0, (
                "decode path did not populate the pointer-array cache"
            )
        finally:
            r.close()
        stats = memory_stats()
        assert stats["active_allocations"] == baseline, (
            f"device allocations leaked across close(): "
            f"active={stats['active_allocations']} baseline={baseline}"
        )
        assert stats["current_allocated_bytes"] == baseline_bytes


def test_gpu_generator_public_api_matches_oracle() -> None:
    """The registered (surya_ocr2, hip_gfx1151, fp32) generator path."""

    from hipengine.generation.registry import GenerationRequest
    from hipengine.generation.surya_contract import SuryaRequestError
    from hipengine.generation.surya_gpu import SuryaOCRGeneratorGPU

    ref_path = FIXTURES / "oracle_greedy.json"
    if not ref_path.exists():
        pytest.skip("oracle_greedy.json not present; capture with transformers")
    ref = json.loads(ref_path.read_text())
    page = FIXTURES / "page_small.png"
    if not page.exists():
        pytest.skip("page_small.png fixture not present")

    gen = SuryaOCRGeneratorGPU(model_path=_model_dir())
    try:
        request = GenerationRequest(
            prompts=["Transcribe this page."],
            max_tokens=64,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=False,
        )
        out = gen.generate_multimodal_detailed(
            "Transcribe this page.", str(page), request
        )
        assert out.text == ref["text"]
        assert out.generated_token_ids is not None
        assert out.finish_details is not None
        assert out.finish_details.reason in {"eos", "stop", "length"}

        # prompt + output must fit the runner context, checked before device work
        with pytest.raises(SuryaRequestError, match="exceeds"):
            gen.generate_multimodal_detailed(
                "Transcribe this page.",
                str(page),
                GenerationRequest(
                    prompts=["Transcribe this page."],
                    max_tokens=gen.runner.max_seq + 1,
                    temperature=0.0,
                    top_p=1.0,
                    ignore_eos=False,
                ),
            )

        # unsupported sampling controls are rejected, not silently ignored
        with pytest.raises(SuryaRequestError, match="temperature"):
            gen.generate_multimodal_detailed(
                "Transcribe this page.",
                str(page),
                GenerationRequest(
                    prompts=["Transcribe this page."],
                    max_tokens=8,
                    temperature=0.7,
                    top_p=1.0,
                    ignore_eos=False,
                ),
            )
    finally:
        gen.close()
