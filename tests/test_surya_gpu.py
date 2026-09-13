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


@pytest.mark.parametrize(
    ("size", "expected_grid"),
    [
        # both below SURYA_MIN_PIXELS: smart_resize rounds *up* to 256x256,
        # exercising the ceil branch rather than the plain factor rounding
        ((64, 64), (1, 16, 16)),
        ((100, 100), (1, 16, 16)),
        # non-square, both orientations: the same patch count through a 22x12
        # and a 12x22 grid, so a row/column mix-up cannot cancel out
        ((128, 256), (1, 22, 12)),
        ((256, 128), (1, 12, 22)),
    ],
)
def test_gpu_vision_geometry_sweep_matches_cpu_reference(
    runner, size: tuple[int, int], expected_grid: tuple[int, int, int]
) -> None:
    """Resize and grid-geometry branches of the vision tower.

    The rect bring-up exposed a tile-addressing bug that only appeared on a
    non-square grid, so the geometry axes are worth pinning explicitly. These
    four cases stay cheap enough for the suite; 3:1 aspect ratios
    (512x1536 / 1536x512 / 1024x192 / 192x1024) also pass but cost 12.8 s of
    CPU vision each.
    """

    r, weights, spec = runner
    from hipengine.kernels.cpu_reference.surya import vision_forward
    from hipengine.loading.surya import preprocess_image_surya
    from PIL import Image, ImageDraw

    width, height = size
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 8
    while y < height - 12:
        draw.rectangle([8, y, max(9, width - 8), y + 4], fill=(40, 40, 40))
        y += 14
    draw.rectangle([8, 8, max(9, width // 2), 20], fill=(10, 10, 10))

    pixel_rows, grid = preprocess_image_surya(img)
    assert tuple(grid) == expected_grid, (
        f"{width}x{height} resized to grid {tuple(grid)}, expected "
        f"{expected_grid}; the resize branch under test changed"
    )

    _, _, merged_cpu = vision_forward(weights, spec, pixel_rows, [grid])
    merged_gpu = r.vision_forward(pixel_rows, [grid])
    assert merged_gpu.shape == merged_cpu.shape
    d = np.abs(merged_gpu - merged_cpu)
    assert d.max() < 0.5, f"{width}x{height} vision diverged: max|d|={d.max():.3e}"
    assert d.mean() < 0.05, f"{width}x{height} vision drift: mean|d|={d.mean():.3e}"
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
            "run scripts/surya_oracle_greedy.py"
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
            "run scripts/surya_oracle_greedy.py"
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


def test_gpu_tiled_vision_matches_dense_and_oracle(runner) -> None:
    """Tiling by query rows must not change the full-image attention result.

    Independently cropping the page would change what each patch attends to;
    the tiled path instead keeps the full key range and only bounds how many
    query rows are live at once. The 4096-patch page under a 8 MiB budget needs
    many tiles, so this is a real partition, not a single dense tile in
    disguise. Checked against both the dense GPU result and the torch oracle.
    """

    r, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner

    path = FIXTURES / "oracle_fullpage.npz"
    if not path.exists():
        pytest.skip("oracle_fullpage.npz not present")
    with np.load(path) as z:
        if "vision_merged" not in z.files:
            pytest.skip("oracle_fullpage.npz has no vision_merged features")
        merged_oracle = z["vision_merged"]

    pixel_rows, grid = _full_page_inputs()
    n = grid[1] * grid[2]
    tiled = SuryaGpuRunner(weights, spec, max_vision_scratch_bytes=8 * 1024 * 1024)
    try:
        block = tiled.vision_block([grid])
        assert 1 <= block < n, f"expected a real partition, got block={block} of {n}"
        merged_tiled = tiled.vision_forward(pixel_rows, [grid])
    finally:
        tiled.close()

    merged_dense = r.vision_forward(pixel_rows, [grid])
    assert np.isfinite(merged_tiled).all()

    d_tiled_oracle = np.abs(merged_tiled - merged_oracle)
    assert d_tiled_oracle.max() < 2e-3, (
        f"tiled vision diverged from the oracle: max|d|={d_tiled_oracle.max():.3e}"
    )
    assert d_tiled_oracle.mean() < 1e-5, (
        f"tiled vision drifted from the oracle: mean|d|={d_tiled_oracle.mean():.3e}"
    )

    d_tiled_dense = np.abs(merged_tiled - merged_dense)
    assert d_tiled_dense.max() < 1e-4, (
        "tiling changed the full-image attention result: "
        f"max|tiled-dense|={d_tiled_dense.max():.3e}"
    )


def test_gpu_tiled_vision_matches_oracle_at_every_geometry(runner) -> None:
    """The tiled path must hold on the non-square grids too.

    The tile boundaries fall at query-row multiples that do not line up with
    the 22x12 / 12x22 geometry, so a row/column mix-up in the block offset
    would show up here and not on a square page.
    """

    _, weights, spec = runner
    from hipengine.kernels.cpu_reference.surya import vision_forward
    from hipengine.loading.surya import preprocess_image_surya
    from hipengine.runtime.surya import SuryaGpuRunner
    from PIL import Image, ImageDraw

    for size in ((128, 256), (256, 128)):
        width, height = size
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        y = 8
        while y < height - 12:
            draw.rectangle([8, y, max(9, width - 8), y + 4], fill=(40, 40, 40))
            y += 14
        pixel_rows, grid = preprocess_image_surya(img)
        n = grid[1] * grid[2]

        _, _, merged_cpu = vision_forward(weights, spec, pixel_rows, [grid])
        tiled = SuryaGpuRunner(
            weights, spec, max_vision_scratch_bytes=2 * 1024 * 1024
        )
        try:
            block = tiled.vision_block([grid])
            assert 1 <= block < n, (size, block, n)
            merged_tiled = tiled.vision_forward(pixel_rows, [grid])
        finally:
            tiled.close()

        d = np.abs(merged_tiled - merged_cpu)
        assert d.max() < 0.5, f"{size} tiled vision diverged: max|d|={d.max():.3e}"
        assert d.mean() < 0.05, f"{size} tiled vision drift: mean|d|={d.mean():.3e}"
        assert np.isfinite(merged_tiled).all()


@pytest.mark.parametrize("page_name", ["columns", "list"])
def test_gpu_ocr_corpus_matches_oracle(runner, page_name: str) -> None:
    """Held-out layouts (two-column, numbered list) vs the torch oracle.

    Both differ structurally from the single-column fixtures, so the GPU lane
    cannot pass them by generalizing from the pages already covered: the
    columns page needs the two prose columns kept apart, and the list page
    needs the ``List-Group`` label, which no other fixture reaches.
    """

    r, _weights, spec = runner
    from hipengine.loading.surya import SuryaTokenizer, preprocess_image_surya
    from PIL import Image

    oracle_path = FIXTURES / "oracle_corpus.json"
    if not oracle_path.exists():
        pytest.skip(
            "oracle_corpus.json not present; run "
            "scripts/surya_oracle_greedy.py --case corpus"
        )
    ref = json.loads(oracle_path.read_text())[page_name]
    page = FIXTURES / f"page_{page_name}.png"
    if not page.exists():
        pytest.skip(f"page_{page_name}.png fixture not present")

    tokenizer = SuryaTokenizer(_model_dir())
    pixel_rows, grid = preprocess_image_surya(Image.open(page).convert("RGB"))
    generated = _gpu_ocr_ids(
        r, spec, tokenizer, pixel_rows, grid, len(ref["ids"]) + 8
    )

    if generated != ref["ids"]:
        n = min(len(generated), len(ref["ids"]))
        first = next((i for i in range(n) if generated[i] != ref["ids"][i]), n)
        raise AssertionError(
            f"GPU greedy decode of page_{page_name}.png diverged from the "
            f"torch fp32 reference at index {first} "
            f"(gpu len={len(generated)}, ref len={len(ref['ids'])})"
        )

    # the oracle decodes a rich layout; a near-constant run would make the
    # parity assertion above vacuous
    assert len(set(generated)) >= 10, (
        f"page_{page_name}.png decoded only {len(set(generated))} distinct ids"
    )
    assert json.loads(tokenizer.decode(generated)), "output is not valid JSON"


def test_gpu_single_token_gemm_uses_sgemv_and_matches_sgemm(runner) -> None:
    """Decode is one row: route it to SGEMV, and SGEMV must equal SGEMM.

    rocBLAS SGEMM is tuned for a wide ``n``; with one row it reaches roughly a
    third of the memory bandwidth SGEMV does, and decode is 187 single-row
    GEMMs per token, so the dispatch is worth ~1.7x end to end. A wrong
    transposition here returns wrong numbers silently rather than erroring,
    which is why the routing and the numerics are both pinned.
    """

    r, _weights, _spec = runner
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    weight = r._w["model.language_model.layers.0.mlp.gate_proj.weight"]
    in_features = 1152
    out_features = weight.nbytes // 4 // in_features
    x = r._buf("probe_x", in_features * 4)
    out = r._buf("probe_out", out_features * 4)
    # sized for the rows=3 dispatch probe below, which writes three rows
    ref = r._buf("probe_ref", 3 * out_features * 4)
    r._upload(x, np.random.default_rng(0).standard_normal(in_features).astype(np.float32) * 0.05)

    seen: list[str] = []
    real_gemv = r.rocblas.sgemv_rowmajor_nt
    real_gemm = r.rocblas.sgemm_rowmajor_nt
    r.rocblas.sgemv_rowmajor_nt = lambda *a, **kw: (seen.append("gemv"), real_gemv(*a, **kw))[1]
    r.rocblas.sgemm_rowmajor_nt = lambda *a, **kw: (seen.append("gemm"), real_gemm(*a, **kw))[1]
    try:
        r._gemm(x.ptr, weight.ptr, out.ptr, 1, in_features, out_features)
        r._gemm(x.ptr, weight.ptr, ref.ptr, 3, in_features, out_features)
    finally:
        r.rocblas.sgemv_rowmajor_nt = real_gemv
        r.rocblas.sgemm_rowmajor_nt = real_gemm
    assert seen == ["gemv", "gemm"], f"rows==1 must route to SGEMV, got {seen}"

    # the GEMV result must equal a one-row SGEMM (fp32 summation order differs)
    real_gemm(x.ptr, weight.ptr, ref.ptr, rows=1, in_features=in_features, out_features=out_features)
    r.runtime.device_synchronize()
    a = np.empty(out_features, dtype=np.float32)
    b = np.empty(out_features, dtype=np.float32)
    copy_device_to_host(host_array_ptr(a), out, a.nbytes)
    copy_device_to_host(host_array_ptr(b), ref, b.nbytes)
    r.runtime.device_synchronize()
    d = np.abs(a - b)
    assert d.max() <= 1e-5 * max(1.0, float(np.abs(a).max())), (
        f"SGEMV disagrees with SGEMM: max|d|={d.max():.3e}"
    )
    assert int(np.argmax(a)) == int(np.argmax(b))


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

    Regression guard for the constructor clobbering the permanent-allocation
    list (now ``_permanent_bufs``) after persistent state/staging allocations
    populated it, and for the untracked cached pointer-array buffers.
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


# -- text-prefill causal score tiling ---------------------------------------


def test_gpu_causal_mask_honors_the_query_offset(runner) -> None:
    """A score tile's mask must use its first query's absolute position.

    The causal condition is ``j <= query_offset + i``. Without the offset every
    tile would behave as if it started at query 0, so a later tile would attend
    to keys it must not see — and the result would still be finite and
    plausible, which is why this is pinned on the kernel directly and not only
    through the end-to-end logits.
    """

    r, _, _ = runner
    from hipengine.core.memory import copy_device_to_host, host_array_ptr, malloc
    from hipengine.core.memory import free as hip_free
    from hipengine.kernels.hip_gfx1100.surya.surya_ops import (
        surya_causal_mask_scale_f32,
    )

    heads, tokens, scale = 2, 12, 0.5
    rng = np.random.default_rng(20260912)
    full = rng.standard_normal((heads, tokens, tokens)).astype(np.float32)
    keys = np.arange(tokens)
    queries = np.arange(tokens)
    # the dense reference: entry (h, i, j) keeps j <= i
    dense = np.where(
        (keys[None, :] <= queries[:, None])[None, :, :], full * scale, -np.inf
    )

    buf = malloc(full.nbytes)
    try:
        # start 0 is the dense case; 7 and 11 fall off the tile grid so an
        # off-by-one or a tile-relative mask cannot pass by luck
        for start, bq in ((0, 5), (7, 5), (tokens - 1, 1)):
            tile = np.ascontiguousarray(full[:, start:start + bq, :])
            r._upload(buf, tile)
            surya_causal_mask_scale_f32(
                buf.ptr, scale, heads, bq, tokens, start,
                library=r.surya_library, runtime=r.runtime,
            )
            r.runtime.device_synchronize()
            out = np.empty((heads, bq, tokens), dtype=np.float32)
            copy_device_to_host(host_array_ptr(out), buf, nbytes=out.nbytes)
            want = dense[:, start:start + bq, :]
            finite = np.isfinite(want)
            assert np.array_equal(np.isfinite(out), finite), (start, bq)
            np.testing.assert_array_equal(out[finite], want[finite])
            assert (out[~finite] == -np.inf).all(), (start, bq)
    finally:
        hip_free(buf)


def test_gpu_tiled_prefill_matches_dense_and_cpu_reference(runner) -> None:
    """Tiling the causal prefill must not change the logits or the greedy ids.

    Each tile still attends over the full key range, so the softmax rows are
    the rows the dense path computed and the result is the dense result. A
    64 KiB budget over a page-scale prompt is a real partition (dozens of
    tiles, partial tile included), not a single dense tile in disguise.
    Checked bit-for-bit against the dense GPU path and independently against
    the NumPy CPU reference.
    """

    r, weights, spec = runner
    from hipengine.kernels.cpu_reference.surya import text_prefill
    from hipengine.loading.surya import (
        SuryaTokenizer,
        compute_mrope_positions,
        render_chat_prompt,
        resolve_surya_path,
    )
    from hipengine.runtime.surya import SuryaGpuRunner

    pixel_rows, grid = _page_inputs()
    tokenizer = SuryaTokenizer(resolve_surya_path(MODEL_ID))
    merged = r.vision_forward(pixel_rows, [grid])
    n_img = (grid[1] // 2) * (grid[2] // 2)
    ids, mm = render_chat_prompt(tokenizer, "Transcribe this page.", n_img)
    ids = np.asarray(ids, dtype=np.int64)
    pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)

    dense_logits = r.prefill(ids, pos, visual_features=merged)

    tiled = SuryaGpuRunner(
        weights, spec, max_seq=r.max_seq, max_prefill_scratch_bytes=64 * 1024
    )
    try:
        block = tiled.prefill_block(len(ids))
        assert 1 <= block < len(ids), (
            f"expected a real partition, got block={block} of {len(ids)}"
        )
        tiled_logits = tiled.prefill(ids, pos, visual_features=merged)
    finally:
        tiled.close()

    assert np.array_equal(dense_logits, tiled_logits), (
        "text-prefill tiling changed the logits: max|"
        f"dense-tiled|={np.abs(dense_logits - tiled_logits).max():.3e}"
    )

    # independent oracle: the NumPy CPU reference on the same features
    hidden, _state = text_prefill(
        weights, spec, ids[None, :], pos, visual_features=merged[None]
    )
    emb = weights["model.language_model.embed_tokens.weight"]
    cpu_logits = (hidden[:, -1] @ emb.T)[0]

    def _prob(x):
        x = np.asarray(x, dtype=np.float64)
        e = np.exp(x - x.max())
        return e / e.sum()

    p, q = _prob(tiled_logits), _prob(cpu_logits)
    kl = float((q * (np.log(q + 1e-300) - np.log(p + 1e-300))).sum())
    assert kl < 1e-4, f"tiled prefill KL vs CPU reference = {kl:.3e}"
    assert int(np.argmax(tiled_logits)) == int(np.argmax(cpu_logits)), (
        "tiled prefill argmax diverged from the CPU reference"
    )


def test_gpu_tiled_prefill_is_bit_identical_at_the_default_budget(runner) -> None:
    """The default budget's page-scale split must not perturb the logits.

    At the 8580 image tokens of a 300-DPI A4 page the default budget splits the
    prefill into 32 tiles of 269 query rows, which is the regime the production
    numerical gate runs in — so this pins the claim that the gate's recorded
    numbers still describe the default path. The tile shape is chosen by
    :func:`plan_score_tiles`, so this also pins that the shape envelope does not
    perturb the arithmetic.
    """

    _, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner

    tokens = 8580
    ids = (np.arange(1, tokens + 1, dtype=np.int64) % 1000) + 1
    axis = np.arange(tokens, dtype=np.int64)
    pos = np.stack([axis, axis, axis])

    dense = SuryaGpuRunner(
        weights, spec, max_seq=16384, max_prefill_scratch_bytes=None
    )
    try:
        dense_logits = dense.prefill(ids, pos)
    finally:
        dense.close()

    tiled = SuryaGpuRunner(weights, spec, max_seq=16384)
    try:
        block = tiled.prefill_block(tokens)
        assert 1 <= block < tokens, (
            f"the default budget must split this prompt, got block={block}"
        )
        tiled_logits = tiled.prefill(ids, pos)
    finally:
        tiled.close()

    assert np.array_equal(dense_logits, tiled_logits), (
        f"the default budget changed the {tokens}-token prefill logits: max|"
        f"dense-tiled|={np.abs(dense_logits - tiled_logits).max():.3e}"
    )


def test_gpu_prefill_scratch_admission_matches_the_allocation(runner) -> None:
    """The admitted prefill tile must match what the score buffer allocates.

    They are separate computations, so drift between them would let a prompt
    through admission and then allocate more than was admitted.
    """

    _, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner, plan_score_tiles

    r = SuryaGpuRunner(weights, spec)
    try:
        for tokens in (64, 257, 1024):
            block, scratch = plan_score_tiles(
                tokens, spec.num_attention_heads, r.max_prefill_scratch_bytes
            )
            assert r.prefill_block(tokens) == block
            admitted = r.prefill_scratch_bytes(tokens)
            tile = r._text_tile_elements(tokens, block)
            assert tile == tokens * block
            assert tile * spec.num_attention_heads * 4 == admitted == scratch
            buf = r._text_scores(tokens, block)
            assert buf.nbytes >= admitted, (
                f"{tokens} tokens admitted {admitted} bytes but allocated "
                f"{buf.nbytes}"
            )
    finally:
        r.close()


def test_gpu_prefill_scratch_admission_matches_the_allocation_when_tiled(
    runner,
) -> None:
    """A prompt that needs many tiles must still be admitted exactly."""

    _, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner

    # 1024 tokens need 32 MiB dense; a 64 KiB budget forces a few query rows
    # per tile, i.e. many tiles, and the admitted size must follow.
    r = SuryaGpuRunner(weights, spec, max_prefill_scratch_bytes=64 * 1024)
    try:
        tokens = 1024
        block = r.prefill_block(tokens)
        assert 1 <= block < tokens
        admitted = r.prefill_scratch_bytes(tokens)
        assert admitted <= 64 * 1024
        assert admitted < spec.num_attention_heads * tokens * tokens * 4
        tile = r._text_tile_elements(tokens, block)
        assert tile * spec.num_attention_heads * 4 == admitted
        assert r._text_scores(tokens, block).nbytes >= admitted
    finally:
        r.close()


def test_gpu_prefill_admits_the_default_context_without_allocating(runner) -> None:
    """The 16384-token default context must pass admission, allocating nothing.

    Dense scores needed 8.59 GB there; the default budget bounds the live tile
    at 512 MiB, so admission is a host-side computation.
    """

    _, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner

    r = SuryaGpuRunner(weights, spec)
    try:
        assert spec.num_attention_heads * 16384 * 16384 * 4 == 8_589_934_592
        need = r.prefill_scratch_bytes(16384)
        assert need <= r.max_prefill_scratch_bytes
        r.check_prefill_capacity(16384)
        assert "scores" not in r._scratch, "admission must not allocate"
    finally:
        r.close()


def test_gpu_prefill_admission_rejects_a_prompt_below_one_query_row(runner) -> None:
    """A budget that cannot hold even one query row is a real rejection."""

    _, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner, SuryaGpuRuntimeError

    tokens = 1024
    floor = spec.num_attention_heads * tokens * 4
    r = SuryaGpuRunner(weights, spec, max_prefill_scratch_bytes=floor - 1)
    try:
        with pytest.raises(SuryaGpuRuntimeError, match="above the .* budget"):
            r.check_prefill_capacity(tokens)
        # and the rejected check allocated nothing
        assert "scores" not in r._scratch
    finally:
        r.close()


def test_gpu_prefill_admission_can_be_disabled(runner) -> None:
    """``max_prefill_scratch_bytes=None`` runs one dense tile."""

    _, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner

    r = SuryaGpuRunner(weights, spec, max_prefill_scratch_bytes=None)
    try:
        assert r.prefill_block(2048) == 2048
        assert r.prefill_scratch_bytes(2048) == spec.num_attention_heads * 2048 * 2048 * 4
        r.check_prefill_capacity(2048)
    finally:
        r.close()


def test_gpu_prefill_rejects_an_over_budget_prompt_before_any_device_work(
    runner,
) -> None:
    """A prompt whose tile cannot fit must be rejected before embed and rope.

    ``prefill`` admitted nothing until it was already inside the layer stack,
    so an over-budget prompt still paid for the embedding lookup and the rope
    tables before failing.
    """

    _, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner, SuryaGpuRuntimeError

    tokens = 1024
    r = SuryaGpuRunner(
        weights, spec, max_prefill_scratch_bytes=spec.num_attention_heads * tokens * 4 - 1
    )
    try:
        ids = (np.arange(1, tokens + 1, dtype=np.int64) % 1000) + 1
        axis = np.arange(tokens, dtype=np.int64)
        pos = np.stack([axis, axis, axis])
        with pytest.raises(SuryaGpuRuntimeError, match="above the .* budget"):
            r.prefill(ids, pos)
        assert "scores" not in r._scratch
        assert r._visual_buf is None
    finally:
        r.close()


def test_gpu_vision_scratch_admission_matches_the_allocation(runner) -> None:
    """The admitted size must match what the score tile actually allocates.

    They are separate computations, so drift between them would let a grid
    through admission and then allocate more than was admitted.
    """

    _, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner, plan_vision_attention

    r = SuryaGpuRunner(weights, spec)
    try:
        for grid in ((1, 16, 16), (1, 22, 12), (1, 32, 32)):
            n = grid[1] * grid[2]
            block, scratch = plan_vision_attention(
                n, spec.vision_num_heads, r.max_vision_scratch_bytes
            )
            buf = r._vis_scores(n, spec.vision_num_heads, block)
            admitted = r.vision_scratch_bytes([grid])
            tile = r._vis_tile_elements(n, block)
            assert tile == n * block
            assert tile * spec.vision_num_heads * 4 == admitted == scratch
            assert buf.nbytes >= admitted, (
                f"grid {grid} admitted {admitted} bytes but allocated "
                f"{buf.nbytes}"
            )
    finally:
        r.close()


def test_gpu_vision_scratch_admission_matches_the_allocation_when_tiled(
    runner,
) -> None:
    """A grid that needs several query tiles must still be admitted exactly."""

    _, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner

    # 4096 patches need 805 MB dense; an 8 MiB budget forces ~40 query rows
    # per tile, i.e. many tiles, and the admitted size must follow.
    r = SuryaGpuRunner(weights, spec, max_vision_scratch_bytes=8 * 1024 * 1024)
    try:
        grid = (1, 64, 64)
        n = 64 * 64
        admitted = r.vision_scratch_bytes([grid])
        assert admitted <= 8 * 1024 * 1024
        assert admitted < spec.vision_num_heads * n * n * 4
        buf = r._vis_scores(n, spec.vision_num_heads, r.vision_block([grid]))
        tile = r._vis_tile_elements(n, r.vision_block([grid]))
        assert tile * spec.vision_num_heads * 4 == admitted
        assert buf.nbytes >= admitted
    finally:
        r.close()


def test_gpu_vision_admission_bounds_the_page_by_memory_and_time(runner) -> None:
    """Both page-scale grids pass the byte budget; the ceiling fails on time.

    Dense scores needed 56.5 GB at 34320 patches (300-DPI A4) and 206 GB at
    65536 (``SURYA_MAX_PIXELS``). Tiling brings both inside the default byte
    budget without allocating anything, but 65536 patches is 1.7e14 FLOPs of
    bidirectional attention -- about 2.7 minutes on gfx1151 -- so the time
    budget rejects it, and the A4 page the benchmark suite measures stays
    admitted.
    """

    _, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner, SuryaGpuRuntimeError

    r = SuryaGpuRunner(weights, spec)
    try:
        for grid in ((1, 220, 156), (1, 256, 256)):
            need = r.vision_scratch_bytes([grid])
            assert need <= r.max_vision_scratch_bytes, (grid, need)
        r.check_vision_capacity([(1, 220, 156)])
        with pytest.raises(SuryaGpuRuntimeError, match="vision time budget"):
            r.check_vision_capacity([(1, 256, 256)])
        assert "vis_scores" not in r._scratch, "admission must not allocate"
        # the declared budget, not a hardware limit: raising or dropping it
        # admits the same grid on the same device
        for budget in (200.0, None):
            raised = SuryaGpuRunner(weights, spec, max_vision_seconds=budget)
            try:
                raised.check_vision_capacity([(1, 256, 256)])
            finally:
                raised.close()
    finally:
        r.close()


def test_gpu_vision_admission_rejects_a_grid_below_one_query_row(runner) -> None:
    """A budget that cannot hold even one query row is a real rejection."""

    _, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner, SuryaGpuRuntimeError

    n = 64 * 64
    floor = spec.vision_num_heads * n * 4
    r = SuryaGpuRunner(weights, spec, max_vision_scratch_bytes=floor - 1)
    try:
        with pytest.raises(SuryaGpuRuntimeError, match="above the .* budget"):
            r.check_vision_capacity([(1, 64, 64)])
        # and the rejected check allocated nothing
        assert "vis_scores" not in r._scratch
    finally:
        r.close()


def test_gpu_vision_admission_can_be_disabled(runner) -> None:
    """``max_vision_scratch_bytes=None`` leaves only the memory check."""

    _, weights, spec = runner
    from hipengine.runtime.surya import SuryaGpuRunner

    r = SuryaGpuRunner(weights, spec, max_vision_scratch_bytes=None)
    try:
        # far past the default budget, but well inside free device memory here
        r.check_vision_capacity([(1, 64, 64)])
    finally:
        r.close()


def test_gpu_generator_checks_capacity_before_vision() -> None:
    """Over-capacity OCR requests must not pay for vision first.

    Regression: ``_generate_ocr`` ran ``vision_forward`` and only then reached
    ``check_prompt_capacity`` inside ``_decode_greedy``, so a request that could
    never fit still ran the whole vision tower and could allocate multi-GB
    scratch before being rejected.
    """

    from hipengine.generation.registry import GenerationRequest
    from hipengine.generation.surya_contract import SuryaRequestError
    from hipengine.generation.surya_gpu import SuryaOCRGeneratorGPU

    page = FIXTURES / "page_small.png"
    if not page.exists():
        pytest.skip("page_small.png fixture not present")

    gen = SuryaOCRGeneratorGPU(model_path=_model_dir())
    ran: list[str] = []
    real_vision = gen.runner.vision_forward
    gen.runner.vision_forward = lambda *a, **kw: (ran.append("vision"), real_vision(*a, **kw))[1]
    try:
        # prompt + max_tokens cannot fit the runner context
        with pytest.raises(SuryaRequestError, match="exceeds"):
            gen.generate_multimodal_detailed(
                "Transcribe this page.",
                str(page),
                GenerationRequest(
                    prompts=["Transcribe this page."],
                    max_tokens=gen.max_seq + 1,
                    temperature=0.0,
                    top_p=1.0,
                    ignore_eos=False,
                ),
            )
        assert ran == [], f"vision ran before the capacity rejection: {ran}"

        # a fitting request does run vision, so the spy is not vacuously empty
        out = gen.generate_multimodal_detailed(
            "Transcribe this page.",
            str(page),
            GenerationRequest(
                prompts=["Transcribe this page."],
                max_tokens=4,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=False,
            ),
        )
        assert ran == ["vision"]
        assert out.text
        # the advertised budget is the runner's, so callers can admit up front
        assert gen.max_seq == gen.runner.max_seq
        assert gen.max_vision_scratch_bytes == gen.runner.max_vision_scratch_bytes
        assert gen.max_vision_seconds == gen.runner.max_vision_seconds
        assert gen.max_prefill_scratch_bytes == gen.runner.max_prefill_scratch_bytes
    finally:
        gen.runner.vision_forward = real_vision
        gen.close()


def test_gpu_generator_rejects_an_over_time_page_before_vision() -> None:
    """A page whose vision forward cannot fit the budget never starts vision.

    The 1024x1024 fixture page is 4096 patches and estimates at ~1.0 s, so a
    0.25 s budget rejects it -- after preprocessing, which is host-side and
    cheap, but before any device work or scratch allocation. The byte budget
    alone would have admitted it: 805 MB dense, ~17 MB tiled.
    """

    from hipengine.generation.registry import GenerationRequest
    from hipengine.generation.surya_gpu import SuryaOCRGeneratorGPU
    from hipengine.runtime.surya import SuryaGpuRuntimeError

    page = FIXTURES / "page_full.png"
    if not page.exists():
        pytest.skip("page_full.png fixture not present")

    gen = SuryaOCRGeneratorGPU(model_path=_model_dir(), max_vision_seconds=0.25)
    ran: list[str] = []
    real_vision = gen.runner.vision_forward
    gen.runner.vision_forward = lambda *a, **kw: (ran.append("vision"), real_vision(*a, **kw))[1]
    try:
        assert gen.max_vision_seconds == 0.25
        estimate = gen.runner.vision_time_seconds([(1, 64, 64)])
        assert estimate > 0.25, estimate
        with pytest.raises(SuryaGpuRuntimeError, match="vision time budget"):
            gen.generate_multimodal_detailed(
                "Transcribe this page.",
                str(page),
                GenerationRequest(
                    prompts=["Transcribe this page."],
                    max_tokens=4,
                    temperature=0.0,
                    top_p=1.0,
                    ignore_eos=False,
                ),
            )
        assert ran == [], f"vision ran before the time-budget rejection: {ran}"
    finally:
        gen.runner.vision_forward = real_vision
        gen.close()


def test_gpu_generator_rejects_a_page_that_cannot_finish_before_the_deadline() -> None:
    """A vision forward that outlasts the request deadline is not started.

    The deadline is otherwise only observed at phase boundaries, so a
    page-scale request would run to completion after the client had already
    given up. Phase 1 is the real estimate on the A4 page against a 5 s
    deadline (43.4 s estimated); phase 2 pins the estimate to isolate the new
    check from the pre-existing expiry checks, which raise the same error.
    """

    import dataclasses
    import time

    from hipengine.generation.deadline import GenerationDeadlineExceeded
    from hipengine.generation.registry import GenerationRequest
    from hipengine.generation.surya_gpu import SuryaOCRGeneratorGPU

    page = FIXTURES / "page_a4.png"
    if not page.exists():
        pytest.skip("page_a4.png fixture not present")

    gen = SuryaOCRGeneratorGPU(model_path=_model_dir())
    ran: list[str] = []
    real_vision = gen.runner.vision_forward
    gen.runner.vision_forward = lambda *a, **kw: (ran.append("vision"), real_vision(*a, **kw))[1]
    real_time = gen.runner.vision_time_seconds
    try:
        request = GenerationRequest(
            prompts=["Transcribe this page."],
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=False,
            deadline_at=time.perf_counter() + 5.0,
        )
        with pytest.raises(GenerationDeadlineExceeded):
            gen.generate_multimodal_detailed(
                "Transcribe this page.", str(page), request
            )
        assert ran == [], f"vision ran past the deadline: {ran}"

        gen.runner.vision_time_seconds = lambda grid: 1e6
        isolated = dataclasses.replace(
            request, deadline_at=time.perf_counter() + 30.0
        )
        with pytest.raises(GenerationDeadlineExceeded):
            gen.generate_multimodal_detailed(
                "Transcribe this page.", str(page), isolated
            )
        assert ran == [], f"vision ran past the deadline: {ran}"
    finally:
        gen.runner.vision_forward = real_vision
        gen.runner.vision_time_seconds = real_time
        gen.close()


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


def test_gpu_generator_multi_prompt_isolation() -> None:
    """``generate`` must be invariant to how many prompts share a request.

    ``generate`` loops ``request.prompts`` on one runner, so the prompts run
    back-to-back over shared KV slots and recurrent state. A leak between them
    shows up as a result that depends on position within the request, which is
    what this checks. Nothing else in the suite passes more than one prompt.

    Liveness: with the ``decode_step`` off-by-one from 91bd23576 reintroduced,
    the long-then-short ordering below fails this test; the single-prompt and
    multimodal->multimodal tests in the suite do not.
    """

    from hipengine.generation.registry import GenerationRequest
    from hipengine.generation.surya_gpu import SuryaOCRGeneratorGPU

    # The first prompt must be much longer than the others. A leak that reads
    # KV past the current request's own slots only corrupts a *short* prompt
    # that follows a longer one; three similarly sized prompts did not expose
    # the decode_step off-by-one fixed in 91bd23576.
    long_prompt = (
        "Transcribe every word on this page exactly as it appears, preserving "
        "all line breaks, headings and table cells. "
    ) * 4
    prompts = [long_prompt, "Read.", "Transcribe this page."]

    def request(prompt_list: list[str]) -> GenerationRequest:
        return GenerationRequest(
            prompts=prompt_list,
            max_tokens=16,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=False,
        )

    gen = SuryaOCRGeneratorGPU(model_path=_model_dir())
    try:
        # a lone prompt, then the same prompt as the head of a longer request
        alone = gen.generate(request([prompts[0]]))
        multi = gen.generate(request(prompts))
        assert len(multi) == len(prompts)
        assert multi[0] == alone[0], (
            "first prompt changed when later prompts joined the request"
        )

        # reversing the request must reverse the outputs, not change them
        rev = gen.generate(request(list(reversed(prompts))))
        assert rev == list(reversed(multi)), (
            "generate() is order-dependent: a prompt's output depends on which "
            "prompts ran before it in the same request"
        )
    finally:
        gen.close()


def test_gpu_generator_forwards_context_and_scratch_budgets() -> None:
    """The registered factory must honor the budgets it advertises.

    ``LLM._factory_capacity_kwargs`` forwards a limit only to a factory that
    declares the parameter by name. ``make_surya_generator_gpu`` used to accept
    ``**_kwargs`` and drop them, so ``LLM(max_sequence_length=...)`` never
    reached the runner. The same applies to the two score-tile budgets and the
    vision time budget: a budget the factory does not declare is silently
    dropped and the runner keeps its default.
    """

    from hipengine.generation.surya_gpu import (
        SuryaOCRGeneratorGPU,
        make_surya_generator_gpu,
    )
    from hipengine.llm import _factory_capacity_kwargs

    forwarded = _factory_capacity_kwargs(
        make_surya_generator_gpu,
        max_sequence_length=16384,
        resident_capacity=None,
        vision_max_scratch_bytes=64 * 1024 * 1024,
        prefill_max_scratch_bytes=32 * 1024 * 1024,
        vision_max_seconds=45.0,
    )
    assert forwarded == {
        "max_sequence_length": 16384,
        "vision_max_scratch_bytes": 64 * 1024 * 1024,
        "prefill_max_scratch_bytes": 32 * 1024 * 1024,
        "vision_max_seconds": 45.0,
    }

    gen = make_surya_generator_gpu(
        model_path=_model_dir(),
        max_sequence_length=16384,
        vision_max_scratch_bytes=64 * 1024 * 1024,
        prefill_max_scratch_bytes=32 * 1024 * 1024,
        vision_max_seconds=45.0,
    )
    try:
        assert gen.max_seq == 16384 == gen.runner.max_seq
        assert gen.max_vision_scratch_bytes == 64 * 1024 * 1024
        assert gen.runner.max_vision_scratch_bytes == 64 * 1024 * 1024
        assert gen.max_prefill_scratch_bytes == 32 * 1024 * 1024
        assert gen.runner.max_prefill_scratch_bytes == 32 * 1024 * 1024
        assert gen.max_vision_seconds == 45.0 == gen.runner.max_vision_seconds
    finally:
        gen.close()

    # a non-positive context budget is a configuration error, not a silent
    # fall back to the default
    with pytest.raises(ValueError, match="max_seq"):
        SuryaOCRGeneratorGPU(model_path=_model_dir(), max_seq=0)
    with pytest.raises(ValueError, match="max_prefill_scratch_bytes"):
        SuryaOCRGeneratorGPU(model_path=_model_dir(), max_prefill_scratch_bytes=0)
    with pytest.raises(ValueError, match="max_vision_seconds"):
        SuryaOCRGeneratorGPU(model_path=_model_dir(), max_vision_seconds=0)


def test_gpu_generator_default_context_admits_a_300dpi_a4_page() -> None:
    """The default context must admit a routine page, not just a small crop.

    A 300-DPI A4 page resizes to a 220x156 grid: 8580 image tokens before the
    prompt and any output. The old 2048-token default rejected every real
    document; upstream Surya budgets 12,288 context tokens per OCR slot.
    """

    from hipengine.generation.surya_contract import check_prompt_capacity
    from hipengine.generation.surya_gpu import SuryaOCRGeneratorGPU

    gen = SuryaOCRGeneratorGPU(model_path=_model_dir())
    try:
        image_tokens = (220 // 2) * (156 // 2)
        assert image_tokens == 8580
        # the prompt and a full-page output budget must both fit
        check_prompt_capacity(image_tokens + 64, 4096, gen.max_seq)
        assert gen.max_seq >= 12288
        # and the page's vision grid must pass the default memory budget
        gen.runner.check_vision_capacity([(1, 220, 156)])
    finally:
        gen.close()


def test_gpu_generator_rejection_names_the_required_context() -> None:
    """An over-context page must fail before vision and say what to raise."""

    from hipengine.generation.registry import GenerationRequest
    from hipengine.generation.surya_contract import SuryaRequestError
    from hipengine.generation.surya_gpu import SuryaOCRGeneratorGPU

    page = FIXTURES / "page_small.png"
    if not page.exists():
        pytest.skip("page_small.png fixture not present")

    gen = SuryaOCRGeneratorGPU(model_path=_model_dir(), max_seq=2048)
    ran: list[str] = []
    real_vision = gen.runner.vision_forward
    gen.runner.vision_forward = lambda *a, **kw: (
        ran.append("vision"),
        real_vision(*a, **kw),
    )[1]
    try:
        with pytest.raises(SuryaRequestError, match="max_sequence_length"):
            gen.generate_multimodal_detailed(
                "Transcribe this page.",
                str(page),
                GenerationRequest(
                    prompts=["Transcribe this page."],
                    max_tokens=4096,
                    temperature=0.0,
                    top_p=1.0,
                    ignore_eos=False,
                ),
            )
        assert ran == [], f"vision ran before the capacity rejection: {ran}"
    finally:
        gen.runner.vision_forward = real_vision
        gen.close()


def test_gpu_generator_recovers_after_an_abandoned_request() -> None:
    """An aborted request must not contaminate the next one on the runner.

    The decode loop raises between steps, so an abandoned request leaves the
    runner holding a partially written KV cache and advanced recurrent state —
    and, because it was abandoned mid-prefill of a *long* prompt, more written
    slots than the next request will ever attend over. `prefill` zeroes the
    conv and GDN state and rewrites KV from slot 0, so the follow-up request
    must be bit-identical to the same request on a runner that never saw the
    abandoned one. The long-then-short ordering is the leak-prone direction.
    """

    from hipengine.generation.deadline import (
        GenerationCancelled,
        GenerationCancellationToken,
    )
    from hipengine.generation.registry import GenerationRequest
    from hipengine.generation.surya_contract import SuryaRequestError
    from hipengine.generation.surya_gpu import SuryaOCRGeneratorGPU
    from hipengine.generation.surya_protocol import FULL_PAGE_HTML_PROMPT

    for name in ("page_table.png", "page_dense.png"):
        if not (FIXTURES / name).exists():
            pytest.skip(f"{name} fixture not present")

    def request(max_tokens: int, token: object = None) -> GenerationRequest:
        return GenerationRequest(
            prompts=[FULL_PAGE_HTML_PROMPT],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=False,
            cancellation_token=token,
        )

    gen = SuryaOCRGeneratorGPU(model_path=_model_dir())
    try:
        # 1. the reference: the short page on a runner that has only seen it
        reference = gen.generate_multimodal_detailed(
            FULL_PAGE_HTML_PROMPT, str(FIXTURES / "page_table.png"), request(512)
        )

        # 2. abandon a much longer page eight decode steps in
        token = GenerationCancellationToken()
        real_step = gen.runner.decode_step
        steps = {"n": 0}

        def cancel_at_eight(token_id: int, position: int) -> np.ndarray:
            steps["n"] += 1
            if steps["n"] == 8:
                token.cancel()
            return real_step(token_id, position)

        gen.runner.decode_step = cancel_at_eight
        try:
            with pytest.raises(GenerationCancelled):
                gen.generate_multimodal_detailed(
                    FULL_PAGE_HTML_PROMPT,
                    str(FIXTURES / "page_dense.png"),
                    request(2048, token),
                )
        finally:
            gen.runner.decode_step = real_step
        assert steps["n"] == 8, "the abandoned request did not reach eight steps"

        # 3. the same short page must reproduce the reference exactly
        after_cancel = gen.generate_multimodal_detailed(
            FULL_PAGE_HTML_PROMPT, str(FIXTURES / "page_table.png"), request(512)
        )
        assert after_cancel.generated_token_ids == reference.generated_token_ids, (
            "the abandoned request left state behind: the next page decoded "
            "differently"
        )

        # 4. a rejected request must also leave the runner usable
        with pytest.raises(SuryaRequestError):
            gen.generate_multimodal_detailed(
                FULL_PAGE_HTML_PROMPT,
                str(FIXTURES / "page_table.png"),
                request(gen.runner.max_seq),
            )
        after_reject = gen.generate_multimodal_detailed(
            FULL_PAGE_HTML_PROMPT, str(FIXTURES / "page_table.png"), request(512)
        )
        assert after_reject.generated_token_ids == reference.generated_token_ids, (
            "a rejected request changed the runner's state"
        )
    finally:
        gen.close()
