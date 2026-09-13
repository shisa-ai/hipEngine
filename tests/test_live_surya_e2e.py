from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.surya import (
    SuryaTokenizer,
    compute_mrope_positions,
    load_surya_spec,
    load_surya_weights,
    preprocess_image_surya,
    render_chat_prompt,
    resolve_surya_path,
    run_surya_ocr,
    smart_resize_surya,
)

FIXTURES = Path("tests/fixtures/surya")


def _model_dir() -> Path | None:
    try:
        return resolve_surya_path("datalab-to/surya-ocr-2")
    except FileNotFoundError:
        return None


@pytest.fixture(scope="module")
def model_dir() -> Path:
    d = _model_dir()
    if d is None:
        pytest.skip("datalab-to/surya-ocr-2 not in local HF cache")
    return d


@pytest.fixture(scope="module")
def tokenizer(model_dir: Path) -> SuryaTokenizer:
    return SuryaTokenizer(model_dir)


@pytest.fixture(scope="module")
def registered_generators() -> None:
    """Populate the generation registry this file's end-to-end test resolves.

    ``(surya_ocr2, cpu_reference, fp32, greedy_one_token)`` is registered as an
    import-time side effect of ``hipengine.generation.surya``, so a test must not
    depend on another test module having imported it. ``run_surya_ocr`` also
    registers the builtins itself; this fixture keeps the dependency explicit at
    the call site as well.
    """

    from hipengine.generation import register_builtin_generators

    register_builtin_generators()


def _oracle(name: str) -> dict[str, np.ndarray]:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{path} not present; run scripts/surya_oracle_torch.py")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def test_smart_resize_bounds() -> None:
    # 256x256 sits exactly at min_pixels: unchanged
    assert smart_resize_surya(256, 256) == (256, 256)
    # below-min areas round UP to the min (rect fixture: 192x320 -> 224x352)
    assert smart_resize_surya(192, 320) == (224, 352)
    # tiny image upscales to the min area, rounded to factor 32
    h, w = smart_resize_surya(100, 100)
    assert h % 32 == 0 and w % 32 == 0 and h * w >= 65_536
    # huge image downscales to the max area
    h, w = smart_resize_surya(8192, 8192)
    assert h * w <= 16_777_216 and h % 32 == 0


def test_tokenizer_matches_oracle_ids(tokenizer: SuryaTokenizer) -> None:
    # char-level artifact: every text token is one character; specials intact
    ids = tokenizer.encode("Transcribe this page.")
    ref = _oracle("oracle_text.npz")["input_ids"][0]
    # fixture: [im_start, "user\n" x5, 21 text chars, im_end, ...]
    assert ids == list(ref[6:27])
    assert tokenizer.token_to_id["<|im_start|>"] == 1
    assert tokenizer.token_to_id["<|im_end|>"] == 2
    roundtrip = tokenizer.decode(ids, skip_special=False)
    assert roundtrip == "Transcribe this page."


def test_render_chat_prompt_text(tokenizer: SuryaTokenizer) -> None:
    ids, mm = render_chat_prompt(tokenizer, "Transcribe this page.")
    ref = _oracle("oracle_text.npz")["input_ids"][0]
    assert ids == list(ref)
    assert all(m == 0 for m in mm)
    assert ids[0] == 1  # <|im_start|>
    # ends with the generation prompt's trailing newline after "assistant"
    assert tokenizer.tokens[ids[-1]] == "\n"
    assert tokenizer.tokens[ids[-11]] == "<|im_start|>"


def test_render_chat_prompt_image_expansion(tokenizer: SuryaTokenizer) -> None:
    ids, mm = render_chat_prompt(tokenizer, "Transcribe this page.", 64)
    ref = _oracle("oracle_image.npz")
    assert ids == list(ref["input_ids"][0])
    assert mm == list(ref["mm_token_type_ids"][0])
    assert sum(mm) == 64


def test_preprocess_image_matches_oracle() -> None:
    from PIL import Image

    page = Image.open(FIXTURES / "page_small.png")
    rows, grid = preprocess_image_surya(page)
    ref = _oracle("oracle_image.npz")
    assert rows.shape == ref["pixel_values"].shape == (256, 1536)
    # torch-free bicubic + normalize + block-major patchify: bitwise-close
    np.testing.assert_allclose(rows, ref["pixel_values"], atol=1e-6)
    assert grid == (1, 16, 16)


def test_mrope_positions_match_oracle() -> None:
    zi = _oracle("oracle_image.npz")
    mm = list(zi["mm_token_type_ids"][0])
    pos = compute_mrope_positions(mm, (1, 16, 16))
    assert np.array_equal(pos, zi["position_ids_prefill"][:, 0])
    # raster image coords offset by the 7-token text prefix; text resumes
    # at prefix + max(merged grid axes) = 7 + 8 = 15
    span = np.flatnonzero(zi["mm_token_type_ids"][0] == 1)
    assert pos[1, span[0]] == 7 and pos[2, span[-1]] == 14
    assert list(pos[0, span[-1] + 1 : span[-1] + 4]) == [15, 16, 17]


def test_end_to_end_greedy_matches_torch_reference(
    model_dir: Path, registered_generators: None
) -> None:
    ref_path = FIXTURES / "oracle_greedy.json"
    if not ref_path.exists():
        pytest.skip("oracle_greedy.json not present; capture with transformers")
    ref = json.loads(ref_path.read_text())
    page = FIXTURES / "page_small.png"
    # The fixture was captured with the early ad-hoc prompt, so name it here
    # rather than relying on the default: `run_surya_ocr` now defaults to the
    # checkpoint's real full-page prompt, which is a different task.
    res = run_surya_ocr(
        model_dir, str(page), prompt="Transcribe this page.", max_new_tokens=64
    )
    assert res.token_ids == ref["ids"], (
        "torch-free greedy decode diverged from the torch fp32 reference"
    )
    assert res.text == ref["text"]


def test_run_surya_ocr_registers_its_generator_from_a_fresh_process() -> None:
    """The public entry point must not need another module's import.

    ``run_surya_ocr`` resolves ``(surya_ocr2, cpu_reference, fp32,
    greedy_one_token)`` from the four-axis generation registry, whose entry is
    an import-time side effect of ``hipengine.generation.surya``. A fresh
    process that imported only ``hipengine.loading.surya`` used to raise
    ``MissingGeneratorError``, and the full suite hid that because another test
    module imported the registration module during collection.

    Run the isolated case in a subprocess, where no other test has run, and
    require the key to be registered after the call. The call itself then fails
    on the bogus model path, which is fine: registration happens before
    resolution.
    """

    script = (
        "from pathlib import Path\n"
        "from hipengine.generation.registry import registered_text_generators\n"
        "from hipengine.loading.surya import run_surya_ocr\n"
        "try:\n"
        "    run_surya_ocr(Path('/nonexistent-surya-model'), 'page.png', max_new_tokens=1)\n"
        "except Exception:\n"
        "    pass\n"
        "print(sorted(\n"
        "    (k.model, k.backend, k.quant, k.mode) for k in registered_text_generators()\n"
        "))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr
    assert "'surya_ocr2', 'cpu_reference', 'fp32', 'greedy_one_token'" in proc.stdout, (
        "run_surya_ocr did not register its generator in a fresh process: "
        f"registry was {proc.stdout.strip() or '<empty>'}"
    )


def test_rect_case_preprocess_matches_oracle(tokenizer: SuryaTokenizer) -> None:
    """Non-square page: upscales across the min-pixels edge, exercises a
    (14, 22) patch grid with 77 merged image tokens."""

    from PIL import Image

    zi = _oracle("oracle_rect.npz")
    page = Image.open(FIXTURES / "page_rect.png")
    rows, grid = preprocess_image_surya(page)
    assert grid == (1, 14, 22)
    assert rows.shape == zi["pixel_values"].shape == (308, 1536)
    np.testing.assert_allclose(rows, zi["pixel_values"], atol=1e-6)
    ids, mm = render_chat_prompt(tokenizer, "Transcribe this page.", 77)
    assert ids == list(zi["input_ids"][0])
    assert mm == list(zi["mm_token_type_ids"][0])
    pos = compute_mrope_positions(mm, grid)
    assert np.array_equal(pos, zi["position_ids_prefill"][:, 0])


def test_rect_case_vision_and_prefill_match_oracle(model_dir: Path) -> None:
    from hipengine.kernels.cpu_reference.surya import vision_forward, text_prefill

    zi = _oracle("oracle_rect.npz")
    grid_t = zi["image_grid_thw"][0]
    grid = (int(grid_t[0]), int(grid_t[1]), int(grid_t[2]))
    spec = load_surya_spec(model_dir)
    weights = load_surya_weights(model_dir)
    patch_out, tower, merged = vision_forward(weights, spec, zi["pixel_values"], [grid])
    np.testing.assert_allclose(patch_out, zi["vision_patch_embed"], atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(tower, zi["vision_tower_out"], atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(merged, zi["vision_merged"], atol=1e-3, rtol=1e-3)
    # text-side: injected prefill logits
    pos = zi["position_ids_prefill"][:, 0].astype(np.int64)
    hidden, _ = text_prefill(
        weights, spec, zi["input_ids"], pos, visual_features=merged[None]
    )
    emb = weights["model.language_model.embed_tokens.weight"]
    logits = hidden[:, -1] @ emb.T
    ref = zi["logits_first"]
    assert int(np.argmax(logits[0])) == int(np.argmax(ref))
    assert float(np.abs(logits[0] - ref).max()) <= 1e-3


def test_image_case_chunked_prefill_matches_single_pass(model_dir: Path) -> None:
    """Chunk boundary inside the image-token span: state threading (GDN
    conv/recurrent + KV cache) must reproduce single-pass prefill."""

    from hipengine.kernels.cpu_reference.surya import text_prefill

    zi = _oracle("oracle_image.npz")
    spec = load_surya_spec(model_dir)
    weights = load_surya_weights(model_dir)
    ids = zi["input_ids"]
    pos = zi["position_ids_prefill"][:, 0].astype(np.int64)
    merged = zi["vision_merged"][None]
    s = ids.shape[1]
    cut = 53  # inside the 64-token image span

    _, state = text_prefill(
        weights, spec, ids[:, :cut], pos[:, :cut], visual_features=merged
    )
    hidden2, state = text_prefill(
        weights, spec, ids[:, cut:], pos[:, cut:], visual_features=merged, state=state
    )
    emb = weights["model.language_model.embed_tokens.weight"]
    logits_chunked = hidden2[:, -1] @ emb.T
    hidden_full, _ = text_prefill(
        weights, spec, ids, pos, visual_features=merged
    )
    logits_full = hidden_full[:, -1] @ emb.T
    assert state.seq_len == s
    assert int(np.argmax(logits_chunked[0])) == int(np.argmax(logits_full[0]))
    np.testing.assert_allclose(logits_chunked, logits_full, atol=1e-3, rtol=1e-3)
