"""Torch-free Surya OCR 2 loading, preprocessing, and prompt assembly.

Everything here runs without torch: config/spec parsing, safetensors
weights (via ``hipengine.loading.safetensors``), byte-BPE tokenization
from the HF ``tokenizer.json`` (via the existing tokenizers-based
encoder used for GGUF models), image preprocessing (PIL resize +
numpy patchify), chat rendering with image-pad expansion, and 3-axis
mRoPE position computation.

Parity targets: ``tests/fixtures/surya/`` (torch oracle). The fixture
validated facts encoded here:

- the processor emits vision patch rows in spatial-merge-block-major
  order (block_row, block_col, in_row, in_col) with the temporal pair
  duplicated for still images;
- decoder mRoPE image positions are RASTER over the merged grid
  offset by the running text position (a transformers quirk kept for
  parity: tower features are block-major while mRoPE indexes raster);
- after an image, the running position advances by
  max(llm_grid_t, llm_grid_h, llm_grid_w); decode steps continue at
  the last prefill position + 1 on all three axes;
- effective EOS is 2 (``<|im_end|>``); text_config.eos_token_id is
  stale metadata and is never used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hipengine.kernels.cpu_reference.surya import (
    SuryaSpec,
    SuryaWeights,
    greedy_generate,
    text_prefill,
)
from hipengine.models.surya import parse_surya_model_spec

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


# ---------------------------------------------------------------------------
# model resolution
# ---------------------------------------------------------------------------


def resolve_surya_path(model_id_or_path: str | Path) -> Path:
    """Resolve a local directory or an HF-cache snapshot for Surya."""

    path = Path(model_id_or_path)
    if path.is_dir():
        return path
    base = Path.home() / ".cache/huggingface/hub"
    repo = "models--datalab-to--surya-ocr-2"
    hits = sorted((base / repo / "snapshots").glob("*")) if (base / repo).exists() else []
    if not hits:
        raise FileNotFoundError(
            f"Surya model {model_id_or_path!r} not found locally; "
            "download datalab-to/surya-ocr-2 with huggingface_hub first"
        )
    return hits[0]


def load_surya_spec(model_dir: Path) -> SuryaSpec:
    """Parse the pinned contract from config.json + generation_config.json."""

    config = json.loads((model_dir / "config.json").read_text())
    generation: dict = {}
    gen_path = model_dir / "generation_config.json"
    if gen_path.exists():
        generation = json.loads(gen_path.read_text())
    return SuryaSpec.from_model_spec(parse_surya_model_spec(config, generation))


def load_surya_weights(model_dir: Path) -> SuryaWeights:
    return SuryaWeights.load(str(model_dir / "model.safetensors"))


# ---------------------------------------------------------------------------
# tokenizer (byte-BPE from tokenizer.json, via the shared HF encoder)
# ---------------------------------------------------------------------------


class SuryaTokenizer:
    """Torch-free Surya tokenizer.

    The shipped tokenizer.json is a CHARACTER-LEVEL tokenizer (a Split
    pre-tokenizer isolating every character feeding a WordLevel lookup)
    with intact Qwen special tokens — an OCR-vocab artifact, consistent
    with the corrupt GGUF tokenizer conversion. Encoding loads the file
    directly through the ``tokenizers`` library (torch-free, exact);
    decoding maps ids back through the vocab and joins.
    """

    def __init__(self, model_dir: Path) -> None:
        path = model_dir / "tokenizer.json"
        data = json.loads(path.read_text())
        self.tokens: list[str] = [""] * len(data["model"]["vocab"])
        for token, idx in data["model"]["vocab"].items():
            self.tokens[idx] = token
        if any(t == "" for t in self.tokens):
            raise ValueError("tokenizer.json vocab has holes")
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.special_ids = {
            int(a["id"]) for a in data.get("added_tokens", []) if a.get("special")
        }
        from tokenizers import Tokenizer

        self._impl = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self._impl.encode(text, add_special_tokens=False).ids]

    def decode(self, token_ids: list[int], *, skip_special: bool = True) -> str:
        pieces = []
        for tid in token_ids:
            tid = int(tid)
            if skip_special and tid in self.special_ids:
                continue
            pieces.append(self.tokens[tid])
        return "".join(pieces)

    def token_string(self, token_id: int) -> str:
        return self.tokens[int(token_id)]


# ---------------------------------------------------------------------------
# chat rendering + image expansion
# ---------------------------------------------------------------------------


def render_chat_prompt(
    tokenizer: SuryaTokenizer,
    text: str,
    n_image_tokens: int | None = None,
) -> tuple[list[int], list[int]]:
    """Render `<|im_start|>user\\n[<|vision_start|><pads><|vision_end|>]text
    <|im_end|>\\n<|im_start|>assistant\\n` (the pinned template's user turn
    with a generation prompt; no thinking prefix).

    Returns (input_ids, mm_token_type_ids) with image pads flagged 1.
    """

    parts = [
        (tokenizer.token_string(2),),  # im_end via literal below
    ]
    del parts
    ids: list[int] = []
    mm: list[int] = []

    def _emit_str(s: str) -> None:
        for tid in tokenizer.encode(s):
            ids.append(tid)
            mm.append(0)

    def _emit_token(token: str, is_image: bool = False) -> None:
        tid = tokenizer.token_to_id.get(token)
        if tid is None:
            raise ValueError(f"token {token!r} not in vocabulary")
        ids.append(tid)
        mm.append(1 if is_image else 0)

    _emit_token(IM_START)
    _emit_str("user\n")
    if n_image_tokens is not None:
        _emit_token(VISION_START)
        for _ in range(n_image_tokens):
            _emit_token(IMAGE_PAD, is_image=True)
        _emit_token(VISION_END)
    _emit_str(text)
    _emit_token(IM_END)
    _emit_str("\n")
    _emit_token(IM_START)
    _emit_str("assistant\n")
    return ids, mm


# ---------------------------------------------------------------------------
# image preprocessing (torch-free)
# ---------------------------------------------------------------------------

# preprocessor_config.json: patch 16, merge 2, temporal 2, bicubic resize,
# rescale 1/255, mean/std 0.5, size bounds are pixel-AREA limits.
SURYA_MIN_PIXELS = 65_536
SURYA_MAX_PIXELS = 16_777_216
SURYA_RESIZE_FACTOR = 32  # patch_size * spatial_merge_size


def smart_resize_surya(
    height: int,
    width: int,
    *,
    factor: int = SURYA_RESIZE_FACTOR,
    min_pixels: int = SURYA_MIN_PIXELS,
    max_pixels: int = SURYA_MAX_PIXELS,
) -> tuple[int, int]:
    """Qwen2VL smart_resize: round both sides down/up to `factor` while
    keeping the pixel area inside [min_pixels, max_pixels]."""

    area = height * width
    if area > max_pixels:
        beta = float(np.sqrt(area / max_pixels))
        height, width = int(height / beta), int(width / beta)
    if area < min_pixels:
        beta = float(np.sqrt(min_pixels / area))
        height, width = int(height * beta), int(width * beta)
    height = max(factor, int(round(height / factor) * factor))
    width = max(factor, int(round(width / factor) * factor))
    if height * width > max_pixels:
        raise ValueError(
            f"resized image {height}x{width} exceeds max_pixels {max_pixels}"
        )
    if height * width < min_pixels:
        # below min after factor rounding: round UP (ceil), as the
        # reference smart_resize does (validated by the rect fixture:
        # 192x320 -> 224x352)
        beta = float(np.sqrt(min_pixels / (height * width)))
        height = int(np.ceil(height * beta / factor) * factor)
        width = int(np.ceil(width * beta / factor) * factor)
    return height, width


def preprocess_image_surya(image: "object") -> tuple[np.ndarray, tuple[int, int, int]]:
    """RGB array/PIL image -> (pixel_rows (n, 3*t*p*p), grid (t, h, w)).

    Patch rows are emitted in spatial-merge-block-major order with the
    temporal pair duplicated (still image), matching the reference
    processor exactly (validated against the torch oracle fixture).
    """

    from PIL import Image

    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    elif isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    else:
        pil = Image.fromarray(np.asarray(image)).convert("RGB")
    w0, h0 = pil.size
    thw_h, thw_w = smart_resize_surya(h0, w0)
    if (thw_h, thw_w) != (h0, w0):
        pil = pil.resize((thw_w, thw_h), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0  # (h, w, 3)
    arr = (arr - 0.5) / 0.5  # mean 0.5 std 0.5

    p, t = 16, 2
    gh, gw = thw_h // p, thw_w // p
    # (gh, p, gw, p, 3) -> per-patch (3, p, p), raster patch order first
    patches = (
        arr.reshape(gh, p, gw, p, 3).transpose(0, 2, 4, 1, 3).reshape(gh * gw, 3, p, p)
    )
    # reorder raster patches -> spatial-merge-block-major
    block_w = gw // 2
    order = []
    for br in range(gh // 2):
        for bc in range(block_w):
            for ir in range(2):
                for ic in range(2):
                    order.append((br * 2 + ir) * gw + (bc * 2 + ic))
    patches = patches[np.array(order)]
    # temporal duplication: (n, 2, 3, p, p) -> (n, 3*t*p*p) rows
    rows = np.stack([patches, patches], axis=1).reshape(patches.shape[0], -1)
    return rows.astype(np.float32), (1, gh, gw)


# ---------------------------------------------------------------------------
# mRoPE positions
# ---------------------------------------------------------------------------


def compute_mrope_positions(
    mm_token_type_ids: list[int],
    grid_thw: tuple[int, int, int] | None,
    spatial_merge_size: int = 2,
) -> np.ndarray:
    """3-axis (t, h, w) decoder positions for one sequence, parity with the
    reference get_rope_index: text runs share one running position; each
    image span takes RASTER (t,h,w) over the merged grid offset by the
    running position; afterwards the running position advances by
    max(grid axes after merge)."""

    seq = len(mm_token_type_ids)
    out = np.zeros((3, seq), dtype=np.int64)
    current_pos = 0
    i = 0
    image_consumed = False
    while i < seq:
        if mm_token_type_ids[i] == 0:
            j = i
            while j < seq and mm_token_type_ids[j] == 0:
                j += 1
            length = j - i
            out[:, i:j] = np.arange(current_pos, current_pos + length)[None, :]
            current_pos += length
            i = j
        else:
            if image_consumed or grid_thw is None:
                raise ValueError("multiple image spans or grid missing")
            image_consumed = True
            j = i
            while j < seq and mm_token_type_ids[j] == 1:
                j += 1
            gt, gh, gw = grid_thw
            llm_h, llm_w = gh // spatial_merge_size, gw // spatial_merge_size
            n = j - i
            if n != llm_h * llm_w:
                raise ValueError(
                    f"image pad span {n} != merged grid {llm_h}x{llm_w}"
                )
            out[0, i:j] = current_pos  # still image: single t value
            hpos = np.repeat(np.arange(llm_h), llm_w)
            wpos = np.tile(np.arange(llm_w), llm_h)
            out[1, i:j] = current_pos + hpos
            out[2, i:j] = current_pos + wpos
            current_pos += max(gt, llm_h, llm_w)
            i = j
    return out


# ---------------------------------------------------------------------------
# end-to-end OCR generation
# ---------------------------------------------------------------------------


@dataclass
class SuryaOCRResult:
    token_ids: list[int]
    text: str
    stopped_on_eos: bool


def run_surya_ocr(
    model_dir: Path,
    image: "object",
    prompt: str = "Transcribe this page.",
    max_new_tokens: int = 256,
) -> SuryaOCRResult:
    """Full torch-free OCR pass: preprocess -> vision tower -> prefill with
    feature injection -> greedy cached decode with EOS stopping.

    CPU-reference path (fp32 numpy); the GPU path registers separately.
    """

    spec = load_surya_spec(model_dir)
    weights = load_surya_weights(model_dir)
    tokenizer = SuryaTokenizer(model_dir)

    pixel_rows, grid = preprocess_image_surya(image)
    _, _, merged = __import__(
        "hipengine.kernels.cpu_reference.surya", fromlist=["vision_forward"]
    ).vision_forward(weights, spec, pixel_rows, [grid])
    n_image_tokens = (grid[1] // 2) * (grid[2] // 2)

    input_ids, mm = render_chat_prompt(tokenizer, prompt, n_image_tokens)
    position_ids = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)

    hidden, state = text_prefill(
        weights,
        spec,
        np.array([input_ids], dtype=np.int64),
        position_ids,
        visual_features=merged[None],
    )
    # greedy decode stepping through the same state container
    emb = weights["model.language_model.embed_tokens.weight"]
    logits = hidden[:, -1] @ emb.T
    generated: list[int] = []
    pos = int(position_ids[:, -1].max())
    stopped = False
    for step in range(max_new_tokens):
        next_token = int(np.argmax(logits[0]))
        if next_token == spec.eos_token_id:
            stopped = True
            break
        generated.append(next_token)
        from hipengine.kernels.cpu_reference.surya import text_decode_step

        logits = text_decode_step(weights, spec, next_token, state, pos + 1 + step)
    return SuryaOCRResult(
        token_ids=generated,
        text=tokenizer.decode(generated, skip_special=True),
        stopped_on_eos=stopped,
    )
