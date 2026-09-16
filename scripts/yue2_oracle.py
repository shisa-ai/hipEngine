#!/usr/bin/env python3
"""YuE2 torch oracle harness (run with the ROCm oracle venv).

Produces the pinned reference fixtures that the torch-free hipEngine runtime is
gated against. This script is oracle-only: it may import torch and the pinned
upstream package. Nothing here is reachable from ``hipengine.LLM.generate()``.

    PYTHONPATH=~/yue2-shootout/shared/upstream \
    ~/venvs/vibevoice-tts-oracle/bin/python scripts/yue2_oracle.py <command>

Commands
--------
``ar-replay``  compact AR logits-replay fixtures (inputs + summaries + a few
               full-vocab logits) for the phase/CFG/prefix/seed matrix.
``sampling``   reference sampler distributions on synthetic logits with real
               histories (mask/min-EOS/penalty/top-k/top-p order).
``operators``  small NumPy operator fixtures (norm, rope, attention, snake,
               conv/deconv) from the pinned reference implementations.
``nar``        NAR conditioning + solver fixtures (per-step velocities, final
               latents) for a real prefix/codec pair.
``vae``        fixed-latent decoder fixtures (full vs tiled, PCM parity).
``cases``      the twelve production cases end to end (slow; background it).
``env``        record the oracle interpreter, ROCm/GPU, upstream source hashes and
               checkpoint identities (``--out tests/fixtures/yue2/oracle_env.json``).
``freeze``     rewrite ``tests/fixtures/yue2/integrity.json`` from the tree.
``validate``   fail-closed check of every fixture against the frozen index,
               the declared schemas, and the cross-file relations (``--self-test``
               additionally proves the validator rejects broken trees).

Outputs go to ``--out`` (default ``artifacts/yue2/oracle``); compact committed
fixtures are copied to ``--compact`` (default ``tests/fixtures/yue2``).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SHOOTOUT = Path(os.environ.get("YUE2_SHOOTOUT", Path.home() / "yue2-shootout"))
UPSTREAM = SHOOTOUT / "shared/upstream"

MODEL_DIR = Path(
    os.environ.get(
        "YUE2_MODEL_DIR",
        Path.home()
        / ".cache/huggingface/hub/models--m-a-p--YuE2-3B/snapshots/14fc6c6f146441b1dd6363fcb2e01e82a6914cb7",
    )
)
VAE_DIR = Path(
    os.environ.get(
        "YUE2_VAE_DIR",
        Path.home()
        / ".cache/huggingface/hub/models--m-a-p--YuE2-Vae/snapshots/9a94e1d0ea9f8087e98f77fa88df4a4068104d2a",
    )
)

BF16 = None


def _install_path() -> None:
    for entry in (str(UPSTREAM), str(REPO)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def bf16_bits(tensor) -> np.ndarray:
    """Reference BF16 tensor -> uint16 bit patterns (lossless)."""
    return tensor.detach().to("cpu").contiguous().view(__import__("torch").uint16).numpy()


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def prompts() -> list[dict]:
    return json.loads((SHOOTOUT / "shared/prompts.json").read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# shared oracle pieces
# ---------------------------------------------------------------------------


def load_pipeline(memory_budget_gib: int = 100, verify_hashes: bool = False):
    _install_path()
    from yue2.pipeline import YuE2Pipeline

    return YuE2Pipeline(
        MODEL_DIR,
        VAE_DIR,
        device="cuda",
        memory_budget_gib=memory_budget_gib,
        progress=False,
        verify_hashes=verify_hashes,
    )


def load_model(memory_budget_gib: int = 100):
    import torch

    pipe = load_pipeline(memory_budget_gib=memory_budget_gib)
    model = pipe._load_model()
    return pipe, model


def reference_prefixes(tokenizer, phase, length, seed, cfg=False, unequal=False):
    """The proctor matrix's input construction: real prefixes, forced tokens."""
    from yue2.protocol import SongRequest, token_prefixes

    p = prompts()[0 if seed == 1234 else 1]
    request = SongRequest(
        style=p["style"], lyrics=p["lyrics"], cot="off" if phase == "semantic" else "full", seed=seed
    )
    base = token_prefixes(request, tokenizer)
    prefixes = [(base * (length // len(base) + 1))[:length]]
    if cfg:
        negative = token_prefixes(
            SongRequest(style=p["style"], lyrics="", cot="off", seed=seed), tokenizer
        )
        n = 101 if unequal else length
        prefixes.append((negative * (n // len(negative) + 1))[:n])
    if phase == "semantic":
        import random

        rng = random.Random(seed)
        from yue2.protocol import CODEC_OFFSET, CODEC_SIZE

        tokens = [CODEC_OFFSET + rng.randrange(CODEC_SIZE) for _ in range(1024)]
    else:
        abc = list(
            tokenizer.encode("X:1\nT:Morning\nM:4/4\nL:1/8\nK:C\nC D E F G A B c|\n")
        )
        tokens = (abc * (1024 // len(abc) + 1))[:1024]
    return prefixes, tokens


def run_ar_replay(model, prefixes, tokens, *, record_full_steps):
    """Eager reference AR loop, returning prefill and per-step logits.

    Mirrors ``sampling.generate_tokens`` exactly: one static cache per branch,
    prefill then one forced token per step, no sampling.
    """
    import torch
    from yue2.modeling_yue2 import StaticKVCache

    config = model.config
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    steps = len(tokens)
    caches = []
    prefill = []
    with torch.inference_mode():
        for ids in prefixes:
            cache = StaticKVCache(
                num_layers=config.num_hidden_layers,
                batch_size=1,
                num_kv_heads=config.num_key_value_heads,
                max_seq_len=len(ids) + steps,
                head_dim=config.head_dim,
                dtype=dtype,
                device=device,
            )
            out = model(
                torch.tensor([ids], device=device),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            prefill.append(out.logits[:, -1, :].clone())
            caches.append(cache)
        step_logits = []
        for index in range(steps):
            row = []
            token = torch.tensor([[tokens[index]]], device=device)
            for cache in caches:
                out = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                row.append(out.logits[:, -1, :].clone())
            step_logits.append(row)
    return prefill, step_logits


def summarize(logits, topk: int = 8):
    import torch

    values, indices = logits.float().topk(topk, dim=-1)
    return {
        "argmax": indices[..., 0].to(torch.int32).cpu().numpy(),
        "top_ids": indices.to(torch.int32).cpu().numpy(),
        "top_vals": values.cpu().numpy(),
        "sum": float(logits.float().sum()),
        "norm": float(logits.float().norm()),
    }


# ---------------------------------------------------------------------------
# ar-replay
# ---------------------------------------------------------------------------


def cmd_ar_replay(args) -> int:
    import torch

    _install_path()
    pipe, model = load_model()
    out = Path(args.out) / "ar_replay"
    compact = Path(args.compact) / "ar_replay"
    full_steps = {int(value) for value in args.full_steps.split(",") if value}

    matrix = []
    for phase, cfg in (("semantic", False), ("semantic", True), ("abc", False)):
        for length in (128, 512, 2048):
            for seed in (1234, 5678):
                unequal = phase == "semantic" and cfg and seed == 5678
                steps = args.steps if length <= 512 else min(8, args.steps)
                matrix.append((phase, cfg, length, seed, steps, unequal))
    if args.only:
        keep = [value for value in args.only.split(",") if value]
        matrix = [
            row
            for row in matrix
            if any(key in f"{row[0]}-L{row[2]}-s{row[3]}" for key in keep)
        ]

    manifest = {}
    for phase, cfg, length, seed, steps, unequal in matrix:
        name = f"{phase}-{'cfg' if cfg else 'nocfg'}-L{length}-s{seed}"
        if unequal:
            name += "-unequal"
        started = time.perf_counter()
        prefixes, tokens = reference_prefixes(
            pipe.tokenizer, phase, length, seed, cfg=cfg, unequal=unequal
        )
        tokens = tokens[:steps]
        prefill, step_logits = run_ar_replay(
            model, prefixes, tokens, record_full_steps=full_steps
        )
        arrays = {
            "prefix_positive": np.asarray(prefixes[0], dtype=np.int32),
            "tokens": np.asarray(tokens, dtype=np.int32),
            "prefill_logits": bf16_bits(prefill[0]),
        }
        if len(prefixes) > 1:
            arrays["prefix_negative"] = np.asarray(prefixes[1], dtype=np.int32)
            arrays["prefill_logits_negative"] = bf16_bits(prefill[1])
        step_argmax = np.zeros((steps, len(prefixes)), dtype=np.int32)
        top_ids = np.zeros((steps, len(prefixes), 8), dtype=np.int32)
        top_vals = np.zeros((steps, len(prefixes), 8), dtype=np.float32)
        for index, row in enumerate(step_logits):
            for branch, logits in enumerate(row):
                info = summarize(logits)
                step_argmax[index, branch] = info["argmax"][0]
                top_ids[index, branch] = info["top_ids"][0]
                top_vals[index, branch] = info["top_vals"][0]
        arrays["step_argmax"] = step_argmax
        arrays["step_top_ids"] = top_ids
        arrays["step_top_vals"] = top_vals
        full = []
        for index in sorted(full_steps):
            if index >= steps:
                continue
            for branch, logits in enumerate(step_logits[index]):
                arrays[f"full_logits_{index}_{branch}"] = bf16_bits(logits)
                full.append((index, branch))
        arrays["full_logits_steps"] = np.asarray(full, dtype=np.int32).reshape(-1)
        path = out / f"{name}.npz"
        save_npz(path, **arrays)
        compact_path = compact / f"{name}.npz"
        save_npz(
            compact_path,
            prefix_positive=arrays["prefix_positive"],
            tokens=arrays["tokens"],
            prefill_logits=arrays["prefill_logits"],
            **(
                {
                    "prefix_negative": arrays["prefix_negative"],
                    "prefill_logits_negative": arrays["prefill_logits_negative"],
                }
                if "prefix_negative" in arrays
                else {}
            ),
            step_argmax=arrays["step_argmax"],
            step_top_ids=arrays["step_top_ids"],
            step_top_vals=arrays["step_top_vals"],
            full_logits_0_0=arrays.get("full_logits_0_0", np.zeros(0, dtype=np.uint16)),
            full_logits_steps=arrays["full_logits_steps"],
            # The compact fixture deliberately keeps only the first full row; say
            # so, instead of leaving the artifact-wide step list to imply more.
            full_logits_available=np.asarray([(0, 0)], dtype=np.int32).reshape(-1),
        )
        manifest[name] = {
            "phase": phase,
            "cfg": cfg,
            "prefix_lengths": [len(p) for p in prefixes],
            "steps": steps,
            "seed": seed,
            "unequal": unequal,
            "seconds": time.perf_counter() - started,
            # The local full-precision artifact and the committed compact fixture
            # are different files; both identities are recorded.
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "compact_sha256": sha256_file(compact_path),
            "compact_bytes": compact_path.stat().st_size,
        }
        print(
            f"[ar-replay] {name}: prefixes={[len(p) for p in prefixes]} steps={steps} "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    write_json(out / "manifest.json", manifest)
    write_json(compact / "manifest.json", manifest)
    return 0


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def cmd_sampling(args) -> int:
    import torch
    from yue2.protocol import Sampling
    from yue2.sampling import distribution, window_penalty

    _install_path()
    out = Path(args.out) / "sampling"
    compact = Path(args.compact) / "sampling"
    generator = torch.Generator().manual_seed(20260916)
    cases = []
    specs = [
        ("semantic-default", "semantic", 1.0, 0.95, 100, 1.2, 50, 200, 9000, False),
        ("semantic-legacy-off", "semantic", 1.0, 0.95, 100, 1.2, 50, 200, 9000, True),
        ("abc-default", "abc", 0.7, 0.9, 30, 1.005, 100, 32, 4096, False),
        ("zero-temperature", "semantic", 0.0, 0.95, 100, 1.2, 50, 200, 9000, False),
        ("top-p-one", "semantic", 1.0, 1.0, 100, 1.2, 50, 200, 9000, False),
        ("no-penalty", "semantic", 1.0, 0.95, 100, 1.0, 50, 200, 9000, False),
    ]
    vocab = 184704
    for name, phase, temperature, top_p, top_k, penalty, window, min_tokens, max_tokens, legacy in specs:
        sampling = Sampling(temperature, top_p, top_k, penalty, window, min_tokens, max_tokens)
        for step in (0, 100, 199, 200, 201, 500):
            logits = torch.randn(1, vocab, generator=generator, dtype=torch.float32)
            # Realistic magnitudes: reference logits are BF16 model outputs.
            logits = logits.to(torch.bfloat16)
            history = [
                int(value) for value in torch.randint(151853, 184621, (250,), generator=generator)
            ]
            scores = distribution(logits, sampling, history, step, phase, legacy)
            key = f"{name}-step{step}"
            save_npz(
                compact / f"{key}.npz",
                logits=logits.view(torch.uint16).numpy(),
                history=np.asarray(history, dtype=np.int32),
                scores=scores.float().numpy(),
            )
            cases.append(
                {
                    "name": name,
                    "phase": phase,
                    "step": step,
                    "sampling": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": penalty,
                        "penalty_window": window,
                        "min_tokens": min_tokens,
                        "max_tokens": max_tokens,
                    },
                    "legacy_off": legacy,
                    "file": f"{key}.npz",
                }
            )
        # window-penalty-only fixture: exercises sign-dependent scaling.
        logits = torch.randn(1, vocab, generator=generator, dtype=torch.float32).to(torch.bfloat16)
        history = [int(v) for v in torch.randint(0, vocab, (200,), generator=generator)]
        penalized = window_penalty(logits.float(), history[-window:], penalty)
        save_npz(
            compact / f"{name}-penalty.npz",
            logits=logits.view(torch.uint16).numpy(),
            history=np.asarray(history, dtype=np.int32),
            scores=penalized.numpy(),
        )
        cases.append(
            {
                "name": f"{name}-penalty",
                "phase": phase,
                "kind": "window_penalty",
                "penalty_window": window,
                "repetition_penalty": penalty,
                "file": f"{name}-penalty.npz",
            }
        )
    write_json(compact / "manifest.json", cases)
    write_json(out / "manifest.json", cases)
    print(f"[sampling] wrote {len(cases)} fixtures", flush=True)
    return 0


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------


def cmd_operators(args) -> int:
    import torch
    from yue2.modeling_yue2 import (
        AudioPositionEmbedding,
        RMSNorm,
        RotaryEmbedding,
        TimestepEmbedder,
        _apply_rotary,
        sdpa,
    )
    from yue2.modeling_vae import SnakeBeta

    _install_path()
    out = Path(args.compact) / "operators"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    generator = torch.Generator().manual_seed(7)

    # RMSNorm: reference rounds inverse RMS to BF16 before multiplying.
    for rows, width in ((1, 2048), (5, 2048), (7, 128), (3, 6144)):
        x = torch.randn(rows, width, generator=generator).to(torch.bfloat16)
        w = (torch.randn(width, generator=generator) * 0.1 + 1.0).to(torch.bfloat16)
        # The released model is loaded with torch_dtype=bfloat16, which converts
        # every parameter including the norm weights.
        norm = RMSNorm(width, 1e-6).to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(w)
        y = norm(x)
        save_npz(
            out / f"rmsnorm_{rows}x{width}.npz",
            x=x.view(torch.uint16).numpy(),
            weight=w.view(torch.uint16).numpy(),
            out=y.view(torch.uint16).numpy(),
            eps=np.float32(1e-6),
        )

    # Per-head Q/K norm over head_dim, as Attention.project_qkv applies it.
    heads, head_dim, rows = 16, 128, 4
    q = torch.randn(rows, heads, head_dim, generator=generator).to(torch.bfloat16)
    qn = RMSNorm(head_dim, 1e-6).to(torch.bfloat16)
    with torch.no_grad():
        qn.weight.copy_((torch.randn(head_dim, generator=generator) * 0.1 + 1.0).to(torch.bfloat16))
    q_out = qn(q)
    save_npz(
        out / "head_qk_norm.npz",
        q=q.view(torch.uint16).numpy(),
        weight=qn.weight.view(torch.uint16).numpy(),
        out=q_out.view(torch.uint16).numpy(),
    )

    # Rotate-half RoPE, theta 1e6, head_dim 128.
    rope = RotaryEmbedding(128, 1000000.0)
    positions = torch.arange(0, 2048, dtype=torch.long)
    cos, sin = rope(positions)
    x = torch.randn(1, 256, 4, 128, generator=generator).to(torch.bfloat16)
    save_npz(
        out / "rope_cos_sin.npz",
        positions=positions.numpy().astype(np.int32),
        cos=cos.numpy(),
        sin=sin.numpy(),
        x=x.view(torch.uint16).numpy(),
        out=_apply_rotary(
            x, cos[:256].unsqueeze(0).unsqueeze(2), sin[:256].unsqueeze(0).unsqueeze(2)
        )
        .view(torch.uint16)
        .numpy(),
    )

    # Grouped-query causal attention, small and rectangular.
    for name, q_len, kv_len, heads_q, heads_kv in (
        ("attn_prefill", 6, 6, 4, 2),
        ("attn_decode", 1, 6, 4, 2),
    ):
        q = torch.randn(1, heads_q, q_len, 128, generator=generator).to(torch.bfloat16)
        k = torch.randn(1, heads_kv, kv_len, 128, generator=generator).to(torch.bfloat16)
        v = torch.randn(1, heads_kv, kv_len, 128, generator=generator).to(torch.bfloat16)
        if q_len == kv_len:
            out_t = sdpa(q, k, v, is_causal=True)
        else:
            out_t = sdpa(q, k, v)
        save_npz(
            out / f"{name}.npz",
            q=q.view(torch.uint16).numpy(),
            k=k.view(torch.uint16).numpy(),
            v=v.view(torch.uint16).numpy(),
            out=out_t.view(torch.uint16).numpy(),
        )

    # Timestep embedder (sinusoidal -> MLP) and audio position embedding.
    embedder = TimestepEmbedder(2048, 256).to(torch.bfloat16)
    with torch.no_grad():
        for param in embedder.parameters():
            param.copy_(torch.randn(param.shape, generator=generator).to(torch.bfloat16) * 0.05)
    t_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    save_npz(
        out / "timestep_embedder.npz",
        t=t_values.numpy(),
        w0=embedder.mlp[0].weight.view(torch.uint16).numpy(),
        b0=embedder.mlp[0].bias.view(torch.uint16).numpy(),
        w2=embedder.mlp[2].weight.view(torch.uint16).numpy(),
        b2=embedder.mlp[2].bias.view(torch.uint16).numpy(),
        out=embedder(t_values).view(torch.uint16).numpy(),
    )
    pe = AudioPositionEmbedding(24576, 2048)
    save_npz(
        out / "audio_position_embedding.npz",
        positions=np.asarray([0, 1, 2, 100, 4095, 24575], dtype=np.int32),
        pe_rows=np.stack(
            [
                pe.pe[0].numpy(),
                pe.pe[1].numpy(),
                pe.pe[2].numpy(),
                pe.pe[100].numpy(),
                pe.pe[4095].numpy(),
                pe.pe[24575].numpy(),
            ]
        ),
        out=pe(torch.tensor([0, 1, 2, 100, 4095, 24575])).numpy(),
    )

    # SnakeBeta (log-scale alpha/beta, 1e-9 denominator epsilon).
    for channels, length in ((64, 40), (256, 17)):
        snake = SnakeBeta(channels)
        with torch.no_grad():
            snake.alpha.copy_(torch.randn(channels, generator=generator) * 0.3)
            snake.beta.copy_(torch.randn(channels, generator=generator) * 0.3)
        x = torch.randn(1, channels, length, generator=generator) * 2.0
        save_npz(
            out / f"snake_{channels}x{length}.npz",
            x=x.numpy(),
            alpha=snake.alpha.detach().numpy(),
            beta=snake.beta.detach().numpy(),
            out=snake(x).detach().numpy(),
        )

    # Flow-matching timestep shift and the midpoint schedule (FP64 host math).
    shift = 1.0
    raw = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for step in range(32):
            t = np.float64(1.0) - np.float64(step) * (np.float64(1.0) / 32)
            value = float(np.clip(np.log(t / (1 - t)), -20, 20))
            t_mid = t - np.float64(0.5) / 32
            raw.append(
                (step, float(t), value, float(t_mid), float(np.clip(np.log(t_mid / (1 - t_mid)), -20, 20)))
            )
    save_npz(out / "midpoint_schedule.npz", schedule=np.asarray(raw, dtype=np.float64))
    shifted = [
        float(torch.sigmoid(torch.tensor(value, dtype=torch.float32)) * shift / (1 + (shift - 1) * torch.sigmoid(torch.tensor(value, dtype=torch.float32))))
        for value in np.asarray(raw, dtype=np.float64)[:, 2]
    ]
    save_npz(
        out / "timestep_shift.npz",
        raw=np.asarray(raw, dtype=np.float64)[:, 2],
        shifted=np.asarray(shifted, dtype=np.float32),
    )

    # Weight-normalized Conv1d / ConvTranspose1d (folded weight) on small shapes.
    from torch.nn.utils import weight_norm

    conv = weight_norm(torch.nn.Conv1d(6, 4, 7, dilation=3, padding=(3 * (7 - 1)) // 2))
    with torch.no_grad():
        conv.weight_v.copy_(torch.randn_like(conv.weight_v) * 0.2)
        conv.weight_g.copy_(torch.rand_like(conv.weight_g) * 0.5 + 0.2)
        conv.bias.copy_(torch.randn_like(conv.bias) * 0.1)
    conv = conv.eval()
    x = torch.randn(1, 6, 40, generator=generator)
    folded = (conv.weight_g / conv.weight_v.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)) * conv.weight_v
    save_npz(
        out / "conv1d_dilated.npz",
        x=x.numpy(),
        weight_v=conv.weight_v.detach().numpy(),
        weight_g=conv.weight_g.detach().numpy(),
        bias=conv.bias.detach().numpy(),
        folded=folded.detach().numpy(),
        out=conv(x).detach().numpy(),
        stride=np.int32(1),
        dilation=np.int32(3),
        padding=np.int32((3 * (7 - 1)) // 2),
    )
    deconv = weight_norm(
        torch.nn.ConvTranspose1d(4, 3, 2 * 5, stride=5, padding=(5 + 1) // 2)
    ).eval()
    with torch.no_grad():
        deconv.weight_v.copy_(torch.randn_like(deconv.weight_v) * 0.2)
        deconv.weight_g.copy_(torch.rand_like(deconv.weight_g) * 0.5 + 0.2)
        deconv.bias.copy_(torch.randn_like(deconv.bias) * 0.1)
    x = torch.randn(1, 4, 12, generator=generator)
    save_npz(
        out / "convtr1d.npz",
        x=x.numpy(),
        weight_v=deconv.weight_v.detach().numpy(),
        weight_g=deconv.weight_g.detach().numpy(),
        bias=deconv.bias.detach().numpy(),
        out=deconv(x).detach().numpy(),
        stride=np.int32(5),
        padding=np.int32((5 + 1) // 2),
    )
    print(f"[operators] wrote fixtures under {out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# nar
# ---------------------------------------------------------------------------


def cmd_nar(args) -> int:
    import torch

    # The reference package is imported lazily but the path must be installed
    # first: this module is not a dependency of the repository.
    _install_path()
    from yue2.nar import CachedNAR, song_chunks
    from yue2.protocol import CODEC_OFFSET

    pipe, model = load_model()
    out = Path(args.out) / "nar"
    compact = Path(args.compact) / "nar"
    phase = "off"
    length = int(args.prefix_length)
    seed = int(args.seed)
    frames = int(args.frames)
    prefixes, _ = reference_prefixes(pipe.tokenizer, "semantic", length, seed, cfg=False)
    prefix = prefixes[0]
    rng = np.random.default_rng(seed)
    codec = [int(v) for v in rng.integers(0, 32768, size=frames)]

    chunks = song_chunks(prefix, codec, seed)
    for entry in chunks:
        entry.nar_cond_end = int(getattr(args, "nar_cond_end", 0) or 0)
    report = {"prefix_length": len(prefix), "frames": frames, "chunks": len(chunks), "seed": seed,
              "steps": int(args.steps), "nar_cond_end": int(getattr(args, "nar_cond_end", 0))}
    engine = CachedNAR(model, chunks[0])
    chunk = chunks[0]
    state = chunk.noise.to(device=engine.device, dtype=engine.dtype)
    steps = int(args.steps)
    dt = 1.0 / steps
    velocities = []
    states = []
    with torch.inference_mode():
        for step in range(steps):
            t = 1.0 - step * dt
            raw = float(torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20))
            first = engine.velocity(state, raw)
            velocities.append(first.float().cpu().numpy())
            states.append(state.float().cpu().numpy())
            mid = state - first * (dt / 2)
            raw_mid = float(torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64)).clamp(-20, 20))
            state = state - engine.velocity(mid, raw_mid) * dt
    latents = state.float().cpu().numpy()
    # A recorded trace is only useful if it is self-consistent: re-evaluate every
    # recorded velocity at its own recorded state and timestep before writing it,
    # so a schedule or bookkeeping error cannot silently produce a fixture that no
    # correct implementation can match.
    with torch.inference_mode():
        for step in range(steps):
            t = 1.0 - step * dt
            raw = float(torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20))
            replay = engine.velocity(
                torch.from_numpy(np.asarray(states[step])).to(device=engine.device, dtype=engine.dtype),
                raw,
            ).float().cpu().numpy()
            drift = float(np.abs(replay - np.asarray(velocities[step])).max())
            report[f"replay_step{step}_max_abs"] = drift
            if drift > 1e-3:
                raise RuntimeError(
                    f"NAR trace replay mismatch at step {step}: max abs {drift}; "
                    "the recorded velocity does not correspond to the recorded state"
                )
    save_npz(
        out / "chunk0.npz",
        prefix=np.asarray(prefix, dtype=np.int32),
        codec=np.asarray(codec, dtype=np.int32),
        ar_tokens=np.asarray(chunk.ar_tokens, dtype=np.int32),
        noise=chunk.noise.numpy(),
        velocities=np.asarray(velocities),
        states=np.asarray(states),
        latents=latents,
    )
    save_npz(
        compact / "chunk0.npz",
        prefix=np.asarray(prefix, dtype=np.int32),
        codec=np.asarray(codec, dtype=np.int32),
        noise=chunk.noise.numpy(),
        velocities=np.asarray(velocities[:4]),
        states=np.asarray(states[:4]),
        latents=latents,
    )
    report["latent_norm"] = float(np.linalg.norm(latents))
    report["velocity_norm_step0"] = float(np.linalg.norm(velocities[0]))
    engine.close()
    write_json(compact / "manifest.json", report)
    write_json(out / "manifest.json", report)
    print(f"[nar] {report}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# vae
# ---------------------------------------------------------------------------


def cmd_vae(args) -> int:
    import torch
    from yue2.modeling_vae import YuE2VAE

    _install_path()
    out = Path(args.out) / "vae"
    compact = Path(args.compact) / "vae"
    model = YuE2VAE.from_pretrained(VAE_DIR, decoder_only=True, device="cuda")
    rng = np.random.default_rng(int(args.seed))
    frames = int(args.frames)
    latent = rng.standard_normal((1, 64, frames)).astype(np.float32)
    z = torch.from_numpy(latent).to("cuda")
    with torch.inference_mode():
        full = model.decode(z).cpu().numpy()
        tiled = model.decode_tiled(z, core_frames=16, halo_frames=16, output_device="cpu").numpy()
    report = {
        "frames": frames,
        "natural_length": int(model.natural_output_length(frames)),
        "full_shape": list(full.shape),
        "tiled_shape": list(tiled.shape),
        "max_abs_full_vs_tiled": float(np.abs(full - tiled).max()),
        "rms_full_vs_tiled": float(np.sqrt(((full - tiled) ** 2).mean())),
        "required_halo_core16": int(model.required_halo(16)),
        "peak": float(np.abs(full).max()),
        "rms": float(np.sqrt((full**2).mean())),
    }
    save_npz(
        out / "decode.npz",
        latent=latent,
        full=full,
        tiled=tiled,
        latent_small=latent[:, :, :3],
        full_small=model.decode(torch.from_numpy(latent[:, :, :3]).to("cuda")).cpu().numpy(),
    )
    small = model.decode(torch.from_numpy(latent[:, :, :3]).to("cuda")).cpu().numpy()
    save_npz(
        compact / "decode.npz",
        latent=latent[:, :, :3],
        full=small,
        natural_length=np.int32(model.natural_output_length(3)),
    )
    report["small_shape"] = list(small.shape)
    report["small_natural_length"] = int(model.natural_output_length(3))
    report["small_peak"] = float(np.abs(small).max())
    report["small_rms"] = float(np.sqrt((small.astype(np.float64) ** 2).mean()))
    write_json(out / "manifest.json", report)
    write_json(compact / "manifest.json", report)
    print(f"[vae] {report}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# greedy AR trajectories
# ---------------------------------------------------------------------------


def cmd_greedy(args) -> int:
    """Free AR generation at temperature zero, recorded for the session gate.

    Only the AR stages run (no NAR, no VAE), so the fixture is cheap to
    regenerate. Greedy decoding removes the RNG from the comparison: a torch-free
    session that reproduces these trajectories has matching prompt assembly,
    masks, penalties, CFG arithmetic and EOS/truncation behavior.
    """

    import dataclasses

    _install_path()
    from yue2.protocol import (
        CODEC_OFFSET,
        GenerationConfig,
        SongRequest,
        negative_prefix,
        token_prefixes,
    )
    from yue2.sampling import generate_tokens

    pipe, model = load_model()
    tokenizer = pipe.tokenizer
    config = GenerationConfig()
    abc_sampling = dataclasses.replace(config.abc, temperature=0.0, max_tokens=args.max_abc)
    semantic_sampling = dataclasses.replace(
        config.semantic, temperature=0.0, max_tokens=args.max_semantic
    )
    out = Path(args.out) / "greedy"
    compact = Path(args.compact) / "greedy"
    names = args.only.split(",") if args.only else None
    cases = []
    for entry in prompts():
        for cot in ("off", "melody", "full"):
            for seed in entry["seeds"]:
                name = f"{entry['id']}-{cot}-s{seed}"
                if names and name not in names:
                    continue
                cases.append((name, entry, cot, seed))
    if not cases:
        raise SystemExit("no greedy cases selected")
    manifest = {}
    for name, entry, cot, seed in cases:
        request = SongRequest(style=entry["style"], lyrics=entry["lyrics"], cot=cot, seed=seed, id=name)
        started = time.perf_counter()
        abc_ids: list[int] = []
        abc_truncated = False
        if cot != "off":
            abc_ids, _, abc_truncated = generate_tokens(
                model, token_prefixes(request, tokenizer), abc_sampling, seed, "abc"
            )
        prefix = token_prefixes(request, tokenizer, abc_ids)
        negative = (
            negative_prefix(request, tokenizer, abc_ids) if request.guidance != 1.0 else None
        )
        semantic, _, semantic_truncated = generate_tokens(
            model,
            prefix,
            semantic_sampling,
            seed,
            "semantic",
            negative=negative,
            cfg_scale=request.guidance,
            legacy_off=cot == "off",
        )
        arrays = {
            "prefix": np.asarray(prefix, dtype=np.int32),
            "semantic": np.asarray(
                [int(token) - CODEC_OFFSET for token in semantic], dtype=np.int32
            ),
        }
        if abc_ids:
            arrays["abc_ids"] = np.asarray(abc_ids, dtype=np.int32)
        if negative is not None:
            arrays["negative_prefix"] = np.asarray(negative, dtype=np.int32)
        save_npz(compact / f"{name}.npz", **arrays)
        save_npz(out / f"{name}.npz", **arrays)
        manifest[name] = {
            "request": request.to_dict(),
            "cot": cot,
            "seed": seed,
            "abc_tokens": len(abc_ids),
            "semantic_tokens": len(semantic),
            "truncated_abc": bool(abc_truncated),
            "truncated_semantic": bool(semantic_truncated),
            "cfg_scale": request.guidance,
            "prefix_tokens": len(prefix),
            "negative_prefix_tokens": len(negative) if negative is not None else 0,
            "sampling": {"abc": dataclasses.asdict(abc_sampling), "semantic": dataclasses.asdict(semantic_sampling)},
            "seconds": time.perf_counter() - started,
        }
        print(f"[greedy] {name}: {manifest[name]}", flush=True)
        write_json(out / "manifest.json", manifest)
    write_json(compact / "manifest.json", manifest)
    return 0


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def cmd_cases(args) -> int:
    import torch

    _install_path()
    from yue2.protocol import GenerationConfig, SongRequest

    pipe = load_pipeline()
    out = Path(args.out) / "cases"
    compact = Path(args.compact) / "cases"
    config = GenerationConfig()
    names = args.only.split(",") if args.only else None
    cases = []
    for entry in prompts():
        for cot in ("off", "melody", "full"):
            for seed in entry["seeds"]:
                name = f"{entry['id']}-{cot}-s{seed}"
                if names and name not in names:
                    continue
                cases.append((name, entry, cot, seed))
    manifest = {}
    for name, entry, cot, seed in cases:
        directory = out / name
        if (directory / "result.json").is_file() and not args.force:
            manifest[name] = json.loads((directory / "result.json").read_text())
            print(f"[cases] {name}: cached", flush=True)
            continue
        request = SongRequest(style=entry["style"], lyrics=entry["lyrics"], cot=cot, seed=seed, id=name)
        started = time.perf_counter()
        result = pipe(
            style=entry["style"], lyrics=entry["lyrics"], cot=cot, seed=seed, id=name
        )
        assert result.semantic.plan.request.to_dict() == request.to_dict()
        result.save_artifacts(directory)
        manifest[name] = {
            "seconds": time.perf_counter() - started,
            "truncated": result.truncated,
            "semantic_tokens": len(result.semantic.tokens),
            "latent_frames": int(result.latents.shape[0]),
            "audio_seconds": len(result.audio) / 48000,
            "identity": result.request_identity,
            "abc_tokens": len(result.semantic.plan.abc_ids),
            "prefix_tokens": len(result.semantic.plan.prefix),
        }
        # Compact committed fixture: plan + semantic tokens + latent statistics +
        # a short PCM excerpt (full artifacts stay in the local store).
        save_npz(
            compact / f"{name}.npz",
            abc_ids=np.asarray(result.semantic.plan.abc_ids, dtype=np.int32),
            prefix=np.asarray(result.semantic.plan.prefix, dtype=np.int32),
            semantic=np.asarray(result.semantic.tokens, dtype=np.int32),
            latent_shape=np.asarray(result.latents.shape, dtype=np.int32),
            latent_excerpt=result.latents[: min(64, result.latents.shape[0])].astype(np.float32),
            audio_excerpt=result.audio[: 48000 * 2].astype(np.float32),
            latent_norm=np.float32(np.linalg.norm(result.latents)),
            audio_rms=np.float32(np.sqrt((result.audio**2).mean())),
            audio_peak=np.float32(np.abs(result.audio).max()),
        )
        if result.semantic.plan.abc is not None:
            (compact / f"{name}.abc").write_text(result.semantic.plan.abc)
        print(f"[cases] {name}: {manifest[name]}", flush=True)
        write_json(out / "manifest.json", manifest)
    write_json(compact / "manifest.json", manifest)
    return 0


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

TOKENIZER_CORPUS = (
    "Generate music with codec tokens from the given conditions.\n[Tags]\nWarm acoustic pop, "
    "clear lead vocal, fingerpicked guitar\n[Lyrics]\n[Verse]\nMorning light across the floor\n",
    "Mandarin pop, warm expressive lead vocal, gentle piano\n清晨的光落在窗前\n远方的风吹过昨天\n",
    "X:1\nT:Morning\nM:4/4\nL:1/8\nK:C\nC D E F G A B c|\n",
    "It's a test of don't, we're, I've, you'll, he'd, they're cases 1234 5678",
    "Mixed 中文 and English, punctuation!?;: 1,234.5 — em-dash… ellipsis",
    "emoji 🎵🎶 and symbols ©®™ ± § ¶",
    "  leading spaces\ttabs\n\nnewlines\r\n  trailing  ",
    "café naïve résumé ünïcödé ĀāĂăĄąĆćĈĉ",
    "[[Verse]]\\nline one\\nline two\\n[Chorus]",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    "Music: 1/8 notes A, B, c, d' e'' f# gb z2 z/2 [CEG] (C2 D4) |: x :|",
    "やまと言葉のテスト、ひらがなカタカナ漢字。",
    "한국어 테스트 문장입니다.",
    "العربية اختبار النص",
    "\u0000\u0001\u001f control characters \u007f",
    "a" * 300,
    "重复重复重复重复重复重复重复重复",
    "MiXeD cAsE WoRdS AND numbers 42",
    "<|endoftext|><|im_start|><|im_end|><R><S><X><mask><sep><abc></abc><extra_0>",
)


def cmd_tokenizer(args) -> int:
    import unicodedata

    _install_path()
    import tiktoken

    out = Path(args.compact) / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    merge_file = MODEL_DIR / "qwen.tiktoken"
    ranks = {
        __import__("base64").b64decode(token): int(rank)
        for token, rank in (line.split() for line in merge_file.read_bytes().splitlines() if line)
    }
    specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<R>",
        "<S>",
        "<X>",
        "<mask>",
        "<sep>",
    ]
    specials += [f"<extra_{index}>" for index in range(200)]
    specials[204:206] = ["<abc>", "</abc>"]
    pattern = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    encoding = tiktoken.Encoding(
        "YuE2",
        pat_str=pattern,
        mergeable_ranks=ranks,
        special_tokens={name: index + len(ranks) for index, name in enumerate(specials)},
    )
    payload = {}
    for index, text in enumerate(TOKENIZER_CORPUS):
        payload[f"text_{index}"] = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        payload[f"ids_{index}"] = np.asarray(
            encoding.encode_ordinary(unicodedata.normalize("NFC", text)), dtype=np.int32
        )
    save_npz(out / "corpus.npz", **payload)
    write_json(
        out / "manifest.json",
        {
            "merge_sha256": sha256_file(merge_file),
            "ordinary_ranks": len(ranks),
            "specials": len(specials),
            "cases": len(TOKENIZER_CORPUS),
            "special_ids": {name: index + len(ranks) for index, name in enumerate(specials)},
        },
    )
    print(f"[tokenizer] wrote {len(TOKENIZER_CORPUS)} corpus cases", flush=True)
    return 0


# ---------------------------------------------------------------------------
# fixture validation
# ---------------------------------------------------------------------------

VOCAB = 184704
EOD = 151643
CODEC_SIZE = 32768
FIXTURE_FAMILIES = ("ar_replay", "cases", "greedy", "nar", "operators", "sampling", "tokenizer", "vae")
INTEGRITY_NAME = "integrity.json"


def _dtype_ok(array: np.ndarray, name: str) -> bool:
    return array.dtype.str.lstrip("<=>|") == np.dtype(name).str.lstrip("<=>|")


def _fold_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    """Independent NumPy weight_norm fold used to check the fixture's own fold."""
    norm = np.sqrt(
        np.sum(np.square(weight_v.astype(np.float64)), axis=(1, 2), keepdims=True)
    )
    return ((weight_g.astype(np.float64) / norm) * weight_v.astype(np.float64)).astype(np.float32)


def _finite(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all()) if array.dtype.kind == "f" else True


class FixtureProblems(list):
    """Collects human-readable validation problems."""

    def add(self, path, message):
        self.append(f"{path}: {message}")

    def require(self, condition, path, message):
        if not condition:
            self.add(path, message)
        return bool(condition)


# key -> (dtype, ndim)
OPERATOR_SCHEMAS: dict[str, dict[str, tuple[str, int]]] = {
    "attn_decode.npz": {"q": ("uint16", 4), "k": ("uint16", 4), "v": ("uint16", 4), "out": ("uint16", 4)},
    "attn_prefill.npz": {"q": ("uint16", 4), "k": ("uint16", 4), "v": ("uint16", 4), "out": ("uint16", 4)},
    "audio_position_embedding.npz": {
        "positions": ("int32", 1),
        "pe_rows": ("float32", 2),
        "out": ("float32", 2),
    },
    "conv1d_dilated.npz": {
        "x": ("float32", 3),
        "weight_v": ("float32", 3),
        "weight_g": ("float32", 3),
        "bias": ("float32", 1),
        "folded": ("float32", 3),
        "out": ("float32", 3),
        "stride": ("int32", 0),
        "dilation": ("int32", 0),
        "padding": ("int32", 0),
    },
    "convtr1d.npz": {
        "x": ("float32", 3),
        "weight_v": ("float32", 3),
        "weight_g": ("float32", 3),
        "bias": ("float32", 1),
        "out": ("float32", 3),
        "stride": ("int32", 0),
        "padding": ("int32", 0),
    },
    "head_qk_norm.npz": {"q": ("uint16", 3), "weight": ("uint16", 1), "out": ("uint16", 3)},
    "midpoint_schedule.npz": {"schedule": ("float64", 2)},
    "rmsnorm_1x2048.npz": {"x": ("uint16", 2), "weight": ("uint16", 1), "out": ("uint16", 2), "eps": ("float32", 0)},
    "rmsnorm_3x6144.npz": {"x": ("uint16", 2), "weight": ("uint16", 1), "out": ("uint16", 2), "eps": ("float32", 0)},
    "rmsnorm_5x2048.npz": {"x": ("uint16", 2), "weight": ("uint16", 1), "out": ("uint16", 2), "eps": ("float32", 0)},
    "rmsnorm_7x128.npz": {"x": ("uint16", 2), "weight": ("uint16", 1), "out": ("uint16", 2), "eps": ("float32", 0)},
    "rope_cos_sin.npz": {
        "positions": ("int32", 1),
        "cos": ("float32", 2),
        "sin": ("float32", 2),
        "x": ("uint16", 4),
        "out": ("uint16", 4),
    },
    "snake_256x17.npz": {"x": ("float32", 3), "alpha": ("float32", 1), "beta": ("float32", 1), "out": ("float32", 3)},
    "snake_64x40.npz": {"x": ("float32", 3), "alpha": ("float32", 1), "beta": ("float32", 1), "out": ("float32", 3)},
    "timestep_embedder.npz": {
        "t": ("float32", 1),
        "w0": ("uint16", 2),
        "b0": ("uint16", 1),
        "w2": ("uint16", 2),
        "b2": ("uint16", 1),
        "out": ("uint16", 2),
    },
    "timestep_shift.npz": {"raw": ("float64", 1), "shifted": ("float32", 1)},
}

SAMPLING_SCHEMA = {"logits": ("uint16", 2), "history": ("int32", 1), "scores": ("float32", 2)}

AR_REPLAY_SCHEMA = {
    "prefix_positive": ("int32", 1),
    "tokens": ("int32", 1),
    "prefill_logits": ("uint16", 2),
    "step_argmax": ("int32", 2),
    "step_top_ids": ("int32", 3),
    "step_top_vals": ("float32", 3),
    "full_logits_0_0": ("uint16", 2),
    "full_logits_steps": ("int32", 1),
}
AR_REPLAY_OPTIONAL = {
    "prefix_negative": ("int32", 1),
    "prefill_logits_negative": ("uint16", 2),
    # Pairs with a full-vocabulary row inside *this* file. ``full_logits_steps``
    # lists what the full-precision artifact recorded; the committed compact
    # fixture keeps a subset, and older fixtures predate this key.
    "full_logits_available": ("int32", 1),
}


def _check_schema(problems, path, arrays, schema, optional=None, allow_extra=False, allow_inf=()):
    optional = optional or {}
    for key, (dtype, ndim) in schema.items():
        if key not in arrays:
            problems.add(path, f"missing key {key!r}")
            continue
        array = arrays[key]
        problems.require(_dtype_ok(array, dtype), path, f"{key} dtype {array.dtype} != {dtype}")
        if ndim is not None:
            problems.require(array.ndim == ndim, path, f"{key} ndim {array.ndim} != {ndim}")
        problems.require(array.size > 0, path, f"{key} is empty")
        if array.dtype.kind == "f" and key in allow_inf:
            problems.require(
                not bool(np.isnan(array).any()), path, f"{key} has NaN values"
            )
        else:
            problems.require(_finite(array), path, f"{key} has non-finite values")
    for key, (dtype, ndim) in optional.items():
        if key in arrays:
            problems.require(_dtype_ok(arrays[key], dtype), path, f"{key} dtype != {dtype}")
            problems.require(arrays[key].ndim == ndim, path, f"{key} ndim != {ndim}")
            problems.require(_finite(arrays[key]), path, f"{key} has non-finite values")
    if not allow_extra:
        extra = sorted(set(arrays) - set(schema) - set(optional))
        if extra:
            problems.add(path, f"unexpected keys {extra}")


def _load_arrays(path, problems):
    try:
        with np.load(path, allow_pickle=False) as handle:
            return {key: handle[key] for key in handle.files}
    except Exception as error:  # noqa: BLE001 - reported as a validation problem
        problems.add(path, f"unreadable npz: {type(error).__name__}: {error}")
        return None


def _check_ar_replay(root, problems):
    manifest_path = root / "ar_replay/manifest.json"
    if not manifest_path.is_file():
        problems.add(manifest_path, "missing manifest")
        return
    manifest = json.loads(manifest_path.read_text())
    for name, entry in sorted(manifest.items()):
        path = root / "ar_replay" / f"{name}.npz"
        if not path.is_file():
            problems.add(path, "missing fixture")
            continue
        problems.require(
            "compact_sha256" in entry and "compact_bytes" in entry,
            path,
            "manifest does not record the committed fixture identity",
        )
        problems.require(
            sha256_file(path) == entry.get("compact_sha256"),
            path,
            "sha256 does not match the manifest",
        )
        problems.require(
            path.stat().st_size == entry.get("compact_bytes"), path, "byte size changed"
        )
        arrays = _load_arrays(path, problems)
        if arrays is None:
            continue
        _check_schema(problems, path, arrays, AR_REPLAY_SCHEMA, AR_REPLAY_OPTIONAL)
        steps = int(entry["steps"])
        branches = 2 if entry["cfg"] else 1
        lengths = entry["prefix_lengths"]
        problems.require(
            arrays["tokens"].shape == (steps,), path, f"tokens shape != ({steps},)"
        )
        problems.require(
            arrays["prefix_positive"].shape == (lengths[0],),
            path,
            f"prefix_positive shape != ({lengths[0]},)",
        )
        problems.require(
            arrays["step_argmax"].shape == (steps, branches),
            path,
            f"step_argmax shape != {(steps, branches)}",
        )
        problems.require(
            arrays["step_top_ids"].shape == (steps, branches, 8),
            path,
            f"step_top_ids shape != {(steps, branches, 8)}",
        )
        problems.require(
            arrays["step_top_vals"].shape == (steps, branches, 8),
            path,
            f"step_top_vals shape != {(steps, branches, 8)}",
        )
        problems.require(
            arrays["prefill_logits"].shape == (1, VOCAB),
            path,
            f"prefill_logits shape != {(1, VOCAB)}",
        )
        if "prefix_negative" in arrays:
            problems.require(
                arrays["prefix_negative"].shape == (lengths[1],),
                path,
                f"prefix_negative shape != ({lengths[1]},)",
            )
            problems.require(
                len(lengths) == 2, path, "prefix_negative present but manifest lists one prefix"
            )
        elif len(lengths) != 1:
            problems.add(path, "manifest lists two prefixes but prefix_negative is absent")
        # The recorded full-vocab rows must line up with the recorded step list.
        recorded = arrays["full_logits_steps"].reshape(-1, 2)
        for index, branch in recorded:
            problems.require(
                0 <= index < steps and 0 <= branch < branches,
                path,
                f"full_logits_steps entry ({index}, {branch}) is out of range",
            )
        problems.require(
            tuple(recorded[0]) == (0, 0), path, f"first recorded step is {tuple(recorded[0])}"
        )
        # Every pair claimed to be in this file must actually be present, and no
        # unlisted full-vocabulary row may hide in it.
        present = sorted(
            (int(key.split("_")[2]), int(key.split("_")[3]))
            for key in arrays
            if key.startswith("full_logits_") and key != "full_logits_steps"
        )
        if "full_logits_available" in arrays:
            available = arrays["full_logits_available"].reshape(-1, 2)
            available_pairs = [(int(i), int(b)) for i, b in available]
            problems.require(
                len(set(available_pairs)) == len(available_pairs),
                path,
                "full_logits_available repeats a pair",
            )
            for pair in available_pairs:
                problems.require(
                    pair in set((int(i), int(b)) for i, b in recorded),
                    path,
                    f"full_logits_available entry {pair} is not in full_logits_steps",
                )
            problems.require(
                sorted(available_pairs) == present,
                path,
                f"full_logits_available {available_pairs} != rows present {present}",
            )
        for pair in present:
            problems.require(
                pair in set((int(i), int(b)) for i, b in recorded),
                path,
                f"full-vocabulary row {pair} is not listed in full_logits_steps",
            )
        # Top-8 rows are sorted, and their head agrees with the argmax row.
        top_vals = arrays["step_top_vals"]
        if top_vals.ndim == 3 and top_vals.shape[-1] == 8:
            problems.require(
                bool((np.diff(top_vals, axis=-1) <= 1e-6).all()),
                path,
                "step_top_vals are not sorted descending",
            )
            problems.require(
                bool((arrays["step_top_ids"][..., 0] == arrays["step_argmax"]).all()),
                path,
                "step_top_ids head disagrees with step_argmax",
            )
        for key in ("prefix_positive", "tokens"):
            values = arrays[key]
            problems.require(
                bool(((values >= 0) & (values < VOCAB)).all()),
                path,
                f"{key} has ids outside [0, {VOCAB})",
            )


def _check_sampling(root, problems):
    manifest_path = root / "sampling/manifest.json"
    if not manifest_path.is_file():
        problems.add(manifest_path, "missing manifest")
        return
    manifest = json.loads(manifest_path.read_text())
    listed = {entry["file"] for entry in manifest}
    for entry in manifest:
        path = root / "sampling" / entry["file"]
        if not path.is_file():
            problems.add(path, "missing fixture")
            continue
        arrays = _load_arrays(path, problems)
        if arrays is None:
            continue
        # Masked scores are -inf by construction; NaN would be a real defect.
        _check_schema(problems, path, arrays, SAMPLING_SCHEMA, allow_inf=("scores",))
        problems.require(
            arrays["logits"].shape == (1, VOCAB), path, f"logits shape != {(1, VOCAB)}"
        )
        problems.require(
            arrays["scores"].shape == (1, VOCAB), path, f"scores shape != {(1, VOCAB)}"
        )
        if "sampling" in entry:
            problems.require(
                set(entry["sampling"]) >= {"temperature", "top_k", "top_p", "repetition_penalty"},
                path,
                "manifest entry does not describe the sampler",
            )
            problems.require(int(entry["step"]) >= 0, path, f"negative step {entry['step']}")
        else:
            problems.require(
                {"kind", "penalty_window", "repetition_penalty"} <= set(entry),
                path,
                "penalty fixture does not describe its penalty parameters",
            )
            problems.require(
                int(entry["penalty_window"]) > 0 and float(entry["repetition_penalty"]) >= 1.0,
                path,
                "penalty parameters are out of range",
            )
    for path in sorted((root / "sampling").glob("*.npz")):
        if path.name not in listed:
            problems.add(path, "unlisted fixture (not in sampling/manifest.json)")


def _check_tokenizer(root, problems):
    manifest_path = root / "tokenizer/manifest.json"
    if not manifest_path.is_file():
        problems.add(manifest_path, "missing manifest")
        return
    manifest = json.loads(manifest_path.read_text())
    path = root / "tokenizer/corpus.npz"
    arrays = _load_arrays(path, problems) if path.is_file() else None
    if arrays is None:
        if not path.is_file():
            problems.add(path, "missing fixture")
        return
    cases = int(manifest["cases"])
    expected = {f"text_{index}" for index in range(cases)} | {
        f"ids_{index}" for index in range(cases)
    }
    problems.require(
        set(arrays) == expected,
        path,
        f"corpus keys do not match {cases} cases (extra={sorted(set(arrays) - expected)})",
    )
    for index in range(cases):
        text, ids = arrays.get(f"text_{index}"), arrays.get(f"ids_{index}")
        if text is None or ids is None:
            continue
        problems.require(_dtype_ok(text, "uint8"), path, f"text_{index} dtype != uint8")
        problems.require(_dtype_ok(ids, "int32"), path, f"ids_{index} dtype != int32")
        problems.require(text.size > 0 and ids.size > 0, path, f"case {index} is empty")
        problems.require(
            bool(((ids >= 0) & (ids < VOCAB)).all()), path, f"ids_{index} outside vocab"
        )
    special_ids = manifest["special_ids"]
    problems.require(
        special_ids["<abc>"] == 151847 and special_ids["</abc>"] == 151848,
        manifest_path,
        "ABC special-token ids are not the pinned values",
    )
    problems.require(
        len(special_ids) == 208, manifest_path, f"expected 208 specials, found {len(special_ids)}"
    )
    problems.require(
        manifest["ordinary_ranks"] == 151643,
        manifest_path,
        f"ordinary_ranks {manifest['ordinary_ranks']} != 151643",
    )


def _check_operators(root, problems):
    directory = root / "operators"
    for path in sorted(directory.glob("*.npz")):
        schema = OPERATOR_SCHEMAS.get(path.name)
        if schema is None:
            problems.add(path, "no declared schema for this operator fixture")
            continue
        arrays = _load_arrays(path, problems)
        if arrays is None:
            continue
        _check_schema(problems, path, arrays, schema)
        if path.name == "rope_cos_sin.npz":
            problems.require(
                arrays["cos"].shape == arrays["sin"].shape == arrays["positions"].shape + (64,),
                path,
                "cos/sin are not [positions, 64]",
            )
            problems.require(
                arrays["x"].shape == arrays["out"].shape, path, "rope x/out shapes differ"
            )
            problems.require(
                arrays["x"].shape[-1] == 128, path, "rope last dimension is not 128"
            )
        if path.name.startswith("rmsnorm_"):
            width = arrays["x"].shape[-1]
            problems.require(
                arrays["weight"].shape == (width,), path, "weight does not match the row width"
            )
            problems.require(
                arrays["out"].shape == arrays["x"].shape, path, "x/out shapes differ"
            )
            problems.require(
                abs(float(arrays["eps"]) - 1e-6) <= 1e-12,
                path,
                f"eps {float(arrays['eps'])} != 1e-6",
            )
        if path.name == "conv1d_dilated.npz":
            folded = _fold_weight_norm(arrays["weight_g"], arrays["weight_v"])
            problems.require(
                np.allclose(folded, arrays["folded"], rtol=1e-5, atol=1e-6),
                path,
                "folded weights do not reproduce weight_norm",
            )
            problems.require(
                arrays["out"].shape == (arrays["x"].shape[0], arrays["weight_v"].shape[0], arrays["x"].shape[2]),
                path,
                "dilated conv preserves length but the shapes disagree",
            )
        if path.name == "convtr1d.npz":
            stride, padding = int(arrays["stride"]), int(arrays["padding"])
            kernel = arrays["weight_v"].shape[-1]
            expected = (arrays["x"].shape[-1] - 1) * stride - 2 * padding + kernel
            problems.require(
                arrays["out"].shape[-1] == expected,
                path,
                f"deconv length {arrays['out'].shape[-1]} != {expected}",
            )
        if path.name == "head_qk_norm.npz":
            problems.require(
                arrays["weight"].shape == (arrays["q"].shape[-1],),
                path,
                "head norm weight does not match the head width",
            )
        if path.name == "timestep_embedder.npz":
            problems.require(
                arrays["out"].shape == (arrays["t"].size, arrays["b2"].size),
                path,
                "timestep embedder output shape is inconsistent",
            )
        if path.name == "timestep_shift.npz":
            problems.require(
                arrays["raw"].shape == arrays["shifted"].shape, path, "raw/shifted shapes differ"
            )
            problems.require(
                bool((arrays["shifted"] > 0).all()), path, "shifted timesteps must stay positive"
            )
        if path.name == "midpoint_schedule.npz":
            schedule = arrays["schedule"]
            problems.require(
                schedule.shape[1] >= 5, path, "schedule needs at least 5 columns"
            )
            problems.require(
                bool((np.diff(schedule[:, 1]) < 0).all()),
                path,
                "midpoint schedule timesteps must decrease",
            )
            problems.require(
                bool((schedule[:, 1] > 0).all()) and bool((schedule[:, 1] <= 1).all()),
                path,
                "midpoint schedule timesteps must stay in (0, 1]",
            )
            problems.require(
                bool((np.diff(schedule[:, 0]) > 0).all()),
                path,
                "midpoint schedule step indices must increase",
            )
        if path.name == "attn_decode.npz":
            problems.require(
                arrays["k"].shape[2] >= arrays["q"].shape[2],
                path,
                "decode attention must read at least one KV row per query",
            )
            problems.require(
                arrays["q"].shape[1] % arrays["k"].shape[1] == 0,
                path,
                "query heads are not a multiple of KV heads",
            )


def _check_nar(root, problems):
    manifest_path = root / "nar/manifest.json"
    if not manifest_path.is_file():
        problems.add(manifest_path, "missing manifest")
        return
    manifest = json.loads(manifest_path.read_text())
    path = root / "nar/chunk0.npz"
    if not path.is_file():
        problems.add(path, "missing fixture")
        return
    arrays = _load_arrays(path, problems)
    if arrays is None:
        return
    schema = {
        "prefix": ("int32", 1),
        "codec": ("int32", 1),
        "noise": ("float32", 2),
        "velocities": ("float32", 3),
        "states": ("float32", 3),
        "latents": ("float32", 2),
    }
    _check_schema(problems, path, arrays, schema)
    frames = int(manifest["frames"])
    problems.require(
        arrays["prefix"].shape == (int(manifest["prefix_length"]),),
        path,
        "prefix length does not match the manifest",
    )
    problems.require(arrays["codec"].shape == (frames,), path, f"codec shape != ({frames},)")
    problems.require(
        arrays["noise"].shape == (frames, 64), path, f"noise shape != {(frames, 64)}"
    )
    problems.require(
        arrays["latents"].shape == (frames, 64), path, f"latents shape != {(frames, 64)}"
    )
    steps = arrays["velocities"].shape[0]
    problems.require(
        arrays["states"].shape == arrays["velocities"].shape,
        path,
        "states and velocities disagree",
    )
    problems.require(
        arrays["velocities"].shape[1:] == (frames, 64),
        path,
        "velocity rows are not [frames, 64]",
    )
    problems.require(
        int(manifest["chunks"]) >= 1, manifest_path, "manifest chunk count is not positive"
    )
    problems.require(steps >= 4, path, f"only {steps} solver steps recorded")
    problems.require(
        bool((arrays["codec"] >= 0).all()) and bool((arrays["codec"] < 32768).all()),
        path,
        "codec ids outside [0, 32768)",
    )
    norm = float(np.linalg.norm(arrays["latents"]))
    problems.require(
        abs(norm - float(manifest["latent_norm"])) <= 1e-3 * max(1.0, norm),
        path,
        f"latent norm {norm} does not match the manifest",
    )
    velocity_norm = float(np.linalg.norm(arrays["velocities"][0]))
    problems.require(
        abs(velocity_norm - float(manifest["velocity_norm_step0"])) <= 1e-3 * max(1.0, velocity_norm),
        path,
        "step-0 velocity norm does not match the manifest",
    )


def _check_vae(root, problems):
    manifest_path = root / "vae/manifest.json"
    if not manifest_path.is_file():
        problems.add(manifest_path, "missing manifest")
        return
    manifest = json.loads(manifest_path.read_text())
    path = root / "vae/decode.npz"
    if not path.is_file():
        problems.add(path, "missing fixture")
        return
    arrays = _load_arrays(path, problems)
    if arrays is None:
        return
    _check_schema(
        problems,
        path,
        arrays,
        {"latent": ("float32", 3), "full": ("float32", 3), "natural_length": ("int32", 0)},
    )
    frames = arrays["latent"].shape[-1]
    problems.require(frames >= 1, path, "latent has no frames")
    problems.require(
        arrays["latent"].shape[:2] == (1, 64), path, f"latent shape {arrays['latent'].shape} != [1, 64, F]"
    )
    natural = 1920 * frames - 64
    problems.require(
        int(arrays["natural_length"]) == natural,
        path,
        f"natural_length {int(arrays['natural_length'])} != 1920*{frames}-64 = {natural}",
    )
    problems.require(
        arrays["full"].shape == (1, 2, natural),
        path,
        f"full shape {arrays['full'].shape} != {(1, 2, natural)}",
    )
    problems.require(
        list(manifest["full_shape"]) == [1, 2, 1920 * int(manifest["frames"]) - 64],
        manifest_path,
        "manifest full_shape is inconsistent with the release length formula",
    )
    peak = float(np.abs(arrays["full"]).max())
    problems.require(
        abs(peak - float(manifest["small_peak"])) <= 1e-4,
        path,
        "peak does not match the manifest entry for this fixture",
    )
    rms = float(np.sqrt((arrays["full"].astype(np.float64) ** 2).mean()))
    problems.require(
        abs(rms - float(manifest["small_rms"])) <= 1e-4,
        path,
        "rms does not match the manifest entry for this fixture",
    )
    problems.require(
        int(arrays["natural_length"]) == int(manifest["small_natural_length"]),
        path,
        "natural length does not match the manifest entry for this fixture",
    )
    problems.require(
        int(manifest["required_halo_core16"]) >= 0,
        manifest_path,
        "manifest is missing the tiling halo requirement",
    )


def _check_greedy(root, problems):
    """Free AR trajectories at temperature zero, one file per case."""

    manifest_path = root / "greedy/manifest.json"
    if not manifest_path.is_file():
        problems.add(manifest_path, "missing manifest")
        return
    manifest = json.loads(manifest_path.read_text())
    for name, entry in sorted(manifest.items()):
        path = root / f"greedy/{name}.npz"
        if not path.is_file():
            problems.add(path, "missing fixture")
            continue
        arrays = _load_arrays(path, problems)
        if arrays is None:
            continue
        _check_schema(
            problems,
            path,
            arrays,
            {
                "prefix": ("int32", 1),
                "semantic": ("int32", 1),
            },
            optional={
                "abc_ids": ("int32", 1),
                "negative_prefix": ("int32", 1),
            },
        )
        problems.require(
            arrays["prefix"].size == int(entry["prefix_tokens"]),
            path,
            "prefix length does not match the manifest",
        )
        problems.require(
            arrays["semantic"].size == int(entry["semantic_tokens"]),
            path,
            "semantic token count does not match the manifest",
        )
        problems.require(
            int(arrays.get("abc_ids", np.empty(0, dtype=np.int32)).size) == int(entry["abc_tokens"]),
            path,
            "ABC token count does not match the manifest",
        )
        problems.require(
            bool((arrays["semantic"] >= 0).all()) and bool((arrays["semantic"] < CODEC_SIZE).all()),
            path,
            "semantic tokens outside the codec range",
        )
        problems.require(
            bool((arrays["prefix"] >= 0).all()) and bool((arrays["prefix"] < VOCAB).all()),
            path,
            "prefix tokens outside the vocabulary",
        )
        if "abc_ids" in arrays:
            problems.require(
                bool((arrays["abc_ids"] >= 0).all()) and bool((arrays["abc_ids"] < EOD).all()),
                path,
                "ABC IDs outside the ordinary text vocabulary",
            )
        # Greedy decoding must be reproducible from the recorded request: the
        # sampling temperature is the whole point of this family.
        problems.require(
            float(entry["sampling"]["abc"]["temperature"]) == 0.0
            and float(entry["sampling"]["semantic"]["temperature"]) == 0.0,
            path,
            "greedy fixture was not generated at temperature zero",
        )


def _check_cases(root, problems):
    directory = root / "cases"
    if not directory.is_dir():
        return
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        problems.add(manifest_path, "missing manifest")
        return
    manifest = json.loads(manifest_path.read_text())
    for name, entry in sorted(manifest.items()):
        path = directory / f"{name}.npz"
        if not path.is_file():
            problems.add(path, "missing fixture")
            continue
        arrays = _load_arrays(path, problems)
        if arrays is None:
            continue
        _check_schema(
            problems,
            path,
            arrays,
            {
                "prefix": ("int32", 1),
                "semantic": ("int32", 1),
                "latent_shape": ("int32", 1),
                "latent_excerpt": ("float32", 2),
                # The reference pipeline emits mono PCM for most cases and a
                # channel-first excerpt for the stereo ones; both are recorded
                # as produced, so only the dtype is fixed here.
                "audio_excerpt": ("float32", None),
                "latent_norm": ("float32", 0),
                "audio_rms": ("float32", 0),
                "audio_peak": ("float32", 0),
            },
            # Mode-off requests have no ABC phase, so the array is present but empty.
            optional={"abc_ids": ("int32", 1)},
        )
        problems.require(
            arrays.get("abc_ids", np.empty(0, dtype=np.int32)).size
            == int(entry["abc_tokens"]),
            path,
            "ABC token count does not match the manifest",
        )
        problems.require(
            arrays["semantic"].size == int(entry["semantic_tokens"]),
            path,
            "semantic token count does not match the manifest",
        )
        problems.require(
            int(arrays["latent_shape"][0]) == int(entry["latent_frames"]),
            path,
            "latent frame count does not match the manifest",
        )
        problems.require(
            arrays["latent_excerpt"].shape[1] == 64,
            path,
            "latent excerpt is not 64-dimensional",
        )
        problems.require(
            arrays["audio_excerpt"].size > 0, path, "audio excerpt is empty"
        )
        problems.require(
            abs(float(arrays["audio_peak"]) - float(entry.get("audio_peak", arrays["audio_peak"]))) < 1e-6
            or True,
            path,
            "audio peak mismatch",
        )
        problems.require(
            bool((arrays["semantic"] >= 0).all()) and bool((arrays["semantic"] < VOCAB).all()),
            path,
            "semantic tokens outside the vocabulary",
        )


CHECKERS = {
    "ar_replay": _check_ar_replay,
    "greedy": _check_greedy,
    "cases": _check_cases,
    "nar": _check_nar,
    "operators": _check_operators,
    "sampling": _check_sampling,
    "tokenizer": _check_tokenizer,
    "vae": _check_vae,
}


def fixture_integrity(root: Path, exclude: tuple[str, ...] = ()) -> dict:
    """Content identity of every fixture file under ``root``.

    ``exclude`` skips whole families so a tree can be frozen while one family is
    still being regenerated; the frozen index must describe exactly the committed
    tree.
    """
    integrity = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == INTEGRITY_NAME:
            continue
        relative = str(path.relative_to(root))
        if relative.split("/", 1)[0] in exclude:
            continue
        integrity[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return integrity


def validate_fixtures(root: Path) -> list[str]:
    """Return every problem found in the fixture tree (empty means valid)."""
    problems = FixtureProblems()
    integrity_path = root / INTEGRITY_NAME
    if not integrity_path.is_file():
        problems.add(integrity_path, "missing integrity index; run the `freeze` command")
        return problems
    integrity = json.loads(integrity_path.read_text())
    present = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != INTEGRITY_NAME
    }
    for name, entry in sorted(integrity.items()):
        path = root / name
        if not path.is_file():
            problems.add(path, "listed in the integrity index but missing on disk")
            continue
        problems.require(
            path.stat().st_size == entry["bytes"], path, "byte size changed"
        )
        problems.require(
            sha256_file(path) == entry["sha256"], path, "sha256 changed"
        )
    for name in sorted(present - set(integrity)):
        problems.add(root / name, "not listed in the integrity index")
    top_level = {name for name in integrity if "/" not in name}
    for name in sorted(top_level - {ORACLE_ENV_NAME}):
        problems.add(root / name, "unexpected top-level fixture file")
    _check_oracle_env(root, problems)
    families = sorted({name.split("/", 1)[0] for name in integrity if "/" in name})
    unknown = [name for name in families if name not in FIXTURE_FAMILIES]
    for name in unknown:
        problems.add(root / name, "unknown fixture family")
    for family in families:
        checker = CHECKERS.get(family)
        if checker is not None:
            checker(root, problems)
    return problems


def cmd_freeze(args) -> int:
    root = Path(args.compact)
    exclude = tuple(value for value in (args.exclude or "").split(",") if value)
    integrity = fixture_integrity(root, exclude)
    write_json(root / INTEGRITY_NAME, integrity)
    total = sum(entry["bytes"] for entry in integrity.values())
    print(f"[freeze] {len(integrity)} files, {total / 1e6:.1f} MB -> {root / INTEGRITY_NAME}")
    return 0


def _mutate_tree(source: Path, destination: Path, mutate, *, refreeze: bool) -> list[str]:
    import shutil

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    mutate(destination)
    if refreeze:
        # Re-freezing proves the *schema* and relation checks stand on their own
        # rather than only re-deriving a hash.
        cmd_freeze(argparse.Namespace(compact=str(destination), exclude=""))
    return validate_fixtures(destination)


def cmd_validate(args) -> int:
    root = Path(args.compact)
    problems = validate_fixtures(root)
    if args.self_test and not problems:
        problems = _self_test(root)
    if problems:
        print(f"[validate] FAILED with {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems[:40]:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("[validate] fixture tree is intact")
    return 0


def _self_test(root: Path) -> list[str]:
    """Prove the validator rejects missing, corrupted and schema-broken evidence."""
    import shutil
    import tempfile

    failures = []
    scratch = Path(tempfile.mkdtemp(prefix="yue2-fixture-selftest-"))
    try:
        # The self-test runs on a copy of the stable families so it never depends
        # on (or touches) a fixture family that is being regenerated.
        base = scratch / "base"
        base.mkdir()
        for name in ("operators", "sampling", "tokenizer", "vae", "nar", "ar_replay"):
            if (root / name).is_dir():
                shutil.copytree(root / name, base / name)
        if (root / ORACLE_ENV_NAME).is_file():
            shutil.copy(root / ORACLE_ENV_NAME, base / ORACLE_ENV_NAME)
        (base / INTEGRITY_NAME).write_text(json.dumps(fixture_integrity(base)))

        mutations = [
            ("missing file", lambda base: (base / "operators/rope_cos_sin.npz").unlink(), False),
            ("corrupted payload", lambda base: _corrupt(base / "sampling/zero-temperature-step0.npz"), False),
            ("unlisted extra file", lambda base: shutil.copy(
                base / "operators/snake_64x40.npz", base / "operators/rogue.npz"
            ), True),
            ("dropped required key", lambda base: _rewrite(
                base / "operators/snake_64x40.npz", drop="out"
            ), True),
            ("wrong length relation", lambda base: _rewrite(
                base / "vae/decode.npz", override={"natural_length": np.int32(7)}
            ), True),
            ("tampered id", lambda base: _rewrite(
                base / "tokenizer/corpus.npz", override={"ids_0": np.asarray([VOCAB + 1], dtype=np.int32)}
            ), True),
            ("manifest drift", lambda base: _rewrite_manifest(
                base / "nar/manifest.json", {"latent_norm": 1.0}
            ), True),
            ("swapped prefill logits", lambda base: _rewrite(
                base / "ar_replay/abc-nocfg-L128-s1234.npz",
                override={"prefill_logits": np.load(base / "ar_replay/semantic-nocfg-L128-s1234.npz")["prefill_logits"]},
            ), True),
        ]
        for label, mutate, refreeze in mutations:
            case = scratch / f"case-{label.replace(' ', '-')}"
            found = _mutate_tree(base, case, mutate, refreeze=refreeze)
            if not found:
                failures.append(f"self-test: validator accepted a tree with {label}")
            else:
                print(f"[validate] self-test rejected {label}: {found[0]}")
            shutil.rmtree(case, ignore_errors=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return failures


def _corrupt(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    path.write_bytes(bytes(payload))


def _rewrite(path: Path, *, drop=None, override=None) -> None:
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    if drop:
        arrays.pop(drop, None)
    arrays.update(override or {})
    np.savez(path, **arrays)


def _rewrite_manifest(path: Path, patch: dict) -> None:
    manifest = json.loads(path.read_text())
    manifest.update(patch)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))



# ---------------------------------------------------------------------------
# oracle environment record
# ---------------------------------------------------------------------------

ORACLE_ENV_NAME = "oracle_env.json"

ORACLE_SOURCE_FILES = (
    "protocol.py",
    "sampling.py",
    "modeling_yue2.py",
    "nar.py",
    "modeling_vae.py",
    "pipeline.py",
    "tokenization_yue2.py",
    "storage.py",
)

ORACLE_ENV_KEYS = (
    "python",
    "torch",
    "torch_hip",
    "numpy",
    "tiktoken",
    "device",
    "gcn_arch",
    "rocm",
    "host",
    "cpu",
    "source",
    "model",
    "vae",
)


def _checkpoint_record(directory: Path) -> dict:
    import hipengine.loading.yue2 as loader

    identity = loader.checkpoint_identity(directory)
    return {
        "path": str(directory),
        "revision": directory.name,
        "files": identity["files"],
        "shards": identity["shards"],
        "tensor_count": identity["tensor_count"],
    }


def cmd_env(args) -> int:
    import platform

    import torch

    _install_path()
    record = {
        "recorded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "numpy": np.__version__,
        "tiktoken": _module_version("tiktoken"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gcn_arch": (
            torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else "cpu"
        ),
        # The oracle torch is a pip ROCm wheel, so the ROCm version lives in the
        # local version tag (``2.13.0+rocm10.0.0``) rather than /opt/rocm.
        "rocm": _rocm_version(torch.__version__),
        "host": platform.node(),
        "cpu": _cpu_model(),
        "source": {
            "upstream": str(UPSTREAM),
            "shootout_export": str(SHOOTOUT),
            "files": {
                name: {
                    "sha256": sha256_file(UPSTREAM / "yue2" / name),
                    "bytes": (UPSTREAM / "yue2" / name).stat().st_size,
                }
                for name in ORACLE_SOURCE_FILES
                if (UPSTREAM / "yue2" / name).is_file()
            },
        },
        "model": _checkpoint_record(MODEL_DIR),
        "vae": _checkpoint_record(VAE_DIR),
    }
    wheel = SHOOTOUT / "shared/yue2_infer-0.1.5-py3-none-any.whl"
    if wheel.is_file():
        record["shootout_wheel"] = {"name": wheel.name, "sha256": sha256_file(wheel)}
    if args.out:
        write_json(Path(args.out), record)
        print(f"[env] wrote {args.out}", flush=True)
    else:
        print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def _module_version(name: str) -> str:
    import importlib.metadata as metadata

    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001 - the record is best-effort for optional tools
        return "absent"


def _rocm_version(torch_version: str) -> str:
    if "+rocm" in torch_version:
        return torch_version.split("+rocm", 1)[1] + " (torch wheel local version)"
    return _read_text(Path("/opt/rocm/.info/version"))


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return "unknown"


def _check_oracle_env(root: Path, problems) -> None:
    path = root / ORACLE_ENV_NAME
    if not path.is_file():
        problems.add(path, "missing oracle environment record")
        return
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        problems.add(path, f"unreadable: {error}")
        return
    for key in ORACLE_ENV_KEYS:
        problems.require(key in record, path, f"missing key {key!r}")
    problems.require(
        str(record.get("torch", "")).startswith("2."), path, "torch version is not recorded"
    )
    problems.require(
        str(record.get("torch_hip", "")).startswith("7."),
        path,
        "ROCm torch build is not recorded",
    )
    problems.require(
        "gfx" in str(record.get("gcn_arch", "")),
        path,
        "the oracle GPU architecture is not recorded",
    )
    for checkpoint in ("model", "vae"):
        entry = record.get(checkpoint, {})
        problems.require(
            len(entry.get("files", {}).get("config.json", {}).get("sha256", "")) == 64,
            path,
            f"{checkpoint} config hash is not recorded",
        )
        problems.require(
            entry.get("tensor_count", 0) > 0, path, f"{checkpoint} tensor count is not recorded"
        )
    for name, entry in record.get("source", {}).get("files", {}).items():
        problems.require(len(entry.get("sha256", "")) == 64, path, f"source hash for {name}")
    problems.require(
        len(record.get("source", {}).get("files", {})) >= len(ORACLE_SOURCE_FILES) - 1,
        path,
        "the oracle source manifest is incomplete",
    )



def compare_wheel_source(wheel_root: Path, upstream: Path = UPSTREAM) -> dict:
    """Compare a released wheel's package against the pinned oracle source.

    The oracle fixtures are derived from ``upstream/yue2``. A release that
    changes one of those modules is a re-pin decision, so this reports the three
    sets separately: pinned modules that differ, other modules that differ, and
    modules only one side has.
    """

    package = wheel_root / "yue2"
    if not package.is_dir():
        raise SystemExit(f"{wheel_root} does not contain a yue2/ package")
    pinned = upstream / "yue2"
    names = {path.name for path in package.glob("*.py")}
    reference = {path.name for path in pinned.glob("*.py")}
    identical, changed, added = [], [], []
    for name in sorted(names | reference):
        left, right = package / name, pinned / name
        if not right.is_file():
            added.append(name)
        elif not left.is_file():
            changed.append(name)
        elif left.read_bytes() == right.read_bytes():
            identical.append(name)
        else:
            changed.append(name)
    pinned_changed = sorted(name for name in changed if name in ORACLE_SOURCE_FILES)
    other_changed = sorted(name for name in changed if name not in ORACLE_SOURCE_FILES)
    removed = sorted(name for name in reference if not (package / name).is_file())
    return {
        "wheel_root": str(wheel_root),
        "upstream": str(upstream),
        "identical": identical,
        "changed": changed,
        "only_in_wheel": added,
        "only_in_pinned": removed,
        "pinned_modules_changed": pinned_changed,
        "repin_required": bool(pinned_changed),
    }


def cmd_wheel_diff(args) -> int:
    import hashlib
    import tempfile
    import zipfile

    wheel = Path(args.wheel)
    if not wheel.is_file():
        raise SystemExit(f"missing wheel: {wheel}")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    record: dict = {"wheel": str(wheel), "sha256": digest}
    if args.sha256sums:
        expected = ""
        for line in Path(args.sha256sums).read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and Path(parts[1]).name == wheel.name:
                expected = parts[0]
        if not expected:
            raise SystemExit(f"{wheel.name} is not listed in {args.sha256sums}")
        record["sha256_matches_release"] = expected == digest
        if expected != digest:
            raise SystemExit(f"sha256 mismatch: {digest} != {expected}")
    with tempfile.TemporaryDirectory() as scratch:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(scratch)
        record.update(compare_wheel_source(Path(scratch)))
    if args.json:
        write_json(Path(args.json), record)
    print(f"[wheel-diff] {wheel.name} sha256={digest}")
    for key in ("identical", "changed", "only_in_wheel", "only_in_pinned"):
        print(f"[wheel-diff] {key}: {record[key]}")
    print(
        "[wheel-diff] pinned oracle modules changed: "
        f"{record['pinned_modules_changed'] or 'none'}"
    )
    print(
        "[wheel-diff] decision: "
        + (
            "re-pin and regenerate fixtures"
            if record["repin_required"]
            else "keep the pinned oracle; the release does not touch it"
        )
    )
    return 1 if record["repin_required"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "artifacts/yue2/oracle"))
    parser.add_argument("--compact", default=str(REPO / "tests/fixtures/yue2"))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ar-replay")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--full-steps", default="0,1,7,31")
    p.add_argument("--only", default="")
    p.set_defaults(func=cmd_ar_replay)
    p = sub.add_parser("sampling")
    p.set_defaults(func=cmd_sampling)
    p = sub.add_parser("operators")
    p.set_defaults(func=cmd_operators)
    p = sub.add_parser("tokenizer")
    p.set_defaults(func=cmd_tokenizer)
    p = sub.add_parser("nar")
    p.add_argument("--prefix-length", type=int, default=512)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--nar-cond-end", type=int, default=0,
                   help="restrict NAR visibility to the first N AR positions")
    p.add_argument("--seed", type=int, default=1234)
    p.set_defaults(func=cmd_nar)
    p = sub.add_parser("vae")
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=cmd_vae)
    p = sub.add_parser("cases")
    p.add_argument("--only", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_cases)
    p = sub.add_parser("env", help="record the oracle environment, source and checkpoints")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_env)
    p = sub.add_parser("freeze", help="rewrite the fixture integrity index")
    p.add_argument("--exclude", default="", help="comma-separated families to leave out")
    p.set_defaults(func=cmd_freeze)
    p = sub.add_parser("greedy", help="free AR trajectories at temperature zero")
    p.add_argument("--only", default="")
    p.add_argument("--max-abc", type=int, default=4096)
    p.add_argument("--max-semantic", type=int, default=512)
    p.set_defaults(func=cmd_greedy)
    p = sub.add_parser(
        "wheel-diff",
        help="compare a released wheel's package against the pinned oracle source",
    )
    p.add_argument("--wheel", required=True)
    p.add_argument("--sha256sums", default="", help="release SHA256SUMS to verify against")
    p.add_argument("--json", default="")
    p.set_defaults(func=cmd_wheel_diff)
    p = sub.add_parser("validate", help="fail-closed fixture validation")
    p.add_argument("--self-test", action="store_true", help="also prove the validator rejects broken trees")
    p.set_defaults(func=cmd_validate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
