#!/usr/bin/env python3
"""Muse Glimmer verify cost vs block rows at a given context (diagnostic).

Times ``verify_with_taps`` for 1..N rows against one AR ``decode_step`` at the
same live length. Perfect amortization would keep verify(N) close to
decode_step; the gap splits into projections (rows x weight reads not
shared) and attention (rows x K/V reads not shared).

``--context`` above the prompt sets the live KV length synthetically (the
cache rows past the prompt are uninitialized), so long-context timing needs no
long prefill. Timing only: logits at a synthetic length are meaningless.

  .venv/bin/python scripts/muse_glimmer_verify_amortization.py --context 131072 --rows 1,2,3,5,8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_DIR = Path("/models/gguf/Muse-Glimmer-30B-GGUF")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--context", type=int, default=0, help="synthetic live length (0 = prompt length)")
    parser.add_argument("--rows", default="1,2,3,5,8,15")
    parser.add_argument("--iters", type=int, default=8)
    args = parser.parse_args(argv)

    from hipengine.generation.muse_glimmer_gguf import make_muse_glimmer_gguf_generator_gfx1151
    from hipengine.kernels.cpu_reference.muse_glimmer_dflash import MuseGlimmerDFlashConfig
    from hipengine.loading.gguf import GGUFReader, scan_gguf
    from hipengine.models.muse_glimmer import MUSE_GLIMMER_GGUF

    target = args.model_dir / "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
    draft = args.model_dir / "dflash-Muse-Glimmer-30B-Q4_K_M.gguf"
    max_rows = max(int(r) for r in args.rows.split(","))
    max_context = max(2048, args.context + max_rows + args.iters + 64)
    gen = make_muse_glimmer_gguf_generator_gfx1151(
        model_path=target, weight_index=scan_gguf(target), model_plugin=MUSE_GLIMMER_GGUF,
        max_sequence_length=max_context,
    )
    try:
        session = gen._ensure_session()
        reader = GGUFReader(draft)
        dcfg = MuseGlimmerDFlashConfig.from_gguf_info(
            reader.info, mask_token_id=int(reader.info.metadata["tokenizer.ggml.mask_token_id"])
        )
        ids = gen.tokenize("Write a Python function to merge overlapping intervals. Return only code.")
        session.reset()
        session.prefill(ids)
        base = max(len(ids), args.context)
        session._length = base  # synthetic live length (timing only)
        tok = ids[-1]

        def timeit(fn) -> float:
            fn()
            session.runtime.device_synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                fn()
            session.runtime.device_synchronize()
            return (time.perf_counter() - t0) / args.iters * 1e3

        def decode() -> None:
            session.truncate(base)
            session.decode_step(tok)

        ar = timeit(decode)
        print(f"live length {base}: decode_step {ar:.2f} ms")
        for rows in (int(r) for r in args.rows.split(",")):
            def verify(rows: int = rows) -> None:
                session.truncate(base)
                session.verify_with_taps([tok] * rows, dcfg.target_layers)

            ms = timeit(verify)
            print(f"  verify {rows:2d} rows: {ms:7.2f} ms  = {ms / ar:.2f}x decode_step  ({ms / rows:.2f} ms/row)")
    finally:
        gen.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
