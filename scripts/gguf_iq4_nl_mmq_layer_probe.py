"""Localize the IQ4_NL dense-MMQ numerics cost to a layer or shape.

CAUTION on how to read this: the probe reports divergence from the *current
default*, which is NOT the gate objective. A 2026-09-08 run attributed 71% of
that divergence to the three early ffn_down tensors, but routing only the other
four measured no better against the teacher (mean 0.0013632 vs 0.0013457 for
all seven). Use this to understand propagation; bisect retention decisions on
the teacher gate itself. See docs/REFACTOR.md "IQ4_NL dense MMQ".


Runs the same prompt twice in one process - once with IQ4_NL held out (the
shipped default) and once with it routed - capturing every layer's output
hidden state, and reports where the two first diverge and how that grows.

The IQ4_NL tensors sit at layers 1/2/3 (ffn_down), 21 (attn_qkv) and
27/27/50 (ffn_gate/up), so the divergence onset says which of those matters.
"""
import sys
from pathlib import Path

import numpy as np

import hipengine.kernels.hip_gfx1151 as be
import hipengine.runtime.gguf_linear as gl
from hipengine.loading.gguf import load_gguf_index
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import load_prompt_rows

MODEL = '/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf'
PROMPT_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 4
# IQ4_NL is routed by default now, so the "off" arm removes it and the "on"
# arm restores it; the probe still reports divergence between the two.
BASE_POLICY = {k: v for k, v in be.GGUF_IQ_DENSE_MMQ_PREFILL_POLICY.items()
               if k != 'gguf_iq4_nl'}

rows = []
for filename in ('mtpbench-code-general-ja.jsonl', 'gdn-prefill-category-heldouts.jsonl'):
    rows.extend(load_prompt_rows(Path('benchmarks/prompts') / filename))
tok = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(MODEL))
row = rows[PROMPT_INDEX]
p = list(build_chat_prompt(tok, row['prompt'], reasoning='off'))
ids = (p * ((512 + len(p) - 1) // len(p)))[:512]
print(f'prompt {PROMPT_INDEX} = {row["id"]}, {len(ids)} tokens', flush=True)


def run(session, enable_iq4_nl, layers, shapes=None):
    entry = {
        "min_rows": 8,
        "max_rows": 131072,
        "variant": "dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out",
    }
    if shapes is not None:
        entry['shapes'] = frozenset(shapes)
    be.GGUF_IQ_DENSE_MMQ_PREFILL_POLICY = (
        {**BASE_POLICY, 'gguf_iq4_nl': entry}
        if enable_iq4_nl else dict(BASE_POLICY))
    gl._DISPATCH_RESOLVE_CACHE.clear()   # policy is not part of the cache key
    session.reset()
    session._last_layer_output_hidden.clear()
    r = session.prefill(ids, use_bulk=True, return_logits=True,
                        capture_layer_output_hidden=layers)
    return ({k: v.astype(np.float64) for k, v in session._last_layer_output_hidden.items()},
            np.asarray(r.logits, dtype=np.float64))


with Qwen35GGUFResidentSession(
        MODEL, backend='hip_gfx1151', max_sequence_length=len(ids) + 16,
        use_wmma_prefill=True, use_gemv_decode=True,
        compiler_version=Path('/tmp/ud-hipcc-version.txt').read_text()) as s:
    n_layers = len(s.runner.weights.config.layer_types)
    layers = list(range(n_layers))
    off, off_logits = run(s, False, layers)
    arms = {
        'all IQ4_NL': None,
        'only (5120,17408) [ffn_down L1-3]': [(17408, 5120)],
        'exclude (5120,17408)': [(5120, 10240), (5120, 17408)],
    }
    # dispatch shapes are keyed (in_features, out_features)
    arms = {
        'all IQ4_NL': None,
        'only ffn_down L1/2/3': [(17408, 5120)],
        'only attn_qkv L21 + ffn_gate/up L27,50': [(5120, 10240), (5120, 17408)],
    }
    results = {}
    for name, shapes in arms.items():
        _, lg = run(s, True, [], shapes)
        results[name] = float(np.linalg.norm(off_logits - lg)
                              / max(np.linalg.norm(off_logits), 1e-30))
        print(f'  {name:42s} logits rel diff {results[name]:.4e}', flush=True)
    on, on_logits = run(s, True, layers)

be.GGUF_IQ_DENSE_MMQ_PREFILL_POLICY = dict(BASE_POLICY)
print(f'\ncaptured {len(off)} layers\n')
print(f'{"layer":>6s}{"rel diff":>12s}{"growth":>9s}   note')
prev = None
iq4nl = {1: 'ffn_down', 2: 'ffn_down', 3: 'ffn_down', 21: 'attn_qkv',
         27: 'ffn_gate+up', 50: 'ffn_gate'}
for layer in layers:
    a, b = off[layer], on[layer]
    rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
    growth = '' if prev is None or prev == 0 else f'{rel/prev:8.2f}x'
    note = f'<-- IQ4_NL {iq4nl[layer]}' if layer in iq4nl else ''
    if layer < 8 or layer in iq4nl or layer % 8 == 0 or layer == n_layers - 1:
        print(f'{layer:6d}{rel:12.3e}{growth:>9s}   {note}')
    prev = rel
lr = float(np.linalg.norm(off_logits - on_logits) / max(np.linalg.norm(off_logits), 1e-30))
print(f'\nfinal logits relative difference: {lr:.3e}')
