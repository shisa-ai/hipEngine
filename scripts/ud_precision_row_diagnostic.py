"""CPU-only inspection of saved UD rate-limiter/control captures.

No runtime routing or qualification decisions. Run from the repository root
with PYTHONPATH=. and the original /tmp captures available.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from hipengine.loading.gguf import GGUFReader, dequantize_gguf_data


def logsoft(x):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 1 and np.isfinite(x).all()
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def distribution_row(teacher, candidate, ids):
    a, b = logsoft(teacher), logsoft(candidate)
    p, q = np.exp(a), np.exp(b)
    mask = np.ones(len(p), dtype=bool)
    mask[ids] = False
    # Partition KL into selected tokens plus one complement bucket. The
    # remainder is the complement's conditional-distribution contribution.
    full = float(p @ (a - b))
    grouped = float(p[ids] @ (a[ids] - b[ids]) +
                    p[mask].sum() * np.log(p[mask].sum() / q[mask].sum()))
    assert grouped <= full + 1e-12
    return dict(kl=full, grouped_kl=grouped, remainder_kl=full-grouped,
                teacher_probabilities=p[ids].tolist(),
                candidate_probabilities=q[ids].tolist(),
                teacher_margin=float(a[ids[0]]-a[ids[1]]),
                candidate_margin=float(b[ids[0]]-b[ids[1]]),
                top1_match=bool(p.argmax() == q.argmax()))


def bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000).view(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    hashes = {}

    def read(path):
        path = Path(path)
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return data

    def js(path):
        return json.loads(read(path))

    def archive(path):
        read(path)
        return np.load(path)

    def logits(path):
        return np.frombuffer(read(path), dtype='<f4').reshape(18, 9, 248320)

    manifest = js('/tmp/ud-layer-teacher-manifest.json')
    for path, expected in manifest['files'].items():
        read(path)
        assert hashes[path] == expected, path
    residual = js('benchmarks/results/2026-09-07-zbook-ud-residual-precision.json')
    for path, expected in residual['input_manifest'].items():
        read(path)
        assert hashes[path] == expected, path
    indices = [0, 4, 10, 17]
    base = archive('/tmp/ud-q56-context512-M-baseline.npz')
    raw = archive('/tmp/ud-q56-raw-context512-M.npz')
    for key in ('inputs', 'tokens'):
        np.testing.assert_array_equal(base[key], raw[key])
    wire = read('/tmp/ud-layer-localize.bin')
    assert wire[:12] == b'Q36Q' + struct.pack('<II', 1, 4)
    cursor = 12
    for index in indices:
        n, m = struct.unpack_from('<II', wire, cursor)
        cursor += 8
        assert (n, m) == (512, 9)
        for count, expected in [(n, base['inputs'][index]),
                                (m, base['tokens'][9*index:9*(index+1)])]:
            np.testing.assert_array_equal(np.frombuffer(wire, '<i4', count, cursor), expected)
            cursor += 4*count
    assert cursor == len(wire)
    teacher_f16 = logits('/tmp/ud-llama-context512-M.f32')
    teacher_bf16 = logits('/tmp/ud-llama-context512-M-bf16kv.f32')
    common = js('benchmarks/results/2026-09-07-zbook-ud-common-kv.json')
    assert hashes['/tmp/ud-llama-context512-M-bf16kv.f32'] == common['teacher_sha256']
    hip_logits = raw['logits'].reshape(18, 9, -1)
    instrumented = np.frombuffer(read('/tmp/ud-layer-localize-teacher.f32'), '<f4')
    np.testing.assert_array_equal(instrumented.view('u4'), teacher_f16[indices].ravel().view('u4'))
    hip = archive('/tmp/ud-hip-layer-localize.npz')
    np.testing.assert_array_equal(hip['indices'], indices)
    np.testing.assert_array_equal(hip['logits'].ravel().view('u4'), hip_logits[indices].ravel().view('u4'))
    data = read('/tmp/ud-layer-localize-teacher.f32.layers')
    records, cursor = {}, 0
    while cursor < len(data):
        p, pos, nb, ne = struct.unpack_from('<IIII', data, cursor)
        cursor += 16
        assert p < 4 and 511 <= pos <= 519 and 0 < nb < 64 and ne == 5120
        name = data[cursor:cursor+nb].decode('ascii')
        cursor += nb
        assert name.startswith('l_out-')
        layer = int(name[6:])
        key = (p, pos-511, layer)
        assert 0 <= layer < 64 and key not in records
        records[key] = np.frombuffer(data, '<f4', ne, cursor)
        cursor += 4*ne
    assert cursor == len(data) and len(records) == 4*9*64
    layers = np.stack([records[p, s, l] for p in range(4) for s in range(9)
                       for l in range(64)]).reshape(4, 9, 64, 5120)
    hip_layers = hip['layers'].reshape(layers.shape)
    assert np.isfinite(layers).all() and np.isfinite(hip_layers).all()
    f32 = archive('/tmp/ud-residual-precision-f32.npz')
    np.testing.assert_array_equal(f32['indices'], indices)
    f32_logits = f32['logits'].reshape(4, 9, -1)
    bf = archive('/tmp/ud-residual-precision-bf16.npz')
    np.testing.assert_array_equal(bf['logits'].ravel().view('u4'), hip_logits[indices].ravel().view('u4'))
    modes = [js('/tmp/ud-residual-precision-'+m+'.json') for m in ('bf16', 'f32')]
    for key in ('actual_resident_manifest', 'prompt_indices', 'rows_per_call',
                'kv_storage_dtype', 'token_embedding_precision', 'smoke'):
        assert modes[0][key] == modes[1][key], key
    reader = GGUFReader('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
    norm = np.asarray(reader.tensor_data('output_norm.weight'), dtype=np.float64)
    eps = reader.info.metadata['qwen35.attention.layer_norm_rms_epsilon']
    vocab = reader.info.metadata['tokenizer.ggml.tokens']
    rows = []
    for p, step in [(10, 0), (10, 1), (10, 2), (4, 2)]:
        local = indices.index(p)
        ids = np.argsort(teacher_bf16[p, step])[-2:][::-1]
        row = dict(prompt_index=p, step_after_prefill=step,
                   absolute_input_position=511+step, token_ids=ids.tolist(),
                   token_strings=[vocab[i] for i in ids], comparisons={})
        for name, teacher, candidate in [
            ('raw_vs_bf16kv', teacher_bf16[p, step], hip_logits[p, step]),
            ('raw_vs_f16kv', teacher_f16[p, step], hip_logits[p, step]),
            ('f32_residual_vs_bf16kv', teacher_bf16[p, step], f32_logits[local, step]),
        ]:
            row['comparisons'][name] = distribution_row(teacher, candidate, ids)
        np.testing.assert_allclose(row['comparisons']['f32_residual_vs_bf16kv']['kl'],
                                   residual['arms']['f32']['kl_by_position'][local*9+step], atol=1e-12)
        packed = np.ascontiguousarray(reader.tensor_data('output.weight')[ids])
        hashes[f'output.weight.rows.{ids.tolist()}'] = hashlib.sha256(packed.tobytes()).hexdigest()
        weights = dequantize_gguf_data(packed, reader.tensor_info('output.weight').ggml_type).astype(float)
        direction = weights[0] - weights[1]
        row['final_head_replay'] = {}
        for name, hidden in [('f16kv_teacher', layers[local, step, -1]),
                             ('hip', hip_layers[local, step, -1])]:
            x = hidden.astype(float)
            normalized = x / np.sqrt(np.mean(x*x)+eps) * norm
            row['final_head_replay'][name] = dict(
                f64_margin=float(direction @ normalized),
                bf16_normalized_margin=float(direction @ bf16(normalized)))
        delta = hip_layers[local, step].astype(float)-layers[local, step]
        row['relative_rms_by_layer_vs_f16kv'] = (
            np.linalg.norm(delta, axis=-1)/np.linalg.norm(layers[local, step].astype(float), axis=-1)).tolist()
        rows.append(row)
    hashes['output_norm.weight'] = hashlib.sha256(norm.tobytes()).hexdigest()
    # Cross-check the target in the saved K_S default capture. This teacher
    # used K_M histories: verify the entire consumed prefix for each selected
    # row instead of applying a whole-suite stale-row exclusion by assumption.
    ks_base = archive('/tmp/ud-q56-context512-S-baseline.npz')
    ks_raw = archive('/tmp/ud-context512-S-current-default.npz')
    ks_teacher = logits('/tmp/ud-llama-context512-S-bf16kv.f32')
    ks_rows = []
    for p, step in [(10, 1), (4, 2)]:
        for capture in (ks_base, ks_raw):
            np.testing.assert_array_equal(capture['inputs'][p], base['inputs'][p])
            np.testing.assert_array_equal(capture['tokens'][9*p:9*p+step],
                                          base['tokens'][9*p:9*p+step])
        ids = np.argsort(ks_teacher[p, step])[-2:][::-1]
        ks_rows.append(dict(prompt_index=p, step_after_prefill=step,
                            token_ids=ids.tolist(), comparisons={
            name: distribution_row(ks_teacher[p, step],
                                   capture['logits'].reshape(18, 9, -1)[p, step], ids)
            for name, capture in [('expanded', ks_base), ('raw', ks_raw)]}))
    read(__file__)
    result = dict(diagnostic_only=True, promotion_qualified=False,
                  capture_host='zbook', capture_hardware='Radeon 8060S / gfx1151',
                  model=str(reader.path), teacher_manifest=manifest, source_sha256=hashes,
                  rows=rows, ks_crosscheck=ks_rows, limitations=[
                      'Saved captures; no new GPU execution or current-default qualification.',
                      'Layer captures use the F16-KV teacher, not the matched BF16-KV teacher.',
                      'No incoming recurrent/convolution/KV state capture; no causal layer attribution.',
                      'F64 output-head replay is a precision-controlled diagnostic, not Q8_1 emulation.',
                      'Model identity inherited from capture manifest; selected live tensor bytes hashed.',
                      'Original layer capture did not record prior resident signatures.'])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
