#!/usr/bin/env python3
"""Verify standalone packaging against source GGUF and original HF frontend."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def equal(a, b):
    if dataclasses.is_dataclass(a):
        for field in dataclasses.fields(a):
            equal(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b, (a,b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--standalone', required=True, type=Path)
    p.add_argument('--hf', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    from hipengine.loading.gguf import GGUFReader, scan_gguf
    from hipengine.loading.vibevoice_assets import read_assets
    from hipengine.loading.vibevoice_frontend_gguf import load_gguf_frontend
    from hipengine.loading.vibevoice_asr import load_vibevoice_encoder, load_vibevoice_connector
    before, after = GGUFReader(args.source), GGUFReader(args.standalone)
    old, new = scan_gguf(args.source), scan_gguf(args.standalone)
    assert {t.name for t in old.tensors} == {t.name for t in new.tensors}
    for t in old.tensors:
        actual = after.tensor_info(t.name)
        assert (t.shape,t.ggml_type) == (actual.shape,actual.ggml_type)
        np.testing.assert_array_equal(before.tensor_data(t.name), after.tensor_data(t.name))
    assets = read_assets(new.metadata)
    for name, value in assets.items():
        assert value.encode() == (args.hf / name).read_bytes(), name
    specs, connectors = load_gguf_frontend(args.standalone)
    for kind in specs:
        equal(specs[kind], load_vibevoice_encoder(args.hf, kind))
        equal(connectors[kind], load_vibevoice_connector(args.hf,kind))
    report = dict(source_sha256=digest(args.source), standalone_sha256=digest(args.standalone),
                  standalone_bytes=args.standalone.stat().st_size,
                  tensors_bit_identical=len(old.tensors), embedded_assets_byte_identical=list(assets),
                  hf_frontend_tensors_and_specs='exact', torch_imported='torch' in sys.modules)
    assert not report['torch_imported']
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
