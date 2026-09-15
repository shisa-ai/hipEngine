"""Versioned, hash-checked HF assets embedded in a VibeVoice GGUF.

Assets stay in memory at inference: no HF lookup, extraction or remote code.
The complete original tokenizer JSON preserves added tokens and normalization.
"""
import hashlib
import json
from pathlib import Path

ASSET_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json',
               'processor_config.json', 'generation_config.json', 'chat_template.jinja')
PREFIX = 'vibevoice.assets.'


def asset_metadata(snapshot):
    root = Path(snapshot)
    missing = [name for name in ASSET_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f'missing standalone assets: {missing}')
    assets = {name: (root / name).read_bytes().decode('utf-8') for name in ASSET_FILES}
    for name, value in assets.items():
        if name.endswith('.json'):
            json.loads(value)
    return {PREFIX + 'version': 1,
            PREFIX + 'manifest': json.dumps({name: hashlib.sha256(value.encode()).hexdigest()
                                             for name, value in assets.items()}, sort_keys=True),
            **{PREFIX + name: value for name, value in assets.items()}}


def read_assets(metadata):
    if metadata.get(PREFIX + 'version') != 1:
        raise ValueError('GGUF lacks supported standalone assets; repackage with vibevoice_asr_to_gguf.py')
    try:
        manifest = json.loads(metadata[PREFIX + 'manifest'])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid standalone asset manifest') from exc
    if set(manifest) != set(ASSET_FILES):
        raise ValueError('standalone asset manifest has missing or unexpected files')
    assets = {}
    for name in ASSET_FILES:
        value = metadata.get(PREFIX + name)
        if not isinstance(value, str) or hashlib.sha256(value.encode()).hexdigest() != manifest[name]:
            raise ValueError(f'standalone asset hash mismatch: {name}')
        if name.endswith('.json'):
            json.loads(value)
        assets[name] = value
    return assets
