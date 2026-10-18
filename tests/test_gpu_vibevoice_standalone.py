"""Offline single-file inference versus the existing HF-frontend Q4 path."""
import ctypes
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest

GGUF = Path(os.environ.get('VIBEVOICE_STANDALONE_GGUF','/nonexistent'))
HF = Path(os.environ.get('VIBEVOICE_REFERENCE_HF','/nonexistent'))
AUDIO = Path(os.environ.get('VIBEVOICE_REFERENCE_AUDIO','/tmp/librispeech-clean-spread'))
try:
    ctypes.CDLL('libamdhip64.so')
    HIP = True
except OSError:
    HIP = False
pytestmark = pytest.mark.skipif(not HIP or not GGUF.is_file() or not HF.is_dir(),
                               reason='requires HIP and explicit standalone/reference artifacts')


def test_offline_transcription_matches_hf_frontend(monkeypatch):
    from hipengine.loading.vibevoice_asr import load_vibevoice_encoder, load_vibevoice_connector
    from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime
    from hipengine.generation.vibevoice_asr_q4 import VibeVoiceASRQ4Generator
    specs = {k: load_vibevoice_encoder(HF,k) for k in ('acoustic','semantic')}
    reference = VibevoiceFrontendRuntime(*specs['acoustic'], *specs['semantic'],
        load_vibevoice_connector(HF,'acoustic'),load_vibevoice_connector(HF,'semantic'),frontend_variant='wmma')
    def denied(*args, **kwargs):
        raise AssertionError('standalone inference accessed the original checkpoint or network')
    import hipengine.loading.hf_cache as cache
    import hipengine.loading.vibevoice_asr as hf_loader
    import huggingface_hub
    monkeypatch.setattr(cache,'resolve_model_path',denied)
    monkeypatch.setattr(hf_loader,'resolve_model_path',denied)
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',denied)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',denied)
    import socket
    monkeypatch.setattr(socket.socket,'connect',denied)
    session = None
    rows = []
    try:
        session = VibeVoiceASRQ4Generator(GGUF,max_sequence_length=1024)
        standalone = session.frontend
        for name in ('1089-134686-0000','121-121726-0000'):
            raw = np.load(AUDIO / (name+'.f32.npy'))
            actual = session.transcribe(raw)
            session.frontend = reference
            try:
                expected = session.transcribe(raw)
            finally:
                session.frontend = standalone
            assert actual == expected
            assert actual.finish_reason == 'eos'
            assert actual.segments is not None
            rows.append({'clip':name,'tokens':list(actual.generated_token_ids),'text':actual.text,
                         'finish_reason':actual.finish_reason,'exact_vs_hf_frontend':True})
        assert 'torch' not in sys.modules
    finally:
        if session is not None:
            session.close()
            session.close()
        reference.close()
    if os.environ.get('VIBEVOICE_STANDALONE_REPORT'):
        Path(os.environ['VIBEVOICE_STANDALONE_REPORT']).write_text(json.dumps(rows,indent=2)+'\n')
