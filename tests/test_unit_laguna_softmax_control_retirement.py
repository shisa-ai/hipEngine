"""Retired softmax session controls leave package selection and primitives intact."""
import inspect
from pathlib import Path

import pytest
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession


def test_softmax_session_control_is_removed():
    name = 'prefill_attention_hipblaslt_wave_rows_softmax'
    assert name not in inspect.signature(LagunaGGUFResidentSession).parameters
    assert not hasattr(LagunaGGUFResidentSession, 'set_' + name)
    for value in (False, True):
        with pytest.raises(TypeError, match='unexpected keyword'):
            LagunaGGUFResidentSession(**{name: value})


def test_obsolete_softmax_comparison_harness_is_removed():
    assert not (Path(__file__).resolve().parents[1] / 'scripts/laguna_attention_softmax_ab.py').exists()
