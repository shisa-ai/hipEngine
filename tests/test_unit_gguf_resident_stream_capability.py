"""Resident slot views cannot run scalar model streams concurrently."""
import pytest

from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator


def test_explicit_resident_scalar_stream_request_names_shared_buffer_capability():
    owner = object.__new__(Qwen35GGUFBringupGenerator)
    owner.ar_stream_decode = True
    owner.ar_packed_decode = False
    with pytest.raises(NotImplementedError, match="disjoint execution and sample buffers"):
        owner.create_resident_model_runner(capacity=3)
