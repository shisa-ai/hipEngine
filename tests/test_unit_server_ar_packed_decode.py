"""Explicit packed AR control across the server's process boundary."""

import pytest

from hipengine import LLM
from hipengine.server.__main__ import build_parser
from hipengine.server import ServerConfig


@pytest.mark.parametrize("selected, option", [(False, "--no-ar-packed-decode"), (True, "--ar-packed-decode")])
def test_server_parser_and_config_preserve_explicit_packed_control(selected, option):
    args = build_parser().parse_args(["--model", "fake.gguf", option])
    config = ServerConfig(model=args.model, ar_packed_decode=args.ar_packed_decode)
    assert config.ar_packed_decode is selected
    llm = LLM("fake.gguf", ar_packed_decode=config.ar_packed_decode)
    assert llm.ar_packed_decode is selected


def test_server_packed_control_is_unset_by_default():
    assert build_parser().parse_args(["--model", "fake.gguf"]).ar_packed_decode is None
    assert LLM("fake.gguf").ar_packed_decode is None


def test_llm_rejects_non_boolean_packed_control():
    with pytest.raises(ValueError, match="ar_packed_decode"):
        LLM("fake.gguf", ar_packed_decode="0")
