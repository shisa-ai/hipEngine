"""Server transport for the MTP proposal vocabulary setting."""

import pytest

from hipengine.server import ServerConfig
from hipengine.server.__main__ import build_parser


@pytest.mark.parametrize("choice", ["auto", "off", "/tmp/map.json", ""])
def test_cli_preserves_explicit_hot_vocabulary(choice):
    args = build_parser().parse_args(["--model", "/tmp/model.gguf", "--mtp-hot-vocab", choice])
    assert args.mtp_hot_vocab == choice
    assert ServerConfig(model="/tmp/model.gguf", mtp_hot_vocab=choice).mtp_hot_vocab == choice


def test_omitted_cli_hot_vocabulary_keeps_compatibility_input(monkeypatch):
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_HOT_VOCAB", "off")
    args = build_parser().parse_args(["--model", "/tmp/model.gguf"])
    assert args.mtp_hot_vocab is None
