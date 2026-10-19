"""Unit: the Flash-Next MTP head is found beside its target and validated.

The target GGUF carries no NextN block, so ``hipengine serve --model <target>``
must locate the separately shipped head itself. Discovery searches by location,
nearest first, and admits a candidate only when its header passes the MTP GGUF
contract; these tests replace that header check so the search order is the
subject.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine import llm
from hipengine.loading import gguf
from hipengine.loading import qwen4_exp_mtp_gguf as mtp_gguf
from hipengine.models.qwen4_exp import QWEN4_EXP_GGUF


@pytest.fixture
def contract(monkeypatch):
    """Treat files named ``mtp-ok*`` as passing the MTP head contract."""

    monkeypatch.setattr(gguf, "load_gguf_index", lambda path: SimpleNamespace(path=Path(path)))
    monkeypatch.setattr(
        mtp_gguf,
        "validate_qwen4_exp_mtp_gguf",
        lambda infos: SimpleNamespace(passed=infos[0].path.name.startswith("mtp-ok")),
    )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _target(root: Path) -> Path:
    shard_dir = root / "unsloth-Flash-Next-UD-Q4_K_XL" / "UD-Q4_K_XL"
    for index in range(1, 3):
        _touch(shard_dir / f"Flash-Next-UD-Q4_K_XL-0000{index}-of-00002.gguf")
    return shard_dir


def test_sibling_mtp_directory_of_the_artifact_repo_is_found(tmp_path, contract) -> None:
    shard_dir = _target(tmp_path)
    head = _touch(tmp_path / "Flash-Next-MTP-Q8_0" / "mtp-ok-Flash-Next-Q8_0.gguf")
    _touch(tmp_path / "Flash-Next-mmproj" / "Flash-Next-BF16.gguf")
    assert QWEN4_EXP_GGUF.resolve_speculative_mtp_draft_model(str(shard_dir)) == head.resolve()
    shard = next(shard_dir.glob("*.gguf"))
    assert QWEN4_EXP_GGUF.resolve_speculative_mtp_draft_model(str(shard)) == head.resolve()


def test_nearest_location_wins_and_invalid_files_are_skipped(tmp_path, contract) -> None:
    shard_dir = _target(tmp_path)
    _touch(tmp_path / "Flash-Next-MTP-Q8_0" / "mtp-ok-far.gguf")
    _touch(shard_dir / "mtp-broken.gguf")
    near = _touch(shard_dir.parent / "MTP" / "mtp-ok-near.gguf")
    assert QWEN4_EXP_GGUF.resolve_speculative_mtp_draft_model(str(shard_dir)) == near.resolve()


def test_no_valid_head_is_not_an_error(tmp_path, contract) -> None:
    shard_dir = _target(tmp_path)
    _touch(tmp_path / "Other-MTP" / "mtp-broken.gguf")
    assert QWEN4_EXP_GGUF.resolve_speculative_mtp_draft_model(str(shard_dir)) is None
    assert QWEN4_EXP_GGUF.resolve_speculative_mtp_draft_model(str(tmp_path / "missing")) is None


class _Plugin:
    name = "plugin"

    def __init__(self, provider=None, head=None):
        self.speculative_mtp_provider = provider
        self._head = head
        self.discovered = 0

    def resolve_speculative_mtp_draft_model(self, model):
        self.discovered += 1
        return self._head


def _owner(plugin, *, provider=None, draft_model=None, mode=None):
    return llm._resolve_speculative_owner(
        provider=provider, draft_model=draft_model, model_plugin=plugin,
        model="target", speculative_mtp_serving=mode,
    )


def test_draft_model_alone_takes_the_plugin_provider() -> None:
    assert _owner(_Plugin("qwen4_exp_mtp"), draft_model="head.gguf") == (
        "qwen4_exp_mtp", "head.gguf",
    )
    with pytest.raises(ValueError, match="declares no MTP provider"):
        _owner(_Plugin(None), draft_model="head.gguf")


def test_discovery_attaches_the_head_unless_mtp_serving_is_off() -> None:
    plugin = _Plugin("qwen4_exp_mtp", head="/models/head.gguf")
    assert _owner(plugin) == ("qwen4_exp_mtp", "/models/head.gguf")
    assert _owner(plugin, mode="auto") == ("qwen4_exp_mtp", "/models/head.gguf")
    off = _Plugin("qwen4_exp_mtp", head="/models/head.gguf")
    assert _owner(off, mode="off") == (None, None)
    assert off.discovered == 0, "an off server must not load an MTP head"
    assert _owner(_Plugin("qwen4_exp_mtp", head=None)) == (None, None)
    assert _owner(_Plugin(None, head="/models/head.gguf")) == (None, None)


def test_explicit_pair_is_used_as_given() -> None:
    plugin = _Plugin("qwen4_exp_mtp", head="/models/head.gguf")
    assert _owner(plugin, provider="dflash", draft_model="d") == ("dflash", "d")
    assert plugin.discovered == 0
