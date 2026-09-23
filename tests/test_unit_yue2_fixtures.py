"""M0 gate: the committed oracle fixtures are intact and the validator fails closed.

The validator lives in ``scripts/yue2_oracle.py`` (oracle-side tooling) and imports
no torch, so it can run as a normal unit test. The ``cases`` family is produced by
the long production-case run and is covered by the same command; these tests cover
the stable families plus the negative (fail-closed) proof.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._torch_absence import run_in_clean_interpreter

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/yue2"
STABLE_FAMILIES = (
    "ar_replay",
    "greedy",
    "nar",
    "nar-condend",
    "nar-multichunk",
    "operators",
    "sampling",
    "tokenizer",
    "vae",
)


def _load_oracle():
    spec = importlib.util.spec_from_file_location("yue2_oracle", REPO / "scripts/yue2_oracle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def oracle():
    return _load_oracle()


@pytest.fixture(scope="module")
def stable_tree(tmp_path_factory):
    """A self-consistent copy of the stable fixture families."""
    root = tmp_path_factory.mktemp("yue2-stable-fixtures")
    for name in STABLE_FAMILIES:
        if (FIXTURES / name).is_dir():
            shutil.copytree(FIXTURES / name, root / name)
    for name in ("oracle_env.json",):
        if (FIXTURES / name).is_file():
            shutil.copy(FIXTURES / name, root / name)
    (root / "integrity.json").write_text(json.dumps(_load_oracle().fixture_integrity(root)))
    return root


def test_oracle_tooling_imports_no_torch(oracle, request):
    if not run_in_clean_interpreter(request.node.nodeid):
        return
    assert "torch" not in sys.modules


def test_fixture_families_present():
    for name in STABLE_FAMILIES:
        assert (FIXTURES / name).is_dir(), f"missing fixture family {name}"


def test_stable_fixture_tree_validates(oracle, stable_tree):
    assert oracle.validate_fixtures(stable_tree) == []


def test_integrity_index_covers_every_stable_file(oracle):
    integrity = json.loads((FIXTURES / "integrity.json").read_text())
    listed = {name for name in integrity if name.split("/", 1)[0] in STABLE_FAMILIES}
    present = {
        str(path.relative_to(FIXTURES))
        for name in STABLE_FAMILIES
        for path in (FIXTURES / name).rglob("*")
        if path.is_file()
    }
    assert present <= set(listed), f"unlisted fixture files: {sorted(present - set(listed))}"
    assert listed <= present


def test_integrity_index_records_size_and_hash(oracle):
    integrity = json.loads((FIXTURES / "integrity.json").read_text())
    for name, entry in integrity.items():
        assert set(entry) == {"bytes", "sha256"}
        assert len(entry["sha256"]) == 64 and entry["bytes"] > 0


@pytest.mark.parametrize(
    "label",
    ["missing-file", "dropped-required-key", "wrong-length-relation", "tampered-id", "manifest-drift"],
)
def test_validator_rejects_broken_trees(oracle, stable_tree, tmp_path, label):
    mutations = {
        "missing-file": (lambda base: (base / "operators/rope_cos_sin.npz").unlink(), False),
        "dropped-required-key": (
            lambda base: _rewrite(base / "operators/snake_64x40.npz", drop="out"),
            True,
        ),
        "wrong-length-relation": (
            lambda base: _rewrite(base / "vae/decode.npz", override={"natural_length": np.int32(7)}),
            True,
        ),
        "tampered-id": (
            lambda base: _rewrite(
                base / "tokenizer/corpus.npz",
                override={"ids_0": np.asarray([oracle.VOCAB + 1], dtype=np.int32)},
            ),
            True,
        ),
        "manifest-drift": (
            lambda base: _rewrite_manifest(base / "nar/manifest.json", {"latent_norm": 1.0}),
            True,
        ),
    }
    mutate, refreeze = mutations[label]
    destination = tmp_path / f"case-{label}"
    problems = oracle._mutate_tree(stable_tree, destination, mutate, refreeze=refreeze)
    assert problems, f"validator accepted a tree with {label}"


def _rewrite(path: Path, *, drop=None, override=None) -> None:
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    if drop:
        arrays.pop(drop, None)
    arrays.update(override or {})
    np.savez(path, **arrays)


def _rewrite_manifest(path: Path, patch: dict) -> None:
    manifest = json.loads(path.read_text())
    manifest.update(patch)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def test_wheel_diff_flags_a_changed_pinned_module(oracle, tmp_path):
    """A release that touches an oracle module is a re-pin decision, not a note."""

    upstream = tmp_path / "upstream" / "yue2"
    wheel = tmp_path / "wheel" / "yue2"
    upstream.mkdir(parents=True)
    wheel.mkdir(parents=True)
    for name in ("protocol.py", "sampling.py"):
        (upstream / name).write_text("PINNED\n")
        (wheel / name).write_text("PINNED\n")
    (upstream / "cli.py").write_text("old\n")
    (wheel / "cli.py").write_text("new\n")
    (wheel / "brand_new.py").write_text("added\n")

    report = oracle.compare_wheel_source(wheel.parent, upstream.parent)
    assert report["identical"] == ["protocol.py", "sampling.py"]
    assert report["changed"] == ["cli.py"]
    assert report["only_in_wheel"] == ["brand_new.py"]
    assert report["pinned_modules_changed"] == []
    assert report["repin_required"] is False

    (wheel / "protocol.py").write_text("CHANGED\n")
    report = oracle.compare_wheel_source(wheel.parent, upstream.parent)
    assert report["pinned_modules_changed"] == ["protocol.py"]
    assert report["repin_required"] is True
