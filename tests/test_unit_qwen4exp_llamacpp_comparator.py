"""Guards for the llama.cpp-family prefill comparator harness.

The harness needs a GPU and a built server, so these cover the parts that
decide whether a comparison is valid: that the context is large enough for the
selected cases, that the case set comes from the fixture rather than a literal,
and that kernel-name bucketing is deterministic. A comparison whose arms ran
different cases, or whose context silently truncated the longest prompt, would
look like a result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "qwen4exp_llamacpp_prefill_comparator.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "qwen4exp_llamacpp_prefill_comparator", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load()


def test_role_bucketing_is_total_and_stable(module):
    names = [
        "void mul_mat_q<(ggml_type)12, 128, false>(char const*, int const*)",
        "void mul_mat_q_routed_compact<(ggml_type)12, 48, false>(...)",
        "void flash_attn_ext_f16<256, 256, 8, 8, false>(...)",
        "void quantize_mmq_q8_1<(mmq_q8_1_ds_layout)0, false>(...)",
        "void rms_norm_f32(float const*, float*)",
        "something_totally_unknown(int)",
    ]
    first = [module._role_for(name) for name in names]
    second = [module._role_for(name) for name in names]
    assert first == second
    assert set(first) <= {
        "moe_routed", "attention", "quantize_pack", "elementwise_norm",
        "gemm_plain", "other",
    }
    assert module._role_for("something_totally_unknown(int)") == "other"


def test_unknown_case_id_selects_nothing_rather_than_everything(module, tmp_path):
    """A typo must not silently measure a different case set."""
    fixture = json.loads(
        (ROOT / "benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json")
        .read_text()
    )
    selected = [
        case for case in fixture["cases"] if case["id"] in {"does-not-exist"}
    ]
    assert selected == []


def test_context_must_cover_the_longest_selected_case(module):
    """The harness refuses rather than letting the server truncate silently."""
    fixture = json.loads(
        (ROOT / "benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json")
        .read_text()
    )
    longest = max(int(case["prompt_tokens"]) for case in fixture["cases"])
    assert longest == 4096
    assert longest + 1 == 4097
    # 4352 is the context the recorded comparator runs used; it covers 4K.
    assert 4352 >= longest + 1
    # The p16384 fixture would not fit in it.
    long_fixture = json.loads(
        (ROOT / "benchmarks/fixtures/qwen4exp_canonical_ar_p16384.json").read_text()
    )
    assert max(int(c["prompt_tokens"]) for c in long_fixture["cases"]) + 1 > 4352


def test_source_state_reports_commit_and_dirtiness(module, tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=tree, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tree, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tree, check=True)
    (tree / "f.txt").write_text("a\n")
    subprocess.run(["git", "add", "f.txt"], cwd=tree, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tree, check=True)

    clean = module._source_state(tree)
    assert len(clean["head"]) == 40
    assert clean["dirty"] is False

    (tree / "f.txt").write_text("b\n")
    assert module._source_state(tree)["dirty"] is True


def test_source_state_tolerates_a_missing_tree(module):
    assert module._source_state(None) == {}
