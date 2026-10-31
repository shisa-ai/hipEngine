"""``main()`` premise and cleanup for the four runner-owned expert-down arms.

The Q5_1 fold and Q8_0 row4/bundle rungs are request-local runner state: no
process key names them. ``main()`` must therefore (a) assert their premise
through the runner rather than through an empty environment key, and (b) leave
the environment alone on the cleanup path. Both used to reach
``os.environ[flag]`` with ``flag == ""``, which raised ``KeyError`` at the
premise and ``OSError`` in the ``finally`` of a run that had otherwise finished.

These tests drive the real ``main()`` end to end with a fake generator, so the
whole premise/cleanup path runs without a GPU. The measured loop is empty by
construction (its fixture case is filtered out), so nothing below the premise is
faked.
"""

import json
import os
from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import (
    KernelKey,
    register,
    registered_keys,
    restore_registry_for_tests,
    resolve,
)
from scripts import qwen4exp_row4_state_gate as gate

PACKAGES = ("q51-fold128", "q51-fold-pair", "q8-down-row4", "q8-down-bundle", "q5k-row4",
            "qsa-h256-wave", "qsa-h256-page256", "q4-bundle", "prefill-bundle", "q51-pair", "gdn-register", "q4-pair", "q8-wave-scale", "gr-wave-scale")

#: The keys ``main()`` resolves the counted leaf through, per package.
LEAF_KEYS = {
    "gr-wave-scale": KernelKey(
        "hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
        "coltile2_branch4_rowbatch4_wave_scale_f32_exact"),
    "q8-wave-scale": KernelKey(
        "hip_gfx1151", "linear", "gguf_q8_0",
        "coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out"),
    "q4-pair": KernelKey(
        "hip_gfx1151", "moe_linear", "gguf_q4_k",
        "selected_dual_grouped_pair2_bf16_bf16_out"),
    "gdn-register": KernelKey(
        "hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
        "qwen4exp_sigmoid_register_prefill"),
    "q51-pair": KernelKey(
        "hip_gfx1151", "moe_linear", "gguf_q5_1",
        "selected_grouped_prefill_pair2_bf16_bf16_out"),
    "q4-bundle": KernelKey(
        "hip_gfx1151", "moe_linear", "gguf_q4_k",
        "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out"),
    "prefill-bundle": KernelKey(
        "hip_gfx1151", "moe_linear", "gguf_q4_k",
        "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out"),
    "qsa-h256-wave": KernelKey(
        "hip_gfx1151", "qsa_sparse_attention", "bf16_kv", "strict_h256_wave_rows_spans"),
    "qsa-h256-page256": KernelKey(
        "hip_gfx1151", "qsa_sparse_attention", "bf16_kv", "strict_h256_page256_wave_rows_spans"),
    "q5k-row4": KernelKey(
        "hip_gfx1151", "linear", "gguf_q5_k",
        "selected_grouped_row4_gemv_bf16_bf16_out"),
    "q51-fold128": KernelKey(
        "hip_gfx1151", "moe_linear", "gguf_q5_1",
        "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"),
    "q51-fold-pair": KernelKey(
        "hip_gfx1151", "moe_linear", "gguf_q5_1",
        "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out"),
    "q8-down-row4": KernelKey(
        "hip_gfx1151", "linear", "gguf_q8_0",
        "selected_grouped_row4_gemv_bf16_bf16_out"),
    "q8-down-bundle": KernelKey(
        "hip_gfx1151", "linear", "gguf_q8_0",
        "selected_grouped_row4_bundle_gemv_bf16_bf16_out"),
}

#: The leaf keys and binder keys of the packages that still own a process key.
PROCESS_KEYED = {
    "q4-iu8-exact": (
        KernelKey("hip_gfx1151", "moe_linear", "gguf_q4_k",
                  "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"),
        "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT"),
}

#: The process keys ``main()`` itself sets before any measurement.
MAIN_OWNED_KEYS = {
    "HIPENGINE_COMPILER_VERSION_FILE",
    "HIPENGINE_REQUIRE_CACHED_BUILD",
}


def _runner(**overrides):
    """A runner carrying the values the production binder publishes."""

    published = {
        "prefill_chunk_size": 1024,
        "gdn_register_prefill": True,
        "q51_pair_prefill": True,
        "q51_exact_grouped_down": True,
        "q51_fold128_prefill": True,
        "q51_fold_pair_prefill": True,
        "q51_register_cache": True,
        "q51_row_publish": True,
        "q5k_row4_prefill": True,
        "q4_bundle_prefill": True,
        "q4_pair_prefill": True,
        "q8_wave_scale_prefill": True,
        "gr_wave_scale": True,
        "qsa_h256_prefill_mode": "page256",
        "q8_down_row4_prefill": True,
        "q8_down_bundle_prefill": True,
        "q8_down_register": True,
        "q8_mapped_down": True,
        "forkb_grouped_down": True,
    }
    published.update(overrides)
    return SimpleNamespace(**published)


@pytest.fixture
def registry():
    """Snapshot and restore the process-global kernel table."""

    before = {key: resolve(
        backend=key.backend, layer=key.layer, quant=key.quant,
        variant=key.variant, missing="none") for key in registered_keys()}
    yield
    restore_registry_for_tests(
        {key: kernel for key, kernel in before.items() if kernel is not None})


@pytest.fixture
def stubs(registry):
    """A registered stub for every leaf ``main()`` resolves."""

    for key in LEAF_KEYS.values():
        register(key, lambda *a, **kw: None, replace=True)
    for key, _env in PROCESS_KEYED.values():
        register(key, lambda *a, **kw: None, replace=True)
    return registry


@pytest.fixture
def premise_env(monkeypatch):
    """A stale process key cannot control the runner-owned pair2 premise."""

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL", "0")


def _install_generator(monkeypatch, runner):
    """Replace the profile/generator construction with a CPU fake."""

    generator = SimpleNamespace(runner=runner, close=lambda: None)
    resolved = SimpleNamespace(
        manifest_sha256="m" * 64,
        strict_manifest_sha256="s" * 64,
        construct_generator=lambda factory: generator,
    )
    monkeypatch.setattr(gate, "resolve_runtime_profile", lambda **kwargs: resolved)
    monkeypatch.setattr(gate, "register_gfx1151_kernels", lambda **kwargs: None)
    monkeypatch.setattr(
        gate, "register_qwen4_exp_gfx1151_profiles", lambda: None)
    monkeypatch.setattr(
        gate, "load_fixture",
        lambda path: (
            {"cases": [{"id": "filtered", "category": "x", "prompt_tokens": 1,
                        "prompt_token_ids": [1]}]},
            "d" * 64,
        ))
    monkeypatch.setattr(gate, "discover_gguf_files", lambda root: ["model.gguf"])
    monkeypatch.setattr(gate, "load_gguf_index", lambda path: SimpleNamespace())
    monkeypatch.setattr(gate, "_git_metadata", lambda root: {"commit": "0" * 40})
    monkeypatch.setattr(gate, "_host_metadata", lambda: {"host": "cpu"})
    monkeypatch.setattr(
        gate, "memory_stats",
        lambda: {"active_allocations": 0, "current_allocated_bytes": 0})
    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_q5_k_q8_1_selected_prefill as selected,
    )
    monkeypatch.setattr(
        selected, "register_gguf_q5_k_q8_1_selected_prefill_kernels",
        lambda **kwargs: None)
    return generator


def _run_main(monkeypatch, tmp_path, package, runner):
    """Call the real ``main()`` for ``package`` and return its report."""

    _install_generator(monkeypatch, runner)
    output = tmp_path / f"{package}.json"
    monkeypatch.setattr("sys.argv", [
        "qwen4exp_row4_state_gate.py",
        "--model-root", str(tmp_path),
        "--compiler-version-file", str(tmp_path / "compiler.txt"),
        "--output", str(output),
        "--route-package", package,
    ])
    gate.main()
    return json.loads(output.read_text())


@pytest.mark.parametrize("package", PACKAGES)
def test_main_writes_no_empty_process_key(
        stubs, premise_env, monkeypatch, tmp_path, package):
    # The reported failure: the premise asserted os.environ[""] (KeyError) and
    # the finally wrote os.environ[""] (OSError).
    before = set(os.environ)
    report = _run_main(monkeypatch, tmp_path, package, _runner())
    assert "" not in os.environ
    assert set(os.environ) - before <= MAIN_OWNED_KEYS
    assert report["status"] == "passed"
    assert report["route_package"] == package


@pytest.mark.parametrize("package", PACKAGES)
def test_main_records_the_runner_owned_premise(
        stubs, premise_env, monkeypatch, tmp_path, package):
    # The premise is still asserted; it is just read from the runner, and the
    # record names the published value the arm replaced.
    report = _run_main(monkeypatch, tmp_path, package, _runner())
    control = report["selection_control"]
    assert control["scope"] == "per measured run; restored on exit"
    if package in {"q4-bundle", "prefill-bundle"}:
        assert control["runner_field"] == "q4_bundle_prefill"
        assert control["before"] is False and control["after"] is True
        if package == "prefill-bundle":
            assert control["additional_control"] == {
                "runner_field": "qsa_h256_prefill_mode",
                "before": "parent", "after": "page256",
            }
    elif package.startswith("qsa-h256-"):
        assert control["runner_field"] == "qsa_h256_prefill_mode"
        assert control["before"] == "parent"
        assert control["after"] == ("wave" if package == "qsa-h256-wave" else "page256")
    elif package == "q5k-row4":
        assert control["runner_field"] == "q5k_row4_prefill"
        assert control["before"] is False and control["after"] is True
    elif package == "gdn-register":
        assert control["runner_field"] == "gdn_register_prefill"
        assert control["before"] is False and control["after"] is True
        assert control["published_value"] is True
        assert control["isolation"] == "tiled and peer prefill off; wave-norm variant unset"
    elif package == "q4-pair":
        from scripts.qwen4exp_halo_box_campaign_ab import Q4_PAIR_ENV
        assert control["runner_field"] == "q4_pair_prefill"
        assert control["before"] is False and control["after"] is True
        assert control["published_value"] is True
        assert control["forced_environment"] == Q4_PAIR_ENV
        assert control["forced_runner_fields"] == {
            "q4_exact_grouped_gate": True, "q4_exact_grouped_all": True,
            "q4_bundle_prefill": True, "routed_f16_enabled": False,
            "production_moe_prefill_enabled": False,
        }
    elif package == "q8-wave-scale":
        from scripts.qwen4exp_halo_box_campaign_ab import Q8_WAVE_ENV
        assert control["runner_field"] == "q8_wave_scale_prefill"
        assert control["before"] is False and control["after"] is True
        assert control["published_value"] is True
        assert control["forced_environment"] == Q8_WAVE_ENV
        assert control["forced_wmma_layers"] == {
            "mode": "empty", "layers": None,
        }
    elif package == "gr-wave-scale":
        from scripts.qwen4exp_halo_box_campaign_ab import GR_WAVE_ENV
        assert control["runner_field"] == "gr_wave_scale"
        assert control["before"] is False and control["after"] is True
        assert control["published_value"] is True
        assert control["forced_environment"] == GR_WAVE_ENV
    elif package == "q51-pair":
        assert control["runner_field"] == gate.Q51_PAIR_FIELD
        assert control["before"] is False and control["after"] is True
        assert control["published_value"] is True
    elif package == "q51-fold-pair":
        assert control["runner_field"] == gate.Q51_FOLD_PAIR_FIELD
        assert control["rung"]["prerequisite"] == gate.Q51_FOLD128_FIELD
    elif package == "q8-down-bundle":
        assert control["runner_field"] == gate.Q8_DOWN_BUNDLE_FIELD
        assert control["rung"]["prerequisite"] == gate.Q8_DOWN_ROW4_FIELD
    elif package == "q51-fold128":
        assert control["runner_field"] == gate.Q51_FOLD128_FIELD
        assert control["rung"]["held_off"] == gate.Q51_FOLD_PAIR_FIELD
    else:
        assert control["runner_field"] == gate.Q8_DOWN_ROW4_FIELD
        assert control["rung"]["held_off"] == gate.Q8_DOWN_BUNDLE_FIELD


@pytest.mark.parametrize(
    "package,field",
    [("q51-fold-pair", gate.Q51_FOLD128_FIELD),
     ("q8-down-bundle", gate.Q8_DOWN_ROW4_FIELD)])
def test_main_still_refuses_a_missing_published_parent(
        stubs, premise_env, monkeypatch, tmp_path, package, field):
    # A parent the plan did not publish is a premise failure, not a silent
    # measurement of the parent route.
    runner = _runner(**{field: False})
    with pytest.raises(AssertionError, match="production must publish"):
        _run_main(monkeypatch, tmp_path, package, runner)
    assert "" not in os.environ


@pytest.mark.parametrize(
    "package,key,message",
    [("q51-fold128", gate.Q51_PAIR_FIELD,
      "production must bind the grouped pair2 Q5_1 selection"),
     ("q51-fold-pair", gate.Q51_PAIR_FIELD,
      "production must bind the grouped pair2 Q5_1 selection")])
def test_main_still_refuses_a_closed_chain_gate(
        stubs, premise_env, monkeypatch, tmp_path, package, key, message):
    with pytest.raises(AssertionError, match=message):
        _run_main(monkeypatch, tmp_path, package, _runner(**{key: False}))
    assert "" not in os.environ


@pytest.mark.parametrize("package", ["q51-fold128", "q51-fold-pair"])
def test_main_refuses_an_unpublished_pair2_chain_gate(
        stubs, premise_env, monkeypatch, tmp_path, package):
    # The pair2 chain's own gate is request-local runner state now, so the
    # premise is the runner's published decision, not a process key.
    with pytest.raises(AssertionError,
                       match="the pair2 chain's own gate must be published"):
        _run_main(monkeypatch, tmp_path, package,
                  _runner(q51_exact_grouped_down=False))
    assert "" not in os.environ


@pytest.mark.parametrize("package", ["q8-down-row4", "q8-down-bundle"])
def test_main_refuses_an_unpublished_q8_chain_gate(
        stubs, premise_env, monkeypatch, tmp_path, package):
    # The sorted chain's own gate is request-local runner state now, so the
    # premise is the runner's published decision, not a process key.
    with pytest.raises(AssertionError,
                       match="production must bind the grouped Q8_0 down chain"):
        _run_main(monkeypatch, tmp_path, package,
                  _runner(forkb_grouped_down=False))
    assert "" not in os.environ


@pytest.mark.parametrize("package", sorted(PROCESS_KEYED))
def test_a_process_keyed_package_still_asserts_its_key(
        stubs, premise_env, monkeypatch, tmp_path, package):
    # The generic premise is intact for the packages that still own a key; the
    # empty-name guard cannot make it pass vacuously.
    monkeypatch.setenv(PROCESS_KEYED[package][1], "0")
    with pytest.raises(AssertionError, match="retained route without an override"):
        _run_main(monkeypatch, tmp_path, package, _runner())
