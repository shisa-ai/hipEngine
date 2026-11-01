"""``main()`` premise and cleanup for the four runner-owned expert-down arms.

The Q5_1 fold and Q8_0 row4/bundle rungs are request-local runner state: no
process key names them. ``main()`` must therefore (a) assert their premise
through the runner rather than through an empty environment key, and (b) leave
the environment alone on the cleanup path. Both used to reach
``os.environ[flag]`` with ``flag == ""``, which raised ``KeyError`` at the
premise and ``OSError`` in the ``finally`` of a run that had otherwise finished.

These tests drive the real ``main()`` end to end with a fake generator, so the
whole premise/cleanup path runs without a GPU. The premise cases are filtered
out of the measured loop; the GR-up DPP tests install one eligible case and a
runner that calls the registered leaf its arm selected, so the loop's counts,
restoration and refusal paths run against the real code below the premise.
"""

import json
import os
from types import SimpleNamespace

import numpy as np

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
            "qsa-h256-wave", "qsa-h256-page256", "q4-bundle", "prefill-bundle", "q51-pair", "gdn-register", "q4-pair", "q8-wave-scale", "gr-wave-scale", "gr-dpp-reduction")

#: The GR-up leaves the DPP package counts: the promoted DPP sibling, the K320
#: specialization the selector binds for complete 512-row chunks, and the
#: wave-scale parent the control arm runs. Both DPP leaves feed the one
#: ``candidate_calls`` count, so the enabled arm is counted whichever of the
#: two the chunk shape selects.
GR_DPP_VARIANT = "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact"
GR_K320_VARIANT = "coltile2_branch4_rowbatch4_wave_scale_dpp_k320_full_rows_f32_exact"
GR_WAVE_VARIANT = "coltile2_branch4_rowbatch4_wave_scale_f32_exact"

#: The keys ``main()`` resolves the counted leaf through, per package.
LEAF_KEYS = {
    "gr-dpp-reduction": KernelKey(
        "hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0", GR_DPP_VARIANT),
    "gr-wave-scale": KernelKey(
        "hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0", GR_WAVE_VARIANT),
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

#: The control arm's leaf for the DPP package: the parent the sibling replaces.
GR_DPP_PARENT_KEY = KernelKey(
    "hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0", GR_WAVE_VARIANT)

#: The K320 specialization a complete 512-row chunk selects. The package
#: installs a second counter on it, so a run that only counted the promoted
#: sibling recorded zero for the arm that actually ran.
GR_K320_KEY = KernelKey(
    "hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0", GR_K320_VARIANT)


def _resolved(key: KernelKey):
    """The entry currently registered for ``key``."""

    return resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                   variant=key.variant)


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
        "gr_dpp_reduction": True,
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
    # The DPP package resolves both DPP leaves it counts and the parent the
    # control arm counts.
    register(GR_DPP_PARENT_KEY, lambda *a, **kw: None, replace=True)
    register(GR_K320_KEY, lambda *a, **kw: None, replace=True)
    for key, _env in PROCESS_KEYED.values():
        register(key, lambda *a, **kw: None, replace=True)
    return registry


@pytest.fixture
def premise_env(monkeypatch):
    """A stale process key cannot control the runner-owned pair2 premise."""

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL", "0")


def _install_generator(monkeypatch, runner, *, cases=None):
    """Replace the profile/generator construction with a CPU fake."""

    fixture_cases = list(cases) if cases is not None else [
        {"id": "filtered", "category": "x", "prompt_tokens": 1,
         "prompt_token_ids": [1]}]
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
        lambda path: ({"cases": fixture_cases}, "d" * 64))
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


def _run_main(monkeypatch, tmp_path, package, runner, *, cases=None,
              extra_argv=()):
    """Call the real ``main()`` for ``package`` and return its report."""

    _install_generator(monkeypatch, runner, cases=cases)
    output = tmp_path / f"{package}.json"
    monkeypatch.setattr("sys.argv", [
        "qwen4exp_row4_state_gate.py",
        "--model-root", str(tmp_path),
        "--compiler-version-file", str(tmp_path / "compiler.txt"),
        "--output", str(output),
        "--route-package", package,
        *extra_argv,
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
    elif package == "gr-dpp-reduction":
        from scripts.qwen4exp_halo_box_campaign_ab import GR_WAVE_ENV
        assert control["runner_field"] == "gr_dpp_reduction"
        assert control["before"] is False and control["after"] is True
        assert control["published_value"] is True
        assert control["forced_environment"] == GR_WAVE_ENV
        # Both arms hold the parent on and close the preemptors, so the only
        # difference between them is the counted leaf.
        assert control["forced_runner_fields"] == {
            "gr_wide": False, "gr_wave_scale": True,
        }
        assert control["counted_variant"] == GR_DPP_VARIANT
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


# --- the measured loop: both counted leaves, and the refusal below the guard --


class _ArmRunner:
    """A CPU runner that calls the registered leaf its bound arm selects.

    ``prefill`` resolves the kernel at call time, so the gate's own counter
    wrappers are what observe the call, and a chunk at or below the 257-row
    guard calls neither leaf - the same shape the selector applies. The
    enabled arm's leaf is ``dpp_variant``; the runtime selector binds the K320
    specialization only for a complete ROW_BATCH=4 group, so this fake applies
    that guard to the same row count.
    """

    GR_READS = 96

    def __init__(self, dpp_variant: str = GR_DPP_VARIANT) -> None:
        self.prefill_chunk_size = 1024
        self.dpp_variant = dpp_variant
        self.gr_wide = True
        # The plan may publish the parent off; the arm holds it on, so the
        # binding is observable from inside the measured call.
        self.gr_wave_scale = False
        self.gr_dpp_reduction = True
        self.runtime = SimpleNamespace(device_synchronize=lambda: None)
        #: (rows, sibling, parent, wide) as the measured prefill saw them.
        self.observed: list[tuple[int, bool, bool, bool]] = []
        #: The variant each engaged prefill resolved, in call order.
        self.leaves: list[str] = []

    def _leaf(self, rows: int):
        """The leaf the runtime selector would bind for ``rows``.

        The K320 specialization replaces the promoted DPP owner only above the
        257-row guard and on a complete ROW_BATCH=4 group; every other shape
        keeps the sibling the package promoted.
        """
        if not self.gr_dpp_reduction:
            variant = GR_WAVE_VARIANT
        elif (self.dpp_variant == GR_K320_VARIANT
              and rows > 256 and rows % 4 == 0):
            variant = GR_K320_VARIANT
        else:
            variant = GR_DPP_VARIANT
        self.leaves.append(variant)
        key = KernelKey("hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
                        variant)
        return resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                       variant=key.variant)

    def prefill(self, token_ids):
        self.observed.append((len(token_ids), self.gr_dpp_reduction,
                              self.gr_wave_scale, self.gr_wide))
        if len(token_ids) > 256:
            leaf = self._leaf(len(token_ids))
            for _ in range(self.GR_READS):
                leaf()
        return SimpleNamespace(logits=np.zeros(4, dtype=np.float32), token_id=0)

    def step(self, token):
        return SimpleNamespace(logits=np.zeros(4, dtype=np.float32), token_id=1)


def _flat_state(runner):
    return {"finite": True, "state_sha256": "s" * 8, "layout_sha256": "l" * 8}


@pytest.mark.parametrize("dpp_variant", (GR_DPP_VARIANT, GR_K320_VARIANT))
def test_main_counts_the_sibling_and_the_parent_in_their_arms(
        stubs, premise_env, monkeypatch, tmp_path, dpp_variant):
    """The measured loop proves both counts and restores every binding.

    The enabled arm is parameterized because a complete 512-row chunk makes the
    runtime selector bind the K320 specialization instead of the promoted
    sibling; both leaves feed the one ``candidate_calls`` count, so a harness
    that counted only the promoted sibling recorded 0 for the arm that ran.
    """

    monkeypatch.setattr(gate, "_state_summary", _flat_state)
    runner = _ArmRunner(dpp_variant=dpp_variant)
    before = set(os.environ)
    originals = {key: _resolved(key) for key in (GR_DPP_PARENT_KEY, GR_K320_KEY)}
    report = _run_main(
        monkeypatch, tmp_path, "gr-dpp-reduction", runner,
        cases=[{"id": "code-p512", "category": "code",
                "prompt_tokens": 512, "prompt_token_ids": [1] * 512}])
    # Both counters the package installs are restored, whichever leaf ran: the
    # wave-scale parent the control arm counts and the K320 specialization the
    # enabled arm can select.
    for key, original in originals.items():
        assert _resolved(key) is original, key.variant
    assert report["status"] == "passed"
    captures = report["cases"][0]["captures"]
    assert [(row["enabled"], row["candidate_calls"], row["wave_parent_calls"])
            for row in captures] == [("0", 0, 96), ("1", 96, 0), ("0", 0, 96)]
    # Inside the measured call, both arms held the parent on and the wide tile
    # off, and only the sibling differed.
    assert runner.observed == [
        (512, False, True, False),
        (512, True, True, False),
        (512, False, True, False),
    ]
    # The enabled arm resolved the leaf this parameter names; the control arms
    # resolved the parent. 96 calls of one leaf are one count, not two.
    assert runner.leaves == [GR_WAVE_VARIANT, dpp_variant, GR_WAVE_VARIANT]
    # Every arm's control is released: the published route is back in force.
    assert runner.gr_dpp_reduction is True
    assert runner.gr_wave_scale is False
    assert runner.gr_wide is True
    # The scope's environment pin is released too.
    assert "" not in os.environ
    assert set(os.environ) - before <= MAIN_OWNED_KEYS


def test_main_keeps_the_promoted_sibling_off_a_complete_row_group(
        stubs, premise_env, monkeypatch, tmp_path):
    """The K320 guard is a shape predicate, not the package's whole arm.

    A 510-row chunk clears the 257-row guard but does not close a ROW_BATCH=4
    group, so the selector keeps the promoted DPP sibling - and that leaf's
    counter still feeds the arm's ``candidate_calls``.
    """

    monkeypatch.setattr(gate, "_state_summary", _flat_state)
    runner = _ArmRunner(dpp_variant=GR_K320_VARIANT)
    report = _run_main(
        monkeypatch, tmp_path, "gr-dpp-reduction", runner,
        cases=[{"id": "code-p510", "category": "code",
                "prompt_tokens": 510, "prompt_token_ids": [1] * 510}],
        extra_argv=("--case-id", "code-p510"))
    assert report["status"] == "passed"
    captures = report["cases"][0]["captures"]
    assert [(row["enabled"], row["candidate_calls"], row["wave_parent_calls"])
            for row in captures] == [("0", 0, 96), ("1", 96, 0), ("0", 0, 96)]
    assert runner.leaves == [GR_WAVE_VARIANT, GR_DPP_VARIANT, GR_WAVE_VARIANT]


def test_main_refuses_a_case_that_cannot_engage_the_sibling(
        stubs, premise_env, monkeypatch, tmp_path):
    """A selected case below the guard is refused, not recorded as 0 == 0.

    The control arm runs first and is allowed to expect nothing; the enabled
    arm then refuses before its count assertion, so the failure names the
    engagement rule rather than a count mismatch.
    """

    monkeypatch.setattr(gate, "_state_summary", _flat_state)
    runner = _ArmRunner()
    with pytest.raises(AssertionError, match="zero expectation"):
        _run_main(
            monkeypatch, tmp_path, "gr-dpp-reduction", runner,
            cases=[{"id": "short-p256", "category": "code",
                    "prompt_tokens": 256, "prompt_token_ids": [1] * 256}],
            extra_argv=("--all-cases",))
    # The refused arm still released its bindings before the raise.
    assert runner.gr_dpp_reduction is True
    assert runner.gr_wave_scale is False
    assert "" not in os.environ


class _BaseParentRunner(_ArmRunner):
    """A control arm that falls through to the fused parent below the sibling.

    The enabled arm still counts its sibling, so the sibling expectation holds
    and only the parent count can catch the fallthrough.
    """

    def prefill(self, token_ids):
        self.observed.append((len(token_ids), self.gr_dpp_reduction,
                              self.gr_wave_scale, self.gr_wide))
        if len(token_ids) > 256 and self.gr_dpp_reduction:
            leaf = self._leaf(len(token_ids))
            for _ in range(self.GR_READS):
                leaf()
        return SimpleNamespace(logits=np.zeros(4, dtype=np.float32), token_id=0)


def test_main_refuses_a_control_arm_that_runs_the_base_parent(
        stubs, premise_env, monkeypatch, tmp_path):
    """The control arm's parent count is checked, not assumed from a zero.

    Without the parent count this arm would record ``0 == 0`` as engaged while
    it measured the fused parent below the wave-scale owner.
    """

    monkeypatch.setattr(gate, "_state_summary", _flat_state)
    runner = _BaseParentRunner()
    # The prefill parent expectation reports the observed/expected pair.
    with pytest.raises(AssertionError, match=r"\(0, 96\)"):
        _run_main(
            monkeypatch, tmp_path, "gr-dpp-reduction", runner,
            cases=[{"id": "code-p512", "category": "code",
                    "prompt_tokens": 512, "prompt_token_ids": [1] * 512}])
    # The arm was bound correctly; only the leaf it ran was wrong.
    assert runner.observed == [(512, False, True, False)]
