#!/usr/bin/env python3
"""Run the canonical p512 AR fixture through hipEngine with a candidate route.

This mirrors ``scripts/qwen4exp_canonical_ar_bench.py``'s ``hipengine`` mode
but lets the caller force a diagnostic routing override. The attention-gate
``--override`` binds generator/runner owner state and prepares the required MMQ
workspace before sampling; it does not change the process key. Generic WMMA
``--override`` also binds owner state. Flash-prefill enable/layer overrides use
the same paired owner policy as profile-gap, as do the independent GR IU8
up/down overrides. Other overrides remain post-binder environment inputs.
This runner produces diagnostic samples, not production benchmark evidence.

Usage:
  python3 scripts/qwen4exp_candidate_ar.py hipengine \
      --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
      --fixture /tmp/qwen4exp_p512_fixture.json \
      --output /tmp/cand.json --warmups 1 --repetitions 1 \
      --override HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE=1 \
      --compiler-version-file /tmp/hipengine-hipcc-version.txt
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import (  # noqa: E402
    _host_metadata,
    _hipengine_case_sample,
    _measurement_order,
    _write_json,
    load_fixture,
    summarize_samples,
)


def split_owner_overrides(values: list[str]) -> tuple[dict[str, str], bool | None]:
    """Preserve CLI spelling while separating the attention owner decision."""
    environment = {}
    attention = None
    for raw in values:
        key, sep, value = raw.partition('=')
        if not sep:
            raise ValueError(f"--override must be KEY=VALUE, got {raw!r}")
        if key == 'HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE':
            attention = value not in {'', '0', 'false', 'False'}
        else:
            environment[key] = value
    return environment, attention


def split_q51_iu8_override(values: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Separate the exact Q5_1 owner input from both process-write stages."""
    key = 'HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT'
    return ({name: value for name, value in values.items() if name != key},
            {name: value for name, value in values.items() if name == key})


def bind_q51_iu8_override(stack: ExitStack, owner, overrides: dict[str, str]):
    """Share the profile-gap parser and restoring owner binding."""
    from scripts.qwen4exp_profile_gap import bind_q51_iu8_override as bind
    return bind(stack, owner, overrides)


def split_gr_iu8_overrides(values: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Keep both GR owner axes out of construction and post-binder writes."""
    keys = {'HIPENGINE_QWEN4_EXP_GR_IU8', 'HIPENGINE_QWEN4_EXP_GR_IU8_DOWN'}
    return ({key: value for key, value in values.items() if key not in keys},
            {key: value for key, value in values.items() if key in keys})


def bind_gr_iu8_override(stack: ExitStack, owner, overrides: dict[str, str]):
    """Share the profile-gap parser, partial-axis binding and effective report."""
    from scripts.qwen4exp_profile_gap import bind_gr_iu8_override as bind
    return bind(stack, owner, overrides)


def split_selected_q8_down_override(values: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Keep selected-down owner requests out of both process-write stages."""
    key = 'HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN'
    return ({name: value for name, value in values.items() if name != key},
            {name: value for name, value in values.items() if name == key})


def bind_selected_q8_down_override(stack: ExitStack, owner, overrides: dict[str, str]):
    """Use the profile-gap binder's parser and effective-value report."""
    from scripts.qwen4exp_profile_gap import bind_selected_q8_down_override as bind
    return bind(stack, owner, overrides)


def split_flash_prefill_overrides(values: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Keep flash requests out of construction and post-binder process writes."""
    keys = {'HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL', 'HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS'}
    return ({key: value for key, value in values.items() if key not in keys},
            {key: value for key, value in values.items() if key in keys})


def bind_flash_prefill_override(stack: ExitStack, owner, overrides: dict[str, str]):
    """Share partial-axis semantics with the profile-gap diagnostic."""
    from scripts.qwen4exp_profile_gap import bind_flash_prefill_override as bind
    return bind(stack, owner, overrides)


def split_generic_wmma_override(values: dict[str, str]) -> tuple[dict[str, str], bool | None]:
    """Separate generic WMMA using its low-level boolean parser."""
    environment = dict(values)
    raw = environment.pop('HIPENGINE_GGUF_WMMA_PREFILL', None)
    return environment, None if raw is None else raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def bind_generic_wmma_override(stack: ExitStack, owner, enabled: bool | None):
    """Keep the generic WMMA choice on the measured owner until stack exit."""
    from hipengine.runtime.qwen4_exp_runner import gguf_wmma_prefill
    if enabled is not None:
        stack.enter_context(gguf_wmma_prefill(owner, enabled))
    return {'gguf_wmma_prefill_enabled': getattr(owner, 'gguf_wmma_prefill_enabled', None)}


@contextmanager
def attention_override_scope(owner, enabled: bool | None):
    """Prepare the explicit MMQ parent and restore both choices after sampling."""
    from hipengine.runtime.qwen4_exp_runner import (
        q8_mmq_attn_gate_enabled, q8_mmq_prefill_enabled, resolve_q8_mmq_prefill,
    )
    with ExitStack() as stack:
        if enabled is not None:
            stack.enter_context(q8_mmq_prefill_enabled(owner, True))
            stack.enter_context(q8_mmq_attn_gate_enabled(owner, enabled))
            owner.runner.configure_mmq_prefill_resources()
        yield {
            'q8_mmq_attn_gate_enabled': getattr(owner, 'q8_mmq_attn_gate_enabled', None),
            'q8_mmq_prefill_enabled': resolve_q8_mmq_prefill(owner),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--compiler-version-file", type=Path)
    args = parser.parse_args()

    try:
        overrides, attention_override = split_owner_overrides(args.override)
        overrides, generic_wmma_override = split_generic_wmma_override(overrides)
        overrides, q51_iu8_overrides = split_q51_iu8_override(overrides)
        overrides, gr_iu8_overrides = split_gr_iu8_overrides(overrides)
        overrides, selected_down_overrides = split_selected_q8_down_override(overrides)
        overrides, flash_overrides = split_flash_prefill_overrides(overrides)
    except ValueError as exc:
        parser.error(str(exc))
    os.environ.update(overrides)

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")

    from hipengine.core.memory import memory_stats
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model

    fixture, _fixture_sha = load_fixture(args.fixture)
    cases = fixture["cases"]
    transitions = int(fixture["decode_transitions"])
    model_root = args.model_root.resolve()
    max_sequence_length = max(int(row["prompt_tokens"]) for row in cases) + transitions + 8

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile.PRODUCTION,
    )

    # Only ambient controls are reapplied after the profile binder. The
    # attention decision is bound below on the fully constructed owner.

    def construct(base_factory, **kwargs):
        generator = resolved.construct_generator(base_factory, **kwargs)
        for key, value in overrides.items():
            os.environ[key] = value
        return generator

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=args.prefill_chunk_size,
        )

    artifact: dict = {
        "schema": 1,
        "kind": "qwen4exp_canonical_ar_engine_run",
        "engine": "hipengine_candidate",
        "surface": "synchronized_direct_runner",
        "status": "running",
        "host": _host_metadata(),
        "fixture": str(args.fixture.resolve()),
        "model_root": str(model_root),
        "source": _git_metadata_import(),
        "profile": {
            "requested": "production",
            "manifest_sha256": resolved.manifest_sha256,
            "fell_back_to_strict": resolved.fell_back_to_strict,
        },
        "overrides": dict(overrides),
        "protocol": {
            "warmups_per_case": int(args.warmups),
            "measured_repetitions": int(args.repetitions),
            "decode_transitions": transitions,
            "prefill_chunk_size": int(args.prefill_chunk_size),
        },
        "warmups": [],
        "samples": [],
    }
    _write_json(args.output, artifact)
    generator = construct(factory)
    owner_stack = ExitStack()
    try:
        artifact['owner_overrides'] = owner_stack.enter_context(
            attention_override_scope(generator, attention_override))
        artifact['owner_overrides'].update(bind_generic_wmma_override(
            owner_stack, generator, generic_wmma_override))
        artifact['owner_overrides'].update(bind_q51_iu8_override(
            owner_stack, generator, q51_iu8_overrides))
        artifact['owner_overrides'].update(bind_gr_iu8_override(
            owner_stack, generator, gr_iu8_overrides))
        artifact['owner_overrides'].update(bind_selected_q8_down_override(
            owner_stack, generator, selected_down_overrides))
        artifact['owner_overrides'].update(bind_flash_prefill_override(
            owner_stack, generator, flash_overrides))
        _write_json(args.output, artifact)
        for warmup in range(args.warmups):
            for case in _measurement_order(cases, warmup):
                row = _hipengine_case_sample(
                    generator.runner, case=case, repetition=warmup, transitions=transitions
                )
                artifact["warmups"].append(row["case_id"])
                print(
                    f"[warmup] hipengine_candidate {row['case_id']} "
                    f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                    flush=True,
                )
        for repetition in range(args.repetitions):
            for case in _measurement_order(cases, repetition):
                row = _hipengine_case_sample(
                    generator.runner, case=case, repetition=repetition, transitions=transitions
                )
                artifact["samples"].append(row)
                artifact["summary"] = summarize_samples(artifact["samples"])
                _write_json(args.output, artifact)
                print(
                    f"[measure {repetition}] hipengine_candidate {row['case_id']} "
                    f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                    flush=True,
                )
        artifact["status"] = "completed"
        artifact["memory_before_close"] = memory_stats()
        _write_json(args.output, artifact)
    finally:
        try:
            owner_stack.close()
        finally:
            generator.close()
    print(
        json.dumps(
            {"kind": artifact["kind"], "status": artifact["status"], "output": str(args.output)},
            indent=2,
        )
    )
    return 0


def _git_metadata_import():
    from scripts.qwen4exp_canonical_ar_bench import _git_metadata

    return _git_metadata(ROOT)


if __name__ == "__main__":
    import json

    raise SystemExit(main())
