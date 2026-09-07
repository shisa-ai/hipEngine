"""Capture N1 packed target logits at fixed strict-teacher contexts.

Target-only diagnostic with host-selected teacher-window commits. No provider,
service, repeated-economics or complete production qualification is implied.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@contextmanager
def target_contexts(generator, budget):
    """Use the adapter's package/profile policy, not copied capability values."""
    from hipengine.generation import qwen35_gguf_mtp2 as m
    adapter = m.Qwen35GGUFMTP2Adapter(
        SimpleNamespace(generator=generator, capacity=1), enabled=True,
        target_verify_mode='bulk', candidate_budget=budget, ngram_enabled=False)
    pairs = [
        ('production_physical_extra_rowtiles', m.q4_t16_physical_extra_rowtiles_session),
        ('production_exact_target_row_counts', m.physical_exact_rowtiles_session),
        ('production_physical_q5_rowtile', m.q5_t16_physical_rowtile_session),
        ('production_physical_q6_rowtile', m.q6_t16_physical_rowtile_session),
        ('production_physical_q6_mixed_rowtiles', m.q6_t16_physical_mixed_rowtiles_session),
        ('moe_physical_c2_numerics', m.moe_physical_c2_numerics_session),
        ('moe_physical_c2_pairreuse', m.moe_physical_c2_pairreuse_session),
        ('moe_physical_c2_exact_linear', m.moe_physical_c2_exact_linear_session),
    ]
    flags = {name: bool(getattr(adapter, name, False)) for name, _ in pairs}
    with ExitStack() as stack:
        stack.enter_context(m.target_verifier_active_slots_session(1))
        for name, context in pairs:
            stack.enter_context(context(flags[name]))
        stack.enter_context(m.target_verifier_wide_q6_shared4_leaf_session(False))
        yield dict(flags, exact_target_rows=list(adapter.production_exact_target_row_counts),
                   active_slots=1, wide_q6_shared4=False)


def candidate_provenance(llm, generator, fixture):
    """Validate production manifest identity and same-model strict teacher scope."""
    from hipengine.execution_profiles import manifest_sha256, validate_variant_manifest
    manifest = validate_variant_manifest(llm.execution_profile_manifest)
    digest = manifest_sha256(manifest)
    teacher = fixture['runtime_manifest']
    if (manifest['execution_profile'] != 'production'
            or digest != llm.execution_profile_manifest_sha256
            or digest != generator.execution_profile_manifest_sha256
            or any(manifest[key] != teacher[key]
                   for key in ('backend', 'model', 'quant', 'kv_policy'))):
        raise ValueError('candidate provenance differs from resolved profile or teacher scope')
    return dict(candidate_manifest=manifest, candidate_manifest_sha256=digest,
                teacher_runtime_manifest_sha256=fixture['runtime_manifest_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('/models/gguf/Qwen3.8-27B-Q4_K_M.gguf'))
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--budget', type=int, choices=range(1, 8), default=3)
    args = parser.parse_args()
    import numpy as np
    from hipengine import LLM
    from hipengine.loading.gguf import scan_gguf
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    from hipengine.runtime.gguf_linear import mtp_serving_target_use_wmma_prefill
    from scripts.qwen38_packed_c1_teacher import load_teacher_prompts
    from scripts.qwen38_packed_c1_teacher_fixture import load_teacher_fixture
    from scripts.qwen38_packed_c1_teacher_candidate import capture_teacher_candidate
    from scripts.qwen38_packed_c1_logits import compare_logits

    args.directory.mkdir(parents=True, exist_ok=False)
    report = dict(kind='packed_c1_teacher_target_diagnostic', complete=False,
                  full_profile_qualification=False, capacity=1, budget=args.budget,
                  execution_profile='production', records=[], host=platform.node(),
                  source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  command=sys.argv, teacher_manifest_sha256=hashlib.sha256(
                      (args.teacher / 'teacher.json').read_bytes()).hexdigest(),
                  limitation='Target only; host-selected commits; no provider/service/task qualification')
    llm = None
    try:
        llm = LLM(str(args.model), backend='hip_gfx1100', execution_profile='production',
                  max_sequence_length=1024, max_active_requests=1)
        llm.prepare(max_sequence_length=1024)
        generator = llm._get_text_generator()
        if str(generator.execution_profile) != 'production':
            raise ValueError('candidate generator did not select production profile')
        artifact = generator._kv_model_artifact_identity()
        if not artifact.content_verified or not artifact.sha256:
            raise ValueError('candidate model content must be verified')
        report['model_sha256'] = artifact.sha256
        report['candidate_manifest_sha256'] = generator.execution_profile_manifest_sha256
        if not report['candidate_manifest_sha256']:
            raise ValueError('candidate variant manifest is missing')
        fixture = load_teacher_fixture(args.teacher, expected_model_sha256=artifact.sha256,
                                       require_runtime_provenance=True)
        report.update(candidate_provenance(llm, generator, fixture))
        tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
        prompts = {r['id']: r for r in load_teacher_prompts()}
        for row in fixture['records']:
            if tuple(tokenizer.encode(prompts[row['prompt_id']]['rendered_prompt'])) != row['prompt_ids']:
                raise ValueError('teacher prompt IDs differ from model tokenizer')
        use_wmma = mtp_serving_target_use_wmma_prefill(
            generator.execution_profile,
            profile_fell_back_to_strict=bool(generator.execution_profile_fell_back_to_strict))
        report['target_use_wmma_prefill'] = use_wmma
        with generator._resident_session_scope(shared_runner=generator._get_shared_runner(),
                pool_name='packed_c1_teacher_candidate') as (session, _reused):
            report['kv_storage_dtype'] = str(session.kv_storage_dtype)
            if report['kv_storage_dtype'] != fixture['kv_storage_dtype']:
                raise ValueError('candidate and teacher actual KV storage differ')
            for row in fixture['records']:
                with target_contexts(generator, args.budget) as flags:
                    result = capture_teacher_candidate(session, session, row, budget=args.budget,
                                                       use_wmma_prefill=use_wmma)
                report['target_contexts'] = flags
                filename = row['prompt_id'] + '.npz'
                np.savez(args.directory / filename, logits=result['logits'],
                         prefill_logits=result['prefill_logits'])
                record = dict(prompt_id=row['prompt_id'], category=row['category'], suite=row['suite'],
                              logits_file=filename, windows=result['windows'],
                              logits_sha256=hashlib.sha256(result['logits'].tobytes()).hexdigest(),
                              prefill_logits_sha256=hashlib.sha256(result['prefill_logits'].tobytes()).hexdigest(),
                              decode=compare_logits(row['logits'], result['logits']),
                              prefill=compare_logits(row['prefill_logits'][None, :], result['prefill_logits'][None, :]))
                report['records'].append(record)
                print(json.dumps(dict(prompt_id=row['prompt_id'], decode=record['decode'])), flush=True)
        llm.close()
        llm = None
        report['complete'] = len(report['records']) == 18
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if llm is not None:
                llm.close()
        finally:
            (args.directory / 'candidate.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
