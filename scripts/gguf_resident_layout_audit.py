"""Resident-layout audit through the real materialization planner.

Reports, per GGUF file, how many weight bytes land in a layout the optimized
(T16/x8/planar) kernel families can consume versus a layout that forces the
GEMV-shaped fallback, and which (quant, layout, role) groups are responsible.

CPU-only: reads GGUF headers and runs the planner. No HIP runtime, no device
allocation, so it is safe to run while the GPU is busy.

    python3 scripts/gguf_resident_layout_audit.py [model.gguf ...]
    AUDIT_NO_VETO=1 python3 scripts/gguf_resident_layout_audit.py model.gguf

``AUDIT_NO_VETO=1`` plans with ``repack_veto=False``, showing what the model
would get if the model-wide raw-IQ decode-repack veto were per-tensor. See
docs/UD-OPTIMIZED-ROUTE-PLAN.md.
"""
import collections
import sys

BACKEND = 'hip_gfx1151'
FILES = sys.argv[1:] or [
    '/models/gguf/Qwen3.8-27B-Q4_K_S.gguf',
    '/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf',
    '/models/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf',
]
FAST = {'gguf_q4_k_t16_v1', 'gguf_q4_k_qmicro_t16_v1', 'gguf_q5_k_t16_v1',
        'gguf_q5_k_qmicro_t16_v1', 'gguf_q6_k_t16_v1',
        'gguf_q6_k_t16_qmicro_planar_v1', 'gguf_q8_0_t16_v1',
        'gguf_q4_k_x8_v1', 'gguf_q5_k_x8_v1', 'gguf_q6_k_x8_v1',
        'gguf_q5_k_qmicro_planar_v1', 'gguf_expert_pack8_v1'}


def audit(path):
    from hipengine.kernels.backends import backend_package_capability
    from hipengine.loading.gguf import load_gguf_index
    from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
    from hipengine.loading.qwen35_gguf_policy import resolve_gguf_dense_flags
    from hipengine.loading.qwen35_gguf_materialize import (
        plan_qwen35_gguf_materialization, gguf_decode_repack_enabled)

    info = load_gguf_index(path)
    model_map = build_qwen35_gguf_tensor_map(info, strict=False)
    flags = resolve_gguf_dense_flags(
        BACKEND, info.file_type_name, capability_reader=backend_package_capability)
    import os
    veto = None if os.environ.get('AUDIT_NO_VETO') != '1' else False
    plan = plan_qwen35_gguf_materialization(
        model_map, decode_repack=gguf_decode_repack_enabled(None),
        repack_veto=veto, **flags)

    print(f'\n=== {path.rsplit("/", 1)[-1]}   file_type={info.file_type_name}   '
          f'decode_repack={gguf_decode_repack_enabled(None)} ===')
    rows = collections.defaultdict(lambda: [0, 0])   # (quant, layout, role, shape) -> count, bytes
    for slot, spec in list(plan.root_specs.items()):
        rows[(spec.source.ggml_type_name, spec.layout, slot, tuple(map(int, spec.source.shape)))][0] += 1
        rows[(spec.source.ggml_type_name, spec.layout, slot, tuple(map(int, spec.source.shape)))][1] += spec.source.nbytes
    for layer in plan.layer_specs:
        for slot, spec in layer.items():
            key = (spec.source.ggml_type_name, spec.layout, slot,
                   tuple(map(int, spec.source.shape)))
            rows[key][0] += 1
            rows[key][1] += spec.source.nbytes

    # group by (quant, layout)
    grouped = collections.defaultdict(lambda: [0, 0, set()])
    for (quant, layout, slot, shape), (count, nb) in rows.items():
        g = grouped[(quant, layout)]
        g[0] += count
        g[1] += nb
        g[2].add((slot, shape))
    fast_b = slow_b = 0
    print(f'  {"quant":9s}{"layout":34s}{"n":>5s}{"GB":>7s}  route      roles')
    for (quant, layout), (count, nb, slots) in sorted(
            grouped.items(), key=lambda kv: -kv[1][1]):
        fast = layout in FAST
        fast_b += nb if fast else 0
        slow_b += 0 if fast else nb
        roles = sorted({s.split('.')[-1] for s, _ in slots})
        shown = ','.join(roles[:5]) + ('…' if len(roles) > 5 else '')
        print(f'  {quant:9s}{layout:34s}{count:5d}{nb/1e9:7.2f}  '
              f'{"FAST" if fast else "gemv":9s}  {shown}')
    total = fast_b + slow_b
    print(f'  --> FAST {fast_b/1e9:.2f} GB ({100*fast_b/total:.1f}%)   '
          f'gemv {slow_b/1e9:.2f} GB ({100*slow_b/total:.1f}%)')
    return grouped


def main():
    results = {}
    for path in FILES:
        results[path] = audit(path)
    # Which fallback groups are biggest across the UD files?
    print('\n\n=== biggest GEMV-fallback groups in the UD files ===')
    agg = collections.defaultdict(float)
    for path, grouped in results.items():
        if 'UD' not in path:
            continue
        for (quant, layout), (count, nb, slots) in grouped.items():
            if layout not in FAST:
                agg[(path.rsplit('/', 1)[-1], quant, layout)] += nb
    for key, nb in sorted(agg.items(), key=lambda kv: -kv[1])[:14]:
        print(f'  {nb/1e9:6.2f} GB  {key[0]:32s} {key[1]:8s} {key[2]}')


if __name__ == '__main__':
    main()


def resident_bytes(path, veto):
    """Total planned device bytes for one file under a given veto setting."""
    from hipengine.kernels.backends import backend_package_capability
    from hipengine.loading.gguf import load_gguf_index
    from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
    from hipengine.loading.qwen35_gguf_policy import resolve_gguf_dense_flags
    from hipengine.loading.qwen35_gguf_materialize import (
        plan_qwen35_gguf_materialization, gguf_decode_repack_enabled,
        planned_qwen35_gguf_weight_allocation_nbytes)
    info = load_gguf_index(path)
    model_map = build_qwen35_gguf_tensor_map(info, strict=False)
    flags = resolve_gguf_dense_flags(
        BACKEND, info.file_type_name, capability_reader=backend_package_capability)
    plan = plan_qwen35_gguf_materialization(
        model_map, decode_repack=gguf_decode_repack_enabled(None),
        repack_veto=veto, **flags)
    total = 0
    for spec in plan.specs:
        try:
            total += sum(nb for _, nb in
                         planned_qwen35_gguf_weight_allocation_nbytes(spec))
        except Exception as exc:
            print(f'    ! {spec.slot_path} {spec.layout}: {type(exc).__name__}: {exc}')
    return total
