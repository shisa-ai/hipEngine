"""Admission-aware Qwen4Exp context capacity, independent of QSA selection budget."""
from dataclasses import replace

from hipengine.loading.qwen4_exp_materialize import (
    Qwen4ExpMemoryAdmissionPlan,
    Qwen4ExpResidencyPlan,
    plan_qwen4_exp_memory_admission,
)


def resolve_qwen4_exp_context(
    residency: Qwen4ExpResidencyPlan, *, available_device_bytes: int,
    requested_context: int | None = None, native_context_length: int | None = None,
    resident_capacity: int = 1, scratch_bytes_per_runner: int = 4 * 1024**3,
    reserve_bytes: int = 4 * 1024**3,
) -> Qwen4ExpMemoryAdmissionPlan:
    native = min(residency.config.context_length,
                 residency.config.context_length if native_context_length is None else int(native_context_length))
    if native <= 0 or resident_capacity <= 0:
        raise ValueError("native context and resident capacity must be positive")
    minimum = residency.config.qsa_compression_ratio
    if native < minimum:
        raise ValueError("native context must contain one QSA compression block")

    def admission(context):
        plan = plan_qwen4_exp_memory_admission(
            residency,available_device_bytes=available_device_bytes,
            context_tokens=context,resident_capacity=resident_capacity,
            scratch_bytes=scratch_bytes_per_runner*resident_capacity,reserve_bytes=reserve_bytes)
        # The logical memory planner omits the runner's physical 256-token KV page tail.
        kv = ((context+255)//256*256)*residency.config.bf16_kv_bytes_per_token*resident_capacity
        return replace(plan,kv_bytes=kv,required_bytes=plan.required_bytes+kv-plan.kv_bytes)

    if requested_context is not None:
        requested = int(requested_context)
        if not minimum <= requested <= native:
            raise ValueError(f"Qwen4Exp allocated context must be within {minimum}..{native}")
        plan = admission(requested)
        if not plan.passed:
            raise MemoryError(
                f"Qwen4Exp context {requested} at c{resident_capacity} needs "
                f"{plan.required_bytes} bytes including reserve; available {available_device_bytes}")
        return plan
    if not admission(minimum).passed:
        raise MemoryError("Qwen4Exp weights, resident scratch and reserve do not fit even minimum context")
    low,high = minimum,native
    while low < high:
        mid = (low+high+1)//2
        if admission(mid).passed:
            low = mid
        else:
            high = mid-1
    return admission(low)
