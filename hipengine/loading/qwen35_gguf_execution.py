"""Load-time GGUF route dependencies and resident compatibility checks.

No payload reads, allocation, backend import or numerical permission. Native
sessions check their resident once at construction, not during model execution.
"""
from dataclasses import dataclass, replace


# _enqueue_native_rows_model's full weight-consuming call plan. Projection,
# auxiliary, embedding, router and selected-call ABIs stay with the accepted
# consumer/selected owners; this is dependency enumeration, not dispatch.
NATIVE_EXECUTION_ROLE_CLASSES = frozenset({
    "projection", "recurrent_alpha_beta", "norm", "gdn_norm", "gdn_scalar", "conv1d",
    "token_embedding", "lm_head", "moe_router", "moe_experts",
})


def execution_operations(routes=("eager",)):
    from hipengine.loading.qwen35_gguf_admission import DEFAULT_AR_OPERATIONS
    result = list(DEFAULT_AR_OPERATIONS)
    for route in routes:
        if route not in {"eager", "native_rows", "native_graph"}:
            raise ValueError(f"unknown GGUF execution route {route!r}")
        if route != "eager" and "ar_decode_native_rows" not in result:
            result.append("ar_decode_native_rows")
    return tuple(result)


def resident_slots(weights):
    return {**{"root." + slot: weight for slot, weight in weights.root_weights.items()},
            **{f"layers.{layer.layer_id}.{slot}": weight
               for layer in weights.layers for slot, weight in layer.weights.items()}}


def resident_snapshot(weights):
    """Bind physical alias/view ownership as well as logical specs."""
    return tuple((slot, replace(weight.spec, slot_path=slot), weight.backend,
                  tuple((name, int(a.tensor.ptr), int(a.buffer.ptr), int(a.buffer.nbytes),
                         tuple(a.tensor.shape), a.tensor.dtype, a.tensor.device, a.owns_buffer)
                        for name, a in sorted(weight.allocations.items())))
                 for slot, weight in sorted(resident_slots(weights).items()))


@dataclass(frozen=True)
class ResidentExecutionBinding:
    certificate: object
    config: object
    preset: str | None
    snapshot: tuple


def bind_resident_execution(weights):
    """Loader publication only; this snapshot does not mint authorization."""
    return ResidentExecutionBinding(weights.admission_certificate, weights.config,
                                    weights.artifact_preset_key, resident_snapshot(weights))


def authorize_native_execution(weights, *, backend, rows, recurrent_state_dtype="f32"):
    """Check actual resident calls against pre-allocation F4 qualification.

    Rebind invocation descriptors to existing specs, never run the planner or
    mint a certificate after allocation. Native scratch.norm supplies BF16.
    This is a construction-time check; private layer calls do not use it.
    """
    from hipengine.loading.qwen35_gguf_admission import (
        Qwen35GGUFAdmissionError, QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS as operation,
        _coverage_for, _role_class_for_slot, certificate_covers_artifact,
        qwen35_gguf_planned_weight_record,
    )
    from hipengine.loading.gguf_selected_contract import default_selected_call_intents, bind_selected_call
    from hipengine.loading.qwen35_gguf_consumer_surface import OPERATION_ROW_LIMITS

    def refuse(reason):
        raise Qwen35GGUFAdmissionError("ar_decode_native_rows execution refused before state mutation/device call: " + reason)

    binding = getattr(weights, "execution_binding", None)
    certificate = getattr(weights, "admission_certificate", None)
    if binding is None or certificate is None or binding.certificate != certificate:
        refuse("missing or stale pre-certified resident execution binding")
    contract = certificate.plan_contract
    if (contract is None or certificate.slot_filter is not None or contract.slot_filter is not None
            or not set(execution_operations(("native_rows",))) <= set(certificate.operations)):
        refuse("partial certificate cannot authorize a full model route")
    if (weights.backend != backend or binding.config != weights.config
            or binding.preset != weights.artifact_preset_key
            or binding.snapshot != resident_snapshot(weights)):
        refuse("backend/resident/geometry/alias ownership changed")
    low, high = OPERATION_ROW_LIMITS[operation]
    if not low <= int(rows) <= high:
        refuse(f"rows {rows} outside certified [{low}, {high}]")
    specs = {}
    records = {}
    invocations = []
    required = []
    try:
        for slot, weight in resident_slots(weights).items():
            spec = replace(weight.spec, slot_path=slot)
            specs[slot] = spec
            if weight.backend != backend or set(weight.allocations) != set(spec.allocation_names):
                refuse(f"{slot}: backend or deferred/missing allocations")
            if any(int(a.tensor.ptr) <= 0 for a in weight.allocations.values()):
                refuse(f"{slot}: invalid allocation")
            records[slot] = qwen35_gguf_planned_weight_record(spec)
            role = _role_class_for_slot(slot)
            if role not in NATIVE_EXECUTION_ROLE_CLASSES:
                refuse(f"{slot}: no declared native execution dependency")
            if role == "moe_experts":
                continue
            coverage = _coverage_for(operation, role, spec.layout, spec.source.ggml_type_name)
            if coverage is None:
                refuse(f"{slot}: no actual native consumer")
            invocation = coverage.invocation(spec, backend=backend, config=weights.config,
                                             recurrent_state_dtype=recurrent_state_dtype)
            invocations.append(invocation)
            required.append((slot, operation))
        expert_quants = {slot: spec.quant_key for slot, spec in specs.items()
                         if _role_class_for_slot(slot) == "moe_experts"}
        intents = default_selected_call_intents(expert_quants, (operation,),
                                               lanes_per_token=max(1, int(weights.config.expert_used_count)))
        selected = tuple(bind_selected_call(intent, specs, records, backend=backend,
                                            row_limits=OPERATION_ROW_LIMITS[operation]) for intent in intents)
        intended = replace(contract, operations=(operation,), f32_input_operations=(),
                           resident_plan_records=tuple(records.values()), required_plan_slots=tuple(specs),
                           invocations=tuple(invocations), required_invocations=tuple(required),
                           selected_invocations=selected, required_selected_intents=intents,
                           operation_scope_refusals=())
        if set(specs) != set(contract.required_plan_slots) or not certificate_covers_artifact(
                certificate, manifest_fingerprint=certificate.manifest_fingerprint,
                plan_contract=intended, backend=backend, operations=(operation,)):
            refuse("certificate does not cover actual full native invocation intent/operands/geometry")
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        refuse(str(exc))
    return (binding, int(rows), recurrent_state_dtype, intended.invocation_digest)
