"""Full execution scopes over the shared GGUF invocation owners.

No payload reads, planning, allocation, backend import or numerical permission.
Load scopes are diagnostic; only a full pre-certified scope can authorize a
native model entry. Native eager/capture/replay callers share this boundary.
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


def native_scratch_identity(scratch, config, *, rows, recurrent_state_dtype):
    """Validate raw-buffer extents and state storage before any native writer.

    Scratch has no element dtype on DeviceBuffer. Its allocator-owned reset
    arrays record state storage; exact byte extents plus stable pointer owners
    bind that declaration. This does not infer numerical contents.
    """
    from hipengine.loading.qwen35_gguf_admission import Qwen35GGUFAdmissionError
    def fail(reason):
        raise Qwen35GGUFAdmissionError("ar_decode_native_rows scratch contract: " + reason)
    if scratch is None or int(getattr(scratch, "slot_count", 0)) < rows:
        fail("missing or insufficient row capacity")
    if str(getattr(getattr(scratch, "recurrent_zero", None), "dtype", "")) != (
            "float16" if recurrent_state_dtype == "fp16" else "float32"):
        fail("recurrent state storage does not match the certified operand")
    qkv = 2 * config.ssm_group_count * config.ssm_state_size + config.ssm_inner_size
    extents = {"norm": config.hidden_size * 2, "post_norm": config.hidden_size * 2,
               "linear_qkv": qkv * 2, "linear_z": config.ssm_inner_size * 2,
               "linear_alpha": config.ssm_time_step_rank * 2, "linear_beta": config.ssm_time_step_rank * 2,
               "conv_out": qkv * 4, "recurrent_out": config.ssm_inner_size * 4,
               "recurrent_bf16": config.ssm_inner_size * 2}
    if config.is_moe:
        lanes = int(config.expert_used_count)
        if lanes <= 0 or int(getattr(scratch, "moe_selected_rows_capacity", 0)) < rows * lanes:
            fail("selected-expert row capacity/ownership mismatch")
        extents.update({"moe_router_logits": config.expert_count * 4,
                        "moe_selected_experts": lanes * 4, "moe_routing_weights": lanes * 4,
                        "ffn_gate_up": lanes * config.expert_feed_forward_length * 4,
                        "ffn_intermediate": lanes * config.expert_feed_forward_length * 2,
                        "moe_down_out": lanes * config.hidden_size * 2})
    signature = []
    def check(name, buf, size):
        if buf is None or int(getattr(buf, "ptr", 0)) <= 0 or int(getattr(buf, "nbytes", 0)) < size:
            fail(f"{name}: missing/undersized operand for rows={rows}")
        signature.append((name, int(buf.ptr), int(buf.nbytes)))
    for name, size in extents.items():
        check(name, getattr(scratch, name, None), rows * size)
    states = []
    for layer, kind in enumerate(config.layer_types):
        if kind != "linear_attention":
            continue
        for name, size in (("layer_conv_states", qkv * config.ssm_conv_kernel * 4),
                           ("layer_recurrent_states", config.ssm_inner_size * config.ssm_state_size *
                            (2 if recurrent_state_dtype == "fp16" else 4))):
            buffers = getattr(scratch, name, ())
            buf = buffers[layer] if layer < len(buffers) else None
            check(f"{name}.{layer}", buf, rows * size)
            extent = (int(buf.ptr), int(buf.ptr) + int(buf.nbytes))
            if any(extent[0] < end and start < extent[1] for start, end in states):
                fail("recurrent/conv state owners overlap")
            states.append(extent)
    # Graphs retain every physical scratch buffer, not only the sampled ABI
    # operands above. Contents such as positions are intentionally not hashed.
    for index, buf in enumerate(getattr(scratch, "buffers", ())):
        signature.append((f"buffer.{index}", int(buf.ptr), int(buf.nbytes)))
    return (id(scratch), int(scratch.slot_count), tuple(signature))


def authorize_native_execution(weights, *, backend, rows, recurrent_state_dtype="f32"):
    """Check actual resident calls against pre-allocation F4 qualification.

    Rebind invocation descriptors to existing specs, never run the planner or
    mint a certificate after allocation. Native scratch.norm supplies BF16.
    Returned immutable identity can be pinned by graph owners, not widened.
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
