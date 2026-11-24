"""Public packed HIP text execution for Inkling GGUF checkpoints."""

import os
from contextlib import ExitStack
from dataclasses import replace
from functools import partial

from tokenizers.decoders import DecodeStream

from hipengine.core.hip import get_hip_runtime
from hipengine.core.rocblas import Rocblas
from hipengine.execution_profiles import (
    RuntimeProfileKey, RuntimeProfilePlan, VariantSelection, register_runtime_profile_plan, registered_runtime_profile_keys,
)
from hipengine.generation.inkling2_gguf_resident import Inkling2ResidentRunner
from hipengine.generation.inkling2_mini import Inkling2ReferenceGenerator
from hipengine.generation.registry import DecodeState, GenerationStreamChunk, GenerationTelemetry, register_text_generator
from hipengine.kernels.backends import hip_target_arch_for_backend
from hipengine.kernels.hip_gfx1100.attention import inkling2_target_attention  # noqa: F401
from hipengine.kernels.hip_gfx1100.norm import (  # noqa: F401
    inkling2_attention_prepare,
    inkling2_controller,
    inkling2_residual,
)
from hipengine.kernels.hip_gfx1100.quant import inkling2_short_conv  # noqa: F401
from hipengine.kernels.registry import resolve
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.inkling2_gguf import Inkling2GGUFWeights
from hipengine.loading.inkling2_gguf_device import Inkling2GGUFDeviceProjections
from hipengine.models.inkling2_mini import INKLING2_GGUF, INKLING2_MINI
from hipengine.runtime.inkling2_device_attention import Inkling2DeviceAttention
from hipengine.runtime.inkling2_device_convolution import Inkling2DeviceConvolution, Inkling2DeviceConvolutionTaps
from hipengine.runtime.inkling2_gguf import Inkling2GGUFText
from hipengine.tokenization.inkling2_gguf import inkling2_gguf_tokenizer


def _device_routes_requested() -> bool:
    return os.environ.get("HIPENGINE_INKLING2_DEVICE_ROUTES", "0") == "1"


def _resident_decode_requested() -> bool:
    return os.environ.get("HIPENGINE_INKLING2_RESIDENT_DECODE", "0") == "1"


class Inkling2GGUFGenerator(Inkling2ReferenceGenerator):
    supports_vision = False
    # Packed target verification changed ordinary fast output on the GGUF prompt suite.
    packed_verification = False
    vision = None
    prefill_block_width = 64
    engine_loop_config_defaults = {
        "prefill_decode_policy": "fair",
        "max_prefill_chunk_tokens": 64,
        "round_prefill_token_budget": 64,
    }

    def __init__(self, *, backend, model_path, weight_index, model_plugin, max_sequence_length=None,
                 fast=False):
        reader = GGUFReader(weight_index.path)
        weights = Inkling2GGUFWeights(reader)
        m = weights.mapping
        if max_sequence_length is not None and not 0 < max_sequence_length <= m.max_context:
            raise ValueError(f"Inkling 2 GGUF context must be between 1 and {m.max_context}")
        self.max_sequence_length = m.max_context if max_sequence_length is None else max_sequence_length
        self.tokenizer, self.eos_token_id = inkling2_gguf_tokenizer(reader.info, m.vocab_size)
        self.model_path = reader.path
        self._draft_provider = None
        runtime = get_hip_runtime()
        arch = hip_target_arch_for_backend(backend)

        def primitive(layer, variant):
            return partial(resolve(backend=backend, layer=layer, quant=INKLING2_MINI.default_quant, variant=variant),
                           target_arch=arch)

        attn = primitive("inkling2_target_attention", "tiled")
        split = primitive("inkling2_target_attention", "split")
        split_chunked = primitive("inkling2_target_attention", "split_chunked") if fast else None
        block = primitive("inkling2_target_attention", "block")
        block_sigmoid = primitive("inkling2_target_attention", "block_sigmoid") if fast else None
        block_table = primitive("inkling2_target_attention", "block_table_sigmoid") if fast else None
        write = primitive("inkling2_target_kv_write", "default")
        conv = primitive("inkling2_short_conv", "default")
        conv_block = primitive("inkling2_short_conv", "block")
        conv_dynamic = primitive("inkling2_short_conv", "projected_dynamic")
        visits = sorted(m.plan.visits, key=lambda v: v.state_slot)
        channels = max(m.hidden_size, max(layer.kv_heads * layer.head_dim for layer in m.layers))
        with ExitStack() as owned:
            blas = Rocblas.load()
            owned.callback(blas.close)
            projection = owned.enter_context(Inkling2GGUFDeviceProjections(
                weights, runtime=runtime, blas=blas, backend=backend, fast_experts=fast, fast_controllers=fast,
                device_routes=_device_routes_requested()))
            model = self.model = Inkling2GGUFText(weights, projection=projection)
            model.resident_decode = _resident_decode_requested()
            model.attention_factory = lambda cap: Inkling2DeviceAttention(
                runtime=runtime, launch=attn, split=split, split_chunked=split_chunked, block=block,
                block_sigmoid=block_sigmoid, block_table=block_table, write=write, geometry=[
                    (min(cap, m.sliding_window) if v.is_local else cap, *model.head_shape(v),
                     m.sliding_window if v.is_local else m.relative_extent) for v in visits
                ])
            tap_cache = Inkling2DeviceConvolutionTaps(runtime)
            owned.callback(tap_cache.close)
            model.convolution_factory = lambda: Inkling2DeviceConvolution(
                runtime=runtime, launch=conv, launch_block=conv_block, launch_dynamic=conv_dynamic,
                channels=channels, width=m.conv_taps, tap_cache=tap_cache)
            self._owned = owned.pop_all()

    @property
    def supports_speculative_mtp(self):
        return self._draft_provider is not None

    def attach_speculative_provider(self, provider):
        if self._draft_provider is not None:
            raise RuntimeError("Inkling GGUF already owns a DSpark provider")
        self._owned.callback(provider.close)
        self._draft_provider = provider

    def generate_speculative_mtp_detailed(self, request):
        if self._draft_provider is None:
            raise NotImplementedError("Inkling GGUF DSpark requires a compatible draft_model GGUF sidecar")
        return self._draft_provider.generate_detailed(request)

    def create_resident_model_runner(self, *, capacity=None):
        return Inkling2ResidentRunner(self, capacity=4 if capacity is None else capacity)

    def _prepare_text_rows(self, request, *, speculative=False):
        if request.kv_storage == "fp32":
            request = replace(request, kv_storage="auto")
        return super()._prepare_text_rows(request, speculative=speculative)

    def stream_detailed(self, request):
        request, rows = self._prepare_text_rows(request)
        if len(rows) != 1:
            raise ValueError("Inkling 2 GGUF streaming requires exactly one prompt")
        decoder = DecodeStream(skip_special_tokens=True)
        prefixes = {sequence[:n] for sequence in request.stop_token_sequences for n in range(1, len(sequence))}
        max_prefix = max(map(len, prefixes), default=0)
        pending = []
        published = ""
        with ExitStack() as owned:
            steps = self._iterate_row(request, 0, rows[0])
            owned.callback(steps.close)
            while True:
                try:
                    token = next(steps)
                except StopIteration as done:
                    output = done.value
                    finish = output.finish_details
                    trim = len(finish.stop_sequence) or int(finish.eos_token_id is not None)
                    ids = output.generated_token_ids
                    visible = ids[:-trim] if trim else ids
                    text = self.detokenize(visible)
                    if not text.startswith(published):
                        raise RuntimeError("Inkling 2 GGUF incremental text disagrees with full decoding")
                    yield GenerationStreamChunk(text=text[len(published):], finish_details=finish,
                                                telemetry=output.telemetry, generated_token_ids=ids)
                    return
                pending.append(token)
                hold = next((n for n in range(min(len(pending), max_prefix), 0, -1)
                             if tuple(pending[-n:]) in prefixes), 0)
                emit_count = len(pending) - hold
                text = "".join(decoder.step(self.tokenizer, tok) or "" for tok in pending[:emit_count])
                del pending[:emit_count]
                if text:
                    published += text
                    yield GenerationStreamChunk(text=text)

    def _decorate_output(self, out):
        diagnostics = {} if out.telemetry is None else dict(out.telemetry.diagnostics)
        diagnostics.update(
            execution_path="inkling2_gguf_packed_hip", kv_storage="fp32",
            speculation="dspark" if "speculative_mtp" in diagnostics else "off",
            expert_mix="ordered_device" if self.model.ffn_device_output else "ordered_host",
            expert_activation="split_device" if self.model.projection.supports_device_activation else "split_host",
        )
        return replace(out, telemetry=GenerationTelemetry(
            decode_state=DecodeState(prompt_tokens=out.prompt_tokens, generated_tokens=len(out.generated_token_ids)),
            diagnostics=diagnostics,
        ))

    def close(self):
        self._owned.close()


def register_inkling2_gguf_profiles():
    model, quant = INKLING2_GGUF.name, INKLING2_GGUF.default_quant
    strict = "selected_register_fp4_f32_f32_out"
    scope = "single_token_top8_nvfp4_n1152_or_2304_m2304"
    evidence = "worklog/entries/20261122T042314.583479Z-lhl-inkling2-mini-fbe423.md"
    activation_evidence = "worklog/entries/20261122T051118.872751Z-lhl-inkling2-mini-978416.md"
    shared_q8 = "selected_shared_input_q8_dot_f32_f32_out"
    shared_q8_evidence = "worklog/entries/20261123T025317.467504Z-lhl-inkling2-mini-e38355.md"
    counted_q8 = "selected_counted_shared_input_q8_dot_f32_f32_out"
    counted_q8_evidence = "worklog/entries/20261123T045710.390593Z-lhl-inkling2-mini-67cb1a.md"
    split_q8 = "selected_split_activation_q8_dot_f32_f32_out"
    split_q8_evidence = "worklog/entries/20261123T024426.568443Z-lhl-inkling2-mini-3f4a1f.md"
    counted_split_q8 = "selected_counted_split_activation_q8_dot_f32_f32_out"
    counted_split_q8_evidence = "worklog/entries/20261123T052005.072756Z-lhl-inkling2-mini-b13bab.md"
    chunked_evidence = "worklog/entries/20261122T161527.676605Z-lhl-inkling2-mini-854a03.md"
    gate_evidence = "worklog/entries/20261123T032340.215296Z-lhl-inkling2-mini-9e502e.md"
    controller_evidence = "worklog/entries/20261124T000430.839711Z-lhl-inkling2-mini-d2aa61.md"
    resident_evidence = "worklog/entries/20261124T093241.447865Z-lhl-inkling2-mini-4b3cf8.md"
    registered = registered_runtime_profile_keys()
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        for profile, variant in (("strict", strict), ("fast", "selected_q8_dot_f32_f32_out")):
            if RuntimeProfileKey(model, backend, quant, profile) in registered:
                continue
            fast = profile == "fast"
            register_runtime_profile_plan(
                model=model, backend=backend, quant=quant, profile=profile,
                plan=RuntimeProfilePlan(
                    selections=(
                        VariantSelection("moe_linear", scope, variant, strict if fast else None,
                                         evidence if fast else None, registry_quant="gguf_nvfp4"),
                        VariantSelection("moe_linear", "single_token_top8_nvfp4_shared_gate_n1152_or_2304_m2304",
                                         shared_q8 if fast else strict, strict if fast else None,
                                         shared_q8_evidence if fast else None, registry_quant="gguf_nvfp4"),
                        VariantSelection("moe_linear", "two_to_four_decode_rows_top8_nvfp4_gate_n1152_or_2304_m2304",
                                         counted_q8 if fast else strict, strict if fast else None,
                                         counted_q8_evidence if fast else None, registry_quant="gguf_nvfp4"),
                        VariantSelection("moe_linear", "single_token_top8_nvfp4_split_down_n1152_m2304",
                                         split_q8 if fast else strict, strict if fast else None,
                                         split_q8_evidence if fast else None, registry_quant="gguf_nvfp4"),
                        VariantSelection("moe_linear", "two_to_four_decode_rows_top8_nvfp4_split_down_n1152_m2304",
                                         counted_split_q8 if fast else strict, strict if fast else None,
                                         counted_split_q8_evidence if fast else None, registry_quant="gguf_nvfp4"),
                        VariantSelection("inkling2_expert_pointwise", "selected_split_gate_up",
                                         "split_activation_device" if fast else "split_activation_host",
                                         "split_activation_host" if fast else None,
                                         activation_evidence if fast else None, registry_quant="inkling2_mixed"),
                        VariantSelection("inkling2_target_attention", "decode_live_ge_513",
                                         "split_chunked" if fast else "split", "split" if fast else None,
                                         chunked_evidence if fast else None, registry_quant="inkling2_mixed"),
                        VariantSelection("inkling2_target_attention", "committed_projected_prefill_gate",
                                         "block_sigmoid" if fast else "block", "block" if fast else None,
                                         gate_evidence if fast else None, registry_quant="inkling2_mixed"),
                        VariantSelection("inkling2_controller", "committed_block_controllers",
                                         "blend_norm" if fast else "host_blend_norm", "host_blend_norm" if fast else None,
                                         controller_evidence if fast else None, registry_quant="inkling2_mixed"),
                        VariantSelection("inkling2_target_attention", "resident_block_relative_bias",
                                         "block_table_sigmoid" if fast else "block_sigmoid",
                                         "block_sigmoid" if fast else None, resident_evidence if fast else None,
                                         registry_quant="inkling2_mixed"),
                        VariantSelection("inkling2_attention_prepare", "resident_block_inputs",
                                         "clip" if fast else "host_clip", "host_clip" if fast else None,
                                         resident_evidence if fast else None, registry_quant="inkling2_mixed"),
                    ),
                    kv_policy="fp32", graph_policy="eager",
                    factory=partial(Inkling2GGUFGenerator, backend=backend, fast=fast),
                ),
            )


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register_text_generator(model=INKLING2_GGUF.name, backend=_backend, quant=INKLING2_GGUF.default_quant,
                            factory=partial(Inkling2GGUFGenerator, backend=_backend))
