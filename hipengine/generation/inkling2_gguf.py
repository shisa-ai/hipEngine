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
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.inkling2_gguf import Inkling2GGUFWeights
from hipengine.loading.inkling2_gguf_device import Inkling2GGUFDeviceProjections
from hipengine.models.inkling2_mini import INKLING2_GGUF, INKLING2_MINI
from hipengine.runtime.inkling2_device_attention import Inkling2DeviceAttention
from hipengine.runtime.inkling2_device_convolution import Inkling2DeviceConvolution, Inkling2DeviceConvolutionTaps
from hipengine.runtime.inkling2_gguf import Inkling2GGUFText
from hipengine.tokenization.inkling2_gguf import inkling2_gguf_tokenizer


ROUTE_DEVICE = "HIPENGINE_INKLING2_DEVICE_ROUTES"
ROUTE_RESIDENT_DECODE = "HIPENGINE_INKLING2_RESIDENT_DECODE"
ROUTE_DECODE_GRAPH = "HIPENGINE_INKLING2_DECODE_GRAPH"
# Backends where the production plan's gate rows pass. Another backend keeps `fast` opt-in until
# its rows exist.
PRODUCTION_BACKENDS = ("hip_gfx1100", "hip_gfx1151")


def _route_flag(name, default=False):
    # An unset flag takes the profile's route: off for `fast`, on for `production`. `1` and `0` are
    # the caller's levers on both.
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value == "1"


def _device_routes_requested(default=False):
    return _route_flag(ROUTE_DEVICE, default)


def _resident_decode_requested(default=False):
    return _route_flag(ROUTE_RESIDENT_DECODE, default)


def _decode_graph_requested(default=False):
    return _route_flag(ROUTE_DECODE_GRAPH, default)


def _prefill_loop_defaults(width):
    # The scheduler hands over a whole block per round, so a block is collected and computed in one round.
    return {
        "prefill_decode_policy": "fair",
        "max_prefill_chunk_tokens": width,
        "round_prefill_token_budget": width,
    }


# The dense Q8_0 projections of a prompt block. `fast` reads them through the tile with one F16
# product per operand. That tile failed the production gate's long-context median (worklog
# 20261126T173747.968486Z-lhl-inkling2-mini-81272f.md), so `production` reads them through the tile
# that adds a residual F16 part to each operand, at 7e-6 of the RMS against its 2.9e-4 and 1.2 to
# 2.2 times its time (worklog
# 20261126T175737.613472Z-lhl-inkling2-mini-652e29.md). `strict` keeps the exact coltile kernel.
DENSE_EXACT = "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
DENSE_WIDE = "dense_wide64x128_f32_f32_out"
DENSE_WIDE_RESIDUAL = "dense_wide64x128_residual_f32_f32_out"


def _wide_dense_variant(fast, production):
    if not fast:
        return None
    return DENSE_WIDE_RESIDUAL if production else DENSE_WIDE


class Inkling2GGUFGenerator(Inkling2ReferenceGenerator):
    supports_vision = False
    # Packed target verification changed ordinary fast output on the GGUF prompt suite.
    packed_verification = False
    vision = None
    # Prompt blocks per profile. A prompt block of `fast` and `production` is 1,024 rows: with 256
    # rows the grouped expert kernel padded each expert's rows to 64 and did four times the work,
    # and every visit paid its host work four times more per token. First token after 575 prompt
    # tokens: 0.937 s at 256 rows against 0.631 at 1,024; after 2,057, 3.130 against 2.381 (W7900,
    # 2026-11-26, worklog 20261125T173325.971349Z-lhl-inkling2-mini-23550f). A wider block is a longer step for the other streams of
    # a multi-request server. `strict` keeps 64 rows, where it is 6 to 8% faster to the first token
    # than at 256 (worklog 20261125T023627.416022Z-lhl-inkling2-mini-91a27c).
    prefill_block_width = 64
    fast_prefill_block_width = 1024
    engine_loop_config_defaults = _prefill_loop_defaults(prefill_block_width)
    fast_engine_loop_config_defaults = _prefill_loop_defaults(fast_prefill_block_width)

    @classmethod
    def prefill_schedule(cls, fast):
        """The prompt block width and loop budgets of one profile, read without loading weights."""
        if fast:
            return cls.fast_prefill_block_width, cls.fast_engine_loop_config_defaults
        return cls.prefill_block_width, cls.engine_loop_config_defaults

    def __init__(self, *, backend, model_path, weight_index, model_plugin, max_sequence_length=None,
                 fast=False, production=False):
        # `production` shares the prompt blocks of `fast`. Its experts keep the selected F32 kernels
        # on every path, its eager decode keeps the ordered attention, and the device-resident
        # single-token traversal with its graph is its shipped route.
        fast = fast or production
        reduced = fast and not production
        reader = GGUFReader(weight_index.path)
        weights = Inkling2GGUFWeights(reader)
        m = weights.mapping
        if max_sequence_length is not None and not 0 < max_sequence_length <= m.max_context:
            raise ValueError(f"Inkling 2 GGUF context must be between 1 and {m.max_context}")
        self.max_sequence_length = m.max_context if max_sequence_length is None else max_sequence_length
        self.tokenizer, self.eos_token_id = inkling2_gguf_tokenizer(reader.info, m.vocab_size)
        self.model_path = reader.path
        self.prefill_block_width, self.engine_loop_config_defaults = self.prefill_schedule(fast)
        self._draft_provider = None
        runtime = get_hip_runtime()
        arch = hip_target_arch_for_backend(backend)

        def primitive(layer, variant):
            return partial(resolve(backend=backend, layer=layer, quant=INKLING2_MINI.default_quant, variant=variant),
                           target_arch=arch)

        attn = primitive("inkling2_target_attention", "tiled")
        split = primitive("inkling2_target_attention", "split")
        split_chunked = primitive("inkling2_target_attention", "split_chunked") if reduced else None
        block = primitive("inkling2_target_attention", "block")
        block_sigmoid = primitive("inkling2_target_attention", "block_sigmoid") if fast else None
        # One 32-lane group per (row, KV head) where the backend registers it. That launcher sends the
        # shapes it does not take to the tiled kernel.
        block_table = None
        if fast:
            waves = KernelKey(backend, "inkling2_target_attention", INKLING2_MINI.default_quant, "block_table_waves")
            block_table = primitive("inkling2_target_attention",
                                    "block_table_waves" if is_registered(waves) else "block_table_sigmoid")
        decode_table = primitive("inkling2_target_attention", "decode_table_sigmoid") if fast else None
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
                device_routes=_device_routes_requested(production), q8_dot_experts=reduced,
                wide_dense=_wide_dense_variant(fast, production)))
            model = self.model = Inkling2GGUFText(weights, projection=projection)
            model.resident_decode = _resident_decode_requested(production)
            model.decode_graph = _decode_graph_requested(production)
            model.attention_factory = lambda cap: Inkling2DeviceAttention(
                runtime=runtime, launch=attn, split=split, split_chunked=split_chunked, block=block,
                block_sigmoid=block_sigmoid, block_table=block_table, decode_table=decode_table, write=write,
                geometry=[
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
        # A graph still recorded against an open session goes before the buffers it names.
        drop = getattr(getattr(self, "model", None), "_drop_graph_state", None)
        if drop is not None:
            drop()
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
    wave_evidence = "worklog/entries/20261125T112427.985223Z-lhl-inkling2-mini-207009.md"
    wmma_evidence = "worklog/entries/20261125T030054.604272Z-lhl-inkling2-mini-1863be.md"
    production_evidence = "worklog/entries/20261125T144054.580076Z-lhl-inkling2-mini-6a7d16.md"
    shared_f32 = "selected_shared_register_fp4_f32_f32_out"
    tile_shared = "selected_tile4_shared_input_f32_f32_out"
    tile = "selected_tile4_f32_f32_out"
    wmma = "selected_wmma_f16_parts_f32_out"
    mixed_evidence = "worklog/entries/20261125T171057.976897Z-lhl-inkling2-mini-851bcb.md"
    dense_evidence = "worklog/entries/20261126T172532.740394Z-lhl-inkling2-mini-3ddb67.md"
    # (layer, scope, strict, fast, fast evidence, production). A production variant equal to the
    # strict one is the F32 kernel the device-resident decode runs, or the ordered path its eager
    # fallback keeps.
    scopes = (
        ("moe_linear", scope, strict, "selected_q8_dot_f32_f32_out", evidence, strict),
        ("moe_linear", "single_token_top8_nvfp4_shared_gate_n1152_or_2304_m2304", strict, shared_q8,
         shared_q8_evidence, shared_f32),
        ("moe_linear", "two_to_four_decode_rows_top8_nvfp4_gate_n1152_or_2304_m2304", strict, counted_q8,
         counted_q8_evidence, strict),
        ("moe_linear", "single_token_top8_nvfp4_split_down_n1152_m2304", strict, split_q8, split_q8_evidence, strict),
        ("moe_linear", "two_to_four_decode_rows_top8_nvfp4_split_down_n1152_m2304", strict, counted_split_q8,
         counted_split_q8_evidence, strict),
        ("moe_linear", "prompt_block_ge_192_rows_top8_nvfp4_gate_n2304_m2304", tile_shared, wmma, wmma_evidence, wmma),
        ("moe_linear", "prompt_block_ge_192_rows_top8_nvfp4_split_down_n1152_m2304", tile, wmma, wmma_evidence, wmma),
        ("moe_linear", "prompt_block_ge_192_rows_top8_q5_k_gate_n2304_m2304", tile_shared, wmma, mixed_evidence, wmma),
        ("moe_linear", "prompt_block_ge_192_rows_top8_q8_0_split_down_n1152_m2304", tile, wmma, mixed_evidence, wmma),
        ("linear", "prompt_block_q8_0_dense_n1024_n2304_n3072", DENSE_EXACT, DENSE_WIDE, dense_evidence,
         DENSE_WIDE_RESIDUAL),
        ("linear", "prompt_block_q8_0_dense_n384", DENSE_EXACT, DENSE_WIDE, dense_evidence, DENSE_WIDE_RESIDUAL),
        ("inkling2_expert_pointwise", "selected_split_gate_up", "split_activation_host", "split_activation_device",
         activation_evidence, "split_activation_device"),
        ("inkling2_target_attention", "decode_live_ge_513", "split", "split_chunked", chunked_evidence, "split"),
        ("inkling2_target_attention", "committed_projected_prefill_gate", "block", "block_sigmoid", gate_evidence,
         "block_sigmoid"),
        ("inkling2_controller", "committed_block_controllers", "host_blend_norm", "blend_norm", controller_evidence,
         "blend_norm"),
        ("inkling2_target_attention", "resident_block_relative_bias", "block_sigmoid", "block_table_waves",
         wave_evidence, "block_table_waves"),
        ("inkling2_attention_prepare", "resident_block_inputs", "host_clip", "clip", resident_evidence, "clip"),
        ("inkling2_target_attention", "resident_single_token_relative_bias", "split", "split", None,
         "decode_table_sigmoid"),
    )
    def selection(profile, layer, name, exact, reduced, reduced_evidence, production):
        """One scope's selection under a profile; every non-strict one names its strict fallback."""
        # The registry quant of an expert bank follows its scope: NVFP4 unless the scope names the layout.
        registry_quant = "inkling2_mixed"
        if layer == "moe_linear":
            registry_quant = next((f"gguf_{tag}" for tag in ("q5_k", "q8_0") if f"_{tag}_" in name), "gguf_nvfp4")
        elif layer == "linear":
            registry_quant = "gguf_q8_0"
        if profile == "strict":
            return VariantSelection(layer, name, exact, None, None, registry_quant=registry_quant)
        if profile == "fast":
            return VariantSelection(layer, name, reduced, exact, reduced_evidence, registry_quant=registry_quant)
        evidence_artifact = production_evidence if production != exact else None
        return VariantSelection(layer, name, production, exact, evidence_artifact, registry_quant=registry_quant)

    registered = registered_runtime_profile_keys()
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        for profile in ("strict", "fast", *((("production",) if backend in PRODUCTION_BACKENDS else ()))):
            if RuntimeProfileKey(model, backend, quant, profile) in registered:
                continue
            register_runtime_profile_plan(
                model=model, backend=backend, quant=quant, profile=profile,
                plan=RuntimeProfilePlan(
                    selections=tuple(selection(profile, *row) for row in scopes),
                    kv_policy="fp32",
                    graph_policy="resident_single_token_whole_token_graph" if profile == "production" else "eager",
                    factory=partial(Inkling2GGUFGenerator, backend=backend, fast=profile == "fast",
                                    production=profile == "production"),
                ),
            )


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register_text_generator(model=INKLING2_GGUF.name, backend=_backend, quant=INKLING2_GGUF.default_quant,
                            factory=partial(Inkling2GGUFGenerator, backend=_backend))
