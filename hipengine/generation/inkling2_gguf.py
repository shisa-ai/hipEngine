"""Public packed HIP text execution for Inkling GGUF checkpoints."""

from contextlib import ExitStack
from dataclasses import replace
from functools import partial

from hipengine.core.hip import get_hip_runtime
from hipengine.core.rocblas import Rocblas
from hipengine.generation.inkling2_mini import Inkling2ReferenceGenerator
from hipengine.generation.registry import DecodeState, GenerationTelemetry, register_text_generator
from hipengine.kernels.backends import hip_target_arch_for_backend
from hipengine.kernels.hip_gfx1100.attention import inkling2_target_attention  # noqa: F401
from hipengine.kernels.hip_gfx1100.quant import inkling2_short_conv  # noqa: F401
from hipengine.kernels.registry import resolve
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.inkling2_gguf import Inkling2GGUFWeights
from hipengine.loading.inkling2_gguf_device import Inkling2GGUFDeviceProjections
from hipengine.models.inkling2_mini import INKLING2_GGUF, INKLING2_MINI
from hipengine.runtime.inkling2_device_attention import Inkling2DeviceAttention
from hipengine.runtime.inkling2_device_convolution import Inkling2DeviceConvolution
from hipengine.runtime.inkling2_gguf import Inkling2GGUFText
from hipengine.tokenization.inkling2_gguf import inkling2_gguf_tokenizer


class Inkling2GGUFGenerator(Inkling2ReferenceGenerator):
    supports_vision = False
    supports_speculative_mtp = False
    vision = None
    prefill_block_width = 64

    def __init__(self, *, backend, model_path, weight_index, model_plugin, max_sequence_length=None):
        reader = GGUFReader(weight_index.path)
        weights = Inkling2GGUFWeights(reader)
        m = weights.mapping
        if max_sequence_length is not None and not 0 < max_sequence_length <= m.max_context:
            raise ValueError(f"Inkling 2 GGUF context must be between 1 and {m.max_context}")
        self.max_sequence_length = m.max_context if max_sequence_length is None else max_sequence_length
        self.tokenizer, self.eos_token_id = inkling2_gguf_tokenizer(reader.info, m.vocab_size)
        self.model_path = reader.path
        runtime = get_hip_runtime()
        arch = hip_target_arch_for_backend(backend)

        def primitive(layer, variant):
            return partial(resolve(backend=backend, layer=layer, quant=INKLING2_MINI.default_quant, variant=variant),
                           target_arch=arch)

        attn = primitive("inkling2_target_attention", "tiled")
        split = primitive("inkling2_target_attention", "split")
        block = primitive("inkling2_target_attention", "block")
        write = primitive("inkling2_target_kv_write", "default")
        conv = primitive("inkling2_short_conv", "default")
        conv_block = primitive("inkling2_short_conv", "block")
        visits = sorted(m.plan.visits, key=lambda v: v.state_slot)
        channels = max(m.hidden_size, max(layer.kv_heads * layer.head_dim for layer in m.layers))
        with ExitStack() as owned:
            blas = Rocblas.load()
            owned.callback(blas.close)
            projection = owned.enter_context(Inkling2GGUFDeviceProjections(weights, runtime=runtime, blas=blas,
                                                                          backend=backend))
            model = self.model = Inkling2GGUFText(weights, projection=projection)
            model.attention_factory = lambda cap: Inkling2DeviceAttention(
                runtime=runtime, launch=attn, split=split, block=block, write=write, geometry=[
                    (min(cap, m.sliding_window) if v.is_local else cap, *model.head_shape(v),
                     m.sliding_window if v.is_local else m.relative_extent) for v in visits
                ])
            model.convolution_factory = lambda: Inkling2DeviceConvolution(
                runtime=runtime, launch=conv, launch_block=conv_block, channels=channels, width=m.conv_taps)
            self._owned = owned.pop_all()

    def _generate_batch(self, request, *, drafter=None):
        if request.kv_storage == "fp32":
            request = replace(request, kv_storage="auto")
        return super()._generate_batch(request, drafter=drafter)

    def _generate_row_with_session(self, request, row, ids, session, draft, **options):
        out = super()._generate_row_with_session(request, row, ids, session, draft, **options)
        return replace(out, telemetry=GenerationTelemetry(
            decode_state=DecodeState(prompt_tokens=len(ids), generated_tokens=len(out.generated_token_ids)),
            diagnostics={"execution_path": "inkling2_gguf_packed_hip", "kv_storage": "fp32", "speculation": "off"},
        ))

    def close(self):
        self._owned.close()


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register_text_generator(model=INKLING2_GGUF.name, backend=_backend, quant=INKLING2_GGUF.default_quant,
                            factory=partial(Inkling2GGUFGenerator, backend=_backend))
