"""gfx1100 Inkling 2 generator with HIP projections and host sequence state."""

from contextlib import ExitStack
from threading import Lock

from hipengine.core.hip import get_hip_runtime
from hipengine.core.rocblas import Rocblas
from hipengine.generation.inkling2_mini import Inkling2ReferenceGenerator
from hipengine.generation.registry import register_text_generator
from hipengine.kernels.hip_gfx1100.attention import inkling2_block_attention, inkling2_target_attention  # noqa: F401
from hipengine.kernels.hip_gfx1100.quant import inkling2_experts, inkling2_short_conv  # noqa: F401
from hipengine.kernels.registry import resolve
from hipengine.loading.inkling2_device_dense import Inkling2DeviceDense
from hipengine.loading.inkling2_device_experts import Inkling2DeviceExperts
from hipengine.models.inkling2_mini import INKLING2_MINI
from hipengine.runtime.inkling2_device_attention import Inkling2DeviceAttention
from hipengine.runtime.inkling2_device_block_attention import Inkling2DeviceBlockAttention
from hipengine.runtime.inkling2_device_convolution import Inkling2DeviceConvolution
from hipengine.runtime.inkling2_device_draft_kv import Inkling2DeviceDraftKV


class Inkling2Gfx1100Generator(Inkling2ReferenceGenerator):
    packed_verification = True
    def __init__(self, *, model_path, weight_index, model_plugin, max_sequence_length=None):
        super().__init__(model_path=model_path, weight_index=weight_index, model_plugin=model_plugin,
                         max_sequence_length=max_sequence_length)
        runtime = get_hip_runtime()
        launch = resolve(backend="hip_gfx1100", layer="inkling2_expert_projection", quant=INKLING2_MINI.default_quant,
                         variant="default")
        grouped = resolve(backend="hip_gfx1100", layer="inkling2_expert_projection", quant=INKLING2_MINI.default_quant,
                          variant="grouped")
        conv = resolve(backend="hip_gfx1100", layer="inkling2_short_conv", quant=INKLING2_MINI.default_quant,
                       variant="default")
        self.model.convolution_factory = lambda: Inkling2DeviceConvolution(
            runtime=runtime, launch=conv, channels=self.model.config["hidden_size"],
            width=self.model.config["sconv_kernel_size"],
        )
        attn = resolve(backend="hip_gfx1100", layer="inkling2_target_attention", quant=INKLING2_MINI.default_quant,
                       variant="default")
        cfg = self.model.config
        block_attn = resolve(backend="hip_gfx1100", layer="inkling2_block_attention", quant=INKLING2_MINI.default_quant,
                             variant="default")
        visits = sorted(self.model.plan.visits, key=lambda v: v.state_slot)

        def attention_factory(capacity):
            return Inkling2DeviceAttention(runtime=runtime, launch=attn, geometry=[
                (min(capacity, cfg["sliding_window_size"]) if v.is_local else capacity,
                 *self.model.head_shape(v), cfg["sliding_window_size"] if v.is_local else cfg["rel_extent"])
                for v in visits
            ])

        self.model.attention_factory = attention_factory
        weights = self.model.weights
        with ExitStack() as owned:
            blas = Rocblas.load()
            owned.callback(blas.close)
            self._blas_lock = Lock()
            dense = owned.enter_context(Inkling2DeviceDense(weights, runtime=runtime, blas=blas, lock=self._blas_lock))
            experts = owned.enter_context(Inkling2DeviceExperts(
                weights, (v.weight_layer for v in self.model.plan.visits), runtime=runtime, launch=launch,
                grouped=grouped, max_selected=cfg["num_experts_per_tok"],
            ))
            self.model.project = dense.project
            self.model.project_pair = dense.project_pair
            self.model.project_expert = experts.project
            self.model.project_experts_group = experts.project_group
            self._block_attention = owned.enter_context(Inkling2DeviceBlockAttention(runtime=runtime, launch=block_attn))
            if self.vision is not None:
                vision_dense = owned.enter_context(Inkling2DeviceDense(
                    self.vision.weights, runtime=runtime, blas=blas, lock=self._blas_lock,
                ))
                self.vision.project = vision_dense.project
                self.vision.attend = self._vision_attention
            self._runtime, self._blas = runtime, blas
            self._draft = None
            self._draft_lock = Lock()
            self._owned = owned.pop_all()

    def _vision_attention(self, q, k, v, *, check):
        check()
        return self._block_attention(q, k, v)

    def _load_drafter(self):
        with self._draft_lock:
            if self._draft is None:
                draft = super()._load_drafter()
                dense = self._owned.enter_context(Inkling2DeviceDense(
                    draft.weights, runtime=self._runtime, blas=self._blas, lock=self._blas_lock,
                ))
                draft.project = dense.project
                draft.kv_factory = lambda capacity, cfg: Inkling2DeviceDraftKV(
                    capacity, cfg, runtime=self._runtime, attend=self._block_attention,
                )
                self._draft = draft
            return self._draft

    def close(self):
        self._owned.close()


register_text_generator(model=INKLING2_MINI.name, backend="hip_gfx1100", quant=INKLING2_MINI.default_quant,
                        factory=Inkling2Gfx1100Generator)
