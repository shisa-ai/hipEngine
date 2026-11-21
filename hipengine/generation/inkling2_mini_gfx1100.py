"""gfx1100 Inkling 2 generator with HIP projections and host sequence state."""

from contextlib import ExitStack
from functools import partial
from threading import Lock

from hipengine.core.hip import get_hip_runtime
from hipengine.core.rocblas import Rocblas
from hipengine.execution_profiles import (
    RuntimeProfileKey, RuntimeProfilePlan, VariantSelection, register_runtime_profile_plan, registered_runtime_profile_keys,
)
from hipengine.generation.inkling2_mini import Inkling2ReferenceGenerator
from hipengine.generation.registry import register_text_generator
from hipengine.kernels.hip_gfx1100.attention import inkling2_block_attention, inkling2_target_attention  # noqa: F401
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_gemv_f32_bf16w_f32_out
from hipengine.kernels.hip_gfx1100.quant import inkling2_experts, inkling2_short_conv  # noqa: F401
from hipengine.kernels.registry import KernelKey, is_registered, register, resolve
from hipengine.loading.inkling2_device_dense import Inkling2DeviceDense
from hipengine.loading.inkling2_device_experts import Inkling2DeviceExperts
from hipengine.models.inkling2_mini import INKLING2_MINI
from hipengine.runtime.inkling2_device_attention import Inkling2DeviceAttention
from hipengine.runtime.inkling2_device_block_attention import Inkling2DeviceBlockAttention
from hipengine.runtime.inkling2_device_convolution import Inkling2DeviceConvolution
from hipengine.runtime.inkling2_device_draft_kv import Inkling2DeviceDraftKV


_BACKEND = "hip_gfx1100"
_HEAD = "inkling2_vocabulary_projection"
_DENSE = "inkling2_dense_projection"
_EXPERTS = "inkling2_routed_ffn"
_HEAD_EVIDENCE = "worklog/entries/20261121T084738.426332Z-lhl-inkling2-mini-d38a06.md"
_DENSE_EVIDENCE = "worklog/entries/20261121T100937.153828Z-lhl-inkling2-mini-557326.md"
_EXPERTS_EVIDENCE = "worklog/entries/20261121T094149.586925Z-lhl-inkling2-mini-732718.md"


def _fp32_head(owner):
    return owner.project


def _native_head(owner):
    return partial(owner.project, native_bf16=dense_gemv_f32_bf16w_f32_out)


def _fp32_dense(owner):
    return owner.project, owner.project_many, owner.project_groups, owner.project_chains, owner.project_device


def _native_dense(owner):
    launch = partial(dense_gemv_f32_bf16w_f32_out, threads=64)
    return (partial(owner.project, native_bf16=launch, pinned=True),
            partial(owner.project_many, native_bf16=launch, pinned=True),
            partial(owner.project_groups, native_bf16=launch, pinned=True),
            partial(owner.project_chains, native_bf16=launch, pinned=True),
            partial(owner.project_device, native_bf16=launch))


def _host_experts(owner):
    return None


def _device_experts(owner):
    return owner.feed_forward


def _bind_profile(generator, profile):
    selections = {item["layer"]: item["selected_variant"] for item in profile.manifest["selections"]}

    def bind(layer, owner):
        return resolve(backend=_BACKEND, layer=layer, quant=INKLING2_MINI.default_quant,
                       variant=selections[layer])(owner)

    generator.model.project_logits = partial(
        bind(_HEAD, generator._dense), "unembed.weight", limit=generator.model.config["unpadded_vocab_size"],
    )
    model = generator.model
    (model.project, model.project_many, model.project_groups, model.project_chains,
     model.project_device) = bind(_DENSE, generator._dense)
    model.feed_experts_group = bind(_EXPERTS, generator._experts)
    # Bound the activation and relative-bias scratch; strict keeps its serial traversal.
    generator.prefill_block_width = 64 if model.feed_experts_group is not None else 1


def register_inkling2_gfx1100_profiles():
    model, quant = INKLING2_MINI.name, INKLING2_MINI.default_quant
    for layer, variant, projection in (
        (_HEAD, "fp32_rocblas", _fp32_head), (_HEAD, "native_bf16", _native_head),
        (_DENSE, "fp32_rocblas", _fp32_dense), (_DENSE, "native_bf16_threads64", _native_dense),
        (_EXPERTS, "host_pointwise", _host_experts), (_EXPERTS, "device_pointwise", _device_experts),
    ):
        key = KernelKey(_BACKEND, layer, quant, variant)
        if not is_registered(key):
            register(key, projection)
    registered = registered_runtime_profile_keys()
    for profile, (head, dense, experts) in (
        ("strict", ("fp32_rocblas", "fp32_rocblas", "host_pointwise")),
        ("production", ("native_bf16", "native_bf16_threads64", "device_pointwise")),
    ):
        if RuntimeProfileKey(model, _BACKEND, quant, profile) not in registered:
            fallback = profile != "strict"
            register_runtime_profile_plan(
                model=model, backend=_BACKEND, quant=quant, profile=profile,
                plan=RuntimeProfilePlan(
                    selections=(
                        VariantSelection(_HEAD, "checkpoint_bf16_head", head,
                                         "fp32_rocblas" if fallback else None, _HEAD_EVIDENCE),
                        VariantSelection(_DENSE, "target_text", dense,
                                         "fp32_rocblas" if fallback else None, _DENSE_EVIDENCE),
                        VariantSelection(_EXPERTS, "target", experts,
                                         "host_pointwise" if fallback else None, _EXPERTS_EVIDENCE),
                    ),
                    kv_policy="fp32", graph_policy="eager", binder=_bind_profile,
                ),
            )


class Inkling2Gfx1100Generator(Inkling2ReferenceGenerator):
    packed_verification = True
    def __init__(self, *, model_path, weight_index, model_plugin, max_sequence_length=None):
        super().__init__(model_path=model_path, weight_index=weight_index, model_plugin=model_plugin,
                         max_sequence_length=max_sequence_length)
        runtime = get_hip_runtime()
        launch = resolve(backend="hip_gfx1100", layer="inkling2_expert_projection", quant=INKLING2_MINI.default_quant,
                         variant="default")
        grouped = resolve(backend=_BACKEND, layer="inkling2_expert_projection", quant=INKLING2_MINI.default_quant,
                          variant="grouped_wave_rows")
        paired = resolve(backend=_BACKEND, layer="inkling2_expert_projection", quant=INKLING2_MINI.default_quant,
                         variant="paired")
        pointwise = resolve(backend=_BACKEND, layer="inkling2_expert_pointwise", quant=INKLING2_MINI.default_quant,
                            variant="default")
        conv = resolve(backend="hip_gfx1100", layer="inkling2_short_conv", quant=INKLING2_MINI.default_quant,
                       variant="default")
        conv_block = resolve(backend=_BACKEND, layer="inkling2_short_conv", quant=INKLING2_MINI.default_quant,
                             variant="block")
        self.model.convolution_factory = lambda: Inkling2DeviceConvolution(
            runtime=runtime, launch=conv, launch_block=conv_block, channels=self.model.config["hidden_size"],
            width=self.model.config["sconv_kernel_size"],
        )
        attn = resolve(backend=_BACKEND, layer="inkling2_target_attention", quant=INKLING2_MINI.default_quant,
                       variant="tiled")
        split = resolve(backend=_BACKEND, layer="inkling2_target_attention", quant=INKLING2_MINI.default_quant,
                        variant="split")
        batch_attn = resolve(backend=_BACKEND, layer="inkling2_target_attention", quant=INKLING2_MINI.default_quant,
                             variant="block")
        cache_write = resolve(backend=_BACKEND, layer="inkling2_target_kv_write", quant=INKLING2_MINI.default_quant,
                              variant="default")
        cfg = self.model.config
        block_attn = resolve(backend="hip_gfx1100", layer="inkling2_block_attention", quant=INKLING2_MINI.default_quant,
                             variant="default")
        visits = sorted(self.model.plan.visits, key=lambda v: v.state_slot)

        def attention_factory(capacity):
            return Inkling2DeviceAttention(runtime=runtime, launch=attn, split=split, block=batch_attn,
                                           write=cache_write, geometry=[
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
                grouped=grouped, max_selected=cfg["num_experts_per_tok"], pointwise=pointwise,
                expand_nvfp4_scales=True, paired=paired,
            ))
            self._experts = experts
            self._dense = dense
            self.model.project = dense.project
            self.model.project_many = dense.project_many
            self.model.project_groups = dense.project_groups
            self.model.project_chains = dense.project_chains
            self.model.project_device = dense.project_device
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


register_text_generator(model=INKLING2_MINI.name, backend=_BACKEND, quant=INKLING2_MINI.default_quant,
                        factory=Inkling2Gfx1100Generator)
