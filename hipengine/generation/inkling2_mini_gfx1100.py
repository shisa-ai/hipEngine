"""gfx1100 Inkling 2 generator with HIP projections and host sequence state."""

from contextlib import ExitStack
from threading import Lock

from hipengine.core.hip import get_hip_runtime
from hipengine.core.rocblas import Rocblas
from hipengine.generation.inkling2_mini import Inkling2ReferenceGenerator
from hipengine.generation.registry import register_text_generator
from hipengine.kernels.hip_gfx1100.quant import inkling2_experts  # noqa: F401
from hipengine.kernels.registry import resolve
from hipengine.loading.inkling2_device_dense import Inkling2DeviceDense
from hipengine.loading.inkling2_device_experts import Inkling2DeviceExperts
from hipengine.models.inkling2_mini import INKLING2_MINI


class Inkling2Gfx1100Generator(Inkling2ReferenceGenerator):
    def __init__(self, *, model_path, weight_index, model_plugin, max_sequence_length=None):
        super().__init__(model_path=model_path, weight_index=weight_index, model_plugin=model_plugin,
                         max_sequence_length=max_sequence_length)
        runtime = get_hip_runtime()
        launch = resolve(backend="hip_gfx1100", layer="inkling2_expert_projection", quant=INKLING2_MINI.default_quant,
                         variant="default")
        weights = self.model.weights
        with ExitStack() as owned:
            blas = Rocblas.load()
            owned.callback(blas.close)
            self._blas_lock = Lock()
            dense = owned.enter_context(Inkling2DeviceDense(weights, runtime=runtime, blas=blas, lock=self._blas_lock))
            experts = owned.enter_context(Inkling2DeviceExperts(
                weights, (v.weight_layer for v in self.model.plan.visits), runtime=runtime, launch=launch,
            ))
            self.model.project = dense.project
            self.model.project_expert = experts.project
            if self.vision is not None:
                vision_dense = owned.enter_context(Inkling2DeviceDense(
                    self.vision.weights, runtime=runtime, blas=blas, lock=self._blas_lock,
                ))
                self.vision.project = vision_dense.project
            self._runtime, self._blas = runtime, blas
            self._draft = None
            self._draft_lock = Lock()
            self._owned = owned.pop_all()

    def _load_drafter(self):
        with self._draft_lock:
            if self._draft is None:
                draft = super()._load_drafter()
                dense = self._owned.enter_context(Inkling2DeviceDense(
                    draft.weights, runtime=self._runtime, blas=self._blas, lock=self._blas_lock,
                ))
                draft.project = dense.project
                self._draft = draft
            return self._draft

    def close(self):
        self._owned.close()


register_text_generator(model=INKLING2_MINI.name, backend="hip_gfx1100", quant=INKLING2_MINI.default_quant,
                        factory=Inkling2Gfx1100Generator)
