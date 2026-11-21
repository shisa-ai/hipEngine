"""gfx1100 Inkling 2 generator with HIP projections and host sequence state."""

from contextlib import ExitStack

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
            dense = owned.enter_context(Inkling2DeviceDense(weights, runtime=runtime, blas=blas))
            experts = owned.enter_context(Inkling2DeviceExperts(
                weights, (v.weight_layer for v in self.model.plan.visits), runtime=runtime, launch=launch,
            ))
            self.model.project = dense.project
            self.model.project_expert = experts.project
            self._owned = owned.pop_all()

    def close(self):
        self._owned.close()


register_text_generator(model=INKLING2_MINI.name, backend="hip_gfx1100", quant=INKLING2_MINI.default_quant,
                        factory=Inkling2Gfx1100Generator)
