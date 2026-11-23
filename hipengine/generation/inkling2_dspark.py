"""Explicit DSpark GGUF sidecar attachment for Inkling text generation."""

from contextlib import ExitStack
from functools import partial
from types import MappingProxyType

from hipengine.models.inkling2_mini import INKLING2_GGUF
from hipengine.runtime.inkling2_dspark_gguf import open_inkling2_dspark_gguf
from hipengine.speculative.registry import SpeculativeProviderKey, register_speculative_provider


class Inkling2DSparkProvider:
    provider_name = "dspark"

    def __init__(self, target, draft, owner):
        self._target = target
        self._draft = draft
        self._owner = owner

    def generate_detailed(self, request):
        return list(self._target._generate_batch(request, drafter=self._draft))

    def stream_detailed(self, request):
        raise NotImplementedError("Inkling GGUF DSpark does not stream verified tokens incrementally")

    def capabilities(self):
        return {"provider": self.provider_name, "candidate_budget": self._draft.cfg["block_size"] - 1,
                "verification": "serial_target", "streaming": False}

    def close(self):
        self._owner.close()


def _build_provider(*, target_generator, config, backend):
    target = target_generator.model
    projection = target.projection
    with ExitStack() as owner:
        draft = owner.enter_context(open_inkling2_dspark_gguf(
            target, config.draft_model, backend=backend, runtime=projection._runtime, blas=projection._blas,
        ))
        max_candidates = draft.cfg["block_size"] - 1
        if config.candidate_budget > max_candidates:
            raise ValueError(f"Inkling DSpark candidate budget {config.candidate_budget} exceeds sidecar limit {max_candidates}")
        draft.cfg = MappingProxyType(dict(draft.cfg, block_size=config.candidate_budget + 1))
        return Inkling2DSparkProvider(target_generator, draft, owner.pop_all())


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register_speculative_provider(
        SpeculativeProviderKey("dspark", INKLING2_GGUF.name, _backend, INKLING2_GGUF.default_quant),
        partial(_build_provider, backend=_backend), candidate_budget=1,
    )
