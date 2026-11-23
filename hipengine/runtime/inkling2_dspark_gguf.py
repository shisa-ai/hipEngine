"""Construct an Inkling DSpark draft with resident GGUF projections."""

from contextlib import contextmanager
from functools import partial

from hipengine.loading.gguf import GGUFReader
from hipengine.loading.inkling2_dspark_gguf import Inkling2DSparkGGUFWeights
from hipengine.loading.inkling2_dspark_gguf_device import Inkling2DSparkDeviceProjections
from hipengine.runtime.inkling2_mini_dspark import Inkling2DSpark


@contextmanager
def open_inkling2_dspark_gguf(target, draft_path, *, backend, runtime, blas):
    weights = Inkling2DSparkGGUFWeights(GGUFReader(draft_path), target.mapping)
    draft = Inkling2DSpark(target, weights, weights.config)
    with Inkling2DSparkDeviceProjections(weights, backend=backend, runtime=runtime, blas=blas) as projection:
        draft.project = partial(projection.project, pinned=True)
        yield draft
