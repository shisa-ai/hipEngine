"""CPU-reference backend.

Importing this package self-registers the first NumPy reference kernels. Tests that clear the
kernel registry can call ``register_cpu_reference_kernels()`` to restore them.
"""

from hipengine.kernels.cpu_reference.fixtures import (
    LayerCheckResult,
    LayerFixture,
    Tolerances,
    load_fixture,
    run_fixture,
    save_fixture,
)
from hipengine.kernels.cpu_reference.ops import (
    attention_decode,
    embed,
    full_attn_prefill,
    linear,
    lm_head,
    o_proj,
    qkv_proj,
    register_cpu_reference_kernels,
    rmsnorm,
    rotate,
)

register_cpu_reference_kernels()

__all__ = [
    "LayerCheckResult",
    "LayerFixture",
    "Tolerances",
    "attention_decode",
    "embed",
    "full_attn_prefill",
    "linear",
    "lm_head",
    "load_fixture",
    "o_proj",
    "qkv_proj",
    "register_cpu_reference_kernels",
    "rmsnorm",
    "rotate",
    "run_fixture",
    "save_fixture",
]
