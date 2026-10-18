"""CPU helpers shared by the Gemma 4 decode-graph tests.

The graph node's ``LLM`` ownership is a CPU contract, but the node itself is a
GPU-tier test that loads a 17 GB model. The helper lives outside any ``test_*``
module so the unit tier can exercise the contract without importing a GPU-tier
test module, which the tier gate in ``scripts/check_test_tiers.py`` forbids.
"""

from __future__ import annotations

import contextlib


@contextlib.contextmanager
def owned_llm(model: str):
    """Open a public ``LLM`` and release its device owner on every exit path.

    The graph session closes itself, but the model weights and KV cache belong
    to the ``LLM``'s text generator. A bare ``LLM(...)`` that is never closed
    keeps those buffers resident for the rest of the process, and a later large
    model load in the same process can then fail with a HIP out-of-memory
    error. The ``finally`` releases the owner on success, assertion failure, or
    any other exception.
    """

    import hipengine

    llm = hipengine.LLM(model=model)
    try:
        yield llm
    finally:
        llm.close()
