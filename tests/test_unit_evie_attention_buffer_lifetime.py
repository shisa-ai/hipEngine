"""CPU ownership regression: attention must never submit a freed operand."""

from types import SimpleNamespace

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import evie


def test_mixed_shape_attention_keeps_operands_alive(monkeypatch):
    # A long query can share its score-cache key with a later vision grid,
    # while their QKV planes have different layouts. No HIP calls are needed
    # to check the allocator/dispatcher ownership contract.
    runner = evie.EvieRunner.__new__(evie.EvieRunner)
    runner.precision = "fp16"
    next_ptr = 1 << 24
    freed = []
    calls = []

    def allocate(nbytes):
        nonlocal next_ptr
        result = DeviceBuffer(next_ptr, nbytes)
        next_ptr += nbytes + 4096
        return result

    def check_operands(args, operation):
        for arg in args:
            ptr = getattr(arg, "value", arg)
            if not isinstance(ptr, int):
                continue
            for owner in freed:
                assert not owner.ptr <= ptr < owner.ptr + owner.nbytes, (
                    f"{operation} submits freed operand {ptr}; "
                    f"owner={owner}; attention calls={calls}"
                )

    def bind(symbol, argtypes):
        def launch(*args):
            check_operands(args, symbol)
            return 0
        return launch

    monkeypatch.setattr(evie, "_malloc_committed", allocate)
    monkeypatch.setattr(evie, "hip_free", freed.append)
    runner._k = bind
    runner.rocblas = SimpleNamespace(
        gemm_ex_strided_batched_f16_f32acc=lambda *args, **kwargs:
        check_operands(args, "GEMM")
    )
    # Small pointers stand for caller-owned inputs, outside fake allocations.
    for kind, tokens in (
        ("text", 560), ("text", 8), ("text", 6),
        ("vision", 560), ("vision", 576), ("vision", 560),
    ):
        calls.append((kind, tokens))
        if kind == "text":
            runner._attention(
                1, 2, 3, 4, tokens, 16, 256, None, 1 / 16, kv_heads=4
            )
        else:
            runner._attention_from_packed(
                1, 2, 3, 4, tokens, 16, 64, 3072, None, 1 / 8
            )
