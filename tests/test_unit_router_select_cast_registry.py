"""Router selection/cast exposes fused and strict variants through one ABI."""
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import resolve


def test_router_select_cast_composite_variants_are_registered():
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        load_backend_kernel_package(backend)
        for variant in ("default", "strict_unfused"):
            assert callable(resolve(backend=backend, layer="router_select+cast",
                                    quant="fp32", variant=variant))


def test_strict_composite_calls_select_then_cast_with_separate_libraries(monkeypatch):
    from hipengine.kernels.hip_gfx1100.moe import router
    from hipengine.kernels.hip_gfx1100.convert import cast

    calls = []
    monkeypatch.setattr(router, "qwen35_router_select",
                        lambda *a, **k: calls.append(("select", a, k)))
    monkeypatch.setattr(cast, "f32_to_bf16",
                        lambda *a, **k: calls.append(("cast", a, k)))
    library, runtime = object(), object()
    router.qwen35_router_select_cast_bf16_unfused(
        10, 20, 30, 40, 50, 3, 512, 512, 10, 2560,
        stream=7, library=library, runtime=runtime,
    )
    assert calls == [
        ("select", (10, 20, 30, 3, 512, 512, 10),
         dict(threads=512, stream=7, library=library, runtime=runtime)),
        ("cast", (40, 50, 3 * 2560), dict(stream=7, runtime=runtime)),
    ]
