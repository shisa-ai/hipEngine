"""Dump the registered prefill kernels for the raw gguf_q8_0 key on this backend.

The dense term dispatches as quant=gguf_q8_0 (raw layout, 206/206 calls), while
the gfx1151 four-wave override is keyed on gguf_q8_0_t16_v1. This prints every
registered (layer, quant, variant) whose quant mentions q8_0, so the gap is read
off the registry rather than inferred.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")


def main() -> int:
    import hipengine.kernels.hip_gfx1151  # noqa: F401  (registers the backend)
    from hipengine.kernels.registry import registered_keys, resolve

    rows = []
    for key in registered_keys():
        quant = str(getattr(key, "quant", ""))
        if "q8_0" not in quant:
            continue
        variant = str(getattr(key, "variant", ""))
        layer = str(getattr(key, "layer", ""))
        try:
            fn = resolve(
                backend=str(getattr(key, "backend", "hip_gfx1151")),
                layer=layer,
                quant=quant,
                variant=variant,
            )
            name = getattr(fn, "__name__", repr(fn))
        except Exception as exc:  # noqa: BLE001
            name = f"<unresolved: {type(exc).__name__}>"
        rows.append((quant, layer, variant, name))

    print(f"registered q8_0 kernels on this backend: {len(rows)}\n")
    for quant, layer, variant, name in sorted(rows):
        print(f"  quant={quant:22s} layer={layer:8s} variant={variant:34s} {name}")

    wave = [r for r in rows if "wave" in r[3]]
    print(f"\nof those, kernels whose symbol mentions wave: {len(wave)}")
    for quant, layer, variant, name in sorted(wave):
        print(f"  quant={quant:22s} layer={layer:8s} variant={variant:34s} {name}")
    raw = [r for r in wave if r[0] == "gguf_q8_0"]
    print(f"\n... and of those, on the RAW gguf_q8_0 key the dense term uses: {len(raw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
