"""Report the Q8_0 T16 dense projections' dispatch keys, tiles, and wave gate.

The four-wave schedule is wired on gfx1151 through a ``(backend, layer, quant,
variant)`` override in ``hip_gfx1151/__init__.py``. An override only fires on an
exact key match, and the wrapper additionally gates on ``tile_m in {32, 64}``,
``tile_n == 32`` and ``out_features >= 2048``. So the only way to know whether
the declared ``GGUF_Q8_T16_PREFILL_FOUR_WAVE = True`` reaches the dense term is
to read the keys and tiles the model actually produces.
"""

from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, ".")


def main() -> int:
    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    llm, runner, info = _resolve_generator(resolve_artifact(), 4096)
    print("resolution:", info["resolution"])

    seen: Counter[tuple[str, str, str]] = Counter()
    shapes: list[tuple[int, int, int]] = []
    for module in _walk(llm):
        weight = getattr(module, "weight", None)
        spec = getattr(weight, "spec", None)
        if spec is None:
            continue
        quant = getattr(spec, "quant_key", None)
        variant = getattr(spec, "variant", None) or getattr(module, "variant", None)
        if quant and "q8_0" in str(quant):
            seen[(str(quant), str(variant), type(module).__name__)] += 1
            shape = getattr(spec, "shape", None) or getattr(weight, "shape", None)
            if shape is not None and len(shape) == 2:
                shapes.append((int(shape[0]), int(shape[1]), 0))

    print("\nQ8_0 dispatch keys seen (quant, variant, module type) -> count")
    for key, count in seen.most_common():
        print(f"  {key} -> {count}")
    if not seen:
        print("  NONE -- the dense term is not Q8_0 by quant_key, so the override")
        print("  key cannot match and four-wave never engages for it.")

    print("\ndistinct 2-D Q8_0 shapes (out_features, in_features):")
    for out_f, in_f, _ in sorted({(a, b, 0) for a, b, _ in shapes}):
        print(f"  out_features={out_f:6d}  in_features={in_f:6d}")

    # Evaluate the gate exactly as the wrapper does, for the default tiles.
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
        _default_tiles,
        _four_wave_prefill_applies,
        _two_wave_prefill_applies,
    )

    print("\nwave gate per distinct shape (default tiles, four_wave_default=True):")
    for out_f, in_f, _ in sorted({(a, b, 0) for a, b, _ in shapes}):
        tile_m, tile_n = _default_tiles(512, in_f, out_f)
        four = _four_wave_prefill_applies(
            tile_m=tile_m, tile_n=tile_n, out_features=out_f, default=True
        )
        two = _two_wave_prefill_applies(
            tile_m=tile_m, tile_n=tile_n, out_features=out_f, default=True
        )
        print(
            f"  out={out_f:6d} in={in_f:6d}  default tile_m={tile_m:3d} tile_n={tile_n:3d}"
            f"  four_wave={four}  two_wave={two}"
        )
    return 0


def _walk(root: object):
    stack = [root]
    seen_ids: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen_ids:
            continue
        seen_ids.add(id(node))
        yield node
        for name in ("_modules",):
            children = getattr(node, name, None)
            if isinstance(children, dict):
                stack.extend(children.values())
        for name in vars(node) if hasattr(node, "__dict__") else ():
            value = getattr(node, name, None)
            if isinstance(value, (list, tuple)) and value and hasattr(value[0], "__dict__"):
                stack.extend(value)


if __name__ == "__main__":
    raise SystemExit(main())
