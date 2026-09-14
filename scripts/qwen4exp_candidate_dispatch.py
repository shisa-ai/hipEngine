"""Count actual candidate calls through registry entries or direct aliases."""

from collections import Counter
from contextlib import contextmanager
from functools import wraps
from importlib import import_module


@contextmanager
def count_candidate_dispatch(*, key=None, direct_target=None, direct_reference=None,
                             shape_positions=None):
    counter = {"calls": 0, "shapes": Counter()}
    if direct_reference is not None and direct_target is None:
        raise ValueError("direct reference requires a direct target")
    if key is None and direct_target is None:
        yield counter
        return
    registered = None
    if key is not None:
        from hipengine.kernels.registry import is_registered, register, resolve

        if not is_registered(key):
            raise ValueError("exact candidate key is not registered")
        registered = resolve(
            backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    if direct_target is not None:
        module = import_module(direct_target[0])
        original = getattr(module, direct_target[1])
        expected = (getattr(import_module(direct_reference[0]), direct_reference[1])
                    if direct_reference is not None else registered)
        if expected is not None and expected is not original:
            raise ValueError("direct alias does not match declared implementation")
    else:
        original = registered

    @wraps(original)
    def counted(*args, **kwargs):
        if shape_positions is not None:
            if len(args) <= max(shape_positions):
                raise ValueError("candidate shape arguments do not match observed ABI")
            counter["shapes"][tuple(int(args[index]) for index in shape_positions)] += 1
        counter["calls"] += 1
        return original(*args, **kwargs)

    if direct_target is not None:
        setattr(module, direct_target[1], counted)
    else:
        register(key, counted, replace=True)
    try:
        yield counter
    finally:
        if direct_target is not None:
            setattr(module, direct_target[1], original)
        else:
            register(key, original, replace=True)


def shape_records(counter):
    return [{"arguments": list(shape), "calls": calls}
            for shape, calls in sorted(counter["shapes"].items())]
