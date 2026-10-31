"""Count actual candidate calls through registry entries or direct aliases."""

from collections import Counter
from contextlib import contextmanager
from functools import wraps
from importlib import import_module


def empty_dispatch_counter():
    """The counter shape every report reads, counted arm or not."""

    return {"calls": 0, "shapes": Counter(), "leaves": Counter()}


@contextmanager
def count_candidate_dispatch(*, key=None, direct_target=None, direct_reference=None,
                             shape_positions=None, extra_direct_targets=()):
    """Count candidate dispatches through a registry entry or direct aliases.

    ``extra_direct_targets`` names further module attributes that implement the
    same mechanism for the same rows. The GR-up wide tile has three: the
    F32-input tile, and the two F16-operand fused epilogues that the F16-direct
    composition takes when its activation workspace is admitted. Counting only
    the F32 tile reports zero calls while the route runs, so the extra leaves
    are patched into the same counter and named individually. Shape records stay
    on the declared primary target, whose argument positions they describe.

    Every counter has the same three keys, so a report can read ``calls``,
    ``shapes`` and ``leaves`` whether or not the arm is counted. Extra targets
    alone are a valid arm: they are counted and named, and a primary target is
    required only when ``shape_positions`` describes its argument list.
    """

    counter = empty_dispatch_counter()
    if direct_reference is not None and direct_target is None:
        raise ValueError("direct reference requires a direct target")
    if key is None and direct_target is None and not extra_direct_targets:
        yield counter
        return
    if key is None and direct_target is None and shape_positions is not None:
        raise ValueError("shape positions require a primary candidate target")
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

    def _counted(original, leaf, positions):
        @wraps(original)
        def counted(*args, **kwargs):
            if positions is not None:
                if len(args) <= max(positions):
                    raise ValueError("candidate shape arguments do not match observed ABI")
                counter["shapes"][tuple(int(args[index]) for index in positions)] += 1
            counter["calls"] += 1
            counter["leaves"][leaf] += 1
            return original(*args, **kwargs)
        return counted

    primary_leaf = (f"{direct_target[0]}.{direct_target[1]}" if direct_target
                    else str(key) if key is not None else None)
    patched = []
    if direct_target is not None:
        counted = _counted(original, primary_leaf, shape_positions)
        setattr(module, direct_target[1], counted)
        patched.append((module, direct_target[1], original))
    elif key is not None:
        register(key, _counted(original, primary_leaf, shape_positions), replace=True)
    for target in extra_direct_targets:
        extra_module = import_module(target[0])
        extra_original = getattr(extra_module, target[1])
        setattr(extra_module, target[1], _counted(
            extra_original, f"{target[0]}.{target[1]}", None))
        patched.append((extra_module, target[1], extra_original))
    try:
        yield counter
    finally:
        if key is not None and direct_target is None:
            register(key, original, replace=True)
        for patched_module, name, patched_original in patched:
            setattr(patched_module, name, patched_original)


def shape_records(counter):
    return [{"arguments": list(shape), "calls": calls}
            for shape, calls in sorted(counter["shapes"].items())]
