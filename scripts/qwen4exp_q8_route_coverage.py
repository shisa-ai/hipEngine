#!/usr/bin/env python3
"""Bound the Q8 dense WMMA prefill route statically, without running the model.

The selector's weight filter in ``hipengine/runtime/qwen4_exp_runner.py`` admits
a weight only when its quant key is ``gguf_q8_0`` **and** its slot path sits
under ``layers.``.  Both halves are readable off the GGUF index, so the set of
weights a layer scope can own is a static fact rather than something a kernel
census has to discover.  Use this before attributing measured drift to a route:
a weight the filter excludes cannot contribute to it at any layer scope.

Usage::

    python3 scripts/qwen4exp_q8_route_coverage.py --model-root PATH \
        --scope 32-47 --scope 0-47
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.loading.gguf import discover_gguf_files, load_gguf_index  # noqa: E402

BLOCK = re.compile(r"^blk\.(\d+)\.(.+)$")
ROUTE_QUANT = "Q8_0"


def _parse_scope(raw: str) -> tuple[int, ...]:
    values: set[int] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            values.update(range(int(start), int(end) + 1))
        else:
            values.add(int(part))
    return tuple(sorted(values))


def _span(layers: Sequence[int]) -> str:
    values = sorted(layers)
    if not values:
        return "(none)"
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return ",".join(str(value) for value in values)


def collect(model_root: Path) -> dict[str, Any]:
    shards = discover_gguf_files(model_root)
    tensors = [tensor for shard in shards for tensor in load_gguf_index(shard).tensors]
    in_block: dict[int, dict[str, int]] = defaultdict(dict)
    outside: list[str] = []
    excluded_roles: Counter[tuple[str, str]] = Counter()
    for tensor in tensors:
        match = BLOCK.match(tensor.name)
        if tensor.ggml_type_name != ROUTE_QUANT:
            if match:
                excluded_roles[(match.group(2), tensor.ggml_type_name)] += 1
            continue
        if match:
            in_block[int(match.group(1))][match.group(2)] = int(tensor.nbytes)
        else:
            outside.append(tensor.name)
    return {
        "shards": [shard.name for shard in shards],
        "tensor_count": len(tensors),
        "quant_mix": dict(Counter(t.ggml_type_name for t in tensors).most_common()),
        "in_block": {layer: dict(roles) for layer, roles in sorted(in_block.items())},
        "outside_blocks": sorted(outside),
        "excluded_block_roles": {
            f"{role}:{quant}": count
            for (role, quant), count in excluded_roles.most_common()
        },
    }


def render(data: dict[str, Any], scopes: Sequence[tuple[int, ...]]) -> str:
    in_block = data["in_block"]
    total = sum(len(roles) for roles in in_block.values())
    total_bytes = sum(sum(roles.values()) for roles in in_block.values())
    lines = [
        f"{data['tensor_count']} tensors across {len(data['shards'])} shards",
        f"whole-model quant mix: {data['quant_mix']}",
        "",
        f"== {ROUTE_QUANT} weights the route can own: {total} tensors, "
        f"{total_bytes / 2**30:.2f} GiB ==",
    ]
    roles: dict[str, list[int]] = defaultdict(list)
    for layer, entry in in_block.items():
        for role in entry:
            roles[role].append(int(layer))
    for role, layers in sorted(roles.items()):
        lines.append(f"  {role:26s} {len(layers):3d} layers  ({_span(layers)})")

    lines += [
        "",
        f"== {ROUTE_QUANT} weights the slot-path filter excludes ==",
        *(f"  {name}" for name in data["outside_blocks"]),
        "",
        "== the largest in-block roles the quant filter excludes ==",
    ]
    for label, count in list(data["excluded_block_roles"].items())[:8]:
        role, _, quant = label.rpartition(":")
        lines.append(f"  {role:26s} {quant:8s} {count:3d}")

    if scopes:
        lines += ["", "== coverage by layer scope =="]
        for scope in scopes:
            owned = sum(len(in_block.get(layer, {})) for layer in scope)
            owned_bytes = sum(
                sum(in_block.get(layer, {}).values()) for layer in scope
            )
            lines.append(
                f"  layers {_span(scope):8s} {owned:4d}/{total} tensors "
                f"({owned / total:5.1%}), {owned_bytes / 2**30:6.2f} GiB "
                f"({owned_bytes / total_bytes:5.1%} of route bytes)"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--scope",
        action="append",
        default=None,
        help="layer scope to report, e.g. 32-47 or 0,1,2 (repeatable)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    data = collect(args.model_root.resolve())
    scopes = [_parse_scope(raw) for raw in (args.scope or [])]
    print(render(data, scopes))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(data)
        payload["scopes"] = {_span(scope): list(scope) for scope in scopes}
        args.output.write_text(json.dumps(payload, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
