"""Benchmark-only borrowing of prefill scratch with unchanged decode ownership."""

from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


PREFILL_FIELDS = (
    "prefill_chunk_size", "gdn_prefill_scratch", "qsa_prefill_scratch",
    "ple_prefill_scratch", "qsa_prefill_metadata", "_prefill_buffers",
)


@dataclass(frozen=True)
class PrefillWorkspace:
    owner: Any
    values: Mapping[str, Any]


def capture_workspace(runner):
    return PrefillWorkspace(
        runner, MappingProxyType({name: getattr(runner, name) for name in PREFILL_FIELDS}))


@contextmanager
def use_workspace(runner, workspace):
    owner = workspace.owner
    if (set(workspace.values) != set(PREFILL_FIELDS)
            or runner.closed or owner.closed or runner.runtime is not owner.runtime
            or runner.resident is not owner.resident
            or runner.max_sequence_length != owner.max_sequence_length
            or runner.backend != owner.backend
            or any(getattr(item, "_q8_mmq_buffers", ())
                   or getattr(item, "_q8_mmq_weight_sidecars", None) is not None
                   for item in (runner, owner))):
        raise ValueError("prefill borrowing requires compatible owners without global MMQ resources")
    runner.runtime.device_synchronize()
    saved = {name: getattr(runner, name) for name in PREFILL_FIELDS}
    try:
        for name, value in workspace.values.items():
            setattr(runner, name, value)
        yield
    finally:
        try:
            runner.runtime.device_synchronize()
        finally:
            for name, value in saved.items():
                setattr(runner, name, value)


def workspace_description(workspace):
    values = workspace.values
    return dict(
        chunk_size=values["prefill_chunk_size"],
        token_capacity=values["_prefill_buffers"][0].nbytes // 8,
        metadata_rows=values["qsa_prefill_metadata"].rows,
        gdn_tile_capacity=values["gdn_prefill_scratch"].moe.group_tile_expert.nbytes // 8,
        qsa_tile_capacity=values["qsa_prefill_scratch"].moe.group_tile_expert.nbytes // 8,
    )
