"""CPU-only census of the real Qwen4Exp runner allocation recipes.

No model weights or device memory are allocated. This is a standalone diagnostic;
temporary memory-stat replacement must not be used concurrently with a live engine.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class CountingRuntime:
    def __init__(self):
        self.next_ptr = 1 << 48
        self.live = {}

    def malloc(self, nbytes):
        nbytes = int(nbytes)
        if nbytes <= 0:
            raise ValueError("positive allocation required")
        ptr = self.next_ptr
        self.next_ptr += (nbytes + 4095) // 4096 * 4096
        self.live[ptr] = nbytes
        return ptr

    def free(self, ptr):
        del self.live[ptr]

    def contains(self, ptr, nbytes):
        return any(base <= ptr and ptr + nbytes <= base + size
                   for base, size in self.live.items())

    def memset(self, ptr, value, nbytes):
        if not self.contains(ptr, nbytes):
            raise ValueError("memset exceeds a live allocation")

    def memcpy(self, dest, source, nbytes, kind):
        from hipengine.core.runtime import MemcpyKind
        if kind != MemcpyKind.HOST_TO_DEVICE:
            raise ValueError("allocation census must not read simulated device data")
        if not self.contains(dest, nbytes):
            raise ValueError("copy exceeds a live allocation")

    def device_synchronize(self):
        pass

    def mem_get_info(self):
        capacity = 1 << 60
        return capacity - sum(self.live.values()), capacity


def attribute_allocations(runner, runtime):
    from hipengine.core.memory import DeviceBuffer

    paths = {}
    seen = set()

    def walk(value, path):
        if isinstance(value, DeviceBuffer):
            if runtime.live.get(value.ptr) == value.nbytes:
                paths.setdefault(value.ptr, path)
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                walk(child, f"{path}.{index}")
        elif is_dataclass(value):
            members = {field.name: getattr(value, field.name) for field in fields(value)}
            members.update(getattr(value, "__dict__", {}))
            for name, child in members.items():
                if name != "runtime":
                    walk(child, f"{path}.{name}")

    for name, value in vars(runner).items():
        if name not in ("runtime", "config"):
            walk(value, name)
    if set(paths) != set(runtime.live):
        raise ValueError("allocation roots missing from owner census")
    records = [{"owner": paths[ptr], "nbytes": nbytes}
               for ptr, nbytes in runtime.live.items()]
    totals = defaultdict(int)
    for row in records:
        totals[row["owner"].split(".")[0]] += row["nbytes"]
    return records, dict(sorted(totals.items()))


@contextmanager
def simulated_runner(config, *, context, chunk):
    from hipengine.core import memory
    from hipengine.runtime import qwen4_exp_runner as module
    if chunk <= 0 or not 0 < context <= config.context_length:
        raise ValueError("invalid context/chunk")
    runtime = CountingRuntime()
    names = (
        "state", "gdn_scratch", "qsa_scratch", "ple_scratch", "head_scratch",
        "gdn_prefill_scratch", "qsa_prefill_scratch", "ple_prefill_scratch",
        "qsa_prefill_metadata", "_target_verify_output", "_device_transaction_snapshot",
        "_q8_mmq_weight_sidecars", "_shared_position_context", "moe_graph_cache",
    )
    runner = module.Qwen4ExpGGUFResidentModelRunner.__new__(module.Qwen4ExpGGUFResidentModelRunner)
    vars(runner).update(
        **dict.fromkeys(names), config=config, runtime=runtime, resident=None, backend="cpu_census",
        max_sequence_length=context, prefill_chunk_size=chunk, closed=False,
        _buffers=[], _prefill_buffers=[], attention_states=(), index_states=(),
        _configure_q8_mmq_prefill_resources=lambda: None,
        _configure_q8_mmq_weight_sidecars=lambda: None)
    tracker = memory._MemoryStatsTracker()
    with patch.object(memory, "_MEMORY_STATS", tracker), patch.object(
        memory, "get_hip_runtime", side_effect=RuntimeError("unexpected GPU runtime")
    ), patch.object(module, "get_hip_runtime", side_effect=RuntimeError("unexpected GPU runtime")), patch.object(
        module, "_configure_qwen4_exp_moe_mmq_scratch", lambda *args, **kwargs: None
    ):
        try:
            module.Qwen4ExpGGUFResidentModelRunner._allocate(runner)
            yield runner, runtime
        finally:
            module.Qwen4ExpGGUFResidentModelRunner.close(runner)
        if runtime.live or memory.memory_stats()["current_allocated_bytes"]:
            raise ValueError("simulated allocations did not close")


def census(config, *, context, chunk):
    from hipengine.core import memory
    from scripts.qwen4exp_chunk_memory_probe import prepare_lazy_group_risk

    with simulated_runner(config, context=context, chunk=chunk) as (runner, runtime):
        constructor_bytes = sum(runtime.live.values())
        queues = prepare_lazy_group_risk(runner)
        records, totals = attribute_allocations(runner, runtime)
        total = sum(runtime.live.values())
        if total != memory.memory_stats()["current_allocated_bytes"]:
            raise ValueError("counting runtime and tracked allocations disagree")
    return dict(context=context, chunk=chunk, constructor_bytes=constructor_bytes,
                prepared_bytes=total, repair_queues=queues, owner_bytes=totals,
                allocations=records, teardown_bytes=0)


def main():
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.loading.qwen4_exp_gguf import qwen4_exp_gguf_config_from_metadata
    from scripts.qwen4exp_canonical_ar_bench import _git_metadata

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--context", type=int, default=4352)
    parser.add_argument("--chunks", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    config = qwen4_exp_gguf_config_from_metadata(index)
    packet = dict(
        schema=1, kind="cpu_allocation_recipe_census", source=_git_metadata(ROOT),
        command=sys.argv, performance_claim=False, promotion_claim=False,
        model_root=str(args.model_root), records=[
            census(config, context=args.context, chunk=chunk) for chunk in args.chunks],
        limits=["Uses real base allocation routines and worst-case repair queues, but no GPU memory.",
                "MMQ resources/sidecars, graph capture, target verification and transactions are excluded.",
                "This is diagnostic evidence for deriving admission, not an installed memory policy."])
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    for row in packet["records"]:
        print(row["chunk"], row["prepared_bytes"], json.dumps(row["owner_bytes"], sort_keys=True))


if __name__ == "__main__":
    main()
