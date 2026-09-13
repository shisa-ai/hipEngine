"""Same-host PLE gather screen using real weights and canonical hash rows.

This is a CPU gather/dequant/scatter screen, not complete-request evidence.
No inference selector is changed. Both arms publish the original row order.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.loading.qwen4_exp_gguf import qwen4_exp_gguf_config_from_metadata
from hipengine.loading.qwen4_exp_materialize import Qwen4ExpPLEMMapTable
from hipengine.quant.gguf import dequantize_gguf_data
from hipengine.kernels.cpu_reference.qwen4_exp import ple_hash_rows
from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture, _host_metadata, _git_metadata
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity


def pread_exact(fd, size, offset, *, read=os.pread):
    chunks = []
    done = 0
    while done < size:
        try:
            chunk = read(fd, size - done, offset + done)
        except InterruptedError:
            continue
        if not chunk:
            raise EOFError(f"PLE short read at {offset + done}, expected {size - done} bytes")
        chunks.append(chunk)
        done += len(chunk)
    return b"".join(chunks)


class PreadGather:
    """Experimental file-scoped reader; independent offsets, bounded workers."""

    def __init__(self, table, workers=0):
        self.table = table
        self.fd = os.open(table.reader.path, os.O_RDONLY)
        self.pool = None
        try:
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
            self.pool = ThreadPoolExecutor(max_workers=workers) if workers else None
        except BaseException:
            os.close(self.fd)
            raise

    def gather(self, indices):
        if self.fd is None or self.table._raw is None:
            raise RuntimeError("PLE reader is closed")
        ids = np.asarray(indices, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("row_indices must have shape [rows]")
        if ids.size and (ids.min() < 0 or ids.max() >= self.table.semantic_rows):
            raise IndexError("PLE row index is outside semantic rows")
        if not ids.size:
            return np.empty((0, self.table.row_width), dtype=np.float32)
        unique, inverse = np.unique(ids, return_inverse=True)
        row_bytes = int(self.table.tensor.byte_shape[1])
        raw = np.empty((unique.size, row_bytes), dtype=np.uint8)

        def read_range(bounds):
            begin, end = bounds
            for i in range(begin, end):
                offset = int(self.table.tensor.data_offset) + int(unique[i]) * row_bytes
                raw[i] = np.frombuffer(pread_exact(self.fd, row_bytes, offset), dtype=np.uint8)

        ranges = [(i, min(i + 128, unique.size)) for i in range(0, unique.size, 128)]
        if self.pool is None:
            for bounds in ranges:
                read_range(bounds)
        else:
            # Wait for every submitted task before releasing/reusing raw or fd.
            futures = [self.pool.submit(read_range, bounds) for bounds in ranges]
            failures = []
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    failures.append(exc)
            if failures:
                raise failures[0]
        values = dequantize_gguf_data(raw, self.table.tensor.ggml_type).astype(np.float32, copy=False)
        self.table.rows_gathered += int(ids.size)
        return values.reshape(unique.size, self.table.row_width)[inverse]

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def gather_sorted_unique(table, row_indices, *, method="sorted_unique"):
    if table._raw is None:
        raise RuntimeError("PLE mmap table is closed")
    indices = np.asarray(row_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("row_indices must have shape [rows]")
    if indices.size and (indices.min() < 0 or indices.max() >= table.semantic_rows):
        raise IndexError("PLE row index is outside semantic rows")
    if not indices.size:
        return np.empty((0, table.row_width), dtype=np.float32)
    if method == "sampled_dedup_elision" and indices.size <= 16:
        method = "copy_elision"
    if method == "sampled_dedup_elision":
        probe = np.concatenate((indices[:64], indices[64::max(1, indices.size // 64)]))
        # This only selects an equivalent algorithm, never output rows.
        # Avoid sorting the complete vector when a bounded probe is unique.
        method = "copy_elision" if np.unique(probe).size == probe.size else "dedup_elision"
    if method == "copy_elision" or (method == "dedup_elision" and indices.size <= 16):
        selected = np.asarray(table._raw[indices])
        values = dequantize_gguf_data(selected, table.tensor.ggml_type).astype(np.float32, copy=False)
        table.rows_gathered += int(indices.size)
        return values.reshape(indices.size, table.row_width)
    unique, inverse = np.unique(indices, return_inverse=True)
    if method == "dedup_elision" and unique.size == indices.size:
        selected = np.asarray(table._raw[indices])
        values = dequantize_gguf_data(selected, table.tensor.ggml_type).astype(np.float32, copy=False)
        table.rows_gathered += int(indices.size)
        return values.reshape(indices.size, table.row_width)
    selected = np.asarray(table._raw[unique])
    values = dequantize_gguf_data(selected, table.tensor.ggml_type).astype(
        np.float32, copy=method == "sorted_unique",
    )
    values = values.reshape(unique.size, table.row_width)[inverse]
    table.rows_gathered += int(indices.size)
    return values


def canonical_rows(case, cfg):
    tokens = case["prompt_token_ids"]
    rows, _ = ple_hash_rows(
        tokens, positions=np.arange(len(tokens)), sequence_ids=np.zeros(len(tokens), dtype=np.int64),
        states={}, eos_token_id=cfg.ple_eos_token_id,
        layer_multipliers=cfg.ple_layer_multipliers,
        head_offsets=cfg.ple_head_offsets, head_vocab_sizes=cfg.ple_head_vocab_sizes,
        heads_per_ngram=cfg.ple_heads_per_ngram, ngram_size=cfg.ple_ngram_size,
    )
    return rows.reshape(-1)


def measure(table, indices, *, cache_mode, repetitions, method="sorted_unique", reader=None):
    candidate_fn = table.gather_rows if method == "mmap_random" else reader.gather if reader is not None else (
        lambda ids: gather_sorted_unique(table, ids, method=method)
    )
    reference = table.gather_rows(indices)
    candidate = candidate_fn(indices)
    if reference.tobytes() != candidate.tobytes():
        raise ValueError("candidate changed PLE values or row order")
    staging = np.empty_like(reference)
    samples = []
    for rep in range(repetitions):
        order = ("parent", "candidate") if rep % 2 == 0 else ("candidate", "parent")
        for name in order:
            if cache_mode == "cold":
                table.advise_cache("cold")
            if method == "mmap_random":
                table._raw._mmap.madvise(mmap.MADV_RANDOM if name == "candidate" else mmap.MADV_NORMAL)
            fn = table.gather_rows if name == "parent" else candidate_fn
            before_io = process_read_bytes()
            started = time.perf_counter_ns()
            values = fn(indices)
            np.copyto(staging, values)
            elapsed = time.perf_counter_ns() - started
            read_bytes = process_read_bytes() - before_io
            if staging.tobytes() != reference.tobytes():
                raise ValueError("timed gather/scatter/staging output changed")
            samples.append({"arm": name, "repetition": rep, "ns": elapsed, "process_read_bytes": read_bytes})
    medians = {arm: statistics.median(s["ns"] for s in samples if s["arm"] == arm)
               for arm in ("parent", "candidate")}
    return {
        "rows": int(indices.size), "unique_rows": int(np.unique(indices).size),
        "cache_mode": cache_mode, "samples": samples, "median_ns": medians,
        "parent_over_candidate": medians["parent"] / medians["candidate"],
        "bit_exact": True,
        "row_ids_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }


def process_read_bytes():
    fields = dict(line.split(": ", 1) for line in Path("/proc/self/io").read_text().splitlines())
    return int(fields["read_bytes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--cache-mode", choices=("warm", "cold"), default="warm")
    parser.add_argument("--method", choices=("sorted_unique", "copy_elision", "dedup_elision", "sampled_dedup_elision", "pread", "pread_workers", "mmap_random"), default="sorted_unique")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--parent-mapping", choices=("normal", "random"), default="normal",
                        help="fixed mapping policy shared by both non-mapping arms")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_host()
    if args.repetitions < 2 or args.repetitions % 2:
        raise ValueError("use an even number of counterbalanced repetitions")
    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixture, fixture_hash = load_fixture(args.fixture)
        if args.case_id and not set(args.case_id) <= {c["id"] for c in fixture["cases"]}:
            raise ValueError("unknown canonical case id")
        identity = model_identity(args.model_root)
        readers = [GGUFReader(path) for path in discover_gguf_files(args.model_root)]
        cfg = qwen4_exp_gguf_config_from_metadata(readers[0].info)
        reader = next(r for r in readers if any(
            t.name == "per_layer_token_embd.weight" for t in r.info.tensors
        ))
        tensor = reader.tensor_info("per_layer_token_embd.weight")
        semantic_rows = max(int(o + n) for o, n in zip(cfg.ple_head_offsets, cfg.ple_head_vocab_sizes))
        table = Qwen4ExpPLEMMapTable(reader, tensor, semantic_rows=semantic_rows)
        table.configure_mapping_access(args.parent_mapping)
        pread = PreadGather(table, workers=8 if args.method == "pread_workers" else 0) if args.method.startswith("pread") else None
        report = {
            "kind": "qwen4exp_ple_sorted_gather_screen", "status": "running",
            "performance_claim": False, "host": _host_metadata(), "source": _git_metadata(ROOT),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_identity": identity, "fixture_sha256": fixture_hash,
            "command": sys.argv, "cases": [],
            "method": args.method,
            "parent_mapping": args.parent_mapping,
            "scope": "CPU gather/dequant/scatter plus staging copy; excludes H2D and inference",
            "cache_limit": "cold is file-scoped advice, not proof every requested page was evicted",
        }
        try:
            for case in fixture["cases"]:
                if args.case_id and case["id"] not in args.case_id:
                    continue
                ids = canonical_rows(case, cfg)
                result = measure(table, ids, cache_mode=args.cache_mode, repetitions=args.repetitions, method=args.method, reader=pread)
                report["cases"].append({"id": case["id"], "category": case["category"], **result})
                print(case["id"], result["parent_over_candidate"], flush=True)
            rng = np.random.default_rng(20260914)
            for count in (() if args.skip_synthetic else (16, 1024, 16384)):
                ids = rng.choice(semantic_rows, size=count, replace=False).astype(np.int64)
                for label, selected in (("all_unique", ids), ("duplicates", np.repeat(ids[:max(1, count // 16)], 16))):
                    result = measure(table, selected, cache_mode=args.cache_mode, repetitions=args.repetitions, method=args.method, reader=pread)
                    report["cases"].append({"id": f"{label}_{count}", "category": "heldout_rows", **result})
                    print(label, count, result["parent_over_candidate"], flush=True)
            report["status"] = "completed"
        finally:
            if pread is not None:
                pread.close()
            table.close()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
