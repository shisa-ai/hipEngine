"""Device builder for Inkling 2 expert-route tiles.

Mirrors ``hipengine.loading.inkling2_device_experts.build_expert_route_tiles``
on the GPU so the resident route path never has to pull route ids back to the
host just to group them into fixed-width tiles. The kernel also gathers an
optional per-expert outer-scale table into per-route scales.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from hipengine.core.build import build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc

_SOURCE = Path(__file__).with_name("inkling2_route_tiles.hip")
_OUTPUT_NAME = "inkling2_route_tiles.so"
_SYMBOL = "hipengine_inkling2_route_tiles_from_ids"
_IDS_SYMBOL = "hipengine_inkling2_route_ids_i32"
_IDS_ARGTYPES = (
    ctypes.c_void_p,  # ids (i64, [routes])
    ctypes.c_int,     # routes
    ctypes.c_void_p,  # scale table (f32, [experts]) or NULL
    ctypes.c_void_p,  # gathered scales (f32, [routes]) or NULL
    ctypes.c_void_p,  # ids out (i32, [routes])
    ctypes.c_void_p,  # stream
)
_ARGTYPES = (
    ctypes.c_void_p,  # ids (i64, [routes])
    ctypes.c_int,     # routes
    ctypes.c_int,     # num_experts
    ctypes.c_int,     # width (2 or 4)
    ctypes.c_void_p,  # scale table (f32, [num_experts]) or NULL
    ctypes.c_void_p,  # gathered scales (f32, [routes]) or NULL
    ctypes.c_void_p,  # records out (int32, [routes * (width + 1)] max)
    ctypes.c_void_p,  # tile_count out (int32[1])
    ctypes.c_int,     # record_capacity (tiles the records buffer can hold)
    ctypes.c_void_p,  # stream
)


def plan_inkling2_route_tiles_build(*, cache_root=None, compiler_version=None, profile="decode"):
    return plan_hip_build(
        sources=[_SOURCE],
        family="inkling2_route_tiles",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_inkling2_route_tiles(*, cache_root=None, compiler_version=None, profile="decode",
                               dry_run: bool = False, load: bool = True):
    return build_hip(
        sources=[_SOURCE],
        family="inkling2_route_tiles",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
    )


def launch_route_tiles(ids_ptr, routes, num_experts, width, scale_table_ptr, gathered_ptr,
                       records_ptr, count_ptr, record_capacity, *, stream: int = 0,
                       library=None, runtime: HipRuntime | None = None):
    """Launch the device tile builder. Pointers are raw device addresses."""
    library = library or build_inkling2_route_tiles(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL, _ARGTYPES, ctypes.c_int)
    err = fn(ids_ptr, int(routes), int(num_experts), int(width),
             scale_table_ptr, gathered_ptr, records_ptr, count_ptr,
             int(record_capacity), stream)
    if err != HIP_SUCCESS:
        runtime.check(err)


def launch_route_ids_i32(ids_ptr, routes, scale_table_ptr, gathered_ptr, ids_out_ptr, *,
                         stream: int = 0, library=None, runtime: HipRuntime | None = None):
    """Convert device i64 route ids to i32 and gather per-route scales."""
    library = library or build_inkling2_route_tiles(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _IDS_SYMBOL, _IDS_ARGTYPES, ctypes.c_int)
    err = fn(ids_ptr, int(routes), scale_table_ptr, gathered_ptr, ids_out_ptr, stream)
    if err != HIP_SUCCESS:
        runtime.check(err)


def build_expert_route_tiles_device(ids, num_experts, *, width: int = 4, scale_table=None,
                                    library=None, runtime: HipRuntime | None = None):
    """Host-side convenience wrapper: upload, build, download, return tiles.

    ``ids`` is an integer array of route expert ids (any int dtype). Returns
    ``(records, gathered_scales)`` where ``records`` is an int32 array of shape
    ``(tiles, width + 1)`` matching ``build_expert_route_tiles`` exactly, and
    ``gathered_scales`` is the per-route fp32 table (or ``None`` when no
    ``scale_table`` was given).
    """
    ids = np.ascontiguousarray(ids, dtype=np.int64).ravel()
    routes = int(ids.size)
    if width not in (2, 4):
        raise ValueError("width must be 2 or 4")
    if not 1 <= int(num_experts) <= 1024:
        raise ValueError("num_experts must be in [1, 1024]")
    runtime = runtime or get_hip_runtime()
    if routes == 0:
        return (np.empty((0, width + 1), dtype=np.int32),
                np.empty(0, dtype=np.float32) if scale_table is not None else None)

    flat = np.ascontiguousarray(scale_table, dtype=np.float32).ravel() if scale_table is not None else None
    if flat is not None and flat.size != num_experts:
        raise ValueError(f"scale_table has {flat.size} entries for {num_experts} experts")

    stride = width + 1
    ids_buf = malloc(ids.nbytes, runtime=runtime)
    rec_buf = malloc(routes * stride * 4, runtime=runtime)
    cnt_buf = malloc(4, runtime=runtime)
    scale_buf = malloc(flat.nbytes, runtime=runtime) if flat is not None else None
    gathered_buf = malloc(routes * 4, runtime=runtime) if flat is not None else None
    try:
        copy_host_to_device(ids_buf, host_array_ptr(ids), ids.nbytes, runtime=runtime)
        if flat is not None:
            copy_host_to_device(scale_buf, host_array_ptr(flat), flat.nbytes, runtime=runtime)
        launch_route_tiles(ids_buf.ptr, routes, num_experts, width,
                           scale_buf.ptr if scale_buf is not None else None,
                           gathered_buf.ptr if gathered_buf is not None else None,
                           rec_buf.ptr, cnt_buf.ptr, routes, stream=0, library=library, runtime=runtime)
        runtime.stream_synchronize(0)
        from hipengine.core.memory import copy_device_to_host
        count = np.empty(1, dtype=np.int32)
        copy_device_to_host(host_array_ptr(count), cnt_buf, 4, runtime=runtime)
        tiles = int(count[0])
        records = np.empty((tiles, stride), dtype=np.int32)
        if tiles:
            copy_device_to_host(host_array_ptr(records), rec_buf, tiles * stride * 4, runtime=runtime)
        gathered = None
        if gathered_buf is not None:
            gathered = np.empty(routes, dtype=np.float32)
            copy_device_to_host(host_array_ptr(gathered), gathered_buf, gathered.nbytes, runtime=runtime)
        return records, gathered
    finally:
        free(ids_buf, runtime=runtime)
        free(rec_buf, runtime=runtime)
        free(cnt_buf, runtime=runtime)
        if scale_buf is not None:
            free(scale_buf, runtime=runtime)
            free(gathered_buf, runtime=runtime)
